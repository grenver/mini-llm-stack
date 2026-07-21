"""Fused attention forward kernel (flash-attention v2 style), in Triton.

Why a custom kernel
-------------------
Naive PyTorch attention does:

    S = Q @ K^T          # materializes [B, H, Sq, Sk] in HBM
    P = softmax(S)       # second full read/write of that matrix
    O = P @ V

The [Sq, Sk] score matrix is O(seq^2) memory traffic and capacity — at
seq 4096 with 8 heads, batch 4 in fp32 that intermediate alone is 2 GiB.
The matmuls are fast; the bottleneck is writing and re-reading the score
matrix through HBM.

This kernel never materializes the score matrix. Each program owns a block
of BLOCK_M query rows, streams over K/V in blocks of BLOCK_N, and maintains
an online softmax (running row max `m_i`, running normalizer `l_i`, and an
unnormalized output accumulator) entirely in registers/SRAM. Memory traffic
drops from O(seq^2) to O(seq * head_dim), and everything happens in one pass.

Causal masking is bottom-right aligned: query i attends to keys
j <= i + (Sk - Sq), matching `train.model.naive_attention`. The kernel also
writes the per-row log-sum-exp (LSE), which the Phase 8 backward pass needs
to recompute softmax probabilities without storing them.
"""

from __future__ import annotations

import os

import torch

import kernels  # noqa: F401  (sets TRITON_INTERPRET before triton import)
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    n_heads, seq_q, seq_k,
    scale,
    CAUSAL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // n_heads
    h = pid_bh % n_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = (Q + b * stride_qb + h * stride_qh
              + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q_mask = (offs_m[:, None] < seq_q) & (offs_d[None, :] < HEAD_DIM)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)   # running row max
    l_i = tl.zeros([BLOCK_M], tl.float32)                 # running normalizer
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)        # unnormalized output

    # causal shift for Sq != Sk (bottom-right aligned)
    shift = seq_k - seq_q
    if CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M + shift, seq_k)
    else:
        hi = seq_k

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = (K + b * stride_kb + h * stride_kh
                  + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        k_mask = (offs_n[:, None] < seq_k) & (offs_d[None, :] < HEAD_DIM)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        scores = tl.dot(q, tl.trans(k)) * scale           # [BLOCK_M, BLOCK_N] fp32
        valid = offs_n[None, :] < seq_k
        if CAUSAL:
            valid = valid & (offs_n[None, :] <= offs_m[:, None] + shift)
        scores = tl.where(valid, scores, float("-inf"))

        # Online softmax update. The m_safe/alpha guards keep fully-masked
        # rows (padding query rows) at exact zero instead of NaN.
        m_ij = tl.max(scores, 1)
        m_new = tl.maximum(m_i, m_ij)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_safe))
        p = tl.exp(scores - m_safe[:, None])              # masked entries -> 0

        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v_ptrs = (V + b * stride_vb + h * stride_vh
                  + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]

    o_ptrs = (O + b * stride_ob + h * stride_oh
              + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=q_mask)

    lse = m_i + tl.log(l_safe)
    lse_ptrs = LSE + pid_bh * seq_q + offs_m
    tl.store(lse_ptrs, lse, mask=offs_m < seq_q)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    causal: bool = True, scale: float | None = None,
                    return_lse: bool = False):
    """Fused attention. q,k,v: [B, H, S, head_dim], q may have Sq != Sk.

    Returns O [B, H, Sq, head_dim] (and fp32 LSE [B, H, Sq] if return_lse).
    """
    B, H, Sq, D = q.shape
    Sk = k.shape[2]
    assert k.shape == (B, H, Sk, D) and v.shape == (B, H, Sk, D)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    scale = scale if scale is not None else D ** -0.5

    o = torch.empty_like(q)
    lse = torch.empty(B, H, Sq, dtype=torch.float32, device=q.device)

    interp = os.environ.get("TRITON_INTERPRET") == "1"
    BLOCK_M = 16 if interp else 64
    BLOCK_N = 16 if interp else 64
    BLOCK_D = max(16, triton.next_power_of_2(D))

    grid = (triton.cdiv(Sq, BLOCK_M), B * H)
    _attn_fwd_kernel[grid](
        q, k, v, o, lse,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        H, Sq, Sk, scale,
        CAUSAL=causal, HEAD_DIM=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return (o, lse) if return_lse else o
