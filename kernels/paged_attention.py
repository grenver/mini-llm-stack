"""Paged-attention decode kernel: attention over block-scattered KV cache.

Why a custom kernel
-------------------
With a paged KV cache (serve/kv_cache.py), a sequence's keys/values live in
non-contiguous fixed-size blocks. PyTorch attention needs contiguous
[H, ctx, D] tensors, so the naive route is gather-copy every sequence's
blocks into a scratch buffer each step — an extra full read+write of the
entire context KV, every token, for every sequence.

This kernel does what vLLM's paged_attention does: each program handles one
(sequence, head), walks the sequence's block table, loads K/V block by
block directly from the pool, and maintains an online softmax as it goes —
zero copies, no materialized score vector longer than one block.

Decode shape notes: the query is a single token, so scores are computed as
a broadcast-multiply + row reduction ([block_size, D] * [D] summed over D)
rather than tl.dot (which needs >=16-row tiles and would force padding).
The new token's K/V must already be written to the cache (write-then-attend)
and context_lens must include it; causality is then automatic — the cache
simply contains nothing later than the current token.
"""

from __future__ import annotations

import torch

import kernels  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _paged_attn_kernel(
    Q, KP, VP, BT, CTX, OUT,
    stride_qs, stride_qh,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_bt_s,
    stride_os, stride_oh,
    scale, max_blocks,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    dmask = offs_d < HEAD_DIM
    q = tl.load(Q + seq * stride_qs + head * stride_qh + offs_d,
                mask=dmask, other=0.0).to(tl.float32)

    ctx = tl.load(CTX + seq)
    n_blocks = tl.cdiv(ctx, BLOCK_SIZE)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], tl.float32)

    for b in range(0, n_blocks):
        blk = tl.load(BT + seq * stride_bt_s + b)
        offs_t = tl.arange(0, BLOCK_SIZE)
        tpos = b * BLOCK_SIZE + offs_t
        tmask = tpos < ctx

        k = tl.load(KP + blk * stride_kb + head * stride_kh
                    + offs_t[:, None] * stride_kt + offs_d[None, :] * stride_kd,
                    mask=tmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * scale        # [BLOCK_SIZE]
        scores = tl.where(tmask, scores, float("-inf"))

        m_ij = tl.max(scores, 0)
        m_new = tl.maximum(m_i, m_ij)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_safe))
        p = tl.exp(scores - m_safe)                            # masked -> 0

        v = tl.load(VP + blk * stride_kb + head * stride_kh
                    + offs_t[:, None] * stride_kt + offs_d[None, :] * stride_kd,
                    mask=tmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, 0)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe
    tl.store(OUT + seq * stride_os + head * stride_oh + offs_d,
             out.to(OUT.dtype.element_ty), mask=dmask)


def paged_attention_decode(q: torch.Tensor, k_pool: torch.Tensor,
                           v_pool: torch.Tensor, block_tables: torch.Tensor,
                           context_lens: torch.Tensor,
                           scale: float | None = None) -> torch.Tensor:
    """q: [n_seqs, H, D]; pools: [num_blocks, H, block_size, D];
    block_tables: [n_seqs, max_blocks] int32; context_lens: [n_seqs] int32.
    Returns [n_seqs, H, D].
    """
    n_seqs, H, D = q.shape
    block_size = k_pool.shape[2]
    scale = scale if scale is not None else D ** -0.5
    out = torch.empty_like(q)
    if n_seqs == 0:
        return out
    q, block_tables = q.contiguous(), block_tables.contiguous()

    grid = (n_seqs, H)
    _paged_attn_kernel[grid](
        q, k_pool, v_pool, block_tables, context_lens, out,
        q.stride(0), q.stride(1),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        block_tables.stride(0),
        out.stride(0), out.stride(1),
        scale, block_tables.shape[1],
        HEAD_DIM=D, BLOCK_SIZE=block_size,
        BLOCK_D=max(16, triton.next_power_of_2(D)),
    )
    return out
