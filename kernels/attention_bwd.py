"""Flash-attention backward kernels.

The backward pass is where fused attention gets genuinely hard. The forward
never stored P = softmax(S) — so the backward must RECOMPUTE it blockwise
from Q, K and the saved log-sum-exp: P = exp(scale·QKᵀ − LSE). Given dO, the
gradient chain is:

    D_i  = rowsum(dO ∘ O)                  (the softmax-Jacobian shortcut:
                                            D_i = Σ_j P_ij dP_ij)
    dV   = Pᵀ dO
    dP   = dO Vᵀ
    dS   = P ∘ (dP − D_i)                  (softmax backward, per row)
    dQ   = scale · dS K
    dK   = scale · dSᵀ Q

Three kernels, mirroring the real FA2 implementation:
  1. preprocess: D = rowsum(dO ∘ O)
  2. dK/dV: one program per KEY block, looping over query blocks that can
     see it (m ≥ n − shift under causal masking) — each program owns its
     dK/dV tile exclusively, so no atomics.
  3. dQ: one program per QUERY block, looping over key blocks it sees
     (n ≤ m + shift) — recomputes P and dS again rather than sharing them
     with kernel 2, trading FLOPs for zero cross-block communication.

Classic silent-bug traps handled here: padding query rows have garbage LSE
(the forward never stored them) — loaded as +inf so exp(s − ∞) = 0; the
scale factor must hit dS exactly once; and the causal shift must match the
forward's bottom-right alignment.
"""

from __future__ import annotations

import os

import torch

import kernels  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _preprocess_kernel(O, DO, D, seq_q, HEAD_DIM: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask = (offs_m[:, None] < seq_q) & (offs_d[None, :] < HEAD_DIM)
    base = pid_bh * seq_q * HEAD_DIM
    o = tl.load(O + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                mask=mask, other=0.0).to(tl.float32)
    do = tl.load(DO + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                 mask=mask, other=0.0).to(tl.float32)
    d = tl.sum(o * do, axis=1)
    tl.store(D + pid_bh * seq_q + offs_m, d, mask=offs_m < seq_q)


@triton.jit
def _bwd_dkdv_kernel(Q, K, V, DO, LSE, Dv, DK, DV,
                     n_heads, seq_q, seq_k, scale,
                     CAUSAL: tl.constexpr, HEAD_DIM: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_D: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    dmask = offs_d[None, :] < HEAD_DIM

    kbase = pid_bh * seq_k * HEAD_DIM
    qbase = pid_bh * seq_q * HEAD_DIM
    nmask = (offs_n[:, None] < seq_k) & dmask
    k = tl.load(K + kbase + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                mask=nmask, other=0.0).to(tl.float32)
    v = tl.load(V + kbase + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                mask=nmask, other=0.0).to(tl.float32)

    dk = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)

    shift = seq_k - seq_q
    if CAUSAL:
        # first query row that can see key n is m = n - shift
        start_m = pid_n * BLOCK_N - shift
        start_m = tl.maximum(start_m, 0)
        start_m = (start_m // BLOCK_M) * BLOCK_M
    else:
        start_m = 0

    for m0 in range(start_m, seq_q, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mmask = (offs_m[:, None] < seq_q) & dmask
        q = tl.load(Q + qbase + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=mmask, other=0.0).to(tl.float32)
        do = tl.load(DO + qbase + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                     mask=mmask, other=0.0).to(tl.float32)
        lse = tl.load(LSE + pid_bh * seq_q + offs_m,
                      mask=offs_m < seq_q, other=float("inf"))
        dvec = tl.load(Dv + pid_bh * seq_q + offs_m,
                       mask=offs_m < seq_q, other=0.0)

        s = tl.dot(q, tl.trans(k)) * scale                  # [M, N]
        valid = (offs_m[:, None] < seq_q) & (offs_n[None, :] < seq_k)
        if CAUSAL:
            valid = valid & (offs_n[None, :] <= offs_m[:, None] + shift)
        p = tl.where(valid, tl.exp(s - lse[:, None]), 0.0)

        dv += tl.dot(tl.trans(p), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - dvec[:, None]) * scale
        dk += tl.dot(tl.trans(ds), q)

    smask = (offs_n[:, None] < seq_k) & dmask
    tl.store(DK + kbase + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
             dk.to(DK.dtype.element_ty), mask=smask)
    tl.store(DV + kbase + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
             dv.to(DV.dtype.element_ty), mask=smask)


@triton.jit
def _bwd_dq_kernel(Q, K, V, DO, LSE, Dv, DQ,
                   n_heads, seq_q, seq_k, scale,
                   CAUSAL: tl.constexpr, HEAD_DIM: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    dmask = offs_d[None, :] < HEAD_DIM
    qbase = pid_bh * seq_q * HEAD_DIM
    kbase = pid_bh * seq_k * HEAD_DIM

    mmask = (offs_m[:, None] < seq_q) & dmask
    q = tl.load(Q + qbase + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                mask=mmask, other=0.0).to(tl.float32)
    do = tl.load(DO + qbase + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                 mask=mmask, other=0.0).to(tl.float32)
    lse = tl.load(LSE + pid_bh * seq_q + offs_m,
                  mask=offs_m < seq_q, other=float("inf"))
    dvec = tl.load(Dv + pid_bh * seq_q + offs_m,
                   mask=offs_m < seq_q, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    shift = seq_k - seq_q
    if CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M + shift, seq_k)
    else:
        hi = seq_k

    for n0 in range(0, hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nmask = (offs_n[:, None] < seq_k) & dmask
        k = tl.load(K + kbase + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=nmask, other=0.0).to(tl.float32)
        v = tl.load(V + kbase + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=nmask, other=0.0).to(tl.float32)

        s = tl.dot(q, tl.trans(k)) * scale
        valid = (offs_m[:, None] < seq_q) & (offs_n[None, :] < seq_k)
        if CAUSAL:
            valid = valid & (offs_n[None, :] <= offs_m[:, None] + shift)
        p = tl.where(valid, tl.exp(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - dvec[:, None]) * scale
        dq += tl.dot(ds, k)

    tl.store(DQ + qbase + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
             dq.to(DQ.dtype.element_ty), mask=mmask)


def flash_attention_backward(q, k, v, o, lse, do, causal=True, scale=None):
    """Returns (dq, dk, dv). All inputs [B, H, S, D] contiguous, lse [B,H,Sq]."""
    B, H, Sq, D = q.shape
    Sk = k.shape[2]
    scale = scale if scale is not None else D ** -0.5
    q, k, v, o, do = (t.contiguous() for t in (q, k, v, o, do))

    interp = os.environ.get("TRITON_INTERPRET") == "1"
    BLOCK_M = 16 if interp else 32
    BLOCK_N = 16 if interp else 32
    BLOCK_D = max(16, triton.next_power_of_2(D))

    dvec = torch.empty(B, H, Sq, dtype=torch.float32, device=q.device)
    _preprocess_kernel[(triton.cdiv(Sq, BLOCK_M), B * H)](
        o, do, dvec, Sq, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    _bwd_dkdv_kernel[(triton.cdiv(Sk, BLOCK_N), B * H)](
        q, k, v, do, lse, dvec, dk, dv,
        H, Sq, Sk, scale, CAUSAL=causal, HEAD_DIM=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D)

    _bwd_dq_kernel[(triton.cdiv(Sq, BLOCK_M), B * H)](
        q, k, v, do, lse, dvec, dq,
        H, Sq, Sk, scale, CAUSAL=causal, HEAD_DIM=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D)

    return dq, dk, dv
