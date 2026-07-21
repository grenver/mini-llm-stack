"""Backward kernels for MoE dispatch/combine.

Gradient structure (see moe_routing.py for the forward):

  gather:  out[i] = x[idx[i]]
    backward: dx[t] = Σ_{i: idx[i]=t} dout[i]   — a scatter-ADD, because with
    top-k routing each token row is gathered k times. Collisions are real,
    so this kernel uses atomic adds (the accumulation-across-gather-scatter
    subtlety the spec calls out: writing dx[idx[i]] = dout[i] would silently
    drop k−1 of every token's k gradient contributions).

  combine: out[t] = Σ_s w[t,s] · eo[pos[t,s]]
    d_eo[pos[t,s]] = w[t,s] · dout[t]  — pos is a permutation (every sorted
    slot appears exactly once), so each output row has exactly one writer:
    plain stores, no atomics.
    d_w[t,s] = <dout[t], eo[pos[t,s]]>  — one dot per (t, s).
"""

from __future__ import annotations

import os

import torch

import kernels  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _scatter_add_rows_kernel(DOUT, IDX, DX, D,
                             stride_do, stride_dx,
                             BLOCK_D: tl.constexpr):
    """DX[IDX[i], :] += DOUT[i, :] with atomics (gather backward)."""
    row = tl.program_id(0)
    dblk = tl.program_id(1)
    dst = tl.load(IDX + row)
    offs = dblk * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D
    val = tl.load(DOUT + row * stride_do + offs, mask=mask, other=0.0)
    tl.atomic_add(DX + dst * stride_dx + offs, val, mask=mask)


@triton.jit
def _combine_bwd_deo_kernel(DOUT, W, POS, DEO, D,
                            stride_do, stride_de,
                            K: tl.constexpr, BLOCK_D: tl.constexpr):
    """DEO[POS[t,s], :] = W[t,s] * DOUT[t, :] (each pos written exactly once)."""
    t = tl.program_id(0)
    dblk = tl.program_id(1)
    offs = dblk * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D
    dout = tl.load(DOUT + t * stride_do + offs, mask=mask, other=0.0)
    for s in range(K):
        pos = tl.load(POS + t * K + s)
        w = tl.load(W + t * K + s)
        tl.store(DEO + pos * stride_de + offs, (w * dout).to(DEO.dtype.element_ty),
                 mask=mask)


@triton.jit
def _combine_bwd_dw_kernel(DOUT, EO, POS, DW, D,
                           stride_do, stride_eo,
                           K: tl.constexpr, BLOCK_D: tl.constexpr,
                           ACC64: tl.constexpr):
    """DW[t,s] = dot(DOUT[t], EO[POS[t,s]]), reduced over D in blocks."""
    t = tl.program_id(0)
    offs_base = tl.arange(0, BLOCK_D)
    for s in range(K):
        pos = tl.load(POS + t * K + s)
        if ACC64:
            acc = tl.zeros([1], tl.float64)
        else:
            acc = tl.zeros([1], tl.float32)
        for d0 in range(0, D, BLOCK_D):
            offs = d0 + offs_base
            mask = offs < D
            dout = tl.load(DOUT + t * stride_do + offs, mask=mask, other=0.0)
            eo = tl.load(EO + pos * stride_eo + offs, mask=mask, other=0.0)
            acc += tl.sum(dout.to(acc.dtype) * eo.to(acc.dtype), axis=0)
        tl.store(DW + t * K + s + tl.arange(0, 1), acc)


def _block_d(D: int) -> int:
    interp = os.environ.get("TRITON_INTERPRET") == "1"
    return min(64 if interp else 1024, triton.next_power_of_2(D))


def scatter_add_rows(dout: torch.Tensor, idx: torch.Tensor,
                     n_rows: int) -> torch.Tensor:
    """dx[n_rows, D] = scatter-add of dout rows by idx (gather backward)."""
    D = dout.shape[1]
    # accumulate in fp32 for half inputs; keep fp32/fp64 native (fp64 matters
    # for finite-difference gradcheck in interpreter mode)
    acc_dtype = torch.float32 if dout.dtype == torch.float16 else dout.dtype
    dx = torch.zeros(n_rows, D, dtype=acc_dtype, device=dout.device)
    BLOCK_D = _block_d(D)
    grid = (dout.shape[0], triton.cdiv(D, BLOCK_D))
    _scatter_add_rows_kernel[grid](dout.to(acc_dtype).contiguous(),
                                   idx.to(torch.int64).contiguous(), dx, D,
                                   D, D, BLOCK_D=BLOCK_D)
    return dx.to(dout.dtype)


def combine_backward(dout: torch.Tensor, expert_out: torch.Tensor,
                     weights: torch.Tensor, pos: torch.Tensor):
    """Returns (d_expert_out, d_weights) for the combine op."""
    T, K = pos.shape
    D = expert_out.shape[1]
    deo = torch.empty_like(expert_out)
    dw_dtype = torch.float32 if weights.dtype == torch.float16 else weights.dtype
    dw = torch.empty(T, K, dtype=dw_dtype, device=dout.device)
    BLOCK_D = _block_d(D)
    dout_c = dout.contiguous()
    pos_c = pos.to(torch.int64).contiguous()
    _combine_bwd_deo_kernel[(T, triton.cdiv(D, BLOCK_D))](
        dout_c, weights.contiguous(), pos_c, deo, D,
        dout_c.stride(0), deo.stride(0), K=K, BLOCK_D=BLOCK_D)
    _combine_bwd_dw_kernel[(T,)](
        dout_c, expert_out.contiguous(), pos_c, dw, D,
        dout_c.stride(0), expert_out.stride(0), K=K, BLOCK_D=BLOCK_D,
        ACC64=dw.dtype == torch.float64)
    return deo, dw.to(weights.dtype)
