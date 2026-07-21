"""MoE token dispatch/combine kernels (gather-scatter routing), in Triton.

Why a custom kernel
-------------------
The naive MoE forward loops over experts and, for each one, does
`torch.where` + `index_select` (gather the tokens routed to it), the expert
FFN, then `index_add_` (scatter-add the weighted result back). That is
2*E separate indexing kernels launched per layer, each re-reading routing
metadata, and `index_add_` on GPU uses atomics.

Real MoE systems (vLLM fused-moe, MegaBlocks) instead:

  1. sort token->expert assignments once, so each expert's tokens become one
     contiguous slice of a dispatch buffer,
  2. gather all T*k rows into that buffer in ONE kernel,
  3. run each expert as a plain dense matmul over its contiguous slice,
  4. combine the k expert outputs per token in ONE kernel, weighted by the
     router probabilities — no atomics, because each output row is owned by
     exactly one program.

This module implements steps 2 and 4 as Triton kernels. The assignment sort
(step 1) stays in `torch.argsort`: it is O(T*k) on scalars, negligible next
to the O(T*k*D) row movement, and a radix sort in Triton would add a lot of
code for no measurable win at this scale. Expert matmuls (step 3) stay
PyTorch GEMMs — the point of the layout is precisely that they become plain
dense matmuls.
"""

from __future__ import annotations

import os

import torch

import kernels  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _gather_rows_kernel(X, OUT, IDX, D,
                        stride_x, stride_o,
                        BLOCK_D: tl.constexpr):
    """OUT[i, :] = X[IDX[i], :] — one program per (row, D-block)."""
    row = tl.program_id(0)
    dblk = tl.program_id(1)
    src = tl.load(IDX + row)
    offs = dblk * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D
    val = tl.load(X + src * stride_x + offs, mask=mask, other=0.0)
    tl.store(OUT + row * stride_o + offs, val, mask=mask)


@triton.jit
def _combine_rows_kernel(EXP_OUT, W, POS, OUT, D,
                         stride_e, stride_o,
                         K: tl.constexpr, BLOCK_D: tl.constexpr):
    """OUT[t, :] = sum_k W[t,k] * EXP_OUT[POS[t,k], :].

    Each program owns one output token row (and one D-block), so the combine
    needs no atomics: it reads its K expert-output rows and accumulates in
    registers.
    """
    t = tl.program_id(0)
    dblk = tl.program_id(1)
    offs = dblk * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D

    acc = tl.zeros([BLOCK_D], tl.float32)
    for slot in range(K):
        pos = tl.load(POS + t * K + slot)
        w = tl.load(W + t * K + slot).to(tl.float32)
        row = tl.load(EXP_OUT + pos * stride_e + offs, mask=mask, other=0.0)
        acc += w * row.to(tl.float32)
    tl.store(OUT + t * stride_o + offs, acc.to(OUT.dtype.element_ty), mask=mask)


def _block_d(D: int) -> int:
    interp = os.environ.get("TRITON_INTERPRET") == "1"
    cap = 64 if interp else 1024
    return min(cap, triton.next_power_of_2(D))


def gather_rows(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """out[i] = x[idx[i]] for 2D x. Kernel equivalent of index_select(0)."""
    n, D = idx.shape[0], x.shape[1]
    out = torch.empty(n, D, dtype=x.dtype, device=x.device)
    BLOCK_D = _block_d(D)
    grid = (n, triton.cdiv(D, BLOCK_D))
    _gather_rows_kernel[grid](x, out, idx.to(torch.int64), D,
                              x.stride(0), out.stride(0), BLOCK_D=BLOCK_D)
    return out


def combine_rows(expert_out: torch.Tensor, weights: torch.Tensor,
                 pos: torch.Tensor) -> torch.Tensor:
    """out[t] = sum_k weights[t,k] * expert_out[pos[t,k]]."""
    T, K = pos.shape
    D = expert_out.shape[1]
    out = torch.empty(T, D, dtype=expert_out.dtype, device=expert_out.device)
    BLOCK_D = _block_d(D)
    grid = (T, triton.cdiv(D, BLOCK_D))
    _combine_rows_kernel[grid](expert_out, weights.contiguous(),
                               pos.to(torch.int64).contiguous(), out, D,
                               expert_out.stride(0), out.stride(0),
                               K=K, BLOCK_D=BLOCK_D)
    return out


def sort_by_expert(experts_idx: torch.Tensor, n_experts: int):
    """Group the T*k (token, slot) assignments by expert.

    Returns:
      token_of_slot: [T*k] source token row for each sorted slot
      counts:        [E] tokens per expert (sorted-buffer slice sizes)
      pos:           [T, k] position of (token, slot) in the sorted buffer
    """
    T, k = experts_idx.shape
    flat = experts_idx.reshape(-1)
    sort_idx = torch.argsort(flat, stable=True)          # [T*k]
    counts = torch.bincount(flat, minlength=n_experts)
    inv = torch.empty_like(sort_idx)
    inv[sort_idx] = torch.arange(T * k, device=flat.device)
    return sort_idx // k, counts, inv.reshape(T, k)


def moe_dispatch_combine(flat: torch.Tensor, weights: torch.Tensor,
                         experts_idx: torch.Tensor, expert_modules) -> torch.Tensor:
    """Full routed MoE forward using the custom dispatch/combine kernels.

    flat: [T, D] tokens, weights: [T, k] combine weights (already
    renormalized), experts_idx: [T, k] int expert ids.
    """
    n_experts = len(expert_modules)
    token_of_slot, counts, pos = sort_by_expert(experts_idx, n_experts)

    dispatched = gather_rows(flat, token_of_slot)        # [T*k, D] expert-sorted

    expert_out = torch.empty_like(dispatched)
    start = 0
    for e, cnt in enumerate(counts.tolist()):
        if cnt:
            expert_out[start:start + cnt] = expert_modules[e](
                dispatched[start:start + cnt])
        start += cnt

    return combine_rows(expert_out, weights, pos)
