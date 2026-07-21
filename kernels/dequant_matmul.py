"""Fused dequantize + matmul kernels (INT8 per-row, INT4 group-wise packed).

Why a custom kernel
-------------------
The lazy way to run a quantized model is `x @ dequant(W).T`: materialize the
full fp16/fp32 weight matrix in HBM, then matmul. That throws away the entire
inference win — at decode time (M is tiny) the matmul is memory-bandwidth
bound on reading W, so what matters is the number of BYTES of weight read
per token, not FLOPs. Dequantizing first means you still read W at full
fp width (plus an extra pass to write the dequantized copy).

These kernels read the weights in their quantized form (1 byte/element for
INT8, 0.5 for INT4) and dequantize tile-by-tile in registers inside the
matmul's K-loop. Weight HBM traffic drops 2x/4x vs fp16, which is the
theoretical decode speedup ceiling.

Layouts
-------
INT8: W_q [N, K] int8, symmetric per-output-row scale s[N]:
      W ~= W_q * s[:, None]. The scale is constant along K, so it factors
      out of the dot product and is applied once after accumulation.
INT4: W packed [N, K/2] uint8 (even k in low nibble), unsigned nibbles with
      offset 8, group-wise scales s[N, K/group]: W ~= (nib - 8) * s_group.
      Scales vary along K, so each K-tile (tile width == group size) applies
      its group scale to the weight tile before the dot.
"""

from __future__ import annotations

import os

import torch

import kernels  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _int8_matmul_kernel(A, W, S, C, M, N, K,
                        stride_am, stride_ak,
                        stride_wn, stride_wk,
                        stride_cm, stride_cn,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(W + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0)
        # dequant in registers; per-row scale factors out of the dot,
        # so multiply raw int values here and scale once at the end
        acc = tl.dot(a, tl.trans(w.to(a.dtype)), acc)

    s = tl.load(S + offs_n, mask=offs_n < N, other=0.0)
    c = acc * s[None, :]
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             c.to(C.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _int4_matmul_kernel(A, Wp, S, C, M, N, K,
                        stride_am, stride_ak,
                        stride_wn, stride_wk,
                        stride_sn, stride_sg,
                        stride_cm, stride_cn,
                        GROUP: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """K-tile width == quantization group size, so one scale per tile row."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for g in range(0, K // GROUP):
        offs_k = g * GROUP + tl.arange(0, GROUP)
        a = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        byte = tl.load(Wp + offs_n[:, None] * stride_wn
                       + (offs_k[None, :] // 2) * stride_wk,
                       mask=offs_n[:, None] < N, other=0)
        nib = tl.where(offs_k[None, :] % 2 == 0, byte & 0x0F,
                       (byte >> 4) & 0x0F)
        s = tl.load(S + offs_n * stride_sn + g * stride_sg,
                    mask=offs_n < N, other=0.0)
        w = (nib.to(tl.float32) - 8.0) * s[:, None]
        acc = tl.dot(a, tl.trans(w.to(a.dtype)), acc)

    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(C.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _blocks():
    interp = os.environ.get("TRITON_INTERPRET") == "1"
    return (16, 16, 16) if interp else (32, 64, 64)


def int8_matmul(a: torch.Tensor, w_q: torch.Tensor, scale: torch.Tensor):
    """a [M, K] fp  @  dequant(w_q [N, K] int8, scale [N]).T  ->  [M, N]"""
    M, K = a.shape
    N = w_q.shape[0]
    c = torch.empty(M, N, dtype=a.dtype, device=a.device)
    BM, BN, BK = _blocks()
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _int8_matmul_kernel[grid](a, w_q, scale.to(torch.float32), c, M, N, K,
                              a.stride(0), a.stride(1),
                              w_q.stride(0), w_q.stride(1),
                              c.stride(0), c.stride(1),
                              BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
    return c


def int4_matmul(a: torch.Tensor, w_packed: torch.Tensor, scale: torch.Tensor,
                group_size: int):
    """a [M, K] @ dequant(int4-packed W [N, K/2], scales [N, K/group]).T"""
    M, K = a.shape
    N = w_packed.shape[0]
    assert K % group_size == 0 and K == w_packed.shape[1] * 2
    c = torch.empty(M, N, dtype=a.dtype, device=a.device)
    BM, BN, _ = _blocks()
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _int4_matmul_kernel[grid](a, w_packed, scale.to(torch.float32), c, M, N, K,
                              a.stride(0), a.stride(1),
                              w_packed.stride(0), w_packed.stride(1),
                              scale.stride(0), scale.stride(1),
                              c.stride(0), c.stride(1),
                              GROUP=group_size, BLOCK_M=BM, BLOCK_N=BN)
    return c
