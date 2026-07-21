"""FP8 (e4m3) matmul kernel with inline scale handling.

HONESTY NOTE — what is and isn't real here
------------------------------------------
No hardware in this project has FP8 tensor cores (Kaggle's T4 is sm_75;
local dev is CPU). So: the QUANTIZATION is real — inputs are genuine
float8_e4m3fn bit patterns with true e4m3 rounding/saturation — but the
ARITHMETIC is fp32: the kernel decodes e4m3 bits in registers and runs an
fp32 dot. On H100-class hardware the same structure maps to native FP8 WGMMA
and the decode disappears. Reported FP8 numbers are therefore about
NUMERICS (does training converge under fp8 rounding + dynamic scaling), not
speed.

What "inline scales" means: the kernel consumes raw fp8 bits plus the two
per-tensor scale factors and applies them inside the kernel after
accumulation (per-tensor scales factor out of the dot product) — there is
no separate "dequantize to a full fp16 tensor, then matmul" pass through
memory.

e4m3fn format: 1 sign, 4 exponent (bias 7), 3 mantissa; no infinities;
NaN = S.1111.111; max finite 448.
"""

from __future__ import annotations

import os

import torch

import kernels  # noqa: F401
import triton
import triton.language as tl

E4M3_MAX = 448.0


@triton.jit
def _decode_e4m3(bits):
    """uint8 e4m3fn bit pattern -> fp32 value (NaN pattern decodes as 0)."""
    sign = (bits >> 7) & 1
    exp = ((bits >> 3) & 0xF).to(tl.int32)
    man = (bits & 0x7).to(tl.float32)
    normal = (1.0 + man / 8.0) * tl.exp2((exp - 7).to(tl.float32))
    subnormal = man * 0.001953125            # man/8 * 2^-6 = man * 2^-9
    val = tl.where(exp == 0, subnormal, normal)
    val = tl.where((exp == 15) & (man == 7), 0.0, val)   # NaN pattern
    return tl.where(sign == 1, -val, val)


@triton.jit
def _fp8_matmul_kernel(A, B, C, sa, sb, M, N, K,
                       stride_am, stride_ak,
                       stride_bn, stride_bk,
                       stride_cm, stride_cn,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr):
    """C[M,N] = sa * sb * (decode(A)[M,K] @ decode(B)[N,K]^T)"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_bits = tl.load(A + offs_m[:, None] * stride_am
                         + offs_k[None, :] * stride_ak,
                         mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                         other=0)
        b_bits = tl.load(B + offs_n[:, None] * stride_bn
                         + offs_k[None, :] * stride_bk,
                         mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
                         other=0)
        a = _decode_e4m3(a_bits)
        b = _decode_e4m3(b_bits)
        acc = tl.dot(a, tl.trans(b), acc)

    c = acc * (sa * sb)                       # inline per-tensor scales
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             c.to(C.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _decode_e4m3_kernel(BITS, OUT, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    bits = tl.load(BITS + offs, mask=mask, other=0)
    tl.store(OUT + offs, _decode_e4m3(bits), mask=mask)


def decode_e4m3(bits: torch.Tensor) -> torch.Tensor:
    """Kernel-decode a uint8 tensor of e4m3fn bit patterns to fp32."""
    flat = bits.reshape(-1).contiguous()
    out = torch.empty(flat.shape, dtype=torch.float32, device=bits.device)
    BLOCK = 256
    _decode_e4m3_kernel[(triton.cdiv(flat.numel(), BLOCK),)](
        flat, out, flat.numel(), BLOCK=BLOCK)
    return out.reshape(bits.shape)


def quantize_e4m3(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Real e4m3 rounding via torch's float8 cast. Returns uint8 bits."""
    q = (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX)
    return q.to(torch.float8_e4m3fn).view(torch.uint8)


def dequantize_e4m3(bits: torch.Tensor, scale: float) -> torch.Tensor:
    return bits.view(torch.float8_e4m3fn).float() * scale


def fp8_matmul(a_bits: torch.Tensor, b_bits: torch.Tensor,
               sa: float, sb: float,
               out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """a_bits [M,K], b_bits [N,K] uint8 e4m3 -> sa*sb*(Â @ B̂ᵀ) [M,N]"""
    M, K = a_bits.shape
    N = b_bits.shape[0]
    c = torch.empty(M, N, dtype=out_dtype, device=a_bits.device)
    interp = os.environ.get("TRITON_INTERPRET") == "1"
    BM, BN, BK = (16, 16, 16) if interp else (32, 64, 64)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _fp8_matmul_kernel[grid](a_bits, b_bits, c, sa, sb, M, N, K,
                             a_bits.stride(0), a_bits.stride(1),
                             b_bits.stride(0), b_bits.stride(1),
                             c.stride(0), c.stride(1),
                             BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
    return c
