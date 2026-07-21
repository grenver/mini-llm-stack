"""Weight-only quantization: INT8 (per-row symmetric) and INT4 (group-wise).

Weight-only because at small batch the matmuls are bandwidth-bound on
weights; activations stay fp. INT8 uses one symmetric scale per output row
(max|row| / 127). INT4 packs two nibbles per byte with offset-8 encoding and
one scale per `group_size` input elements per row — per-row scaling is too
coarse at 4 bits, group-wise keeps outlier damage local.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from train.model import Transformer


def quantize_int8(w: torch.Tensor):
    scale = (w.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
    w_q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return w_q, scale


def dequantize_int8(w_q: torch.Tensor, scale: torch.Tensor):
    return w_q.float() * scale[:, None]


def quantize_int4(w: torch.Tensor, group_size: int = 32):
    N, K = w.shape
    assert K % group_size == 0
    g = w.reshape(N, K // group_size, group_size)
    scale = (g.abs().amax(dim=2) / 7.0).clamp(min=1e-8)        # [N, K/gs]
    q = torch.round(g / scale[:, :, None]).clamp(-8, 7) + 8    # unsigned 0..15
    q = q.reshape(N, K).to(torch.uint8)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()     # even k -> low nibble
    return packed, scale


def dequantize_int4(packed: torch.Tensor, scale: torch.Tensor, group_size: int):
    N = packed.shape[0]
    K = packed.shape[1] * 2
    q = torch.empty(N, K, dtype=torch.uint8, device=packed.device)
    q[:, 0::2] = packed & 0x0F
    q[:, 1::2] = packed >> 4
    w = q.float() - 8.0
    w = w.reshape(N, K // group_size, group_size) * scale[:, :, None]
    return w.reshape(N, K)


class QuantLinear(nn.Module):
    """Drop-in nn.Linear replacement holding quantized weights.

    use_kernel=True runs the fused Triton dequant+matmul; False runs the
    reference path (explicit dequant then matmul) — the thing the kernel
    exists to avoid, kept for correctness testing and fast CPU fallback.
    """

    def __init__(self, linear: nn.Linear, bits: int = 8, group_size: int = 32,
                 use_kernel: bool = True):
        super().__init__()
        assert linear.bias is None
        assert bits in (4, 8)
        self.bits, self.group_size, self.use_kernel = bits, group_size, use_kernel
        self.out_features, self.in_features = linear.weight.shape
        w = linear.weight.detach().float()
        if bits == 8:
            w_q, scale = quantize_int8(w)
            self.register_buffer("w_q", w_q)
            self.register_buffer("scale", scale)
        else:
            packed, scale = quantize_int4(w, group_size)
            self.register_buffer("w_q", packed)
            self.register_buffer("scale", scale)

    def dequant(self) -> torch.Tensor:
        if self.bits == 8:
            return dequantize_int8(self.w_q, self.scale)
        return dequantize_int4(self.w_q, self.scale, self.group_size)

    def weight_bytes(self) -> int:
        return self.w_q.numel() * self.w_q.element_size() + \
            self.scale.numel() * self.scale.element_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.in_features)
        if self.use_kernel:
            from kernels.dequant_matmul import int4_matmul, int8_matmul
            if self.bits == 8:
                y = int8_matmul(flat, self.w_q, self.scale)
            else:
                y = int4_matmul(flat, self.w_q, self.scale, self.group_size)
        else:
            y = flat @ self.dequant().to(flat.dtype).t()
        return y.reshape(*shape[:-1], self.out_features)


def quantize_model(model: Transformer, bits: int = 8, group_size: int = 32,
                   use_kernel: bool = True) -> Transformer:
    """Replace all block linears with QuantLinear (router/embed/head stay fp:
    the router is tiny and precision-critical, and the head is usually tied
    to the embedding)."""
    import copy
    model = copy.deepcopy(model)
    for blk in model.blocks:
        a = blk.attn
        for name in ("wq", "wk", "wv", "wo"):
            setattr(a, name, QuantLinear(getattr(a, name), bits, group_size,
                                         use_kernel))
        mods = blk.mlp.experts if blk.is_moe else [blk.mlp]
        for m in mods:
            for name in ("w_gate", "w_up", "w_down"):
                setattr(m, name, QuantLinear(getattr(m, name), bits,
                                             group_size, use_kernel))
    return model


def model_weight_bytes(model: nn.Module) -> int:
    total = 0
    seen = set()
    for m in model.modules():
        if isinstance(m, QuantLinear):
            total += m.weight_bytes()
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel() * p.element_size()
    return total
