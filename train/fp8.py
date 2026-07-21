"""FP8 mixed-precision training with per-tensor dynamic (delayed) scaling.

The problem dynamic scaling solves: e4m3's smallest normal is 2⁻⁶ ≈ 0.016
and max is 448. Transformer activations/weights (~1e-2) and especially
gradients (~1e-4..1e-6) sit mostly BELOW the normal range — cast naively at
scale 1.0 and most values land in subnormals (3 bits of resolution) or
flush to zero, and training stalls. Per-tensor scaling maps each tensor's
observed amax to the top of the fp8 range before casting.

Delayed scaling (the Transformer Engine recipe, simplified): the scale for
step t comes from the max of an amax HISTORY of previous steps — cheap
(no extra pass over the tensor before casting) but can clip after a sudden
amax jump; the history window absorbs that. First step falls back to
just-in-time amax.

Forward tensors use e4m3 (more mantissa); gradients use e5m2 (more range) —
matching production practice. Weight master copies stay fp32; the optimizer
never sees fp8.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn

from kernels.fp8_matmul import E4M3_MAX, dequantize_e4m3, fp8_matmul, quantize_e4m3

E5M2_MAX = 57344.0


class DynamicScaler:
    """Per-tensor delayed scaling: scale_t = max(amax history) / fp8_max."""

    def __init__(self, fp8_max: float = E4M3_MAX, history: int = 16,
                 enabled: bool = True):
        self.fp8_max = fp8_max
        self.hist: deque = deque(maxlen=history)
        self.enabled = enabled

    def scale_for(self, t: torch.Tensor) -> float:
        if not self.enabled:
            return 1.0
        amax = float(t.detach().abs().amax())
        if not self.hist:                     # first call: just-in-time
            self.hist.append(max(amax, 1e-12))
        s = max(self.hist) / self.fp8_max
        self.hist.append(max(amax, 1e-12))
        return max(s, 1e-12)


def _quant_e5m2(x: torch.Tensor, scale: float) -> torch.Tensor:
    q = (x.float() / scale).clamp(-E5M2_MAX, E5M2_MAX)
    return q.to(torch.float8_e5m2).view(torch.uint8)


def _dequant_e5m2(bits: torch.Tensor, scale: float) -> torch.Tensor:
    return bits.view(torch.float8_e5m2).float() * scale


class _FP8LinearFn(torch.autograd.Function):
    """y = fp8(x) @ fp8(W)ᵀ. Backward re-quantizes dy to e5m2 and computes
    both grads against the SAME fp8 operands the forward used (true fp8
    training semantics, not fp32-with-noise)."""

    @staticmethod
    def forward(ctx, x, w, sx, sw, use_kernel, scale_grads):
        xb = quantize_e4m3(x, sx)
        wb = quantize_e4m3(w, sw)
        ctx.save_for_backward(xb, wb)
        ctx.scales = (sx, sw)
        ctx.scale_grads = scale_grads
        if use_kernel:
            y = fp8_matmul(xb, wb, sx, sw)
        else:                                  # emulation path (fast on CPU)
            y = (dequantize_e4m3(xb, sx)) @ (dequantize_e4m3(wb, sw)).t()
        return y

    @staticmethod
    def backward(ctx, dy):
        xb, wb = ctx.saved_tensors
        sx, sw = ctx.scales
        # gradients get e5m2 (range over precision) with just-in-time scale;
        # the no-scaling ablation casts them at 1.0 like everything else
        if ctx.scale_grads:
            sdy = max(float(dy.detach().abs().amax()) / E5M2_MAX, 1e-12)
        else:
            sdy = 1.0
        dyh = _dequant_e5m2(_quant_e5m2(dy, sdy), sdy)
        xh = dequantize_e4m3(xb, sx)
        wh = dequantize_e4m3(wb, sw)
        dx = dyh @ wh
        dw = dyh.t() @ xh
        return dx, dw, None, None, None, None


class FP8Linear(nn.Module):
    def __init__(self, linear: nn.Linear, enabled: bool = True,
                 use_kernel: bool = False):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.sx = DynamicScaler(enabled=enabled)
        self.sw = DynamicScaler(enabled=enabled)
        self.use_kernel = use_kernel

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        y = _FP8LinearFn.apply(flat, self.weight,
                               self.sx.scale_for(flat),
                               self.sw.scale_for(self.weight),
                               self.use_kernel, self.sx.enabled)
        return y.reshape(*shape[:-1], -1)


def convert_linears_to_fp8(model: nn.Module, enabled: bool = True,
                           use_kernel: bool = False) -> nn.Module:
    """Swap every block linear (attn + MLP/experts) for FP8Linear.
    Embedding/head/norms stay fp32, like real mixed-precision recipes."""
    import copy
    from train.model import Attention, SwiGLU
    model = copy.deepcopy(model)
    for m in model.modules():
        if isinstance(m, Attention):
            for name in ("wq", "wk", "wv", "wo"):
                setattr(m, name, FP8Linear(getattr(m, name), enabled, use_kernel))
        elif isinstance(m, SwiGLU):
            for name in ("w_gate", "w_up", "w_down"):
                setattr(m, name, FP8Linear(getattr(m, name), enabled, use_kernel))
    return model
