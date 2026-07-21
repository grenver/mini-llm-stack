"""torch.autograd.Function wrappers: custom forward AND backward kernels.

These make the model trainable end-to-end through the Triton kernels.
Every Function here pairs a Phase 1-4 forward kernel with its hand-written
backward kernel(s); PyTorch autograd only ever sees opaque (input, output,
grad) triples, so a wrong backward is a *silent* bug — the correctness
gate is tests/test_backward.py (gradcheck + grads-vs-reference + does the
loss actually go down).
"""

from __future__ import annotations

import torch

from kernels.attention_bwd import flash_attention_backward
from kernels.attention_fwd import flash_attention
from kernels.dequant_matmul import int8_dgrad, int8_matmul
from kernels.moe_backward import combine_backward, scatter_add_rows
from kernels.moe_routing import combine_rows, gather_rows, sort_by_expert


class FlashAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, scale):
        o, lse = flash_attention(q, k, v, causal=causal, scale=scale,
                                 return_lse=True)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal, ctx.scale = causal, scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = flash_attention_backward(q, k, v, o, lse, do,
                                              causal=ctx.causal,
                                              scale=ctx.scale)
        return dq, dk, dv, None, None


def flash_attention_train(q, k, v, causal=True, scale=None):
    return FlashAttentionFn.apply(q, k, v, causal, scale)


class GatherRowsFn(torch.autograd.Function):
    """out[i] = x[idx[i]]; backward is atomic scatter-add (rows repeat)."""

    @staticmethod
    def forward(ctx, x, idx):
        ctx.save_for_backward(idx)
        ctx.n_rows = x.shape[0]
        return gather_rows(x, idx)

    @staticmethod
    def backward(ctx, dout):
        (idx,) = ctx.saved_tensors
        return scatter_add_rows(dout, idx, ctx.n_rows), None


class CombineRowsFn(torch.autograd.Function):
    """out[t] = Σ_s w[t,s] · eo[pos[t,s]]"""

    @staticmethod
    def forward(ctx, expert_out, weights, pos):
        ctx.save_for_backward(expert_out, weights, pos)
        return combine_rows(expert_out, weights, pos)

    @staticmethod
    def backward(ctx, dout):
        expert_out, weights, pos = ctx.saved_tensors
        deo, dw = combine_backward(dout, expert_out, weights, pos)
        return deo, dw, None


def moe_dispatch_combine_train(flat, weights, experts_idx, expert_modules):
    """Differentiable MoE forward through the custom kernels.

    Kernels handle gather/combine (and their backwards); the expert MLPs
    stay ordinary autograd modules operating on contiguous slices.
    """
    n_experts = len(expert_modules)
    token_of_slot, counts, pos = sort_by_expert(experts_idx, n_experts)
    dispatched = GatherRowsFn.apply(flat, token_of_slot)

    outs = []
    start = 0
    for e, cnt in enumerate(counts.tolist()):
        if cnt:
            outs.append(expert_modules[e](dispatched[start:start + cnt]))
        start += cnt
    expert_out = torch.cat(outs, dim=0) if outs else dispatched[:0]

    return CombineRowsFn.apply(expert_out, weights, pos)


class Int8MatmulSTE(torch.autograd.Function):
    """y = x @ dequant(W_q)ᵀ with straight-through weight gradients.

    Forward uses the fused INT8 kernel. Backward:
      dx = (dy ∘ s) @ W_q            — custom kernel (the quantized operand)
      dW_master = dyᵀ @ x            — plain fp GEMM: nothing quantized in
                                       it, so no custom kernel is warranted;
                                       STE passes it to the fp32 master
                                       weight unchanged.
    """

    @staticmethod
    def forward(ctx, x, w_q, scale, master_weight):
        ctx.save_for_backward(x, w_q, scale)
        return int8_matmul(x, w_q, scale)

    @staticmethod
    def backward(ctx, dy):
        x, w_q, scale = ctx.saved_tensors
        dy = dy.contiguous()
        dx = int8_dgrad(dy, w_q, scale)
        dw = dy.t() @ x                       # gradient for the fp32 master
        return dx, None, None, dw


class QuantLinearSTE(torch.nn.Module):
    """Train-time quantized linear: fp32 master weights, INT8 forward."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, in_features) * 0.02)

    def forward(self, x):
        from serve.quantize import quantize_int8
        w_q, scale = quantize_int8(self.weight.detach())
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        y = Int8MatmulSTE.apply(flat, w_q, scale, self.weight)
        return y.reshape(*shape[:-1], -1)
