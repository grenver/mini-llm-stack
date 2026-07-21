"""Phase 8 correctness: hand-written backward kernels.

Layers of evidence, weakest to strongest:
  1. analytic grads from the custom Functions == PyTorch autograd grads on
     the unfused reference implementation (fp32, tight tolerance, many
     shapes) — catches math errors, which are O(1), not O(eps);
  2. true finite-difference gradcheck (torch.autograd.gradcheck, fp64) for
     the ops whose kernels preserve fp64 (gather/combine);
  3. coordinate-sampled finite differences for the attention Function
     (its kernels accumulate in fp32 internally, so full fp64 gradcheck
     would report precision noise, not bugs — documented tradeoff);
  4. end-to-end: a model trained THROUGH the kernels matches the reference
     model's gradients step-0 and its loss actually decreases.
"""

import pytest
import torch

from kernels.autograd_ops import (CombineRowsFn, FlashAttentionFn,
                                  GatherRowsFn, Int8MatmulSTE, QuantLinearSTE,
                                  flash_attention_train,
                                  moe_dispatch_combine_train)
from train.config import tiny_config
from train.model import MoE, Transformer, naive_attention

torch.manual_seed(0)


# ------------------------------------------------------------- attention

ATTN_SHAPES = [
    (1, 1, 16, 16, 16),
    (1, 2, 33, 33, 16),
    (2, 2, 48, 48, 32),
    (1, 2, 8, 24, 16),      # Sq < Sk: cached-decode alignment in backward
]


@pytest.mark.parametrize("shape", ATTN_SHAPES)
@pytest.mark.parametrize("causal", [True, False])
def test_attention_grads_match_reference(shape, causal):
    B, H, Sq, Sk, D = shape
    torch.manual_seed(3)
    q = torch.randn(B, H, Sq, D, requires_grad=True)
    k = torch.randn(B, H, Sk, D, requires_grad=True)
    v = torch.randn(B, H, Sk, D, requires_grad=True)
    do = torch.randn(B, H, Sq, D)

    out = flash_attention_train(q, k, v, causal=causal)
    out.backward(do)
    dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    ref = naive_attention(q2, k2, v2, causal=causal)
    ref.backward(do)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dq, q2.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dk, k2.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dv, v2.grad, rtol=1e-4, atol=1e-4)


def test_attention_finite_difference_spot_check():
    """Coordinate-sampled numeric gradcheck of the fused attention Function.

    Catches systematic analytic-gradient errors (wrong scale, shifted mask)
    that agree with nothing; tolerant of fp32 finite-difference noise.
    """
    torch.manual_seed(4)
    B, H, S, D = 1, 1, 20, 16
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    w = torch.randn(B, H, S, D)         # fixed projection -> scalar loss

    def loss_fn(q_, k_, v_):
        return (FlashAttentionFn.apply(q_, k_, v_, True, None) * w).sum()

    qg = q.clone().requires_grad_(True)
    kg = k.clone().requires_grad_(True)
    vg = v.clone().requires_grad_(True)
    loss_fn(qg, kg, vg).backward()

    gen = torch.Generator().manual_seed(0)
    eps = 1e-2
    for tensor, grad in [(q, qg.grad), (k, kg.grad), (v, vg.grad)]:
        flat = tensor.reshape(-1)
        for _ in range(6):
            i = int(torch.randint(0, flat.numel(), (1,), generator=gen))
            orig = flat[i].item()
            flat[i] = orig + eps
            hi = loss_fn(q, k, v).item()
            flat[i] = orig - eps
            lo = loss_fn(q, k, v).item()
            flat[i] = orig
            numeric = (hi - lo) / (2 * eps)
            analytic = grad.reshape(-1)[i].item()
            assert abs(numeric - analytic) < 5e-2 * max(1.0, abs(analytic)), (
                f"coord {i}: numeric {numeric:.5f} vs analytic {analytic:.5f}")


# ------------------------------------------------------------ MoE routing

def test_gather_rows_gradcheck_fp64():
    x = torch.randn(12, 8, dtype=torch.float64, requires_grad=True)
    idx = torch.randint(0, 12, (20,))    # repeats -> exercises atomic adds
    assert torch.autograd.gradcheck(
        lambda t: GatherRowsFn.apply(t, idx), (x,), eps=1e-6)


def test_combine_rows_gradcheck_fp64():
    T, K, D = 6, 2, 8
    eo = torch.randn(T * K, D, dtype=torch.float64, requires_grad=True)
    w = torch.rand(T, K, dtype=torch.float64, requires_grad=True)
    pos = torch.randperm(T * K).reshape(T, K)
    assert torch.autograd.gradcheck(
        lambda a, b: CombineRowsFn.apply(a, b, pos), (eo, w), eps=1e-6)


def test_moe_layer_grads_match_naive():
    """Full MoE layer: custom-kernel path grads == naive-loop path grads
    for input, router weights, and every expert weight."""
    cfg_n = tiny_config(n_experts=4, top_k=2, moe_every=1, routing_impl="naive")
    cfg_t = tiny_config(n_experts=4, top_k=2, moe_every=1, routing_impl="triton")
    torch.manual_seed(7)
    layer_n = MoE(cfg_n)
    layer_t = MoE(cfg_t)
    layer_t.load_state_dict(layer_n.state_dict())

    x = torch.randn(2, 12, cfg_n.d_model)
    xn = x.clone().requires_grad_(True)
    xt = x.clone().requires_grad_(True)
    g = torch.randn(2, 12, cfg_n.d_model)

    layer_n(xn).backward(g)
    layer_t(xt).backward(g)

    torch.testing.assert_close(xt.grad, xn.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(layer_t.router.weight.grad,
                               layer_n.router.weight.grad,
                               rtol=1e-4, atol=1e-4)
    for et, en in zip(layer_t.experts, layer_n.experts):
        for pt, pn in zip(et.parameters(), en.parameters()):
            torch.testing.assert_close(pt.grad, pn.grad, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------- INT8 STE matmul

def test_int8_ste_input_grad_matches_reference():
    from serve.quantize import dequantize_int8, quantize_int8
    torch.manual_seed(1)
    x = torch.randn(9, 32, requires_grad=True)
    w = torch.randn(24, 32)
    w_q, s = quantize_int8(w)
    master = w.clone().requires_grad_(True)

    y = Int8MatmulSTE.apply(x, w_q, s, master)
    dy = torch.randn_like(y)
    y.backward(dy)

    x2 = x.detach().clone().requires_grad_(True)
    ref = x2 @ dequantize_int8(w_q, s).t()
    ref.backward(dy)

    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(x.grad, x2.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(master.grad, dy.t() @ x.detach(),
                               rtol=1e-5, atol=1e-5)


@pytest.mark.slow
def test_quant_ste_linear_learns():
    """Toy regression through the INT8 kernel: loss must collapse."""
    torch.manual_seed(2)
    w_true = torch.randn(16, 32)
    lin = QuantLinearSTE(32, 16)
    opt = torch.optim.Adam(lin.parameters(), lr=3e-2)
    first = last = None
    for step in range(150):
        x = torch.randn(64, 32)
        loss = ((lin(x) - x @ w_true.t()) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first * 0.1, f"loss {first:.3f} -> {last:.3f}"


# ------------------------------------------------------------- end to end

@pytest.mark.slow
def test_model_grads_through_kernels_match_reference():
    """Same weights, same batch: full-model gradients via the custom
    fwd+bwd kernels must match the all-PyTorch reference model."""
    cfg_ref = tiny_config(d_model=32, n_layers=2, n_heads=2, n_experts=2,
                          top_k=2, moe_every=2, attn_impl="naive",
                          routing_impl="naive", max_seq_len=64)
    cfg_ker = tiny_config(d_model=32, n_layers=2, n_heads=2, n_experts=2,
                          top_k=2, moe_every=2, attn_impl="triton",
                          routing_impl="triton", max_seq_len=64)
    torch.manual_seed(11)
    m_ref = Transformer(cfg_ref)
    m_ker = Transformer(cfg_ker)
    m_ker.load_state_dict(m_ref.state_dict())

    tokens = torch.randint(0, cfg_ref.vocab_size, (2, 24))
    for m in (m_ref, m_ker):
        _, loss, aux = m(tokens, tokens)
        (loss + 0.01 * aux).backward()

    mismatches = []
    for (name, p_ref), (_, p_ker) in zip(m_ref.named_parameters(),
                                         m_ker.named_parameters()):
        try:
            torch.testing.assert_close(p_ker.grad, p_ref.grad,
                                       rtol=2e-4, atol=2e-4)
        except AssertionError as e:
            mismatches.append(f"{name}: {e}")
    assert not mismatches, "\n".join(mismatches[:5])


@pytest.mark.slow
def test_training_through_kernels_decreases_loss():
    """The spec's bottom line: numerical closeness isn't enough — train a
    toy model entirely through the custom fwd+bwd kernels and watch loss."""
    from train.config import ModelConfig, TrainConfig
    from train.train_loop import train
    mc = ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=2,
                     max_seq_len=64, attn_impl="triton")
    tc = TrainConfig(steps=30, seq_len=32, batch_size=4, lr=2e-3,
                     log_every=1000, device="cpu")
    result = train(mc, tc, log=lambda *_: None)
    assert result["final_loss"] < result["first_loss"] * 0.85, (
        f"{result['first_loss']:.3f} -> {result['final_loss']:.3f}")
