"""Phase 9 correctness: e4m3 kernel decode, scaled matmul, FP8 training."""

import pytest
import torch

from kernels.fp8_matmul import (E4M3_MAX, decode_e4m3, dequantize_e4m3,
                                fp8_matmul, quantize_e4m3)
from train.config import ModelConfig, TrainConfig, tiny_config
from train.fp8 import DynamicScaler, FP8Linear, convert_linears_to_fp8
from train.model import Transformer
from train.train_loop import train

torch.manual_seed(0)


def test_kernel_decode_matches_torch_all_256_patterns():
    """The in-kernel e4m3 bit decode must agree with PyTorch's float8 view
    for every possible byte (NaN patterns 0x7F/0xFF map to 0 in-kernel)."""
    bits = torch.arange(256, dtype=torch.uint8)
    ours = decode_e4m3(bits)
    ref = bits.view(torch.float8_e4m3fn).float()
    nan_mask = ref.isnan()
    assert nan_mask.sum().item() == 2                    # 0x7F and 0xFF
    torch.testing.assert_close(ours[~nan_mask], ref[~nan_mask],
                               rtol=0, atol=0)           # bit-exact
    assert (ours[nan_mask] == 0).all()


def test_quantize_roundtrip_error_bound():
    """e4m3 has a 3-bit mantissa: relative error <= 2^-4 for scaled values."""
    x = torch.randn(1000) * 3
    s = float(x.abs().amax()) / E4M3_MAX
    xh = dequantize_e4m3(quantize_e4m3(x, s), s)
    rel = ((x - xh).abs() / x.abs().clamp(min=1e-6))
    assert rel.median() < 0.04
    assert (xh == 0).float().mean() < 0.01               # no underflow flush


@pytest.mark.parametrize("M,N,K", [(8, 16, 32), (17, 33, 48), (1, 16, 16)])
def test_fp8_matmul_kernel_matches_emulation(M, N, K):
    a = torch.randn(M, K)
    b = torch.randn(N, K)
    sa = float(a.abs().amax()) / E4M3_MAX
    sb = float(b.abs().amax()) / E4M3_MAX
    ab, bb = quantize_e4m3(a, sa), quantize_e4m3(b, sb)
    out = fp8_matmul(ab, bb, sa, sb)
    ref = dequantize_e4m3(ab, sa) @ dequantize_e4m3(bb, sb).t()
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_scale1_underflows_small_values():
    """The failure dynamic scaling exists to prevent: typical gradient-sized
    values flush to zero (or worse) when cast at scale 1.0."""
    g = torch.randn(1000) * 1e-4              # gradient-magnitude values
    dead = dequantize_e4m3(quantize_e4m3(g, 1.0), 1.0)
    scaled_s = float(g.abs().amax()) / E4M3_MAX
    alive = dequantize_e4m3(quantize_e4m3(g, scaled_s), scaled_s)
    assert (dead == 0).float().mean() > 0.5              # mass murdered
    assert (alive == 0).float().mean() < 0.01


def test_delayed_scaler_tracks_amax():
    s = DynamicScaler(history=4)
    t = torch.ones(10)
    s1 = s.scale_for(t * 448)                # first call: JIT amax
    assert abs(s1 - 1.0) < 1e-6
    s2 = s.scale_for(t)                      # history still remembers 448
    assert abs(s2 - 1.0) < 1e-6
    for _ in range(4):                       # roll 448 out of the window
        s.scale_for(t)
    assert s.scale_for(t) < 0.01


def test_fp8_linear_grads_match_emulation_reference():
    torch.manual_seed(1)
    lin = torch.nn.Linear(16, 12, bias=False)
    fp8 = FP8Linear(lin)
    x = torch.randn(5, 16, requires_grad=True)
    y = fp8(x)
    dy = torch.randn_like(y)
    y.backward(dy)

    # reference: same quantized operands through plain autograd
    from train.fp8 import _dequant_e5m2, _quant_e5m2
    sx = fp8.sx.scale_for(x.detach())        # scalers advanced; recompute
    # rebuild exactly what the Function saw on its first call
    fp8_2 = FP8Linear(lin)
    x2 = x.detach().clone().requires_grad_(True)
    sx2 = DynamicScaler().scale_for(x2)
    sw2 = DynamicScaler().scale_for(fp8_2.weight)
    xh = dequantize_e4m3(quantize_e4m3(x2.detach(), sx2), sx2)
    wh = dequantize_e4m3(quantize_e4m3(fp8_2.weight.detach(), sw2), sw2)
    sdy = max(float(dy.abs().amax()) / 57344.0, 1e-12)
    dyh = _dequant_e5m2(_quant_e5m2(dy, sdy), sdy)
    torch.testing.assert_close(x.grad, dyh @ wh, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fp8.weight.grad, dyh.t() @ xh,
                               rtol=1e-5, atol=1e-5)


@pytest.mark.slow
def test_fp8_training_converges_close_to_fp32():
    """The Phase 9 question: does fp8 + dynamic scaling actually train?

    The positive claim is asserted here: scaled fp8 must track the fp32
    baseline. The scaled-vs-unscaled ABLATION is deliberately NOT a hard
    assertion: empirically, at this toy scale (2 layers, d=64, Adam), even
    scale-1.0 casts survive — activations sit ~1 and Adam's per-parameter
    normalization shrugs off gradient-magnitude damage. The underflow
    mechanism itself is proven at tensor level above
    (test_scale1_underflows_small_values); the training-level comparison is
    reported as data by bench/bench_fp8.py rather than asserted.
    """
    mc = ModelConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=2,
                     max_seq_len=64)
    tc = TrainConfig(steps=80, seq_len=48, batch_size=8, lr=1e-3,
                     log_every=10_000, device="cpu")

    torch.manual_seed(42)
    base = Transformer(mc)

    def run(model):
        return train(mc, tc, model=model, log=lambda *_: None)["final_loss"]

    import copy
    fp32_loss = run(copy.deepcopy(base))
    fp8_loss = run(convert_linears_to_fp8(base, enabled=True))

    assert fp8_loss < fp32_loss * 1.15, (
        f"fp8+scaling {fp8_loss:.3f} vs fp32 {fp32_loss:.3f}")
