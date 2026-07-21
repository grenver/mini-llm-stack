"""Phase 4 correctness: quantization roundtrip, fused kernels, model accuracy."""

import pytest
import torch

from kernels.dequant_matmul import int4_matmul, int8_matmul
from serve.quantize import (QuantLinear, dequantize_int4, dequantize_int8,
                            quantize_int4, quantize_int8, quantize_model)
from train.config import ModelConfig, TrainConfig
from train.model import Transformer
from train.train_loop import train

torch.manual_seed(0)


def test_int8_roundtrip_error_bounded():
    w = torch.randn(64, 48)
    w_q, s = quantize_int8(w)
    err = (w - dequantize_int8(w_q, s)).abs()
    assert (err <= s[:, None] / 2 + 1e-6).all()


def test_int4_roundtrip_error_bounded():
    w = torch.randn(32, 64)
    packed, s = quantize_int4(w, group_size=16)
    dq = dequantize_int4(packed, s, group_size=16)
    per_group_bound = s.repeat_interleave(16, dim=1) / 2 + 1e-6
    assert ((w - dq).abs() <= per_group_bound).all()


@pytest.mark.parametrize("M,N,K", [(4, 32, 48), (17, 33, 64), (1, 64, 32)])
def test_int8_fused_matmul_matches_reference(M, N, K):
    a = torch.randn(M, K)
    w = torch.randn(N, K)
    w_q, s = quantize_int8(w)
    out = int8_matmul(a, w_q, s)
    ref = a @ dequantize_int8(w_q, s).t()
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("M,N,K,gs", [(4, 32, 64, 16), (9, 33, 32, 16),
                                      (1, 16, 64, 32)])
def test_int4_fused_matmul_matches_reference(M, N, K, gs):
    a = torch.randn(M, K)
    w = torch.randn(N, K)
    packed, s = quantize_int4(w, group_size=gs)
    out = int4_matmul(a, packed, s, group_size=gs)
    ref = a @ dequantize_int4(packed, s, group_size=gs).t()
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_quantlinear_kernel_vs_reference_path():
    lin = torch.nn.Linear(48, 32, bias=False)
    x = torch.randn(2, 7, 48)
    for bits in (8, 4):
        qk = QuantLinear(lin, bits=bits, group_size=16, use_kernel=True)
        qr = QuantLinear(lin, bits=bits, group_size=16, use_kernel=False)
        torch.testing.assert_close(qk(x), qr(x), rtol=1e-4, atol=1e-4)


@pytest.mark.slow
def test_quantized_model_accuracy():
    """Train briefly, then check quantized loss degradation is bounded.

    Thresholds are generous but real: they catch broken quantization
    (garbage output blows the loss up immediately) while tolerating normal
    rounding damage.
    """
    mc = ModelConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=2,
                     max_seq_len=64)
    tc = TrainConfig(steps=60, seq_len=48, batch_size=8, lr=1e-3,
                     log_every=1000, device="cpu")
    result = train(mc, tc, log=lambda *_: None)
    model, ds = result["model"], result["dataset"]

    torch.manual_seed(9)
    x, y = ds.batch(16)
    with torch.no_grad():
        _, fp_loss, _ = model(x, y)
        _, int8_loss, _ = quantize_model(model, bits=8, use_kernel=False)(x, y)
        _, int4_loss, _ = quantize_model(model, bits=4, group_size=16,
                                         use_kernel=False)(x, y)

    assert int8_loss < fp_loss * 1.03, f"int8 {int8_loss} vs fp {fp_loss}"
    assert int4_loss < fp_loss * 1.30, f"int4 {int4_loss} vs fp {fp_loss}"


def test_quantized_model_memory_shrinks():
    from serve.quantize import model_weight_bytes
    mc = ModelConfig(vocab_size=64, d_model=128, n_layers=2, n_heads=4,
                     max_seq_len=64)
    model = Transformer(mc)
    fp = model_weight_bytes(model)
    q8 = model_weight_bytes(quantize_model(model, bits=8))
    q4 = model_weight_bytes(quantize_model(model, bits=4))
    assert q8 < fp * 0.5   # block linears dominate; embeddings stay fp32
    assert q4 < q8
