"""Model-level tests: forward shapes, attn impl equivalence, loss decreases."""

import pytest
import torch

from train.config import ModelConfig, TrainConfig, tiny_config
from train.model import Transformer
from train.train_loop import train

torch.manual_seed(0)


def test_forward_shapes():
    cfg = tiny_config()
    model = Transformer(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 32))
    logits = model(tokens)
    assert logits.shape == (2, 32, cfg.vocab_size)


def test_triton_attention_matches_naive_in_model():
    """Same weights, same input: model output must match across attn impls."""
    cfg_naive = tiny_config(attn_impl="naive")
    cfg_triton = tiny_config(attn_impl="triton")
    torch.manual_seed(42)
    m1 = Transformer(cfg_naive)
    m2 = Transformer(cfg_triton)
    m2.load_state_dict(m1.state_dict())

    tokens = torch.randint(0, cfg_naive.vocab_size, (2, 40))
    with torch.no_grad():
        out1 = m1(tokens)
        out2 = m2(tokens)
    torch.testing.assert_close(out1, out2, rtol=2e-4, atol=2e-4)


def test_causality():
    """Changing a future token must not change past logits."""
    cfg = tiny_config()
    model = Transformer(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, 24))
    with torch.no_grad():
        base = model(tokens)
        mutated = tokens.clone()
        mutated[0, -1] = (mutated[0, -1] + 1) % cfg.vocab_size
        out = model(mutated)
    torch.testing.assert_close(base[:, :-1], out[:, :-1], rtol=1e-5, atol=1e-5)


@pytest.mark.slow
def test_loss_decreases():
    mc = ModelConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=2,
                     max_seq_len=64)
    tc = TrainConfig(steps=80, seq_len=48, batch_size=8, log_every=1000,
                     device="cpu", lr=1e-3)
    result = train(mc, tc, log=lambda *_: None)
    assert result["final_loss"] < result["first_loss"] * 0.8, (
        f"loss did not decrease: {result['first_loss']:.3f} -> "
        f"{result['final_loss']:.3f}")
