"""Phase 1 correctness: fused Triton attention vs naive PyTorch reference."""

import pytest
import torch

from kernels.attention_fwd import flash_attention
from train.model import naive_attention

torch.manual_seed(0)

SHAPES = [
    # (B, H, Sq, Sk, D) — odd sizes exercise masking; Sq < Sk exercises
    # the decode-with-cache causal alignment.
    (1, 1, 16, 16, 16),
    (1, 2, 17, 17, 16),
    (2, 4, 64, 64, 32),
    (1, 1, 33, 33, 64),
    (1, 2, 5, 37, 16),
    (2, 2, 1, 40, 32),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [True, False])
def test_matches_reference(shape, causal):
    B, H, Sq, Sk, D = shape
    q = torch.randn(B, H, Sq, D)
    k = torch.randn(B, H, Sk, D)
    v = torch.randn(B, H, Sk, D)

    out = flash_attention(q, k, v, causal=causal)
    ref = naive_attention(q, k, v, causal=causal)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_lse_matches_reference():
    B, H, S, D = 1, 2, 33, 16
    q, k, v = (torch.randn(B, H, S, D) for _ in range(3))
    _, lse = flash_attention(q, k, v, causal=True, return_lse=True)

    scale = D ** -0.5
    scores = (q @ k.transpose(-1, -2)) * scale
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    ref_lse = torch.logsumexp(scores.float(), dim=-1)

    torch.testing.assert_close(lse, ref_lse, rtol=1e-4, atol=1e-4)


def test_values_not_just_shape():
    """Guard against a kernel that returns zeros/garbage with correct shape."""
    q, k, v = (torch.randn(1, 1, 32, 16) for _ in range(3))
    out = flash_attention(q, k, v, causal=True)
    assert out.abs().sum() > 0
    # Row 0 with causal mask attends only to key 0 -> output == v[0]
    torch.testing.assert_close(out[0, 0, 0], v[0, 0, 0], rtol=1e-5, atol=1e-5)


def test_scale_override():
    q, k, v = (torch.randn(1, 1, 24, 16) for _ in range(3))
    out = flash_attention(q, k, v, causal=False, scale=0.5)
    ref = naive_attention(q, k, v, causal=False, scale=0.5)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
