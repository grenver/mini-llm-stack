"""Phase 12: auto-generated fused kernels vs PyTorch and the hand-written chain."""

import pytest
import torch
import torch.nn.functional as F

from kernels.fusion_compiler import FusedKernel, Graph, softmax_graph

torch.manual_seed(0)


def test_elementwise_chain():
    """relu(a*b + c) — pure pointwise fusion."""
    g = Graph()
    a, b, c = g.input("mn"), g.input("mn"), g.input("mn")
    g.output(g.relu(g.add(g.mul(a, b), c)))
    fk = FusedKernel(g)

    ta, tb, tc = (torch.randn(20, 33) for _ in range(3))
    torch.testing.assert_close(fk(ta, tb, tc), F.relu(ta * tb + tc),
                               rtol=1e-6, atol=1e-6)


def test_scalar_and_unary_ops():
    """exp(-(a - 2.5)) exercises const, neg, sub, exp."""
    g = Graph()
    a = g.input("mn")
    g.output(g.exp(g.neg(g.sub(a, g.const(2.5)))))
    fk = FusedKernel(g)
    ta = torch.randn(7, 19)
    torch.testing.assert_close(fk(ta), torch.exp(-(ta - 2.5)),
                               rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("M,N", [(16, 16), (20, 33), (5, 100), (1, 7)])
def test_softmax_graph_matches_torch(M, N):
    """Row softmax with reductions — masking must keep padded lanes out."""
    fk = FusedKernel(softmax_graph(scale=1.0))
    x = torch.randn(M, N)
    torch.testing.assert_close(fk(x), F.softmax(x, dim=-1),
                               rtol=1e-5, atol=1e-6)


def test_attention_score_chain_matches_handwritten():
    """The Phase 12 validation the spec asks for: auto-generate
    softmax(Q·Kᵀ·scale) and check it against the same quantity computed by
    the hand-written path (naive reference probabilities)."""
    M, N, D = 24, 24, 32
    q = torch.randn(M, D)
    k = torch.randn(N, D)
    scale = D ** -0.5

    fk = FusedKernel(softmax_graph(with_matmul=True, scale=scale))
    auto = fk(q, k)

    ref = F.softmax((q @ k.t()) * scale, dim=-1)
    torch.testing.assert_close(auto, ref, rtol=1e-4, atol=1e-5)

    # cross-check vs the hand-written fused attention: P @ V == flash output
    from kernels.attention_fwd import flash_attention
    v = torch.randn(N, D)
    flash = flash_attention(q[None, None], k[None, None], v[None, None],
                            causal=False)[0, 0]
    torch.testing.assert_close(auto @ v, flash, rtol=1e-4, atol=1e-4)


def test_matmul_odd_k_masking():
    fk = FusedKernel(softmax_graph(with_matmul=True, scale=0.3))
    q = torch.randn(9, 21)          # K=21 not a multiple of BLOCK_K
    k = torch.randn(13, 21)
    ref = F.softmax((q @ k.t()) * 0.3, dim=-1)
    torch.testing.assert_close(fk(q, k), ref, rtol=1e-4, atol=1e-5)


def test_generated_source_is_inspectable():
    fk = FusedKernel(softmax_graph(with_matmul=True, scale=0.5))
    src = fk.source
    assert "tl.dot" in src and "tl.max" in src and "tl.sum" in src
    assert "@triton.jit" in src
    # no unfused temporaries: exactly one store
    assert src.count("tl.store") == 1
