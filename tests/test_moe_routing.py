"""Phase 2 correctness: Triton MoE dispatch/combine vs naive reference."""

import pytest
import torch

from kernels.moe_routing import (combine_rows, gather_rows,
                                 moe_dispatch_combine, sort_by_expert)
from train.config import tiny_config
from train.model import MoE, SwiGLU, Transformer

torch.manual_seed(0)


def test_gather_rows_matches_index_select():
    x = torch.randn(50, 33)
    idx = torch.randint(0, 50, (77,))
    torch.testing.assert_close(gather_rows(x, idx), x.index_select(0, idx))


def test_combine_rows_matches_reference():
    T, K, D = 40, 2, 24
    expert_out = torch.randn(T * K, D)
    weights = torch.rand(T, K)
    pos = torch.randperm(T * K).reshape(T, K)
    out = combine_rows(expert_out, weights, pos)
    ref = (expert_out[pos] * weights[..., None]).sum(dim=1)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_sort_by_expert_roundtrip():
    T, k, E = 30, 2, 4
    experts_idx = torch.randint(0, E, (T, k))
    token_of_slot, counts, pos = sort_by_expert(experts_idx, E)
    assert counts.sum().item() == T * k
    # pos must be the inverse mapping: sorted_buffer[pos[t,s]] holds token t
    flat_expert = experts_idx.reshape(-1)
    sorted_experts = flat_expert[torch.argsort(flat_expert, stable=True)]
    for t in range(T):
        for s in range(k):
            assert token_of_slot[pos[t, s]].item() == t
            assert sorted_experts[pos[t, s]].item() == experts_idx[t, s].item()


@pytest.mark.parametrize("T,E,k,D", [(64, 4, 2, 32), (37, 8, 1, 16),
                                     (128, 8, 2, 64), (16, 2, 2, 24)])
def test_dispatch_combine_matches_naive(T, E, k, D):
    experts = torch.nn.ModuleList([SwiGLU(D, 2 * D) for _ in range(E)])
    flat = torch.randn(T, D)
    experts_idx = torch.randint(0, E, (T, k))
    # ensure distinct experts per token like topk would give (when k>1)
    if k == 2:
        experts_idx[:, 1] = (experts_idx[:, 0] + 1 +
                             torch.randint(0, E - 1, (T,))) % E
    weights = torch.rand(T, k)
    weights = weights / weights.sum(-1, keepdim=True)

    out = moe_dispatch_combine(flat, weights, experts_idx, experts)

    ref = torch.zeros_like(flat)
    for e in range(E):
        tok, slot = torch.where(experts_idx == e)
        if tok.numel():
            ref.index_add_(0, tok, experts[e](flat[tok]) * weights[tok, slot, None])

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_all_tokens_one_expert():
    """Degenerate routing: every token to expert 0; others must stay unused."""
    T, E, D = 20, 4, 16
    experts = torch.nn.ModuleList([SwiGLU(D, 32) for _ in range(E)])
    flat = torch.randn(T, D)
    experts_idx = torch.zeros(T, 1, dtype=torch.long)
    weights = torch.ones(T, 1)
    out = moe_dispatch_combine(flat, weights, experts_idx, experts)
    torch.testing.assert_close(out, experts[0](flat), rtol=1e-5, atol=1e-5)


def test_zero_weights_zero_output():
    T, E, D = 10, 2, 16
    experts = torch.nn.ModuleList([SwiGLU(D, 32) for _ in range(E)])
    out = moe_dispatch_combine(torch.randn(T, D), torch.zeros(T, 2),
                               torch.randint(0, E, (T, 2)), experts)
    assert out.abs().max().item() == 0.0


def test_moe_layer_triton_matches_naive():
    """Full MoE layer inside the model: triton routing == naive routing."""
    cfg_n = tiny_config(n_experts=4, top_k=2, moe_every=1, routing_impl="naive")
    cfg_t = tiny_config(n_experts=4, top_k=2, moe_every=1, routing_impl="triton")
    torch.manual_seed(3)
    layer_n = MoE(cfg_n)
    layer_t = MoE(cfg_t)
    layer_t.load_state_dict(layer_n.state_dict())

    x = torch.randn(2, 16, cfg_n.d_model)
    with torch.no_grad():
        torch.testing.assert_close(layer_t(x), layer_n(x), rtol=1e-4, atol=1e-4)


def test_moe_model_trains():
    """MoE transformer end-to-end forward/backward runs and aux loss is live."""
    cfg = tiny_config(n_experts=4, top_k=2, moe_every=1)
    model = Transformer(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 24))
    _, loss, aux = model(tokens, tokens)
    assert aux.item() > 0
    (loss + 0.01 * aux).backward()
    router_grad = model.blocks[0].mlp.router.weight.grad
    assert router_grad is not None and router_grad.abs().sum() > 0
