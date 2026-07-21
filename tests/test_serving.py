"""Phase 5 correctness: block manager, paged-attention kernel, full engine.

The engine test is the load-bearing one: continuous batching + paged cache +
paged attention must reproduce naive full-reforward generation token for
token, across concurrent requests with different prompt lengths.
"""

import pytest
import torch

from kernels.paged_attention import paged_attention_decode
from serve.engine import Engine
from serve.kv_cache import PagedKVCache
from train.config import tiny_config
from train.model import Transformer, naive_attention

torch.manual_seed(0)


# ------------------------------------------------------------ block manager

def test_block_accounting():
    c = PagedKVCache(n_layers=1, n_heads=2, head_dim=8, num_blocks=8,
                     block_size=4)
    assert c.can_allocate(32) and not c.can_allocate(33)
    c.allocate(0, 9)                       # 3 blocks
    assert c.num_free_blocks() == 5
    c.allocate(1, 4)                       # 1 block
    assert c.num_free_blocks() == 4
    c.free(0)
    assert c.num_free_blocks() == 7
    c.free(1)
    assert c.num_free_blocks() == 8


def test_append_slot_claims_block_only_at_boundary():
    c = PagedKVCache(1, 2, 8, num_blocks=4, block_size=4)
    c.allocate(0, 3)
    c.set_len(0, 3)
    assert c.append_slot(0) and len(c.tables[0]) == 1     # slot 4 fits block 1
    c.set_len(0, 4)
    assert c.append_slot(0) and len(c.tables[0]) == 2     # slot 5 needs block 2
    # exhaust the pool: 2 free blocks left
    c.allocate(1, 8)
    c.set_len(1, 8)
    assert not c.append_slot(1)                            # no free blocks


def test_write_read_roundtrip():
    c = PagedKVCache(2, 2, 8, num_blocks=8, block_size=4)
    c.allocate(0, 10)
    k = torch.randn(2, 10, 8)
    v = torch.randn(2, 10, 8)
    for layer in range(2):
        c.write_prefill(layer, 0, k, v)
    c.set_len(0, 10)
    kk, vv = c.gather_contiguous(1, 0)
    torch.testing.assert_close(kk, k)
    torch.testing.assert_close(vv, v)


# ------------------------------------------------------ paged attention kernel

@pytest.mark.parametrize("lens", [[7], [16], [5, 23, 12], [1, 1, 40]])
def test_paged_attention_matches_contiguous(lens):
    H, D, bs = 4, 32, 8
    n_seqs = len(lens)
    cache = PagedKVCache(1, H, D, num_blocks=32, block_size=bs)
    ks, vs = [], []
    for sid, L in enumerate(lens):
        cache.allocate(sid, L)
        k = torch.randn(H, L, D)
        v = torch.randn(H, L, D)
        cache.write_prefill(0, sid, k, v)
        cache.set_len(sid, L)
        ks.append(k)
        vs.append(v)

    q = torch.randn(n_seqs, H, D)
    tables, ctx = cache.batch_tables(list(range(n_seqs)))
    out = paged_attention_decode(q, cache.k_pool[0], cache.v_pool[0],
                                 tables, ctx)

    for i, L in enumerate(lens):
        ref = naive_attention(q[i][:, None], ks[i], vs[i], causal=False)[:, 0]
        torch.testing.assert_close(out[i], ref, rtol=1e-4, atol=1e-4)


def test_paged_attention_respects_context_len():
    """Tokens past ctx in a partially-filled block must not leak in."""
    H, D, bs = 2, 16, 8
    cache = PagedKVCache(1, H, D, num_blocks=4, block_size=bs)
    cache.allocate(0, 5)
    k = torch.randn(H, 5, D)
    v = torch.randn(H, 5, D)
    cache.write_prefill(0, 0, k, v)
    cache.set_len(0, 5)
    # poison the unused slots of the block
    blk = cache.tables[0][0]
    cache.k_pool[0][blk, :, 5:] = 1e6
    cache.v_pool[0][blk, :, 5:] = 1e6

    q = torch.randn(1, H, D)
    tables, ctx = cache.batch_tables([0])
    out = paged_attention_decode(q, cache.k_pool[0], cache.v_pool[0],
                                 tables, ctx)
    ref = naive_attention(q[0][:, None], k, v, causal=False)[:, 0]
    torch.testing.assert_close(out[0], ref, rtol=1e-4, atol=1e-4)


# ------------------------------------------------------------- full engine

@pytest.mark.slow
def test_engine_matches_naive_generation():
    cfg = tiny_config(d_model=64, n_layers=2, n_heads=2, max_seq_len=128)
    torch.manual_seed(5)
    model = Transformer(cfg).eval()

    prompts = [
        list(range(1, 9)),
        [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],
        [42],
        list(range(20, 45)),
    ]
    max_new = 12

    refs = []
    for p in prompts:
        t = torch.tensor([p])
        refs.append(model.generate_naive(t, max_new)[0, len(p):].tolist())

    # use_kernels=False: reference gather path (fast on CPU). The kernel path
    # is covered by test_paged_attention_* and the GPU suite.
    eng = Engine(model, num_blocks=64, block_size=8, max_batch=4,
                 use_kernels=False)
    reqs = [eng.submit(p, max_new_tokens=max_new) for p in prompts]
    eng.run_until_done()

    for req, ref in zip(reqs, refs):
        assert req.generated == ref, (
            f"req {req.req_id}: engine {req.generated} != naive {ref}")


@pytest.mark.slow
def test_engine_kernel_path_matches_naive_generation():
    """Same check through the actual Triton kernels (interpreter on CPU)."""
    cfg = tiny_config(d_model=32, n_layers=1, n_heads=2, max_seq_len=64)
    torch.manual_seed(6)
    model = Transformer(cfg).eval()
    prompts = [[5, 6, 7], [9, 8, 7, 6, 5, 4]]
    max_new = 5

    refs = [model.generate_naive(torch.tensor([p]), max_new)[0, len(p):].tolist()
            for p in prompts]

    eng = Engine(model, num_blocks=32, block_size=8, max_batch=4,
                 use_kernels=True)
    reqs = [eng.submit(p, max_new_tokens=max_new) for p in prompts]
    eng.run_until_done()
    for req, ref in zip(reqs, refs):
        assert req.generated == ref


@pytest.mark.slow
def test_continuous_batching_admits_when_blocks_free():
    """More requests than cache capacity: later ones must wait, then run."""
    cfg = tiny_config(d_model=32, n_layers=1, n_heads=2, max_seq_len=64)
    model = Transformer(cfg).eval()
    # tiny pool: 8 blocks of 4 = 32 token slots total
    eng = Engine(model, num_blocks=8, block_size=4, max_batch=8,
                 use_kernels=False)
    reqs = [eng.submit([1, 2, 3, 4, 5], max_new_tokens=6) for _ in range(6)]
    stats = eng.run_until_done()
    assert all(r.state == "finished" for r in reqs)
    assert all(len(r.generated) == 6 for r in reqs)
    assert stats["requests"] == 6
    # sanity: with 32 slots and 11-token footprints, not all 6 fit at once,
    # so the engine must have taken more steps than one static batch would
    assert eng.steps > 6
