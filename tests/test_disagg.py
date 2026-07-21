"""Phase 11: disaggregated prefill/decode must match the unified engine."""

import pytest
import torch

from serve.disagg import run_disaggregated
from serve.engine import Engine
from train.config import tiny_config
from train.model import Transformer


@pytest.mark.slow
def test_disaggregated_matches_unified_engine():
    cfg = tiny_config(d_model=64, n_layers=2, n_heads=2, max_seq_len=128)
    torch.manual_seed(21)
    model = Transformer(cfg).eval()

    prompts = [
        (list(range(1, 9)), 8),
        ([3, 1, 4, 1, 5, 9, 2, 6], 6),
        ([42, 43], 10),
        (list(range(30, 50)), 5),
    ]

    eng = Engine(model, num_blocks=128, block_size=8, max_batch=8,
                 use_kernels=False)
    reqs = [eng.submit(p, max_new_tokens=m) for p, m in prompts]
    eng.run_until_done()
    unified = {r.req_id: r.generated for r in reqs}

    results, _ = run_disaggregated(model, prompts, num_blocks=128,
                                   block_size=8)

    assert set(results.keys()) == set(unified.keys())
    for rid, res in results.items():
        assert res.generated == unified[rid], (
            f"req {rid}: disagg {res.generated} != unified {unified[rid]}")
        assert res.kv_bytes > 0
        assert res.prefill_ms >= 0 and res.decode_ms >= 0
