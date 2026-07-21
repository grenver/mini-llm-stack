"""Phase 11 benchmark: disaggregated vs unified serving (SIMULATED split).

Both pools time-share one physical device, so disaggregation can only ADD
cost here (KV serialization + queue IPC + a second model replica) — the
point of this benchmark is the overhead breakdown, not a win. Rows report
where each request's wall time went: prefill compute, KV transfer
(serialize + IPC + ingest into the paged pool), decode.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from bench._util import env_tag, save_results
from serve.disagg import run_disaggregated
from serve.engine import Engine
from train.config import ModelConfig
from train.model import Transformer


def run(quick: bool = False):
    cuda = torch.cuda.is_available()
    mc = ModelConfig(vocab_size=256, d_model=128 if not cuda else 512,
                     n_layers=4 if not cuda else 8, n_heads=4,
                     max_seq_len=512)
    torch.manual_seed(0)
    model = Transformer(mc).eval()

    n_req = 4 if quick or not cuda else 16
    g = torch.Generator().manual_seed(2)
    requests = [(torch.randint(0, mc.vocab_size, (48,), generator=g).tolist(), 16)
                for _ in range(n_req)]

    # unified engine baseline
    import time
    eng = Engine(model, num_blocks=1024, block_size=16, max_batch=n_req,
                 use_kernels=False)
    reqs = [eng.submit(p, max_new_tokens=m) for p, m in requests]
    t0 = time.perf_counter()
    eng.run_until_done()
    unified_wall = time.perf_counter() - t0

    results, disagg_wall = run_disaggregated(model, requests,
                                             num_blocks=1024, block_size=16)

    total_kv_mb = sum(r.kv_bytes for r in results.values()) / 2**20
    rows = [
        {"config": "unified_engine", "wall_s": round(unified_wall, 3),
         "kv_transferred_mb": 0.0, "mean_transfer_ms": 0.0},
        {"config": "disaggregated_simulated", "wall_s": round(disagg_wall, 3),
         "kv_transferred_mb": round(total_kv_mb, 2),
         "mean_transfer_ms": round(sum(r.transfer_ms for r in results.values())
                                   / len(results), 2)},
    ]
    for r in rows:
        print(r)
    breakdown = {
        "mean_prefill_ms": round(sum(r.prefill_ms for r in results.values())
                                 / len(results), 2),
        "mean_transfer_ms": rows[1]["mean_transfer_ms"],
        "mean_decode_ms": round(sum(r.decode_ms for r in results.values())
                                / len(results), 2),
    }
    print("overhead breakdown:", breakdown)
    save_results("disagg", rows, extra={
        "simulated": True, "per_request_breakdown": breakdown,
        "note": "both pools share one device: the split can only add "
                "overhead in this environment; breakdown shows where it goes"})


if __name__ == "__main__":
    print(env_tag())
    run(quick="--quick" in sys.argv)
