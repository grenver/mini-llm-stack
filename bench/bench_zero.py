"""Phase 10 benchmark: ZeRO optimizer-state sharding.

Per-rank optimizer state memory is REAL — the moments genuinely live in
separate processes, so the saving is directly measurable even on one
machine. Step-time rows are SIMULATED (ranks share one device; the async
reduce "overlap" hides nothing here) — they measure orchestration overhead
vs a single process, not distributed speedup.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist

from bench._util import RESULTS_DIR, env_tag, save_results
from train.config import ModelConfig
from train.model import Transformer
from train.parallel import init_dist, launch_workers
from train.zero import ZeroAdamW, full_adamw_state_bytes

MC = ModelConfig(vocab_size=256, d_model=256, n_layers=6, n_heads=4,
                 max_seq_len=128)
STEPS = 4


def _zero_worker(rank, world, store_path, out_path, overlap):
    init_dist(rank, world, store_path)
    torch.manual_seed(0)
    model = Transformer(MC)
    opt = ZeroAdamW(model.parameters(), rank, world, lr=1e-3,
                    overlap=overlap)
    tokens = torch.randint(0, MC.vocab_size, (4, 64))
    times = []
    for _ in range(STEPS):
        dist.barrier()
        t0 = time.perf_counter()
        _, loss, _ = model(tokens, tokens)
        opt.zero_grad()
        loss.backward()
        opt.step()
        dist.barrier()
        times.append(time.perf_counter() - t0)
    state_mb = opt.optimizer_state_bytes() / 2**20
    all_mb = [None] * world
    dist.all_gather_object(all_mb, state_mb)
    if rank == 0:
        Path(out_path).write_text(json.dumps({
            "per_rank_state_mb": [round(m, 2) for m in all_mb],
            "step_ms": round(sorted(times)[len(times) // 2] * 1000, 1)}))
    dist.destroy_process_group()


def run(quick: bool = False):
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(0)
    model = Transformer(MC)
    full_mb = full_adamw_state_bytes(model) / 2**20

    # unsharded single-process baseline
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, MC.vocab_size, (4, 64))
    times = []
    for _ in range(STEPS):
        t0 = time.perf_counter()
        _, loss, _ = model(tokens, tokens)
        opt.zero_grad()
        loss.backward()
        opt.step()
        times.append(time.perf_counter() - t0)
    rows = [{"config": "adamw_1proc", "per_rank_state_mb": round(full_mb, 2),
             "step_ms": round(sorted(times)[len(times) // 2] * 1000, 1),
             "memory_real": True, "timing_simulated": False}]
    print(rows[-1])

    worlds = [2] if quick else [2, 4]
    for world in worlds:
        for overlap in ([True] if quick else [True, False]):
            tmp = RESULTS_DIR / "_zero_tmp.json"
            launch_workers(_zero_worker, world, str(tmp), overlap)
            blob = json.loads(tmp.read_text())
            tmp.unlink()
            rows.append({"config": f"zero_w{world}"
                                   f"{'_overlap' if overlap else '_sync'}",
                         "per_rank_state_mb": max(blob["per_rank_state_mb"]),
                         "step_ms": blob["step_ms"],
                         "memory_real": True, "timing_simulated": True})
            print(rows[-1])

    save_results("zero", rows, extra={
        "full_state_mb": round(full_mb, 2),
        "note": "memory columns real (states live in separate processes); "
                "step_ms simulated — ranks share one device"})


if __name__ == "__main__":
    print(env_tag())
    run(quick="--quick" in sys.argv)
