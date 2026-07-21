"""Phase 3 benchmark: SIMULATED parallel step times.

All ranks share ONE physical device (that is the whole compute budget), so
these numbers measure orchestration + communication overhead only — they can
NOT show real parallel speedup and are tagged simulated=True in the results.
On real hardware TP/PP would distribute the FLOPs; here the same FLOPs are
serialized onto one device plus IPC cost, so parallel configs are expected
to be SLOWER than dense. The interesting output is the overhead breakdown,
not a scaling curve.
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
from train.parallel import (PipelineStage, init_dist, launch_workers,
                            pipeline_run, shard_model)

MC = ModelConfig(vocab_size=256, d_model=256, n_layers=4, n_heads=4,
                 max_seq_len=256, tie_embeddings=False)
BATCH, SEQ, STEPS = 4, 128, 5


def _step_dense():
    torch.manual_seed(0)
    model = Transformer(MC)
    tokens = torch.randint(0, MC.vocab_size, (BATCH, SEQ))
    times = []
    for _ in range(STEPS):
        t0 = time.perf_counter()
        _, loss, _ = model(tokens, tokens)
        loss.backward()
        model.zero_grad(set_to_none=True)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2] * 1000


def _tp_bench_worker(rank, world, store_path, out_path):
    init_dist(rank, world, store_path)
    torch.manual_seed(0)
    dense = Transformer(MC)
    model = shard_model(dense, rank, world)
    tokens = torch.randint(0, MC.vocab_size, (BATCH, SEQ))
    times = []
    for _ in range(STEPS):
        dist.barrier()
        t0 = time.perf_counter()
        _, loss, _ = model(tokens, tokens)
        loss.backward()
        model.zero_grad(set_to_none=True)
        dist.barrier()
        times.append(time.perf_counter() - t0)
    if rank == 0:
        Path(out_path).write_text(json.dumps(
            {"tp_step_ms": sorted(times)[len(times) // 2] * 1000}))
    dist.destroy_process_group()


def _pp_bench_worker(rank, world, store_path, out_path):
    init_dist(rank, world, store_path)
    torch.manual_seed(0)
    dense = Transformer(MC)
    stage = PipelineStage(dense, rank, world)
    tokens = torch.randint(0, MC.vocab_size, (BATCH, SEQ))
    times = []
    for _ in range(STEPS):
        dist.barrier()
        t0 = time.perf_counter()
        pipeline_run(stage, tokens, tokens, n_microbatches=2,
                     d_model=MC.d_model)
        stage.zero_grad(set_to_none=True)
        dist.barrier()
        times.append(time.perf_counter() - t0)
    if rank == 0:
        Path(out_path).write_text(json.dumps(
            {"pp_step_ms": sorted(times)[len(times) // 2] * 1000}))
    dist.destroy_process_group()


def run():
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = [{"config": "dense_1proc", "step_ms": round(_step_dense(), 1)}]

    tmp = RESULTS_DIR / "_tp_tmp.json"
    launch_workers(_tp_bench_worker, 2, str(tmp))
    rows.append({"config": "tp2_simulated",
                 "step_ms": round(json.loads(tmp.read_text())["tp_step_ms"], 1)})
    tmp.unlink()

    tmp = RESULTS_DIR / "_pp_tmp.json"
    launch_workers(_pp_bench_worker, 2, str(tmp))
    rows.append({"config": "pp2_mb2_simulated",
                 "step_ms": round(json.loads(tmp.read_text())["pp_step_ms"], 1)})
    tmp.unlink()

    for r in rows:
        print(r)
    save_results("parallel", rows, extra={"simulated": True, "note":
                 "all ranks share one physical device; no real parallel "
                 "speedup is possible in this environment"})


if __name__ == "__main__":
    print(env_tag())
    run()
