"""Shared benchmark helpers.

Every bench script dumps a JSON blob into bench/results/ tagged with the
environment (GPU name or "cpu-interpreter"). Report generation (Phase 7)
reads those blobs. Timings collected on CPU/interpreter are flagged
not_meaningful=True and the report renders them as such — interpreter-mode
numbers say nothing about GPU kernel performance.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

RESULTS_DIR = Path(__file__).parent / "results"


def env_tag() -> dict:
    if torch.cuda.is_available():
        return {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "meaningful_timings": True,
        }
    return {
        "device": "cpu (Triton interpreter)",
        "torch": torch.__version__,
        "meaningful_timings": False,
    }


def timeit(fn, warmup: int = 3, iters: int = 20) -> float:
    """Median wall time in milliseconds. Synchronizes CUDA if present."""
    cuda = torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def peak_mem_mb(fn) -> float:
    """Peak CUDA memory in MiB while running fn (nan on CPU)."""
    if not torch.cuda.is_available():
        return float("nan")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def save_results(name: str, rows: list[dict], extra: dict | None = None):
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {"bench": name, "env": env_tag(), "rows": rows}
    if extra:
        payload.update(extra)
    path = RESULTS_DIR / f"{name}.json"

    # Don't let a local CPU run silently clobber real GPU numbers: divert to
    # a sidecar file instead (set BENCH_FORCE_OVERWRITE=1 to override).
    if (path.exists() and not payload["env"]["meaningful_timings"]
            and not payload.get("timings_irrelevant")
            and not os.environ.get("BENCH_FORCE_OVERWRITE")):
        try:
            old = json.loads(path.read_text())
            if old.get("env", {}).get("meaningful_timings"):
                path = RESULTS_DIR / f"{name}.cpu.json"
                print(f"[kept GPU results; CPU run diverted to {path.name}]")
        except (json.JSONDecodeError, OSError):
            pass

    path.write_text(json.dumps(payload, indent=2))
    print(f"[saved] {path}")
    return path
