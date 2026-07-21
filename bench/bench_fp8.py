"""Phase 9 experiment: FP8 training convergence vs fp32 baseline.

Three arms, same init/seed/data: fp32, fp8 with per-tensor dynamic scaling,
fp8 with all casts at scale 1.0 (the ablation). Reported as loss
trajectories — see test_fp8.py for why the ablation is reported rather than
asserted: at toy scale Adam largely absorbs gradient-magnitude damage, so
"unscaled fp8 also trains" is a legitimate (and interesting) finding at
this size. The tensor-level underflow the scaling prevents is demonstrated
in tests/test_fp8.py::test_scale1_underflows_small_values.

Arithmetic is fp32 emulation with REAL e4m3/e5m2 rounding — no FP8 hardware
here (T4 = sm_75); see kernels/fp8_matmul.py.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from bench._util import env_tag, save_results
from train.config import ModelConfig, TrainConfig
from train.fp8 import convert_linears_to_fp8
from train.model import Transformer
from train.train_loop import train


def run(quick: bool = False):
    cuda = torch.cuda.is_available()
    steps = 400 if cuda and not quick else 150
    mc = ModelConfig(vocab_size=64, d_model=128, n_layers=4, n_heads=4,
                     max_seq_len=128)
    tc = TrainConfig(steps=steps, seq_len=96, batch_size=8, lr=1e-3,
                     log_every=10_000, device="cuda" if cuda else "cpu")

    torch.manual_seed(42)
    base = Transformer(mc)

    arms = {
        "fp32": copy.deepcopy(base),
        "fp8_dynamic_scaling": convert_linears_to_fp8(base, enabled=True),
        "fp8_no_scaling": convert_linears_to_fp8(base, enabled=False),
    }

    rows = []
    for name, model in arms.items():
        r = train(mc, tc, model=model, log=lambda *_: None)
        losses = r["losses"]
        checkpoints = {f"loss@{i}": round(losses[min(i, len(losses) - 1)], 4)
                       for i in [0, steps // 4, steps // 2, steps - 1]}
        diverged = any(l != l or l > losses[0] * 3 for l in losses)  # nan/blowup
        rows.append({"arm": name, **checkpoints,
                     "final_loss": round(r["final_loss"], 4),
                     "diverged": diverged})
        print(rows[-1])

    save_results("fp8", rows, extra={
        "note": "fp32 arithmetic with real e4m3/e5m2 rounding (no FP8 "
                "hardware available); convergence comparison, not a speed "
                "benchmark"})


if __name__ == "__main__":
    print(env_tag())
    run(quick="--quick" in sys.argv)
