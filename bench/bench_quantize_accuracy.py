"""Phase 4 accuracy: quantized model quality degradation (device-independent).

Trains the toy char model briefly, then evaluates held-out loss / perplexity
under fp32, INT8 (per-row) and INT4 (group-wise) weights. Completes the
Phase 4 story: memory and latency say nothing if quantization wrecked the
model. Pure numerics — CPU and GPU produce the same answer, so this runs
anywhere and is flagged timings_irrelevant.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from bench._util import env_tag, save_results
from serve.quantize import quantize_model
from train.config import ModelConfig, TrainConfig
from train.train_loop import train


def run(quick: bool = False):
    mc = ModelConfig(vocab_size=64, d_model=128, n_layers=4, n_heads=4,
                     max_seq_len=128)
    tc = TrainConfig(steps=80 if quick else 200, seq_len=96, batch_size=8,
                     lr=1e-3, log_every=10_000, device="cpu")
    torch.manual_seed(0)
    result = train(mc, tc, log=lambda *_: None)
    model, ds = result["model"], result["dataset"]

    def eval_loss(m) -> float:
        g_state = ds.g.get_state()
        ds.g.manual_seed(1234)              # same held-out batches for all
        losses = []
        with torch.no_grad():
            for _ in range(8):
                x, y = ds.batch(16)
                _, loss, _ = m(x, y)
                losses.append(loss.item())
        ds.g.set_state(g_state)
        return sum(losses) / len(losses)

    fp32 = eval_loss(model)
    rows = []
    for tag, m in [("fp32", model),
                   ("int8", quantize_model(model, bits=8, use_kernel=False)),
                   ("int4_g32", quantize_model(model, bits=4, group_size=32,
                                               use_kernel=False))]:
        loss = eval_loss(m)
        rows.append({"variant": tag, "val_loss": round(loss, 4),
                     "perplexity": round(math.exp(loss), 3),
                     "loss_increase_pct": round(100 * (loss - fp32) / fp32, 2)})
        print(rows[-1])

    save_results("quantize_accuracy", rows,
                 extra={"timings_irrelevant": True,
                        "note": "held-out loss on the toy char corpus after "
                                f"{tc.steps} training steps; reference path "
                                "(kernel and reference give identical logits "
                                "— see tests/test_quantize.py)"})


if __name__ == "__main__":
    print(env_tag())
    run(quick="--quick" in sys.argv)
