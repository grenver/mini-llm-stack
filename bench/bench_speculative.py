"""Phase 6 benchmark: speculative vs plain autoregressive decoding.

Trains a target and a smaller draft on the same synthetic corpus first —
acceptance rate is only meaningful when the draft actually approximates the
target. Reports end-to-end latency, acceptance rate, and tokens/round as a
function of gamma.

Speedup logic: each round costs 1 target forward + gamma draft forwards and
emits (acceptance*gamma + 1) tokens on average. If the draft is much cheaper
than the target and acceptance is high, tokens/sec rises; a bad draft makes
it SLOWER than autoregressive (worth showing honestly).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from bench._util import env_tag, save_results
from serve.speculative import autoregressive_generate, speculative_generate
from train.config import ModelConfig, TrainConfig
from train.train_loop import train


def run(quick: bool = False):
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    steps = 300 if cuda else 120
    target_mc = ModelConfig(vocab_size=64, d_model=256 if cuda else 128,
                            n_layers=6 if cuda else 4, n_heads=4,
                            max_seq_len=512)
    draft_mc = ModelConfig(vocab_size=64, d_model=64, n_layers=1, n_heads=2,
                           max_seq_len=512)
    tc = TrainConfig(steps=steps, seq_len=128, batch_size=8, lr=1e-3,
                     log_every=10_000, device=device)

    print("training target...")
    target = train(target_mc, tc, log=lambda *_: None)["model"].eval()
    print("training draft...")
    result = train(draft_mc, tc, log=lambda *_: None)
    draft = result["model"].eval()
    ds = result["dataset"]

    prompt = ds.data[:32].tolist()
    max_new = 128 if cuda else 48

    t0 = time.perf_counter()
    ref = autoregressive_generate(target, prompt, max_new, device=device)
    ar_wall = time.perf_counter() - t0
    rows = [{"impl": "autoregressive", "gamma": 0, "wall_s": round(ar_wall, 3),
             "tokens_per_s": round(max_new / ar_wall, 2),
             "acceptance_rate": None, "tokens_per_round": 1.0}]
    print(rows[-1])

    for gamma in [2, 4] + ([] if quick else [8]):
        t0 = time.perf_counter()
        out, stats = speculative_generate(target, draft, prompt, max_new,
                                          gamma=gamma)
        wall = time.perf_counter() - t0
        assert out == ref, "greedy speculative output diverged from target!"
        rows.append({"impl": "speculative", "gamma": gamma,
                     "wall_s": round(wall, 3),
                     "tokens_per_s": round(max_new / wall, 2),
                     "acceptance_rate": round(stats.acceptance_rate, 3),
                     "tokens_per_round": round(stats.tokens_per_round, 2)})
        print(rows[-1])

    save_results("speculative", rows)


if __name__ == "__main__":
    print(env_tag())
    run(quick="--quick" in sys.argv)
