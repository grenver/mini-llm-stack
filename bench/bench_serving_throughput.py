"""Phase 5 benchmark: continuous batching engine vs sequential naive generate.

Baseline = one-request-at-a-time full-reforward generation (each token
recomputes the whole prefix — no KV cache at all), which is what a plain
`model.generate()` loop does. The engine amortizes weight reads across the
batch and reuses cached KV, so tokens/s should scale with concurrency until
compute saturates.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from bench._util import env_tag, save_results
from serve.engine import Engine
from train.config import ModelConfig
from train.model import Transformer


def run(quick: bool = False):
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    if cuda:
        mc = ModelConfig(vocab_size=512, d_model=512, n_layers=8, n_heads=8,
                         max_seq_len=1024)
        n_requests, max_new, prompt_len = 32, 64, 64
        num_blocks, max_batch = 2048, 32
    else:
        mc = ModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=2,
                         max_seq_len=256)
        n_requests, max_new, prompt_len = 6, 16, 12
        num_blocks, max_batch = 256, 8

    torch.manual_seed(0)
    model = Transformer(mc).eval().to(device)
    g = torch.Generator().manual_seed(1)
    prompts = [torch.randint(0, mc.vocab_size, (prompt_len,), generator=g).tolist()
               for _ in range(n_requests)]

    # ---- baseline: sequential, full re-forward each token ----
    t0 = time.perf_counter()
    with torch.no_grad():
        for p in prompts:
            model.generate_naive(torch.tensor([p], device=device), max_new)
    if cuda:
        torch.cuda.synchronize()
    seq_wall = time.perf_counter() - t0
    seq_tps = n_requests * max_new / seq_wall

    rows = [{"impl": "sequential_naive", "requests": n_requests,
             "max_new": max_new, "wall_s": round(seq_wall, 3),
             "tokens_per_s": round(seq_tps, 2)}]
    print(rows[-1])

    # ---- engine: continuous batching + paged KV (+ kernels on GPU) ----
    for concurrency in ([max_batch] if quick else [1, max_batch // 4, max_batch]):
        eng = Engine(model, num_blocks=num_blocks, block_size=16,
                     max_batch=concurrency, use_kernels=cuda, device=device,
                     dtype=torch.float32)
        for p in prompts:
            eng.submit(p, max_new_tokens=max_new)
        stats = eng.run_until_done(max_steps=100_000)
        rows.append({"impl": f"engine_batch{concurrency}",
                     "requests": n_requests, "max_new": max_new,
                     "wall_s": round(stats["wall_s"], 3),
                     "tokens_per_s": round(stats["tokens_per_s"], 2)})
        print(rows[-1])

    save_results("serving_throughput", rows)


if __name__ == "__main__":
    print(env_tag())
    run(quick="--quick" in sys.argv)
