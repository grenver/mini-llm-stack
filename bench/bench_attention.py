"""Phase 1 benchmark: fused Triton attention vs naive PyTorch vs SDPA.

Measures median latency and peak memory across sequence lengths. The naive
baseline materializes the full [B, H, S, S] score matrix — its memory column
is the point of the comparison: flash-style attention keeps memory flat in S
(beyond the Q/K/V/O tensors themselves) while naive grows O(S^2).

Run on GPU for meaningful numbers; on CPU it runs tiny shapes purely to
exercise the code path.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

from bench._util import env_tag, peak_mem_mb, save_results, timeit
from kernels.attention_fwd import flash_attention
from train.model import naive_attention


def run(quick: bool = False):
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    dtype = torch.float16 if cuda else torch.float32
    if cuda:
        B, H, D = 4, 8, 64
        seqs = [256, 512, 1024, 2048] + ([] if quick else [4096])
        iters = 20
    else:
        B, H, D = 1, 2, 32
        seqs = [64]
        iters = 2

    rows = []
    for S in seqs:
        q, k, v = (torch.randn(B, H, S, D, device=device, dtype=dtype)
                   for _ in range(3))

        impls = {
            "naive_pytorch": lambda: naive_attention(q, k, v, causal=True),
            "sdpa": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
            "triton_flash": lambda: flash_attention(q, k, v, causal=True),
        }
        for name, fn in impls.items():
            try:
                lat = timeit(fn, warmup=2 if not cuda else 5, iters=iters)
                mem = peak_mem_mb(fn)
                rows.append({"seq": S, "impl": name, "latency_ms": round(lat, 3),
                             "peak_mem_mb": round(mem, 1)})
                print(f"S={S:5d} {name:15s} {lat:9.3f} ms  {mem:9.1f} MiB")
            except torch.cuda.OutOfMemoryError:
                rows.append({"seq": S, "impl": name, "latency_ms": None,
                             "peak_mem_mb": None, "oom": True})
                print(f"S={S:5d} {name:15s} OOM")
                torch.cuda.empty_cache()

    save_results("attention", rows)


if __name__ == "__main__":
    print(env_tag())
    run(quick="--quick" in sys.argv)
