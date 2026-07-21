"""Phase 8 benchmark: fwd+bwd step time through custom kernels vs reference.

Compares a full forward+backward of attention through (a) the custom
fwd/bwd Triton kernels and (b) PyTorch autograd over the naive reference.
The flash backward recomputes P twice (once for dK/dV, once for dQ) —
extra FLOPs traded for O(S·d) memory; the reference stores the full [S,S]
probability matrix for backward, so its memory column explodes with S.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from bench._util import env_tag, peak_mem_mb, save_results, timeit
from kernels.autograd_ops import flash_attention_train
from train.model import naive_attention


def run(quick: bool = False):
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    if cuda:
        B, H, D = 4, 8, 64
        seqs = [256, 512, 1024] + ([] if quick else [2048])
        iters = 10
    else:
        B, H, D = 1, 2, 16
        seqs = [32]
        iters = 2

    rows = []
    for S in seqs:
        def make():
            torch.manual_seed(0)
            q = torch.randn(B, H, S, D, device=device, requires_grad=True)
            k = torch.randn(B, H, S, D, device=device, requires_grad=True)
            v = torch.randn(B, H, S, D, device=device, requires_grad=True)
            return q, k, v

        do = torch.randn(B, H, S, D, device=device)

        def step_kernels():
            q, k, v = make()
            flash_attention_train(q, k, v, causal=True).backward(do)

        def step_reference():
            q, k, v = make()
            naive_attention(q, k, v, causal=True).backward(do)

        for name, fn in [("triton_fwd_bwd", step_kernels),
                         ("reference_fwd_bwd", step_reference)]:
            lat = timeit(fn, warmup=2, iters=iters)
            mem = peak_mem_mb(fn)
            rows.append({"seq": S, "impl": name, "latency_ms": round(lat, 3),
                         "peak_mem_mb": round(mem, 1)})
            print(f"S={S:5d} {name:18s} {lat:10.3f} ms  {mem:9.1f} MiB")

    save_results("backward", rows)


if __name__ == "__main__":
    print(env_tag())
    run(quick="--quick" in sys.argv)
