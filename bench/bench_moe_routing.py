"""Phase 2 benchmark: MoE routing overhead, naive loop vs fused dispatch/combine.

Two measurements per (tokens, experts) point:
  * full layer: expert MLPs included (what the model actually pays)
  * routing only: identity experts, isolating pure gather/scatter overhead —
    the naive loop's per-expert index_select/index_add_ launches vs the two
    fused kernels.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from bench._util import env_tag, save_results, timeit
from kernels.moe_routing import moe_dispatch_combine
from train.model import SwiGLU


def naive_route(flat, weights, experts_idx, experts):
    out = torch.zeros_like(flat)
    E = len(experts)
    for e in range(E):
        tok, slot = torch.where(experts_idx == e)
        if tok.numel():
            out.index_add_(0, tok, experts[e](flat[tok]) * weights[tok, slot, None])
    return out


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def run(quick: bool = False):
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    dtype = torch.float16 if cuda else torch.float32
    D, k = (1024, 2) if cuda else (64, 2)
    if cuda:
        points = [(4096, 8), (16384, 8), (16384, 32)]
        if not quick:
            points += [(65536, 8), (65536, 32), (65536, 64)]
        iters = 20
    else:
        points = [(256, 4)]
        iters = 2

    rows = []
    for T, E in points:
        torch.manual_seed(0)
        flat = torch.randn(T, D, device=device, dtype=dtype)
        experts_idx = torch.randint(0, E, (T, k), device=device)
        weights = torch.rand(T, k, device=device, dtype=dtype)
        weights = weights / weights.sum(-1, keepdim=True)

        for tag, mods in [("mlp", [SwiGLU(D, 2 * D).to(device, dtype) for _ in range(E)]),
                          ("routing_only", [Identity() for _ in range(E)])]:
            mods = torch.nn.ModuleList(mods)
            for impl, fn in [
                ("naive", lambda: naive_route(flat, weights, experts_idx, mods)),
                ("triton", lambda: moe_dispatch_combine(flat, weights, experts_idx, mods)),
            ]:
                lat = timeit(fn, warmup=3, iters=iters)
                rows.append({"tokens": T, "experts": E, "top_k": k, "d": D,
                             "experts_kind": tag, "impl": impl,
                             "latency_ms": round(lat, 3)})
                print(f"T={T:6d} E={E:3d} {tag:13s} {impl:7s} {lat:9.3f} ms")

    save_results("moe_routing", rows)


if __name__ == "__main__":
    print(env_tag())
    run(quick="--quick" in sys.argv)
