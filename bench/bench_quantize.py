"""Phase 4 benchmark: quantized weight memory + fused dequant-matmul latency.

The latency comparison that matters is decode-shaped (M small): fp16 matmul
vs fused INT8 vs fused INT4 vs the anti-pattern (dequantize whole W to fp,
then matmul). Memory footprint numbers are real on any device.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from bench._util import env_tag, save_results, timeit
from kernels.dequant_matmul import int4_matmul, int8_matmul
from serve.quantize import (dequantize_int4, dequantize_int8, model_weight_bytes,
                            quantize_int4, quantize_int8, quantize_model)
from train.config import ModelConfig
from train.model import Transformer


def run(quick: bool = False):
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    dtype = torch.float16 if cuda else torch.float32
    rows = []

    # ---- memory footprint (real on any device) ----
    mc = ModelConfig(vocab_size=256, d_model=512, n_layers=8, n_heads=8,
                     max_seq_len=256)
    model = Transformer(mc)
    for tag, m in [("fp32", model),
                   ("int8", quantize_model(model, bits=8)),
                   ("int4", quantize_model(model, bits=4))]:
        mb = model_weight_bytes(m) / 2**20
        rows.append({"kind": "memory", "variant": tag,
                     "weight_mb": round(mb, 2)})
        print(f"weights {tag:5s} {mb:8.2f} MiB")

    # ---- matmul latency ----
    N = K = 2048 if cuda else 128
    gs = 64 if cuda else 32
    Ms = [1, 16, 128] if cuda else [4]
    for M in Ms:
        a = torch.randn(M, K, device=device, dtype=dtype)
        w = torch.randn(N, K, device=device)
        w_q8, s8 = quantize_int8(w)
        w_q4, s4 = quantize_int4(w, group_size=gs)
        w_fp = w.to(device=device, dtype=dtype)
        w_q8, s8, w_q4, s4 = (t.to(device) for t in (w_q8, s8, w_q4, s4))

        impls = {
            "fp_matmul": lambda: a @ w_fp.t(),
            "dequant_then_matmul_int8":
                lambda: a @ dequantize_int8(w_q8, s8).to(dtype).t(),
            "fused_int8": lambda: int8_matmul(a, w_q8, s8),
            "fused_int4": lambda: int4_matmul(a, w_q4, s4, group_size=gs),
        }
        for name, fn in impls.items():
            lat = timeit(fn, warmup=3, iters=10 if not cuda else 30)
            rows.append({"kind": "latency", "M": M, "N": N, "K": K,
                         "impl": name, "latency_ms": round(lat, 4)})
            print(f"M={M:4d} {name:26s} {lat:9.4f} ms")

    save_results("quantize", rows)


if __name__ == "__main__":
    print(env_tag())
    run(quick="--quick" in sys.argv)
