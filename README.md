# mini-llm-stack

A small-scale, end-to-end LLM stack that mirrors real production systems:
a from-scratch MoE transformer with (simulated) multi-GPU training
parallelism, and an inference engine with the optimizations real serving
systems use — continuous batching, paged KV-cache attention, weight
quantization, speculative decoding — with custom Triton kernels replacing
naive PyTorch at every bottleneck.

Built under a hard $0 compute constraint: development and all correctness
tests run on CPU (Triton interpreter mode); real kernel timings come from a
single free Kaggle T4 session. **Anything this environment cannot actually
measure is labeled simulated, both here and in the generated report.**

## Architecture

```
            ┌────────────────────── train/ ──────────────────────┐
            │  model.py     MoE transformer (RMSNorm, RoPE,      │
            │               SwiGLU, top-k routed experts)        │
            │  parallel.py  Megatron-style TP + GPipe PP         │
            │               (multi-process, gloo — SIMULATED     │
            │               ranks on one device)                 │
            │  train_loop.py                                     │
            └────────────┬───────────────────────────────────────┘
                         │ same model definition
            ┌────────────▼───────────────── serve/ ──────────────┐
            │  engine.py    continuous batching scheduler        │
            │     │         (admit/preempt per step)             │
            │  kv_cache.py  paged KV cache (fixed-size blocks,   │
            │     │         block tables, free-list)             │
            │  quantize.py  INT8 / INT4 weight-only quant        │
            │  speculative.py  draft + one-forward verification  │
            │  api.py       CLI chat + HTTP endpoint             │
            └────────────┬───────────────────────────────────────┘
                         │ hot paths
            ┌────────────▼──────────────── kernels/ ─────────────┐
            │  attention_fwd.py   flash-attn-style fused fwd     │
            │  attention_bwd.py   hand-written flash backward    │
            │  paged_attention.py decode over scattered blocks   │
            │  moe_routing.py     gather / combine (no atomics)  │
            │  moe_backward.py    atomic scatter-add, combine bwd│
            │  dequant_matmul.py  INT8/INT4 dequant inside GEMM  │
            │  fp8_matmul.py      e4m3 bit-decode GEMM, inline   │
            │                     scales (emulated arithmetic)   │
            │  autograd_ops.py    torch.autograd.Function glue   │
            │  fusion_compiler.py graph -> generated Triton      │
            └────────────────────────────────────────────────────┘
   also: train/fp8.py (dynamic scaling), train/zero.py (ZeRO sharding),
         serve/disagg.py (split prefill/decode pools)
```

## Run it

```bash
pip install -r requirements.txt

python run_all.py                 # full test suite (CPU-safe, ~1 min)
python run_all.py --bench         # + benchmarks + regenerate bench/report.md

python -m serve.api train         # train the toy char-level model
python -m serve.api chat -p "kernel triton "
python -m serve.api chat -p "kernel " --quant int8 --speculative
python -m serve.api serve         # HTTP: POST /generate {"prompt": "..."}
```

No GPU needed for tests: `conftest.py` sets `TRITON_INTERPRET=1` when CUDA
is absent, so every Triton kernel runs (slowly, correctly) on CPU. On a
CUDA machine the same code compiles to real kernels.

## What's real vs simulated

| Component | Status |
|---|---|
| Kernel correctness (all phases) | **Real** — tested vs PyTorch references, CPU + GPU |
| Kernel benchmarks | **Real on GPU** (Kaggle T4); interpreter timings are labeled meaningless |
| Serving engine + scheduler | **Real** — reproduces naive generation token-for-token |
| Quantization accuracy/memory | **Real** |
| Speculative decoding | **Real** — greedy variant asserted identical to target output |
| Backward kernels (Phase 8) | **Real** — fp64 gradcheck where kernels preserve precision; analytic-vs-reference + finite-difference spot checks + loss-decrease elsewhere (see tests/test_backward.py docstring for the tolerance reasoning) |
| FP8 training (Phase 9) | **Rounding real, arithmetic emulated** — true e4m3/e5m2 casts + dynamic scaling, fp32 matmuls (no FP8 hardware on T4/CPU). Finding: scaled fp8 tracks fp32; unscaled fp8 stalls via gradient-cast underflow. |
| ZeRO sharding (Phase 10) | **Memory savings real** (states live in separate processes); step timing simulated |
| Disaggregated serving (Phase 11) | **Correctness real, split simulated** — one device time-shared; transfer overhead dominates by construction and is reported as such |
| Fusion compiler (Phase 12) | **Real** — generated kernels validated against torch and the hand-written attention chain |
| Tensor/pipeline parallelism | **Logic real, speedup SIMULATED** — ranks are processes sharing one device over gloo; correctness (outputs/grads match dense) is the tested claim. On real hardware you'd get NCCL, true compute/comm overlap, and actual scaling. |
| "Multi-GPU" benchmarks (phases 3/10/11) | **Simulated, labeled as such** — they measure orchestration overhead, not parallel speedup |

## GPU benchmark run (Kaggle)

One consolidated T4 session runs a CUDA-tensor correctness smoke suite
(`kaggle/gpu_smoke.py` — hard gate: no benchmarks on unverified kernels),
then every benchmark at GPU sizes:

```bash
python kaggle/push_and_fetch.py push     # start the session (uses kaggle CLI)
python kaggle/push_and_fetch.py status
python kaggle/push_and_fetch.py fetch    # pull results into bench/results/
python bench/make_report.py              # refresh report with GPU numbers
```

## Known limitations

* The model is a toy char-level LM (a few M params) — the systems work is
  the point, not language quality.
* Tokenizer = raw chars; vocab comes from the synthetic corpus.
* Speculative decoding uses its own contiguous KV cache; it is not composed
  into the paged engine (real systems do both).
* MoE layers are not tensor-parallel (real systems use expert parallelism).
* No CUDA graphs, no NCCL, no multi-node anything.

## Benchmarks

See [bench/report.md](bench/report.md) — regenerated by
`python bench/make_report.py` from `bench/results/*.json`. Each kernel file
documents what PyTorch does by default, where the bottleneck is, and what
the kernel changes.
