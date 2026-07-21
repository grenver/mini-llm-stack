# Benchmark report

_Generated from bench/results/*.json — rerun benches then `python bench/make_report.py` to refresh._

## Phase 1 — Fused attention

**Bottleneck:** naive attention materializes the [S, S] score matrix in HBM (O(S²) traffic + capacity). **Kernel:** flash-style single pass, online softmax in registers, O(S·d) traffic. Peak-memory column is the direct evidence: naive grows quadratically, fused stays flat.

> ⚠️ **Correctness-only run** on `cpu (Triton interpreter)` — interpreter timings are meaningless; rerun on GPU for real numbers.

| seq | impl | latency_ms | peak_mem_mb |
|---|---|---|---|
| 64 | naive_pytorch | 0.07 | nan |
| 64 | sdpa | 0.022 | nan |
| 64 | triton_flash | 67.213 | nan |

## Phase 2 — MoE routing

**Bottleneck:** the per-expert loop launches 2·E indexing kernels (index_select + atomic index_add) per layer. **Kernel:** sort once, one gather kernel into expert-contiguous layout, dense per-expert GEMMs, one atomic-free combine kernel. `routing_only` rows isolate dispatch/combine overhead; `mlp` rows show the full layer.

> ⚠️ **Correctness-only run** on `cpu (Triton interpreter)` — interpreter timings are meaningless; rerun on GPU for real numbers.

| tokens | experts | top_k | d | experts_kind | impl | latency_ms |
|---|---|---|---|---|---|---|
| 256 | 4 | 2 | 64 | mlp | naive | 0.584 |
| 256 | 4 | 2 | 64 | mlp | triton | 639.274 |
| 256 | 4 | 2 | 64 | routing_only | naive | 0.196 |
| 256 | 4 | 2 | 64 | routing_only | triton | 556.165 |

## Phase 3 — Tensor/pipeline parallelism (SIMULATED)

**All ranks share one physical device** — these numbers measure orchestration+IPC overhead only and can NOT show speedup; parallel configs are expected to be slower than dense here. The correctness tests (sharded outputs/grads == dense reference) are the real deliverable of this phase.

> ⚠️ **Correctness-only run** on `cpu (Triton interpreter)` — interpreter timings are meaningless; rerun on GPU for real numbers.

> ⚠️ **Simulated parallelism:** all ranks share one physical device; no real parallel speedup is possible in this environment

| config | step_ms |
|---|---|
| dense_1proc | 40.3 |
| tp2_simulated | 96.4 |
| pp2_mb2_simulated | 58.0 |

## Phase 4 — Quantization + fused dequant-matmul

**Bottleneck:** decode-shaped matmuls are bandwidth-bound on weight bytes; dequantizing to fp before the matmul reads W at full width anyway. **Kernel:** loads INT8/INT4 weights, dequantizes in registers inside the K-loop — 2×/4× less weight traffic.

> ⚠️ **Correctness-only run** on `cpu (Triton interpreter)` — interpreter timings are meaningless; rerun on GPU for real numbers.

| kind | variant | weight_mb |
|---|---|---|
| memory | fp32 | 97.03 |
| memory | int8 | 24.82 |
| memory | int4 | 15.61 |
| latency |  |  |
| latency |  |  |
| latency |  |  |
| latency |  |  |

## Phase 5 — Continuous batching + paged KV cache

**Bottleneck:** sequential decode re-reads all weights per token per request; static batches waste slots on finished sequences and padding. **Engine:** paged cache (block-granular admission, zero padding waste) + per-step rescheduling; paged-attention kernel reads K/V directly from scattered blocks.

> ⚠️ **Correctness-only run** on `cpu (Triton interpreter)` — interpreter timings are meaningless; rerun on GPU for real numbers.

| impl | requests | max_new | wall_s | tokens_per_s |
|---|---|---|---|---|
| sequential_naive | 6 | 16 | 0.086 | 1121.96 |
| engine_batch1 | 6 | 16 | 0.066 | 1445.14 |
| engine_batch2 | 6 | 16 | 0.056 | 1709.85 |
| engine_batch8 | 6 | 16 | 0.031 | 3073.7 |

## Phase 6 — Speculative decoding

**Bottleneck:** one bandwidth-bound target forward per token. **Mechanism:** draft proposes γ tokens, target verifies all of them in one forward; greedy variant is token-identical to the target (asserted inside this very benchmark). Low acceptance ⇒ slower than autoregressive — reported as measured.

> ⚠️ **Correctness-only run** on `cpu (Triton interpreter)` — interpreter timings are meaningless; rerun on GPU for real numbers.

| impl | gamma | wall_s | tokens_per_s | acceptance_rate | tokens_per_round |
|---|---|---|---|---|---|
| autoregressive | 0 | 0.063 | 764.4 | None | 1.0 |
| speculative | 2 | 0.089 | 541.44 | 0.345 | 1.69 |
| speculative | 4 | 0.078 | 612.74 | 0.26 | 2.04 |
| speculative | 8 | 0.125 | 383.79 | 0.13 | 2.04 |
