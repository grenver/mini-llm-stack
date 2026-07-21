# Benchmark report

_Generated from bench/results/*.json — rerun benches then `python bench/make_report.py` to refresh._

## Phase 1 — Fused attention

**Bottleneck:** naive attention materializes the [S, S] score matrix in HBM (O(S²) traffic + capacity). **Kernel:** flash-style single pass, online softmax in registers, O(S·d) traffic. Peak-memory column is the direct evidence: naive grows quadratically, fused stays flat.

_Measured on `Tesla T4`._

| seq | impl | latency_ms | peak_mem_mb |
|---|---|---|---|
| 256 | naive_pytorch | 0.736 | 31.1 |
| 256 | sdpa | 0.107 | 12.1 |
| 256 | triton_flash | 1.381 | 12.2 |
| 512 | naive_pytorch | 2.4 | 86.1 |
| 512 | sdpa | 0.238 | 16.1 |
| 512 | triton_flash | 3.991 | 16.2 |
| 1024 | naive_pytorch | 8.366 | 292.1 |
| 1024 | sdpa | 0.401 | 24.1 |
| 1024 | triton_flash | 12.621 | 24.2 |
| 2048 | naive_pytorch | 27.91 | 1088.2 |
| 2048 | sdpa | 1.361 | 40.1 |
| 2048 | triton_flash | 47.065 | 40.4 |
| 4096 | naive_pytorch | 110.183 | 4216.2 |
| 4096 | sdpa | 5.428 | 72.1 |
| 4096 | triton_flash | 180.977 | 72.6 |

## Phase 2 — MoE routing

**Bottleneck:** the per-expert loop launches 2·E indexing kernels (index_select + atomic index_add) per layer. **Kernel:** sort once, one gather kernel into expert-contiguous layout, dense per-expert GEMMs, one atomic-free combine kernel. `routing_only` rows isolate dispatch/combine overhead; `mlp` rows show the full layer.

_Measured on `Tesla T4`._

| tokens | experts | top_k | d | experts_kind | impl | latency_ms |
|---|---|---|---|---|---|---|
| 4096 | 8 | 2 | 1024 | mlp | naive | 9.876 |
| 4096 | 8 | 2 | 1024 | mlp | triton | 6.165 |
| 4096 | 8 | 2 | 1024 | routing_only | naive | 1.875 |
| 4096 | 8 | 2 | 1024 | routing_only | triton | 0.974 |
| 16384 | 8 | 2 | 1024 | mlp | naive | 23.879 |
| 16384 | 8 | 2 | 1024 | mlp | triton | 21.805 |
| 16384 | 8 | 2 | 1024 | routing_only | naive | 4.295 |
| 16384 | 8 | 2 | 1024 | routing_only | triton | 2.035 |
| 16384 | 32 | 2 | 1024 | mlp | naive | 26.67 |
| 16384 | 32 | 2 | 1024 | mlp | triton | 21.073 |
| 16384 | 32 | 2 | 1024 | routing_only | naive | 6.303 |
| 16384 | 32 | 2 | 1024 | routing_only | triton | 2.243 |
| 65536 | 8 | 2 | 1024 | mlp | naive | 88.665 |
| 65536 | 8 | 2 | 1024 | mlp | triton | 81.82 |
| 65536 | 8 | 2 | 1024 | routing_only | naive | 13.811 |
| 65536 | 8 | 2 | 1024 | routing_only | triton | 6.719 |
| 65536 | 32 | 2 | 1024 | mlp | naive | 99.091 |
| 65536 | 32 | 2 | 1024 | mlp | triton | 93.017 |
| 65536 | 32 | 2 | 1024 | routing_only | naive | 15.994 |
| 65536 | 32 | 2 | 1024 | routing_only | triton | 6.84 |
| 65536 | 64 | 2 | 1024 | mlp | naive | 100.856 |
| 65536 | 64 | 2 | 1024 | mlp | triton | 94.92 |
| 65536 | 64 | 2 | 1024 | routing_only | naive | 18.82 |
| 65536 | 64 | 2 | 1024 | routing_only | triton | 7.003 |

## Phase 3 — Tensor/pipeline parallelism (SIMULATED)

**All ranks share one physical device** — these numbers measure orchestration+IPC overhead only and can NOT show speedup; parallel configs are expected to be slower than dense here. The correctness tests (sharded outputs/grads == dense reference) are the real deliverable of this phase.

_Measured on `Tesla T4`._

> ⚠️ **Simulated parallelism:** all ranks share one physical device; no real parallel speedup is possible in this environment

| config | step_ms |
|---|---|
| dense_1proc | 109.1 |
| tp2_simulated | 238.6 |
| pp2_mb2_simulated | 133.0 |

## Phase 4 — Quantization + fused dequant-matmul

**Bottleneck:** decode-shaped matmuls are bandwidth-bound on weight bytes; dequantizing to fp before the matmul reads W at full width anyway. **Kernel:** loads INT8/INT4 weights, dequantizes in registers inside the K-loop — 2×/4× less weight traffic.

_Measured on `Tesla T4`._

| kind | variant | weight_mb |
|---|---|---|
| memory | fp32 | 97.03 |
| memory | int8 | 24.82 |
| memory | int4 | 15.61 |

| kind | M | N | K | impl | latency_ms |
|---|---|---|---|---|---|
| latency | 1 | 2048 | 2048 | fp_matmul | 0.0679 |
| latency | 1 | 2048 | 2048 | dequant_then_matmul_int8 | 0.4531 |
| latency | 1 | 2048 | 2048 | fused_int8 | 1.2003 |
| latency | 1 | 2048 | 2048 | fused_int4 | 0.879 |
| latency | 16 | 2048 | 2048 | fp_matmul | 0.0948 |
| latency | 16 | 2048 | 2048 | dequant_then_matmul_int8 | 0.4956 |
| latency | 16 | 2048 | 2048 | fused_int8 | 1.2098 |
| latency | 16 | 2048 | 2048 | fused_int4 | 0.869 |
| latency | 128 | 2048 | 2048 | fp_matmul | 0.1206 |
| latency | 128 | 2048 | 2048 | dequant_then_matmul_int8 | 0.5225 |
| latency | 128 | 2048 | 2048 | fused_int8 | 4.0002 |
| latency | 128 | 2048 | 2048 | fused_int4 | 1.9409 |

### Phase 4 (cont.) — Quantized model accuracy

Held-out loss/perplexity of the same trained model under fp32, INT8 and INT4 weights — the memory/latency numbers above only matter if quality survives. Device-independent numerics.

_Measured on `cpu (Triton interpreter)` (device-independent numerics — no timings involved)._

| variant | val_loss | perplexity | loss_increase_pct |
|---|---|---|---|
| fp32 | 0.5054 | 1.658 | 0.0 |
| int8 | 0.5055 | 1.658 | 0.02 |
| int4_g32 | 0.5081 | 1.662 | 0.54 |

## Phase 5 — Continuous batching + paged KV cache

**Bottleneck:** sequential decode re-reads all weights per token per request; static batches waste slots on finished sequences and padding. **Engine:** paged cache (block-granular admission, zero padding waste) + per-step rescheduling; paged-attention kernel reads K/V directly from scattered blocks.

_Measured on `Tesla T4`._

| impl | requests | max_new | wall_s | tokens_per_s |
|---|---|---|---|---|
| sequential_naive | 32 | 64 | 23.759 | 86.2 |
| engine_batch1 | 32 | 64 | 21.525 | 95.15 |
| engine_batch8 | 32 | 64 | 3.663 | 559.13 |
| engine_batch32 | 32 | 64 | 1.863 | 1099.37 |

## Phase 6 — Speculative decoding

**Bottleneck:** one bandwidth-bound target forward per token. **Mechanism:** draft proposes γ tokens, target verifies all of them in one forward; greedy variant is token-identical to the target (asserted inside this very benchmark). Low acceptance ⇒ slower than autoregressive — reported as measured.

_Measured on `Tesla T4`._

| impl | gamma | wall_s | tokens_per_s | acceptance_rate | tokens_per_round |
|---|---|---|---|---|---|
| autoregressive | 0 | 1.136 | 112.71 | None | 1.0 |
| speculative | 2 | 0.739 | 173.19 | 0.694 | 2.39 |
| speculative | 4 | 0.666 | 192.09 | 0.447 | 2.79 |
| speculative | 8 | 0.74 | 173.09 | 0.321 | 3.57 |

## Phase 8 — Custom backward kernels (training)

Forward/backward through the custom autograd Functions vs PyTorch autograd on the reference implementation. Correctness = gradcheck + loss-decreases tests, not this table.

_Measured on `Tesla T4`._

| seq | impl | latency_ms | peak_mem_mb |
|---|---|---|---|
| 256 | triton_fwd_bwd | 4.514 | 16.1 |
| 256 | reference_fwd_bwd | 2.059 | 60.3 |
| 512 | triton_fwd_bwd | 14.462 | 48.4 |
| 512 | reference_fwd_bwd | 5.143 | 168.5 |
| 1024 | triton_fwd_bwd | 27.385 | 80.5 |
| 1024 | reference_fwd_bwd | 18.64 | 577.2 |
| 2048 | triton_fwd_bwd | 105.322 | 144.8 |
| 2048 | reference_fwd_bwd | 71.144 | 2164.2 |

## Phase 9 — FP8 (e4m3) training emulation

Storage/rounding in true float8_e4m3fn with per-tensor dynamic scaling; matmul arithmetic in fp32 (no FP8 tensor cores on T4 — see README). Convergence curves are the result here.

_Measured on `Tesla T4`._

| arm | loss@0 | loss@100 | loss@200 | loss@399 | final_loss | diverged |
|---|---|---|---|---|---|---|
| fp32 | 4.1617 | 0.67 | 0.4201 | 0.3549 | 0.3628 | False |
| fp8_dynamic_scaling | 4.1645 | 0.6569 | 0.4208 | 0.3575 | 0.3634 | False |
| fp8_no_scaling | 4.1617 | 2.9359 | 2.8699 | 2.8656 | 2.8743 | False |

## Phase 10 — ZeRO-style optimizer sharding (SIMULATED ranks)

Per-rank optimizer-state memory is REAL (states genuinely live in separate processes); step-time comparisons are not meaningful on shared hardware.

_Measured on `Tesla T4`._

| config | per_rank_state_mb | step_ms | memory_real | timing_simulated |
|---|---|---|---|---|
| adamw_1proc | 37.28 | 140.1 | True | False |
| zero_w2_overlap | 18.88 | 263.1 | True | True |
| zero_w2_sync | 18.88 | 305.9 | True | True |
| zero_w4_overlap | 9.5 | 844.0 | True | True |
| zero_w4_sync | 9.5 | 809.1 | True | True |

## Phase 11 — Disaggregated prefill/decode (SIMULATED)

Two processes time-share one device; KV-cache transfer cost is real, pool separation benefits are not observable. Correctness + overhead breakdown only.

_Measured on `Tesla T4`._

> ⚠️ **Simulated parallelism:** both pools share one device: the split can only add overhead in this environment; breakdown shows where it goes

| config | wall_s | kv_transferred_mb | mean_transfer_ms |
|---|---|---|---|
| unified_engine | 2.18 | 0.0 | 0.0 |
| disaggregated_simulated | 4.999 | 21.0 | 95.31 |
