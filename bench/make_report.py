"""Generate bench/report.md from bench/results/*.json.

Renders whatever result files exist; each section is tagged with the
environment it was measured on. Results with meaningful_timings=False
(CPU / Triton interpreter) are labeled "correctness-only run" — the
interpreter executes kernels as numpy loops, so its timings say nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
OUT = Path(__file__).parent / "report.md"

EXPLAIN = {
    "attention": (
        "## Phase 1 — Fused attention\n\n"
        "**Bottleneck:** naive attention materializes the [S, S] score matrix "
        "in HBM (O(S²) traffic + capacity). **Kernel:** flash-style single "
        "pass, online softmax in registers, O(S·d) traffic. Peak-memory column "
        "is the direct evidence: naive grows quadratically, fused stays flat.\n"),
    "moe_routing": (
        "## Phase 2 — MoE routing\n\n"
        "**Bottleneck:** the per-expert loop launches 2·E indexing kernels "
        "(index_select + atomic index_add) per layer. **Kernel:** sort once, "
        "one gather kernel into expert-contiguous layout, dense per-expert "
        "GEMMs, one atomic-free combine kernel. `routing_only` rows isolate "
        "dispatch/combine overhead; `mlp` rows show the full layer.\n"),
    "parallel": (
        "## Phase 3 — Tensor/pipeline parallelism (SIMULATED)\n\n"
        "**All ranks share one physical device** — these numbers measure "
        "orchestration+IPC overhead only and can NOT show speedup; parallel "
        "configs are expected to be slower than dense here. The correctness "
        "tests (sharded outputs/grads == dense reference) are the real "
        "deliverable of this phase.\n"),
    "quantize": (
        "## Phase 4 — Quantization + fused dequant-matmul\n\n"
        "**Bottleneck:** decode-shaped matmuls are bandwidth-bound on weight "
        "bytes; dequantizing to fp before the matmul reads W at full width "
        "anyway. **Kernel:** loads INT8/INT4 weights, dequantizes in "
        "registers inside the K-loop — 2×/4× less weight traffic.\n"),
    "quantize_accuracy": (
        "### Phase 4 (cont.) — Quantized model accuracy\n\n"
        "Held-out loss/perplexity of the same trained model under fp32, "
        "INT8 and INT4 weights — the memory/latency numbers above only "
        "matter if quality survives. Device-independent numerics.\n"),
    "serving_throughput": (
        "## Phase 5 — Continuous batching + paged KV cache\n\n"
        "**Bottleneck:** sequential decode re-reads all weights per token per "
        "request; static batches waste slots on finished sequences and "
        "padding. **Engine:** paged cache (block-granular admission, zero "
        "padding waste) + per-step rescheduling; paged-attention kernel reads "
        "K/V directly from scattered blocks.\n"),
    "speculative": (
        "## Phase 6 — Speculative decoding\n\n"
        "**Bottleneck:** one bandwidth-bound target forward per token. "
        "**Mechanism:** draft proposes γ tokens, target verifies all of them "
        "in one forward; greedy variant is token-identical to the target "
        "(asserted inside this very benchmark). Low acceptance ⇒ slower than "
        "autoregressive — reported as measured.\n"),
    "backward": (
        "## Phase 8 — Custom backward kernels (training)\n\n"
        "Forward/backward through the custom autograd Functions vs PyTorch "
        "autograd on the reference implementation. Correctness = gradcheck + "
        "loss-decreases tests, not this table.\n"),
    "fp8": (
        "## Phase 9 — FP8 (e4m3) training emulation\n\n"
        "Storage/rounding in true float8_e4m3fn with per-tensor dynamic "
        "scaling; matmul arithmetic in fp32 (no FP8 tensor cores on T4 — "
        "see README). Convergence curves are the result here.\n"),
    "zero": (
        "## Phase 10 — ZeRO-style optimizer sharding (SIMULATED ranks)\n\n"
        "Per-rank optimizer-state memory is REAL (states genuinely live in "
        "separate processes); step-time comparisons are not meaningful on "
        "shared hardware.\n"),
    "disagg": (
        "## Phase 11 — Disaggregated prefill/decode (SIMULATED)\n\n"
        "Two processes time-share one device; KV-cache transfer cost is real, "
        "pool separation benefits are not observable. Correctness + overhead "
        "breakdown only.\n"),
}


def group_rows(rows: list[dict]) -> list[list[dict]]:
    """Split rows into groups of identical field-sets (a bench may emit
    differently-shaped rows, e.g. memory rows and latency rows)."""
    groups: dict[frozenset, list[dict]] = {}
    order: list[frozenset] = []
    for r in rows:
        key = frozenset(r.keys())
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)
    return [groups[k] for k in order]


def fmt_table(rows: list[dict]) -> str:
    """Render rows as one or more markdown tables — one per row shape.

    Invariant (enforced, not assumed): every key of every row must appear as
    a column of its table. A violation raises instead of silently rendering
    blank cells; that exact failure shipped once (Phase 4 latency rows were
    dropped because columns came from the first — memory — row only).
    """
    if not rows:
        return "_no rows_\n"
    parts = []
    for group in group_rows(rows):
        cols = list(group[0].keys())
        for r in group:
            missing = set(r.keys()) - set(cols)
            if missing:
                raise RuntimeError(
                    f"report table would drop fields {missing} of row {r}")
        out = ["| " + " | ".join(cols) + " |",
               "|" + "|".join("---" for _ in cols) + "|"]
        for r in group:
            out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        parts.append("\n".join(out) + "\n")
    return "\n".join(parts)


def main():
    sections = ["# Benchmark report\n",
                "_Generated from bench/results/*.json — rerun benches then "
                "`python bench/make_report.py` to refresh._\n"]
    order = ["attention", "moe_routing", "parallel", "quantize",
             "quantize_accuracy", "serving_throughput", "speculative",
             "backward", "fp8", "zero", "disagg"]
    seen = set()
    files = {p.stem: p for p in RESULTS.glob("*.json")}
    for name in order + sorted(set(files) - set(order)):
        if name not in files or name in seen:
            continue
        seen.add(name)
        blob = json.loads(files[name].read_text())
        env = blob.get("env", {})
        sections.append(EXPLAIN.get(name, f"## {name}\n"))
        tag = env.get("device", "?")
        if blob.get("timings_irrelevant"):
            sections.append(f"_Measured on `{tag}` (device-independent "
                            "numerics — no timings involved)._\n")
        elif not env.get("meaningful_timings", True):
            sections.append(f"> ⚠️ **Correctness-only run** on `{tag}` — "
                            "interpreter timings are meaningless; rerun on "
                            "GPU for real numbers.\n")
        else:
            sections.append(f"_Measured on `{tag}`._\n")
        if blob.get("simulated"):
            sections.append(f"> ⚠️ **Simulated parallelism:** "
                            f"{blob.get('note', '')}\n")
        sections.append(fmt_table(blob.get("rows", [])))
    OUT.write_text("\n".join(sections), encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
