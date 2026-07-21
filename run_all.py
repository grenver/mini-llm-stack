"""Single entrypoint: full test suite + all benchmarks + report.

    python run_all.py                # tests only (fast-ish; CPU-safe)
    python run_all.py --bench        # tests + benchmarks + report
    python run_all.py --bench --quick  # smaller benchmark grids
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
BENCHES = [
    "bench/bench_attention.py",
    "bench/bench_moe_routing.py",
    "bench/bench_parallel.py",
    "bench/bench_quantize.py",
    "bench/bench_serving_throughput.py",
    "bench/bench_speculative.py",
    "bench/bench_backward.py",
    "bench/bench_fp8.py",
    "bench/bench_zero.py",
    "bench/bench_disagg.py",
]


def run(cmd: list[str]) -> int:
    print(f"\n=== {' '.join(cmd)} ===", flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def main():
    bench = "--bench" in sys.argv
    quick = "--quick" in sys.argv

    rc = run([sys.executable, "-m", "pytest", "tests/", "-q"])
    if rc != 0:
        print("\nTESTS FAILED — not running benchmarks on unverified kernels.")
        sys.exit(rc)

    if bench:
        failures = []
        for b in BENCHES:
            if not (ROOT / b).exists():
                continue
            cmd = [sys.executable, b] + (["--quick"] if quick else [])
            if run(cmd) != 0:
                failures.append(b)
        run([sys.executable, "bench/make_report.py"])
        if failures:
            print(f"\nBENCH FAILURES: {failures}")
            sys.exit(1)
    print("\nALL OK")


if __name__ == "__main__":
    main()
