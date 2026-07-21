"""Kaggle GPU benchmark session (single script, one T4 session).

Flow: clone the public repo → GPU smoke correctness suite (hard gate) →
run every benchmark at full GPU sizes → copy bench/results/*.json to
/kaggle/working/results for retrieval via `kaggle kernels output`.

Designed to finish in well under an hour of the 30 GPU-hr/week budget.
"""

import os
import shutil
import subprocess
import sys
import time

REPO = "https://github.com/grenver/mini-llm-stack.git"
WORK = "/kaggle/working"

t_start = time.time()
os.chdir(WORK)
if os.path.exists("mini-llm-stack"):
    shutil.rmtree("mini-llm-stack")
subprocess.run(["git", "clone", "--depth", "1", REPO], check=True)
os.chdir("mini-llm-stack")

import torch  # noqa: E402

print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# ---- gate: GPU correctness before any benchmark ----
r = subprocess.run([sys.executable, "kaggle/gpu_smoke.py"])
smoke_ok = r.returncode == 0
os.makedirs(f"{WORK}/results", exist_ok=True)
with open(f"{WORK}/results/SMOKE_STATUS.txt", "w") as f:
    f.write("PASS" if smoke_ok else "FAIL")
if not smoke_ok:
    print("SMOKE FAILED — skipping benchmarks (no numbers for broken kernels)")
    sys.exit(1)

BENCHES = [
    "bench/bench_attention.py",
    "bench/bench_moe_routing.py",
    "bench/bench_quantize.py",
    "bench/bench_serving_throughput.py",
    "bench/bench_speculative.py",
    "bench/bench_backward.py",
    "bench/bench_fp8.py",
    "bench/bench_parallel.py",
    "bench/bench_zero.py",
    "bench/bench_disagg.py",
]
failures = []
for b in BENCHES:
    print(f"\n===== {b} (t+{time.time()-t_start:.0f}s) =====", flush=True)
    if subprocess.run([sys.executable, b]).returncode != 0:
        failures.append(b)

shutil.copytree("bench/results", f"{WORK}/results", dirs_exist_ok=True)
with open(f"{WORK}/results/BENCH_FAILURES.txt", "w") as f:
    f.write("\n".join(failures) if failures else "none")
print(f"\nDONE in {time.time()-t_start:.0f}s; failures: {failures or 'none'}")
