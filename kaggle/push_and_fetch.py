"""Local driver for the Kaggle GPU benchmark run.

    python kaggle/push_and_fetch.py push    # push + start the GPU kernel
    python kaggle/push_and_fetch.py status  # poll
    python kaggle/push_and_fetch.py fetch   # download results into bench/results

Uses the `kaggle` CLI (needs ~/.kaggle credentials). The kernel runs
kaggle_bench.py on a T4 and writes bench/results/*.json to its output.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE.parent / "bench" / "results"
SLUG = "mini-llm-stack-bench"


def username() -> str:
    out = subprocess.run(["kaggle", "config", "view"], capture_output=True,
                         text=True).stdout
    for line in out.splitlines():
        if "username" in line.lower():
            return line.split(":")[-1].strip().strip("'\"")
    raise SystemExit(f"could not find kaggle username in config:\n{out}")


def push():
    user = username()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shutil.copy(HERE / "kaggle_bench.py", td / "kaggle_bench.py")
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{SLUG}",
            "title": SLUG,
            "code_file": "kaggle_bench.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }, indent=1))
        subprocess.run(["kaggle", "kernels", "push", "-p", str(td)], check=True)
    print(f"pushed {user}/{SLUG}; poll with: python kaggle/push_and_fetch.py status")


def status():
    subprocess.run(["kaggle", "kernels", "status", f"{username()}/{SLUG}"])


def fetch():
    user = username()
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["kaggle", "kernels", "output", f"{user}/{SLUG}",
                        "-p", td], check=True)
        src = Path(td) / "results"
        if not src.exists():
            src = Path(td)
        RESULTS.mkdir(exist_ok=True)
        n = 0
        for f in src.glob("*.json"):
            shutil.copy(f, RESULTS / f.name)
            n += 1
        for f in src.glob("*.txt"):
            print(f.name, "->", f.read_text().strip())
        print(f"copied {n} result files into {RESULTS}")
    print("regenerate the report with: python bench/make_report.py")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "push"
    {"push": push, "status": status, "fetch": fetch}[cmd]()
