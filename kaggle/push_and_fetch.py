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
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
RESULTS = ROOT / "bench" / "results"
SLUG = "mini-llm-stack-bench"
DATA_SLUG = "mini-llm-stack-src"


def username() -> str:
    out = subprocess.run(["kaggle", "config", "view"], capture_output=True,
                         text=True).stdout
    for line in out.splitlines():
        if "username" in line.lower():
            return line.split(":")[-1].strip().strip("'\"")
    raise SystemExit(f"could not find kaggle username in config:\n{out}")


def push_dataset() -> str:
    """Upload the repo snapshot (tracked files at HEAD) as a private dataset
    so the kernel needs no internet. Returns the dataset ref."""
    user = username()
    ref = f"{user}/{DATA_SLUG}"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tar = td / "src.tar"
        subprocess.run(["git", "archive", "HEAD", "-o", str(tar)],
                       cwd=ROOT, check=True)
        src = td / "repo"
        src.mkdir()
        with tarfile.open(tar) as tf:
            tf.extractall(src)
        tar.unlink()
        (src / "dataset-metadata.json").write_text(json.dumps({
            "title": DATA_SLUG, "id": ref,
            "licenses": [{"name": "CC0-1.0"}]}, indent=1))
        r = subprocess.run(["kaggle", "datasets", "create", "-p", str(src),
                            "--dir-mode", "zip"],
                           capture_output=True, text=True)
        out = r.stdout + r.stderr
        if "already exists" in out or "409" in out or r.returncode != 0:
            subprocess.run(["kaggle", "datasets", "version", "-p", str(src),
                            "--dir-mode", "zip", "-m", "update"], check=True)
        print(f"dataset {ref} pushed")
    return ref


def push(offline: bool = False):
    user = username()
    sources = [push_dataset()] if offline else []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shutil.copy(HERE / "kaggle_bench.py", td / "kaggle_bench.py")
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{SLUG}",
            "title": SLUG,
            "code_file": "kaggle_bench.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": not offline,
            "dataset_sources": sources,
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
    if cmd == "push":
        push(offline="--offline" in sys.argv)
    else:
        {"status": status, "fetch": fetch}[cmd]()
