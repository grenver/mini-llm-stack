"""Custom Triton kernels.

Importing this package (which happens before any submodule import) selects
Triton interpreter mode when no CUDA device is present, so the same kernel
source runs CPU-interpreted for correctness work and compiled on GPU.
"""

import os

import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

INTERPRETER = os.environ.get("TRITON_INTERPRET") == "1"
