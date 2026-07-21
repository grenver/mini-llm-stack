"""Root conftest: force Triton interpreter mode on machines without CUDA.

TRITON_INTERPRET must be set before `triton` is imported anywhere in the
process. pytest imports this file before collecting test modules, so kernels
developed here run under the (slow, CPU, numpy-backed) interpreter locally and
compile to real PTX on a CUDA machine with no code changes.
"""

import os

import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
