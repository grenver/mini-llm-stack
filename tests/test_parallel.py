"""Phase 3: tensor/pipeline parallelism correctness via real multi-process runs."""

import pytest
import torch

from train.parallel import launch_workers
from tests._parallel_workers import pp_worker, tp_worker


@pytest.mark.slow
def test_tensor_parallel_2rank_matches_dense():
    launch_workers(tp_worker, 2)


@pytest.mark.slow
def test_pipeline_parallel_2stage_matches_dense():
    launch_workers(pp_worker, 2)
