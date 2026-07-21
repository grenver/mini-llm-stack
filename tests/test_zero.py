"""Phase 10: ZeRO sharded optimizer — equivalence with plain AdamW + memory."""

import pytest
import torch

from train.parallel import launch_workers
from train.zero import partition_params
from tests._zero_workers import zero_equivalence_worker, zero_memory_worker


def test_partition_balances_and_covers():
    params = [torch.nn.Parameter(torch.zeros(n))
              for n in [1000, 900, 500, 400, 300, 200, 100, 50]]
    owner = partition_params(params, 3)
    assert len(owner) == len(params)
    loads = [sum(p.numel() for p, o in zip(params, owner) if o == r)
             for r in range(3)]
    assert sum(loads) == sum(p.numel() for p in params)
    assert max(loads) < sum(loads) * 0.5      # nobody hoards


@pytest.mark.slow
def test_zero_matches_plain_adamw_2ranks():
    """3 optimizer steps with ZeRO across 2 ranks must produce the same
    parameters as single-process AdamW on the combined batch."""
    launch_workers(zero_equivalence_worker, 2)


@pytest.mark.slow
def test_zero_memory_actually_sharded():
    """Each rank's Adam-moment bytes must be roughly half the unsharded
    total, and the shards must sum to it exactly."""
    launch_workers(zero_memory_worker, 2)
