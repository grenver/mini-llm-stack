"""Spawned workers for ZeRO tests (module-level for Windows pickling)."""

import torch
import torch.distributed as dist

from train.config import tiny_config
from train.model import Transformer
from train.parallel import init_dist
from train.zero import ZeroAdamW, full_adamw_state_bytes

LR, BETAS, WD, STEPS = 1e-2, (0.9, 0.95), 0.01, 3


def _batches(cfg, world):
    torch.manual_seed(99)
    return [torch.randint(0, cfg.vocab_size, (2, 16)) for _ in range(STEPS * world)]


def zero_equivalence_worker(rank, world, store_path):
    init_dist(rank, world, store_path)
    try:
        torch.manual_seed(0)
        cfg = tiny_config(d_model=64, n_layers=2, n_heads=2)
        model = Transformer(cfg)
        batches = _batches(cfg, world)

        # single-process reference: full batch = concat of all rank batches
        torch.manual_seed(0)
        ref = Transformer(cfg)
        ref_opt = torch.optim.AdamW(ref.parameters(), lr=LR, betas=BETAS,
                                    weight_decay=WD)
        for s in range(STEPS):
            full = torch.cat([batches[s * world + r] for r in range(world)])
            _, loss, _ = ref(full, full)
            ref_opt.zero_grad()
            loss.backward()
            ref_opt.step()

        opt = ZeroAdamW(model.parameters(), rank, world, lr=LR, betas=BETAS,
                        weight_decay=WD, overlap=True, shard_grads=True)
        for s in range(STEPS):
            mine = batches[s * world + rank]
            _, loss, _ = model(mine, mine)
            opt.zero_grad()
            loss.backward()
            opt.step()

        for (n, p), (_, pr) in zip(model.named_parameters(),
                                   ref.named_parameters()):
            torch.testing.assert_close(p, pr, rtol=2e-4, atol=2e-4)
    finally:
        dist.destroy_process_group()


def zero_memory_worker(rank, world, store_path):
    init_dist(rank, world, store_path)
    try:
        torch.manual_seed(0)
        cfg = tiny_config(d_model=128, n_layers=4, n_heads=4)
        model = Transformer(cfg)
        opt = ZeroAdamW(model.parameters(), rank, world, lr=1e-3)
        tokens = torch.randint(0, cfg.vocab_size, (2, 16))
        _, loss, _ = model(tokens, tokens)
        loss.backward()
        opt.step()                                   # states materialize here

        mine = torch.tensor([opt.optimizer_state_bytes()], dtype=torch.float64)
        total = mine.clone()
        dist.all_reduce(total)
        full = full_adamw_state_bytes(model)

        assert int(total.item()) == full, (total.item(), full)
        # balanced-ish: this rank holds 35-65% of the states
        frac = mine.item() / full
        assert 0.35 < frac < 0.65, f"rank {rank} holds {frac:.0%}"
    finally:
        dist.destroy_process_group()
