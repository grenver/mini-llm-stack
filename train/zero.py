"""ZeRO-style optimizer state sharding (stage 1 + gradient sharding à la 2).

What ZeRO fixes: with data parallelism, every rank redundantly holds the
full Adam state (2 fp32 moments = 8 bytes/param — for mixed-precision
training the optimizer state dwarfs the fp16 weights). ZeRO stage 1
partitions parameters across ranks; each rank keeps moments ONLY for its
shard, steps its shard, and broadcasts updated params. Stage 2 additionally
avoids keeping full replicated gradients: each gradient is REDUCED to its
owner rank (not all-reduced everywhere) and dropped elsewhere.

Overlap: gradients become ready back-to-front during backward, so each
parameter registers a post-accumulate-grad hook that immediately launches
an async reduce toward the owner. step() waits on the handles, steps local
shards, then broadcasts. This is the real orchestration pattern; on this
hardware (all ranks share one machine/device) the "overlap" hides nothing —
the memory sharding is the measurable, real result here, the timing is not.

Partitioning is whole-tensor greedy bin-packing (largest first). Real ZeRO
flattens and splits at arbitrary element boundaries for perfect balance;
whole-tensor keeps the optimizer untouched and the imbalance is small for
transformer-shaped parameter lists.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def partition_params(params: list[torch.nn.Parameter],
                     world: int) -> list[int]:
    """Greedy balanced assignment; returns owner rank per param."""
    order = sorted(range(len(params)), key=lambda i: -params[i].numel())
    load = [0] * world
    owner = [0] * len(params)
    for i in order:
        r = min(range(world), key=lambda x: load[x])
        owner[i] = r
        load[r] += params[i].numel()
    return owner


class ZeroAdamW:
    def __init__(self, params, rank: int, world: int, lr: float = 1e-3,
                 betas=(0.9, 0.95), weight_decay: float = 0.0,
                 overlap: bool = True, shard_grads: bool = True):
        self.params = [p for p in params if p.requires_grad]
        self.rank, self.world = rank, world
        self.owner = partition_params(self.params, world)
        self.local = [p for p, o in zip(self.params, self.owner) if o == rank]
        self.inner = torch.optim.AdamW(self.local, lr=lr, betas=betas,
                                       weight_decay=weight_decay) \
            if self.local else None
        self.shard_grads = shard_grads
        self.overlap = overlap
        self._handles: list = []
        if overlap:
            for p, o in zip(self.params, self.owner):
                p.register_post_accumulate_grad_hook(self._make_hook(o))

    def _make_hook(self, owner_rank: int):
        def hook(p):
            # launch the reduce the moment this grad is finalized
            if self.shard_grads:
                h = dist.reduce(p.grad, dst=owner_rank, async_op=True)
            else:
                h = dist.all_reduce(p.grad, async_op=True)
            self._handles.append(h)
        return hook

    def _sync_grads_now(self):
        for p, o in zip(self.params, self.owner):
            if p.grad is None:
                continue
            if self.shard_grads:
                dist.reduce(p.grad, dst=o)
            else:
                dist.all_reduce(p.grad)

    @property
    def param_groups(self):
        return self.inner.param_groups if self.inner else []

    def set_lr(self, lr: float):
        for g in self.param_groups:
            g["lr"] = lr

    def step(self):
        if self.overlap:
            for h in self._handles:
                h.wait()
            self._handles.clear()
        else:
            self._sync_grads_now()

        # reduce delivered SUMs; owners take the data-parallel mean
        for p in self.local:
            if p.grad is not None:
                p.grad.div_(self.world)
        if self.inner:
            self.inner.step()

        # stage-2 flavor: non-owned grads are dead weight — drop immediately
        for p, o in zip(self.params, self.owner):
            if o != self.rank:
                p.grad = None

        # everyone receives the updated shards
        for p, o in zip(self.params, self.owner):
            dist.broadcast(p.data, src=o)

    def zero_grad(self):
        for p in self.params:
            p.grad = None
        self._handles.clear()

    # ------------------------------------------------------------- metrics

    def optimizer_state_bytes(self) -> int:
        """Bytes of Adam moments held by THIS rank (the ZeRO saving)."""
        if not self.inner:
            return 0
        total = 0
        for state in self.inner.state.values():
            for v in state.values():
                # count the moment buffers; skip AdamW's 0-dim step counters
                if torch.is_tensor(v) and v.dim() > 0:
                    total += v.numel() * v.element_size()
        return total


def full_adamw_state_bytes(model: torch.nn.Module) -> int:
    """What an unsharded AdamW would hold: 2 fp32 moments per param."""
    return sum(p.numel() * 8 for p in model.parameters() if p.requires_grad)
