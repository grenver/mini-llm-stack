"""Tensor parallelism (Megatron-style) and pipeline parallelism (GPipe-style).

SIMULATION NOTE: this machine has one device. "Ranks" here are separate OS
processes communicating over torch.distributed's gloo backend, all sharing
the same CPU (or, on Kaggle, time-sharing one GPU). The sharding math, the
collectives, and the microbatch schedule are the real thing; the *speedups*
are not — there is no independent hardware for ranks to run on. Benchmarks
built on this module are labeled simulated. On real multi-GPU hardware you
would additionally see: NCCL instead of gloo, true compute/comm overlap
(async all-reduce during backward), and bandwidth-bound collective costs.

Tensor parallelism (Megatron-LM scheme)
---------------------------------------
Attention: QKV projections are column-parallel (each rank owns H/tp heads),
the output projection is row-parallel. MLP: gate/up are column-parallel,
down is row-parallel. Each block then needs exactly two all-reduces per
forward (after attn wo, after mlp w_down) — communication only where a
row-parallel layer produces partial sums.

Autograd handles the backward collectives via two conjugate ops:
  _CopyToTP:     forward identity, backward all-reduce (input feeds all ranks)
  _ReduceFromTP: forward all-reduce, backward identity
Embeddings and the LM head stay replicated for simplicity (Megatron shards
those too). MoE layers are out of scope for TP here — real systems use
expert parallelism for them instead.

Pipeline parallelism
--------------------
Layers are split into contiguous stages; stage 0 owns the embedding, the
last stage owns final norm + LM head + loss. The schedule is GPipe
fill-drain: all microbatch forwards (activations sent P2P), then all
backwards in reverse order (gradients sent P2P). Peak activation memory
grows with the number of in-flight microbatches — 1F1B would cap that, but
fill-drain keeps the schedule easy to verify for correctness.
"""

from __future__ import annotations

import copy
import os
import tempfile
import uuid

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from train.config import ModelConfig
from train.model import Attention, RMSNorm, SwiGLU, Transformer, naive_attention


# ---------------------------------------------------------------- dist setup

def init_dist(rank: int, world_size: int, store_path: str):
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group("gloo", store=store, rank=rank,
                            world_size=world_size)


def launch_workers(fn, world_size: int, *args):
    """Spawn world_size processes running fn(rank, world_size, store_path, *args)."""
    store_path = os.path.join(tempfile.gettempdir(),
                              f"mini_llm_store_{uuid.uuid4().hex}")
    mp.spawn(fn, args=(world_size, store_path) + args, nprocs=world_size,
             join=True)


# ------------------------------------------------------- TP collective ops

class _CopyToTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        dist.all_reduce(grad)
        return grad


class _ReduceFromTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


# ------------------------------------------------------------- TP layers

class ColumnParallelLinear(nn.Module):
    """y_r = x @ W_r^T where W is split along the output dim across ranks."""

    def __init__(self, weight_shard: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight_shard.clone())

    def forward(self, x):
        return F.linear(_CopyToTP.apply(x), self.weight)

    @staticmethod
    def shard(linear: nn.Linear, rank: int, world: int) -> "ColumnParallelLinear":
        out = linear.weight.shape[0]
        assert out % world == 0
        sl = out // world
        return ColumnParallelLinear(linear.weight[rank * sl:(rank + 1) * sl])


class RowParallelLinear(nn.Module):
    """y = all_reduce_r(x_r @ W_r^T) where W is split along the input dim.

    Input is assumed already split (it is the output of a column-parallel
    layer), so no scatter is needed on the way in.
    """

    def __init__(self, weight_shard: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight_shard.clone())

    def forward(self, x):
        return _ReduceFromTP.apply(F.linear(x, self.weight))

    @staticmethod
    def shard(linear: nn.Linear, rank: int, world: int) -> "RowParallelLinear":
        inp = linear.weight.shape[1]
        assert inp % world == 0
        sl = inp // world
        return RowParallelLinear(linear.weight[:, rank * sl:(rank + 1) * sl])


class TPAttention(nn.Module):
    """Attention with heads split across ranks."""

    def __init__(self, dense: Attention, cfg: ModelConfig, rank: int, world: int):
        super().__init__()
        assert cfg.n_heads % world == 0
        self.cfg = cfg
        self.local_heads = cfg.n_heads // world
        self.wq = ColumnParallelLinear.shard(dense.wq, rank, world)
        self.wk = ColumnParallelLinear.shard(dense.wk, rank, world)
        self.wv = ColumnParallelLinear.shard(dense.wv, rank, world)
        self.wo = RowParallelLinear.shard(dense.wo, rank, world)
        self.rotary = dense.rotary

    def forward(self, x, pos=None, kv_cache=None):
        assert kv_cache is None, "TP attention is train-time only here"
        B, S, _ = x.shape
        H, hd = self.local_heads, self.cfg.head_dim
        if pos is None:
            pos = torch.arange(S, device=x.device)
        q = self.wq(x).view(B, S, H, hd).transpose(1, 2)
        k = self.wk(x).view(B, S, H, hd).transpose(1, 2)
        v = self.wv(x).view(B, S, H, hd).transpose(1, 2)
        q, k = self.rotary(q, pos), self.rotary(k, pos)
        o = naive_attention(q, k, v, causal=True)
        return self.wo(o.transpose(1, 2).reshape(B, S, H * hd))


class TPSwiGLU(nn.Module):
    def __init__(self, dense: SwiGLU, rank: int, world: int):
        super().__init__()
        self.w_gate = ColumnParallelLinear.shard(dense.w_gate, rank, world)
        self.w_up = ColumnParallelLinear.shard(dense.w_up, rank, world)
        self.w_down = RowParallelLinear.shard(dense.w_down, rank, world)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def shard_model(model: Transformer, rank: int, world: int) -> Transformer:
    """Replace attention/MLP modules of a dense model with TP shards."""
    model = copy.deepcopy(model)
    cfg = model.cfg
    for blk in model.blocks:
        assert not blk.is_moe, "TP over MoE layers not supported (use EP)"
        blk.attn = TPAttention(blk.attn, cfg, rank, world)
        blk.mlp = TPSwiGLU(blk.mlp, rank, world)
    return model


# ------------------------------------------------------------- pipeline

class PipelineStage(nn.Module):
    """One contiguous slice of a Transformer's blocks.

    Stage 0 additionally owns the embedding; the last stage owns final norm
    and LM head.
    """

    def __init__(self, model: Transformer, stage: int, n_stages: int):
        super().__init__()
        cfg = model.cfg
        per = cfg.n_layers // n_stages
        lo = stage * per
        hi = cfg.n_layers if stage == n_stages - 1 else lo + per
        self.stage, self.n_stages = stage, n_stages
        self.is_first = stage == 0
        self.is_last = stage == n_stages - 1
        self.blocks = nn.ModuleList(copy.deepcopy(model.blocks[lo:hi]))
        self.embed = copy.deepcopy(model.embed) if self.is_first else None
        self.final_norm = copy.deepcopy(model.final_norm) if self.is_last else None
        self.lm_head = copy.deepcopy(model.lm_head) if self.is_last else None
        if self.is_last and model.cfg.tie_embeddings:
            # untie: last stage doesn't own the embedding
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.lm_head.weight = nn.Parameter(model.embed.weight.detach().clone())

    def forward(self, x):
        if self.is_first:
            x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        if self.is_last:
            x = self.lm_head(self.final_norm(x))
        return x


def pipeline_run(stage: PipelineStage, tokens: torch.Tensor,
                 targets: torch.Tensor, n_microbatches: int,
                 d_model: int, backward: bool = True):
    """GPipe fill-drain schedule. Returns mean loss (valid on last stage).

    P2P protocol: activations flow stage r -> r+1 during the forward phase,
    gradients flow r+1 -> r in reverse microbatch order during backward.
    Shapes are static (microbatch size and seq len fixed), so no shape
    handshake is needed.
    """
    rank, world = stage.stage, stage.n_stages
    mbs_tok = tokens.chunk(n_microbatches)
    mbs_tgt = targets.chunk(n_microbatches)
    B_mb, S = mbs_tok[0].shape

    stash = []          # (input_with_grad, output) per microbatch
    losses = []

    for m in range(n_microbatches):
        if stage.is_first:
            x_in = mbs_tok[m]
        else:
            x_in = torch.empty(B_mb, S, d_model)
            dist.recv(x_in, src=rank - 1)
            x_in.requires_grad_(True)
        out = stage(x_in)
        if not stage.is_last:
            dist.send(out.detach(), dst=rank + 1)
        stash.append((x_in, out))

    if not backward:
        return torch.stack([
            F.cross_entropy(o.reshape(-1, o.shape[-1]), t.reshape(-1))
            for (_, o), t in zip(stash, mbs_tgt)
        ]).mean() if stage.is_last else None

    for m in reversed(range(n_microbatches)):
        x_in, out = stash[m]
        if stage.is_last:
            loss = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                   mbs_tgt[m].reshape(-1))
            losses.append(loss.detach())
            # scale so summed microbatch grads equal the full-batch mean grad
            (loss / n_microbatches).backward()
        else:
            grad = torch.empty_like(out)
            dist.recv(grad, src=rank + 1)
            out.backward(grad)
        if not stage.is_first:
            dist.send(x_in.grad, dst=rank - 1)

    return torch.stack(losses).mean() if stage.is_last else None
