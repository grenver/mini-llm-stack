"""Spawned worker functions for parallelism tests (module-level for pickling).

Each worker builds the SAME dense reference model (same seed, same data),
runs the reference forward/backward locally, then runs the sharded/staged
version through real torch.distributed collectives and asserts the outputs
and gradients match its own slice of the reference. Assertion failures
propagate to the parent via mp.spawn.
"""

import torch
import torch.distributed as dist

from train.config import tiny_config
from train.model import Transformer
from train.parallel import PipelineStage, init_dist, pipeline_run, shard_model

TOL = dict(rtol=2e-4, atol=2e-4)


def tp_worker(rank, world, store_path):
    init_dist(rank, world, store_path)
    try:
        torch.manual_seed(0)
        cfg = tiny_config(d_model=64, n_layers=2, n_heads=4)
        dense = Transformer(cfg)
        tokens = torch.randint(0, cfg.vocab_size, (2, 16))

        _, ref_loss, _ = dense(tokens, tokens)
        ref_loss.backward()

        tp = shard_model(dense, rank, world)
        logits, loss, _ = tp(tokens, tokens)

        with torch.no_grad():
            ref_logits = dense(tokens)

        torch.testing.assert_close(logits, ref_logits, **TOL)
        torch.testing.assert_close(loss, ref_loss.detach(), **TOL)

        loss.backward()
        for i, blk in enumerate(tp.blocks):
            dblk = dense.blocks[i]
            # column-parallel: rows are sharded
            for name in ("wq", "wk", "wv"):
                sl = getattr(dblk.attn, name).weight.shape[0] // world
                ref = getattr(dblk.attn, name).weight.grad[rank * sl:(rank + 1) * sl]
                got = getattr(blk.attn, name).weight.grad
                torch.testing.assert_close(got, ref, **TOL)
            # row-parallel: columns are sharded
            sl = dblk.attn.wo.weight.shape[1] // world
            ref = dblk.attn.wo.weight.grad[:, rank * sl:(rank + 1) * sl]
            torch.testing.assert_close(blk.attn.wo.weight.grad, ref, **TOL)
            for name in ("w_gate", "w_up"):
                sl = getattr(dblk.mlp, name).weight.shape[0] // world
                ref = getattr(dblk.mlp, name).weight.grad[rank * sl:(rank + 1) * sl]
                torch.testing.assert_close(getattr(blk.mlp, name).weight.grad,
                                           ref, **TOL)
            sl = dblk.mlp.w_down.weight.shape[1] // world
            ref = dblk.mlp.w_down.weight.grad[:, rank * sl:(rank + 1) * sl]
            torch.testing.assert_close(blk.mlp.w_down.weight.grad, ref, **TOL)
    finally:
        dist.destroy_process_group()


def pp_worker(rank, world, store_path):
    init_dist(rank, world, store_path)
    try:
        torch.manual_seed(0)
        cfg = tiny_config(d_model=64, n_layers=4, n_heads=4,
                          tie_embeddings=False)
        dense = Transformer(cfg)
        tokens = torch.randint(0, cfg.vocab_size, (4, 16))

        _, ref_loss, _ = dense(tokens, tokens)
        ref_loss.backward()

        stage = PipelineStage(dense, rank, world)
        loss = pipeline_run(stage, tokens, tokens, n_microbatches=2,
                            d_model=cfg.d_model)
        if stage.is_last:
            torch.testing.assert_close(loss, ref_loss.detach(), **TOL)

        # every stage checks its blocks' grads against the dense reference
        per = cfg.n_layers // world
        lo = rank * per
        for i, blk in enumerate(stage.blocks):
            dblk = dense.blocks[lo + i]
            for (n1, p), (n2, dp) in zip(blk.named_parameters(),
                                         dblk.named_parameters()):
                assert n1 == n2
                torch.testing.assert_close(p.grad, dp.grad, **TOL)
        if stage.is_first:
            torch.testing.assert_close(stage.embed.weight.grad,
                                       dense.embed.weight.grad, **TOL)
        if stage.is_last:
            torch.testing.assert_close(stage.lm_head.weight.grad,
                                       dense.lm_head.weight.grad, **TOL)
            torch.testing.assert_close(stage.final_norm.weight.grad,
                                       dense.final_norm.weight.grad, **TOL)
    finally:
        dist.destroy_process_group()
