"""Serving engine: continuous batching over a paged KV cache.

Continuous batching
-------------------
Static batching waits for a whole batch to finish before admitting new
requests, so one long generation holds the batch hostage and short requests
queue behind it. Continuous batching (Orca/vLLM) reschedules every step:

  each engine step:
    1. finished sequences release their cache blocks immediately
    2. waiting requests are admitted if enough free blocks exist
       (new requests join mid-flight; admission control is just block math)
    3. newly admitted prompts are prefilled (Phase 1 flash kernel),
       their K/V written into paged blocks
    4. ONE batched decode step runs for all running sequences
       (Phase 5 paged-attention kernel walks each block table)

The decode batch therefore grows/shrinks token-by-token — GPU work per step
tracks live demand instead of the slowest member of a static batch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from serve.kv_cache import PagedKVCache
from train.model import Transformer, naive_attention

WAITING, RUNNING, FINISHED = "waiting", "running", "finished"


@dataclass
class Request:
    req_id: int
    prompt: list[int]
    max_new_tokens: int = 32
    eos_token: int | None = None
    state: str = WAITING
    generated: list[int] = field(default_factory=list)
    arrive_time: float = field(default_factory=time.perf_counter)
    first_token_time: float | None = None
    finish_time: float | None = None

    @property
    def tokens(self) -> list[int]:
        return self.prompt + self.generated


def _rope_single(rotary, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to per-sequence single-token q/k: x [n, H, D], pos [n]."""
    cos = rotary.cos[pos].to(x.dtype)[:, None, :]     # [n, 1, D/2]
    sin = rotary.sin[pos].to(x.dtype)[:, None, :]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class PagedRunner:
    """Runs a Transformer forward against the paged KV cache.

    use_kernels=False swaps the paged-attention kernel for a gather-copy
    reference path (used by correctness tests and as CPU fallback — the
    interpreter is far too slow to serve with).
    """

    def __init__(self, model: Transformer, cache: PagedKVCache,
                 use_kernels: bool = True):
        self.model = model
        self.cache = cache
        self.use_kernels = use_kernels
        self.cfg = model.cfg

    @torch.no_grad()
    def prefill(self, seq_id: int, tokens: list[int]) -> torch.Tensor:
        """Run the prompt, write K/V into the cache, return last-token logits."""
        cfg = self.cfg
        dev = self.cache.device
        t = torch.tensor([tokens], dtype=torch.long, device=dev)
        S = t.shape[1]
        pos = torch.arange(S, device=dev)
        x = self.model.embed(t)
        for li, blk in enumerate(self.model.blocks):
            h = blk.attn_norm(x)
            B = 1
            H, hd = cfg.n_heads, cfg.head_dim
            q = blk.attn.wq(h).view(B, S, H, hd).transpose(1, 2)
            k = blk.attn.wk(h).view(B, S, H, hd).transpose(1, 2)
            v = blk.attn.wv(h).view(B, S, H, hd).transpose(1, 2)
            q = blk.attn.rotary(q, pos)
            k = blk.attn.rotary(k, pos)
            if self.use_kernels:
                from kernels.attention_fwd import flash_attention
                o = flash_attention(q, k, v, causal=True)
            else:
                o = naive_attention(q, k, v, causal=True)
            self.cache.write_prefill(li, seq_id, k[0].to(self.cache.dtype),
                                     v[0].to(self.cache.dtype))
            x = x + blk.attn.wo(o.transpose(1, 2).reshape(B, S, cfg.d_model))
            x = x + blk.mlp(blk.mlp_norm(x))
        self.cache.set_len(seq_id, S)
        logits = self.model.lm_head(self.model.final_norm(x))
        return logits[0, -1]

    @torch.no_grad()
    def decode_step(self, seq_ids: list[int], last_tokens: list[int]) -> torch.Tensor:
        """One batched decode step for all running sequences.

        Returns logits [n_seqs, vocab]. Write-then-attend: each layer first
        writes the new token's K/V, then runs paged attention over a context
        that includes it.
        """
        cfg = self.cfg
        cache = self.cache
        dev = cache.device
        n = len(seq_ids)
        H, hd = cfg.n_heads, cfg.head_dim

        t = torch.tensor(last_tokens, dtype=torch.long, device=dev)
        new_pos = [cache.lens[s] for s in seq_ids]             # slot for new token
        pos = torch.tensor(new_pos, device=dev)
        x = self.model.embed(t)                                # [n, D]

        tables, _ = cache.batch_tables(seq_ids)
        ctx = (pos + 1).to(torch.int32)                        # includes new token

        for li, blk in enumerate(self.model.blocks):
            h = blk.attn_norm(x)
            q = blk.attn.wq(h).view(n, H, hd)
            k = blk.attn.wk(h).view(n, H, hd)
            v = blk.attn.wv(h).view(n, H, hd)
            q = _rope_single(blk.attn.rotary, q, pos)
            k = _rope_single(blk.attn.rotary, k, pos)

            for i, s in enumerate(seq_ids):
                cache.write_decode(li, s, new_pos[i], k[i].to(cache.dtype),
                                   v[i].to(cache.dtype))

            if self.use_kernels:
                from kernels.paged_attention import paged_attention_decode
                o = paged_attention_decode(q, cache.k_pool[li], cache.v_pool[li],
                                           tables, ctx)
            else:
                outs = []
                for i, s in enumerate(seq_ids):
                    kk, vv = cache.gather_contiguous(li, s, n=new_pos[i] + 1)
                    oo = naive_attention(q[i][:, None], kk, vv, causal=False)
                    outs.append(oo[:, 0])
                o = torch.stack(outs)
            x = x + blk.attn.wo(o.reshape(n, cfg.d_model))
            x = x + blk.mlp(blk.mlp_norm(x[:, None]))[:, 0]

        for i, s in enumerate(seq_ids):
            cache.set_len(s, new_pos[i] + 1)
        return self.model.lm_head(self.model.final_norm(x))


class Engine:
    """Continuous-batching scheduler."""

    def __init__(self, model: Transformer, num_blocks: int = 256,
                 block_size: int = 16, max_batch: int = 16,
                 use_kernels: bool = True, device: str = "cpu",
                 dtype: torch.dtype = torch.float32):
        cfg = model.cfg
        self.cache = PagedKVCache(cfg.n_layers, cfg.n_heads, cfg.head_dim,
                                  num_blocks, block_size, device, dtype)
        self.runner = PagedRunner(model.to(device), self.cache, use_kernels)
        self.max_batch = max_batch
        self.max_seq_len = cfg.max_seq_len
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self._next_id = 0
        self.steps = 0

    def submit(self, prompt: list[int], max_new_tokens: int = 32,
               eos_token: int | None = None) -> Request:
        assert len(prompt) + 1 <= self.max_seq_len, "prompt too long"
        req = Request(self._next_id, list(prompt), max_new_tokens, eos_token)
        self._next_id += 1
        self.waiting.append(req)
        return req

    def _sample(self, logits: torch.Tensor) -> int:
        return int(logits.argmax(-1))

    def _admit(self):
        while (self.waiting and len(self.running) < self.max_batch):
            req = self.waiting[0]
            # need blocks for all tokens so far (prompt, or prompt+generated
            # when re-admitting a preempted request) plus one decode slot
            if not self.cache.can_allocate(len(req.tokens) + 1):
                break
            self.waiting.pop(0)
            self.cache.allocate(req.req_id, len(req.tokens) + 1)
            logits = self.runner.prefill(req.req_id, req.tokens)
            tok = self._sample(logits)
            req.generated.append(tok)
            if req.first_token_time is None:
                req.first_token_time = time.perf_counter()
            req.state = RUNNING
            self.running.append(req)
            self._maybe_finish(req, tok)

    def _preempt(self, req: Request):
        """Out of blocks: release the request's cache and requeue it. On
        re-admission its full token list is re-prefilled (recompute-style
        preemption, like vLLM's) — greedy decoding makes the continuation
        deterministic, so preemption never changes output."""
        self.running.remove(req)
        self.cache.free(req.req_id)
        req.state = WAITING
        self.waiting.insert(0, req)

    def _maybe_finish(self, req: Request, tok: int):
        done = (len(req.generated) >= req.max_new_tokens
                or (req.eos_token is not None and tok == req.eos_token)
                or len(req.tokens) >= self.max_seq_len)
        if done and req.state == RUNNING:
            req.state = FINISHED
            req.finish_time = time.perf_counter()
            self.running.remove(req)
            self.finished.append(req)
            self.cache.free(req.req_id)

    def step(self) -> bool:
        """One scheduler iteration. Returns True while work remains."""
        self._admit()
        if self.running:
            # reserve the slot each new token's KV will occupy; if the pool
            # is exhausted, preempt the newest request(s) to make room
            i = 0
            while i < len(self.running):
                req = self.running[i]
                if self.cache.append_slot(req.req_id):
                    i += 1
                    continue
                victim = self.running[-1]
                if victim is req and len(self.running) == 1 and not self.waiting:
                    raise RuntimeError(
                        "single request exceeds total KV cache capacity")
                self._preempt(victim)
                # if we preempted ourselves, the loop end condition handles it
            if self.running:
                seq_ids = [r.req_id for r in self.running]
                last = [r.generated[-1] for r in self.running]
                logits = self.runner.decode_step(seq_ids, last)
                for i, req in enumerate(list(self.running)):
                    tok = self._sample(logits[i])
                    req.generated.append(tok)
                    self._maybe_finish(req, tok)
        self.steps += 1
        return bool(self.running or self.waiting)

    def run_until_done(self, max_steps: int = 10_000):
        t0 = time.perf_counter()
        while self.step():
            if self.steps >= max_steps:
                raise RuntimeError("engine exceeded max_steps")
        wall = time.perf_counter() - t0
        toks = sum(len(r.generated) for r in self.finished)
        return {"wall_s": wall, "generated_tokens": toks,
                "tokens_per_s": toks / wall if wall > 0 else float("inf"),
                "requests": len(self.finished)}
