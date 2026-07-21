"""Disaggregated prefill/decode serving (single-GPU SIMULATION).

Why real systems do this: prefill is compute-bound (long sequences, big
GEMMs) and decode is memory-bandwidth-bound (one token per step, weights
re-read every step). Mixing them in one pool makes decode latency spiky —
a long prompt's prefill stalls every in-flight decode. Splitting them onto
separate pools with a KV-cache handoff isolates the two regimes and lets
each pool batch its own kind of work.

WHAT THIS ENVIRONMENT CANNOT SHOW: with both "pools" time-sharing one
physical device, there is no isolation win to measure — the split ADDS
KV serialization + IPC while removing nothing. Expect the unified engine
to beat the disaggregated one here; the honest deliverables are (a) the
split produces identical output to the unified engine and (b) the overhead
breakdown (where the transfer cost actually goes).

Topology: router (parent) → prefill worker → decode worker → router,
connected by multiprocessing queues. The KV handoff is serialized to a
byte buffer (torch.save) and deserialized on the decode side — the same
wire-format discipline a real disaggregated system uses over NIC/NVLink,
and deliberately NOT torch.multiprocessing's shared-memory tensor passing:
on Linux that path moves tensors via file-descriptor passing, and a feeder-
thread serialization failure silently DROPS the item (observed in CI as
"all payloads lost, sentinel delivered"). Bytes go through the plain
pickler on every platform.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass

import torch
import torch.multiprocessing as mp

from serve.kv_cache import PagedKVCache
from train.config import ModelConfig
from train.model import Transformer

SENTINEL = None


@dataclass
class DisaggResult:
    req_id: int
    generated: list[int]
    prefill_ms: float
    transfer_ms: float          # enqueue -> decode-side ingest complete
    kv_bytes: int
    decode_ms: float


def _build_model(cfg_dict: dict, state_dict) -> Transformer:
    cfg = ModelConfig(**cfg_dict)
    model = Transformer(cfg)
    model.load_state_dict(state_dict)
    return model.eval()


def prefill_worker(cfg_dict, state_dict, req_q, kv_q):
    """Runs prompts, ships (first_token, per-layer KV, timings) downstream."""
    from serve.speculative import CachedModel
    model = _build_model(cfg_dict, state_dict)
    while True:
        item = req_q.get()
        if item is SENTINEL:
            kv_q.put(SENTINEL)
            return
        req_id, prompt, max_new = item
        t0 = time.perf_counter()
        cm = CachedModel(model)
        logits = cm.logits_for(list(prompt))
        first_tok = int(logits[-1].argmax())
        kv = [(k[0].contiguous(), v[0].contiguous()) for k, v in cm.caches]
        buf = io.BytesIO()
        torch.save(kv, buf)
        payload = buf.getvalue()
        prefill_ms = (time.perf_counter() - t0) * 1000
        kv_q.put((req_id, list(prompt), first_tok, payload, max_new,
                  prefill_ms, len(payload), time.perf_counter()))


def decode_worker(cfg_dict, state_dict, kv_q, done_q, num_blocks, block_size):
    """Ingests prefilled KV into a paged pool; runs batched decode steps."""
    from serve.engine import PagedRunner
    model = _build_model(cfg_dict, state_dict)
    cfg = model.cfg
    cache = PagedKVCache(cfg.n_layers, cfg.n_heads, cfg.head_dim,
                         num_blocks, block_size)
    runner = PagedRunner(model, cache, use_kernels=False)

    running: dict[int, dict] = {}
    drained = False
    while not (drained and not running):
        # ingest everything waiting (continuous admission)
        while True:
            try:
                item = kv_q.get_nowait() if running or drained else kv_q.get()
            except Exception:
                break
            if item is SENTINEL:
                drained = True
                break
            (req_id, prompt, first_tok, payload, max_new,
             prefill_ms, kv_bytes, sent_ts) = item
            kv = torch.load(io.BytesIO(payload), weights_only=True)
            S = len(prompt)
            cache.allocate(req_id, S + 1)
            for li, (k, v) in enumerate(kv):
                cache.write_prefill(li, req_id, k, v)
            cache.set_len(req_id, S)
            transfer_ms = (time.perf_counter() - sent_ts) * 1000
            running[req_id] = {
                "generated": [first_tok], "max_new": max_new,
                "prefill_ms": prefill_ms, "transfer_ms": transfer_ms,
                "kv_bytes": kv_bytes, "decode_t0": time.perf_counter(),
            }
            if max_new <= 1:
                _finish(req_id, running, cache, done_q)

        if not running:
            continue
        seq_ids = list(running.keys())
        for sid in seq_ids:
            if not cache.append_slot(sid):
                raise RuntimeError("decode pool out of KV blocks")
        last = [running[s]["generated"][-1] for s in seq_ids]
        logits = runner.decode_step(seq_ids, last)
        for i, sid in enumerate(seq_ids):
            tok = int(logits[i].argmax())
            rec = running[sid]
            rec["generated"].append(tok)
            if len(rec["generated"]) >= rec["max_new"]:
                _finish(sid, running, cache, done_q)
    done_q.put(SENTINEL)


def _finish(req_id, running, cache, done_q):
    rec = running.pop(req_id)
    cache.free(req_id)
    done_q.put(DisaggResult(
        req_id, rec["generated"], round(rec["prefill_ms"], 2),
        round(rec["transfer_ms"], 2), rec["kv_bytes"],
        round((time.perf_counter() - rec["decode_t0"]) * 1000, 2)))


def run_disaggregated(model: Transformer, requests: list[tuple[list[int], int]],
                      num_blocks: int = 256, block_size: int = 16):
    """requests: [(prompt, max_new), ...] -> (results by req_id, wall_s)."""
    ctx = mp.get_context("spawn")
    req_q, kv_q, done_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    cfg_dict = {k: v for k, v in vars(model.cfg).items()}
    state = {k: v.cpu() for k, v in model.state_dict().items()}

    pf = ctx.Process(target=prefill_worker, args=(cfg_dict, state, req_q, kv_q))
    dc = ctx.Process(target=decode_worker, args=(cfg_dict, state, kv_q, done_q,
                                                 num_blocks, block_size))
    pf.start()
    dc.start()
    t0 = time.perf_counter()
    for i, (prompt, max_new) in enumerate(requests):
        req_q.put((i, prompt, max_new))
    req_q.put(SENTINEL)

    results = {}
    while True:
        item = done_q.get(timeout=300)   # fail loudly, never hang CI
        if item is SENTINEL:
            break
        results[item.req_id] = item
    wall = time.perf_counter() - t0
    pf.join()
    dc.join()
    return results, wall
