"""GPU correctness smoke suite — run BEFORE any GPU benchmark.

The pytest suite exercises every kernel under the CPU interpreter; this
script re-verifies each one with COMPILED kernels on CUDA tensors, where a
whole different class of bugs lives (block-size constraints, dtype
conversions inside tl.dot, atomics, actual masking behavior). Benchmarks on
unverified kernels are worthless — kaggle_bench.py refuses to bench if this
fails.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root importable

import torch

assert torch.cuda.is_available(), "GPU smoke test needs CUDA"
DEV = "cuda"
FAILS = []


def check(name):
    def deco(fn):
        def run():
            try:
                fn()
                print(f"PASS  {name}")
            except Exception:
                print(f"FAIL  {name}")
                traceback.print_exc()
                FAILS.append(name)
        run()
    return deco


@check("attention fwd fp32 vs naive")
def _():
    from kernels.attention_fwd import flash_attention
    from train.model import naive_attention
    torch.manual_seed(0)
    for (B, H, Sq, Sk, D) in [(2, 4, 128, 128, 64), (1, 2, 100, 100, 32),
                              (1, 2, 5, 133, 64)]:
        for causal in (True, False):
            q = torch.randn(B, H, Sq, D, device=DEV)
            k = torch.randn(B, H, Sk, D, device=DEV)
            v = torch.randn(B, H, Sk, D, device=DEV)
            out = flash_attention(q, k, v, causal=causal)
            ref = naive_attention(q, k, v, causal=causal)
            torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@check("attention fwd fp16 vs sdpa")
def _():
    from kernels.attention_fwd import flash_attention
    import torch.nn.functional as F
    q, k, v = (torch.randn(2, 4, 256, 64, device=DEV, dtype=torch.float16)
               for _ in range(3))
    out = flash_attention(q, k, v, causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)


@check("attention bwd vs autograd")
def _():
    from kernels.autograd_ops import flash_attention_train
    from train.model import naive_attention
    torch.manual_seed(1)
    B, H, S, D = 2, 2, 96, 64
    q, k, v = (torch.randn(B, H, S, D, device=DEV, requires_grad=True)
               for _ in range(3))
    do = torch.randn(B, H, S, D, device=DEV)
    flash_attention_train(q, k, v, causal=True).backward(do)
    grads = [t.grad.clone() for t in (q, k, v)]
    refs = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    naive_attention(*refs, causal=True).backward(do)
    for g, r in zip(grads, refs):
        torch.testing.assert_close(g, r.grad, rtol=1e-4, atol=1e-4)


@check("moe dispatch/combine fwd+bwd vs naive")
def _():
    from kernels.autograd_ops import moe_dispatch_combine_train
    torch.manual_seed(2)
    T, E, k, D = 512, 8, 2, 128
    experts = torch.nn.ModuleList(
        [torch.nn.Linear(D, D, bias=False) for _ in range(E)]).to(DEV)
    flat = torch.randn(T, D, device=DEV, requires_grad=True)
    idx = torch.randint(0, E, (T, k), device=DEV)
    w = torch.rand(T, k, device=DEV)
    w = (w / w.sum(-1, keepdim=True)).requires_grad_(True)

    out = moe_dispatch_combine_train(flat, w, idx, experts)
    g = torch.randn_like(out)
    out.backward(g)
    dflat, dw = flat.grad.clone(), w.grad.clone()

    flat2 = flat.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    ref = torch.zeros_like(flat2)
    for e in range(E):
        tok, slot = torch.where(idx == e)
        if tok.numel():
            ref = ref.index_add(0, tok, experts[e](flat2[tok]) *
                                w2[tok, slot, None])
    ref.backward(g)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dflat, flat2.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dw, w2.grad, rtol=1e-4, atol=1e-4)


@check("int8/int4 fused matmul vs dequant reference")
def _():
    from kernels.dequant_matmul import int4_matmul, int8_dgrad, int8_matmul
    from serve.quantize import (dequantize_int4, dequantize_int8,
                                quantize_int4, quantize_int8)
    torch.manual_seed(3)
    M, N, K = 33, 256, 512
    a = torch.randn(M, K, device=DEV)
    w = torch.randn(N, K, device=DEV)
    w8, s8 = quantize_int8(w)
    torch.testing.assert_close(int8_matmul(a, w8, s8),
                               a @ dequantize_int8(w8, s8).t(),
                               rtol=1e-3, atol=1e-3)
    w4, s4 = quantize_int4(w, 64)
    torch.testing.assert_close(int4_matmul(a, w4, s4, 64),
                               a @ dequantize_int4(w4, s4, 64).t(),
                               rtol=1e-3, atol=1e-3)
    dy = torch.randn(M, N, device=DEV)
    torch.testing.assert_close(int8_dgrad(dy, w8, s8),
                               (dy * s8[None, :]) @ w8.float(),
                               rtol=1e-3, atol=1e-3)


@check("paged attention vs contiguous")
def _():
    from kernels.paged_attention import paged_attention_decode
    from serve.kv_cache import PagedKVCache
    from train.model import naive_attention
    torch.manual_seed(4)
    H, D, bs = 8, 64, 16
    lens = [7, 200, 45, 128]
    cache = PagedKVCache(1, H, D, num_blocks=64, block_size=bs, device=DEV)
    ks, vs = [], []
    for sid, L in enumerate(lens):
        cache.allocate(sid, L)
        kk = torch.randn(H, L, D, device=DEV)
        vv = torch.randn(H, L, D, device=DEV)
        cache.write_prefill(0, sid, kk, vv)
        cache.set_len(sid, L)
        ks.append(kk)
        vs.append(vv)
    q = torch.randn(len(lens), H, D, device=DEV)
    tables, ctx = cache.batch_tables(list(range(len(lens))))
    out = paged_attention_decode(q, cache.k_pool[0], cache.v_pool[0],
                                 tables, ctx)
    for i, L in enumerate(lens):
        ref = naive_attention(q[i][:, None], ks[i], vs[i], causal=False)[:, 0]
        torch.testing.assert_close(out[i], ref, rtol=1e-4, atol=1e-4)


@check("engine on GPU matches naive generation")
def _():
    from serve.engine import Engine
    from train.config import tiny_config
    from train.model import Transformer
    torch.manual_seed(5)
    cfg = tiny_config(d_model=64, n_layers=2, n_heads=2, max_seq_len=128)
    model = Transformer(cfg).eval().to(DEV)
    prompts = [[1, 2, 3, 4, 5], [9, 8, 7], list(range(20, 40))]
    refs = [model.generate_naive(torch.tensor([p], device=DEV), 10)[0, len(p):]
            .tolist() for p in prompts]
    eng = Engine(model, num_blocks=64, block_size=16, use_kernels=True,
                 device=DEV)
    reqs = [eng.submit(p, max_new_tokens=10) for p in prompts]
    eng.run_until_done()
    for r, ref in zip(reqs, refs):
        assert r.generated == ref, (r.generated, ref)


@check("fp8 e4m3 decode + matmul")
def _():
    from kernels.fp8_matmul import (E4M3_MAX, decode_e4m3, dequantize_e4m3,
                                    fp8_matmul, quantize_e4m3)
    bits = torch.arange(256, dtype=torch.uint8, device=DEV)
    ref = bits.view(torch.float8_e4m3fn).float()
    ours = decode_e4m3(bits)
    ok = ~ref.isnan()
    torch.testing.assert_close(ours[ok], ref[ok], rtol=0, atol=0)
    a = torch.randn(64, 128, device=DEV)
    b = torch.randn(96, 128, device=DEV)
    sa = float(a.abs().amax()) / E4M3_MAX
    sb = float(b.abs().amax()) / E4M3_MAX
    ab, bb = quantize_e4m3(a, sa), quantize_e4m3(b, sb)
    torch.testing.assert_close(fp8_matmul(ab, bb, sa, sb),
                               dequantize_e4m3(ab, sa) @
                               dequantize_e4m3(bb, sb).t(),
                               rtol=1e-4, atol=1e-4)


@check("fusion compiler generated kernel")
def _():
    import torch.nn.functional as F
    from kernels.fusion_compiler import FusedKernel, softmax_graph
    q = torch.randn(64, 96, device=DEV)
    k = torch.randn(80, 96, device=DEV)
    fk = FusedKernel(softmax_graph(with_matmul=True, scale=96 ** -0.5))
    ref = F.softmax((q @ k.t()) * 96 ** -0.5, dim=-1)
    torch.testing.assert_close(fk(q, k), ref, rtol=1e-4, atol=1e-5)


print(f"\n{'ALL GPU SMOKE TESTS PASSED' if not FAILS else 'FAILURES: ' + str(FAILS)}")
sys.exit(1 if FAILS else 0)
