"""CLI + minimal HTTP endpoint tying the stack together.

The model is a tiny char-level LM trained on the synthetic corpus (this repo
is a systems project — the "LLM" is deliberately small; every serving
optimization is real).

Composition matrix (what runs with what):
  * engine path  (`chat`, `serve`): paged KV cache + continuous batching
    (+ Triton kernels on GPU), optionally over INT8/INT4-quantized weights.
  * speculative path (`chat --speculative`): draft+verify decoding via its
    own contiguous KV cache — speculative-inside-the-paged-engine is not
    composed here (real systems do this; see README limitations).

Usage:
  python -m serve.api train                 # train + save checkpoint
  python -m serve.api chat -p "kernel "     # generate a continuation
  python -m serve.api chat -p "..." --quant int8 --speculative
  python -m serve.api serve --port 8123     # HTTP: POST /generate
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch

from serve.engine import Engine
from serve.quantize import quantize_model
from serve.speculative import speculative_generate
from train.config import ModelConfig, TrainConfig
from train.model import Transformer
from train.train_loop import CharDataset, synthetic_corpus, train

CKPT = Path(__file__).parent.parent / "checkpoints"


def cmd_train(args):
    mc = ModelConfig(vocab_size=64, d_model=128, n_layers=4, n_heads=4,
                     max_seq_len=512)
    tc = TrainConfig(steps=args.steps, seq_len=128, batch_size=8, lr=1e-3)
    result = train(mc, tc)
    ds = result["dataset"]
    CKPT.mkdir(exist_ok=True)
    torch.save({"state_dict": result["model"].state_dict(), "config": vars(mc) |
                {"_head_dim": mc.head_dim}, "itos": ds.itos}, CKPT / "model.pt")
    print(f"saved {CKPT/'model.pt'}  (final loss {result['final_loss']:.3f})")


def _load():
    path = CKPT / "model.pt"
    if not path.exists():
        raise SystemExit("no checkpoint — run `python -m serve.api train` first")
    blob = torch.load(path, weights_only=False)
    cfg_d = {k: v for k, v in blob["config"].items() if not k.startswith("_")}
    cfg = ModelConfig(**cfg_d)
    model = Transformer(cfg)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    itos = blob["itos"]
    stoi = {c: i for i, c in enumerate(itos)}
    return model, cfg, stoi, itos


def _encode(text, stoi):
    toks = [stoi[c] for c in text if c in stoi]
    return toks or [0]


def _generate(model, cfg, stoi, itos, prompt: str, max_new: int,
              quant: str | None = None, speculative: bool = False,
              use_kernels: bool | None = None) -> str:
    toks = _encode(prompt, stoi)
    cuda = torch.cuda.is_available()
    if use_kernels is None:
        use_kernels = cuda
    if quant:
        bits = 8 if quant == "int8" else 4
        model = quantize_model(model, bits=bits, use_kernel=use_kernels)
    if speculative:
        draft_cfg = ModelConfig(vocab_size=cfg.vocab_size, d_model=64,
                                n_layers=1, n_heads=2,
                                max_seq_len=cfg.max_seq_len)
        torch.manual_seed(0)
        draft = Transformer(draft_cfg).eval()   # untrained draft: still exact
        out, stats = speculative_generate(model, draft, toks, max_new)
        text = "".join(itos[t] for t in out)
        return text + f"\n[speculative: acceptance {stats.acceptance_rate:.0%}, " \
                      f"{stats.tokens_per_round:.2f} tok/round]"
    eng = Engine(model, num_blocks=512, block_size=16,
                 use_kernels=use_kernels,
                 device="cuda" if cuda else "cpu")
    req = eng.submit(toks, max_new_tokens=max_new)
    eng.run_until_done()
    return "".join(itos[t] for t in req.generated)


def cmd_chat(args):
    model, cfg, stoi, itos = _load()
    out = _generate(model, cfg, stoi, itos, args.prompt, args.max_new,
                    quant=args.quant, speculative=args.speculative)
    print(f"{args.prompt!r} -> {out!r}")


def cmd_serve(args):
    model, cfg, stoi, itos = _load()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/generate":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(
                int(self.headers["Content-Length"])))
            text = _generate(model, cfg, stoi, itos,
                             body.get("prompt", ""),
                             int(body.get("max_new_tokens", 64)),
                             quant=body.get("quant"))
            payload = json.dumps({"completion": text}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    print(f"listening on http://127.0.0.1:{args.port}  "
          f"POST /generate {{\"prompt\": \"...\"}}")
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=400)
    c = sub.add_parser("chat")
    c.add_argument("-p", "--prompt", required=True)
    c.add_argument("--max-new", type=int, default=64)
    c.add_argument("--quant", choices=["int8", "int4"])
    c.add_argument("--speculative", action="store_true")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8123)
    args = ap.parse_args()
    {"train": cmd_train, "chat": cmd_chat, "serve": cmd_serve}[args.cmd](args)


if __name__ == "__main__":
    main()
