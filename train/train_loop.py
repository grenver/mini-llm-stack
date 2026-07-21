"""Minimal training loop with a self-contained synthetic dataset.

The dataset is a deterministic char-level corpus generated in-process (no
downloads), structured enough that a tiny model's loss drops fast — which is
what the correctness-oriented phases need: a "does loss decrease with these
kernels in the path" signal, not a real language model.
"""

from __future__ import annotations

import math
import time

import torch

from train.config import ModelConfig, TrainConfig
from train.model import Transformer

_WORDS = (
    "kernel triton block warp tensor expert router cache page token stream "
    "batch fuse scale shard rank pipe stage decode prefill spec draft grad "
    "loss adam norm rope swiglu logit vocab layer head query key value "
).split()


def synthetic_corpus(n_chars: int = 65536, seed: int = 7) -> str:
    """Deterministic pseudo-text: markov-ish word chains with local structure."""
    g = torch.Generator().manual_seed(seed)
    words = []
    idx = 0
    while sum(len(w) + 1 for w in words) < n_chars:
        # biased transition: mostly nearby words in the list -> learnable bigrams
        step = int(torch.randint(0, 5, (1,), generator=g))
        idx = (idx + step) % len(_WORDS)
        words.append(_WORDS[idx])
        if int(torch.randint(0, 12, (1,), generator=g)) == 0:
            words.append(".")
    return " ".join(words)[:n_chars]


class CharDataset:
    def __init__(self, text: str, seq_len: int, seed: int = 0):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.seq_len = seq_len
        self.g = torch.Generator().manual_seed(seed)

    def batch(self, batch_size: int, device: str = "cpu"):
        ix = torch.randint(0, len(self.data) - self.seq_len - 1, (batch_size,),
                           generator=self.g)
        x = torch.stack([self.data[i:i + self.seq_len] for i in ix])
        y = torch.stack([self.data[i + 1:i + self.seq_len + 1] for i in ix])
        return x.to(device), y.to(device)


def lr_at(step: int, tc: TrainConfig) -> float:
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    t = (step - tc.warmup_steps) / max(1, tc.steps - tc.warmup_steps)
    return tc.lr * 0.5 * (1 + math.cos(math.pi * t))


def train(mc: ModelConfig, tc: TrainConfig, model: Transformer | None = None,
          log=print) -> dict:
    device = tc.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(tc.seed)

    ds = CharDataset(synthetic_corpus(), tc.seq_len, seed=tc.seed)
    if mc.vocab_size < ds.vocab_size:
        raise ValueError(f"vocab_size {mc.vocab_size} < corpus vocab {ds.vocab_size}")

    if model is None:
        model = Transformer(mc)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, betas=tc.betas,
                            weight_decay=tc.weight_decay)

    losses = []
    t0 = time.perf_counter()
    for step in range(tc.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, tc)
        x, y = ds.batch(tc.batch_size, device)
        _, loss, aux = model(x, y)
        total = loss + tc.aux_loss_coef * aux
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        losses.append(loss.item())
        if step % tc.log_every == 0 or step == tc.steps - 1:
            log(f"step {step:4d}  loss {loss.item():.4f}  aux {aux.item():.4f}  "
                f"lr {lr_at(step, tc):.2e}")

    return {
        "losses": losses,
        "first_loss": losses[0],
        "final_loss": sum(losses[-10:]) / min(10, len(losses)),
        "wall_time_s": time.perf_counter() - t0,
        "model": model,
        "dataset": ds,
    }


if __name__ == "__main__":
    mc = ModelConfig(vocab_size=64, d_model=128, n_layers=4, n_heads=4,
                     max_seq_len=256)
    tc = TrainConfig(steps=200, seq_len=128, batch_size=8)
    result = train(mc, tc)
    print(f"first loss {result['first_loss']:.3f} -> "
          f"final loss {result['final_loss']:.3f}")
