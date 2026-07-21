"""Speculative decoding: draft proposes, target verifies in one forward.

The mechanism
-------------
Autoregressive decode runs one target forward per token, and each forward is
memory-bound on reading the weights — the GPU is idle-ish while a single
token's activations stream through. Speculative decoding buys back that
bandwidth: a cheap draft model proposes `gamma` tokens sequentially, then the
target model scores ALL of them in ONE forward over gamma+1 positions (the
"batched verification step" — same weight reads as a single-token step, just
a taller activation batch). Accepted prefix + one target-chosen token are
emitted per round, so each round costs ~1 target forward but can emit up to
gamma+1 tokens.

The verification forward is exactly a Sq = gamma+1, Sk = context+gamma+1
bottom-right-aligned causal attention — the shape the Phase 1 flash kernel
supports natively (set attn_impl="triton" on the models to use it).

Acceptance rules
----------------
greedy=True: accept draft token i iff it equals the target's argmax at that
position; on first mismatch emit the target argmax instead. Output is
token-identical to plain greedy decoding of the target — draft quality only
affects speed, never content.

greedy=False implements Leviathan et al. rejection sampling: accept d_i with
prob min(1, p_target(d_i)/p_draft(d_i)); on rejection resample from
norm(max(0, p_target - p_draft)). Marginal distribution provably equals
sampling the target directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from train.model import Transformer


class CachedModel:
    """Incremental forward wrapper: feeds only unprocessed tokens, keeps
    per-layer KV caches, supports rollback (truncate) after rejections."""

    def __init__(self, model: Transformer, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device
        self.caches: list[tuple] = [(None, None) for _ in model.blocks]
        self.processed = 0

    @torch.no_grad()
    def logits_for(self, tokens: list[int]) -> torch.Tensor:
        """Feed tokens[processed:]; return logits [n_fed, vocab]."""
        feed = tokens[self.processed:]
        assert feed, "nothing new to feed"
        t = torch.tensor([feed], dtype=torch.long, device=self.device)
        pos = torch.arange(self.processed, self.processed + len(feed),
                           device=self.device)
        x = self.model.embed(t)
        new_caches = []
        for blk, cache in zip(self.model.blocks, self.caches):
            x, c = blk(x, pos=pos, kv_cache=cache)
            new_caches.append(c)
        self.caches = new_caches
        self.processed = len(tokens)
        logits = self.model.lm_head(self.model.final_norm(x))
        return logits[0]

    def truncate(self, n: int):
        """Roll the cache back to the first n tokens."""
        if n >= self.processed:
            return
        self.caches = [(k[:, :, :n], v[:, :, :n]) for k, v in self.caches]
        self.processed = n


@dataclass
class SpecStats:
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    emitted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_round(self) -> float:
        return self.emitted / self.rounds if self.rounds else 0.0


def _residual_sample(p_t: torch.Tensor, p_d: torch.Tensor,
                     gen: torch.Generator) -> int:
    """Sample from norm(max(0, p_target - p_draft)) — the rejection branch."""
    resid = (p_t - p_d).clamp(min=0)
    total = resid.sum()
    if total <= 0:                      # distributions identical
        return int(torch.multinomial(p_t, 1, generator=gen))
    return int(torch.multinomial(resid / total, 1, generator=gen))


@torch.no_grad()
def speculative_generate(target: Transformer, draft: Transformer,
                         prompt: list[int], max_new: int, gamma: int = 4,
                         greedy: bool = True, temperature: float = 1.0,
                         seed: int = 0, device: str = "cpu"):
    """Returns (generated_tokens, SpecStats)."""
    M = CachedModel(target, device)
    D = CachedModel(draft, device)
    gen = torch.Generator().manual_seed(seed)
    tokens = list(prompt)
    n_prompt = len(prompt)
    stats = SpecStats()

    # prefill both up to (but not including) the last prompt token: the loop
    # below always feeds the unprocessed suffix, so just seed the caches
    if n_prompt > 1:
        M.logits_for(tokens[:-1])
        D.logits_for(tokens[:-1])
        M.truncate(n_prompt - 1)
        D.truncate(n_prompt - 1)

    while len(tokens) - n_prompt < max_new:
        # ---- draft phase: propose gamma tokens sequentially ----
        proposals: list[int] = []
        draft_probs: list[torch.Tensor] = []
        for _ in range(gamma):
            dl = D.logits_for(tokens + proposals)[-1]
            if greedy:
                proposals.append(int(dl.argmax()))
            else:
                p = F.softmax(dl.float() / temperature, dim=-1)
                draft_probs.append(p)
                proposals.append(int(torch.multinomial(p, 1, generator=gen)))

        # ---- verification: ONE target forward over gamma+1 positions ----
        L = M.logits_for(tokens + proposals)[-(gamma + 1):]

        emitted: list[int] = []
        if greedy:
            for i, d in enumerate(proposals):
                t_choice = int(L[i].argmax())
                if t_choice == d:
                    emitted.append(d)
                else:
                    emitted.append(t_choice)   # correction token
                    break
            else:
                emitted.append(int(L[gamma].argmax()))   # bonus token
            n_accepted = len(emitted) - 1
        else:
            n_accepted = 0
            for i, d in enumerate(proposals):
                p_t = F.softmax(L[i].float() / temperature, dim=-1)
                p_d = draft_probs[i]
                u = torch.rand((), generator=gen)
                if u < (p_t[d] / p_d[d]).clamp(max=1.0):
                    emitted.append(d)
                    n_accepted += 1
                else:
                    emitted.append(_residual_sample(p_t, p_d, gen))
                    break
            else:
                p_t = F.softmax(L[gamma].float() / temperature, dim=-1)
                emitted.append(int(torch.multinomial(p_t, 1, generator=gen)))

        tokens.extend(emitted)
        stats.rounds += 1
        stats.proposed += gamma
        stats.accepted += n_accepted
        stats.emitted += len(emitted)

        # ---- rollback: caches must hold exactly tokens[:-1] or less ----
        M.truncate(len(tokens) - 1)
        D.truncate(min(D.processed, len(tokens) - 1))

    generated = tokens[n_prompt:][:max_new]
    return generated, stats


@torch.no_grad()
def autoregressive_generate(model: Transformer, prompt: list[int],
                            max_new: int, device: str = "cpu") -> list[int]:
    """Plain cached greedy decode — the fair baseline for speculative."""
    M = CachedModel(model, device)
    tokens = list(prompt)
    for _ in range(max_new):
        tokens.append(int(M.logits_for(tokens)[-1].argmax()))
    return tokens[len(prompt):]
