"""Decoder-only MoE transformer, written from scratch.

Architecture: pre-norm blocks with RMSNorm, rotary position embeddings
(GPT-NeoX style rotate-half), causal self-attention, and SwiGLU MLPs. Layers
can be dense or Mixture-of-Experts (top-k token routing, Phase 2).

Every compute-heavy op has a reference PyTorch path and a custom-kernel path,
selected by `ModelConfig.attn_impl` / `ModelConfig.routing_impl`. The
reference paths double as the ground truth for kernel correctness tests.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class Rotary(nn.Module):
    """Precomputed RoPE cos/sin tables, rotate-half application."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)                    # [S, D/2]
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # x: [B, H, S, D]; pos: [S] absolute positions (supports KV-cache decode)
        cos = self.cos[pos].to(x.dtype)                     # [S, D/2]
        sin = self.sin[pos].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def naive_attention(q, k, v, causal: bool = True, scale: float | None = None):
    """Reference attention: materializes the full [Sq, Sk] score matrix.

    Causal masking is bottom-right aligned (query i sees keys j <= i + Sk-Sq),
    which reduces to standard causal when Sq == Sk and handles the
    decode-with-cache case where Sq < Sk.
    """
    scale = scale if scale is not None else q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if causal:
        sq, sk = q.shape[-2], k.shape[-2]
        qi = torch.arange(sq, device=q.device)[:, None] + (sk - sq)
        kj = torch.arange(sk, device=q.device)[None, :]
        scores = scores.masked_fill(kj > qi, float("-inf"))
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, hd = cfg.d_model, cfg.head_dim
        self.n_heads = cfg.n_heads
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.rotary = Rotary(hd, cfg.max_seq_len, cfg.rope_theta)

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None,
                kv_cache: tuple | None = None):
        B, S, D = x.shape
        H, hd = self.n_heads, self.cfg.head_dim
        if pos is None:
            pos = torch.arange(S, device=x.device)

        q = self.wq(x).view(B, S, H, hd).transpose(1, 2)    # [B, H, S, hd]
        k = self.wk(x).view(B, S, H, hd).transpose(1, 2)
        v = self.wv(x).view(B, S, H, hd).transpose(1, 2)
        q = self.rotary(q, pos)
        k = self.rotary(k, pos)

        new_cache = None
        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)

        impl = self.cfg.attn_impl
        if impl == "triton":
            if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad
                                            or v.requires_grad):
                from kernels.autograd_ops import flash_attention_train
                o = flash_attention_train(q.contiguous(), k.contiguous(),
                                          v.contiguous(), causal=True)
            else:
                from kernels.attention_fwd import flash_attention
                o = flash_attention(q.contiguous(), k.contiguous(),
                                    v.contiguous(), causal=True)
        elif impl == "sdpa":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=(S == k.shape[2]))
        else:
            o = naive_attention(q, k, v, causal=True)

        o = o.transpose(1, 2).reshape(B, S, D)
        out = self.wo(o)
        return (out, new_cache) if kv_cache is not None else out


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoE(nn.Module):
    """Top-k token-choice MoE layer.

    The router picks top_k experts per token; expert outputs are combined
    weighted by the (renormalized) router softmax. The naive path loops over
    experts with index_select / index_add_ — that loop over gathers is exactly
    the dispatch/combine bottleneck the Phase 2 Triton kernel replaces.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.ffn_hidden) for _ in range(cfg.n_experts)]
        )
        self.aux_loss = torch.zeros(())    # set on each forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        flat = x.reshape(-1, D)                                  # [T, D]
        logits = self.router(flat)                               # [T, E]
        probs = F.softmax(logits.float(), dim=-1)
        weights, experts = torch.topk(probs, self.cfg.top_k, dim=-1)  # [T, k]
        weights = weights / weights.sum(dim=-1, keepdim=True)
        self.aux_loss = self._load_balancing_loss(probs, experts)

        if self.cfg.routing_impl == "triton":
            if torch.is_grad_enabled() and flat.requires_grad:
                from kernels.autograd_ops import moe_dispatch_combine_train
                out = moe_dispatch_combine_train(flat, weights.to(flat.dtype),
                                                 experts, self.experts)
            else:
                from kernels.moe_routing import moe_dispatch_combine
                out = moe_dispatch_combine(flat, weights.to(flat.dtype),
                                           experts, self.experts)
        else:
            out = self._naive_route(flat, weights.to(flat.dtype), experts)
        return out.reshape(B, S, D)

    def _naive_route(self, flat, weights, experts):
        out = torch.zeros_like(flat)
        for e in range(self.cfg.n_experts):
            tok, slot = torch.where(experts == e)                # tokens routed to e
            if tok.numel() == 0:
                continue
            expert_in = flat.index_select(0, tok)
            expert_out = self.experts[e](expert_in)
            out.index_add_(0, tok, expert_out * weights[tok, slot, None])
        return out

    def _load_balancing_loss(self, probs, experts):
        # Switch-transformer style: E * sum_e (frac_tokens_e * mean_prob_e)
        E = self.cfg.n_experts
        counts = torch.zeros(E, device=probs.device, dtype=probs.dtype)
        counts.scatter_add_(0, experts.reshape(-1),
                            torch.ones_like(experts.reshape(-1), dtype=probs.dtype))
        frac = counts / max(1, experts.numel())
        mean_prob = probs.mean(dim=0)
        return E * torch.sum(frac * mean_prob)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.is_moe = cfg.is_moe_layer(layer_idx)
        self.mlp = MoE(cfg) if self.is_moe else SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x, pos=None, kv_cache=None):
        if kv_cache is not None:
            attn_out, new_cache = self.attn(self.attn_norm(x), pos, kv_cache)
            x = x + attn_out
            x = x + self.mlp(self.mlp_norm(x))
            return x, new_cache
        x = x + self.attn(self.attn_norm(x), pos)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.final_norm(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               targets.reshape(-1))
        aux = self.moe_aux_loss()
        return logits, loss, aux

    def moe_aux_loss(self) -> torch.Tensor:
        losses = [b.mlp.aux_loss for b in self.blocks if b.is_moe]
        if not losses:
            return torch.zeros((), device=self.embed.weight.device)
        return torch.stack(losses).sum()

    @torch.no_grad()
    def generate_naive(self, tokens: torch.Tensor, max_new: int,
                       temperature: float = 0.0) -> torch.Tensor:
        """Baseline generation: full re-forward each step, no KV cache."""
        for _ in range(max_new):
            logits = self(tokens[:, -self.cfg.max_seq_len:])[:, -1]
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            tokens = torch.cat([tokens, nxt], dim=1)
        return tokens

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
