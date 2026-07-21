"""Model / training configuration."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    max_seq_len: int = 512
    ffn_mult: float = 4.0          # SwiGLU hidden = 2/3 * ffn_mult * d_model
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    # Mixture of Experts. n_experts == 0 means every layer is dense.
    n_experts: int = 0
    top_k: int = 2
    moe_every: int = 2             # every moe_every-th layer is MoE (1 = all)

    # Implementation switches (reference PyTorch vs custom Triton kernels).
    attn_impl: str = "naive"       # "naive" | "sdpa" | "triton"
    routing_impl: str = "naive"    # "naive" | "triton"

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @property
    def ffn_hidden(self) -> int:
        h = int(2 * self.ffn_mult * self.d_model / 3)
        return ((h + 31) // 32) * 32  # round up to multiple of 32

    def is_moe_layer(self, layer_idx: int) -> bool:
        if self.n_experts == 0:
            return False
        return (layer_idx % self.moe_every) == (self.moe_every - 1)


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    batch_size: int = 8
    seq_len: int = 128
    steps: int = 200
    warmup_steps: int = 20
    log_every: int = 20
    seed: int = 1234
    device: str = "auto"           # "auto" | "cpu" | "cuda"
    aux_loss_coef: float = 0.01    # MoE load-balancing loss weight


def tiny_config(**overrides) -> ModelConfig:
    """Small config that runs quickly on CPU; used by tests."""
    defaults = dict(vocab_size=128, d_model=64, n_layers=2, n_heads=2, max_seq_len=128)
    defaults.update(overrides)
    return ModelConfig(**defaults)
