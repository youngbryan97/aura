"""LatticeConfig — typed configuration for the Aura Lattice model."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LatticeConfig:
    """Hyperparameters for ``LatticeLM``.

    Defaults are tuned for a small CPU-runnable smoke configuration so
    tests can boot a real model in milliseconds.  Production training
    overrides these per checkpoint.
    """

    vocab_size: int = 8192
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_state: int = 48
    n_experts: int = 8
    top_k: int = 2
    d_ff_mult: int = 4
    dropout: float = 0.05
    max_seq_len: int = 2048
    attention_window: int = 256
    rope_base: float = 10_000.0
    aux_moe_weight: float = 0.01
    aux_world_weight: float = 0.05
    residual_scale: float = 1.0
    eps: float = 1e-5
    tie_embeddings: bool = True

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
            )
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError(
                f"top_k={self.top_k} must satisfy 1 <= top_k <= n_experts={self.n_experts}"
            )
        if self.attention_window < 1:
            raise ValueError(f"attention_window={self.attention_window} must be >= 1")
        if self.max_seq_len < 2:
            raise ValueError(f"max_seq_len={self.max_seq_len} must be >= 2")
        if self.d_state < 1:
            raise ValueError(f"d_state={self.d_state} must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout={self.dropout} must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
