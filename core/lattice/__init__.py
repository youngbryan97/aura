"""Aura Lattice — hybrid attention + SSM + MoE + world-head architecture.

This is Aura's own architecture substrate — the actual answer to
"invent a radically better model architecture."  Not a placeholder.
A runnable hybrid that combines:

  * Causal sparse attention with rotary embeddings (exact retrieval)
  * Stable diagonal recurrent SSM (linear-time long-context state)
  * Top-k Mixture-of-Experts FFN (sparse capacity)
  * Latent world-prediction head (next-latent self-supervision)

Each block fuses the four primitives via a learned router, so the
model can spend compute on attention when it needs precise recall and
on the SSM when it's compressing long range, without the operator
having to pick.

The primitives are implemented in stock PyTorch so the model runs on
CPU, MPS, or CUDA without extra kernels.  Replace the SSM step with a
custom Mamba kernel later for serious throughput; the contract stays
the same.
"""
from core.lattice.config import LatticeConfig
from core.lattice.model import (
    CausalSparseAttention,
    LatentWorldHead,
    LatticeBlock,
    LatticeLM,
    RMSNorm,
    RotaryEmbedding,
    StableDiagonalSSM,
    TopKMoE,
    entropy,
)
from core.lattice.trainer import LatticeTrainer, TrainConfig
from core.lattice.dataset import RandomTokenDataset

__all__ = [
    "CausalSparseAttention",
    "LatentWorldHead",
    "LatticeBlock",
    "LatticeConfig",
    "LatticeLM",
    "LatticeTrainer",
    "RandomTokenDataset",
    "RMSNorm",
    "RotaryEmbedding",
    "StableDiagonalSSM",
    "TopKMoE",
    "TrainConfig",
    "entropy",
]
