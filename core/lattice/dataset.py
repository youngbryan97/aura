"""Synthetic token datasets for plumbing tests of LatticeLM.

Real training plugs in a tokenized corpus.  These datasets exist so
the unit + stress tests can boot a full forward+backward pass on
deterministic data without an external dependency.
"""
from __future__ import annotations

import random
from typing import Dict

import torch
from torch.utils.data import Dataset


class RandomTokenDataset(Dataset):
    """Reproducible synthetic next-token dataset.

    Each sequence has a hidden additive period so a model can actually
    learn to lower its loss — random uniform tokens would leave the
    loss flat and turn the trainer into a no-op.
    """

    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        vocab_size: int,
        seed: int = 0,
        noise_p: float = 0.08,
    ):
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if not 0.0 <= noise_p < 1.0:
            raise ValueError("noise_p must be in [0, 1)")
        self.n_samples = int(n_samples)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.noise_p = float(noise_p)
        rng = random.Random(seed)
        self.seeds = [rng.randrange(2**31) for _ in range(self.n_samples)]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self.seeds[idx])
        start = rng.randrange(self.vocab_size)
        step = rng.randrange(1, 17)
        ids = []
        val = start
        for t in range(self.seq_len + 1):
            if rng.random() < self.noise_p:
                val = rng.randrange(self.vocab_size)
            ids.append(val % self.vocab_size)
            val += step + (t % 5)
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}
