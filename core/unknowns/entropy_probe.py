"""EmbeddingEntropyProbe — find LatticeLM blind spots in embedding space.

Optimization: starting from a known input's token embeddings, walk
gradient ascent on output entropy in a small ε-ball.  Resulting
"adversarial embeddings" are the model's continuous-input blind
spots — places where its predictions are maximally uncertain.

These are not token-valid prompts; they're model-internal probes.
Use them to stress-test routing, hidden classifiers, decoding
heads, or to seed F9's curriculum on cases the model can't answer.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from core.lattice.model import LatticeLM


class EmbeddingEntropyProbe:
    def __init__(
        self,
        *,
        epsilon: float = 0.05,
        steps: int = 6,
        step_size: float = 0.01,
    ):
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if step_size <= 0:
            raise ValueError("step_size must be > 0")
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.step_size = float(step_size)

    def generate(
        self,
        model: LatticeLM,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, seq]")
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        was_training = model.training
        model.eval()
        try:
            base = model.embed(input_ids).detach()
            adv = base.clone().requires_grad_(True)
            for _ in range(self.steps):
                x = model.dropout(adv)
                for block in model.blocks:
                    x, _ = block(x)
                logits = model.head(model.norm(x))[:, -1]
                probs = F.softmax(logits.float(), dim=-1)
                ent = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean()
                grad = torch.autograd.grad(ent, adv, retain_graph=False, create_graph=False)[0]
                with torch.no_grad():
                    adv = adv + self.step_size * grad.sign()
                    delta = torch.clamp(adv - base, min=-self.epsilon, max=self.epsilon)
                    adv = (base + delta).detach().requires_grad_(True)
            return adv.detach()
        finally:
            if was_training:
                model.train()

    def measure_entropy(self, model: LatticeLM, embeddings: torch.Tensor) -> float:
        """Compute the mean per-position entropy at the final token."""
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                x = model.dropout(embeddings)
                for block in model.blocks:
                    x, _ = block(x)
                logits = model.head(model.norm(x))[:, -1]
                probs = F.softmax(logits.float(), dim=-1)
                ent = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean()
                return float(ent)
        finally:
            if was_training:
                model.train()
