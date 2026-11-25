"""Her sampler: mlx_lm's, which can also say the distribution it draws from.

A draft of her head's is checked against what she would have said. Where all that can be asked of her sampler is a token, a draft
is kept only where the one drawn is the draft, which is as often as that one token is likely: her head drafted its likeliest, and
LIVE 25 November 2026 it was kept 58 times in a hundred, though it was trained to predict her and does. Where the distribution
can be asked for, a draft drawn from her head's own is kept as often as the two distributions overlap, and where it is not, what
she says instead is drawn from what hers has over her head's. What is said follows her distribution exactly either way
(Leviathan, Kalman and Matias 2023; Chen et al. 2023).
"""
from __future__ import annotations

from typing import Any

# mlx is brought in where it is used: the worker's module names this one, and the parent imports that without a GPU of its own.
__all__ = ["HerSampler", "kept_or_instead", "sampler_of"]


class HerSampler:
    """A token for a row of logprobs, and the distribution it is drawn from."""

    def __init__(self, temp: float = 0.0, top_p: float = 0.0, min_p: float = 0.0, top_k: int = 0, min_tokens_to_keep: int = 1) -> None:
        from mlx_lm.sample_utils import make_sampler

        from core.brain.llm.several_tokens_a_pass import Sampling

        self._draw = make_sampler(temp=temp, top_p=top_p, min_p=min_p, min_tokens_to_keep=min_tokens_to_keep, top_k=top_k)
        self._of = Sampling(temperature=temp, top_p=top_p, min_p=min_p, top_k=top_k, min_tokens_to_keep=min_tokens_to_keep)

    def __call__(self, logprobs: Any) -> Any:
        return self._draw(logprobs)

    @property
    def draws(self) -> bool:
        """Whether it draws, and does not take the likeliest."""
        return self._of.temperature > 0.0

    def probabilities(self, logits: Any) -> Any:
        """The distribution it draws from, for rows of logits."""
        return self._of.probabilities(logits)


def sampler_of(make: Any, **controls: Any) -> Any:
    """``make``'s sampler for these controls, as hers where they are all ones hers has."""
    known = {"temp", "top_p", "min_p", "top_k", "min_tokens_to_keep"}
    if set(controls) <= known:
        try:
            return HerSampler(**{name: value for name, value in controls.items() if value is not None})
        # not a failure: controls hers cannot be made of make mlx_lm's, which says so itself.
        except (TypeError, ValueError, ImportError):
            pass
    return make(**controls)


def kept_or_instead(p: Any, q: Any, drafts: Any) -> tuple[Any, Any]:
    """For rows of her distribution ``p``, of her head's ``q``, and a draft drawn from each row of ``q``: what she says, and
    whether it is the draft."""
    import mlx.core as mx

    at = drafts.reshape(-1, 1).astype(mx.int32)
    hers, heads = mx.take_along_axis(p, at, axis=-1)[:, 0], mx.take_along_axis(q, at, axis=-1)[:, 0]
    keep = mx.random.uniform(shape=hers.shape) * heads < hers
    over = mx.maximum(p - q, 0.0)
    # Where hers has nothing over her head's the two are one, and the draft is kept.
    over = mx.where(mx.sum(over, axis=-1, keepdims=True) > 0, over, p)
    instead = mx.random.categorical(mx.log(over), axis=-1)
    return mx.where(keep, at[:, 0], instead.astype(mx.int32)), keep
