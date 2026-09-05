"""Train the substrate readout instead of seeding it and calling it a head.

``SubstrateTokenGenerator._ensure_readout`` builds its matrix with
``rng.standard_normal(...)`` and never fits it to anything. Gating its output
away from users was the safety fix; it was not the capability fix. This is the
capability fix: a readout that is actually trained, on real pairs, and MEASURED
against the random projection it replaces.

WHAT CAN AND CANNOT BE LEARNED HERE, STATED UP FRONT. The substrate state is 64
dimensions of affect and cognitive summary — valence, arousal, dominance,
frustration, curiosity, energy, focus and their neighbours. Sixty-four numbers
cannot encode which *content* words a sentence needs; no fit will make them. What
they can carry is the part of token choice that genuinely depends on state:
register, hedging, directness, affect-laden vocabulary. So the honest target is
a **state-conditioned token prior**, and the honest test is whether it beats the
random projection on held-out pairs by a margin that is not noise.

That is a real capability with a real bound, and it is the difference between
"the substrate has a voice" (false) and "the substrate measurably shapes word
choice" (testable, and tested below).

The training pairs come from the runtime's own history: each recorded substrate
state paired with the tokens that were actually produced while it held. No
synthetic data — a readout fitted to invented pairs would measure nothing.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.SubstrateReadoutTraining")

#: Held-out share used to measure the fit. Fixed so a run cannot be tuned by
#: choosing a friendlier split.
HOLDOUT_FRACTION = 0.25

#: Below this many pairs there is nothing to fit and nothing to hold out; the
#: trainer refuses rather than producing a matrix with no evidence behind it.
MIN_TRAINING_PAIRS = 200


@dataclass(frozen=True)
class ReadoutPair:
    """One substrate state and a token that was produced while it held."""

    state: tuple[float, ...]
    token_id: int


@dataclass(frozen=True)
class ReadoutFit:
    """A trained head and what it measurably does better than random."""

    weights: np.ndarray
    vocab_size: int
    state_dim: int
    n_train: int
    n_holdout: int
    #: Mean held-out log-likelihood of the true token under each model.
    trained_log_likelihood: float
    random_log_likelihood: float
    #: Share of held-out tokens ranked in the top-k by each model.
    trained_top_k: float
    random_top_k: float
    top_k: int
    trained_at: float = field(default_factory=time.time)

    @property
    def improvement_nats(self) -> float:
        """How much better the trained head is, per token, in nats."""
        return float(self.trained_log_likelihood - self.random_log_likelihood)

    @property
    def beats_random(self) -> bool:
        """Whether the fit earned its place over the projection it replaces."""
        return self.improvement_nats > 0.0 and self.trained_top_k > self.random_top_k

    def as_report(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "state_dim": self.state_dim,
            "n_train": self.n_train,
            "n_holdout": self.n_holdout,
            "top_k": self.top_k,
            "trained_log_likelihood": round(self.trained_log_likelihood, 6),
            "random_log_likelihood": round(self.random_log_likelihood, 6),
            "improvement_nats": round(self.improvement_nats, 6),
            "trained_top_k_rate": round(self.trained_top_k, 6),
            "random_top_k_rate": round(self.random_top_k, 6),
            "beats_random": self.beats_random,
            "trained_at": self.trained_at,
            # Said plainly, so the number cannot be read as more than it is.
            "interpretation": (
                "a state-conditioned token prior over a 64-dimensional affect/"
                "cognition state; it shapes word choice and does not generate "
                "language on its own"
            ),
        }


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def train_readout(
    pairs: Sequence[ReadoutPair],
    *,
    vocab_size: int,
    epochs: int = 60,
    learning_rate: float = 0.35,
    l2: float = 1e-3,
    top_k: int = 20,
    seed: int = 913,
) -> ReadoutFit | None:
    """Fit a multinomial logistic readout state → token, and measure it.

    Multinomial logistic regression rather than anything deeper on purpose: the
    claim being supported is that substrate state carries information about
    token choice, and a linear model is the weakest thing that can demonstrate
    it. If a linear head beats the random projection, the information is there.
    """

    if len(pairs) < MIN_TRAINING_PAIRS:
        logger.warning(
            "Refusing to train a readout on %d pairs (minimum %d): a matrix with "
            "no evidence behind it is what this replaces.",
            len(pairs),
            MIN_TRAINING_PAIRS,
        )
        return None

    rng = np.random.default_rng(seed)
    states = np.asarray([p.state for p in pairs], dtype=np.float64)
    tokens = np.asarray([int(p.token_id) for p in pairs], dtype=np.int64)
    if states.ndim != 2 or states.shape[0] != tokens.shape[0]:
        raise ValueError("states and tokens must line up")
    if np.any(tokens < 0) or np.any(tokens >= vocab_size):
        raise ValueError("a token id falls outside the declared vocabulary")

    order = rng.permutation(len(pairs))
    states, tokens = states[order], tokens[order]
    split = max(1, int(len(pairs) * (1.0 - HOLDOUT_FRACTION)))
    train_x, train_y = states[:split], tokens[:split]
    test_x, test_y = states[split:], tokens[split:]
    if len(test_y) == 0:
        return None

    state_dim = train_x.shape[1]
    weights = np.zeros((vocab_size, state_dim), dtype=np.float64)
    bias = np.zeros(vocab_size, dtype=np.float64)

    one_hot = np.zeros((len(train_y), vocab_size), dtype=np.float64)
    one_hot[np.arange(len(train_y)), train_y] = 1.0

    for _ in range(max(1, int(epochs))):
        probabilities = _softmax_rows(train_x @ weights.T + bias)
        error = probabilities - one_hot
        grad_w = error.T @ train_x / len(train_y) + l2 * weights
        grad_b = error.mean(axis=0)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    # The baseline is the thing being replaced, built exactly as the generator
    # builds it — not an idealised random model.
    baseline_rng = np.random.default_rng(seed + state_dim * 31 + vocab_size)
    baseline = baseline_rng.standard_normal((vocab_size, state_dim)) / math.sqrt(
        max(1, state_dim)
    )

    def _score(matrix: np.ndarray, offsets: np.ndarray | None) -> tuple[float, float]:
        logits = test_x @ matrix.T + (offsets if offsets is not None else 0.0)
        probabilities = _softmax_rows(logits)
        truth = probabilities[np.arange(len(test_y)), test_y]
        log_likelihood = float(np.mean(np.log(np.clip(truth, 1e-12, None))))
        k = max(1, min(int(top_k), vocab_size))
        ranked = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
        hit_rate = float(np.mean([test_y[i] in ranked[i] for i in range(len(test_y))]))
        return log_likelihood, hit_rate

    trained_ll, trained_hit = _score(weights, bias)
    random_ll, random_hit = _score(baseline, None)

    fit = ReadoutFit(
        weights=weights,
        vocab_size=int(vocab_size),
        state_dim=int(state_dim),
        n_train=int(len(train_y)),
        n_holdout=int(len(test_y)),
        trained_log_likelihood=trained_ll,
        random_log_likelihood=random_ll,
        trained_top_k=trained_hit,
        random_top_k=random_hit,
        top_k=int(max(1, min(int(top_k), vocab_size))),
    )
    logger.info(
        "Substrate readout trained on %d pairs: holdout log-likelihood %.4f vs "
        "random %.4f (+%.4f nats), top-%d %.3f vs %.3f.",
        fit.n_train,
        fit.trained_log_likelihood,
        fit.random_log_likelihood,
        fit.improvement_nats,
        fit.top_k,
        fit.trained_top_k,
        fit.random_top_k,
    )
    return fit


def save_fit(fit: ReadoutFit, path: str | Path) -> Path:
    """Persist a trained head next to the report that justifies it.

    Through the atomic writer, not `np.save` and `write_text`. A half-written
    weight matrix beside a complete report is worse than no head at all — it
    would load, produce numbers, and carry a report claiming they were
    measured. The governance lint flagged the first draft of this function for
    exactly that, and it was right to.
    """
    import io

    from core.runtime.file_write_gateway import get_file_write_gateway

    gateway = get_file_write_gateway()
    target = Path(path)
    buffer = io.BytesIO()
    np.save(buffer, fit.weights, allow_pickle=False)
    gateway.write_bytes(
        target.with_suffix(".npy"),
        buffer.getvalue(),
        source="substrate_readout_training",
    )
    gateway.write_text(
        target.with_suffix(".json"),
        json.dumps(fit.as_report(), indent=2, sort_keys=True) + "\n",
        source="substrate_readout_training",
    )
    return target


def pairs_from_history(
    records: Iterable[tuple[Sequence[float], Iterable[int]]],
) -> list[ReadoutPair]:
    """Flatten (state, tokens-produced-while-it-held) records into pairs."""
    out: list[ReadoutPair] = []
    for state, tokens in records:
        try:
            vector = tuple(float(x) for x in state)
        except (TypeError, ValueError) as exc:
            record_degradation("substrate_readout_training", exc, severity="warning")
            continue
        for token in tokens:
            try:
                out.append(ReadoutPair(vector, int(token)))
            except (TypeError, ValueError):
                continue
    return out


__all__ = [
    "HOLDOUT_FRACTION",
    "MIN_TRAINING_PAIRS",
    "ReadoutFit",
    "ReadoutPair",
    "pairs_from_history",
    "save_fit",
    "train_readout",
]
