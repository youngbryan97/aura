"""Can something the prompt never said still reach the words?

This is the strongest experiment the architecture argument suggests. Hold the
prompt fixed, put a fact only into z_Aura, and ask whether the reply carries
it. If a reader can tell which state produced which reply — from the text
alone, with the prompt identical in both arms — then information travelled
through the endogenous pathway rather than through the context window.

What is being tested is recoverability, not fluency. The reader here is
deliberately weak: token counts and a linear classifier. A weak reader that
succeeds is a strong result, because it means the signal is coarse enough to
find without a model doing the finding. A model scoring its own outputs would
be a confound rather than a measurement.

The null is label permutation. Shuffle which reply belongs to which arm, refit
the reader, and score again. If the true labelling scores no better than the
shuffles, the arms are indistinguishable and the honest answer is that nothing
crossed — which is a result, and gets reported as one.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.SubstrateOnlyChannel")

#: Minimum replies per arm. Below this the classifier is fitting noise and the
#: permutation test has nothing to resolve.
MIN_PER_ARM = 8

#: Vocabulary the reader is allowed. Capped so a rare word appearing in one
#: reply cannot become the whole classifier.
MAX_READER_VOCAB = 400

#: Shuffles of the arm labels. The p-value's resolution.
PERMUTATIONS = 500

_WORD = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class Recoverability:
    """Whether a reader could tell the arms apart, and how sure that is."""

    accuracy: float
    null_mean: float
    null_max: float
    p_value: float
    permutations: int
    n_per_arm: tuple[int, int]
    reader_vocab: int
    top_words_low: tuple[str, ...] = ()
    top_words_high: tuple[str, ...] = ()

    @property
    def recovered(self) -> bool:
        return self.accuracy > 0.5 and self.p_value <= 0.05

    def as_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 6),
            "null_mean": round(self.null_mean, 6),
            "null_max": round(self.null_max, 6),
            "p_value": round(self.p_value, 6),
            "permutations": self.permutations,
            "n_per_arm": list(self.n_per_arm),
            "reader_vocab": self.reader_vocab,
            "top_words_low": list(self.top_words_low),
            "top_words_high": list(self.top_words_high),
            "recovered": self.recovered,
            "what_this_means": (
                "the arms are distinguishable from the text alone, with the "
                "prompt held identical, so the state reached the words"
                if self.recovered
                else "the arms are indistinguishable; nothing crossed by this route"
            ),
        }


def _counts(texts: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Bag of words over the pooled vocabulary, capped by document frequency."""
    tokenised = [_WORD.findall(str(text or "").lower()) for text in texts]
    frequency: dict[str, int] = {}
    for words in tokenised:
        for word in set(words):
            frequency[word] = frequency.get(word, 0) + 1
    vocabulary = [
        word
        for word, _ in sorted(frequency.items(), key=lambda kv: (-kv[1], kv[0]))
        if frequency[word] >= 2
    ][:MAX_READER_VOCAB]
    index = {word: i for i, word in enumerate(vocabulary)}
    matrix = np.zeros((len(texts), len(vocabulary)), dtype=np.float64)
    for row, words in enumerate(tokenised):
        for word in words:
            column = index.get(word)
            if column is not None:
                matrix[row, column] += 1.0
    # Length-normalise: a longer reply must not be easier to classify because
    # it is longer.
    lengths = matrix.sum(axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return matrix / lengths, vocabulary


def _fit_reader(
    x: np.ndarray, y: np.ndarray, *, epochs: int = 200, learning_rate: float = 1.0
) -> np.ndarray:
    """Logistic regression with a bias column, trained by plain gradient descent."""
    design = np.hstack([x, np.ones((x.shape[0], 1))])
    weights = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(epochs):
        prediction = 1.0 / (1.0 + np.exp(-design @ weights))
        gradient = design.T @ (prediction - y) / max(1, len(y)) + 0.01 * weights
        weights -= learning_rate * gradient
    return weights


def _leave_one_out_accuracy(x: np.ndarray, y: np.ndarray) -> float:
    """Every reply is classified by a reader that never saw it.

    Leave-one-out rather than a single split: the sample here is dozens of
    replies, not thousands, and a single held-out fifth of it would put the
    whole measurement at the mercy of which five landed there.
    """
    correct = 0
    for held in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[held] = False
        if len(set(y[mask].tolist())) < 2:
            continue
        weights = _fit_reader(x[mask], y[mask])
        score = float(np.hstack([x[held], [1.0]]) @ weights)
        correct += int((score > 0.0) == bool(y[held]))
    return correct / max(1, len(y))


def measure_recoverability(
    low_texts: Sequence[str],
    high_texts: Sequence[str],
    *,
    permutations: int = PERMUTATIONS,
    seed: int = 17,
) -> Recoverability | None:
    """Can a weak reader tell which arm each reply came from?"""
    if len(low_texts) < MIN_PER_ARM or len(high_texts) < MIN_PER_ARM:
        logger.warning(
            "Refusing to measure recoverability on %d/%d replies (minimum %d each).",
            len(low_texts),
            len(high_texts),
            MIN_PER_ARM,
        )
        return None
    texts = list(low_texts) + list(high_texts)
    labels = np.concatenate(
        [np.zeros(len(low_texts)), np.ones(len(high_texts))]
    ).astype(np.float64)
    x, vocabulary = _counts(texts)
    if x.shape[1] < 2:
        logger.warning("The replies share fewer than two words; no reader is possible.")
        return None

    accuracy = _leave_one_out_accuracy(x, labels)

    rng = np.random.default_rng(seed)
    nulls: list[float] = []
    for _ in range(max(0, int(permutations))):
        shuffled = rng.permutation(labels)
        # The permutation null uses the cheap in-sample reader: leave-one-out
        # five hundred times over would cost more than the experiment that
        # produced the replies. The asymmetry is deliberate and runs one way.
        # In-sample accuracy on shuffled labels is biased UP, and leave-one-out
        # accuracy on noise is biased DOWN — measured at 0.18 on arms built to
        # be identical, because a reader that memorised the held-out reply's
        # neighbours anti-predicts it. Both biases raise the bar the true
        # labelling has to clear, so this test errs towards reporting that
        # nothing crossed.
        weights = _fit_reader(x, shuffled)
        scores = np.hstack([x, np.ones((x.shape[0], 1))]) @ weights
        nulls.append(float(np.mean((scores > 0.0) == (shuffled > 0.5))))

    array = np.asarray(nulls, dtype=np.float64)
    p_value = (
        float((np.sum(array >= accuracy) + 1) / (array.size + 1)) if array.size else 1.0
    )
    weights = _fit_reader(x, labels)
    order = np.argsort(weights[:-1])
    top = min(6, len(vocabulary))
    return Recoverability(
        accuracy=accuracy,
        null_mean=float(np.mean(array)) if array.size else 0.0,
        null_max=float(np.max(array)) if array.size else 0.0,
        p_value=p_value,
        permutations=int(array.size),
        n_per_arm=(len(low_texts), len(high_texts)),
        reader_vocab=len(vocabulary),
        top_words_low=tuple(vocabulary[i] for i in order[:top]),
        top_words_high=tuple(vocabulary[i] for i in order[::-1][:top]),
    )


async def run_substrate_only_channel(
    client: Any,
    prompt: str,
    state: Any,
    feature: str,
    *,
    low: float = 0.05,
    high: float = 0.95,
    replies_per_arm: int = 12,
    temp: float = 0.7,
    max_tokens: int = 120,
    permutations: int = PERMUTATIONS,
) -> dict[str, Any]:
    """The live experiment: one prompt, two states, and a reader.

    Sampling temperature is deliberately not zero. At temperature zero each arm
    produces one reply repeated, and a reader distinguishing two constants
    measures nothing about how reliably the state reaches language.
    """
    from core.brain.llm.endogenous_intervention import run_text_arm

    arms: dict[str, list[str]] = {"low": [], "high": []}
    receipts: list[dict[str, Any]] = []
    for name, value in (("low", low), ("high", high)):
        for _ in range(max(1, int(replies_per_arm))):
            arm = await run_text_arm(
                client,
                prompt,
                state.do(**{feature: value}),
                name=f"{feature}={value}",
                max_tokens=max_tokens,
                temp=temp,
            )
            arms[name].append(arm.text)
            receipts.append(arm.receipt)

    applied = sum(1 for r in receipts if r.get("applied"))
    result = measure_recoverability(
        arms["low"], arms["high"], permutations=permutations
    )
    return {
        "prompt": prompt,
        "feature": feature,
        "low": low,
        "high": high,
        "replies_per_arm": replies_per_arm,
        "bias_applied_generations": applied,
        "bias_applied_share": round(applied / max(1, len(receipts)), 4),
        "arms": arms,
        "recoverability": result.as_dict() if result else None,
        "verdict": (
            "not_measured"
            if result is None
            else ("recovered" if result.recovered else "nothing_crossed")
        ),
        # Without the bias attached, the two arms differ in nothing the model
        # can see, so a positive result would be sampling noise.
        "arms_were_actually_differentiated": applied == len(receipts),
    }


def state_entropy_bound(n_dimensions: int) -> float:
    """How much a reader could recover at best, in bits, if z were read exactly.

    Stated because the ceiling is low and worth saying out loud: a reader
    recovering one binary contrast from a state of this width has recovered one
    bit, not a memory. Nothing here supports the claim that an episodic record
    crossed.
    """
    return float(max(0.0, math.log2(max(1, int(n_dimensions)))))


__all__ = [
    "MAX_READER_VOCAB",
    "MIN_PER_ARM",
    "PERMUTATIONS",
    "Recoverability",
    "measure_recoverability",
    "run_substrate_only_channel",
    "state_entropy_bound",
]
