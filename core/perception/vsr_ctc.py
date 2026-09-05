"""core/perception/vsr_ctc.py
──────────────────────────
CTC decoding for open-vocabulary visual speech recognition.

The decoder is the part that turns a frame-by-frame character
probability sequence (the ONNX model's output) into an actual
transcript. Two decoders, both exact and independently tested:

- greedy: argmax per frame, collapse repeats, drop blanks — the CTC
  baseline, O(T·V).
- beam search: prefix-beam search that correctly merges paths differing
  only by blanks/repeats, with an optional character-level language
  model prior. This is what real VSR systems use; it recovers
  transcripts greedy decode gets wrong.

Vocabulary convention: index 0 is the CTC blank; indices 1..N are
characters given by ``alphabet``.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

BLANK = 0


@dataclass(frozen=True)
class Vocabulary:
    """CTC label alphabet. Index 0 is blank; the rest map to characters."""

    alphabet: str  # characters for indices 1..len(alphabet)

    def __post_init__(self) -> None:
        if not self.alphabet:
            raise ValueError("alphabet must be non-empty")
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("alphabet has duplicate characters")

    @property
    def size(self) -> int:
        return len(self.alphabet) + 1  # +1 for blank

    def char(self, index: int) -> str:
        if index == BLANK:
            return ""
        return self.alphabet[index - 1]

    def decode_indices(self, indices: Sequence[int]) -> str:
        return "".join(self.char(int(i)) for i in indices)


DEFAULT_ALPHABET = " abcdefghijklmnopqrstuvwxyz'"


def default_vocabulary() -> Vocabulary:
    return Vocabulary(DEFAULT_ALPHABET)


def _as_log_probs(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last axis."""
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("logits must be (frames, vocab)")
    shifted = arr - arr.max(axis=1, keepdims=True)
    log_denom = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    return shifted - log_denom


def greedy_decode(logits: np.ndarray, vocab: Vocabulary) -> str:
    """Argmax path, collapse repeats, drop blanks."""
    if logits.shape[1] != vocab.size:
        raise ValueError(f"logits width {logits.shape[1]} != vocab size {vocab.size}")
    path = np.argmax(logits, axis=1)
    collapsed: list[int] = []
    previous = -1
    for index in path:
        index = int(index)
        if index != previous and index != BLANK:
            collapsed.append(index)
        previous = index
    return vocab.decode_indices(collapsed)


def _logsumexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi, lo = (a, b) if a > b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


def beam_search_decode(
    logits: np.ndarray,
    vocab: Vocabulary,
    *,
    beam_width: int = 12,
    lm: Callable[[str, str], float] | None = None,
    lm_weight: float = 0.3,
) -> tuple[str, float]:
    """Prefix-beam search. Returns (transcript, normalized_confidence).

    ``lm(prefix, next_char)`` returns a log-probability prior over the
    next character; it steers decoding without overriding the acoustic
    model. Exact CTC prefix merging: beams differing only in blank/repeat
    padding collapse into one prefix with summed probability."""
    if logits.shape[1] != vocab.size:
        raise ValueError(f"logits width {logits.shape[1]} != vocab size {vocab.size}")
    log_probs = _as_log_probs(logits)
    frames = log_probs.shape[0]

    # For each prefix track (p_blank, p_non_blank) in log space: the
    # probability the prefix ends in a blank vs. a real character.
    beams: dict[str, tuple[float, float]] = {"": (0.0, -math.inf)}

    for t in range(frames):
        next_beams: dict[str, tuple[float, float]] = defaultdict(
            lambda: (-math.inf, -math.inf))
        # Prune to the most probable prefixes each frame.
        pruned = sorted(
            beams.items(),
            key=lambda kv: _logsumexp(kv[1][0], kv[1][1]),
            reverse=True,
        )[:beam_width]
        for prefix, (p_blank, p_non_blank) in pruned:
            total = _logsumexp(p_blank, p_non_blank)
            # Extend with blank: prefix unchanged, accumulate into p_blank.
            blank_lp = log_probs[t, BLANK]
            nb, nnb = next_beams[prefix]
            next_beams[prefix] = (_logsumexp(nb, total + blank_lp), nnb)
            # Extend with each real character.
            for index in range(1, vocab.size):
                char = vocab.char(index)
                char_lp = float(log_probs[t, index])
                last = prefix[-1] if prefix else ""
                if char == last:
                    # Same char: only the blank-ending mass can extend it
                    # (repeat via blank); non-blank mass keeps same prefix.
                    new_prefix = prefix + char
                    nb2, nnb2 = next_beams[new_prefix]
                    next_beams[new_prefix] = (
                        nb2, _logsumexp(nnb2, p_blank + char_lp))
                    same_nb, same_nnb = next_beams[prefix]
                    next_beams[prefix] = (
                        same_nb, _logsumexp(same_nnb, p_non_blank + char_lp))
                else:
                    new_prefix = prefix + char
                    prior = 0.0 if lm is None else lm_weight * lm(prefix, char)
                    nb2, nnb2 = next_beams[new_prefix]
                    next_beams[new_prefix] = (
                        nb2, _logsumexp(nnb2, total + char_lp + prior))
        beams = dict(next_beams)

    scored = sorted(
        beams.items(),
        key=lambda kv: _logsumexp(kv[1][0], kv[1][1]),
        reverse=True,
    )
    best_prefix, (p_blank, p_non_blank) = scored[0]
    best_score = _logsumexp(p_blank, p_non_blank)
    # Confidence: exp of the per-frame-normalized best log-prob, bounded.
    confidence = float(math.exp(best_score / max(1, frames)))
    return best_prefix, max(0.0, min(1.0, confidence))
