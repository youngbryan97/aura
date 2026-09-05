"""core/creativity/novelty_value.py — making something new that is still legible.

Novelty is easy to get and worthless on its own. A random byte string is
maximally unlike everything ever made and nobody wants it. Legibility is also
easy and also worthless on its own: a copy of what is already there is
perfectly intelligible and adds nothing. Whatever creativity is, it lives
where those two are both high, and the interesting fact is that they can both
be high at once.

They can because they are measured against different things. Something can be
unlike every individual thing in the corpus while being made entirely of parts
the corpus explains. That is the ordinary description of a new piece of work —
recognisably in a tradition, not a copy of anything in it — and it is the case
a one-dimensional novelty score cannot represent.

So this module keeps two measurements:

**Novelty** is the distance to the nearest thing already made. Normalised
compression distance, from Cilibrasi and Vitányi: an approximation to the
information distance that needs no features, no embedding model, and no
parameters. Two objects are close when knowing one helps you describe the
other.

**Intelligibility** is how much the whole corpus helps describe this. It is
the conditional encoding cost against everything, rather than against the
nearest neighbour, which is what lets a fresh recombination of familiar
material score well on both.

Their product is the value, and Berlyne's inverted curve — mid-novelty
preferred over both the obvious and the incomprehensible — comes out of the
arithmetic rather than being imposed on it.

## It uses itself up

``absorb`` puts an artifact in the corpus. Everything near it is then less
novel, so the same move made twice is worth less the second time, and a
generator that repeats itself has a falling score with no boredom heuristic
anywhere in the code. That is the property to check when this is wired to
anything: if repeating is not punished, the corpus is not being updated.

## What it does not know

Compression sees structure and nothing else. It cannot tell whether a thing is
good, whether anyone wants it, or whether the tradition it fits is worth
fitting. It is a filter that removes noise and copies, and everything past
that is somebody else's judgement.

One consequence is worth knowing before trusting a ranking. Something very
regular in an unfamiliar vocabulary scores well: far from everything in the
corpus, and cheap to describe on its own. Measured against a corpus of
melodies built from four motifs, a run of a single unrelated syllable beats a
fresh rearrangement of the four, because it is further out and still
structured. Whether that is the right answer depends on what is being asked —
it is a plausible account of introducing a new element, and a poor account of
inventiveness inside a tradition. Both readings are on the returned object;
only the product is a single number, and a caller that needs the distinction
should weight the two rather than sort on the product.
"""

from __future__ import annotations

import logging
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.Creativity.Novelty")

#: Compression level. Nine because the measurement is a proxy for the shortest
#: description and a lazier compressor is a worse proxy; these payloads are
#: small enough that the time does not signify.
LEVEL = 9

#: Corpus bytes kept for the conditional encoding. The corpus is a working
#: model rather than an archive, and zlib's window makes material beyond about
#: this distance invisible to the measurement anyway.
MAX_CORPUS_BYTES = 262_144

#: Artifacts kept for the nearest-neighbour search.
MAX_ARTIFACTS = 512


def _size(payload: bytes) -> int:
    return len(zlib.compress(payload, LEVEL))


def normalised_compression_distance(a: bytes, b: bytes) -> float:
    """Distance in [0, 1]. Zero for identical, near one for unrelated.

    ``(C(ab) - min(C(a), C(b))) / max(C(a), C(b))``. The value can drift a
    little above one on short inputs where the compressor's own header
    dominates, so it is clamped, and it is not clamped below because a
    negative would mean the compressor found the concatenation shorter than
    its own larger half, which is worth seeing rather than hiding.
    """
    if not a or not b:
        return 1.0
    ca, cb = _size(a), _size(b)
    cab = _size(a + b)
    denominator = max(ca, cb)
    if denominator <= 0:
        return 1.0
    return min(1.0, (cab - min(ca, cb)) / denominator)


@dataclass(frozen=True)
class Valuation:
    """What one artifact is worth, and the two readings it came from."""

    key: str
    novelty: float
    intelligibility: float
    value: float
    nearest: str | None
    corpus_size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "novelty": round(self.novelty, 4),
            "intelligibility": round(self.intelligibility, 4),
            "value": round(self.value, 4),
            "nearest": self.nearest,
            "corpus": self.corpus_size,
        }


@dataclass
class Artifact:
    key: str
    payload: bytes
    at: float = 0.0


class NoveltyValuer:
    """A corpus, and what a new thing would be worth against it.

    The corpus is the state. Everything the object reports is relative to it,
    which is why the same artifact is worth different amounts to different
    observers and why it is worth less the second time.
    """

    def __init__(self) -> None:
        self._artifacts: list[Artifact] = []
        self._corpus: bytes = b""

    def __len__(self) -> int:
        return len(self._artifacts)

    def absorb(self, key: str, payload: bytes) -> None:
        """Add something to what has been seen. The step that uses novelty up."""
        self._artifacts.append(Artifact(key=key, payload=payload))
        if len(self._artifacts) > MAX_ARTIFACTS:
            del self._artifacts[: len(self._artifacts) - MAX_ARTIFACTS]
        self._corpus = (self._corpus + payload)[-MAX_CORPUS_BYTES:]

    def novelty(self, payload: bytes) -> tuple[float, str | None]:
        """Distance to the nearest thing already made, and which one that was."""
        if not self._artifacts:
            return 1.0, None
        best = min(
            (
                (normalised_compression_distance(payload, a.payload), a.key)
                for a in self._artifacts
            ),
            key=lambda pair: pair[0],
        )
        return best[0], best[1]

    def intelligibility(self, payload: bytes) -> float:
        """How much the whole corpus helps describe this, in [0, 1].

        The conditional encoding cost — the joint size minus the corpus's own
        size — against the cost of describing the artifact alone. One means
        the corpus accounts for all of it; zero means the corpus is no help
        and the thing is noise as far as this observer can tell.
        """
        if not payload:
            return 0.0
        alone = _size(payload)
        if alone <= 0:
            return 0.0
        if not self._corpus:
            # With nothing to condition on there is no conditional cost to
            # measure. Reporting one here would say the empty corpus explains
            # everything, and reporting zero would say the first artifact is
            # noise; neither is a measurement, so this is the prior.
            return 0.5
        conditional = _size(self._corpus + payload) - _size(self._corpus)
        return min(1.0, max(0.0, 1.0 - conditional / alone))

    def value(self, key: str, payload: bytes) -> Valuation:
        """Score an artifact without adding it."""
        novelty, nearest = self.novelty(payload)
        legible = self.intelligibility(payload)
        return Valuation(
            key=key, novelty=novelty, intelligibility=legible,
            value=novelty * legible, nearest=nearest,
            corpus_size=len(self._artifacts),
        )

    def offer(self, key: str, payload: bytes) -> Valuation:
        """Score it and keep it. What a generator calls."""
        scored = self.value(key, payload)
        self.absorb(key, payload)
        return scored

    def curve(self, samples: Sequence[tuple[str, bytes]]) -> list[Valuation]:
        """Score several candidates against the same corpus, none absorbed.

        The way to see the shape: hand it a run from copy through recombination
        to noise and read the values. If the middle does not win, either the
        corpus is empty or the samples were not what they were said to be.
        """
        return [self.value(key, payload) for key, payload in samples]

    def status(self) -> dict[str, Any]:
        return {
            "artifacts": len(self._artifacts),
            "corpus_bytes": len(self._corpus),
            "keys": [a.key for a in self._artifacts[-8:]],
        }


_VALUER: NoveltyValuer | None = None


def get_novelty_valuer() -> NoveltyValuer:
    global _VALUER
    if _VALUER is None:
        _VALUER = NoveltyValuer()
    return _VALUER


def reset_novelty_valuer_for_test() -> None:
    global _VALUER
    _VALUER = None
