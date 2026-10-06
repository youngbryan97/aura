"""Where no test says pass or fail: many readers comparing two at a time, on named criteria, both ways round.

"Write an essay that changes minds about interest rates" and "design a screen
more intuitive than Apple's" have no check that returns pass or fail. One
model asked "is this good?" says yes, leaning whichever way its last sentence
leaned, and the same question asked again gets another answer. What settles
such things among people is comparison: two versions side by side, which does
this better, asked of several readers who differ, on what the goal is made of.
Here that is the fitness:

- the goal's criteria, each a question asked of a pair ("which answers the
  strongest objection better?"), and measures that need no reader wherever one
  exists (how easily it reads, how long its sentences run, how much of a
  screen's text is readable against its background, how big its targets are);
- every pair asked of every reader both ways round, so the side a version was
  shown on cancels out instead of deciding;
- the votes made into strengths (Bradley-Terry) with intervals from
  resampling the votes, and a winner said only where its interval stands clear
  of the rest; otherwise it is said to be a tie, which is often the truth.

A reader is any callable ``(criterion, first, second) -> 1 | 2 | 0`` (0: no
preference): her model in a stance, a persona of the audience, a person.
Nothing here knows what is being judged.
"""
from __future__ import annotations

import math
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import combinations

__all__ = ["Standing", "contrast_ratio", "fitts_difficulty", "judged_by_many", "reading_ease", "strengths"]

Reader = Callable[[str, str, str], int]


@dataclass
class Standing:
    """How the candidates stand on one criterion, or on all of them together."""

    strength: dict[int, float]
    low: dict[int, float]
    high: dict[int, float]
    winner: int | None
    votes: int
    per_criterion: dict[str, dict[int, float]] = field(default_factory=dict)

    def says(self, names: Sequence[str] | None = None) -> str:
        name = (lambda i: names[i]) if names else (lambda i: f"version {i + 1}")
        order = sorted(self.strength, key=self.strength.get, reverse=True)
        ranked = ", ".join(f"{name(i)} {self.strength[i]:.2f} ({self.low[i]:.2f} to {self.high[i]:.2f})" for i in order)
        verdict = f"{name(self.winner)} is better, clear of the rest" if self.winner is not None else "no version stands clear of the others: a tie"
        return f"{verdict}; from {self.votes} comparisons: {ranked}"


def strengths(n: int, wins: list[tuple[int, int, float]], *, rounds: int = 200) -> list[float]:
    """Bradley-Terry strengths of ``n`` candidates from (winner, loser, weight) votes; ties are half a win each way. Shares that sum to 1."""
    won = [0.0] * n
    met: dict[tuple[int, int], float] = {}
    for a, b, w in wins:
        won[a] += w
        key = (min(a, b), max(a, b))
        met[key] = met.get(key, 0.0) + w
    p = [1.0 / n] * n
    for _ in range(rounds):
        nxt = []
        for i in range(n):
            denominator = sum(count / (p[i] + p[j if i == a else a]) for (a, j), count in met.items() if i in (a, j))
            nxt.append((won[i] + 0.5) / denominator if denominator else p[i])  # half a win of prior: none is certain from few votes
        total = sum(nxt)
        p = [x / total for x in nxt]
    return p


def judged_by_many(candidates: Sequence[str], criteria: Sequence[str], readers: Sequence[Reader], *,
                   measures: dict[str, Callable[[str], float]] | None = None, resamples: int = 300, seed: int = 0) -> Standing:
    """The candidates compared two at a time by every reader on every criterion, both ways round, and measured where a measure exists."""
    n = len(candidates)
    votes: list[tuple[str, int, int, float]] = []  # (criterion, winner, loser, weight)
    for criterion in criteria:
        for a, b in combinations(range(n), 2):
            for reader in readers:
                for first, second in ((a, b), (b, a)):
                    said = reader(criterion, candidates[first], candidates[second])
                    if said == 1:
                        votes.append((criterion, first, second, 1.0))
                    elif said == 2:
                        votes.append((criterion, second, first, 1.0))
                    else:
                        votes += [(criterion, first, second, 0.5), (criterion, second, first, 0.5)]
    for name, measure in (measures or {}).items():
        # A measure is a reader that is never unsure and never moved by order: each pair once, by its numbers.
        scores = [measure(c) for c in candidates]
        for a, b in combinations(range(n), 2):
            if scores[a] != scores[b]:
                hi, lo = (a, b) if scores[a] > scores[b] else (b, a)
                votes.append((name, hi, lo, float(len(readers))))
    overall = strengths(n, [(w, lost, x) for _c, w, lost, x in votes])
    per = {c: dict(enumerate(strengths(n, [(w, lost, x) for cc, w, lost, x in votes if cc == c]))) for c in {v[0] for v in votes}}
    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        drawn = [votes[rng.randrange(len(votes))] for _ in votes] if votes else []
        samples.append(strengths(n, [(w, lost, x) for _c, w, lost, x in drawn], rounds=60))
    low = {i: sorted(s[i] for s in samples)[int(0.05 * len(samples))] if samples else overall[i] for i in range(n)}
    high = {i: sorted(s[i] for s in samples)[int(0.95 * len(samples)) - 1] if samples else overall[i] for i in range(n)}
    best = max(range(n), key=lambda i: overall[i])
    clear = all(low[best] > high[j] for j in range(n) if j != best)
    return Standing(dict(enumerate(overall)), low, high, best if clear else None, len(votes), per)


# ── measures that need no reader ──────────────────────────────────────

def _syllables(word: str) -> int:
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups) - (1 if word.endswith("e") and len(groups) > 1 and not word.endswith("le") else 0)
    return max(1, count)


def reading_ease(text: str) -> float:
    """Flesch reading ease: higher reads more easily (60-70 is plain English)."""
    sentences = max(1, len(re.findall(r"[.!?]+(?:\s|$)", text)) or 1)
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return 0.0
    return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (sum(_syllables(w) for w in words) / len(words))


def _luminance(colour: str) -> float:
    value = colour.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    channels = [int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(ink: str, ground: str) -> float:
    """WCAG contrast ratio of text colour against its background: 4.5 is the floor for body text, 7 is comfortable."""
    a, b = sorted((_luminance(ink), _luminance(ground)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def fitts_difficulty(distance_px: float, width_px: float) -> float:
    """Fitts's index of difficulty of reaching a target: bits; lower is quicker to hit."""
    return math.log2(distance_px / max(1.0, width_px) + 1)
