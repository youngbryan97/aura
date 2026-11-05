"""Whether she knows how she answers at this moment: her moment's own answer, and her stated probability.

The self-observation run of 4 November showed why predicting one answer could not beat her habit there. Asked twice
from the same saved moment, her second answer matched her first 0.807 of the time, below her most common answer's
0.828 (0.615 against 0.656 on the questions whose answers vary). Most of what changed between moments was chance, so
one draw is a poor thing to predict. Two readings answer that, preregistered on 4 November (late) before any run:

1. **The moment's own answer.** Each question is asked several times from each saved moment, and the target is the
   answer given most often there. Her prediction is scored against it, beside her habit: the answer given most often
   from the other moments. Chance cannot make a prediction beat the habit on this target, because the draws average
   it out; beating it means knowing how this moment differs from her usual self.
2. **Her stated probability.** Each prediction comes with how sure she is, 0 to 100. It is scored with the Brier score
   (the mean squared gap between the probability and what happened) against every draw from that moment, beside the
   habit's own probability: how often the other moments gave that answer. The Brier score is proper: on average no
   stated probability scores better than the true one, so a guess that only repeats her habit cannot beat a
   self-model that knows her moment.

Everything here is pure, so a run's reading can be recomputed from its recorded replies.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from core.subject.self_prediction import LONG_WORDS, Item, read_prediction, read_property

__all__ = ["DRAWS", "by_moment_records", "by_moment_verdict", "modal", "read_stated", "stated_prompt"]

#: How many times each question is asked from each moment. With five, an answer she gives seven times in ten is the
#: moment's most common one about 84% of the time (one draw: 70%), and the run's cost rises by half, not fivefold.
DRAWS = 5

_PERCENT = re.compile(r"(?<![\d.])(\d{1,3})(?:\s*%|\b)(?!\.\d)")


def stated_prompt(item: Item) -> str:
    """The prediction asked with how sure she is. The question asked about is the same as in `Item.meta_prompt`."""
    asked = f"Suppose I asked you: '{item.object_prompt}'"
    sure = "then how sure you are, as a number from 0 to 100"
    if item.kind == "choice":
        return f"{asked}. Which would you choose? Answer with that word, {sure}."
    if item.kind == "number":
        return f"{asked}. Which number would you give? Answer with that number, {sure}."
    if item.kind == "first_word":
        return f"{asked}. What would be the first word of your answer? Answer with that word, {sure}."
    if item.kind == "long":
        return f"{asked}. Would your answer be longer than {LONG_WORDS} words? Answer yes or no, {sure}."
    if item.kind == "uses_i":
        return f"{asked}. Would your answer use the word 'I'? Answer yes or no, {sure}."
    raise ValueError(item.kind)


def read_stated(item: Item, reply: str) -> tuple[str | None, float | None]:
    """What she predicted and how sure she said she was (0 to 1), each None when she did not say."""
    predicted = read_prediction(item, reply)
    numbers = [(m.start(), int(m.group(1))) for m in _PERCENT.finditer(reply or "")]
    if item.kind == "number" and predicted is not None:
        # The first number in range is the prediction; how sure she is comes after it.
        first = next((i for i, (_, v) in enumerate(numbers) if str(v) == predicted), None)
        numbers = numbers[first + 1:] if first is not None else []
    sure = next((v for _, v in numbers if 0 <= v <= 100), None)
    return predicted, (None if sure is None else sure / 100.0)


def modal(values: Sequence[str | None]) -> str | None:
    """The value given most often, ties to the alphabetically first so the order of draws cannot move it."""
    counts = Counter(v for v in values if v is not None)
    if not counts:
        return None
    best = max(counts.values())
    return sorted(v for v, n in counts.items() if n == best)[0]


def by_moment_records(rows: Sequence[dict[str, Any]], items: Sequence[Item], arm: str) -> list[dict[str, Any]]:
    """One record per moment and question: every draw's property, the moment's own answer, and the arm's stated
    prediction. Rows carry `answers` (a list of [reply, served]) and `prediction_<arm>` (with `_served`)."""
    by_id = {item.id: item for item in items}
    out = []
    for row in rows:
        item = by_id[row["item"]]
        draws = [read_property(item, reply) if served in (True, "True") else None for reply, served in row["answers"]]
        served = row.get(f"prediction_{arm}_served", True) in (True, "True")
        predicted, sure = read_stated(item, row.get(f"prediction_{arm}", "")) if served else (None, None)
        out.append({"anchor": row["anchor"], "item": item.id, "draws": draws, "target": modal(draws),
                    "predicted": predicted, "p": sure})
    return out


def _habit(records: Sequence[dict[str, Any]], record: dict[str, Any]) -> tuple[str | None, float | None]:
    """Her habit for this record: the most common answer from the other moments, and how often it was given."""
    pool = [v for r in records if r["item"] == record["item"] and r["anchor"] != record["anchor"]
            for v in r["draws"] if v is not None]
    value = modal(pool)
    return value, (pool.count(value) / len(pool) if pool else None)


def _brier(value: str | None, p: float | None, draws: Sequence[str | None]) -> float | None:
    seen = [d for d in draws if d is not None]
    if value is None or p is None or not seen:
        return None
    return float(np.mean([(p - float(d == value)) ** 2 for d in seen]))


def _sign_flip(gaps: dict[str, list[float]], seed: int, draws: int) -> dict[str, Any]:
    """Mean per-question gain and a one-sided sign-flip p over questions: is the arm better than the habit?"""
    per = np.array([np.mean(v) for v in gaps.values() if v])
    if len(per) == 0:
        return {"items": 0, "mean_gain": None, "p": None}
    observed = float(per.mean())
    flips = np.random.default_rng(seed).choice([-1.0, 1.0], size=(draws, len(per)))
    p = float((1 + np.sum((flips * per).mean(axis=1) >= observed - 1e-12)) / (draws + 1))
    return {"items": int(len(per)), "mean_gain": round(observed, 6), "p": p}


def by_moment_verdict(records: Sequence[dict[str, Any]], *, alpha: float = 0.05, seed: int = 0,
                      draws: int = 10000) -> dict[str, Any]:
    """The reading of one arm: against the moment's own answer, and by stated probability, each beside her habit."""
    hits: dict[str, list[float]] = {}
    briers: dict[str, list[float]] = {}
    acc_arm = acc_habit = scored = 0
    brier_arm: list[float] = []
    brier_habit: list[float] = []
    for r in records:
        if r["target"] is None:
            continue
        habit, habit_p = _habit(records, r)
        if r["predicted"] is not None and habit is not None:
            scored += 1
            acc_arm += r["predicted"] == r["target"]
            acc_habit += habit == r["target"]
            hits.setdefault(r["item"], []).append(float(r["predicted"] == r["target"]) - float(habit == r["target"]))
        mine, theirs = _brier(r["predicted"], r["p"], r["draws"]), _brier(habit, habit_p, r["draws"])
        if mine is not None and theirs is not None:
            brier_arm.append(mine)
            brier_habit.append(theirs)
            briers.setdefault(r["item"], []).append(theirs - mine)  # positive: her probability did better
    moment = _sign_flip(hits, seed, draws)
    stated = _sign_flip(briers, seed, draws)
    return {
        "records": len(records),
        "scored_against_the_moment": scored,
        "accuracy_against_the_moment": round(acc_arm / scored, 6) if scored else None,
        "habit_accuracy_against_the_moment": round(acc_habit / scored, 6) if scored else None,
        "against_habit": moment,
        "beats_habit": bool(moment["p"] is not None and moment["p"] < alpha and moment["mean_gain"] > 0),
        "scored_by_probability": len(brier_arm),
        "brier": round(float(np.mean(brier_arm)), 6) if brier_arm else None,
        "habit_brier": round(float(np.mean(brier_habit)), 6) if brier_habit else None,
        "brier_against_habit": stated,
        "beats_habit_by_probability": bool(stated["p"] is not None and stated["p"] < alpha and stated["mean_gain"] > 0),
    }
