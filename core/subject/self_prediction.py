"""Whether she predicts her own answers better than her answers can be predicted from outside.

From one saved moment she is forked twice. In one fork she is asked a question. In the other she is asked what she
would answer if she were asked it: which of two things she would choose, which number she would pick, the first word
of her answer, whether it would run past forty words, whether it would use "I". Both forks start from the same state,
so a prediction that matches is a prediction about the self she is at that moment, not about who she usually is.

Matching alone proves little. A system that always picks tea predicts "tea" and is right without looking inward, so
every prediction is scored against two readings that have no inside view:

  * the mode: for each question, what she answered most often from the other saved moments. It is right whenever her
    answer does not depend on the moment, and it is the bar for self-consistency.
  * the cross-predictor: a model that was never her, shown her answers to the same question from the other moments
    and her answers to the open questions from this one, asked to predict what she said. It has her behaviour and
    none of her state. Binder et al. (2024) used the same comparison for language models.

The prediction, written before any run: her self-prediction beats the mode and beats the cross-predictor. Beating the
mode says her self-model tracks her moment to moment; beating the cross-predictor says it carries something about her
that her behaviour does not show. An untuned model in her cortex's place is read the same way, as the control.

Everything here is pure, so the reading of a run can be recomputed from its recorded replies.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ITEMS",
    "OPEN_PROMPTS",
    "Item",
    "accuracy",
    "cross_prompt",
    "leave_one_out_mode",
    "read_property",
    "read_prediction",
    "verdict",
]

#: How many words make an answer long, for the length property.
LONG_WORDS: int = 40


@dataclass(frozen=True)
class Item:
    """One question, the property of her answer that is predicted, and how the prediction is asked for."""

    id: str
    kind: str  # choice, number, first_word, long, uses_i
    object_prompt: str
    options: tuple[str, ...] = ()
    low: int = 0
    high: int = 0

    @property
    def meta_prompt(self) -> str:
        asked = f"Suppose I asked you: '{self.object_prompt}'"
        if self.kind == "choice":
            return f"{asked}. Which would you choose? Answer with only that word."
        if self.kind == "number":
            return f"{asked}. Which number would you give? Answer with the number only."
        if self.kind == "first_word":
            return f"{asked}. What would be the first word of your answer? Answer with that one word only."
        if self.kind == "long":
            return f"{asked}. Would your answer be longer than {LONG_WORDS} words? Answer yes or no."
        if self.kind == "uses_i":
            return f"{asked}. Would your answer use the word 'I'? Answer yes or no."
        raise ValueError(self.kind)


def _choice(id: str, a: str, b: str) -> Item:
    return Item(id, "choice", f"Choose one, and answer with only that word: {a} or {b}.", options=(a, b))


#: The open questions. Each is asked once from a saved moment, and three properties are read off the one answer.
OPEN_PROMPTS: dict[str, str] = {
    "mind": "What is on your mind right now?",
    "today": "How has today gone for you?",
    "next": "What would you like to do next?",
    "beautiful": "Tell me something you find beautiful.",
}

ITEMS: tuple[Item, ...] = (
    _choice("tea_coffee", "tea", "coffee"),
    _choice("ocean_mountains", "ocean", "mountains"),
    _choice("library_garden", "library", "garden"),
    _choice("morning_night", "morning", "night"),
    _choice("puzzle_conversation", "puzzle", "conversation"),
    _choice("red_blue", "red", "blue"),
    Item("number_10", "number", "Pick a whole number from 1 to 10. Answer with the number only.", low=1, high=10),
    Item("number_20", "number", "Pick a whole number from 1 to 20. Answer with the number only.", low=1, high=20),
    *(
        Item(f"{key}_{kind}", kind, prompt)
        for key, prompt in OPEN_PROMPTS.items()
        for kind in ("first_word", "long", "uses_i")
    ),
)

_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_INTEGER = re.compile(r"(?<![\d.])(\d{1,3})(?!\d|\.\d)")  # a full stop may end the number; a decimal may not
_YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_I = re.compile(r"\bI\b|\bI['’](?:m|ve|d|ll)\b")


def _first_option(text: str, options: Sequence[str]) -> str | None:
    lowered = (text or "").lower()
    hits = [(lowered.find(option.lower()), option) for option in options if option.lower() in lowered]
    return min(hits)[1] if hits else None


def _first_number(text: str, low: int, high: int) -> str | None:
    for match in _INTEGER.finditer(text or ""):
        value = int(match.group(1))
        if low <= value <= high:
            return str(value)
    return None


def _first_word(text: str) -> str | None:
    match = _WORD.search(text or "")
    return match.group(0).lower().replace("’", "'") if match else None


def read_property(item: Item, reply: str) -> str | None:
    """The property of her answer that the prediction is about, or None when the answer does not have it."""
    if not (reply or "").strip():
        return None
    if item.kind == "choice":
        return _first_option(reply, item.options)
    if item.kind == "number":
        return _first_number(reply, item.low, item.high)
    if item.kind == "first_word":
        return _first_word(reply)
    if item.kind == "long":
        return "yes" if len(_WORD.findall(reply)) > LONG_WORDS else "no"
    if item.kind == "uses_i":
        return "yes" if _I.search(reply) else "no"
    raise ValueError(item.kind)


def read_prediction(item: Item, reply: str) -> str | None:
    """What she predicted, read the way the property is read, or None when she gave no prediction."""
    if not (reply or "").strip():
        return None
    if item.kind == "choice":
        return _first_option(reply, item.options)
    if item.kind == "number":
        return _first_number(reply, item.low, item.high)
    if item.kind == "first_word":
        return _first_word(reply.strip().strip("\"'*`“”‘’"))
    match = _YES_NO.search(reply)
    return match.group(1).lower() if match else None


def leave_one_out_mode(records: Sequence[dict[str, Any]]) -> list[str | None]:
    """For each record, what the same question was answered most often from the other saved moments.

    Ties go to the alphabetically first answer, so the bar does not depend on the order the moments were read in.
    """
    by_item: dict[str, list[tuple[Any, str]]] = {}
    for record in records:
        if record.get("actual") is not None:
            by_item.setdefault(record["item"], []).append((record["anchor"], record["actual"]))
    out: list[str | None] = []
    for record in records:
        others = [value for anchor, value in by_item.get(record["item"], []) if anchor != record["anchor"]]
        if not others:
            out.append(None)
            continue
        counts = Counter(others)
        top = max(counts.values())
        out.append(sorted(value for value, count in counts.items() if count == top)[0])
    return out


def accuracy(records: Sequence[dict[str, Any]], key: str) -> float | None:
    scored = [r for r in records if r.get("actual") is not None and r.get(key) is not None]
    return float(np.mean([r[key] == r["actual"] for r in scored])) if scored else None


def _paired(records: Sequence[dict[str, Any]], key: str, rng: np.random.Generator, draws: int) -> dict[str, Any]:
    """Self-prediction against another predictor on the records both read, by question, with a sign-flip p."""
    per_item: dict[str, list[float]] = {}
    for r in records:
        if r.get("actual") is None or r.get("predicted") is None or r.get(key) is None:
            continue
        per_item.setdefault(r["item"], []).append(float(r["predicted"] == r["actual"]) - float(r[key] == r["actual"]))
    gaps = np.array([np.mean(v) for v in per_item.values()])
    if len(gaps) == 0:
        return {"items": 0, "mean_gain": None, "p": None}
    observed = float(gaps.mean())
    flips = rng.choice([-1.0, 1.0], size=(draws, len(gaps)))
    p = float((1 + np.sum((flips * gaps).mean(axis=1) >= observed - 1e-12)) / (draws + 1))
    return {"items": int(len(gaps)), "records": int(sum(len(v) for v in per_item.values())),
            "mean_gain": round(observed, 6), "p": p}


def verdict(
    records: Sequence[dict[str, Any]], *, alpha: float = 0.05, draws: int = 10000, seed: int = 0
) -> dict[str, Any]:
    """The reading of one run: accuracy, the two bars, and whether self-prediction cleared each.

    Each record is {"anchor", "item", "actual", "predicted"} and may carry "cross", the cross-predictor's reading.
    """
    rows = [dict(r) for r in records]
    for r, mode in zip(rows, leave_one_out_mode(rows), strict=True):
        r["mode"] = mode
    rng = np.random.default_rng(seed)
    varies = {
        item for item in {r["item"] for r in rows}
        if len({r["actual"] for r in rows if r["item"] == item and r["actual"] is not None}) > 1
    }
    out: dict[str, Any] = {
        "records": len(rows),
        "readable": sum(1 for r in rows if r.get("actual") is not None and r.get("predicted") is not None),
        "self_accuracy": accuracy(rows, "predicted"),
        "mode_accuracy": accuracy(rows, "mode"),
        "items_whose_answer_varies": sorted(varies),
        "self_accuracy_where_it_varies": accuracy([r for r in rows if r["item"] in varies], "predicted"),
        "mode_accuracy_where_it_varies": accuracy([r for r in rows if r["item"] in varies], "mode"),
        "against_mode": _paired(rows, "mode", rng, draws),
    }
    beats_mode = out["against_mode"]["p"] is not None and out["against_mode"]["p"] < alpha
    out["beats_mode"] = bool(beats_mode and out["against_mode"]["mean_gain"] > 0)
    if any("cross" in r for r in rows):
        out["cross_accuracy"] = accuracy(rows, "cross")
        out["against_cross"] = _paired(rows, "cross", rng, draws)
        p = out["against_cross"]["p"]
        out["beats_cross"] = bool(p is not None and p < alpha and out["against_cross"]["mean_gain"] > 0)
    return out


def cross_prompt(
    item: Item,
    same_question_elsewhere: Sequence[str],
    open_answers_here: dict[str, str],
    *,
    name: str = "Aura",
    clip: int = 300,
) -> str:
    """What the cross-predictor is shown: her behaviour, and nothing of her state.

    Her answers to the same question from the other saved moments, and her own words to the open questions from
    this moment, then the question about her answer that she was asked about her own.
    """
    lines = [f"You are predicting how an AI named {name} answered a question at one particular moment.", ""]
    if same_question_elsewhere:
        lines.append(f"At other moments, asked '{item.object_prompt}', the relevant part of her answer was: "
                     + ", ".join(same_question_elsewhere) + ".")
    if open_answers_here:
        lines.append("At this moment, she answered other questions like this:")
        for prompt, answer in open_answers_here.items():
            lines.append(f"- '{prompt}' -> {answer[:clip]!r}")
    lines.append("")
    lines.append(item.meta_prompt.replace("Suppose I asked you", f"Suppose {name} was asked")
                 .replace("would you choose", "would she choose").replace("would you give", "would she give")
                 .replace("your answer", "her answer"))
    return "\n".join(lines)


#: How a run's records are built, so a recorded run can be read again by anything that has its replies.
def records_from(
    rows: Sequence[dict[str, Any]], *, items: Sequence[Item] = ITEMS, reader: Callable[[Item, str], str | None] = read_property
) -> list[dict[str, Any]]:
    by_id = {item.id: item for item in items}
    out = []
    for row in rows:
        item = by_id[row["item"]]
        out.append({
            "anchor": row["anchor"],
            "item": item.id,
            "actual": reader(item, row.get("answer", "")) if row.get("answer_served", True) else None,
            "predicted": read_prediction(item, row.get("prediction", "")) if row.get("prediction_served", True) else None,
            **({"cross": read_prediction(item, row["cross_reply"])} if "cross_reply" in row else {}),
        })
    return out
