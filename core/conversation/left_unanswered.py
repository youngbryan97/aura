"""Whether a reply leaves a question of a message unanswered, by what it means.

A reply that shares no word with a question is not one that leaves it out: "Yeah, more than I expected" answers "did you
enjoy it?". LIVE 24 November 2026, 23:27, such a reply was taken for one that left the question out; the question was asked of
her model again by itself, and its answer was put under the first. He was given two whole replies as one, marked "Partial",
fifty seconds later than the first was written.

Her own encoder reads the question as a question and each sentence of the reply as what might answer it. The bound is the
lowest that any pair on record that does answer comes to, so what is said to be left out is further from its question than
any answer on record is. Where no encoder can be had it is not said either way, and the words decide as they did.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

__all__ = ["ANSWERED", "how_near", "left_out", "settle_by_meaning", "warm"]

_TASK = "answer"
#: What a worker names a part it takes to be left out.
_PART = "unanswered_question_part"
Vectors = Callable[[Iterable[str]], list[list[float]]]
_BOUND: dict[Any, float] = {}

#: A question, and words that answer it, in the ways people do: without repeating it.
ANSWERED: tuple[tuple[str, str], ...] = (
    ("did you enjoy it?", "Yeah, actually — more than I expected. Something clicked around the middle of it."),
    ("how do you feel about the games?", "Quietly good. The gap stopped being vague and became specific."),
    ("Are you free to look at it tomorrow?", "Sure, any time after nine works for me."),
    ("what color did you pick?", "I went with the deep green in the end."),
    ("do you think I should take the job?", "If the commute doesn't scare you, yes. The work sounds like what you've wanted."),
    ("why did it break?", "The belt had worn through, so the drum slipped under load."),
    ("is it safe to eat after that long?", "No, I'd throw it out. Rice left out overnight is a real risk."),
    ("how long does it take to get there?", "About forty minutes by car, a bit more at rush hour."),
    ("who wrote it?", "Stanislaw Lem, in 1961."),
    ("What was the hardest part?", "Keeping the two clocks in step. Everything else was bookkeeping."),
    ("have you heard from her?", "Not since Sunday. She said she'd call after the move."),
    ("which one would you keep?", "The red one. It's lighter and the zip doesn't stick."),
    ("what did you learn?", "That patience beats cleverness in those positions."),
    ("can you explain it simply?", "Think of it as a queue: first in, first out, and nobody cuts the line."),
    ("Does it bother you?", "A little. I'd rather know than guess."),
    ("what should I cook?", "Something slow: lentils with lemon, and bread to mop it up."),
    ("would you do it again?", "In a heartbeat, though I'd sleep first."),
    ("is my plan realistic?", "Mostly. The second month is too full; the rest holds up."),
    ("where did you put the keys?", "On the hook by the back door."),
    ("How are you today?", "Steady, a bit tired, glad you're here."),
    ("what's your favorite so far?", "The one with the lighthouse. I keep coming back to it."),
    ("did watching it bore you?", "Not at all — I was leaning in by the end."),
    ("how does that make you feel?", "Seen, honestly. And a little exposed."),
    ("should we wait for him?", "I'd start without him; he'll catch up."),
)


def _vectors(texts: Iterable[str]) -> list[list[float]]:
    # The encoder her app holds. A worker has none, and what it would cost there is the time of the words being written.
    from core.cognition.evidence_relevance import vectors_if_ready

    return vectors_if_ready(texts)


def _sentences(body: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?;])\s+|\s+—\s+|\n+", body) if s.strip()][:40] or [body]


def _asked(question: str) -> str:
    from core.memory.embedding_model import query_prompt

    return query_prompt(_TASK) + str(question)


def _nearest(asked: Sequence[float], said: Sequence[Sequence[float]]) -> float:
    from core.language.learned_matcher import cosine

    return max((cosine(asked, one) for one in said), default=0.0)


def how_near(question: str, body: str, vectors: Vectors | None = None) -> float | None:
    """How near the nearest of a reply is to answering a question; None where no encoder can be had."""
    body = str(body or "").strip()
    if not str(question or "").strip() or not body:
        return None
    texts = [_asked(question), *_sentences(body), body[:2000]]
    made = (vectors or _vectors)(texts)
    return _nearest(made[0], made[1:]) if len(made) == len(texts) else None


def _bound(vectors: Vectors) -> float | None:
    if vectors not in _BOUND:
        texts: list[str] = []
        spans: list[tuple[int, int]] = []
        for question, answer in ANSWERED:
            said = [*_sentences(answer), answer]
            spans.append((len(texts), len(texts) + 1 + len(said)))
            texts += [_asked(question), *said]
        made = vectors(texts)
        if len(made) != len(texts):
            return None
        _BOUND[vectors] = min(_nearest(made[at], made[at + 1:end]) for at, end in spans)
    return _BOUND[vectors]


def left_out(question: str, body: str, vectors: Vectors | None = None) -> bool | None:
    """Whether a reply leaves a question out: whether nothing in it is as near to it as the furthest answer on record is.
    None where it cannot be read."""
    vectors = vectors or _vectors
    near = how_near(question, body, vectors)
    bound = _bound(vectors) if near is not None else None
    return None if near is None or bound is None else near < bound


def warm() -> bool:
    """Read the pairs on record, so no reply waits for them; whether they could be."""
    return _bound(_vectors) is not None


def settle_by_meaning(receipt: dict[str, Any], asked: str, reply: str) -> bool:
    """Put right, in place, a worker's verdict that a reply left a question out, where what it means answers it; whether it was.

    A worker has the words to judge by and nothing else. Only that is settled here: a reply cut off, or short of something
    else that was asked, stays as it was told.
    """
    if not receipt.get("semantic_completion_missing_part_indexes") or not str(reply or "").strip():
        return False
    other = set(receipt.get("semantic_completion_quality_reasons") or ()) - {_PART}
    if other or receipt.get("semantic_completion_unfulfilled_discourse_count") or receipt.get("semantic_completion_epistemic_partition_covered") is False:
        return False
    if not (receipt.get("semantic_completion_eos_boundary") or receipt.get("semantic_completion_terminal_boundary")):
        return False
    from core.conversation.request_coverage import unanswered_question_parts
    from core.runtime.structured_input import analyze_prompt_shape

    if unanswered_question_parts(reply, analyze_prompt_shape(asked)):
        return False
    receipt.update({"semantic_completion_missing_part_indexes": [], "semantic_completion_missing_part_count": 0, "semantic_completion_incomplete": False,
                    "semantic_completion_satisfied": True, "semantic_completion_settled_by": "meaning"})
    for name in ("surface_quality_gate_reasons", "semantic_completion_quality_reasons"):
        if receipt.get(name):
            receipt[name] = [reason for reason in receipt[name] if reason != _PART]
    return True
