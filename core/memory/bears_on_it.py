"""Whether a memory bears on what is said now, by what it means.

Recall gave a turn the four memories its search and the clock ranked highest, whether or not they were about anything in it.
LIVE 25 November 2026, 02:00, a question about basil was given three thousand characters of a conversation about chess, read
in some two and a half seconds, and the words of an earlier answer, which she gave again.

Her own encoder reads what is said as a question of memory. The bound is the furthest that a memory on record which does bear
on its question comes to, so what is left out is further from what is said than any of those. Where her encoder is not loaded
every memory is kept, as it was.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

__all__ = ["BEARS", "bear_on", "warm"]

_TASK = "memory_recall"
#: How much of a memory is read, in characters.
READ = 800
Vectors = Callable[[Iterable[str]], list[list[float]]]
_BOUND: dict[Any, float] = {}

#: What is said, and a memory that bears on it without saying it again.
BEARS: tuple[tuple[str, str], ...] = (
    ("What did I tell you about my sister's wedding?", "My sister is getting married in Asheville in June. That sounds lovely. Are you giving a toast?"),
    ("how's the knee doing, remember?", "I twisted my knee on the trail Saturday. Ice it and stay off it a few days."),
    ("Any ideas for dinner?", "I'm allergic to shellfish and I don't eat pork. Noted, I'll keep both in mind."),
    ("I have an interview tomorrow", "I applied to the museum for the archivist job. That suits you."),
    ("should I repot the fern?", "The fern by the window keeps dropping fronds. It may be root-bound."),
    ("what was that movie you liked?", "We watched The Third Man last night. The sewer chase was my favorite part."),
    ("can you help me with the budget again", "I'm trying to save four hundred a month for a car. Then let's cut the subscriptions first."),
    ("my dog is limping", "Biscuit is eleven and has arthritis in her hips. Poor thing."),
    ("did the landlord ever answer?", "The heat's been out since Tuesday and the landlord won't call back. Put it in writing."),
    ("I want to play again", "You lost three games to the engine last night. I'd like another go."),
    ("what should I read next", "I loved Piranesi and hated the last thriller you picked. Then something strange and quiet."),
    ("the app you made me is broken", "Make me an app to track my houseplants. It's in your Applications folder."),
)


def _vectors(texts: Iterable[str]) -> list[list[float]]:
    from core.cognition.evidence_relevance import vectors_if_ready

    return vectors_if_ready(texts)


def _as_asked(said: str) -> str:
    from core.memory.embedding_model import query_prompt

    return query_prompt(_TASK) + str(said)[:READ]


def _bound(vectors: Vectors) -> float | None:
    from core.language.learned_matcher import cosine

    if vectors not in _BOUND:
        made = vectors([text for said, memory in BEARS for text in (_as_asked(said), memory)])
        if len(made) != 2 * len(BEARS):
            return None
        _BOUND[vectors] = min(cosine(made[2 * i], made[2 * i + 1]) for i in range(len(BEARS)))
    return _BOUND[vectors]


def warm() -> bool:
    """Read the pairs on record, so no turn waits for them; whether they could be."""
    return _bound(_vectors) is not None


def bear_on(said: str, memories: Sequence[str], vectors: Vectors | None = None) -> list[bool]:
    """For each memory, whether it bears on what is said: every one, where that cannot be read."""
    from core.language.learned_matcher import cosine

    vectors = vectors or _vectors
    said, texts = " ".join(str(said or "").split()), [" ".join(str(m or "").split())[:READ] for m in memories]
    bound = _bound(vectors) if said and texts else None
    made = vectors([_as_asked(said), *texts]) if bound is not None else []
    if bound is None or len(made) != len(texts) + 1:
        return [True] * len(texts)
    return [cosine(made[0], vector) >= bound for vector in made[1:]]
