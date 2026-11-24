"""What people say when they want something brought about, and what they say with the same words when they do not.

``heard/asked.tsv`` and ``heard/not_asked.tsv`` hold one utterance a line with the kind it is. ``heard/in_conversation.tsv``
adds what was said before it, because the same words can be either: "Could you build a chess clock?" after "I'm curious
what you're capable of" is a question about her, and after "My club keeps arguing about time" it is a request. A file
that came with an utterance is named at its end in square brackets, as it is in a live transcript.

Every line is invented. Nothing a person said to her is kept here.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from core.understanding.effects import BY_NAME

__all__ = ["Example", "Turn", "attached", "examples", "transcript"]

_HEARD = Path(__file__).resolve().parent / "heard"

#: The most characters of an earlier turn that are kept: its end, which is what the next turn answers.
MOST_OF_AN_EARLIER_TURN = 420
#: The most characters kept of either end of what was said, with what lies between counted and left out.
MOST_OF_AN_END = 700


@dataclass(frozen=True)
class Turn:
    #: ``user`` or ``assistant``.
    role: str
    said: str


@dataclass(frozen=True)
class Example:
    effect: str
    said: str
    before: tuple[Turn, ...] = field(default=())


def attached(said: str, names: Sequence[str] = ()) -> str:
    """What was said, with what came with it named at its end."""
    marks = " ".join(f"[attached: {' '.join(str(name).split())}]" for name in names if str(name).strip())
    return f"{said} {marks}".strip() if marks else str(said)


def _kept(said: str) -> str:
    """What was said, whole where it is short; both its ends and the size of what lies between where it is long."""
    said = str(said or "").strip()
    if len(said) <= 2 * MOST_OF_AN_END:
        return said
    head, tail = said[:MOST_OF_AN_END], said[-MOST_OF_AN_END:]
    between = len(said[MOST_OF_AN_END:-MOST_OF_AN_END].split())
    return f"{head} [pasted text, {between} words] {tail}"


def transcript(said: str, before: Sequence[Turn] = ()) -> str:
    """The exchange as it stood, in the bare chat markup, with her turn opened after it and no system message."""
    parts = []
    for turn in before:
        kept = " ".join(str(turn.said or "").split())[-MOST_OF_AN_EARLIER_TURN:]
        if kept:
            parts.append(f"<|im_start|>{turn.role}\n{kept}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{_kept(said)}<|im_end|>\n<|im_start|>assistant\n")
    return "".join(parts)


def _turns(text: str) -> tuple[Turn, ...]:
    said = []
    for part in (piece.strip() for piece in text.split("|")):
        if part.startswith(("U:", "A:")):
            said.append(Turn("user" if part[0] == "U" else "assistant", part[2:].strip()))
    return tuple(said)


def _read(name: str) -> list[Example]:
    rows = []
    for line in (_HEARD / name).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0] in BY_NAME and parts[1].strip():
            rows.append(Example(parts[0], parts[1].strip(), _turns(parts[2]) if len(parts) > 2 else ()))
    return rows


@lru_cache(maxsize=1)
def examples() -> tuple[Example, ...]:
    """Every example, in a fixed order."""
    return tuple([*_read("asked.tsv"), *_read("not_asked.tsv"), *_read("in_conversation.tsv")])
