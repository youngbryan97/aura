"""A second voice that answers rather than argues, in the register the first lacked.

Two of these records are built on it and they are built the same way. Oddisee
raps "Contradiction's Maze" at a pitch locked to 80 Hz and a level locked to
-9.0 dB for four minutes, and Maimouna Youssef's chorus comes in at 232 Hz with
a spectral centroid of 3233 against his 2500 — 152 per cent higher in pitch,
brighter, and shorter. It does not disagree with the verse. It asks the
question the verse spends four minutes not asking: is this the phase, or is
this the way? The same shape carries "You Know Who You Are", where Olivier
Daysoul's sung hook answers a rapped verse.

She has two things that are not this. The council argues, which is voices
holding positions against each other. The drafts compete, which is alternatives
for one slot where only one survives. Antiphony is neither: both voices are
kept, the second exists to carry what the first could not, and what makes it an
answer is that its register is deliberately unlike.

`toward` in this package computes the moves that bring her register nearer
somebody's, because matching is what says "I am with you". This is the other
direction, and it is not a failure to match. The measurement of the axis to
move on is the same; the sign is opposite.

    answer(statement) -> the register that would answer it

An utterance that reports and never asks is answered by a question. One that
testifies in the first person is answered by address. One that piles up long
clauses is answered by a short one. Nothing here writes the words.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.expression.register import _AXES, Move, Register, comparable, distance

__all__ = ["Answer", "answer", "answers"]

#: How far apart two registers have to be before the second is answering the
#: first rather than continuing it. Half the coordinate space: nearer than that
#: and the two voices are doing the same thing, which is a duet rather than a
#: call and its response.
APART: float = 0.5


@dataclass(frozen=True)
class Answer:
    """The register that would answer a statement, and how far off it is."""

    moves: tuple[Move, ...] = ()
    apart: float = 0.0
    measured: bool = False
    why: str = "nothing was said to answer"

    def as_dict(self) -> dict[str, Any]:
        return {
            "moves": [
                {"axis": m.axis, "from": round(m.mine, 4), "to": round(m.theirs, 4)}
                for m in self.moves
            ],
            "apart": round(self.apart, 4),
            "measured": self.measured,
            "why": self.why,
        }


def answer(statement: Register, *, most: int = 3) -> Answer:
    """The register that would answer this one, largest inversion first.

    Each share is sent to its own complement rather than to a chosen target, so
    the answer to a register is derived from that register and from nothing
    else. A voice already balanced on an axis is left alone there: the
    complement of a half is a half, and there is nothing to answer.
    """
    # The same width rule the package already applies to comparing two
    # registers. A message with fewer words than there are axes puts a whole
    # share on one of them, so its shape is a reading of its length and the
    # answer to it would be an answer to that.
    if not statement.measured or statement.words < len(_AXES):
        return Answer(why="the statement had no shape to answer")
    moves = tuple(
        sorted(
            (
                Move(axis=axis, mine=getattr(statement, axis), theirs=1.0 - getattr(statement, axis))
                for axis in _AXES
            ),
            key=lambda m: -abs(m.gap),
        )[: max(1, most)]
    )
    apart = sum(abs(m.gap) for m in moves) / max(1, len(moves))
    return Answer(
        moves=moves,
        apart=apart,
        measured=True,
        why=(
            "answering on "
            + ", ".join(f"{m.axis} {m.mine:.2f} to {m.theirs:.2f}" for m in moves)
        ),
    )


def answers(statement: Register, reply: Register) -> bool:
    """Whether the reply answers the statement rather than continuing it.

    Distance rather than direction: a reply can answer on any axis, and
    requiring it to take the particular inversion `answer` computed would be
    scoring it against one way of answering rather than against the thing that
    makes an answer one.
    """
    if not comparable(statement, reply):
        return False
    return distance(statement, reply) >= APART
