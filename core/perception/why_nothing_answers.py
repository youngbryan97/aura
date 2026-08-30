"""Nothing is answering. There are three reasons, and they want three answers.

"Nothing I do changes anything any more" is a good ending test and a poor
diagnosis. It is equally true when a game is over, when a dialog is sitting on
top of the thing, and when she is looking at somebody else's window — and those
want opposite responses. Treating them as one produced the worst behaviour she
has: pressing keys into a finished board and narrating moves as though a game
were happening, because the loop could not tell "it has ended" from "I cannot
reach it".

The three, and what each is:

    she is looking elsewhere    her window is not the one in front, so every
                                reading is of somebody else's screen and every
                                keystroke goes to them. FIXABLE: bring it back.
    something is in front       a window sits over the part she is using.
                                FIXABLE: decline it, and if it will not go,
                                say so — it is somebody's to answer, not hers.
    it has ended                her window is in front, nothing is over it,
                                and the thing she was acting on is no longer
                                there. NOT fixable, and not a fault: finished
                                is a real outcome and stopping is the right
                                response to it.

The last is decided structurally rather than by reading words. A page that has
ended says so in its own language — Game Over, Session expired, Thanks for your
submission — and no list of those covers the next one. What is common to all of
them is that the THING is gone: she was acting on something laid out, and where
it was there is now prose.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

__all__ = ["ELSEWHERE", "ENDED", "IN_FRONT", "UNKNOWN", "WhyNothingAnswers", "work_out_why"]

logger = logging.getLogger("Aura.WhyNothingAnswers")

#: Her window is not the one in front.
ELSEWHERE = "she is looking elsewhere"
#: Something is sitting over the part she is using.
IN_FRONT = "something is in front of it"
#: The thing she was acting on is not there any more.
ENDED = "it has ended"
#: Her window is in front, nothing is over it, and the thing is still there —
#: so it is none of the three, and saying so is better than picking one.
UNKNOWN = "it is there and not answering"


@dataclass(frozen=True)
class WhyNothingAnswers:
    """Which of the three it is, what it is about, and whether she can fix it."""

    because: str
    what: str = ""
    can_fix: bool = False

    def says(self) -> str:
        """Said the way she would say it, for whoever has to answer for it."""
        if self.because == ELSEWHERE:
            return (
                f"I am not looking at it — {self.what or 'something else'} is in "
                "front. Bringing it back."
            )
        if self.because == IN_FRONT:
            return f"{self.what} is over the part I am using. Trying to move it."
        if self.because == ENDED:
            return "This is finished — what I was working on is not there any more."
        return (
            "It is in front of me, nothing is over it, and nothing I do changes "
            "anything. I do not know why."
        )


def work_out_why(
    *,
    mine: str,
    in_front: str,
    on_top: str,
    still_there: bool,
) -> WhyNothingAnswers:
    """Which of the three reasons this is.

    Every argument is something she can find out: which application is in
    front, what sits above her window over the part she is using, and whether
    the thing she was acting on is still in the reading.

    Ordered by what would make the others meaningless. If she is looking at
    the wrong window then the reading is not of her task at all, so nothing it
    says about the thing being gone means anything; if something is over the
    part she uses then the same holds. Only when she is looking at her own
    work, unobstructed, does the thing's absence mean it has ended.
    """
    hers = str(mine or "").strip().lower()
    front = str(in_front or "").strip().lower()
    if hers and front and front != hers:
        return WhyNothingAnswers(ELSEWHERE, what=str(in_front), can_fix=True)
    if str(on_top or "").strip():
        return WhyNothingAnswers(IN_FRONT, what=str(on_top), can_fix=True)
    if not still_there:
        return WhyNothingAnswers(ENDED)
    return WhyNothingAnswers(UNKNOWN)
