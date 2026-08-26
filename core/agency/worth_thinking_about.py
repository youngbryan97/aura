"""Whether thinking about this one would change what she does.

Effort was decided by a counter: every fifth move got a language pass whatever
the move was, and the four in between got none whatever they were. A counter
cannot tell a forced move from the one that decides the shape of the next
thirty, so it spends the same on both and is wrong about both.

What a thought is worth is the difference it could make. Where she can see
where each move leads and one of them is plainly better, words will not change
the answer and buying them is buying nothing. Where the best two are too close
to call, or she cannot see ahead at all, a thought is the only thing that
could decide it, and it is worth what it costs.

Two guards sit around that. Something at stake buys a pass regardless, because
being wrong there costs more than the thinking. And a run that never uses words
stops being hers, so a horizon brings the question back however clear the
arithmetic looks.
"""

from __future__ import annotations

from typing import Mapping

from core.agency.how_good_is_this import ROOM_MATTERS

__all__ = ["TOO_CLOSE_TO_CALL", "WORTH_A_PASS", "worth_a_pass"]

#: Below this, two futures differ only in the weakest of the reasons a
#: situation can be good, and calling one better than the other is reading
#: noise. Tied to the smallest term the score is built from rather than picked.
TOO_CLOSE_TO_CALL = ROOM_MATTERS

#: What has to be riding on a move before it is worth a pass whatever the
#: arithmetic says. Above this a wrong choice costs more than the thinking.
WORTH_A_PASS = 0.7


def worth_a_pass(
    ahead: Mapping[str, tuple[float, str]] | None,
    *,
    stakes: float = 0.5,
    since_words: int = 0,
    horizon: int = 5,
    unusual: bool = False,
) -> tuple[bool, str]:
    """Whether to spend a language pass on this decision, and why.

    Returns the reason as well as the answer, because a decision about how to
    decide is a decision, and one nobody can account for is indistinguishable
    from a habit.
    """
    if unusual:
        return True, "this is not a routine moment"
    if float(stakes) >= WORTH_A_PASS:
        return True, "there is enough riding on this to be sure"
    if since_words >= max(1, int(horizon)):
        return True, f"{since_words} move(s) without saying anything"
    if not ahead:
        return True, "she cannot see where these lead"
    scores = sorted((score for score, _why in ahead.values()), reverse=True)
    if len(scores) < 2:
        return False, "there is only one way to go"
    gap = scores[0] - scores[1]
    if gap < TOO_CLOSE_TO_CALL:
        return True, f"the best two are {gap:.2f} apart, too close to call"
    return False, f"the best is {gap:.2f} clear, and words would not change it"
