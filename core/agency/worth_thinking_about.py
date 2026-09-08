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
    recognised: str = "",
    costs_moves: float = 1.0,
    how_sure: float = 0.0,
) -> tuple[bool, str]:
    """Whether to spend a language pass on this decision, and why.

    Returns the reason as well as the answer, because a decision about how to
    decide is a decision, and one nobody can account for is indistinguishable
    from a habit.

    ``recognised`` is what has worked before from a position of this kind, if
    anything has. Recognition is what frees her from deciding — and where it
    disagrees with the arithmetic, that disagreement is the surest sign this
    position is not the routine one it looked like.

    ``costs_moves`` is what a pass costs, in moves not made. A pass on a small
    model costs a fraction of a move; on a resident model under memory
    pressure it costs the time of ten, and the same "the best two are close"
    that is worth buying at one is not worth buying at ten. Measured by the
    caller from its own clock, so nothing here is tuned for a machine.

    LIVE 2026-09-04, playing on a real board: a language pass every move at
    about twenty-seven seconds each, on a game that takes hundreds of moves.
    What the answer was worth never met what asking for it cost.

    The bars it moves are the ones about VALUE. Nothing here excuses her from
    thinking when there is something she cannot see, when what worked before
    disagrees with what she can see now, or when enough rides on it — those
    are necessities and a price does not change them.
    """
    dear = max(1.0, float(costs_moves or 1.0))
    if unusual:
        return True, "this is not a routine moment"
    best = _the_best(ahead)
    if recognised and best and recognised != best:
        # What worked here before and what looking ahead says now do not
        # agree. One of them is wrong about this position, and finding out
        # which is exactly what thinking is for.
        return True, f"{recognised} has worked from positions like this, but {best} looks better now"
    if float(stakes) >= WORTH_A_PASS:
        return True, "there is enough riding on this to be sure"
    if since_words >= max(1, int(horizon * dear)):
        return True, f"{since_words} move(s) without saying anything"
    if recognised and recognised == best:
        return False, f"a position of a kind she has met, where {recognised} has been working"
    if not ahead:
        return True, "she cannot see where these lead"
    scores = sorted((score for score, _why in ahead.values()), reverse=True)
    if len(scores) < 2:
        return False, "there is only one way to go"
    gap = scores[0] - scores[1]
    # Two futures being close is not a reason to think about them.
    #
    # It was: anything inside the smallest term of the score bought a
    # language pass. On a board where the best two moves are nearly always
    # close, that is a pass on almost every move — measured live, forty-eight
    # of them for nineteen moves, at a cost of about ten moves each. And the
    # thing being bought is the difference between two futures worth the same,
    # which is worth about nothing.
    #
    # Thinking is for when the arithmetic is UNRELIABLE, not for when it is
    # tied. So the bar is her own error rather than a fixed distance: a rule
    # right most of the time makes scores she can trust to sort two futures a
    # hair apart, and a rule right half the time does not. Where the gap is
    # inside what her own model could have got wrong, she cannot tell, and
    # that is worth a pass. Where it is outside, she can, however small it is.
    #
    # It tightens itself as she learns. Nothing to turn off, and nothing set:
    # the number is how often her own rule has been right about this world.
    #
    # The scale is the spread of what is on offer, floored at the smallest
    # difference that means anything at all.
    #
    # Without the floor the rule is degenerate wherever there are exactly two
    # options: the spread IS the gap being tested, so `gap < (1 - how_sure) *
    # spread` reduces to `gap < gap` and is false at every value of how_sure. A
    # choice between two futures could never buy a pass however unreliable her
    # arithmetic was, which is the case the bar exists for.
    spread = max(scores[0] - scores[-1], TOO_CLOSE_TO_CALL)
    inside_her_own_error = (1.0 - max(0.0, min(1.0, float(how_sure)))) * spread
    if gap < inside_her_own_error / dear:
        return True, (
            f"the best two are {gap:.2f} apart, inside what her model of this "
            f"world gets wrong ({how_sure:.0%} right)"
        )
    return (
        False,
        f"the best is {gap:.2f} clear, and words costing {dear:.0f} move(s) "
        "would not change it"
        if dear > 1.0
        else f"the best is {gap:.2f} clear, and words would not change it",
    )


def _the_best(ahead: Mapping[str, tuple[float, str]] | None) -> str:
    """Whichever move looking ahead puts first, if it puts anything first."""
    if not ahead:
        return ""
    return max(ahead.items(), key=lambda move: move[1][0])[0]
