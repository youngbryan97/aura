"""From a turn's reading to the request it opens or belongs to.

"Where is it?" and "try again" are about something already asked for, and which is in what she was asked and not in the
words. So a kind about earlier work is settled against the requests of the conversation: where there is one, the turn
belongs to it, and where there is none, the turn cannot mean that and the next likeliest kind is taken instead.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from core.runtime.what_this_turn_asks import Asked
from core.understanding.effects import ABOUT_WORK_ASKED_FOR
from core.understanding.requests_of_her import MET, WORKING, Request, RequestsOfHer
from core.understanding.what_is_asked import decide

__all__ = ["Settled", "settle"]


@dataclass(frozen=True)
class Settled:
    asked: Asked
    #: The request the turn belongs to, where it belongs to one.
    request: Request | None = None
    #: ``new``, ``on_with`` (taken up again), ``told_more`` (under way, and given more), ``how_it_stands``, or ``nothing``.
    what: str = "nothing"


def _without(asked: Asked, *kinds: str) -> Asked:
    """The reading where the turn cannot mean these kinds."""
    left = {name: p for name, p in asked.by_effect.items() if name not in kinds}
    total = sum(left.values())
    if not left or total <= 0.0:
        return replace(asked, effect="talk", decision="talk", asking=0.0)
    p = {name: value / total for name, value in left.items()}
    effect, decision, asking = decide(p)
    return replace(asked, effect=effect, decision=decision, asking=asking, sure=p.get(effect, 0.0), by_effect=p)


def settle(asked: Asked, ledger: RequestsOfHer, *, conversation: str, before: Sequence[str] = ()) -> Settled:
    """The turn's reading, settled against what she has been asked in this conversation. ``before`` is the words of the
    person's turns just before it, which is how a turn that adds to what was just asked is told from a new request."""
    held = ledger.of(conversation)
    latest = held[-1] if held else None
    unfinished = ledger.unfinished(conversation)
    if not held:
        asked = _without(asked, "resume", "status")
    if asked.decision == "ask_first" and asked.effect == "resume" and unfinished is not None:
        # Going on with what was asked for and is not done is not something nobody asked for.
        asked = replace(asked, decision="act")
    if not asked.acts:
        return Settled(asked)
    if asked.effect == "status":
        return Settled(asked, latest, "how_it_stands")
    if asked.effect == "resume":
        request = unfinished or latest
        assert request is not None
        if request.stands == MET and request.effect == "program" and request.made.get("page"):
            return Settled(replace(asked, effect="program_changed"), request, "on_with")
        return Settled(replace(asked, effect=request.effect), request,
                       "told_more" if request.stands == WORKING else "on_with")
    if asked.effect == "program_changed":
        made = next((r for r in reversed(held) if r.made.get("page")), None)
        return Settled(asked, made, "on_with" if made is not None else "new")
    # Asked for the kind of thing that was asked for just before, and not yet had: more of the same request.
    if (unfinished is not None and unfinished.effect == asked.effect and asked.effect not in ABOUT_WORK_ASKED_FOR
            and any(_same(unfinished.said, words) or _same((unfinished.added or [""])[-1], words) for words in before)):
        return Settled(asked, unfinished, "told_more" if unfinished.stands == WORKING else "on_with")
    return Settled(asked, None, "new")


def _same(one: str, other: str) -> bool:
    return bool(one) and " ".join(str(one).split()) == " ".join(str(other).split())
