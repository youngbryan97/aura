"""The form of regard a particular person welcomes.

Two lines from the fourth seven are about the form regard takes rather than
whether it is there. "People Watching" appreciates being cared about and does
not want it displayed. "Special" gives regard that names the damage ("in case
nobody told you today") instead of talking past it. `core/social/the_kind_it_was.py`
reads the form of what arrives, and nothing learned, per person, which form of
what she sends is welcome.

Two forms are read from a reply she is about to send:

    shown     regard said outright: that they matter, that something they did
              was good, a question about how they are. One or zero, from
              `core/social/never_told.py`.
    named     when they are telling her about something of theirs, the share
              of their words her reply carries. A reply that names what they said
              scores high; one that reassures in general scores low.

After a reply goes out, the next thing that person says is read as welcome
(warm, with no correction or frustration in it), unwelcome (a correction or
frustration), or neither. Per person and form:

    fit = sum(form value * reaction) / (sum(form value) + 2)

which is the Laplace-smoothed welcome rate moved to (-1, 1): zero with no
evidence, and approaching one or minus one only as welcomes or refusals pile
up against a form the reply actually carried.

`form_fit` scores a candidate reply for the person she is talking to: each
form it carries, times how that person has taken the form. It is a feature of
every candidate, weighed by the taste model at the prior the other fit
features carry, so the reply that wins is the one shaped the way this person
takes regard. The reply that wins is recorded, and their next message teaches
the fit again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "FORMS",
    "FormLedger",
    "forms_of",
    "get_form_ledger",
    "note_heard",
    "note_sent",
    "reset_for_test",
]

#: The forms read from a reply, in a fixed order.
FORMS: tuple[str, ...] = ("shown", "named")


def forms_of(reply: str, their_message: str = "") -> dict[str, float]:
    """How much of each form a reply carries, given what it answers."""
    text = str(reply or "")
    shown = 0.0
    named = 0.0
    try:
        from core.social.never_told import kinds_in

        shown = 1.0 if kinds_in(text) else 0.0
    except (ImportError, AttributeError, TypeError, ValueError):
        shown = 0.0
    try:
        from core.expression.register import read

        theirs = read(str(their_message or ""))
        # Telling her about something of theirs rather than asking for help
        # with it. The register's stricter reading, which also wants the
        # telling to hold on, is for records nobody wants fixed; one telling is
        # enough for a reply to name it or talk past it.
        if theirs.measured and theirs.stance() == "testimony" and not theirs.asks_for_help():
            said = {word for word in str(their_message).lower().split() if len(word) > 2}
            mine = {word for word in text.lower().split() if len(word) > 2}
            named = len(said & mine) / len(said) if said else 0.0
    except (ImportError, AttributeError, TypeError, ValueError):
        named = 0.0
    return {"shown": shown, "named": max(0.0, min(1.0, named))}


@dataclass
class _Person:
    carried: dict[str, float] = field(default_factory=lambda: dict.fromkeys(FORMS, 0.0))
    welcomed: dict[str, float] = field(default_factory=lambda: dict.fromkeys(FORMS, 0.0))
    pending: dict[str, float] | None = None

    def fit(self, form: str) -> float:
        return self.welcomed.get(form, 0.0) / (self.carried.get(form, 0.0) + 2.0)


class FormLedger:
    """Per person, how each form of regard she sent was taken."""

    def __init__(self) -> None:
        self._people: dict[str, _Person] = {}
        self._partner = ""

    def _person(self, name: str) -> _Person:
        held = self._people.get(name)
        if held is None:
            held = _Person()
            self._people[name] = held
        return held

    def set_partner(self, partner: str) -> None:
        """Who she is talking to now, so candidates are scored for them."""
        self._partner = str(partner or "").strip()

    @property
    def partner(self) -> str:
        return self._partner

    def note_reply(self, reply: str, their_message: str = "", *, partner: str = "") -> None:
        """The reply that went out, and the forms it carried."""
        name = str(partner or self._partner or "").strip()
        if not name:
            return
        self._person(name).pending = forms_of(reply, their_message)

    def note_reaction(self, partner: str, *, welcomed: bool, unwelcomed: bool) -> None:
        """What they said next, read as a reaction to the reply before it."""
        name = str(partner or "").strip()
        held = self._people.get(name)
        if held is None or held.pending is None:
            return
        forms, held.pending = held.pending, None
        if welcomed == unwelcomed:
            # Neither, or both at once: nothing to learn about the form.
            return
        sign = 1.0 if welcomed else -1.0
        for form, value in forms.items():
            if value <= 0.0:
                continue
            held.carried[form] = held.carried.get(form, 0.0) + value
            held.welcomed[form] = held.welcomed.get(form, 0.0) + sign * value

    def fit(self, partner: str, form: str) -> float:
        held = self._people.get(str(partner or "").strip())
        return held.fit(form) if held is not None else 0.0

    def form_fit(self, reply: str, their_message: str = "", *, partner: str = "") -> float:
        """How well a candidate's forms suit the person, in [-1, 1]."""
        name = str(partner or self._partner or "").strip()
        held = self._people.get(name)
        if held is None:
            return 0.0
        forms = forms_of(reply, their_message)
        score = sum(value * held.fit(form) for form, value in forms.items())
        return max(-1.0, min(1.0, score))

    def status(self) -> dict[str, Any]:
        return {
            name: {form: round(held.fit(form), 4) for form in FORMS}
            for name, held in sorted(self._people.items())
        }


def note_heard(partner: str, warm: bool, objected: bool) -> None:
    """Their message arrived: read it as a reaction to her last reply, and
    score the next candidates for them. Warm with no objection is a welcome;
    a correction or frustration is not.
    """
    ledger = get_form_ledger()
    ledger.set_partner(partner)
    ledger.note_reaction(partner, welcomed=bool(warm) and not objected, unwelcomed=bool(objected))


def note_sent(reply: str, their_message: str, partner: str) -> None:
    """A reply went out: keep its forms for their next message to teach, and
    count the regard in it as given, whether or not anything comes back.
    See core/social/what_passes_between.py.
    """
    get_form_ledger().note_reply(reply, their_message, partner=partner)
    if forms_of(reply, their_message).get("shown", 0.0) > 0.0:
        from core.social.what_passes_between import get_between_ledger

        get_between_ledger().note_given(partner)


_LEDGER: FormLedger | None = None


def get_form_ledger() -> FormLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = FormLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
