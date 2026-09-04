"""What a change has to survive before it is hers, and the record that says it did.

A change that installs itself the moment it looks good is how a system talks
itself into anything. The parts of her differ in how much depends on them, so
what they cost to be wrong about differs, and the evidence should differ with
it. Nothing here is a policy about safety in the abstract; it is arithmetic
about blast radius.

Five tiers, and the tier is read off the part rather than assigned:

    a word, a way of building     nothing else is written over it yet
    a way of computing, a rule    other terms may be written over it
    the order, the proposer       every later search runs through it
    what a change is worth        every later decision runs through it
    the gate                      not a destination at all

A change to a word can be tried on a handful of families. A change to what a
change is worth decides every later change, so being wrong about it is wrong
about everything after, and the evidence wanted is proportionally larger. The
number of families each tier wants comes from Hoeffding at that tier's claim
size, so the ladder is derived rather than declared.

Three states and a stack
------------------------
A change arrives in shadow: installed, measured, not yet believed. It becomes
canary when the probe says it pays, and active when it has paid over a stretch
of ordinary work. Anything can go back, because every promotion pushes what it
replaced onto a stack, and going back is the ordinary outcome rather than the
failure — a change rolled back after the probe regressed is the governance
working, not development failing.

**Shadow here does not mean isolated.** The word usually means a copy running
beside the real thing with its output thrown away, and that is not what
happens: a shadow change is installed and in use from the moment it is made,
and the state is a record of how much is believed about it rather than a limit
on what it can do. What shadow buys is not containment. It is that the change
is reversible and that the record says what it replaced.

That second half was a promise for a long time and not a fact. No caller
anywhere passed ``replaced``, so the stack was always empty, ``put_it_back``
returned None every time, and this paragraph described a mechanism nothing
could reach. It is armed now, and ``test_a_change_that_can_go_back.py``
fails if a promotion stops carrying what it replaced.

Retiring archives rather than destroys. An artifact removed under a size budget
may be wanted again, and re-deriving it can cost more than it ever saved.

Receipts
--------
Every promotion writes a line carrying who started it, what it replaced, what
the evidence was, and a field for the external command that caused it — which
is empty for everything she started. The lines chain: each carries a digest of
the one before, so the record cannot be quietly rewritten to say a decision was
hers. That is the difference between a claim of autonomy and a checkable one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "AReceipt",
    "WHAT_A_TIER_WANTS",
    "a_ledger_of_its_own",
    "archived",
    "forget_the_receipts",
    "how_far_it_reaches",
    "nothing_installs_to_the_gate",
    "promote",
    "put_it_back",
    "the_receipts",
    "the_stack",
    "what_it_replaced",
]

logger = logging.getLogger("Aura.HowAChangeIsPromoted")

#: How far being wrong about each kind of part reaches, and so how large a
#: claim its evidence has to support. Read off the part, not assigned to it:
#: a word affects the terms containing it, an order affects every search, and
#: what a change is worth affects every change after.
HOW_FAR: dict[str, float] = {
    "word": 0.5,
    "what is done": 0.5,
    "way of building": 0.4,
    "way of computing": 0.3,
    "rule": 0.3,
    "the search": 0.2,
    "the deciding": 0.1,
}


def how_far_it_reaches(at: str) -> float:
    """The claim size a change to this part has to support.

    Smaller means further-reaching, because a smaller claim needs more
    evidence — which is the relation wanted: the parts everything runs through
    are the ones a mistake is expensive in.
    """
    return HOW_FAR.get(str(at).split("/", 1)[0], 0.5)


def _families_wanted(at: str) -> int:
    from core.cognition.how_sure_she_is import enough_families_to_say

    return enough_families_to_say(at_least=how_far_it_reaches(at))


#: What each tier wants, derived from the claim its part has to support rather
#: than chosen. Held here so the ladder can be read without running it.
WHAT_A_TIER_WANTS: dict[str, int] = {
    kind: _families_wanted(kind) for kind in HOW_FAR
}


@dataclass(frozen=True, slots=True)
class AReceipt:
    """One line saying a change happened and who caused it."""

    at: str
    #: shadow, canary, active, retired
    became: str
    started_by: str
    evidence: str
    #: What caused it from outside, and empty for everything she started. The
    #: field exists so that its being empty is a fact rather than a silence.
    asked_from_outside: str | None = None
    replaced: str = ""
    when: float = field(default_factory=time.time)
    #: A digest of the line before, so a record cannot be quietly rewritten to
    #: say a decision was hers.
    after: str = ""

    def digest(self) -> str:
        body = json.dumps(
            {
                "at": self.at,
                "became": self.became,
                "started_by": self.started_by,
                "evidence": self.evidence,
                "asked_from_outside": self.asked_from_outside,
                "replaced": self.replaced,
                "when": round(self.when, 6),
                "after": self.after,
            },
            sort_keys=True,
        )
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def describes(self) -> str:
        who = self.asked_from_outside or self.started_by
        return f"{self.at} became {self.became} ← {who}: {self.evidence}"


_RECEIPTS: list[AReceipt] = []
_STACK: list[tuple[str, Any]] = []
_ARCHIVE: dict[str, Any] = {}
_PRIVATE_LEDGER: ContextVar[
    tuple[list[AReceipt], list[tuple[str, Any]], dict[str, Any]] | None
] = ContextVar("aura_private_promotion_ledger", default=None)


def _ledger_stores() -> tuple[
    list[AReceipt], list[tuple[str, Any]], dict[str, Any]
]:
    private = _PRIVATE_LEDGER.get()
    return private if private is not None else (_RECEIPTS, _STACK, _ARCHIVE)


def the_receipts() -> tuple[AReceipt, ...]:
    receipts, _stack, _archive = _ledger_stores()
    return tuple(receipts)


def forget_the_receipts() -> None:
    receipts, stack, archive = _ledger_stores()
    receipts.clear()
    stack.clear()
    archive.clear()


@contextmanager
def a_ledger_of_its_own() -> Iterator[None]:
    """Promote and roll back without writing into the record of what she did.

    The invariant that checks a promotion can go back has to perform one, and
    performing one wrote two lines into the receipt chain every time anybody
    read the health report. The chain is the evidence that a decision was
    hers; filling it with the checks that read it is a way of losing that.

    The lines still chain and the digests still hold inside the scope, so what
    the check verifies is the real mechanism and not a stub.
    """
    token = _PRIVATE_LEDGER.set(([], [], {}))
    try:
        yield
    finally:
        _PRIVATE_LEDGER.reset(token)


def the_stack() -> tuple[tuple[str, Any], ...]:
    """What each promotion replaced, newest last. Going back reads this."""
    _receipts, stack, _archive = _ledger_stores()
    return tuple(stack)


def archived() -> dict[str, Any]:
    """What was retired but not destroyed.

    Re-deriving an artifact can cost more than it ever saved, so a size budget
    is a reason to stop carrying something in the active search and never a
    reason to lose it.
    """
    _receipts, _stack, archive = _ledger_stores()
    return dict(archive)


def nothing_installs_to_the_gate() -> list[str]:
    """Destinations that would let a change decide what is kept. Empty, or a bug.

    The gate stays outside the space for the reason
    `a_gate_inside_the_space_cannot_hold` already executes: a rule that can
    rewrite the thing judging it can pass by changing what passing means. What
    a change is WORTH is inside — she may value her own work differently — and
    what is KEPT is not, because that is the judgement.
    """
    from core.cognition.what_she_could_do_next import WHERE_A_TERM_CAN_GO

    forbidden = {"the gate", "what is kept", "the ruler"}
    return [one for one in WHERE_A_TERM_CAN_GO if one in forbidden]


def promote(
    at: str,
    *,
    became: str,
    started_by: str,
    evidence: str,
    replaced: Any = None,
    asked_from_outside: str | None = None,
) -> AReceipt:
    """Move a change up a state and write the line that says so."""
    receipts, stack, archive = _ledger_stores()
    if replaced is not None:
        stack.append((at, replaced))
        if len(stack) > 64:
            del stack[:-64]
    if became == "retired" and replaced is not None:
        archive[at] = replaced
    made = AReceipt(
        at=at,
        became=became,
        started_by=started_by,
        evidence=evidence,
        asked_from_outside=asked_from_outside,
        replaced="" if replaced is None else str(at),
        after=receipts[-1].digest() if receipts else "",
    )
    receipts.append(made)
    if len(receipts) > 512:
        del receipts[:-512]
    if _PRIVATE_LEDGER.get() is None:
        logger.info("%s", made.describes())
    return made


def what_it_replaced(at: str) -> Any | None:
    """The most recent thing this address held before, or nothing."""
    _receipts, stack, _archive = _ledger_stores()
    for where, was in reversed(stack):
        if where == at:
            return was
    return None


def put_it_back(at: str) -> Any | None:
    """Undo the last promotion at this address. The ordinary outcome, not a failure.

    This used to hand the replaced thing to the caller and leave the undoing
    to them, and no caller ever did it — nor passed ``replaced`` in the first
    place, so the stack was empty and this returned None every time. The
    docstring at the top of this module said anything can go back; nothing
    could. Where the replaced thing knows how to restore itself, this now
    restores it, and the receipt says whether that worked.
    """
    _receipts, stack, _archive = _ledger_stores()
    for index in range(len(stack) - 1, -1, -1):
        where, was = stack[index]
        if where != at:
            continue
        del stack[index]
        went_back = True
        stubborn: Any = ()
        restore = getattr(was, "restore", None)
        if callable(restore):
            try:
                stubborn = restore()
            except Exception as exc:  # noqa: BLE001 - a failed undo is a result
                went_back = False
                stubborn = (f"{type(exc).__name__}: {exc}",)
        promote(
            at,
            became="rolled back" if went_back and not stubborn else "would not go back",
            started_by="she",
            evidence=(
                "the probe did not hold"
                if went_back and not stubborn
                else f"the probe did not hold, and these did not come back: {stubborn}"
            ),
        )
        return was
    return None


def the_chain_holds() -> bool:
    """Does every line still follow the one before?

    A record that can be edited to say a decision was hers is not evidence of
    anything. This is what makes the trace checkable rather than trusted.
    """
    receipts, _stack, _archive = _ledger_stores()
    for at, one in enumerate(receipts):
        expected = receipts[at - 1].digest() if at else ""
        if one.after != expected:
            return False
    return True
