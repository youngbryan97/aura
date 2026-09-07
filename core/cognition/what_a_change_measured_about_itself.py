"""An action that says it judges itself, and what it has to show for that.

The generic gate in :mod:`core.cognition.what_she_could_do_next` takes a
held-out reading before and after a change and puts the change back where it
did not pay. An action can opt out of that by declaring ``judges_itself``, on
the grounds that it runs a stronger test of its own.

Four do. The declaration is a boolean in a table, nothing reads it except the
gate deciding to stop asking, and nothing checks the claim. So the strongest
statement the development layer can make about a self-judging change is that
its author said it was fine — which is the shape an external review named:
individual mechanisms sometimes approach a real evidence gate, and the layer
they all pass through does not impose one.

What closes it is small. A self-judging action returns what it measured
alongside what it did:

* **on** — the families or cases it scored, which have to be ones the change
  was not chosen for. A test run on the evidence that motivated the change
  measures how well the change fits its own reason.
* **before** and **after** — the two numbers.
* **why_it_counts** — a sentence, because a number with no claim attached
  cannot be disagreed with.

Where the evidence is there, the gate records `judged itself` and the record
carries the numbers. Where it is not, the change is recorded as `unmeasured`
like any other unprobed keep, and :func:`claiming_without_showing` names the
action. The opt-out stops being a way to be believed.

One of the four installs nothing at all — asking for an example produces a
question and changes nothing about her, so there is no installed change for
any probe to weigh. That is a different thing from judging yourself and it
says so separately: an action that changes nothing has nothing to prove, and
counting it beside the ones that do is how three real claims hide behind one
honest one.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.WhatAChangeMeasuredAboutItself")

__all__ = [
    "WhatItMeasured",
    "changes_nothing",
    "note_a_claim",
    "the_evidence_in",
    "claiming_without_showing",
    "how_the_self_judged_stand",
    "forget_everything",
]


@dataclass(frozen=True, slots=True)
class WhatItMeasured:
    """One action's own evidence, returned with what it did."""

    #: What the action did, in the words it would have returned anyway.
    said: str
    #: The families or cases it scored. Not the ones that motivated the change.
    on: tuple[str, ...]
    before: float
    after: float
    why_it_counts: str

    @property
    def paid(self) -> bool:
        return self.after > self.before

    @property
    def enough_to_be_believed(self) -> bool:
        """Whether this is evidence at all, before asking what it says."""
        return bool(self.on) and bool(self.why_it_counts.strip())

    def __str__(self) -> str:
        return self.said

    def __bool__(self) -> bool:
        # The gate keeps what is truthy, and an action that measured itself
        # and did not pay must not be kept because it returned an object.
        return bool(self.said) and self.paid

    def to_dict(self) -> dict[str, Any]:
        return {
            "said": self.said,
            "on": list(self.on),
            "before": self.before,
            "after": self.after,
            "paid": self.paid,
            "why_it_counts": self.why_it_counts,
        }


@dataclass(frozen=True, slots=True)
class ChangesNothing:
    """An action that installs nothing, so there is nothing to weigh.

    Asking a question is the case. It is not self-judgment and must not be
    counted as it: three actions that really do run a held-out test and one
    that changes nothing read identically once they share a flag.
    """

    said: str
    because: str = "it installs nothing"

    def __str__(self) -> str:
        return self.said

    def __bool__(self) -> bool:
        return bool(self.said)


def changes_nothing(said: str, because: str = "it installs nothing") -> ChangesNothing:
    return ChangesNothing(said=str(said), because=str(because))


_CLAIMS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def the_evidence_in(said: Any) -> WhatItMeasured | None:
    """The evidence an action returned, where it returned any."""
    return said if isinstance(said, WhatItMeasured) else None


def note_a_claim(name: str, said: Any) -> str:
    """Record what a self-judging action showed, and say how it was judged.

    Returns the verdict the gate should count: ``judged itself`` where the
    action showed its working, ``changes nothing`` where it installs nothing,
    and ``unmeasured`` where it claimed the opt-out and showed nothing.
    """
    measured = the_evidence_in(said)
    if measured is not None:
        verdict = (
            "judged itself" if measured.enough_to_be_believed else "unmeasured"
        )
        row: dict[str, Any] = {"verdict": verdict, **measured.to_dict()}
    elif isinstance(said, ChangesNothing):
        verdict = "changes nothing"
        row = {"verdict": verdict, "said": said.said, "because": said.because}
    else:
        verdict = "unmeasured"
        row = {
            "verdict": verdict,
            "said": str(said)[:200],
            "why": "it declared it judges itself and showed no measurement",
        }
    with _LOCK:
        _CLAIMS[str(name)] = row
    if verdict == "unmeasured":
        logger.info(
            "%s claims to judge itself and showed no measurement; counted as "
            "unmeasured", name
        )
    return verdict


def claiming_without_showing() -> tuple[str, ...]:
    """Self-judging actions that have run and shown nothing.

    The ratchet. An action here is one whose keeps rest on its author's word.
    """
    with _LOCK:
        return tuple(
            sorted(
                name
                for name, row in _CLAIMS.items()
                if row.get("verdict") == "unmeasured"
            )
        )


def how_the_self_judged_stand() -> dict[str, Any]:
    """For the health report: who opted out, and what each of them showed."""
    with _LOCK:
        rows = {name: dict(row) for name, row in sorted(_CLAIMS.items())}
    counted: dict[str, int] = {}
    for row in rows.values():
        counted[row["verdict"]] = counted.get(row["verdict"], 0) + 1
    return {
        "schema": "aura.development.self_judged.v1",
        "ran": len(rows),
        "counts": counted,
        "claiming_without_showing": list(claiming_without_showing()),
        "each": rows,
    }


def forget_everything() -> None:
    with _LOCK:
        _CLAIMS.clear()
