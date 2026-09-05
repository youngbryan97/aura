"""core/cognition/what_the_whole_organism_costs.py — evidence from all of her.

The developmental policy chooses what to change about herself from a record of
what her own work has cost. The record is right and its intake was narrow:
almost everything in it came from one cognitive ecology — rule, sequence and
representation induction — because those were the only paths that called
``note_an_episode``.

That is a scope problem rather than a language problem. The developmental
language can express a change to any part of her; the developmental agent only
ever saw evidence from one library. It is a universal programming language
hooked to an optimiser that receives benchmark results from a single module,
and the optimiser is not wrong, it is uninformed.

So this is the intake. Two halves, because cost and failure arrive
differently.

**Failure already has a universal sink.** Every subsystem in the tree reports
degradations to one tracker. Attaching there makes the failure half of the
ledger organism-wide with no new seam anywhere — perception, retrieval,
consolidation, routing, recurrence, planning, motor, social, verification and
governance all already speak into it, and none of them had to be edited to
start saying something the developmental policy could read.

**Cost has to be reported by whoever spends it.** ``while_doing`` is what a
subsystem wraps its work in: it times the work, notes whether anything came
of it, and writes one episode. Deliberately trivial, because an intake that
costs anything to adopt does not get adopted, and the previous intake's whole
problem was that it was only ever adopted once.

What "universal" means here is checkable rather than asserted. The ecologies
below are the ones the developmental record has to hear from before its
optimisation loop covers the organism, and ``coverage`` says which of them
have actually spoken.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("Aura.Cognition.WhatItAllCosts")


def _checked_lock(name: str, *, reentrant: bool = False):
    """The repo's instrumented lock, so lockdep can see this one too.

    A raw threading.Lock is invisible to the ABBA detector, and a detector
    that only sees some of the locks reports clean while the deadlock it
    exists to find is being assembled out of the others.
    """

    from core.runtime.lockdep import checked_lock

    return checked_lock(name, reentrant=reentrant)


__all__ = [
    "WHAT_SHE_IS_MADE_OF",
    "coverage",
    "read_what_does_not_report",
    "hear_from_every_subsystem",
    "note_a_failure",
    "while_doing",
]


#: The parts of her a developmental policy has to have evidence about before
#: it is choosing for the organism rather than for one library. Written down
#: so "the intake is narrow" is a measurement rather than an impression.
WHAT_SHE_IS_MADE_OF: tuple[str, ...] = (
    "perception",
    "retrieval",
    "memory consolidation",
    "model routing",
    "latent recurrence",
    "tool planning",
    "motor policy",
    "social inference",
    "architecture",
    "verification",
    "governance",
    "rule induction",
)

#: Which subsystem name belongs to which ecology. Degradation records carry a
#: subsystem string chosen by whoever wrote the call site, so this is the
#: translation, and an unrecognised one is kept under its own name rather than
#: dropped — an ecology nobody anticipated is evidence too.
_WHICH_PART: dict[str, str] = {
    "perception": "perception",
    "screen_perception": "perception",
    "perception_daemon": "perception",
    "sensory_integration": "perception",
    "memory_facade": "retrieval",
    "memory_provider": "retrieval",
    "intentional_retrieval": "retrieval",
    "consolidation": "memory consolidation",
    "sleep_consolidation": "memory consolidation",
    "memory_consolidation": "memory consolidation",
    "llm_client": "model routing",
    "inference_gate": "model routing",
    "model_registry": "model routing",
    "latent_cortex": "latent recurrence",
    "recurrent_cortex": "latent recurrence",
    "capability_engine": "tool planning",
    "tool_registry": "tool planning",
    "skill_engine": "tool planning",
    "desktop_action_gateway": "motor policy",
    "action_executor": "motor policy",
    "interpersonal_store": "social inference",
    "interpersonal_observer": "social inference",
    "theory_of_mind": "social inference",
    "morphogenesis": "architecture",
    "service_container": "architecture",
    "verify": "verification",
    "invariants": "verification",
    "will": "governance",
    "governance": "governance",
    "values_engine": "governance",
}


def which_part(subsystem: str) -> str:
    """Which ecology a subsystem belongs to, or its own name."""

    plain = str(subsystem or "").strip().lower()
    if plain in _WHICH_PART:
        return _WHICH_PART[plain]
    # A dotted name reports under its head: "memory.interpersonal_store.render"
    # is social inference, and losing it because of the suffix would make the
    # intake narrow again by accident.
    for part in plain.split("."):
        if part in _WHICH_PART:
            return _WHICH_PART[part]
    return plain or "unnamed"


_ATTACHED = [False]
_LOCK = _checked_lock("what_the_whole_organism_costs")
#: Ecologies that have actually reported. The measurement the module exists
#: for, and it starts empty on purpose.
_HEARD_FROM: dict[str, int] = {}


def _heard(part: str) -> None:
    with _LOCK:
        _HEARD_FROM[part] = _HEARD_FROM.get(part, 0) + 1


def note_a_failure(record: Any) -> None:
    """One subsystem's degradation, as developmental evidence. O(1).

    A failure is the cheapest developmental signal there is: it says a part
    of her spent something and got nothing, which is exactly the shape the
    policy prices changes in.
    """

    subsystem = str(getattr(record, "subsystem", "") or "")
    if not subsystem:
        return
    part = which_part(subsystem)
    severity = str(getattr(record, "severity", "") or "")
    if severity in {"debug"}:
        # Demoted lifecycle noise. Counting it would make the loudest
        # ecology the one with the chattiest logging.
        return
    try:
        from core.cognition.the_record_of_her_own_work import note_an_episode

        note_an_episode(
            part,
            route=None,
            walked=0,
            tried=str(getattr(record, "action", "") or "") or None,
            admitted=None,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("could not note a failure from %s: %s", subsystem, exc)
        return
    _heard(part)


def hear_from_every_subsystem() -> bool:
    """Attach the developmental record to the one sink they all report to."""

    with _LOCK:
        if _ATTACHED[0]:
            return False
        _ATTACHED[0] = True
    try:
        from core.runtime.errors import get_degradation_tracker

        return bool(get_degradation_tracker().add_listener(note_a_failure))
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("could not hear from every subsystem: %s", exc)
        with _LOCK:
            _ATTACHED[0] = False
        return False


@contextmanager
def while_doing(
    subsystem: str,
    what: str = "",
    *,
    route: str = "",
    used: tuple[str, ...] = (),
) -> Iterator[dict[str, Any]]:
    """Wrap work so the developmental record hears what it cost.

    Yields a small dict the caller may put ``admitted`` or ``walked`` into.
    Everything else is measured: how long it took, and whether it raised.

    Cost is reported in milliseconds where the caller does not count
    candidates, because the record prices everything in one unit and a
    subsystem that cannot count its own search can still count its clock.
    An episode with no cost at all is an episode the policy cannot rank.
    """

    part = which_part(subsystem)
    started = time.monotonic()
    said: dict[str, Any] = {"admitted": None, "walked": 0}
    failed = False
    try:
        yield said
    except BaseException:
        failed = True
        raise
    finally:
        spent = int((time.monotonic() - started) * 1000)
        try:
            from core.cognition.the_record_of_her_own_work import note_an_episode

            note_an_episode(
                f"{part}: {what}" if what else part,
                route=None if failed else (route or "an answer"),
                walked=max(int(said.get("walked") or 0), spent),
                used=used,
                admitted=said.get("admitted"),
                tried=route or what or part,
            )
            _heard(part)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("could not note what %s cost: %s", part, exc)


def read_what_does_not_report() -> tuple[str, ...]:
    """Take evidence from the parts that keep it rather than send it.

    Verification and governance both hold what they have found and neither
    pushes it anywhere: a verifier keeps its last report, and a value
    registry keeps its refusals. Neither is a degradation — a refused change
    is the governance working, and recording correct refusals as faults is
    the mistake a whole test file in this tree is named after — so they are
    read rather than waited for.
    """

    took: list[str] = []
    try:
        from core.verify.invariants import verify

        report = verify()
        checked = int(getattr(report, "checked", 0) or 0)
        if checked:
            _note("verification", spent=checked, worked=not report.violations)
            took.append("verification")
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("could not read the verifier: %s", exc)
    try:
        from core.governance.value_levels import registry

        held = registry()
        decided = len(held.refusals()) + len(held.changes())
        if decided:
            _note("governance", spent=decided, worked=bool(held.changes()))
            took.append("governance")
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("could not read the value registry: %s", exc)
    return tuple(took)


def _note(part: str, *, spent: int, worked: bool) -> None:
    try:
        from core.cognition.the_record_of_her_own_work import note_an_episode

        note_an_episode(
            part,
            route="an answer" if worked else None,
            walked=max(0, int(spent)),
            tried=part,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("could not note what %s found: %s", part, exc)
        return
    _heard(part)


def coverage(*, read_the_quiet_ones: bool = True) -> dict[str, Any]:
    """Which parts of her the developmental policy has evidence about.

    The measurement behind the criticism. A policy fed by one ecology is
    narrow however general its language, and this says how narrow, in the
    only terms that can be argued with: which parts have spoken, and how
    often.
    """

    if read_the_quiet_ones:
        read_what_does_not_report()
    with _LOCK:
        heard = dict(_HEARD_FROM)
    covered = [part for part in WHAT_SHE_IS_MADE_OF if heard.get(part)]
    return {
        "declared": list(WHAT_SHE_IS_MADE_OF),
        "heard_from": sorted(heard),
        "covered": covered,
        "missing": [part for part in WHAT_SHE_IS_MADE_OF if part not in covered],
        "share": round(len(covered) / len(WHAT_SHE_IS_MADE_OF), 3),
        "episodes": sum(heard.values()),
    }


def forget_what_was_heard() -> None:
    """Used by tests, and by nothing else."""

    with _LOCK:
        _HEARD_FROM.clear()
