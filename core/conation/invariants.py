"""Structural invariants for the conative organ.

These sit next to what they protect. Each exists because there is a specific
way a motivational system can quietly become dishonest, and because the
dishonest version looks healthier from the outside than the honest one.

1. **Text never writes motivation.** The whole architecture rests on state
   causing the report rather than the report causing the state. A language
   layer that can write conative values manufactures the motivations it then
   describes, and no reader downstream can tell which happened. This is
   enforced at the import graph, because a rule about discipline that depends
   on discipline is not a rule.

2. **A magnitude implies evidence.** An origin reporting a number must be able
   to name what was measured. The failure this prevents is the one CP126
   named: an unavailable channel emitting plausible values that nothing can
   distinguish from measurements.

3. **Borrowed value keeps its source.** A vicarious transfer without a named
   agent is exactly the invisible borrowing the origin exists to make
   visible. Once the source is gone, a borrowed want and an original one are
   the same number forever.

4. **A refused act carries no value.** The enactive gate returns zero on
   refusal. If a refusal ever came back with a magnitude, a caller that read
   only the number would act on a decision that had already been made against
   it.

5. **The vector's shape is fixed.** A learned weight vector indexes into
   ``VECTOR_FIELDS`` by position. If the state's vector and that tuple ever
   disagree in length, every learned weight is pointing at the wrong field
   and the arbitration is silently scrambled.
"""

from __future__ import annotations

from collections.abc import Iterator

from core.conation.origins import REQUIRED_TOPOLOGY, ValueOrigin
from core.conation.state import VECTOR_FIELDS
from core.verify.invariants import Severity, Violation, invariant

#: Modules a motivational state must never be produced from. Text generation
#: is the one that matters; the rest are the paths that reach it.
_FORBIDDEN_IMPORTS = ("core.brain", "core.llm", "llm.", "interface.")


def _engine() -> object | None:
    try:
        from core.conation.engine import _ENGINE

        return _ENGINE
    except (ImportError, RuntimeError, OSError, ValueError, TypeError):
        return None


@invariant(
    "conation.text_never_writes_motivation",
    scope="conation",
    owner="core/conation/engine.py",
    description="no conation module imports a text-generation path",
)
def _no_language_writeback() -> Iterator[Violation]:
    """The load-bearing one. State causes the report, never the reverse."""
    import sys
    from pathlib import Path

    package = sys.modules.get("core.conation")
    if package is None:
        return
    root = Path(getattr(package, "__file__", "") or "").parent
    if not root.is_dir():
        return

    for module in root.glob("*.py"):
        try:
            source = module.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in source.splitlines():
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            for forbidden in _FORBIDDEN_IMPORTS:
                if forbidden in stripped:
                    yield Violation(
                        subject=f"core/conation/{module.name}",
                        message=f"imports a text-generation path: {stripped[:80]}",
                        remedy=(
                            "compute the motivational state from measured runtime "
                            "state; a language layer may render it afterwards and "
                            "must never write it back"
                        ),
                        severity=Severity.CRITICAL,
                    )


@invariant(
    "conation.magnitude_implies_evidence",
    scope="conation",
    owner="core/conation/origins.py",
    description="every origin reporting a magnitude names what it measured",
)
def _magnitude_has_evidence() -> Iterator[Violation]:
    engine = _engine()
    last = getattr(engine, "_last_state", None) if engine is not None else None
    if last is None:
        return
    for origin, reading in last.readings.items():
        if reading.available and not str(reading.evidence).strip():
            yield Violation(
                subject=f"conation/{origin}",
                message="reported a magnitude with no evidence string",
                remedy="name the measurement, or report the origin unavailable",
            )
        if not reading.available and reading.magnitude != 0.0:
            yield Violation(
                subject=f"conation/{origin}",
                message=f"unavailable origin carries magnitude {reading.magnitude}",
                remedy="an origin without evidence contributes nothing",
            )


@invariant(
    "conation.borrowed_value_keeps_its_source",
    scope="conation",
    owner="core/conation/vicarious.py",
    description="every mimetic transfer names the agent it was borrowed from",
)
def _transfers_name_their_source() -> Iterator[Violation]:
    engine = _engine()
    vicarious = getattr(engine, "vicarious", None) if engine is not None else None
    transfers = getattr(vicarious, "_transfers", None)
    if not transfers:
        return
    for transfer in transfers:
        if not str(getattr(transfer, "agent", "") or "").strip():
            yield Violation(
                subject=f"conation/vicarious/{transfer.target}",
                message="a value transfer has no source agent",
                remedy=(
                    "refuse anonymous valuations at observe_valuation; a transfer "
                    "nobody can audit is the invisible borrowing this origin exists "
                    "to prevent"
                ),
                severity=Severity.CRITICAL,
            )


@invariant(
    "conation.refusal_carries_no_value",
    scope="conation",
    owner="core/conation/enactive.py",
    description="a refused act on another mind reports zero",
)
def _refusals_are_zero() -> Iterator[Violation]:
    engine = _engine()
    last = getattr(engine, "_last_state", None) if engine is not None else None
    if last is None or not last.refusals:
        return
    magnitude = last.magnitude_of(ValueOrigin.ENACTIVE)
    if magnitude > 0.0:
        yield Violation(
            subject=f"conation/enactive/{last.incentive_key}",
            message=f"refused ({', '.join(last.refusals)}) yet carries {magnitude}",
            remedy="the gate must zero the reading and mark it unavailable",
            severity=Severity.CRITICAL,
        )


@invariant(
    "conation.social_origins_carry_their_topology",
    scope="conation",
    owner="core/conation/origins.py",
    description="vicarious value flows inward and enactive value outward",
)
def _social_topology_holds() -> Iterator[Violation]:
    engine = _engine()
    last = getattr(engine, "_last_state", None) if engine is not None else None
    if last is None or last.dominant_origin not in REQUIRED_TOPOLOGY:
        return
    expected = REQUIRED_TOPOLOGY[last.dominant_origin]
    if last.topology is not expected:
        yield Violation(
            subject=f"conation/{last.incentive_key}",
            message=f"{last.dominant_origin} reported topology {last.topology}",
            remedy=f"a {last.dominant_origin} state is always {expected}",
        )


@invariant(
    "conation.vector_shape_is_fixed",
    scope="conation",
    owner="core/conation/state.py",
    description="the motivational vector matches VECTOR_FIELDS position for position",
)
def _vector_shape_holds() -> Iterator[Violation]:
    engine = _engine()
    last = getattr(engine, "_last_state", None) if engine is not None else None
    if last is None:
        return
    vector = last.motivational_vector()
    if len(vector) != len(VECTOR_FIELDS):
        yield Violation(
            subject="conation/motivational_vector",
            message=f"vector has {len(vector)} fields, contract has {len(VECTOR_FIELDS)}",
            remedy=(
                "VECTOR_FIELDS is append-only; a learned weight vector indexes it "
                "by position and a mismatch scrambles every weight"
            ),
            severity=Severity.CRITICAL,
        )
