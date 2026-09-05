"""Structural invariants for an identity held as its practices.

One check, and it is the architectural claim of
:mod:`core.identity.constitutive_identity` rather than a guard on its
arithmetic. If declaring an identity could establish it, the whole design
collapses into the essentialism it was built against, and — worse for a
running system — it becomes a layer that manufactures the state it then
reports. Nothing downstream could tell the manufactured reading from a
measured one.

The check is on the import graph rather than on behaviour, because a rule
about discipline that depends on discipline is not a rule.
"""

from __future__ import annotations

from collections.abc import Iterator

from core.verify.invariants import Severity, Violation, invariant

#: Paths that generate or hold text about the self. A constitutive identity
#: must be computable with every one of them absent.
_FORBIDDEN = ("core.brain", "core.llm", "llm.", "interface.", "core.language")


@invariant(
    "identity.constitution_is_one_way",
    scope="identity",
    owner="core/identity/constitutive_identity.py",
    description="coherence is computed from enactments and never from a declaration",
)
def _label_never_writes_coherence() -> Iterator[Violation]:
    """Practices cause the reading. A claim about the identity never does."""
    import sys
    from pathlib import Path

    module = sys.modules.get("core.identity.constitutive_identity")
    path = Path(getattr(module, "__file__", "") or "") if module else None
    if path is None or not path.is_file():
        return
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in source.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        for forbidden in _FORBIDDEN:
            if forbidden in stripped:
                yield Violation(
                    subject="core/identity/constitutive_identity.py",
                    message=f"reaches a language path: {stripped[:80]}",
                    remedy=(
                        "compute coherence from the enactment record; a label may "
                        "be recorded by declare() and must never enter the "
                        "arithmetic"
                    ),
                    severity=Severity.ERROR,
                )


@invariant(
    "identity.coherence_carries_its_null",
    scope="identity",
    owner="core/identity/constitutive_identity.py",
    description="every coherence reading reports the floor it has to clear",
)
def _reading_carries_its_floor() -> Iterator[Violation]:
    """A number with no null beside it invites reading the null as a result.

    Unrelated phases already give a nonzero order parameter, so a coherence
    of 0.4 over four practices is the null rather than weak coherence. The
    field is on the frozen dataclass, so this checks the shape rather than
    any particular value.
    """
    from core.identity.constitutive_identity import Coherence

    for required in ("incoherent_floor", "n_active", "k_critical"):
        if required not in getattr(Coherence, "__annotations__", {}):
            yield Violation(
                subject="core.identity.constitutive_identity.Coherence",
                message=f"a reading no longer carries {required}",
                remedy="keep the null and the threshold on the reading itself",
                severity=Severity.ERROR,
            )
