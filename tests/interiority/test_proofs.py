"""The three proofs every faculty has to survive.

These are not tests that the code runs. Each one is a claim about
causation that the package makes about itself, run against every
mechanism at once:

* a declared intervention produces the declared direction;
* the declared neutral world produces no activation;
* removing the faculty changes a quantity some other subsystem reads.

The third is the one that matters most and the one an emotion module
usually cannot pass. A faculty that survives ablation without moving
anything downstream is decorative, and the assertion names it rather
than averaging it into a pass.
"""

from __future__ import annotations

import pytest

from core.interiority.faculties import load_all
from core.interiority.faculty import registry
from core.interiority.proving import (
    ablation_report,
    counterfactual_report,
    null_report,
)


@pytest.fixture(scope="module", autouse=True)
def _loaded() -> None:
    load_all()


def test_every_faculty_is_registered() -> None:
    faculties = registry().all()
    assert len(faculties) == 43, f"expected 43 faculties, found {len(faculties)}"
    numbers = sorted(f.number for f in faculties)
    assert numbers == list(range(1, 44)), f"gaps or duplicates in numbering: {numbers}"


def test_declared_counterfactuals_hold() -> None:
    failures = [c for c in counterfactual_report() if not c.held]
    assert not failures, "\n".join(
        f"{c.faculty} :: {c.name} expected {c.expect} but got {c.detail}\n"
        f"    because: {c.because}"
        for c in failures
    )


def test_no_faculty_fires_in_its_own_null_world() -> None:
    failures = [n for n in null_report() if not n.held]
    assert not failures, "\n".join(
        f"{n.faculty} fired at {n.intensity:.4f} with nothing present "
        f"(tolerance {n.tolerance})"
        for n in failures
    )


def test_every_faculty_reaches_behaviour() -> None:
    decorative = [a for a in ablation_report() if not a.reaches_behaviour]
    assert not decorative, (
        "these faculties change nothing any subsystem reads when removed, "
        "which makes them descriptions rather than mechanisms: "
        + ", ".join(a.faculty for a in decorative)
    )


def test_ablation_deltas_are_attributable() -> None:
    """A faculty's effect must be traceable to a named quantity."""
    for result in ablation_report():
        moved = {k: v for k, v in result.deltas.items() if abs(v) > 1e-9}
        assert moved or result.unblocked or result.unheld, result.faculty
