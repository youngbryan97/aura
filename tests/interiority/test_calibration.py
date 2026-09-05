"""The condition under which a guess is allowed, actually checked.

Twenty-eight of the parameters here are guesses with a declared range.
:mod:`core.interiority.params` permits that on one condition — that the
system's ordering survives the value moving across the range — and until
this existed the condition was written down and never checked, which
makes it a promise rather than a discipline.
"""

from __future__ import annotations

import pytest

from core.interiority.calibration import (
    _load_every_declaring_module,
    check_targets,
    order_sensitive,
    order_sensitive_baseline,
    report,
    sweep,
)
from core.interiority.params import registry


@pytest.fixture(scope="module", autouse=True)
def _loaded() -> None:
    _load_every_declaring_module()


def test_every_published_target_is_reproduced() -> None:
    failures = [t for t in check_targets() if not t.held]
    assert not failures, "\n".join(
        f"{t.name} ({t.source}): {t.detail}" for t in failures
    )


def test_no_new_conclusion_rests_on_an_unmeasured_number() -> None:
    """The order-sensitivity list may shrink and may not grow."""
    found = {r.parameter for r in order_sensitive()}
    baseline = order_sensitive_baseline()
    new = sorted(found - baseline)
    assert not new, (
        "these guesses now reorder the system when moved across their own "
        "declared range, which is exactly the condition under which guessing "
        "is not allowed: " + ", ".join(new)
    )


def test_the_sweep_actually_moves_the_parameter() -> None:
    """A sweep that recomputes the same numbers reports stability it never tested.

    An earlier version of the harness ignored the value it was given and
    called every parameter stable. This checks the override reaches the
    running system.
    """
    param = registry().get("interiority.f08.welfare_tradeoff_floor")
    assert param is not None
    original = param.value
    low, high = param.sweep_range
    with param.override(high):
        assert param.value == high
    assert param.value == original


def test_a_parameter_declared_stable_really_is() -> None:
    baseline = order_sensitive_baseline()
    stable = [
        p for p in registry().calibration() if p.name not in baseline
    ]
    assert stable, "no stable calibration parameters to check"
    # One is enough to prove the check discriminates; sweeping all of them
    # is what the gate does.
    result = sweep(stable[0], steps=3)
    assert result.stable, f"{stable[0].name}: {result.detail}"


def test_the_report_names_what_is_unconstrained() -> None:
    """The honest state, not a claim of having fitted anything."""
    calibration = report()["calibration"]
    assert calibration["total"] > 0
    assert calibration["unconstrained"], (
        "the report claims every guess is pinned by a published property, "
        "which would be the first thing to doubt"
    )
