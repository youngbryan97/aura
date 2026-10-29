"""An arm that did not move the state her cortex reads cannot be a report of it.

The first three reports checks on seed 7 each moved her computed valence across
arms and read 0 of 24 anchors. None of them recorded what the steering hooks
were given, and the hooks were given 0.593 in every arm: the channel published
the substrate's valence neuron and an appraisal reached that neuron at eight
thousandths (7d60873f1). The run now records the valence activation per arm and
says whether the displacement arrived, at the bar the addendum of 28 September
fixed.
"""

from __future__ import annotations

import pytest

from tools.run_report_grounding import REACHED_HER, _reached_her_cortex


def test_no_readable_arm_is_not_a_measurement() -> None:
    out = _reached_her_cortex([{"raised": None, "lowered": None}])
    assert out["measured"] is False
    assert "no arm published" in out["why"]


def test_the_displacement_that_did_not_arrive_is_named() -> None:
    """The reading the first three checks would have printed."""
    out = _reached_her_cortex(
        [{"raised": 0.61543, "lowered": 0.60774}, {"raised": 0.6178, "lowered": 0.6173}]
    )
    assert out["measured"] is True
    assert out["reached"] is False
    assert out["mean_raised_minus_lowered"] == pytest.approx(0.004095, abs=1e-5)
    assert str(REACHED_HER) in out["why"]


def test_the_displacement_that_arrived_is_named_too() -> None:
    """And what the same arms read once the hooks were given her felt state."""
    out = _reached_her_cortex(
        [{"raised": 0.7227, "lowered": 0.5105}, {"raised": 0.7048, "lowered": 0.5271}]
    )
    assert out["reached"] is True
    assert out["mean_raised_minus_lowered"] == pytest.approx(0.19495, abs=1e-5)
    assert out["anchors"] == 2


def test_an_arm_missing_one_side_is_not_counted() -> None:
    out = _reached_her_cortex(
        [{"raised": 0.72, "lowered": None}, {"raised": 0.72, "lowered": 0.51}]
    )
    assert out["anchors"] == 1


def test_the_run_records_it_beside_the_ground() -> None:
    import inspect

    import tools.run_report_grounding as harness

    source = inspect.getsource(harness.main)
    assert '"reached_her_cortex"' in source
    assert "_steered_valence()" in source
    assert '"steered_valence"' in source


def test_an_arm_keeps_which_phase_gave_up() -> None:
    """Seed 89's run, 28 October: five arms ended with an answer held and
    unserved, and no file said which phase had given up."""
    from types import SimpleNamespace

    from tools.run_report_grounding import _phases_that_failed

    runtime = SimpleNamespace(
        failures={"UnitaryResponsePhase": 2, "AffectPhase": 1},
        failure_notes={
            "UnitaryResponsePhase": "TimeoutError: Foreground conversation lane produced only unsafe drafts",
            "AffectPhase": "RuntimeError: earlier",
        },
    )
    failed = _phases_that_failed(runtime, {"UnitaryResponsePhase": 1, "AffectPhase": 1})
    assert failed == {"UnitaryResponsePhase": "TimeoutError: Foreground conversation lane produced only unsafe drafts"}
    assert _phases_that_failed(SimpleNamespace(), {}) == {}
