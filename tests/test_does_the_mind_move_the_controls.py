"""What the uncalibrated control policy actually does, swept exhaustively.

LIVE_MIND_CONTROL_POLICY says it is a hand-tuned heuristic and that
LIVE_MIND_CONTROL_POLICY_CALIBRATED is False. That is honest and it is not
evidence. An external review made the point precisely: the code proves
internal state CAN actuate model computation; it does not prove this mapping
is beneficial, learned, or even discriminative.

Discrimination is the part that can be settled offline, and the one that would
embarrass the claim most if it failed — a policy returning the same
temperature for every mind state would still be causal and would still carry
no information.

The findings, held down here so they cannot quietly change:

* `clean_user_surface_steering_alpha` takes one value across all 5,760 states.
  A declared control that never moves.
* Nociceptive pressure accounts for most of the variance in every control
  that does move. This is close to being a pain policy.
* `integration` moves top_p and nothing else, and `dominant_intensity` moves
  almost nothing, though the mapping reads both.
"""
from __future__ import annotations

import pytest

from core.brain.does_the_mind_move_the_controls import (
    how_much_the_mind_moves_the_controls,
    the_states_it_reads,
)


@pytest.fixture(scope="module")
def swept():
    return how_much_the_mind_moves_the_controls()


def _control(swept, name):
    return next(one for one in swept["controls"] if one["control"] == name)


# ---------------------------------------------------------------- the sweep


def test_the_sweep_covers_the_readings_the_policy_consults(swept):
    """A sweep over inputs it ignores measures the sweep."""
    assert set(the_states_it_reads()) == {
        "dominant_intensity", "curiosity", "pain", "integration", "self_presence",
    }
    assert swept["states_swept"] == 5760


def test_the_sweep_is_the_same_twice(swept):
    """Deterministic and exhaustive: nothing in it could differ between runs."""
    again = how_much_the_mind_moves_the_controls()
    assert again["controls"] == swept["controls"]


def test_the_policy_still_says_it_is_not_calibrated(swept):
    assert swept["calibrated"] is False
    assert swept["policy"] == "hand_tuned_heuristic.v1"


def test_the_result_says_what_it_does_not_show(swept):
    """A caveat beside a number gets read separately from it."""
    assert "held-out set" in swept["what_this_does_not_show"]


# ------------------------------------------------------------- what moves


def test_temperature_and_top_p_do_move_with_her_state(swept):
    """The mechanism is real: 44 and 38 distinct values across the space."""
    assert _control(swept, "temperature")["distinct"] > 10
    assert _control(swept, "top_p")["distinct"] > 10


def test_a_declared_control_that_never_moves_is_named(swept):
    """One value across 5,760 states. Causal in shape, constant in fact."""
    assert "clean_user_surface_steering_alpha" in swept["controls_that_never_move"]
    assert _control(swept, "clean_user_surface_steering_alpha")["distinct"] == 1


def test_the_policy_does_not_yet_discriminate_on_every_control(swept):
    """False because of the constant one. A finding, not a passing grade."""
    assert swept["discriminates"] is False


# --------------------------------------------------------- what drives it


@pytest.mark.parametrize("control", ["temperature", "top_p"])
def test_pain_accounts_for_most_of_the_variance(swept, control):
    """This is close to being a pain policy, and that is worth knowing."""
    explained = _control(swept, control)["explained_by"]
    assert explained["pain"] > 0.5
    assert explained["pain"] == max(explained.values())


def test_integration_moves_top_p_and_nothing_else(swept):
    assert _control(swept, "top_p")["explained_by"]["integration"] > 0.05
    assert _control(swept, "temperature")["explained_by"]["integration"] == 0.0


def test_the_dominant_affects_intensity_barely_moves_anything(swept):
    """The mapping reads it. Almost nothing follows from it."""
    for control in ("temperature", "top_p"):
        assert _control(swept, control)["explained_by"]["dominant_intensity"] < 0.01


def test_every_share_is_a_fraction(swept):
    for control in swept["controls"]:
        for share in control["explained_by"].values():
            assert 0.0 <= share <= 1.0


def test_the_sweep_is_in_the_health_report():
    """Through the registry: core/runtime may not import core.brain."""
    import core.brain.cognitive_engine  # noqa: F401 — importing registers it
    from core.runtime.health_contract import runtime_health_report

    block = runtime_health_report()["integrity"]["the_control_policy"]
    assert set(block) >= {"policy", "calibrated", "controls_that_never_move"}


def test_health_reads_it_through_the_registry_and_not_by_importing():
    """A health block that needed that edge is a layering violation dressed up."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "core" / "runtime" / "health_contract.py"
    ).read_text("utf-8")
    assert "does_the_mind_move_the_controls" not in source
    assert 'get_runtime_service("the_control_policy_sweep"' in source
