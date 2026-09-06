"""The protocol's own controls, held so a result can be read.

A protocol that cannot produce an informative negative is not a measurement.
Three properties make the run readable, and all three are checkable without
loading a model:

  * The null family is declared before the run, so a margin there is this
    measurement's own bias rather than a win.
  * A faculty arm whose lesion cannot fire is refused, not scored. The first
    version skipped an unregistered channel, so three ablation arms were
    byte-identical to intact and the run reported three deltas of exactly
    0.000 as though that were a finding.
  * A zero delta that IS arithmetic reads as NOT_MEASURED with the reason,
    because NOT_MEASURED and 0.000 are different readings and only one is
    true.
"""

from __future__ import annotations

import pytest

from tools.matched.run_matched_substrate import (
    ARMS,
    INTACT,
    MAX_TOKENS,
    NO_RECURRENT,
    SUBSTRATE,
    WHAT_EACH_ARM_REMOVES,
    _budgets,
    _faculty_reading,
    _prompt_for,
)
from tools.matched.tasks import THE_NULL_FAMILY, THE_POSITIVE_CONTROL, by_family


def test_every_arm_declares_the_same_budget():
    """Parity by construction, and then checked."""
    from core.evaluation.matched_budget import check_budget_parity

    assert check_budget_parity(_budgets()).matched


def test_the_null_family_is_declared_and_present():
    families = by_family()
    assert THE_NULL_FAMILY in families
    assert all(len(one.turns) == 1 for one in families[THE_NULL_FAMILY]), (
        "the null family must be single-turn, or context could legitimately help"
    )


def test_the_positive_control_needs_the_earlier_turn():
    families = by_family()
    assert THE_POSITIVE_CONTROL in families
    assert all(len(one.turns) >= 2 for one in families[THE_POSITIVE_CONTROL])


def test_a_zero_delta_that_cannot_be_measured_says_so():
    said = _faculty_reading({"delta_mean": 0.0, "separated": False}, NO_RECURRENT)
    assert said["outcome"] == "NOT_MEASURED"
    assert "cognitive engine" in said["why"]
    assert said["what_would_measure_it"]


def test_a_real_delta_is_reported_as_measured():
    said = _faculty_reading({"delta_mean": 0.2, "separated": True}, NO_RECURRENT)
    assert said["outcome"] == "MEASURED"
    assert said["delta_mean"] == 0.2


def test_only_the_prompt_differs_between_the_base_arms():
    """The base arms are the same model; what separates them is context."""
    from tools.matched.run_matched_substrate import BASE, SCAFFOLDED

    task = by_family()[THE_POSITIVE_CONTROL][0]
    history = [task.turns[0], "Noted."]
    bare = _prompt_for(BASE, task, history)
    assert task.turns[0] not in bare, "the base arm must not see the earlier turn"
    assert task.turns[0] in _prompt_for(INTACT, task, history)
    assert _prompt_for(SCAFFOLDED, task, history) != bare


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_is_declared_in_the_ablation_table(arm):
    """An arm missing from the table would silently ablate nothing."""
    assert arm in WHAT_EACH_ARM_REMOVES


def test_the_substrate_and_budget_are_named_in_one_place():
    assert "1.5B" in SUBSTRATE
    assert MAX_TOKENS > 0
