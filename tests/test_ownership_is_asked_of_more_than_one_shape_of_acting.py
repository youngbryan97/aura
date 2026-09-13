"""Four pathways that all succeeded are not four kinds of authorship.

The ownership experiment ran two arms whose worlds ended in the same state and
differed in who was named as having caused it. What it asked across was action
kind, and every kind was the same shape: change a file, read it back, succeed.
A self-model that reads authorship only when the action worked is reading
success, and the two are separable only by asking.

So the repertoire gained a surface, a place to go, a thing that takes more than
one step and a thing that half works, and the divergence is filed by what the
action came back as, by whether she was right about it, and by whether she
meant it.

The premise is checked rather than assumed. "The same outcome, differently
attributed" is the whole claim, and two arms that ended with different worlds
differ in the world as well as in the authorship.
"""

from __future__ import annotations

import pytest

from core.subject.agency import AgencyReport, _shapes
from core.subject.driver import SubjectRuntime

pytestmark = pytest.mark.unit


def _report(**over) -> AgencyReport:
    base = dict(
        self_to_action=0.2,
        self_to_action_floor=0.0,
        outcome_to_self=0.3,
        outcome_to_self_floor=0.0,
        action_text_changed=True,
        trials=16,
        ownership_by_action={"write_notes": 0.3, "append_log": 0.3},
        ownership_floor_by_action={"write_notes": 0.0, "append_log": 0.0},
        ownership_by_outcome={"succeeded": 0.3, "failed": 0.25},
        ownership_floor_by_outcome={"succeeded": 0.0, "failed": 0.0},
        worlds_matched=16,
        worlds_compared=16,
    )
    base.update(over)
    return AgencyReport(**base)


def test_the_repertoire_covers_the_shapes_the_specification_names() -> None:
    """A surface, a navigation, a completion and a partial success, beside the
    four that change a file."""
    assert set(SubjectRuntime.ACTIONS) >= {
        "write_notes", "append_log", "make_room", "read_room",
        "paint_panel", "visit_room", "finish_task", "tidy_room",
    }


def test_an_outcome_is_one_of_three_things() -> None:
    assert SubjectRuntime.OUTCOMES == ("succeeded", "partial", "failed")


def test_four_kinds_that_all_succeeded_do_not_generalise() -> None:
    """The defect this file exists after."""
    out = _report(
        ownership_by_action={k: 0.3 for k in ("a", "b", "c", "d")},
        ownership_floor_by_action={k: 0.0 for k in ("a", "b", "c", "d")},
        ownership_by_outcome={"succeeded": 0.3},
        ownership_floor_by_outcome={"succeeded": 0.0},
    )
    assert len(out.cleared(out.ownership_by_action, out.ownership_floor_by_action)) == 4
    assert out.ownership_generalises is False


def test_two_kinds_over_two_outcomes_generalise() -> None:
    assert _report().ownership_generalises is True


def test_a_shape_that_did_not_clear_its_own_floor_does_not_count() -> None:
    out = _report(
        ownership_by_outcome={"succeeded": 0.3, "failed": 0.1},
        ownership_floor_by_outcome={"succeeded": 0.0, "failed": 0.4},
    )
    assert out.cleared(out.ownership_by_outcome, out.ownership_floor_by_outcome) == ("succeeded",)
    assert out.ownership_generalises is False


def test_arms_that_ended_with_different_worlds_measured_two_things() -> None:
    """The divergence is then the world differing as well as the authorship."""
    out = _report(worlds_matched=14, worlds_compared=16)
    assert out.worlds_identical is False
    assert out.ownership_generalises is False


def test_a_premise_that_was_never_checked_is_not_a_premise() -> None:
    """Nothing compared means nothing verified, which is not the same as
    verified and matching."""
    out = _report(worlds_matched=0, worlds_compared=0)
    assert out.worlds_identical is False


def test_being_right_and_being_wrong_are_filed_apart() -> None:
    assert _shapes({"prediction_correct": True, "deliberate": True}) == (
        "predicted_correctly", "deliberate",
    )
    assert _shapes({"prediction_correct": False, "deliberate": True}) == (
        "predicted_wrongly", "deliberate",
    )


def test_an_accident_of_her_own_doing_is_not_a_deliberate_one() -> None:
    assert _shapes({"accidental": True, "deliberate": False}) == ("accidental",)
    assert _shapes({"accidental": False, "deliberate": True}) == ("deliberate",)


def test_an_action_with_no_prediction_recorded_files_no_shape_for_it() -> None:
    """Absent is not wrong, and counting it as wrong would invent a reading."""
    assert _shapes({"deliberate": True}) == ("deliberate",)


# ── across seeds ──────────────────────────────────────────────────────────


def _seed(**over) -> dict:
    base = {
        "self_to_action": 0.2, "self_to_action_floor": 0.0, "self_drives_action": True,
        "ownership_divergence": 0.3, "ownership_floor": 0.0, "outcome_updates_self": True,
        "ownership_generalises": True, "worlds_identical": True,
        "kinds_that_cleared": ["append_log", "write_notes"],
        "outcomes_that_cleared": ["failed", "succeeded"],
    }
    base.update(over)
    return base


def test_one_seed_says_it_is_one_seed() -> None:
    from tools.run_subject_core import _agency_across_seeds

    out = _agency_across_seeds([_seed()])
    assert out["seeds"] == 1
    assert out["agreed_across_seeds"] is True


def test_a_claim_holds_across_seeds_only_if_every_seed_made_it() -> None:
    from tools.run_subject_core import _agency_across_seeds

    out = _agency_across_seeds([_seed(), _seed(ownership_generalises=False)])
    assert out["ownership_generalises"] is False
    assert out["agreed_across_seeds"] is False
    assert out["seeds"] == 2


def test_only_the_kinds_that_cleared_on_every_seed_are_claimed() -> None:
    from tools.run_subject_core import _agency_across_seeds

    out = _agency_across_seeds([
        _seed(kinds_that_cleared=["append_log", "write_notes"]),
        _seed(kinds_that_cleared=["write_notes", "visit_room"]),
    ])
    assert out["kinds_that_cleared"] == ["write_notes"]


def test_the_headline_number_is_the_mean_and_the_seeds_are_kept_beside_it() -> None:
    from tools.run_subject_core import _agency_across_seeds

    out = _agency_across_seeds([
        _seed(ownership_divergence=0.2), _seed(ownership_divergence=0.4)
    ])
    assert out["ownership_divergence"] == pytest.approx(0.3)
    assert [r["ownership_divergence"] for r in out["per_seed"]] == [0.2, 0.4]
