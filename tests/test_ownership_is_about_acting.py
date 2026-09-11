"""Ownership has to be about acting, not about one pathway.

The experiment runs two arms whose worlds end in the same state — the same file
holding the same bytes, the same fact recorded, the same goal text — differing
only in who is named as having caused it. Run six times on whichever action the
state happened to pick, that is one pathway exercised six times, and a report
saying ownership holds would be saying it about `write_notes`.

The four things she can do are cycled across the trials now, and the divergence
is reported per kind against its own same-arm floor.
"""

from __future__ import annotations

from core.subject.agency import AgencyReport
from core.subject.driver import SubjectRuntime


def _report(**kwargs) -> AgencyReport:
    base = dict(
        self_to_action=0.2,
        self_to_action_floor=0.0,
        outcome_to_self=0.8,
        outcome_to_self_floor=0.03,
        action_text_changed=False,
        trials=4,
    )
    base.update(kwargs)
    return AgencyReport(**base)


def test_one_pathway_is_not_generalisation() -> None:
    report = _report(
        ownership_by_action={"write_notes": 0.8},
        ownership_floor_by_action={"write_notes": 0.03},
    )
    assert not report.ownership_generalises


def test_two_pathways_each_over_their_own_floor_is() -> None:
    report = _report(
        ownership_by_action={"write_notes": 0.8, "append_log": 0.6},
        ownership_floor_by_action={"write_notes": 0.03, "append_log": 0.02},
    )
    assert report.ownership_generalises


def test_a_pathway_that_does_not_clear_its_own_floor_does_not_count() -> None:
    report = _report(
        ownership_by_action={"write_notes": 0.8, "read_room": 0.01},
        ownership_floor_by_action={"write_notes": 0.03, "read_room": 0.05},
    )
    assert not report.ownership_generalises


def test_the_experiment_cycles_every_action_she_has() -> None:
    import inspect

    from core.subject import agency

    source = inspect.getsource(agency.run_agency)
    assert "SubjectRuntime.ACTIONS" in source
    assert "trial % len(kinds)" in source
    assert len(SubjectRuntime.ACTIONS) >= 4


def test_holding_her_to_an_action_overrides_what_she_would_have_picked() -> None:
    """Both arms of an ownership trial must do the same thing, or the
    difference between them is what they did rather than who did it."""
    runtime = SubjectRuntime.__new__(SubjectRuntime)
    runtime.forced_action = "append_log"
    assert runtime._chosen_action() == "append_log"


def test_the_record_says_which_action_it_was() -> None:
    import inspect

    source = inspect.getsource(SubjectRuntime._act)
    assert '"kind": kind' in source
