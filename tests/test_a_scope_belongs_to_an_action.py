"""A step is not authorised for what the task is authorised for.

LIVE 2026-08-26: "make a file on my Desktop called aura_note.txt" was refused
with standing_authority_effect_scope_mismatch — the lease held
'desktop_file_io' and the call derived 'foreground_desktop_control', which is
the TASK's declared scope arriving through the step context. She could not
write a file at all.

A scope belongs to an action. The task's scope is the widest thing in its
plan; a step that types a key is foreground control and a step that writes a
file is file io, and each step's lease is issued for its own. Inherited, the
parent's value is presented against the child's lease and the two can never
agree.
"""
from __future__ import annotations

import pytest

from core.executive.execution_policy import resolve_execution_effect_scope
from core.skills.desktop_task import DesktopTaskSkill


def test_a_step_does_not_inherit_the_tasks_scope():
    child = DesktopTaskSkill._child_step_context(
        {
            "effect_scope": "foreground_desktop_control",
            "risk_level": "medium",
            "origin": "desktop_ui",
            "objective": "write a file",
        }
    )
    assert "effect_scope" not in child
    assert "risk_level" not in child
    # What identifies the turn is kept.
    assert child["origin"] == "desktop_ui"
    assert child["objective"] == "write a file"


def test_a_step_does_not_inherit_the_tasks_expectations_either():
    """The rule that was already here, still here."""
    child = DesktopTaskSkill._child_step_context(
        {"acceptance_criteria": ["the file exists"], "origin": "desktop_ui"}
    )
    assert "acceptance_criteria" not in child


@pytest.mark.parametrize(
    ("action", "scope"),
    [
        ("write_text_file", "desktop_file_io"),
        ("move_file", "desktop_file_io"),
        ("type", "foreground_desktop_control"),
        ("click", "foreground_desktop_control"),
        ("read_screen_text", "read_only"),
    ],
)
def test_each_action_resolves_to_its_own_scope(action, scope):
    assert resolve_execution_effect_scope("computer_use", {"action": action}) == scope


def test_a_plan_is_governed_by_the_widest_thing_in_it():
    """The task's own scope is still the widest step, which is what makes
    inheriting it wrong for the narrower ones."""
    plan = {"steps": [{"action": "open_app"}, {"action": "write_text_file"}]}
    assert resolve_execution_effect_scope("desktop_task", plan) == "desktop_file_io"
