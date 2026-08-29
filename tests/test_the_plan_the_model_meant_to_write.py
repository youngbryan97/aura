"""The planner read the first "[" to the last "]" and hoped.

A thinking block that mentions a bracket, prose around the JSON, a fenced block
with a comment above it, a trailing comma — each made the span wrong rather
than absent, so the failure arrived as "Expecting ',' delimiter" at a column
nobody could act on.

From the planner's own log on 2026-08-29: 24 empty responses, 15 "No JSON array
in response", and a run of delimiter errors at the same column. Every one fell
through to a single generic step, and 102 of 106 completed plans then read
"Success=True (1/1 steps)" whatever the goal was.
"""

from __future__ import annotations

import pytest

from core.utils.json_utils import extract_json_list

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("shape", "raw"),
    [
        ("plain", '[{"tool": "think", "description": "a"}]'),
        ("fenced", '```json\n[{"tool": "think", "description": "a"}]\n```'),
        (
            "after a thinking block that has a bracket in it",
            '<think>I should use [brackets] here</think>\n'
            '[{"tool": "think", "description": "a"}]',
        ),
        (
            "prose on both sides",
            'Here is the plan:\n[{"tool": "think", "description": "a"}]\nThat should do it.',
        ),
        ("a trailing comma", '[{"tool": "think", "description": "a",},]'),
        ("smart quotes", '[{“tool”: “think”, “description”: “a”}]'),
    ],
)
def test_the_plan_survives_how_the_model_wrote_it(shape: str, raw: str) -> None:
    steps = extract_json_list(raw)
    assert steps, shape
    assert steps[0]["tool"] == "think"


def test_one_step_written_as_an_object_is_a_plan_of_one() -> None:
    """A model asked for a list of one usually writes the one."""

    assert extract_json_list('{"tool": "web_search", "description": "look"}') == [
        {"tool": "web_search", "description": "look"}
    ]


def test_nothing_parseable_is_an_empty_plan_not_a_guess() -> None:
    assert extract_json_list("I could not make a plan.") == []
    assert extract_json_list("") == []
    assert extract_json_list(None) == []


def test_the_outer_array_wins_over_a_nested_one() -> None:
    steps = extract_json_list('[{"tool": "think", "args": {"items": [1, 2]}}]')
    assert len(steps) == 1
    assert steps[0]["args"]["items"] == [1, 2]


def test_the_planner_uses_the_shared_reader() -> None:
    """A third parser would drift from the two that already exist."""

    from pathlib import Path

    source = Path("core/agency/autonomous_task_engine.py").read_text(encoding="utf-8")
    assert "extract_json_list(raw)" in source
    assert 'raw.find("[")' not in source
