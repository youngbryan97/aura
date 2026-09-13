"""A skill she dispatched that failed arrives in her percept stream as an error.

The affect phase maps an `error` percept to fear and frustration, and the
subject-core harness emits one when an action in its scratch world fails. No
production code emitted one, so the harness exercised a route she never takes
live. Action grounding is where a dispatched skill's failure is learned, inside
the response phase that holds the state, and that is where it is perceived now.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.phases.action_grounding import GroundingResult, perceive_failed_actions

ROOT = Path(__file__).resolve().parents[1]


class _World:
    def __init__(self) -> None:
        self.recent_percepts: list[dict] = []


def _result(*hits: dict) -> GroundingResult:
    result = GroundingResult(grounded_text="")
    result.marker_hits = list(hits)
    return result


def test_a_skill_that_ran_and_failed_arrives_as_an_error() -> None:
    world = _World()
    emitted = perceive_failed_actions(
        world,
        _result({"skill": "write_file", "status": "executed_failed", "ok": False, "error": "permission denied"}),
    )
    assert emitted == 1
    (percept,) = world.recent_percepts
    assert percept["type"] == "error"
    assert "write_file" in percept["content"] and "permission denied" in percept["content"]
    assert percept["source"] == "action_grounding"


def test_a_dispatch_that_raised_is_perceived_too() -> None:
    world = _World()
    assert perceive_failed_actions(world, _result({"skill": "search", "status": "dispatch_error", "error": "OSError()"})) == 1


@pytest.mark.parametrize("status", ["executed", "unverified", "unverified_no_engine"])
def test_a_success_or_a_marker_that_dispatched_nothing_is_not_an_error(status: str) -> None:
    world = _World()
    assert perceive_failed_actions(world, _result({"skill": "search", "status": status, "ok": status == "executed"})) == 0
    assert world.recent_percepts == []


def test_the_affect_phase_reads_an_error_percept() -> None:
    source = (ROOT / "core" / "phases" / "affect_update.py").read_text(encoding="utf-8")
    assert '"error"' in source


def test_the_response_phase_perceives_the_actions_it_grounded() -> None:
    source = (ROOT / "core" / "phases" / "response_generation_unitary.py").read_text(encoding="utf-8")
    grounded = source.index("grounding_res = await ground_response(")
    perceived = source.index("perceive_failed_actions(new_state.world, grounding_res)")
    assert perceived > grounded
