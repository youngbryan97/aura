"""A decision point that only ever answers one way is not deciding anything.

Two failures, the same defect from opposite sides. A gate nobody can pass takes
a working system and gates it into a coma; a gate nobody can fail is
decorative. Neither is findable in the source, because the code branches both
ways and only the traffic says which branch is real.

LIVE 2026-09-07: `/api/readyz` answered 503 for the whole of every turn while
the runtime answered perfectly; the prompt cache reported a miss on every turn
for the life of the process; the foreground non-parametric memory refused every
turn against a threshold nothing on this host will reach.

Counted and reported. Never enforced — a runtime that refuses to start because
a counter looks lopsided is the coma arriving by another route.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.verify.one_way_decisions import (
    ENOUGH_TO_BE_A_PATTERN,
    decision_census,
    one_way_decisions,
    record_decision,
    reset_decisions,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_decisions()
    yield
    reset_decisions()


def test_a_gate_that_never_admits_is_found() -> None:
    for _ in range(ENOUGH_TO_BE_A_PATTERN):
        record_decision("readyz", admitted=False, reason="conversation_lane_not_ready")
    found = {item.name: item for item in one_way_decisions()}
    assert found["readyz"].verdict == "never admits"
    assert found["readyz"].last_reason == "conversation_lane_not_ready"


def test_a_gate_that_never_refuses_is_found_too() -> None:
    """The decorative half costs nothing visible, which is why it survives."""
    for _ in range(ENOUGH_TO_BE_A_PATTERN):
        record_decision("always_yes", admitted=True)
    found = {item.name: item for item in one_way_decisions()}
    assert found["always_yes"].verdict == "never refuses"


def test_a_gate_that_decides_both_ways_is_not_a_finding() -> None:
    for index in range(ENOUGH_TO_BE_A_PATTERN * 2):
        record_decision("real_gate", admitted=index % 3 != 0)
    assert [item.name for item in one_way_decisions()] == []


def test_a_short_run_of_one_answer_is_a_tuesday() -> None:
    """Ten decisions going the same way is ordinary; a hundred is a mechanism."""
    for _ in range(ENOUGH_TO_BE_A_PATTERN - 1):
        record_decision("quiet", admitted=False)
    assert [item.name for item in one_way_decisions()] == []


def test_the_census_reports_everything_it_counted() -> None:
    record_decision("a", admitted=True)
    record_decision("a", admitted=False, reason="because")
    census = decision_census()
    assert census["a"]["admitted"] == 1
    assert census["a"]["refused"] == 1
    assert census["a"]["last_refusal_reason"] == "because"


def test_it_reports_and_never_enforces() -> None:
    """The whole point: this must not become the next thing that refuses."""
    source = Path("core/verify/one_way_decisions.py").read_text()
    assert "raise" not in source.replace("raises", ""), (
        "this module must never raise; it is the thing that watches for gates"
    )
    health = Path("core/runtime/health_contract.py").read_text()
    start = health.index('block["one_way_decisions"]')
    assert "healthy" not in health[start : start + 600], (
        "the census must not be allowed to flip the health verdict"
    )


def test_the_live_decision_points_are_wired() -> None:
    """A census nobody feeds reports nothing, forever."""
    for path, name in (
        ("interface/routes/system.py", '"readyz"'),
        ("interface/routes/chat.py", '"conversation_resume_handle"'),
        ("core/conversation/surface_disposition.py", '"tool_receipt.custody"'),
    ):
        source = Path(path).read_text()
        assert "record_decision" in source and name in source, path
