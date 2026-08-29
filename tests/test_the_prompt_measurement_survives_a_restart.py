"""A measurement that cannot cross a restart is one that was never taken.

The section volatility table decides which parts of an assembled prompt are
stable, and a cached prefix is only worth anything if it is byte-identical —
measured live, the runtime was reusing 558 tokens of 27,298 and prefilling the
rest at about forty-six seconds a turn.

``load_volatility`` was written to take the table back at boot and nothing
called it. The writer ran every twenty observed prompts, so a session shorter
than that measured for nothing — and on this machine restarts are minutes
apart, which made that every session. LIVE 2026-08-29: the store did not exist
at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.brain.llm import context_budget

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _own_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Never the live store: this test writes, and that one is real."""

    monkeypatch.setattr(
        "core.runtime.state_ownership.state_root", lambda: tmp_path, raising=False
    )
    changed = dict(context_budget._CHANGED)
    seen = dict(context_budget._LAST_SEEN)
    taken = context_budget._taken_back
    context_budget._CHANGED.clear()
    context_budget._LAST_SEEN.clear()
    yield tmp_path
    context_budget._CHANGED.clear()
    context_budget._CHANGED.update(changed)
    context_budget._LAST_SEEN.clear()
    context_budget._LAST_SEEN.update(seen)
    context_budget._taken_back = taken


def test_what_one_run_measured_is_read_back_by_the_next(_own_state: Path) -> None:
    (_own_state / "section_volatility.json").write_text(
        json.dumps({"changed": {"[PRESENT MOMENT]": [40, 39], "[IDENTITY]": [40, 0]}})
    )
    context_budget._taken_back = False
    context_budget.observe_sections("head\n\n[PRESENT MOMENT]\nnow\n")
    assert context_budget._CHANGED["[IDENTITY]"] == [40, 0], "the table was not taken back"
    assert context_budget._CHANGED["[PRESENT MOMENT]"][0] >= 40


def test_it_is_read_once_and_does_not_undo_this_session(_own_state: Path) -> None:
    (_own_state / "section_volatility.json").write_text(
        json.dumps({"changed": {"[IDENTITY]": [40, 0]}})
    )
    context_budget._taken_back = False
    context_budget.observe_sections("head\n\n[IDENTITY]\naura\n")
    context_budget.observe_sections("head\n\n[IDENTITY]\naura\n")
    seen_now = context_budget._CHANGED["[IDENTITY]"][0]
    context_budget.observe_sections("head\n\n[IDENTITY]\naura\n")
    assert context_budget._CHANGED["[IDENTITY]"][0] > seen_now


def test_a_session_shorter_than_the_write_cadence_is_kept(_own_state: Path) -> None:
    """The cadence keeps writes off the answer path; exit keeps the session."""

    context_budget._taken_back = True
    # Twice: volatility is a comparison, so one prompt teaches nothing.
    context_budget.observe_sections("head\n\n[MEMORY]\nnothing\n")
    context_budget.observe_sections("head\n\n[MEMORY]\nsomething\n")
    assert context_budget._CHANGED, "nothing was measured"
    context_budget._keep_what_this_session_measured()
    stored = json.loads((_own_state / "section_volatility.json").read_text())
    assert "[MEMORY]" in stored["changed"]


def test_a_missing_store_is_not_an_error(_own_state: Path) -> None:
    context_budget._taken_back = False
    context_budget.observe_sections("head\n\n[MEMORY]\nnothing\n")
    context_budget.observe_sections("head\n\n[MEMORY]\nsomething\n")
    assert context_budget._CHANGED
