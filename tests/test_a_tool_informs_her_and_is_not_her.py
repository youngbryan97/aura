"""A tool informs her and is not her; what she does with it is hers.

Bryan, on where he stops: a calculator is not him, and the page he writes is
his because he chose and combined what went on it. Her capability beliefs were
keyed on the tool she used, so a search engine that worked made her better at
searching and one that broke made her worse, whoever had chosen well. See
core/agency/intention_loop.py `_record_authorship`.
"""

from __future__ import annotations

import pytest

from core.agency import authorship
from core.agency.intention_loop import IntentionLoop

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_ledger():
    authorship.reset_agency_ledger_for_test()
    yield
    authorship.reset_agency_ledger_for_test()


def _act(loop: IntentionLoop, intention: str, tool: str, *, tool_worked: bool, met: bool) -> None:
    identifier = loop.intend(intention=intention, drive="curiosity", expected_outcome="the papers are found")
    loop.record_action(identifier, tool_name=tool, args={}, result="ok" if tool_worked else "error",
                       success=tool_worked, duration_ms=1.0)
    outcome = "the papers are found" if met else "the papers are found — it did not"
    loop.observe(identifier, observation="the papers are found", actual_outcome=outcome)


def test_her_capability_is_the_act_and_the_tool_is_the_worlds(tmp_path) -> None:
    loop = IntentionLoop(db_path=str(tmp_path / "intentions.db"))
    _act(loop, "Research sleep and memory", "web_search", tool_worked=True, met=True)
    ledger = authorship.get_agency_ledger()
    assert "research" in ledger.by_capability, "her act was not credited to her"
    assert "web_search" not in ledger.by_capability, "the tool was credited as her capability"
    assert ledger.acted == 1 and ledger.observed == 1
    assert ledger.last_event is not None and ledger.last_event.actor == authorship.SELF


def test_a_broken_tool_is_evidence_about_the_tool(tmp_path) -> None:
    loop = IntentionLoop(db_path=str(tmp_path / "intentions.db"))
    _act(loop, "Research sleep and memory", "web_search", tool_worked=False, met=False)
    ledger = authorship.get_agency_ledger()
    assert ledger.observed == 1 and ledger.observed_succeeded == 0
    assert ledger.by_capability["research"] == [1, 0], "the intention she chose to rely on it for is hers"


def test_a_tool_named_for_her_act_is_her_hand(tmp_path) -> None:
    loop = IntentionLoop(db_path=str(tmp_path / "intentions.db"))
    _act(loop, "write the notes file", "write_file", tool_worked=True, met=True)
    ledger = authorship.get_agency_ledger()
    assert ledger.observed == 0 and ledger.by_capability == {"write_file": [1, 1]}


def test_her_own_hand_is_not_a_second_actor(tmp_path) -> None:
    """The subject driver's acts are her own writes: the tool is named for the act."""
    loop = IntentionLoop(db_path=str(tmp_path / "intentions.db"))
    _act(loop, "append_log for turn 3: notes", "append_log", tool_worked=True, met=True)
    ledger = authorship.get_agency_ledger()
    assert ledger.observed == 0 and ledger.by_capability == {"append_log": [1, 1]}
