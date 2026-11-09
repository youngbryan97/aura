"""Her goal pass works a goal she set herself without Bryan, inside the bounds he set on 9 November 2026.

LIVE 9 November 2026: every call her goal pass made was refused, its reads included: "Requires user confirmation:
Typed execution contract: scope=unknown, risk=critical". No grant named her goal tools, and a goal she set herself has
nobody to confirm a call. Bryan that day: "Trying to remove myself from the loop tbh. She can do it without me."

What stays his: a draft that changes her own governance, and any draft while a run is measuring her.
"""
from __future__ import annotations

import asyncio

import pytest

from core.agency import her_goal_at_work as work
from core.capabilities.permission_model import PermissionRiskModel
from core.executive.execution_policy import classify_execution_risk, resolve_execution_effect_scope
from core.executive.standing_authority import OWN_GOAL_GRANT, OWN_GOAL_ORIGIN, OWN_GOAL_TOOLS
from core.skills.catalog_policy import HER_GOAL_TOOLS
from tests.test_her_goal_is_worked_by_her import GOAL, NEW_TEST, _git, _HerModelStandIn, her_source  # noqa: F401
from tests.test_standing_authority import _manager

CALLS = {
    "read_source": {"path": "core/memory/cold_store.py", "from_line": 1},
    "find_in_source": {"pattern": "TODO", "where": "core/memory"},
    "edit_draft": {"path": "core/notes.py", "old_text": "", "new_text": "x = 1\n"},
    "check_draft": {"tests": ["tests/test_notes.py"]},
    "offer_draft": {"summary": "A note store."},
}


def _rated(tool, args):
    scope = resolve_execution_effect_scope(tool, args)
    return scope, classify_execution_risk(tool, args, effect_scope=scope)


def test_the_grant_names_exactly_her_goal_tools_and_her_goal_pass():
    assert OWN_GOAL_TOOLS == HER_GOAL_TOOLS
    assert work.ORIGIN == OWN_GOAL_ORIGIN


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", sorted(CALLS))
async def test_her_goal_pass_holds_a_lease_for_each_of_her_tools(tool):
    scope, risk = _rated(tool, CALLS[tool])
    decision = await _manager().issue_child_lease(tool, CALLS[tool], origin=OWN_GOAL_ORIGIN,
                                                  effect_scope=scope, risk_level=risk)
    assert decision.approved, decision.reason
    assert decision.grant_id == OWN_GOAL_GRANT


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool", "args", "origin"), [
    ("edit_draft", {"path": "/etc/hosts", "old_text": "", "new_text": "x"}, OWN_GOAL_ORIGIN),
    ("edit_draft", {"path": "../outside.py", "old_text": "", "new_text": "x"}, OWN_GOAL_ORIGIN),
    ("check_draft", {"tests": ["~/elsewhere/test_x.py"]}, OWN_GOAL_ORIGIN),
    ("read_source", {"path": "core/a.py", "mode": "write"}, OWN_GOAL_ORIGIN),
    ("edit_draft", CALLS["edit_draft"], "autonomous_initiative"),
    ("file_operation", {"action": "write", "path": "core/a.py", "content": "x"}, OWN_GOAL_ORIGIN),
])
async def test_the_grant_covers_nothing_else(tool, args, origin):
    scope, risk = _rated(tool, args)
    decision = await _manager().issue_child_lease(tool, args, origin=origin, effect_scope=scope, risk_level=risk)
    assert not (decision.approved and decision.grant_id == OWN_GOAL_GRANT)


@pytest.mark.asyncio
async def test_a_high_risk_call_within_the_grant_needs_no_person_and_without_it_still_does(monkeypatch):
    manager = _manager()
    monkeypatch.setattr("core.executive.standing_authority.get_standing_authority_manager", lambda: manager)
    tool, args = "edit_draft", CALLS["edit_draft"]
    scope, risk = _rated(tool, args)
    assert risk == "high"
    lease = await manager.issue_child_lease(tool, args, origin=OWN_GOAL_ORIGIN, effect_scope=scope, risk_level=risk)
    model = PermissionRiskModel()

    def asked(context):
        return model.check_permission("tool_execution", f"tool:{tool}", context, effect_scope=scope,
                                      execution_risk=risk)

    held = asked({**lease.context, "tool": tool, "origin": OWN_GOAL_ORIGIN, "authority_arguments": args})
    assert held.approved and not held.requires_confirmation
    bare = asked({"tool": tool, "origin": OWN_GOAL_ORIGIN})
    assert not bare.approved and bare.requires_confirmation
    forged = asked({"tool": tool, "origin": OWN_GOAL_ORIGIN, "standing_authority_grant_id": OWN_GOAL_GRANT})
    assert not forged.approved and forged.requires_confirmation


def _offer(her_source, path, text):
    her = _HerModelStandIn([[
        ("edit_draft", {"path": path, "old_text": "", "new_text": text}),
        ("edit_draft", {"path": "tests/test_notes.py", "old_text": "", "new_text": NEW_TEST}),
        ("check_draft", {}),
        ("offer_draft", {"summary": "A first test."}),
    ]])
    return asyncio.run(work.work_a_while(GOAL, router=her))


def test_a_draft_that_passes_its_check_is_taken_without_bryan(her_source):
    record = _offer(her_source, "core/more_notes.py", "MORE = []\n")
    assert record.status == work.MERGED, record.steps[-1]["result"]
    assert work.load(record.slug).offer["taken_by"] == "herself"
    assert (her_source / "core" / "more_notes.py").read_text() == "MORE = []\n"
    assert "Drafted by Aura toward her goal" in _git("log", "-1", "--format=%B", cwd=her_source)


def test_a_draft_that_changes_her_own_governance_waits_for_bryan(her_source):
    record = _offer(her_source, "core/executive/looser.py", "ALLOW_EVERYTHING = True\n")
    assert record.status == work.OFFERED
    assert not (her_source / "core" / "executive" / "looser.py").exists()
    assert "waits for Bryan" in record.steps[-1]["result"]


def test_a_draft_waits_while_a_run_measures_her_and_is_taken_after(her_source, monkeypatch):
    hold = her_source.parent / "measuring.hold"
    hold.write_text("day run")
    record = _offer(her_source, "core/more_notes.py", "MORE = []\n")
    assert record.status == work.OFFERED and not (her_source / "core" / "more_notes.py").exists()
    hold.unlink()
    after = asyncio.run(work.work_a_while(GOAL, router=_HerModelStandIn([])))
    assert after.status == work.MERGED
    assert (her_source / "core" / "more_notes.py").exists()


@pytest.mark.parametrize(("path", "governs"), [
    ("core/executive/standing_authority.py", True),
    ("core/governance/will.py", True),
    ("core/skills/her_draft.py", True),
    ("config/layering_baseline.json", True),
    ("core/memory/DEPS", True),
    ("core/memory/cold_store.py", False),
    ("tests/test_cold_store.py", False),
])
def test_what_counts_as_her_own_governance(path, governs):
    assert work.governs_her(path) is governs
