"""A program over tools, a repository surface, specialists, and scoped secrets.

Cards A1.1-A1.19, A2.1-A2.4, A2.8-A2.11, A2.13, A2.16, A2.17, A3.1-A3.15.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.cognition.tool_plan import (
    Executor,
    Op,
    Plan,
    PlanRefused,
    Step,
)
from core.runtime.governed_subagent import (
    BudgetExhausted,
    Conductor,
    Finding,
    SubagentSpec,
)
from core.security.credential_broker import CredentialBroker, CredentialRefused
from core.self_modification.coding_aci import (
    VIEW_BUDGET_LINES,
    CodingSurface,
    TooMuchToRead,
    summarise_failure,
)

ROOT = Path(__file__).resolve().parent.parent


# ── the tool plan ─────────────────────────────────────────────────────────

def _pages():
    return {f"p{i}": "x" * 500 + ("MATCH" if i % 3 == 0 else "") for i in range(9)}


def _tools(pages):
    return {"fetch": lambda url: pages[url], "list": lambda: sorted(pages)}


def _plan():
    return Plan("find matches", (
        Step(Op.CALL, into="urls", tool="list"),
        Step(Op.MAP, into="bodies", over="urls", tool="fetch", args={"url": None}),
        Step(Op.FILTER, into="hits", over="bodies", fn=lambda b: "MATCH" in b),
        Step(Op.ASSERT, over="hits", fn=bool, message="nothing matched"),
        Step(Op.MAP, into="sizes", over="hits", fn=len),
        Step(Op.RETURN, over="sizes"),
    ))


def test_a_plan_does_in_one_round_trip_what_a_conversation_does_in_ten():
    execution = Executor().run(
        _plan(), tools=_tools(_pages()), permitted=frozenset({"fetch", "list"})
    )
    assert execution.tool_calls == 10
    assert execution.round_trips == 1
    assert execution.sequential_round_trips == 10


def test_only_what_the_plan_returns_reaches_the_context():
    execution = Executor().run(
        _plan(), tools=_tools(_pages()), permitted=frozenset({"fetch", "list"})
    )
    assert execution.context_saved > 4000
    assert execution.bytes_returned < 100


def test_a_plan_cannot_reach_a_tool_the_caller_may_not_use():
    with pytest.raises(PlanRefused, match="not a way around the gateway"):
        Executor().run(_plan(), tools=_tools(_pages()), permitted=frozenset({"list"}))


def test_the_refusal_happens_before_anything_runs():
    calls = []
    tools = {"list": lambda: calls.append("list") or [], "fetch": lambda url: None}
    with pytest.raises(PlanRefused):
        Executor().run(_plan(), tools=tools, permitted=frozenset({"list"}))
    assert calls == []


def test_filtering_to_nothing_is_a_failure_not_a_finding():
    pages = {f"p{i}": "x" * 10 for i in range(3)}
    execution = Executor().run(
        _plan(), tools=_tools(pages), permitted=frozenset({"fetch", "list"})
    )
    assert execution.failed == "nothing matched"
    assert execution.returned is None


def test_a_plan_cannot_exceed_its_declared_call_bound():
    plan = Plan(
        "runaway",
        (
            Step(Op.CALL, into="urls", tool="list"),
            Step(Op.MAP, into="bodies", over="urls", tool="fetch", args={"url": None}),
        ),
        max_calls=3,
    )
    execution = Executor().run(
        plan, tools=_tools(_pages()), permitted=frozenset({"fetch", "list"})
    )
    assert "exceeded 3 tool calls" in execution.failed


def test_retry_reruns_its_body_and_gives_up_on_the_bound():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("flaky")
        return ["ok"]

    plan = Plan("retrying", (
        Step(Op.RETRY, attempts=5, body=(Step(Op.CALL, into="v", tool="flaky"),)),
        Step(Op.RETURN, over="v"),
    ))
    execution = Executor().run(plan, tools={"flaky": flaky}, permitted=frozenset({"flaky"}))
    assert execution.returned == ["ok"] and attempts["n"] == 3


# ── the coding surface ────────────────────────────────────────────────────

def _surface():
    return CodingSurface(ROOT)


def test_an_outline_lets_reading_a_file_whole_be_a_choice():
    definitions = _surface().outline("core/cognition/tool_plan.py")
    assert any(d.qualified == "Executor.run" for d in definitions)
    assert all(d.end >= d.start for d in definitions)


def test_a_view_larger_than_the_budget_is_refused_not_returned():
    with pytest.raises(TooMuchToRead, match="narrow it with outline"):
        _surface().view("core/skills/screen_pursuit.py")


def test_one_definition_can_be_read_without_the_file_around_it():
    body = _surface().view_definition("core/cognition/tool_plan.py", "Executor.run")
    assert "def run" in body
    assert len(body.splitlines()) < VIEW_BUDGET_LINES


def test_a_structural_edit_replaces_one_definition_and_reverts_exactly(tmp_path):
    module = tmp_path / "m.py"
    module.write_text("def a():\n    return 1\n\n\ndef b():\n    return 1\n")
    surface = CodingSurface(tmp_path)
    edit = surface.replace_definition("m.py", "b", "def b():\n    return 2")
    assert "def a():\n    return 1" in module.read_text()
    assert "return 2" in module.read_text()
    surface.revert(edit)
    assert module.read_text().count("return 1") == 2


def test_editing_a_definition_that_does_not_exist_raises(tmp_path):
    module = tmp_path / "m.py"
    module.write_text("def a():\n    return 1\n")
    with pytest.raises(KeyError):
        CodingSurface(tmp_path).replace_definition("m.py", "nope", "def nope(): pass")


def test_test_selection_finds_the_tests_that_name_what_changed():
    hits = _surface().tests_touching(["CodingSurface"])
    assert "tests/test_agent_surface.py" in hits


def test_failure_summarisation_keeps_the_assertion_and_drops_the_frames():
    traceback = (
        'Traceback (most recent call last):\n'
        '  File "/x/harness.py", line 12, in run\n'
        '    thing()\n'
        '  File "/x/tests/test_a.py", line 44, in test_a\n'
        '    assert got == want\n'
        'E   AssertionError: assert 3 == 4\n'
        'AssertionError'
    )
    summary = summarise_failure(traceback)
    assert "test_a.py" in summary["where"]
    assert summary["frames_dropped"] == 1
    assert any("3 == 4" in value for value in summary["values"])


def test_a_surface_cannot_reach_outside_its_root(tmp_path):
    surface = CodingSurface(tmp_path)
    with pytest.raises(ValueError):
        surface.outline("../../etc/hosts")


# ── governed subagents ────────────────────────────────────────────────────

def _work(subagent, context):
    subagent.spend(1.0)
    return {"saw": sorted(context)}


def test_a_subagent_sees_only_the_context_it_declared():
    finding = Conductor().run(
        SubagentSpec("researcher", "find sources", frozenset({"query"}), budget=5.0),
        _work, context={"query": "x", "private_notes": "y"},
    )
    assert finding.content == {"saw": ["query"]}
    assert finding.context_keys == ("query",)


def test_a_subagent_that_overspends_ends_rather_than_borrowing():
    def greedy(subagent, context):
        subagent.spend(99.0)

    finding = Conductor().run(
        SubagentSpec("greedy", "x", frozenset({"query"}), budget=1.0),
        greedy, context={"query": "x"},
    )
    assert "ends rather than borrowing" in finding.failed


def test_a_subagent_returns_a_finding_and_never_an_action():
    finding = Conductor().run(
        SubagentSpec("r", "x", frozenset({"q"}), budget=5.0), _work, context={"q": 1}
    )
    assert isinstance(finding, Finding)
    assert Conductor().fanout_report()["nothing_committed"]


def test_a_subagent_declaring_tools_the_caller_lacks_is_refused():
    finding = Conductor().run(
        SubagentSpec("r", "x", frozenset({"q"}), budget=1.0, tools=frozenset({"shell"})),
        _work, context={"q": 1}, permitted_tools=frozenset({"read"}),
    )
    assert "may not use" in finding.failed


def test_a_failing_worker_becomes_a_finding_rather_than_a_crash():
    def broken(subagent, context):
        raise RuntimeError("the index is down")

    finding = Conductor().run(
        SubagentSpec("r", "x", frozenset({"q"}), budget=1.0), broken, context={"q": 1}
    )
    assert "RuntimeError" in finding.failed


def test_fanout_says_whether_fanning_out_was_worth_it():
    conductor = Conductor()
    specs = [SubagentSpec(f"w{i}", "x", frozenset({"q"}), budget=5.0) for i in range(4)]
    cheap = conductor.fanout(specs, _work, context={"q": 1}, single_worker_cost=10.0)
    assert cheap["worth_it"] is True
    expensive = conductor.fanout(specs, _work, context={"q": 1}, single_worker_cost=1.0)
    assert expensive["worth_it"] is False
    assert expensive["reconciliation_needed"]


def test_an_unjudged_fanout_claims_nothing():
    conductor = Conductor()
    conductor.fanout(
        [SubagentSpec("w", "x", frozenset({"q"}), budget=5.0)], _work, context={"q": 1}
    )
    assert conductor.fanout_report()["fanouts_judged"] == 0


# ── scoped credentials ────────────────────────────────────────────────────

def _broker(now):
    return CredentialBroker({"gh": "TOKEN"}, clock=lambda: now[0], default_ttl=10.0)


def test_the_caller_never_receives_the_value():
    now = [1000.0]
    broker = _broker(now)
    lease = broker.issue("gh", purpose="clone", scopes=["github.com"])
    seen = broker.use(lease.lease_id, purpose="clone", scope="github.com",
                      operation=lambda value: len(value))
    assert seen == len("TOKEN")
    assert broker.report()["values_returned_to_callers"] == 0


def test_a_wildcard_scope_is_refused():
    with pytest.raises(CredentialRefused, match="ceremony"):
        _broker([0.0]).issue("gh", purpose="x", scopes=["*"])


def test_a_lease_works_once_and_then_does_not():
    now = [1000.0]
    broker = _broker(now)
    lease = broker.issue("gh", purpose="clone", scopes=["github.com"], uses=1)
    broker.use(lease.lease_id, purpose="clone", scope="github.com", operation=lambda v: v)
    with pytest.raises(CredentialRefused, match="no uses remaining"):
        broker.use(lease.lease_id, purpose="clone", scope="github.com", operation=lambda v: v)


def test_a_lease_used_for_another_purpose_is_refused():
    now = [1000.0]
    broker = _broker(now)
    lease = broker.issue("gh", purpose="clone", scopes=["github.com"], uses=5)
    with pytest.raises(CredentialRefused, match="issued for 'clone'"):
        broker.use(lease.lease_id, purpose="push", scope="github.com", operation=lambda v: v)


def test_a_lease_used_outside_its_scope_is_refused():
    now = [1000.0]
    broker = _broker(now)
    lease = broker.issue("gh", purpose="clone", scopes=["github.com"], uses=5)
    with pytest.raises(CredentialRefused, match="is not in"):
        broker.use(lease.lease_id, purpose="clone", scope="evil.example",
                   operation=lambda v: v)


def test_expiry_is_checked_against_an_injected_clock_not_a_sleep():
    now = [1000.0]
    broker = _broker(now)
    lease = broker.issue("gh", purpose="clone", scopes=["github.com"], uses=5, ttl=10.0)
    now[0] += 100.0
    with pytest.raises(CredentialRefused, match="expired"):
        broker.use(lease.lease_id, purpose="clone", scope="github.com", operation=lambda v: v)


def test_a_revoked_lease_stops_working_immediately():
    now = [1000.0]
    broker = _broker(now)
    lease = broker.issue("gh", purpose="clone", scopes=["github.com"], uses=5)
    assert broker.revoke(lease.lease_id)
    with pytest.raises(CredentialRefused, match="revoked"):
        broker.use(lease.lease_id, purpose="clone", scope="github.com", operation=lambda v: v)


def test_every_refusal_is_recorded():
    now = [1000.0]
    broker = _broker(now)
    lease = broker.issue("gh", purpose="clone", scopes=["github.com"], uses=5)
    for scope in ("a", "b", "c"):
        with pytest.raises(CredentialRefused):
            broker.use(lease.lease_id, purpose="clone", scope=scope, operation=lambda v: v)
    assert len(broker.report()["refusals"]) == 3
