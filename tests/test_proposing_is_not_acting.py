"""Agency pathways are asked what they want, and used to do it while answering.

Every registered pathway runs on every pulse, then arbitration picks one
winner. Several pathways changed the world during that evaluation: they spent
the shared skill cooldown, dispatched swarm shards, and claimed the dispatch in
their own message. A proposal that lost — or that the action gate blocked —
had already spent the cooldown that stops real work from spamming, so a losing
pathway suppressed a winning one for the next half hour, and a shard the
capacity gate refused still became a memory of work in flight.
"""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.agency.agency_core import AgencyCore, _skill_cooldown_effect

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "core" / "agency" / "agency_core.py").read_text("utf-8")
#: The shard's work, lifted out of agency_core on 21 September.
SHARD_SOURCE = (ROOT / "core" / "agency" / "agency_shard_work.py").read_text("utf-8")


def test_no_pathway_spends_the_cooldown_while_being_asked():
    """The assignment may appear only inside the deferred effect."""
    tree = ast.parse(SOURCE)

    def _direct_body(node: ast.AST):
        """Everything in this function except its nested functions.

        A cooldown assignment inside a deferred effect is the fix, not the
        defect, and a plain ast.walk cannot tell the two apart — it descends
        into the closure and reports it as an evaluation-time write.
        """
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            yield child
            yield from _direct_body(child)

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("_pathway_"):
            continue
        for inner in _direct_body(node):
            if (
                isinstance(inner, ast.Attribute)
                and inner.attr == "last_skill_use"
                and isinstance(inner.ctx, ast.Store)
            ):
                offenders.append((node.name, inner.lineno))
    assert not offenders, f"pathways still spend the cooldown at evaluation: {offenders}"


def test_the_cooldown_effect_spends_it_when_run():
    state = SimpleNamespace(last_skill_use=0.0)

    effect = _skill_cooldown_effect(state, 1234.0)
    assert state.last_skill_use == 0.0, "building the effect already spent it"

    effect()
    assert state.last_skill_use == 1234.0


def test_the_aesthetic_pathway_declares_rather_than_spends():
    agency = SimpleNamespace(
        state=SimpleNamespace(
            initiative_energy=0.9,
            curiosity_pressure=0.9,
            last_skill_use=0.0,
        ),
        _resolve_component=lambda _name: None,
    )

    # 5% chance per evaluation; try until it proposes.
    action = None
    for _ in range(500):
        action = AgencyCore._pathway_aesthetic_creation(agency, now=9000.0, idle_seconds=600.0)
        if action:
            break
    assert action is not None, "the pathway never proposed in 500 evaluations"

    assert agency.state.last_skill_use == 0.0
    effects = dict(action["_deferred_effects"])
    effects["aesthetic_creation.cooldown"]()
    assert agency.state.last_skill_use == 9000.0


def test_a_refused_shard_is_not_reported_as_dispatched():
    """spawn_shard's result was discarded and the reflection claimed dispatch
    either way."""
    calls: list[bool] = []

    class Swarm:
        @staticmethod
        async def spawn_shard(**_kwargs):
            calls.append(False)
            return False

    agency = SimpleNamespace(
        state=SimpleNamespace(curiosity_pressure=0.9, last_skill_use=0.0),
        swarm=Swarm(),
    )

    action = None
    for _ in range(500):
        action = AgencyCore._pathway_autonomous_research(agency, now=7000.0, idle_seconds=600.0)
        if asyncio.iscoroutine(action):
            action = asyncio.run(action)
        if action:
            break
    assert action is not None

    # Nothing dispatched while merely proposing.
    assert calls == []
    assert "I've dispatched" not in action["thought"]

    effects = dict(action["_deferred_effects"])
    asyncio.run(effects["autonomous_research.dispatch"]())
    assert calls == [False]


def test_the_research_target_is_the_implementation_not_the_facade():
    """core/agency_core.py is a 48-line re-export shim; the shard spent its
    analysis on it while the implementation next door went unread."""
    assert '"agency/agency_core.py"' in SOURCE
    assert 'target_file = "core/agency/agency_core.py"' in SOURCE


def test_the_self_architect_cooldown_is_set_on_every_branch():
    """The assignment sat after every branch that returns, so it ran only when
    the pathway proposed nothing — a cooldown that applied on failure alone."""
    marker = SOURCE.index("async def _pathway_self_architect")
    body = SOURCE[marker : SOURCE.index("def _pathway_environmental_explorer")]

    assert body.count("_audit_cooldown") >= 5, body.count("_audit_cooldown")
    assert "self._last_meta_audit = now\n        return None" not in body


def test_the_shard_count_is_published_when_the_map_changes():
    """It was captured before capacity admission and before the task existed,
    and cleanup removed tasks without telling the registry, so the advertised
    number only ever went up."""
    marker = SOURCE.index("self.active_shards[shard_id] = task")
    block = SOURCE[marker : marker + 900]

    assert "self._publish_shard_count()" in block
    cleanup = SOURCE[SOURCE.index("def _cleanup(t: asyncio.Task)") :][:400]
    assert "self._publish_shard_count()" in cleanup


@pytest.mark.asyncio
async def test_the_winner_runs_its_declared_effects():
    ran: list[str] = []

    async def _async_effect():
        ran.append("async")

    def _sync_effect():
        ran.append("sync")

    core = AgencyCore.__new__(AgencyCore)
    core.state = SimpleNamespace(
        unshared_observations=[],
        topics_to_discuss=[],
        last_observation_comment=0.0,
        last_self_initiated_contact=0.0,
    )

    action = {
        "type": "internal_reflection",
        "_deferred_effects": [("a", _sync_effect), ("b", _async_effect)],
    }

    assert await core._commit_action_side_effects(action, now=1.0) is True
    assert ran == ["sync", "async"]


@pytest.mark.asyncio
async def test_a_failing_effect_does_not_abort_the_approved_action():
    ran: list[str] = []

    def _boom():
        raise RuntimeError("effect failed")

    def _after():
        ran.append("after")

    core = AgencyCore.__new__(AgencyCore)
    core.state = SimpleNamespace(
        unshared_observations=[],
        topics_to_discuss=[],
        last_observation_comment=0.0,
        last_self_initiated_contact=0.0,
    )

    action = {"type": "internal_reflection", "_deferred_effects": [("boom", _boom), ("after", _after)]}

    assert await core._commit_action_side_effects(action, now=1.0) is True
    assert ran == ["after"]


def test_a_shard_keeps_nothing_and_writes_nothing_for_her():
    """When structured generation collapsed, the shard wrote her answer itself
    ("I am experiencing cognitive formatting degradation. Taking a defensive
    posture."), and before a guard was added the crucible committed it as a
    belief and the abstraction engine learned from it. The fallback is gone, and
    a shard hands its reply to neither: both reworded it before keeping it."""
    tree = ast.parse(SHARD_SOURCE)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body.pop(0)  # its docstring quotes the old fallback; the code must not
    code = ast.unparse(tree)
    for kept_by in ("run_crucible", "abstract_from_success", "_record_durable_insight", "add_insight"):
        assert kept_by not in code
    for fabricated in ("formatting degradation", "defensive posture", "ShardResponse", "StructuredLLM"):
        assert fabricated not in code
    assert 'pulse_hypha("collective", "distributed_agency", success=True)' not in SOURCE


def test_the_internet_pulse_waits_for_the_research_to_win():
    """The agency->internet edge was pulsed with success=True while the pathway
    was only building a proposal: no request made, nothing returned, and it
    might not even win. The network's picture of reachability was a record of
    how often Aura felt like looking."""
    marker = SOURCE.index("def _pathway_world_monitor")
    block = SOURCE[marker : SOURCE.index("def _pathway_self_development")]

    assert "world_monitor.pulse" in block
    assert "_deferred_effects" in block
    # The pulse call itself may only appear inside the deferred closure.
    pulse_at = block.index('pulse_hypha("agency", "internet"')
    closure_at = block.index("def _pulse_research_intent")
    assert closure_at < pulse_at


def test_identity_writes_happen_only_for_the_winner():
    """Social reflection, creative synthesis and metacognitive audit wrote
    beliefs into identity while every pathway was still being evaluated."""
    tree = ast.parse(SOURCE)

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("_pathway_"):
            continue
        for child in ast.iter_child_nodes(node):
            for inner in ast.walk(child):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == "_record_durable_insight"
                ):
                    # Only acceptable inside a nested effect closure.
                    offenders.append((node.name, inner.lineno))
    # Every remaining call must sit inside a nested function, which the walk
    # above only reaches through one — so verify by source position instead.
    for name, lineno in offenders:
        line = SOURCE.splitlines()[lineno - 1]
        assert line.startswith("            ") or line.startswith("                "), (
            f"{name}:{lineno} writes identity at pathway top level"
        )


def test_an_insight_write_reads_its_own_disposition():
    """add_insight returns 'denied' | 'duplicate' | 'saved' | 'persist_failed'
    precisely so a caller can tell durable success from a mutation that never
    reached disk. Every call site discarded it."""
    assert "_record_durable_insight" in SOURCE
    assert SOURCE.count("identity.add_insight(") == 1, "a raw write came back"
    marker = SOURCE.index("def _record_durable_insight")
    block = SOURCE[marker : marker + 1400]
    assert 'disposition in {"saved", "duplicate"}' in block
    assert "not durable" in block


def test_the_trust_claim_is_grounded_in_what_she_has():
    """"Our interactions feel increasingly grounded in mutual trust" was
    asserted from the mere existence of a kinship entry — no trend, no
    comparison, nothing that could have come out the other way."""
    # Checked against string literals, not the file text: the comment
    # explaining the removal quotes the old sentence, and a substring search
    # cannot tell an explanation from the thing it explains.
    tree = ast.parse(SOURCE)
    spoken = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            spoken.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    spoken.append(part.value)

    assert not any("increasingly grounded in mutual trust" in t for t in spoken)
    assert any("What I have to go on: kinship level" in t for t in spoken)
