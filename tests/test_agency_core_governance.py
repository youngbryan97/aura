from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_swarm_shard_tool_execution_creates_governed_subprocess_scope(monkeypatch):
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "production")

    from core.agency.agency_core import SovereignSwarm
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    class SubprocessBackedTool:
        async def route_and_execute(self, _name, _payload):
            result = get_subprocess_gateway().run(
                ["true"],
                capture_output=True,
                timeout=1.0,
                source="test.agency_core_governance.shard_tool",
                accelerator_capability="none",
            )
            return f"ok:{result.returncode}"

    owner = SimpleNamespace(tool_orchestrator=SubprocessBackedTool(), _last_tool_routing_error=None)
    swarm = SovereignSwarm(SimpleNamespace(), agency_core=owner)

    assert await swarm._execute_shard_tool("python_sandbox", "print('safe')") == "ok:0"
    assert owner._last_tool_routing_error is None


def test_agency_degradation_reporting_cannot_cascade(monkeypatch):
    from core.agency import agency_core

    def raising_reporter(*_args, **_kwargs):
        raise RuntimeError("failure policy fail-closed")

    monkeypatch.setattr(agency_core, "record_degradation", raising_reporter)

    agency_core._record_agency_degradation(
        RuntimeError("original shard failure"),
        action="shard cleanup failed",
    )


def test_ungoverned_subprocess_still_fails_closed(monkeypatch):
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "production")

    from core.governance_context import GovernanceViolation
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    with pytest.raises(GovernanceViolation):
        get_subprocess_gateway().run(
            ["true"],
            capture_output=True,
            timeout=1.0,
            source="test.agency_core_governance.ungoverned",
            accelerator_capability="none",
        )


def test_both_high_risk_restraints_are_resolvable_or_every_tool_is_refused():
    """The high-risk gate refuses unless BOTH restraints actually ran.

    That is correct fail-closed behaviour — but only one of the pair was ever
    registered. `causal_world_model` went on the container at boot;
    `dynamic_value_graph` existed solely as a module singleton, so
    `ServiceContainer.get("dynamic_value_graph")` always returned None and the
    gate concluded the restraint "could not run". Measured live: every
    python_sandbox / shell_executor / file_operations request refused with
    `unavailable restraints: dynamic_value_graph` and a CRITICAL degradation
    per attempt, while the graph was healthy and in use by mind_tick.
    """
    from core.adaptation.dynamic_value_graph import (
        get_dynamic_value_graph,
        register_dynamic_value_graph,
    )
    from core.brain.causal_world_model import register_causal_world_model
    from core.container import ServiceContainer

    register_causal_world_model()
    graph = register_dynamic_value_graph()

    assert ServiceContainer.get("causal_world_model", default=None) is not None
    resolved = ServiceContainer.get("dynamic_value_graph", default=None)
    assert resolved is not None, (
        "an unregistered value graph makes every high-risk tool permanently refused"
    )
    assert resolved is graph
    # Registration is idempotent — two callers must not split the graph's state.
    assert register_dynamic_value_graph() is graph
    assert get_dynamic_value_graph() is graph

    # The gate reads .get_status()["nodes"] and inspects each node's "status";
    # that shape is the contract between the organ and the restraint.
    nodes = graph.get_status().get("nodes", {})
    assert isinstance(nodes, dict) and nodes, "the value graph must expose value nodes"
    assert all(isinstance(node, dict) for node in nodes.values())
    assert all("weight" in node and "status" in node for node in nodes.values()), (
        "the high-risk gate sorts on weight and refuses on provisional status"
    )


def test_high_risk_gate_falls_back_to_the_value_graph_accessor():
    """Resolving ONLY through the container cannot be the single point of failure.

    An unregistered-but-healthy restraint was indistinguishable from a broken
    one, and fail-closed then disabled tool execution for the life of the
    process. The gate now falls back to the module accessor.
    """
    from core.agency import agency_core, agency_shard_work
    from tests.source_contract import declared_in

    # The gate moved with `_shard_wrapper` when the shard work was lifted
    # into its own module. What it DOES is unchanged, and the location was
    # never the point.
    _module, source = declared_in(
        'ServiceContainer.get("dynamic_value_graph"', agency_core, agency_shard_work
    )
    gate = source[source.index('ServiceContainer.get("dynamic_value_graph"'):]
    gate = gate[: gate.index("_value_check_done = True")]
    assert "get_dynamic_value_graph" in gate, (
        "the gate must fall back to the accessor when the container has no entry"
    )
