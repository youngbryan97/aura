import ast
import inspect

import pytest

# ── The container-independence gate ──────────────────────────────────────
#
# These modules must reach services through core.runtime.service_registry and
# never through the container directly. That rule was enforced by substring
# search over the module's source, at twelve sites, which had two problems.
#
# It fired on prose. A comment in core_baseline explaining a service-name
# collision — the word ServiceContainer inside a `#` line — failed the gate,
# and the only ways out were to delete the explanation or to loosen the check.
# A rule that punishes documentation gets documentation removed.
#
# And it was weaker than it looked. `import core.container as c` was caught by
# accident (the string appears), while a substring check cannot tell an import
# from a mention, so nobody could rely on which it was actually testing.
#
# What follows parses the module and asks structurally: what does it import,
# what names does it reference, what does it build strings out of. Comments and
# docstrings are free. Everything the old check caught in code is still caught,
# and dynamic access through a string literal — importlib.import_module(
# "core.container"), getattr(m, "ServiceContainer") — is caught now too, which
# the structural pass alone would have missed.

#: Import paths that mean "this module reached for the container".
#:
#: The bare name is here because a relative ``from .container import x`` reaches
#: the AST as module="container", which is what the old
#: ``"from container import"`` check was after.
_CONTAINER_MODULES = frozenset({"core.container", "container"})

#: What counts as a container reference inside a *string*.
#:
#: Only the dotted path. The bare word is an ordinary English noun and is used
#: as a dict key in the health reports — ``{"container": container_report}`` —
#: which is not a module reference by any reading. Flagging it would make the
#: gate fire on health telemetry, and a gate that fires on unrelated code gets
#: switched off. The dotted form has no innocent use: nothing writes
#: "core.container" in a string except to import it by name.
_CONTAINER_PATHS_IN_STRINGS = frozenset({"core.container"})


def _facts_from_source(source: str) -> tuple[set[str], set[str], set[str]]:
    """(imported modules, referenced names, non-docstring string literals)."""
    tree = ast.parse(source)

    docstrings = {
        ast.get_docstring(node, clean=False)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    imports: set[str] = set()
    names: set[str] = set()
    literals: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # A relative import carries no package prefix, so `from .container
            # import x` arrives as module="container" — which is exactly what
            # the old `"from container import"` check was after.
            if node.module:
                imports.add(node.module)
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in docstrings:
                literals.add(node.value)

    return imports, names, literals


def _container_violations(
    source: str,
    *,
    forbid_names: tuple[str, ...] = ("ServiceContainer",),
    require_import: str | None = "core.runtime.service_registry",
    require_names: tuple[str, ...] = (),
) -> list[str]:
    """Every way this source breaks the rule, named. Empty means it holds.

    Pure, and separate from the assertion, so the gate itself can be tested
    against sources that are supposed to fail. A gate nobody points at a
    violation is one that can quietly stop catching anything.
    """
    imports, names, literals = _facts_from_source(source)
    violations: list[str] = []

    for offending in sorted(imports & _CONTAINER_MODULES):
        violations.append(f"imports {offending}")
    for forbidden in forbid_names:
        if forbidden in names:
            violations.append(f"references {forbidden}")
    for text in sorted(literals):
        if text in _CONTAINER_PATHS_IN_STRINGS or text in set(forbid_names):
            violations.append(f"names {text!r} in a string literal")
    if require_import is not None and require_import not in imports:
        violations.append(f"does not import {require_import}")
    for required in require_names:
        if required not in names:
            violations.append(f"never references {required}")
    return violations


def _assert_reaches_services_through_the_registry(
    module,
    *,
    forbid_names: tuple[str, ...] = ("ServiceContainer",),
    require_import: str | None = "core.runtime.service_registry",
    require_names: tuple[str, ...] = (),
) -> None:
    """The gate. Raises with the module and the offending symbol named.

    ``forbid_names`` varies by site on purpose and is not unified: cognitive_engine
    publishes its own ``get_container`` shim, so forbidding that name everywhere
    would fail a module the original gate deliberately permitted it in.
    """
    where = getattr(module, "__name__", repr(module))
    violations = _container_violations(
        inspect.getsource(module),
        forbid_names=forbid_names,
        require_import=require_import,
        require_names=require_names,
    )
    assert not violations, f"{where} " + "; ".join(violations)


_REGISTRY_IMPORT = "import core.runtime.service_registry\n"

#: Every bypass the gate must catch. The first four are what the substring
#: check caught; the last four are what it could not, because a substring
#: cannot tell an import from a mention or find a name assembled at runtime.
_MUST_CATCH = {
    "direct import": "from core.container import ServiceContainer\n" + _REGISTRY_IMPORT,
    "aliased module import": "import core.container as c\n" + _REGISTRY_IMPORT,
    "aliased symbol": "from core.container import ServiceContainer as SC\n" + _REGISTRY_IMPORT,
    "relative import": "from container import thing\n" + _REGISTRY_IMPORT,
    "bare name": _REGISTRY_IMPORT + "x = ServiceContainer\n",
    "attribute access": _REGISTRY_IMPORT + "import m\nx = m.ServiceContainer\n",
    "dynamic import": _REGISTRY_IMPORT + 'import importlib\nm = importlib.import_module("core.container")\n',
    "getattr by string": _REGISTRY_IMPORT + 'x = getattr(m, "ServiceContainer")\n',
    "no registry import": "x = 1\n",
}

#: Patterns that must stay legal. The first two are the reason this gate is
#: structural: a rule that fires on an explanation gets the explanation
#: deleted. The third is live code — the health report keys a section
#: "container", which is a noun, not a module.
_MUST_ALLOW = {
    "comment": "# ServiceContainer was the other object under this name\n" + _REGISTRY_IMPORT,
    "docstring": '"""Explains core.container and ServiceContainer."""\n' + _REGISTRY_IMPORT,
    "container as a dict key": _REGISTRY_IMPORT + 'report = {"container": {}}\n',
    "clean module": _REGISTRY_IMPORT,
}


@pytest.mark.parametrize("label", sorted(_MUST_CATCH))
def test_the_container_gate_catches_the_bypass(label):
    assert _container_violations(_MUST_CATCH[label]), f"{label} slipped past the gate"


@pytest.mark.parametrize("label", sorted(_MUST_ALLOW))
def test_the_container_gate_allows_legitimate_code(label):
    assert _container_violations(_MUST_ALLOW[label]) == [], f"{label} was wrongly refused"


def test_record_degradation_does_not_import_service_container():
    import inspect
    import core.runtime.errors as errors

    source = inspect.getsource(errors.record_degradation)
    assert "from core.container import ServiceContainer" not in source
    assert "core.observability.metrics" not in source
    assert "core.runtime.service_registry" in source


def test_record_degradation_uses_low_level_metric_sink():
    from core.runtime.errors import record_degradation
    from core.runtime.service_registry import install_metric_counter_sink

    counters: list[tuple[str, int]] = []
    install_metric_counter_sink(lambda name, amount=1: counters.append((name, amount)))

    try:
        record_degradation(
            "metric_bridge",
            RuntimeError("sample"),
            severity="warning",
            action="verify low-level metric sink",
        )
    finally:
        install_metric_counter_sink(None)

    assert ("degradation_metric_bridge_warning", 1) in counters


def test_record_degradation_preserves_fail_closed_policy(monkeypatch):
    from core.runtime.errors import record_degradation
    from core.runtime.service_registry import install_failure_policy_resolver

    install_failure_policy_resolver(
        lambda name: "fail-closed" if name == "critical_service" else None
    )
    monkeypatch.setenv("AURA_MODE", "production")

    try:
        with pytest.raises(RuntimeError, match="CRITICAL SERVICE FAILURE"):
            record_degradation(
                "critical_service",
                RuntimeError("boom"),
                severity="warning",
                action="test failure-policy bridge",
            )
    finally:
        install_failure_policy_resolver(None)


def test_timeout_on_fail_closed_subsystem_does_not_escalate(monkeypatch):
    """A timeout is backpressure, not a service death.

    Bounded background timeouts on fail-closed subsystems (sovereign_pruner,
    dialectical_crucible, cognitive_engine→agency_core) repeatedly cascaded to
    unified_failure_lockdown 1.00 by raising a CRITICAL SERVICE FAILURE
    (observed live 2026-07-04/05). Genuine faults must still fail closed; bare
    timeouts must not.
    """
    import asyncio

    from core.runtime.errors import record_degradation
    from core.runtime.service_registry import install_failure_policy_resolver

    install_failure_policy_resolver(
        lambda name: "fail-closed" if name == "critical_service" else None
    )
    monkeypatch.setenv("AURA_MODE", "production")

    try:
        # TimeoutError and asyncio.TimeoutError are demoted: recorded, never raised.
        for timeout_error in (TimeoutError("slow"), asyncio.TimeoutError()):
            record = record_degradation(
                "critical_service",
                timeout_error,
                severity="degraded",
                action="bounded background wait expired",
            )
            assert "CRITICAL SERVICE FAILURE" not in (record.error_message or "")
            assert record.severity != "critical"

        # A genuine fault on the same fail-closed subsystem still fails closed.
        with pytest.raises(RuntimeError, match="CRITICAL SERVICE FAILURE"):
            record_degradation(
                "critical_service",
                RuntimeError("corruption"),
                severity="warning",
                action="genuine fault",
            )
    finally:
        install_failure_policy_resolver(None)


def test_metrics_reads_services_through_low_level_registry(monkeypatch):
    import inspect
    import numpy as np
    import core.observability.metrics as metrics_module
    from core.runtime.service_registry import install_service_resolver

    class Substrate:
        def get_state_summary(self):
            return {"valence": 0.25, "step_count": 7}

        def get_state_vector(self):
            return np.array([0.1, 0.2, 0.3])

    class DriveEngine:
        def get_state(self):
            return {"drives": {"curiosity": 0.8}}

    services = {
        "continuous_substrate": Substrate(),
        "drive_engine": DriveEngine(),
    }

    install_service_resolver(lambda name, default=None: services.get(name, default))
    monkeypatch.setattr(metrics_module, "_metrics_instance", metrics_module.MetricsCollector())

    try:
        samples = metrics_module.get_metrics().collect()
        sample_map = {(sample.name, tuple(sample.labels.items())): sample.value for sample in samples}
        assert sample_map[("aura_substrate_valence", ())] == 0.25
        assert sample_map[("aura_substrate_step_count", ())] == 7.0
        assert sample_map[("aura_drive_level", (("drive", "curiosity"),))] == 0.8
        assert metrics_module.check_readiness()["status"] in {"ready", "not_ready"}
    finally:
        install_service_resolver(None)

    source = inspect.getsource(metrics_module)
    assert "from core.container import ServiceContainer" not in source


def test_degradation_repair_default_lookup_uses_low_level_registry():
    import inspect
    import core.resilience.degradation_repair as repair_module
    from core.runtime.service_registry import install_service_resolver

    class Resilience:
        def record_failure(self, **_kwargs):
            return "recorded"

    install_service_resolver(
        lambda name, default=None: Resilience() if name == "resilience_engine" else default
    )

    try:
        router = repair_module.DegradationRepairRouter(cooldown_seconds=0)
        action = router.route(
            record=type("Record", (), {"subsystem": "demo", "severity": "degraded"})(),
            error=RuntimeError("boom"),
            incident=None,
            extra={},
        )
        assert action.resilience_state == "recorded"
    finally:
        install_service_resolver(None)

    assert "from core.container import ServiceContainer" not in inspect.getsource(
        repair_module._service_get
    )


def test_health_contract_uses_low_level_runtime_registry():
    import inspect
    import time
    import core.runtime.health_contract as health_contract
    from core.runtime.service_registry import install_service_resolver

    class Orchestrator:
        start_time = time.time() - 12.0

    class Stakes:
        def get_status(self):
            return {
                "existential_threat": 0.01,
                "lag_threat": 0.0,
                "memory_threat": 0.0,
            }

    class Monitor:
        def get_status(self):
            return {"last_lag_s": 0.0, "last_failure_reason": ""}

    services = {
        "orchestrator": Orchestrator(),
        "existential_stakes": Stakes(),
        "event_loop_monitor": Monitor(),
        "hypervisor": Monitor(),
    }
    install_service_resolver(lambda name, default=None: services.get(name, default))

    try:
        assert health_contract._process_uptime_seconds() >= 0.0
        status = health_contract._runtime_pressure_status()
        assert status.present is True
        assert status.liveness_ok is True
    finally:
        install_service_resolver(None)

    source = inspect.getsource(health_contract)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_governance_context_uses_runtime_registry_predicates(monkeypatch):
    import inspect
    import core.governance_context as governance_context
    from core.runtime.service_registry import (
        install_registration_locked_resolver,
        install_service_presence_resolver,
    )

    monkeypatch.delenv("AURA_GOVERNANCE_MODE", raising=False)
    monkeypatch.delenv("AURA_REQUIRE_GOVERNANCE", raising=False)

    install_service_presence_resolver(lambda name: name == "kernel_interface")
    install_registration_locked_resolver(lambda: False)
    try:
        assert governance_context.governance_runtime_active() is True
    finally:
        install_service_presence_resolver(None)
        install_registration_locked_resolver(None)

    install_service_presence_resolver(lambda _name: False)
    install_registration_locked_resolver(lambda: True)
    try:
        assert governance_context.governance_runtime_active() is True
    finally:
        install_service_presence_resolver(None)
        install_registration_locked_resolver(None)

    source = inspect.getsource(governance_context.governance_runtime_active)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_event_loop_monitor_uses_runtime_registry_for_foreground_status():
    import inspect
    import core.utils.concurrency as concurrency
    from core.runtime.service_registry import install_service_resolver

    class Gate:
        def get_conversation_status(self):
            return {
                "active": False,
                "foreground_owned": True,
                "warmup_in_flight": False,
                "active_generations": 0,
                "current_request_started_at": 0.0,
                "state": "ready",
            }

    install_service_resolver(lambda name, default=None: Gate() if name == "inference_gate" else default)
    try:
        monitor = concurrency.EventLoopMonitor()
        assert monitor._active_runtime_reason() == "foreground_generation"
    finally:
        install_service_resolver(None)

    source = inspect.getsource(concurrency.EventLoopMonitor._active_runtime_reason)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source

    robust_lock_source = inspect.getsource(concurrency.RobustLock.acquire_robust)
    assert "core.observability.metrics" not in robust_lock_source
    assert "core.container" not in robust_lock_source


def test_degraded_events_forwarding_uses_runtime_registry():
    import inspect
    import core.health.degraded_events as degraded_events
    from core.runtime.service_registry import install_service_resolver

    calls: list[tuple[BaseException, dict, str, str]] = []

    class SelfModifier:
        def on_error(self, error, context, *, skill_name, goal):
            calls.append((error, context, skill_name, goal))
            return None

    class Orchestrator:
        self_modifier = SelfModifier()

    event = {
        "classification": "foreground_blocking",
        "subsystem": "demo",
        "reason": "blocked",
        "detail": "detail",
        "severity": "critical",
        "context": {"source": "test"},
    }
    degraded_events._LAST_FORWARDED.clear()
    install_service_resolver(lambda name, default=None: Orchestrator() if name == "orchestrator" else default)

    try:
        degraded_events._forward_to_error_intelligence(
            ("demo", "blocked", "critical", "foreground_blocking"),
            event,
            exc=RuntimeError("boom"),
        )
    finally:
        install_service_resolver(None)
        degraded_events._LAST_FORWARDED.clear()

    assert len(calls) == 1
    error, context, skill_name, goal = calls[0]
    assert isinstance(error, RuntimeError)
    assert context["source"] == "test"
    assert skill_name == "demo"
    assert goal == "blocked"

    source = inspect.getsource(degraded_events._forward_to_error_intelligence)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_terminal_monitor_reliability_lookup_uses_runtime_registry():
    import inspect
    import core.terminal_monitor as terminal_monitor

    source = inspect.getsource(terminal_monitor.TerminalMonitor.check_for_errors)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_terminal_monitor_world_state_publish_uses_runtime_registry():
    import inspect
    import core.terminal_monitor as terminal_monitor

    source = inspect.getsource(terminal_monitor.TerminalMonitor._ingest_error)
    assert "from core.world_state import get_world_state" not in source
    assert "core.runtime.service_registry" in source


def test_world_state_registration_uses_runtime_registry(monkeypatch):
    import inspect
    import core.world_state as world_state
    from core.runtime.service_registry import (
        install_service_presence_resolver,
        install_service_registration_sink,
    )

    registered: list[tuple[str, object, bool, dict[str, str | None]]] = []
    monkeypatch.setattr(world_state, "_ws_instance", None)
    install_service_presence_resolver(lambda _name: False)
    install_service_registration_sink(
        lambda name, instance, required, metadata: registered.append(
            (name, instance, required, metadata)
        )
    )
    try:
        instance = world_state.get_world_state()
    finally:
        install_service_presence_resolver(None)
        install_service_registration_sink(None)
        monkeypatch.setattr(world_state, "_ws_instance", None)

    assert registered
    name, registered_instance, required, metadata = registered[0]
    assert name == "world_state"
    assert registered_instance is instance
    assert required is False
    assert metadata["owner"] == "core/world_state.py"

    source = inspect.getsource(world_state)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_ice_sentinel_registration_uses_runtime_registry(monkeypatch):
    import inspect
    import core.security.ice_sentinel as ice_sentinel
    from core.runtime.service_registry import (
        install_service_registration_sink,
        install_service_resolver,
    )

    registered: list[tuple[str, object, bool, dict[str, str | None]]] = []
    monkeypatch.setattr(ice_sentinel, "_INSTANCE", None)
    install_service_resolver(lambda _name, default=None: default)
    install_service_registration_sink(
        lambda name, instance, required, metadata: registered.append(
            (name, instance, required, metadata)
        )
    )
    try:
        instance = ice_sentinel.register_ice_sentinel()
    finally:
        install_service_resolver(None)
        install_service_registration_sink(None)
        monkeypatch.setattr(ice_sentinel, "_INSTANCE", None)

    names = [item[0] for item in registered]
    assert "ice" in names
    assert any(item[1] is instance and item[2] is False for item in registered)

    source = inspect.getsource(ice_sentinel.register_ice_sentinel)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_engine_support_brain_resolution_uses_runtime_registry():
    import inspect
    import core.utils.engine_support as engine_support
    from core.runtime.service_registry import install_service_resolver

    class Brain:
        pass

    class Orchestrator:
        brain = Brain()

    install_service_resolver(lambda name, default=None: Orchestrator() if name == "orchestrator" else default)
    try:
        assert isinstance(engine_support.resolve_brain(), Brain)
    finally:
        install_service_resolver(None)

    source = inspect.getsource(engine_support.resolve_brain)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_hierarchical_agency_handlers_use_runtime_registry():
    import inspect
    import core.agency.hierarchical_agency as hierarchical_agency
    from core.runtime.service_registry import install_service_resolver

    class GoalEngine:
        pass

    signals: list[tuple[str, str]] = []

    class RsiLoop:
        def record_signal(self, source, kind, **_kwargs):
            signals.append((source, kind))

    def resolver(name, default=None):
        if name == "goal_engine":
            return GoalEngine()
        if name == "recursive_self_improvement":
            return RsiLoop()
        return default

    agency = hierarchical_agency.HierarchicalAgency(ledger_enabled=False)
    install_service_resolver(resolver)
    try:
        strategic = agency._strategic(hierarchical_agency.Situation("plan", goal_horizon=0.8))
        self_improve = agency._self_improvement(
            hierarchical_agency.Situation("missing skill", capability_gap=0.9)
        )
    finally:
        install_service_resolver(None)

    assert strategic.detail["routed_to"] == "goal_engine"
    assert self_improve.detail["recorded"] == "capability_gap_signal"
    assert signals == [("hierarchical_agency", "capability_gap")]

    source = inspect.getsource(hierarchical_agency.HierarchicalAgency._strategic)
    source += inspect.getsource(hierarchical_agency.HierarchicalAgency._self_improvement)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_emergency_protocol_minimal_mode_uses_runtime_registry():
    import inspect
    import core.security.emergency_protocol as emergency_protocol
    from core.runtime.service_registry import install_service_resolver

    class BackgroundService:
        running = True

    services = {
        "curiosity_explorer": BackgroundService(),
        "skill_synthesizer": BackgroundService(),
    }
    install_service_resolver(lambda name, default=None: services.get(name, default))
    try:
        protocol = emergency_protocol.EmergencyProtocol()
        protocol._enter_minimal_mode()
    finally:
        install_service_resolver(None)

    assert services["curiosity_explorer"].running is False
    assert services["skill_synthesizer"].running is False

    source = inspect.getsource(emergency_protocol.EmergencyProtocol._enter_minimal_mode)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_scientific_engine_belief_publish_uses_runtime_registry():
    import inspect
    import core.cognition.scientific_engine as scientific_engine
    from core.runtime.service_registry import install_service_resolver

    beliefs: list[tuple[str, object, float, str]] = []

    class WorldState:
        def set_belief(self, key, value, *, confidence, source):
            beliefs.append((key, value, confidence, source))

    engine = scientific_engine.ScientificEngine.__new__(scientific_engine.ScientificEngine)
    hypothesis = scientific_engine.Hypothesis(
        hypothesis_id="hyp-test",
        claim="runtime registry routes belief updates",
        predicted_observable="belief_written",
        expected=1.0,
        confidence=0.8,
        status="supported",
    )
    install_service_resolver(lambda name, default=None: WorldState() if name == "world_state" else default)
    try:
        engine._publish_belief(hypothesis)
    finally:
        install_service_resolver(None)

    assert beliefs == [
        (
            "hypothesis:runtime registry routes belief updates",
            {"status": "supported", "expected": 1.0},
            0.8,
            "scientific_engine",
        )
    ]

    source = inspect.getsource(scientific_engine.ScientificEngine._publish_belief)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_intention_loop_uses_runtime_registry(tmp_path, monkeypatch):
    import inspect
    import core.agency.intention_loop as intention_loop
    from core.runtime.service_registry import (
        install_service_registration_sink,
        install_service_resolver,
    )

    class Embedder:
        def similarity(self, _expected, _actual):
            return 0.75

    services = {
        "cognitive_ledger": object(),
        "belief_revision_engine": object(),
        "embedding_engine": Embedder(),
    }
    original_class = intention_loop.IntentionLoop
    install_service_resolver(lambda name, default=None: services.get(name, default))
    try:
        loop = intention_loop.IntentionLoop(db_path=str(tmp_path / "intentions.db"))
        assert loop._get_ledger() is services["cognitive_ledger"]
        assert loop._get_belief_engine() is services["belief_revision_engine"]
        assert loop._calculate_surprise("expected outcome", "actual outcome") == 0.25
    finally:
        install_service_resolver(None)
        loop.close()

    registered: list[tuple[str, object, bool, dict[str, str | None]]] = []
    instance = object()
    monkeypatch.setattr(intention_loop, "_instance", None)
    monkeypatch.setattr(intention_loop, "IntentionLoop", lambda: instance)
    install_service_registration_sink(
        lambda name, value, required, metadata: registered.append(
            (name, value, required, metadata)
        )
    )
    try:
        assert intention_loop.get_intention_loop() is instance
    finally:
        install_service_registration_sink(None)
        monkeypatch.setattr(intention_loop, "_instance", None)

    assert registered
    assert registered[0][0] == "intention_loop"
    assert registered[0][1] is instance

    source = inspect.getsource(original_class._get_ledger)
    source += inspect.getsource(original_class._get_belief_engine)
    source += inspect.getsource(original_class._calculate_surprise)
    source += inspect.getsource(intention_loop.get_intention_loop)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_consciousness_system_publication_uses_runtime_registry():
    import inspect
    import core.consciousness.system as consciousness_system

    _assert_reaches_services_through_the_registry(
        consciousness_system,
        require_import=None,
        require_names=("register_runtime_service", "get_runtime_service"),
    )


def test_being_runtime_publish_uses_runtime_registry():
    import inspect
    import core.being.runtime as being_runtime
    from core.runtime.service_registry import install_service_registration_sink

    registered: list[tuple[str, object, bool, dict[str, str | None]]] = []
    runtime = being_runtime.BeingRuntime.__new__(being_runtime.BeingRuntime)
    now = object()
    install_service_registration_sink(
        lambda name, instance, required, metadata: registered.append(
            (name, instance, required, metadata)
        )
    )
    try:
        runtime._publish(now)
    finally:
        install_service_registration_sink(None)

    assert [item[0] for item in registered] == ["aura_now", "being_runtime"]
    assert registered[0][1] is now
    assert registered[1][1] is runtime
    assert registered[0][2] is False
    assert registered[1][2] is False

    source = inspect.getsource(being_runtime.BeingRuntime._publish)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_cognitive_engine_service_lookup_uses_runtime_registry_adapter():
    import inspect
    import core.brain.cognitive_engine as cognitive_engine
    from core.runtime.service_registry import install_service_resolver

    class Router:
        last_tier = "primary"

    router = Router()
    install_service_resolver(lambda name, default=None: router if name == "llm_router" else default)
    try:
        engine = cognitive_engine.CognitiveEngine()
        assert engine._current_tier == "primary"
        assert cognitive_engine.get_container().get("llm_router") is router
    finally:
        install_service_resolver(None)

    _assert_reaches_services_through_the_registry(cognitive_engine)


def test_cryptolalia_decoder_uses_runtime_registry():
    import inspect
    import core.brain.cryptolalia_decoder as cryptolalia_decoder
    from core.runtime.service_registry import (
        install_service_registration_sink,
        install_service_resolver,
    )

    class ConceptBridge:
        _concept_cache = {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
        }

    install_service_resolver(
        lambda name, default=None: ConceptBridge() if name == "concept_bridge" else default
    )
    try:
        decoder = cryptolalia_decoder.CryptolaliaDecoder()
        assert decoder.approximate_translation([1.0, 0.0], top_n=1) == "[alpha]"
    finally:
        install_service_resolver(None)

    registered: list[tuple[str, object, bool, dict[str, str | None]]] = []
    install_service_registration_sink(
        lambda name, instance, required, metadata: registered.append(
            (name, instance, required, metadata)
        )
    )
    try:
        instance = cryptolalia_decoder.register_cryptolalia_decoder()
    finally:
        install_service_registration_sink(None)

    assert registered
    assert registered[0][0] == "cryptolalia_decoder"
    assert registered[0][1] is instance
    assert registered[0][2] is True

    _assert_reaches_services_through_the_registry(cryptolalia_decoder)


def test_meta_cognition_structural_review_uses_runtime_registry():
    import asyncio
    import inspect
    import core.meta.meta_cognition as meta_cognition
    from core.runtime.service_registry import install_service_resolver

    queued: list[tuple[str, str]] = []

    class MetaEvolution:
        def queue_optimization(self, *, target_area, context):
            queued.append((target_area, context))

    loop = meta_cognition.MetaCognition()
    loop.error_history = [
        {"decision": f"failure-{idx}", "outcome": "failure", "context": {}, "timestamp": 1.0}
        for idx in range(6)
    ]

    install_service_resolver(
        lambda name, default=None: MetaEvolution() if name == "meta_evolution" else default
    )
    try:
        asyncio.run(loop._trigger_structural_review())
    finally:
        install_service_resolver(None)

    assert queued
    assert queued[0][0] == "cognitive_patterns"
    assert "failure-5" in queued[0][1]
    assert len(loop.error_history) == 2

    _assert_reaches_services_through_the_registry(meta_cognition)


def test_startup_boot_validator_uses_runtime_registry_presence():
    import inspect
    import core.startup.boot_validator as boot_validator
    from core.runtime.service_registry import install_service_presence_resolver

    required = {"event_bus", "orchestrator", "llm_interface", "state_repo"}
    install_service_presence_resolver(lambda name: name in required)
    try:
        result = boot_validator.BootValidator.validate_boot()
    finally:
        install_service_presence_resolver(None)

    assert result.passed is True
    assert result.failures == []

    install_service_presence_resolver(lambda name: name == "event_bus")
    try:
        result = boot_validator.BootValidator.validate_boot()
    finally:
        install_service_presence_resolver(None)

    assert result.passed is False
    assert "Core Orchestrator Ready" in result.failures
    assert "LLM Interface Bound" in result.failures
    assert "State Repository (Persistence) Ready" in result.failures

    class ExplicitContainer:
        def has(self, name):
            return name in required

    assert boot_validator.BootValidator.validate_boot(ExplicitContainer()).passed is True

    _assert_reaches_services_through_the_registry(boot_validator)


def test_small_runtime_service_batch_uses_registry():
    import asyncio
    import inspect
    from types import SimpleNamespace

    import core.affect.affect_facade as affect_facade
    import core.agency.latent_distiller as latent_distiller
    import core.brain.personality_bridge as personality_bridge
    import core.brain.scratchpad as scratchpad
    import core.identity.identity_anchor as identity_anchor
    import core.ops.singularity_monitor as singularity_monitor
    import core.resilience.hotfix_engine as hotfix_engine
    from core.brain.types import ThinkingMode, Thought
    from core.runtime.service_registry import install_service_resolver
    from core.state.aura_state import AuraState

    class AffectEngine:
        def get_status(self):
            return {
                "mood": "steady",
                "energy": 60,
                "curiosity": 55,
                "frustration": 0,
                "stability": 100,
                "valence": 0.2,
                "arousal": 0.3,
            }

    class CognitiveEngine:
        async def think(self, objective, **_kwargs):
            return Thought(
                id="registry-thought",
                content=f"summary:{objective[:20]}",
                mode=ThinkingMode.FAST,
            )

    class Memory:
        def __init__(self):
            self.records = []

        async def store_memory(self, **kwargs):
            self.records.append(kwargs)

    class Repo:
        def __init__(self):
            self._current = AuraState.default()
            self._current.identity.name = "Aura"
            self._current.state_id = "abcdefgh1234"

        async def get_current(self):
            return self._current

    class Mirror:
        def get_audit_summary(self):
            return {"health_score": 0.95}

    class MetaCognition:
        mirror = Mirror()

    repo = Repo()
    memory = Memory()
    services = {
        "affect_engine": AffectEngine(),
        "cognitive_engine": CognitiveEngine(),
        "state_repo": repo,
        "state_repository": repo,
        "metacognition": MetaCognition(),
    }
    install_service_resolver(lambda name, default=None: services.get(name, default))
    try:
        anchor = identity_anchor.IdentityAnchor()
        # Identity comes from the DURABLE ENTITY KEY, not from the state
        # repository this batch installs. That dependency was removed on
        # purpose — reading identity out of AuraState is what made it
        # transient, so it changed between the first tick after boot and the
        # ten-thousandth. Asserting the old state-derived value here would
        # pin the defect: the mocked repo below must not be able to move it.
        identity = anchor.get_identity()
        assert identity
        assert identity == identity_anchor.IdentityAnchor().get_identity(), (
            "identity differs between two anchors in the same process"
        )
        assert "abcdefgh" not in identity, (
            "identity is being derived from the mocked state repository again"
        )

        facade = affect_facade.AffectFacade()
        assert facade.is_ready() is True
        assert facade.get_status()["mood"] == "steady"

        bridge = personality_bridge.PersonalityBridge()
        assert asyncio.run(bridge.sync_embodiment(SimpleNamespace(model=None)))["damping_mult"] > 0

        pad = scratchpad.ScratchpadEngine()
        plan = asyncio.run(pad.think_recursive("plan carefully", {"history": []}, depth=0))
        # CP126 92172bb9: the result separates a user-safe strategy from the
        # private monologue; str() yields the safe one.
        assert plan.ok is True
        assert plan.strategy.startswith("summary:")
        assert plan.monologue.startswith("[Plan] summary:")

        monitor = singularity_monitor.SingularityMonitor(orchestrator=SimpleNamespace(container=True))
        monitor.pulse()
        assert monitor.is_accelerated is True
        assert monitor.acceleration_factor == 1.5

        distiller = latent_distiller.LatentSpaceDistiller(memory_provider=memory)
        long_history = [{"content": "This session produced a useful design insight. " * 8}]
        asyncio.run(distiller.distill_session(long_history))
        assert memory.records
        assert memory.records[0]["metadata"]["type"] == "distilled_wisdom"
    finally:
        install_service_resolver(None)

    for module in (
        identity_anchor,
        affect_facade,
        personality_bridge,
        scratchpad,
        singularity_monitor,
        latent_distiller,
        hotfix_engine,
    ):
        _assert_reaches_services_through_the_registry(module, require_import=None)


def test_runtime_service_registry_supports_lazy_factory_publication():
    from core.runtime.service_registry import (
        install_service_factory_registration_sink,
        register_runtime_factory,
    )

    captured = []
    factory = lambda: "ready"
    install_service_factory_registration_sink(
        lambda name, fn, lifetime, required, metadata: captured.append(
            (name, fn, lifetime, required, metadata)
        )
    )
    try:
        assert register_runtime_factory(
            "demo_factory",
            factory,
            lifetime="singleton",
            required=False,
            owner="tests",
            registered_by="test_runtime_service_registry_supports_lazy_factory_publication",
            metadata={"extra": "value"},
        )
    finally:
        install_service_factory_registration_sink(None)

    assert captured == [
        (
            "demo_factory",
            factory,
            "singleton",
            False,
            {
                "owner": "tests",
                "registered_by": "test_runtime_service_registry_supports_lazy_factory_publication",
                "required_for": None,
                "failure_policy": None,
                "extra": "value",
            },
        )
    ]


def test_runtime_registry_batch_two_service_seams():
    import asyncio
    import inspect
    from types import SimpleNamespace

    import core.capabilities.source_summarizer as source_summarizer
    import core.affect.emotional_coloring as emotional_coloring
    import core.evals.adaptive_test_chamber as adaptive_test_chamber
    import core.memory.provenance as provenance
    import core.orchestrator.coordinators.affect as affect_coordinator
    import core.plasticity.plasticity_controller as plasticity_controller
    import core.senses.voice_socket_logic as voice_socket_logic
    import core.learning.skill_evolution as skill_evolution
    import core.utils.telemetry_enrichment as telemetry_enrichment
    from core.brain import concept_vector_bridge
    from core.runtime.service_registry import (
        install_service_factory_registration_sink,
        install_service_registration_sink,
        install_service_resolver,
    )

    class EventBus:
        def __init__(self):
            self.events = []

        async def publish(self, event, payload):
            self.events.append((event, payload))

    class Client:
        async def generate_embedding(self, text):
            return [float(len(text)), 1.0]

    class Router:
        async def route(self, **_kwargs):
            return SimpleNamespace(text="synthesized summary")

        def get_health_report(self):
            return {"foreground_tier": "primary"}

    class Browser:
        async def extract_article_text(self, url):
            return SimpleNamespace(
                url=url,
                title="Article",
                body="Useful article body.",
                author="Reporter",
                source_domain="example.com",
                word_count=3,
            )

    class LiquidState:
        def get_status(self):
            return {
                "energy": 0.6,
                "curiosity": 0.7,
                "frustration": 0.1,
                "confidence": 0.8,
            }

        def get_valence(self):
            return 0.25

    class Homeostasis:
        def get_health(self):
            return {"will_to_live": 0.9}

    class Memory:
        def search(self, _topic, limit=5):
            return [{"emotion": "joy", "arousal": 0.4}]

    class BeliefGraph:
        def get_all_beliefs(self):
            return [1, 2, 3]

    class InsightJournal:
        def get_highest_confidence_insights(self, limit=10):
            return list(range(8))

    class Omni:
        _execution_logs = {"web_search": [{"status": "error"}] * 4}

    class Swarm:
        def __init__(self):
            self.objectives = []

        async def spawn_shard(self, **kwargs):
            self.objectives.append(kwargs)

    class Reflex:
        def __init__(self):
            self.commands = []

        async def process_emergency_interrupt(self, command, context):
            self.commands.append((command, context))

    class FakeWhisper:
        def transcribe(self, _audio_np, beam_size=1):
            return [SimpleNamespace(text=" stop ")], None

    event_bus = EventBus()
    swarm = Swarm()
    reflex = Reflex()
    services = {
        "event_bus": event_bus,
        "cognitive_engine": SimpleNamespace(client=Client()),
        "llm_router": Router(),
        "browser_controller": Browser(),
        "liquid_state": LiquidState(),
        "homeostasis": Homeostasis(),
        "memory": Memory(),
        "free_energy_engine": SimpleNamespace(current=SimpleNamespace(free_energy=0.2)),
        "belief_graph": BeliefGraph(),
        "insight_journal": InsightJournal(),
        "omni_tool": Omni(),
        "sovereign_swarm": swarm,
        "orchestrator": SimpleNamespace(reflex_engine=reflex),
        "affect_engine": SimpleNamespace(
            get_mood=lambda: "Curious",
            get_status=lambda: {"mood": "Curious"},
        ),
    }
    registered = []
    factories = []
    install_service_resolver(lambda name, default=None: services.get(name, default))
    install_service_registration_sink(
        lambda name, instance, required, metadata: registered.append(
            (name, instance, required, metadata)
        )
    )
    install_service_factory_registration_sink(
        lambda name, factory, lifetime, required, metadata: factories.append(
            (name, factory, lifetime, required, metadata)
        )
    )
    try:
        bridge = concept_vector_bridge.register_concept_bridge()
        assert ("concept_bridge", bridge, True, registered[-1][3]) == registered[-1]
        assert asyncio.run(bridge.transmit("a", "b", [1.0])) .startswith("latent_")
        assert event_bus.events[0][0] == "cryptolalia_transmission"
        assert asyncio.run(bridge.generate_concept_vector("cat")) == [3.0, 1.0]

        summarizer = source_summarizer.SourceSummarizer()
        asyncio.run(summarizer.start())
        assert any(item[0] == "source_summarizer" for item in registered)
        result = asyncio.run(
            summarizer.summarize_urls(["https://example.com"], objective="brief")
        )
        assert result.summary == "synthesized summary"

        stamped = provenance.wrap({"fact": "sample"})
        assert round(stamped.provenance.confidence, 2) == 0.8

        enriched = telemetry_enrichment.enrich_telemetry({})
        assert enriched["energy"] == 60.0
        assert enriched["llm_tier"] == "primary"

        emotional_coloring.register_emotional_coloring()
        assert any(item[0] == "emotional_coloring" for item in factories)
        texture = asyncio.run(emotional_coloring.EmotionalColoring().get_texture_for_topic("cat"))
        assert texture.tone_hint == "warm/exploratory"

        plasticity_controller.register_plasticity_controller()
        assert any(item[0] == "plasticity_controller" for item in factories)
        plasticity = plasticity_controller.PlasticityController()
        assert asyncio.run(plasticity.update_plasticity()) == 0.8

        skill_evolution.register_skill_evolution()
        assert any(item[0] == "skill_evolution" for item in factories)
        engine = skill_evolution.SkillEvolutionEngine()
        assert asyncio.run(engine.identify_evolution_targets()) == ["web_search"]
        asyncio.run(engine.spawn_evolution_shard("web_search"))
        assert swarm.objectives[0]["context"] == {"target_skill": "web_search"}

        chamber = adaptive_test_chamber.register_test_chamber()
        # A freshly registered chamber has recorded nothing, so it reports
        # healthy=None with a reason — not True.
        #
        # This asserted True from when "healthy" was a hard-coded literal. The
        # chamber deliberately stopped doing that: a health field that cannot
        # return False is a decoration on a dictionary, and it reads as a
        # PASSED CHECK to everything downstream. Demanding True here would
        # require reinstating exactly that.
        fresh = chamber.get_status()
        assert fresh["healthy"] is None
        assert fresh["reason"] == "no results recorded yet"
        assert [item[0] for item in registered if item[0] in {"glados_test_chamber", "glados"}] == [
            "glados_test_chamber",
            "glados",
        ]

        processor = voice_socket_logic.VoiceStreamProcessor(model_instance=FakeWhisper())
        processor.speech_buffer = [(b"\0\0" * 160)]
        assert asyncio.run(processor.get_transcript()) == "stop"
        assert reflex.commands == [("STOP", "audio_stream")]

        coordinator = affect_coordinator.AffectCoordinator(
            orchestrator=SimpleNamespace(_get_service=lambda _name: None)
        )
        assert coordinator.get_mood() == "Curious"
    finally:
        install_service_resolver(None)
        install_service_registration_sink(None)
        install_service_factory_registration_sink(None)

    for module in (
        concept_vector_bridge,
        source_summarizer,
        provenance,
        telemetry_enrichment,
        emotional_coloring,
        plasticity_controller,
        skill_evolution,
        adaptive_test_chamber,
        voice_socket_logic,
        affect_coordinator,
    ):
        _assert_reaches_services_through_the_registry(module)


def test_runtime_registry_batch_three_live_path_service_seams():
    import asyncio
    import inspect
    import time
    from types import SimpleNamespace

    import numpy as np

    import core.agency.autonomy_latitude as autonomy_latitude
    import core.agency.desktop_planner as desktop_planner
    import core.brain.composer_node as composer_node
    import core.brain.llm.substrate_token_generator as substrate_token_generator
    import core.brain.metacognitive_monitor as metacognitive_monitor
    import core.brain.morphic_forking as morphic_forking
    import core.brain.narrator as narrator
    import core.self_modification.code_refiner as code_refiner
    import core.collective.swarm_protocol as swarm_protocol
    import core.conversation.conversational_momentum_engine as momentum_engine
    import core.guardians.user_advocate as user_advocate
    import core.maintenance.dream_coordinator as dream_coordinator
    import core.being.panzer_soul as panzer_soul
    import core.phenomenal_substrate.philosophical_stance as philosophical_stance
    import core.pneuma.topological_memory as topological_memory
    import core.resilience.reflex_engine as reflex_engine
    import core.resilience.phenomenal_error_map as phenomenal_error_map
    import core.runtime.derived_runtime_context as derived_runtime_context
    import core.runtime.live_mind_snapshot as live_mind_snapshot
    import core.senses.tts_stream as tts_stream
    import interface.memory_ui as memory_ui
    import interface.routes.rpc as rpc_routes
    from core.runtime.service_registry import (
        install_service_factory_registration_sink,
        install_service_presence_resolver,
        install_service_registration_sink,
        install_service_resolver,
    )
    from core.state.aura_state import AuraState

    class Sync:
        def __init__(self):
            self.calls = []

        async def handle_rpc_request(self, route, payload):
            self.calls.append((route, payload))
            return {"route": route, "payload": payload}

        async def handle_incoming_beliefs(self, payload):
            self.calls.append(("beliefs", payload))

        async def handle_incoming_principles(self, payload):
            self.calls.append(("principles", payload))

    class Router:
        async def think(self, prompt, **_kwargs):
            if "Respond in JSON" in str(prompt):
                return '{"coherent": true, "score": 0.91, "violations": [], "metrics": {"clarity": 1, "logic": 1, "factuality": 1, "persona": 1}}'
            return "narrated response"

        async def generate(self, *_args, **_kwargs):
            return "momentum response"

    class Cognition:
        async def think(self, *_args, **_kwargs):
            return "VERDICT: ACCEPTED"

    class Substrate:
        def __init__(self):
            self._snapshot_buffer = [np.zeros(64, dtype=np.float32)]

        def get_state_vector(self):
            return np.ones(64, dtype=np.float32) * 0.2

        def get_state_summary(self):
            return {"phi": 0.42}

        def get_mood(self):
            return {"valence": 0.4}

    class Affect:
        def __init__(self):
            self.signals = []
            self.modifications = []

        def apply_signal(self, **kwargs):
            self.signals.append(kwargs)

        def modify(self, **kwargs):
            self.modifications.append(kwargs)

        def get_status(self):
            return {"mood": "steady"}

    class Skill:
        def __init__(self):
            self.calls = []

        async def safe_execute(self, payload, _context):
            # `safe_execute`, not `execute`. DesktopAdapter._dispatch gates on
            # `hasattr(skill, "safe_execute")` and calls it; a double that only
            # offers `execute` sends the adapter down its "computer_use
            # unavailable" branch, where it logs a no-op and returns. The
            # assertion below then reads an empty list. The double has to speak
            # the seam the caller actually uses.
            self.calls.append(payload)
            return {"ok": True}

        async def execute(self, payload, context):
            return await self.safe_execute(payload, context)

    class Vision:
        frame_buffer = [b"frame"]

        async def query_visual_context(self, prompt, cognition):
            assert cognition is not None
            return f"visual:{prompt[:12]}"

    class Facade:
        def __init__(self):
            now_ms = time.time() * 1000
            self.vector = SimpleNamespace(
                memories=[{"text": "remembered event", "created": now_ms, "access_count": 1}]
            )

        def setup(self):
            return None

    class Reflex:
        def __init__(self):
            self.commands = []

        async def process_emergency_interrupt(self, command, context):
            self.commands.append((command, context))

    class Mycelium:
        def __init__(self):
            self.pulsed = False

        def pulse_hypha(self, *_args, success=True):
            self.pulsed = bool(success)
            return True

    sync = Sync()
    affect = Affect()
    skill = Skill()
    mycelium = Mycelium()
    substrate = Substrate()
    agency = SimpleNamespace(_action_queue=[1], state=SimpleNamespace(safemode=False))
    orchestrator = SimpleNamespace(
        status=SimpleNamespace(is_processing=False),
        _last_user_interaction_time=0.0,
        process_user_input=lambda *_args, **_kwargs: None,
        memory=SimpleNamespace(get_recent_texts=lambda limit=10: []),
    )
    services = {
        "belief_sync": sync,
        "affective_steering_engine": SimpleNamespace(telemetry=SimpleNamespace(level=1)),
        "continuous_substrate": substrate,
        "liquid_state": substrate,
        "conscious_substrate": substrate,
        "voice_engine": object(),
        "soul": SimpleNamespace(),
        "continuous_vision": Vision(),
        "cognitive_engine": Cognition(),
        "mycelium": mycelium,
        "mycelial_network": mycelium,
        "affect_engine": affect,
        "neurochemical_regulator": SimpleNamespace(nudge=lambda *_args, **_kwargs: None),
        "skill:computer_use": skill,
        "llm_router": Router(),
        "orchestrator": orchestrator,
        "memory_facade": Facade(),
        "agency": agency,
        "global_workspace": SimpleNamespace(get_snapshot=lambda: {"winner": "test"}),
        "nociception": SimpleNamespace(snapshot=lambda: {"pain": 0}),
        "affect_grounding": SimpleNamespace(gather=lambda: SimpleNamespace(assess=lambda: [], dominant=lambda: "steady")),
        "drive_integration": SimpleNamespace(state=lambda: {"curiosity": 0.6}),
        "outcome_ledger": SimpleNamespace(stats=lambda: {"n": 1}),
        "scientific_engine": SimpleNamespace(stats=lambda: {"hypotheses": 1}),
        "unified_world_model": SimpleNamespace(status=lambda: {"ok": True}),
        "phenomenal_engine": SimpleNamespace(last_state=SimpleNamespace(valence=0.1)),
        "phenomenal_knowing": SimpleNamespace(snapshot=lambda: {"knows": True}),
        "recursive_self_knowing": SimpleNamespace(snapshot=lambda: {"depth": 2}),
        "automatic_self_knowing": SimpleNamespace(snapshot=lambda: {"automatic": True}),
        "screen_perception": SimpleNamespace(get_status=lambda: {"seeing": True}),
        "perceptual_pump": SimpleNamespace(get_status=lambda: {"running": True}),
        "safe_surf": SimpleNamespace(scan=lambda _message: {"level": "low", "categories": ["test"], "advice": "watch"}),
        "ice": SimpleNamespace(
            inspect_input=lambda _message: {"level": "none"},
            inspect_output=lambda _text: {"level": "none", "recommended_action": "allow"},
        ),
        "samantha": SimpleNamespace(attune=lambda _message: {"recommended_tone": "warm", "valence": 0.2, "arousal": 0.1, "resonance": 0.7}),
        "hal": SimpleNamespace(is_safe_to_proceed=lambda: (True, [])),
        "data": SimpleNamespace(vet_output=lambda text, confidence=None: f"{text}"),
        "drive_engine": SimpleNamespace(get_state=lambda: {"curiosity": 0.5}),
    }
    registered = []
    factories = []
    install_service_resolver(lambda name, default=None: services.get(name, default))
    install_service_presence_resolver(lambda name: name in services)
    install_service_registration_sink(
        lambda name, instance, required, metadata: registered.append(
            (name, instance, required, metadata)
        )
    )
    install_service_factory_registration_sink(
        lambda name, factory, lifetime, required, metadata: factories.append(
            (name, factory, lifetime, required, metadata)
        )
    )
    try:
        assert asyncio.run(rpc_routes.rpc_query_beliefs({"x": 1}, None, None))["route"] == "query_beliefs"
        assert asyncio.run(rpc_routes.rpc_receive_beliefs({"belief": 1}, None, None)) == {"status": "accepted"}

        autonomy_latitude.reset_autonomy_latitude_for_test()
        assert autonomy_latitude.get_autonomy_latitude().classify("reflection").latitude == "autonomous"
        assert any(item[0] == "autonomy_latitude" for item in registered)

        generator = substrate_token_generator.get_substrate_token_generator()
        assert generator.substrate is substrate
        assert any(item[0] == "substrate_token_generator" for item in registered)

        assert topological_memory.get_runtime_service("drive_engine") is services["drive_engine"]

        collector = philosophical_stance.BehavioralProofCollector()
        asyncio.run(collector.start())
        assert any(item[0] == "behavioral_proof" for item in registered)
        assert collector.generate_proof_bundle()["metrics"]["functional_phi"] == 0.42

        assert tts_stream.FastMouth().engine is services["voice_engine"]
        soul = panzer_soul.get_panzer_soul()
        assert soul.version == panzer_soul.version

        composed = asyncio.run(composer_node.ComposerNode().stylize_desktop("ink wash"))
        assert composed["ok"] is True
        assert mycelium.pulsed is True

        phenomenal_error_map._notify_substrate(phenomenal_error_map.PHENOMENAL_STATES["tool_failure"])
        assert affect.signals

        assert user_advocate.register_user_advocate().get_status()["healthy"] is True
        assert any(item[0] == "tron" for item in registered)

        fork = morphic_forking.register_morphic_forking()
        assert asyncio.run(fork.absorb_insight("test", "VERDICT: ACCEPTED")) is True

        code_refiner.register_code_refiner()
        assert any(item[0] == "code_refiner" for item in factories)

        swarm = swarm_protocol.SwarmProtocol()
        asyncio.run(swarm._process_gossip({"type": "mood_sync", "node_id": "peer", "mood": {"valence": 0.5}}))
        assert affect.modifications

        asyncio.run(desktop_planner.DesktopAdapter().open_app("Notes"))
        assert skill.calls[-1]["action"] == "open_app"

        state = AuraState.default()
        state.identity.current_narrative = "Aura"
        report = asyncio.run(metacognitive_monitor.MetacognitiveMonitor().evaluate("hello", state))
        assert report.is_coherent is True

        momentum = momentum_engine.ConversationalMomentumEngine()
        assert momentum.orchestrator is orchestrator

        narrator.register_narrator_service()
        assert any(item[0] == "narrator" for item in factories)
        n = narrator.NarratorService()
        assert n.llm_router is services["llm_router"]

        memory_payload = asyncio.run(memory_ui.get_vault_stats())
        assert memory_payload["status"] == "online"
        assert memory_payload["total_nodes"] == 1

        snapshot = live_mind_snapshot.collect_live_mind_snapshot(lane={"origin": "test"})
        assert snapshot["services_present"]["global_workspace"] is True
        assert snapshot["global_workspace"]["winner"] == "test"

        engine = reflex_engine.ReflexEngine()
        assert asyncio.run(engine.process_emergency_interrupt("STOP")) is True
        assert agency._action_queue == []

        derived = derived_runtime_context.collect_derived_runtime_context("hello")
        assert derived["input"]["threat_watch"]["level"] == "low"
        assert derived_runtime_context.guard_user_facing_output("safe") == "safe"

        coord = dream_coordinator.get_dream_coordinator()
        assert isinstance(coord.status(), dict)
    finally:
        install_service_resolver(None)
        install_service_presence_resolver(None)
        install_service_registration_sink(None)
        install_service_factory_registration_sink(None)
        autonomy_latitude.reset_autonomy_latitude_for_test()

    for module in (
        autonomy_latitude,
        rpc_routes,
        substrate_token_generator,
        topological_memory,
        philosophical_stance,
        tts_stream,
        panzer_soul,
        composer_node,
        phenomenal_error_map,
        user_advocate,
        morphic_forking,
        code_refiner,
        dream_coordinator,
        swarm_protocol,
        desktop_planner,
        metacognitive_monitor,
        momentum_engine,
        narrator,
        memory_ui,
        live_mind_snapshot,
        reflex_engine,
        derived_runtime_context,
    ):
        _assert_reaches_services_through_the_registry(module)


def test_runtime_registry_batch_four_boot_sensory_health_seams():
    import asyncio
    import inspect
    from types import SimpleNamespace

    import core.brain.cognitive_manager as cognitive_manager
    import core.brain.llm.lazarus_brainstem as lazarus_brainstem
    import core.brain.llm.web_augmentor as web_augmentor
    import core.coordinators.dream_coordinator as dream_coordinator
    import core.health.system_health as system_health
    import core.initializers.self_knowing as self_knowing
    import core.morality.master_moral_integration as master_moral_integration
    import core.memory.attention as attention
    import core.observability.neural_feed as neural_feed
    import core.orchestrator.initializers.core_baseline as core_baseline
    import core.phases.consciousness_phase as consciousness_phase
    import core.phases.executive_closure as executive_closure
    import core.social.presence_integration as presence_integration
    import core.runtime.response_policy as response_policy
    import core.senses.sensory_instincts as sensory_instincts
    import core.ops.system_monitor as system_monitor
    import interface.helpers as interface_helpers
    import interface.routes.interaction_signals as interaction_signals
    from core.runtime.service_registry import (
        get_runtime_container_health_report,
        install_container_health_report_resolver,
        install_service_factory_registration_sink,
        install_service_registration_sink,
        install_service_resolver,
    )

    class RecoveryLayer:
        def __init__(self):
            self.initialized = False

        async def initialize(self):
            self.initialized = True

    class AsyncBus:
        def __init__(self):
            self.events = []

        async def emit(self, topic, payload):
            self.events.append((topic, payload))

    class LiquidState:
        def __init__(self):
            self.updates = []

        def update(self, **kwargs):
            self.updates.append(kwargs)

    class CapabilityEngine:
        def get(self, name):
            return object() if name == "search_web" else None

        async def execute(self, *_args, **_kwargs):
            return {
                "ok": True,
                "answer": "world signal",
                "citations": [{"title": "source", "url": "https://example.test"}],
            }

    class SignalEngine:
        async def publish_typing(self, payload):
            self.typing = payload

        async def publish_voice(self, payload):
            self.voice = payload

        async def publish_vision_frame(self, frame, metadata=None):
            self.vision = (frame, metadata)

        def get_status(self):
            return {"typing": hasattr(self, "typing")}

    class Brain:
        async def think(self, *_args, **_kwargs):
            return SimpleNamespace(content="seed thought")

    class Graph:
        def __init__(self):
            self.beliefs = []

        def update_belief(self, **kwargs):
            self.beliefs.append(kwargs)

    recovery = RecoveryLayer()
    bus = AsyncBus()
    liquid = LiquidState()
    graph = Graph()
    workspace = SimpleNamespace(
        history=[
            SimpleNamespace(winner=SimpleNamespace(source="sys", content=f"event {idx}"))
            for idx in range(55)
        ]
    )
    orchestrator = SimpleNamespace(
        _last_user_interaction_time=0.0,
        dream_cycle=SimpleNamespace(process_dreams=lambda: None),
        status=SimpleNamespace(brain_connected=False),
    )
    services = {
        "executive_closure": object(),
        "causal_world_model": SimpleNamespace(get_prompt_context=lambda: "causal context"),
        "cognitive_engine": Brain(),
        "skill_router": object(),
        "cognitive_integration_layer": recovery,
        "mycelium": bus,
        "orchestrator": orchestrator,
        "liquid_state": liquid,
        "global_workspace": workspace,
        "belief_graph": graph,
        "capability_engine": CapabilityEngine(),
        "drive_engine": SimpleNamespace(satisfy=lambda *_args, **_kwargs: None),
        "state_repository": SimpleNamespace(_current=SimpleNamespace(marker="state")),
        "tricorder": SimpleNamespace(healthy=True, scan=lambda _state: {"tricorder": True}),
        "interaction_signals": SignalEngine(),
        "voice_engine": SimpleNamespace(microphone_enabled=True),
    }
    registered = []
    factories = []

    install_service_resolver(lambda name, default=None: services.get(name, default))
    install_service_registration_sink(
        lambda name, instance, required, metadata: registered.append(
            (name, instance, required, metadata)
        )
    )
    install_service_factory_registration_sink(
        lambda name, factory, lifetime, required, metadata: factories.append(
            (name, factory, lifetime, required, metadata)
        )
    )
    install_container_health_report_resolver(lambda: {"container": "ok"})
    try:
        assert get_runtime_container_health_report() == {"container": "ok"}

        assert executive_closure.ExecutiveClosurePhase()._get_engine() is services["executive_closure"]

        manager = cognitive_manager.CognitiveManager()
        asyncio.run(manager.on_start_async())
        assert manager.get_status()["initialized"] is True

        lazarus = lazarus_brainstem.LazarusBrainstem(orchestrator=orchestrator)
        assert asyncio.run(lazarus.attempt_recovery()) is True
        assert recovery.initialized is True
        assert bus.events[-1][0] == "aura.system.recovery"

        sensory = sensory_instincts.SensoryInstincts(orchestrator=orchestrator)
        assert sensory.trigger_spike("audio", 0.4, emotion="curiosity") is True
        assert liquid.updates[-1]["delta_curiosity"] == 0.4

        summarizer = attention.AttentionSummarizer(SimpleNamespace(cognitive_engine=Brain()))
        assert asyncio.run(summarizer._generate_seed_thought(workspace.history[:2])) == "seed thought"

        augmentor = web_augmentor.SovereignWebAugmentor()
        asyncio.run(augmentor.refresh_world_state(force=True))
        assert "world signal" in augmentor.world_context

        assert system_health._current_state_for_scan().marker == "state"
        assert interaction_signals._get_engine() is services["interaction_signals"]
        assert interaction_signals._microphone_signal_allowed() is True

        feed = object()
        services["neural_feed"] = feed
        assert neural_feed.get_feed() is feed

        system_monitor.register_system_monitor()
        assert any(item[0] == "system_monitor" for item in factories)

        interface_helpers._notify_user_spoke("hello")
        assert orchestrator._last_user_interaction_time >= 0.0
    finally:
        install_service_resolver(None)
        install_service_registration_sink(None)
        install_service_factory_registration_sink(None)
        install_container_health_report_resolver(None)

    for module in (
        cognitive_manager,
        lazarus_brainstem,
        web_augmentor,
        dream_coordinator,
        system_health,
        self_knowing,
        master_moral_integration,
        attention,
        neural_feed,
        core_baseline,
        consciousness_phase,
        executive_closure,
        presence_integration,
        response_policy,
        sensory_instincts,
        system_monitor,
        interface_helpers,
        interaction_signals,
    ):
        _assert_reaches_services_through_the_registry(module)


def test_runtime_registry_provider_lifetime_bridge_removes_container_imports():
    import inspect

    import core.providers.cognitive_provider as cognitive_provider
    import core.providers.consciousness_provider as consciousness_provider
    import core.providers.memory_provider as memory_provider
    import core.providers.ops_provider as ops_provider
    import core.providers.sensory_provider as sensory_provider
    from core.container import ServiceContainer
    from core.runtime.service_registry import SERVICE_LIFETIME_SINGLETON

    counter = {"count": 0}

    def factory():
        counter["count"] += 1
        return object()

    service_name = "test_runtime_registry_provider_lifetime_bridge"
    ServiceContainer.register(
        service_name,
        factory=factory,
        lifetime=SERVICE_LIFETIME_SINGLETON,
        required=False,
    )

    first = ServiceContainer.get(service_name)
    second = ServiceContainer.get(service_name)
    assert first is second
    assert counter["count"] == 1

    for module in (
        cognitive_provider,
        consciousness_provider,
        memory_provider,
        ops_provider,
        sensory_provider,
    ):
        source = inspect.getsource(module)
        assert "from core.container import ServiceLifetime" not in source
        assert "ServiceLifetime." not in source
        assert "SERVICE_LIFETIME_SINGLETON" in source
        assert "core.runtime.service_registry" in source


def test_runtime_registry_batch_five_safety_memory_morality_seams():
    import asyncio
    import inspect
    from types import SimpleNamespace

    import core.actuators.sandbox_operator as sandbox_operator
    import core.brain.ontology_genesis as ontology_genesis
    import core.capabilities.clipboard_manager as clipboard_manager
    import core.consciousness.self_report as self_report
    import core.conversation.memory as conversation_memory
    import core.identity.identity_guard as identity_guard
    import core.maintenance.dream_cycle as dream_cycle
    import core.morality.aggregate_harm as aggregate_harm
    import core.morality.honesty_governor as honesty_governor
    import core.orchestrator.handlers.recovery as recovery_handler
    import core.phases.repair_phase as repair_phase
    import core.runtime.organism_status as organism_status
    import core.safety.self_preservation_safe as self_preservation_safe
    import core.senses.ears as ears
    import core.soul as soul
    from core.runtime.service_registry import (
        install_service_registration_sink,
        install_service_resolver,
    )

    class Mycelium:
        def __init__(self):
            self.reflexes = []

        async def emit_reflex(self, topic, payload):
            self.reflexes.append((topic, payload))

    class Workspace:
        def __init__(self):
            self.candidates = []

        async def submit(self, candidate):
            self.candidates.append(candidate)

    class State:
        def __init__(self):
            self.viability = 0.9
            self.energy = 0.8
            self.integrity = 0.95
            self.degradation_events = 2

    class ResourceStakes:
        def state(self):
            return State()

        def action_envelope(self, _mode):
            return SimpleNamespace(as_dict=lambda: {"mode": "normal"})

    mycelium = Mycelium()
    workspace = Workspace()
    services = {
        "aura_state": SimpleNamespace(identity=SimpleNamespace(name="Aura")),
        "mycelial_network": mycelium,
        "voice_engine": SimpleNamespace(should_auto_listen=lambda: True),
        "homeostasis": SimpleNamespace(anxiety=0.25),
        "aura_kernel": SimpleNamespace(volition_level=3),
        "global_workspace": workspace,
        "self_prediction": SimpleNamespace(get_surprise_signal=lambda: 0.6),
        "resource_stakes": ResourceStakes(),
    }
    registered = []
    install_service_resolver(lambda name, default=None: services.get(name, default))
    install_service_registration_sink(
        lambda name, instance, required, metadata: registered.append(
            (name, instance, required, metadata)
        )
    )
    try:
        cb = clipboard_manager.ClipboardManager()
        asyncio.run(cb.start())
        assert any(item[0] == "clipboard_manager" for item in registered)

        gate = identity_guard.PersonaEnforcementGate()
        ok, reason, _score = gate.validate_output("I am Aura.", enforce_supervision=False)
        assert ok is True and reason == "OK"

        e = ears.SovereignEars()
        e.capabilities.hearing_enabled = True
        assert e.should_auto_listen() is True

        genesis = ontology_genesis.register_ontology_genesis()
        assert genesis._get_resource_anxiety() == 0.25
        assert any(item[0] == "ontology_genesis" for item in registered)

        assert honesty_governor.register_honesty_governor().get_status()["healthy"] is True
        assert aggregate_harm.register_aggregate_harm().get_status()["healthy"] is True

        status = organism_status.get_organism_status(orchestrator=None)
        assert status["resource_stakes"]["viability"] == 0.9

        s = soul.Soul(SimpleNamespace(boredom=0.1))
        assert s.get_dominant_drive().name == "curiosity"

        connection = soul.Drive("connection", 1.0, "test")
        asyncio.run(s.satisfy_drive(connection))
        assert workspace.candidates
    finally:
        install_service_resolver(None)
        install_service_registration_sink(None)

    for module in (
        sandbox_operator,
        ontology_genesis,
        clipboard_manager,
        self_report,
        conversation_memory,
        identity_guard,
        dream_cycle,
        aggregate_harm,
        honesty_governor,
        recovery_handler,
        repair_phase,
        organism_status,
        self_preservation_safe,
        ears,
        soul,
    ):
        _assert_reaches_services_through_the_registry(
            module,
            forbid_names=("ServiceContainer", "get_container"),
            # repair_phase reaches services another way and always has.
            require_import=(
                None if module is repair_phase else "core.runtime.service_registry"
            ),
        )


def test_runtime_registry_batch_six_large_scc_service_seams(monkeypatch, tmp_path):
    import asyncio
    import inspect
    from types import SimpleNamespace

    import core.actuators.doc_ingest as doc_ingest
    import core.agency.goal_planner as goal_planner
    import core.agency.self_play as self_play
    import core.brain.deliberation as deliberation
    import core.brain.inference_feedback as inference_feedback
    import core.brain.llm.compiler as compiler
    import core.collective.strategic_synthesis as strategic_synthesis
    import core.consciousness.predictive_engine as predictive_engine
    import core.embodiment.voice_presence as voice_presence
    import core.environment.embodied_simulator as embodied_simulator
    import core.goals.directive_conflict_sentinel as directive_conflict_sentinel
    import core.governance.need_to_know as need_to_know
    import core.guardians.memory_guard as memory_guard
    import core.guardians.threat_watch as threat_watch
    import core.knowledge.bottling as bottling
    import core.managers.memory_manager as memory_manager
    import core.memory.black_hole as black_hole
    import core.memory.sovereign_pruner as sovereign_pruner
    import core.orchestrator.handlers.aegis as aegis
    import core.phases.bonding_phase as bonding_phase
    import core.phases.inference_phase as inference_phase
    import core.phases.initiative_generation as initiative_generation
    import core.phases.motivation_update as motivation_update
    import core.reliability_engine as reliability_engine
    import core.scheduler as scheduler
    import core.sim.outcome_simulator as outcome_simulator
    import core.sim.scenario_forge as scenario_forge
    import core.state.state_authority as state_authority
    import core.values.values_engine as values_engine
    import core.voice.voice_session as voice_session
    from core.runtime.service_registry import (
        install_registration_locked_resolver,
        install_service_factory_registration_sink,
        install_service_presence_resolver,
        install_service_registration_sink,
        install_service_resolver,
    )
    from core.state.aura_state import AuraState

    class Memory:
        def query_knowledge(self, topic):
            return "remembered truth" if topic == "topic" else None

        def recall(self, _topic):
            return None

    class VectorMemory:
        def retrieve_context(self, _topic, top_k=1):
            return [{"content": "vector truth"}]

        def search(self, _query, limit=5):
            return [{"score": 0.9, "content": "match"}]

        def search_similar(self, _query, limit=5, **_kwargs):
            return [{"score": 0.9}]

    class Router:
        high_pressure_mode = True

        async def think(self, *_args, **_kwargs):
            return (
                '{"implicit_intent":"answer directly","user_subtext":"needs clarity",'
                '"momentum":"flowing","conversation_hooks":["clarity"]}'
            )

    class Gate:
        async def generate_response(self, prompt, **_kwargs):
            return f"generated:{prompt[:8]}"

        def _background_local_deferral_reason(self, *, origin):
            return f"{origin}:deferred"

    class Tom:
        known_selves = {"bryan": SimpleNamespace(rapport=0.8)}

    class Mycelium:
        def __init__(self):
            self.pulsed = []

        def pulse_hypha(self, source, target, *, success=True):
            self.pulsed.append((source, target, success))
            return True

    class TTS:
        def __init__(self):
            self.messages = []

        async def speak(self, message):
            self.messages.append(message)

    reliability = object()
    mycelium = Mycelium()
    tts = TTS()
    services = {
        "reliability_engine": reliability,
        "memory": Memory(),
        "vector_memory": VectorMemory(),
        "llm_router": Router(),
        "inference_gate": Gate(),
        "theory_of_mind": Tom(),
        "mycelial_network": mycelium,
        "tts_engine": tts,
        "free_energy_engine": SimpleNamespace(accept_surprise_signal=lambda *_args, **_kwargs: None),
    }
    registered: list[tuple[str, object, bool, dict[str, str | None]]] = []
    factories: list[tuple[str, object, object, bool, dict[str, str | None]]] = []

    monkeypatch.setattr(scheduler.Scheduler, "_instance", None)
    install_service_resolver(lambda name, default=None: services.get(name, default))
    install_service_presence_resolver(lambda name: name in services)
    install_registration_locked_resolver(lambda: False)
    install_service_registration_sink(
        lambda name, instance, required, metadata: registered.append(
            (name, instance, required, metadata)
        )
    )
    install_service_factory_registration_sink(
        lambda name, factory, lifetime, required, metadata: factories.append(
            (name, factory, lifetime, required, metadata)
        )
    )
    try:
        assert reliability_engine.get_reliability_engine() is reliability

        compiler.register_prompt_compiler()
        state_authority.register_state_authority()
        assert any(item[0] == "prompt_compiler" for item in factories)
        assert any(item[0] == "state_authority" for item in factories)

        authority = state_authority.StateAuthority()
        assert authority._check_knowledge_base("topic") == "remembered truth"
        assert authority._check_vector_memory("missing") == "vector truth"

        state = AuraState.default()
        state.cognition.current_origin = "user"
        inferred = asyncio.run(inference_phase.InferencePhase().execute(state, objective="help me"))
        assert inferred.cognition.modifiers["inferred_intent"] == "answer directly"

        bonded = AuraState.default()
        bonded.cognition.current_origin = "user"
        bonded = asyncio.run(bonding_phase.BondingPhase().execute(bonded, objective="A personal note " * 20))
        assert bonded.identity.bonding_level > 0

        assert initiative_generation.InitiativeGenerationPhase._autonomy_pause_reason() == "memory_pressure"
        assert motivation_update._background_curiosity_allowed() in {True, False}

        manager = memory_manager.MemoryManager()
        assert manager._get_mycelium() is mycelium
        assert manager.search_similar("query") == [{"score": 0.9}]

        session = voice_session.VoiceSessionManager()
        asyncio.run(session.start())
        session.begin_session("hello")
        asyncio.run(session.narrate("checking voice"))
        assert any(item[0] == "voice_session" for item in registered)
        assert tts.messages == ["checking voice"]

        scheduler.Scheduler()
        assert any(item[0] == "scheduler" for item in registered)

        # BlackHole without Horcrux must come up ENCRYPTING, not disabled.
        # This previously asserted `_aesgcm is None` — i.e. it pinned the defect:
        # "Horcrux unavailable" meant encryption silently off, and every memory
        # written on that boot went to disk in the clear. on_start() now
        # provisions a local key instead (weaker than Horcrux and reported as
        # such via key_provenance, but never plaintext).
        #
        # The key dir is redirected because on_start() genuinely writes one, and
        # a service-seam test must not leave key material in the live ~/.aura.
        monkeypatch.setenv(
            "AURA_BLACK_HOLE_KEY_DIR", str(tmp_path / "black_hole_keys")
        )
        black = black_hole.BlackHole()
        black.on_start()
        assert black._aesgcm is not None
        assert black.key_provenance == "local"
        assert black.encrypt(b"probe") != b"probe"

        assert goal_planner.GoalPlanner().classify("explain this") == "reasoning"
        assert asyncio.run(goal_planner.GoalPlanner()._default_generate("question", 0.2)).startswith("generated:")

        assert outcome_simulator.register_outcome_simulator().get_status()["healthy"] is True
        assert bottling.register_knowledge_bottling().get_status()["healthy"] is True
        assert scenario_forge.register_scenario_forge().get_status()["healthy"] is True
        assert threat_watch.register_threat_watch().get_status()["healthy"] is True
        assert directive_conflict_sentinel.register_directive_sentinel().get_status()["healthy"] is True
        assert need_to_know.register_need_to_know().get_status()["healthy"] is True

        registered_names = [item[0] for item in registered]
        assert "culture_mind" in registered_names
        assert "brainiac" in registered_names
        assert "caine" in registered_names
        assert "safe_surf" in registered_names
        assert "hal" in registered_names
        assert "the_machine" in registered_names
    finally:
        install_service_resolver(None)
        install_service_presence_resolver(None)
        install_registration_locked_resolver(None)
        install_service_registration_sink(None)
        install_service_factory_registration_sink(None)

    for module in (
        doc_ingest,
        goal_planner,
        self_play,
        deliberation,
        inference_feedback,
        compiler,
        strategic_synthesis,
        predictive_engine,
        voice_presence,
        embodied_simulator,
        directive_conflict_sentinel,
        need_to_know,
        memory_guard,
        threat_watch,
        bottling,
        memory_manager,
        black_hole,
        sovereign_pruner,
        aegis,
        # bonding_phase left this batch in 2961f214: rapport now derives
        # from bounded internal state — no service resolution remains, so
        # the registry-seam source contract no longer applies to it.
        inference_phase,
        initiative_generation,
        # motivation_update is asserted separately below: three of its four
        # service reads moved to the registry, and the fourth cannot, because
        # the registry resolves with `peek` and the drive engine is registered
        # lazily. A lifecycle caller that must instantiate is a different
        # thing from a phase that reached past the seam.
        reliability_engine,
        scheduler,
        outcome_simulator,
        scenario_forge,
        state_authority,
        values_engine,
        voice_session,
    ):
        _assert_reaches_services_through_the_registry(
            module,
            forbid_names=("ServiceContainer", "ServiceLifetime", "get_container"),
        )


def test_motivation_update_reads_through_the_registry_and_instantiates_once():
    """Every observation through the seam; the one lifecycle call named.

    `get_runtime_service` resolves with `ServiceContainer.peek`, which never
    invokes a factory — right for diagnostics and error sinks, which must not
    boot an organ while they are looking at one. Three reads here are exactly
    that and now go through it: the free-energy urgency, the soma exertion,
    and the world-model surprise. Each of them used to be able to instantiate
    the thing it was reading.

    The fourth gathers drive signals, and the drive engine is registered
    lazily, so the read-only seam returns None for ever. That one stays on the
    container, and this test pins the count so a second one cannot appear
    beside it without a reader noticing.
    """
    import ast
    import inspect

    import core.phases.motivation_update as motivation_update

    source = inspect.getsource(motivation_update)
    assert "from core.runtime.service_registry import" in source
    assert source.count("get_runtime_service(") >= 4

    tree = ast.parse(source)
    container_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "core.container"
    ]
    assert len(container_imports) == 1, (
        f"{len(container_imports)} container imports; the lifecycle caller is "
        "one, and an observer belongs on the registry"
    )
    resolves = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ServiceContainer"
    ]
    assert [call.args[0].value for call in resolves] == ["drive_integration"]


def test_runtime_registry_batch_seven_consciousness_adaptation_seams(monkeypatch, tmp_path):
    import asyncio
    import inspect
    from types import SimpleNamespace

    import core.adaptation.abstraction_engine as abstraction_engine
    import core.adaptation.epistemic_humility as epistemic_humility
    import core.adaptation.heuristic_synthesizer as heuristic_synthesizer
    import core.adaptation.immune_system as immune_system
    import core.affect.affective_resonance as affective_resonance
    import core.agency.canvas_manager as canvas_manager
    import core.autonomy.sleep_trigger as sleep_trigger
    import core.autonomy.autonomy_guardian as autonomy_guardian
    import core.brain.causal_world_model as causal_world_model
    import core.brain.deep_deliberation as deep_deliberation
    import core.brain.discourse_tracker as discourse_tracker
    import core.brain.narrative_memory as narrative_memory
    import core.brain.predictive_engine as predictive_engine
    import core.consciousness.evidence_engine as evidence_engine
    import core.consciousness.integration as integration
    import core.consciousness.liquid_substrate_bridge as liquid_substrate_bridge
    import core.consciousness.resource_stakes as resource_stakes
    import core.consciousness.world_model as consciousness_world_model
    import core.final_engines as final_engines
    import core.cognition.meta_cognition as meta_cognition
    import core.orchestrator.initializers.hardening as hardening
    import core.orchestrator.mixins.cognitive_background as cognitive_background
    import core.pneuma.pneuma as pneuma
    import core.pneuma.precision_engine as precision_engine
    import core.self_modification.kernel_refiner as kernel_refiner
    import core.senses.screen_vision as screen_vision
    import core.sovereignty.integrity_guard as integrity_guard
    import core.planning.strategic_planner as strategic_planner
    import core.world_model.acg as acg
    import interface.routes.privacy as privacy
    from core.runtime.service_registry import (
        SERVICE_LIFETIME_SINGLETON,
        install_registration_locked_resolver,
        install_service_factory_registration_sink,
        install_service_presence_resolver,
        install_service_registration_sink,
        install_service_resolver,
    )
    from core.service_names import ServiceNames
    from core.state.aura_state import AuraState

    registered: list[tuple[str, object, bool, dict[str, str | None]]] = []
    factories: list[tuple[str, object, object, bool, dict[str, str | None]]] = []

    class Router:
        async def think(self, *args, **kwargs):
            prompt = kwargs.get("prompt") or (args[0] if args else "")
            if "Compare these two" in str(prompt):
                return "0.25"
            return "topic response"

    class CognitiveEngine:
        async def think(self, **_kwargs):
            return SimpleNamespace(content="first principle")

    class MemoryFacade:
        def __init__(self):
            self.stored: list[tuple[str, dict[str, str]]] = []

        def query_memory_sync(self, query, limit=1):
            if query == "type:narrative_arc":
                return [{"text": "Aura learned from a prior exchange."}]
            return []

        def store(self, *, content, metadata):
            self.stored.append((content, metadata))

    class AffectEngine:
        def __init__(self):
            self.applied: list[tuple[str, float]] = []

        async def apply_stimulus(self, stimulus, intensity):
            self.applied.append((stimulus, intensity))

    class Workspace:
        ignition_level = 0.7
        ignited = True
        current_phi = 0.4

    class FakeExperiencer:
        phenomenal_context_string = "felt continuity marker"

        def __init__(self):
            self.started = False
            self.refs = None
            self.broadcasts = []

        def set_refs(self, **kwargs):
            self.refs = kwargs

        async def start(self):
            self.started = True

        async def stop(self):
            self.started = False

        def on_broadcast(self, snap):
            self.broadcasts.append(snap)

        def get_status(self):
            return {"started": self.started}

    class GlobalWorkspace:
        def __init__(self):
            self.subscribers = []

        def subscribe(self, callback):
            self.subscribers.append(callback)

    services = {
        "llm_router": Router(),
        "cognitive_engine": CognitiveEngine(),
        "memory_facade": MemoryFacade(),
        "affect_engine": AffectEngine(),
        "orchestrator": SimpleNamespace(
            conversation_history=[{"role": "user", "content": "hello"}],
            reply_queue=SimpleNamespace(__class__=SimpleNamespace(__name__="TaggedReplyQueue")),
            _inference_gate=SimpleNamespace(is_inference_ready=lambda: True),
            agency=True,
            status=SimpleNamespace(last_error=None),
        ),
        "personality_engine": SimpleNamespace(
            get_emotional_context_for_response=lambda: {"dominant_emotions": ["curiosity"]}
        ),
        "self_report_engine": SimpleNamespace(generate_state_report=lambda: "state report"),
        "phenomenological_experiencer": SimpleNamespace(
            phenomenal_context_string="phenomenal fragment",
            to_dict=lambda: {"is_stale": False},
        ),
        "self_model": object(),
        "global_workspace": Workspace(),
        "homeostasis": object(),
        "opinion_engine": object(),
        "spine": object(),
        "volition_engine": object(),
        "executive_closure": SimpleNamespace(get_status=lambda: {"closure_score": 0.8}),
        "liquid_state": object(),
    }

    monkeypatch.setattr(
        final_engines,
        "WorldModelEngine",
        lambda: SimpleNamespace(name="world"),
    )
    monkeypatch.setattr(
        final_engines,
        "NarrativeIdentityEngine",
        lambda: SimpleNamespace(name="identity"),
    )
    monkeypatch.setattr(
        final_engines,
        "MetacognitiveCalibrator",
        lambda: SimpleNamespace(name="metacognition"),
    )
    monkeypatch.setattr(
        causal_world_model,
        "CausalWorldModel",
        lambda: SimpleNamespace(name="causal"),
    )
    monkeypatch.setattr(
        epistemic_humility,
        "EpistemicHumility",
        lambda orchestrator: SimpleNamespace(orchestrator=orchestrator),
    )

    install_service_resolver(lambda name, default=None: services.get(name, default))
    install_service_presence_resolver(lambda name: name in services)
    install_registration_locked_resolver(lambda: False)
    install_service_registration_sink(
        lambda name, instance, required, metadata: registered.append(
            (name, instance, required, metadata)
        )
    )
    install_service_factory_registration_sink(
        lambda name, factory, lifetime, required, metadata: factories.append(
            (name, factory, lifetime, required, metadata)
        )
    )
    try:
        aff = affective_resonance.register_affective_resonance()
        deep = deep_deliberation.register_deep_deliberation()
        epistemic_humility.register_epistemic_humility(SimpleNamespace(name="orch"))
        final_engines.register_final_engines()
        causal_world_model.register_causal_world_model()
        abstraction_engine.register_abstraction_engine()

        registered_names = [item[0] for item in registered]
        assert ServiceNames.SAMANTHA in registered_names
        assert "samantha" in registered_names
        assert ServiceNames.DEEP_THOUGHT in registered_names
        assert "deep_thought" in registered_names
        assert "epistemic_humility" in registered_names
        assert "world_model" in registered_names
        assert "narrative_identity" in registered_names
        assert "metacognitive_calibrator" in registered_names
        assert "causal_world_model" in registered_names
        assert aff.get_status()["healthy"] is True
        assert deep.refine_question("fix this").startswith("fix this")

        assert factories[0][0] == "abstraction_engine"
        assert factories[0][2] == SERVICE_LIFETIME_SINGLETON

        pred = predictive_engine.PredictiveEngine()
        assert pred._get_router() is services["llm_router"]
        state = AuraState.default()
        state.cognition.working_memory.append({"role": "user", "content": "hello"})
        prediction = asyncio.run(pred.predict(state))
        error = asyncio.run(pred.evaluate(prediction, "actual answer", state))
        assert error.error_magnitude == 0.25

        tracker = discourse_tracker.DiscourseTracker()
        assert tracker._get_brain() is services["cognitive_engine"]

        narrative = narrative_memory.NarrativeEngine(SimpleNamespace(cognitive_engine=None))
        assert "Aura learned" in narrative.get_narrative_context()

        snapshot = evidence_engine.ConsciousnessEvidenceEngine().snapshot()
        assert snapshot["subjectivity_evidence"] > 0.0
        assert snapshot["dimensions"]["reliability"] > 0.0

        fake_experiencer = FakeExperiencer()
        monkeypatch.setattr(integration, "get_experiencer", lambda: fake_experiencer)
        global_workspace = GlobalWorkspace()
        consciousness = integration.ConsciousnessIntegration(
            SimpleNamespace(
                affect_module="affect",
                liquid_substrate="substrate",
                drive_engine="drives",
                credit_engine="credit",
                global_workspace=global_workspace,
            )
        )
        asyncio.run(consciousness.initialize())
        assert fake_experiencer.started is True
        assert fake_experiencer.refs["substrate"] == "substrate"
        assert len(global_workspace.subscribers) == 1

        graph = acg.ActionConsequenceGraph(persist_path=tmp_path / "aura-test-acg.json")
        graph._save = lambda *args, **kwargs: None
        graph.record_outcome({"tool": "inspect", "params": {"path": "x"}}, "ctx", "ok", True)
        assert graph.query_consequences("inspect", {"path": "x"})
    finally:
        install_service_resolver(None)
        install_service_presence_resolver(None)
        install_registration_locked_resolver(None)
        install_service_registration_sink(None)
        install_service_factory_registration_sink(None)

    for module in (
        liquid_substrate_bridge,
        screen_vision,
        acg,
        affective_resonance,
        deep_deliberation,
        predictive_engine,
        integrity_guard,
        discourse_tracker,
        autonomy_guardian,
        evidence_engine,
        cognitive_background,
        consciousness_world_model,
        final_engines,
        pneuma,
        kernel_refiner,
        strategic_planner,
        sleep_trigger,
        meta_cognition,
        epistemic_humility,
        resource_stakes,
        canvas_manager,
        immune_system,
        integration,
        hardening,
        abstraction_engine,
        precision_engine,
        privacy,
        heuristic_synthesizer,
        causal_world_model,
        narrative_memory,
    ):
        _assert_reaches_services_through_the_registry(
            module,
            forbid_names=("ServiceContainer", "ServiceLifetime", "get_container"),
        )
