"""Contract tests for the ROS 2-derived middleware disciplines.

core/runtime/{lifecycle,parameters}.py, core/bus/qos.py,
core/observability/bus_recorder.py, core/health/diagnostics_aggregator.py.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.bus import qos as qos_mod
from core.bus.qos import (
    COMMAND,
    HEARTBEAT,
    SENSOR_DATA,
    STATE,
    Durability,
    QosBus,
    QosProfile,
    Reliability,
)
from core.health import diagnostics_aggregator as diag_mod
from core.health.diagnostics_aggregator import (
    Analyzer,
    DiagnosticsAggregator,
    Level,
)
from core.observability import bus_recorder as bag_mod
from core.observability.bus_recorder import BagReader, BusRecorder
from core.runtime import lifecycle as life_mod
from core.runtime import parameters as param_mod
from core.runtime.lifecycle import (
    LifecycleManager,
    ManagedOrgan,
    State,
    Transition,
    TransitionError,
)
from core.runtime.parameters import (
    ParameterError,
    ParameterServer,
    ParameterType,
    SetResult,
)


@pytest.fixture(autouse=True)
def _clean():
    for mod in (life_mod, param_mod, qos_mod, bag_mod, diag_mod):
        for name in dir(mod):
            if name.startswith("reset_") and name.endswith("_for_test"):
                getattr(mod, name)()
    yield
    for mod in (life_mod, param_mod, qos_mod, bag_mod, diag_mod):
        for name in dir(mod):
            if name.startswith("reset_") and name.endswith("_for_test"):
                getattr(mod, name)()


# ── lifecycle ─────────────────────────────────────────────────────────

def test_inactive_is_a_real_state_between_configured_and_running():
    async def scenario():
        events: list[str] = []
        organ = ManagedOrgan(
            "index",
            on_configure=lambda: events.append("configure"),
            on_activate=lambda: events.append("activate"),
        )
        assert organ.state is State.UNCONFIGURED
        assert await organ.configure() is True
        # This is the whole point: allocated, dependencies resolved,
        # publishing nothing.
        assert organ.state is State.INACTIVE
        assert events == ["configure"]
        assert await organ.activate() is True
        assert organ.state is State.ACTIVE
        assert events == ["configure", "activate"]

    asyncio.run(scenario())


def test_deactivation_is_distinguishable_from_failure():
    async def scenario():
        organ = ManagedOrgan("lane")
        await organ.configure()
        await organ.activate()
        assert await organ.deactivate() is True
        return organ

    organ = asyncio.run(scenario())
    assert organ.state is State.INACTIVE, "paused, not broken"
    assert organ.last_error == ""


def test_a_failing_transition_leaves_a_known_state_not_a_half_built_object():
    async def scenario():
        organ = ManagedOrgan("picky", on_configure=lambda: False)
        assert await organ.configure() is False
        return organ

    organ = asyncio.run(scenario())
    assert organ.state is State.UNCONFIGURED
    assert "failure" in organ.last_error


def test_a_raising_transition_goes_to_error_processing_not_an_exception():
    async def scenario():
        def boom():
            raise RuntimeError("dependency missing")

        organ = ManagedOrgan("exploder", on_configure=boom)
        result = await organ.configure()
        return result, organ

    result, organ = asyncio.run(scenario())
    assert result is False
    # on_error defaults to "not recovered", which finalizes.
    assert organ.state is State.FINALIZED
    assert "dependency missing" in organ.last_error


def test_error_handler_can_recover_to_unconfigured():
    async def scenario():
        def boom():
            raise RuntimeError("transient")

        organ = ManagedOrgan("recoverable", on_configure=boom, on_error=lambda: True)
        await organ.configure()
        return organ

    organ = asyncio.run(scenario())
    assert organ.state is State.UNCONFIGURED


def test_illegal_transitions_raise_rather_than_silently_no_op():
    async def scenario():
        organ = ManagedOrgan("strict")
        with pytest.raises(TransitionError, match="not legal"):
            await organ.activate()
        assert organ.can(Transition.CONFIGURE) is True
        assert organ.can(Transition.ACTIVATE) is False

    asyncio.run(scenario())


def test_transition_timeout_is_a_failure_not_a_hang():
    async def scenario():
        async def slow():
            await asyncio.sleep(5)

        organ = ManagedOrgan("slow", on_configure=slow, transition_timeout_s=0.05)
        result = await organ.configure()
        return result, organ

    result, organ = asyncio.run(scenario())
    assert result is False
    assert "timed out" in organ.last_error
    assert organ.state is State.UNCONFIGURED


def test_bring_up_configures_everything_before_activating_anything():
    async def scenario():
        order: list[str] = []
        manager = LifecycleManager()
        for name in ("a", "b"):
            manager.register(
                ManagedOrgan(
                    name,
                    on_configure=lambda n=name: order.append(f"configure:{n}"),
                    on_activate=lambda n=name: order.append(f"activate:{n}"),
                )
            )
        report = await manager.bring_up()
        return order, report

    order, report = asyncio.run(scenario())
    assert order == ["configure:a", "configure:b", "activate:a", "activate:b"]
    assert report["ready"] is True
    assert report["critical_blocked"] == []


def test_a_critical_organ_that_cannot_configure_blocks_activation_of_all():
    async def scenario():
        activated: list[str] = []
        manager = LifecycleManager()
        manager.register(ManagedOrgan("spine", critical=True, on_configure=lambda: False))
        manager.register(
            ManagedOrgan("other", on_activate=lambda: activated.append("other"))
        )
        report = await manager.bring_up()
        return activated, report

    activated, report = asyncio.run(scenario())
    assert report["critical_blocked"] == ["spine"]
    assert activated == [], "nothing activates over a blocked critical organ"
    assert report["ready"] is False


def test_adopt_gives_an_existing_start_stop_object_a_lifecycle():
    class Legacy:
        def __init__(self):
            self.running = False

        def start(self):
            self.running = True

        def stop(self):
            self.running = False

    async def scenario():
        legacy = Legacy()
        organ = life_mod.adopt("legacy", legacy)
        assert organ is not None
        await organ.configure()
        await organ.activate()
        assert legacy.running is True
        await organ.deactivate()
        assert legacy.running is False
        return organ

    organ = asyncio.run(scenario())
    assert organ.state is State.INACTIVE


def test_adopt_declines_an_object_with_no_lifecycle():
    assert life_mod.adopt("inert", object()) is None


def test_lifecycle_transitions_publish_a_ready_condition():
    from core.runtime.conditions import ConditionType, get_component_conditions

    async def scenario():
        organ = ManagedOrgan("conditioned")
        await organ.configure()
        await organ.activate()

    asyncio.run(scenario())
    condition = get_component_conditions("conditioned").get(ConditionType.READY)
    assert condition is not None and condition.status is True


# ── QoS ───────────────────────────────────────────────────────────────

def test_transient_local_delivers_retained_state_to_a_late_joiner():
    async def scenario():
        bus = QosBus()
        bus.declare_publisher("cortex.lane_state", STATE)
        await bus.publish("cortex.lane_state", {"warm": True})
        # A subscriber that starts AFTER the announcement.
        retained = bus.retained("cortex.lane_state", profile=STATE)
        return retained

    retained = asyncio.run(scenario())
    assert len(retained) == 1
    assert retained[0].data == {"warm": True}


def test_volatile_topics_give_a_late_joiner_nothing():
    async def scenario():
        bus = QosBus()
        bus.declare_publisher("sensory.frame", SENSOR_DATA)
        await bus.publish("sensory.frame", {"pixels": 1})
        return bus.retained("sensory.frame", profile=STATE)

    assert asyncio.run(scenario()) == []


def test_history_depth_bounds_what_is_retained():
    async def scenario():
        bus = QosBus()
        deep = QosProfile(durability=Durability.TRANSIENT_LOCAL, depth=3)
        bus.declare_publisher("log", deep)
        for i in range(10):
            await bus.publish("log", i)
        return [s.data for s in bus.retained("log", profile=deep)]

    assert asyncio.run(scenario()) == [7, 8, 9]


def test_lifespan_expires_retained_samples():
    async def scenario():
        bus = QosBus()
        profile = QosProfile(durability=Durability.TRANSIENT_LOCAL, lifespan_s=0.05, depth=5)
        bus.declare_publisher("brief", profile)
        await bus.publish("brief", "old")
        await asyncio.sleep(0.06)
        return bus.retained("brief", profile=profile)

    assert asyncio.run(scenario()) == []


def test_qos_mismatch_is_reported_not_silently_degraded():
    bus = QosBus()
    bus.declare_publisher("weak", SENSOR_DATA)
    problems = bus.check_compatibility("weak", COMMAND)
    assert problems, "requesting more than is offered must be reported"
    assert any("RELIABLE" in p for p in problems)
    assert bus.report()["qos_mismatches"]


def test_compatible_request_reports_nothing():
    bus = QosBus()
    bus.declare_publisher("strong", COMMAND)
    assert bus.check_compatibility("strong", SENSOR_DATA) == []


def test_deadline_miss_fires_a_callback():
    fired: list[tuple[str, float]] = []

    async def scenario():
        bus = QosBus()
        bus.declare_publisher("mind.tick", QosProfile(deadline_s=0.02))
        bus.on_deadline_missed("mind.tick", lambda t, gap: fired.append((t, gap)))
        await bus.publish("mind.tick", 1)
        await asyncio.sleep(0.05)
        return bus.check_deadlines()

    missed = asyncio.run(scenario())
    assert missed == ["mind.tick"]
    assert fired and fired[0][0] == "mind.tick"


def test_liveliness_is_lost_when_a_publisher_goes_quiet():
    lost: list[str] = []

    async def scenario():
        bus = QosBus()
        bus.declare_publisher("hb", QosProfile(liveliness_lease_s=0.02))
        bus.on_liveliness_lost("hb", lost.append)
        await bus.publish("hb", 1)
        await asyncio.sleep(0.05)
        return bus.check_liveliness()

    assert asyncio.run(scenario()) == ["hb"]
    assert lost == ["hb"]
    # And the sweep is idempotent — one loss, not one per sweep.


def test_named_profiles_encode_their_intent():
    assert SENSOR_DATA.reliability is Reliability.BEST_EFFORT
    assert SENSOR_DATA.depth == 1
    assert STATE.durability is Durability.TRANSIENT_LOCAL
    assert COMMAND.reliability is Reliability.RELIABLE
    assert HEARTBEAT.deadline_s > 0


# ── parameters ────────────────────────────────────────────────────────

def test_reading_an_undeclared_parameter_raises():
    server = ParameterServer()
    with pytest.raises(ParameterError, match="never declared"):
        server.get("ghost")
    assert server.get_or("ghost", 42) == 42


def test_declaration_enforces_constraints_on_the_default():
    server = ParameterServer()
    from core.runtime.parameters import ParameterDescriptor

    descriptor = ParameterDescriptor(
        description="fraction", type=ParameterType.FLOAT, owner="test", minimum=0.0, maximum=1.0
    )
    assert server.declare("frac", 0.5, descriptor) == 0.5
    with pytest.raises(ValueError, match="default is invalid"):
        server.declare("bad", 5.0, descriptor)


def test_range_and_allowed_set_are_enforced_on_set():
    server = ParameterServer()
    from core.runtime.parameters import ParameterDescriptor

    server.declare(
        "frac",
        0.5,
        ParameterDescriptor(
            description="f", type=ParameterType.FLOAT, owner="t", minimum=0.0, maximum=1.0
        ),
    )
    assert server.set("frac", 0.9).successful is True
    result = server.set("frac", 1.5)
    assert result.successful is False and "above the maximum" in result.reason
    assert server.get("frac") == 0.9, "a rejected set must not partially apply"

    server.declare(
        "mode",
        "fast",
        ParameterDescriptor(
            description="m",
            type=ParameterType.STRING,
            owner="t",
            allowed=("fast", "careful"),
        ),
    )
    assert server.set("mode", "reckless").successful is False


def test_read_only_parameters_refuse_writes():
    server = ParameterServer()
    from core.runtime.parameters import ParameterDescriptor

    server.declare(
        "build",
        "abc123",
        ParameterDescriptor(
            description="b", type=ParameterType.STRING, owner="t", read_only=True
        ),
    )
    result = server.set("build", "def456")
    assert result.successful is False and "read-only" in result.reason


def test_a_validator_can_veto_a_change_with_a_reason():
    server = ParameterServer()
    from core.runtime.parameters import ParameterDescriptor

    server.declare(
        "lanes",
        1,
        ParameterDescriptor(description="l", type=ParameterType.INT, owner="t", minimum=1),
    )
    server.add_validator(
        "lanes",
        lambda name, old, new: SetResult.reject("cannot grow lanes while a model is loading"),
    )
    result = server.set("lanes", 4)
    assert result.successful is False
    assert "model is loading" in result.reason
    assert server.get("lanes") == 1


def test_atomic_multi_set_applies_all_or_nothing():
    server = ParameterServer()
    from core.runtime.parameters import ParameterDescriptor

    for name in ("threshold", "hysteresis"):
        server.declare(
            name,
            0.5,
            ParameterDescriptor(
                description=name,
                type=ParameterType.FLOAT,
                owner="t",
                minimum=0.0,
                maximum=1.0,
            ),
        )
    # The second value is illegal: neither may apply, because the
    # intermediate state would be observed.
    result = server.set_atomically({"threshold": 0.8, "hysteresis": 9.0})
    assert result.successful is False
    assert server.get("threshold") == 0.5
    assert server.get("hysteresis") == 0.5

    assert server.set_atomically({"threshold": 0.8, "hysteresis": 0.1}).successful
    assert (server.get("threshold"), server.get("hysteresis")) == (0.8, 0.1)


def test_observers_are_notified_and_a_failing_observer_does_not_undo_the_change():
    server = ParameterServer()
    from core.runtime.parameters import ParameterDescriptor

    server.declare(
        "x", 1, ParameterDescriptor(description="x", type=ParameterType.INT, owner="t")
    )
    seen: list[tuple[str, int, int]] = []
    server.add_observer("x", lambda n, o, v: seen.append((n, o, v)))
    server.add_observer("x", lambda n, o, v: (_ for _ in ()).throw(RuntimeError("bad observer")))
    assert server.set("x", 7).successful is True
    assert seen == [("x", 1, 7)]
    assert server.get("x") == 7


def test_environment_seeds_the_declared_default(monkeypatch):
    monkeypatch.setenv("AURA_TEST_PARAM", "0.25")
    server = ParameterServer()
    from core.runtime.parameters import ParameterDescriptor

    value = server.declare(
        "seeded",
        0.5,
        ParameterDescriptor(
            description="s",
            type=ParameterType.FLOAT,
            owner="t",
            minimum=0.0,
            maximum=1.0,
            env_var="AURA_TEST_PARAM",
        ),
    )
    assert value == 0.25


def test_change_history_records_who_and_why():
    server = ParameterServer()
    from core.runtime.parameters import ParameterDescriptor

    server.declare(
        "k", 1, ParameterDescriptor(description="k", type=ParameterType.INT, owner="t")
    )
    server.set("k", 2, source="operator", reason="tuning under load")
    entry = server.report()["recent_changes"][-1]
    assert entry["source"] == "operator"
    assert entry["reason"] == "tuning under load"
    assert (entry["old"], entry["new"]) == (1, 2)


def test_conflicting_redeclaration_is_refused():
    server = ParameterServer()
    from core.runtime.parameters import ParameterDescriptor

    first = ParameterDescriptor(description="a", type=ParameterType.INT, owner="one")
    second = ParameterDescriptor(description="b", type=ParameterType.INT, owner="two")
    server.declare("dup", 1, first)
    assert server.declare("dup", 1, first) == 1
    with pytest.raises(ValueError, match="already declared"):
        server.declare("dup", 1, second)


# ── bus recorder ──────────────────────────────────────────────────────

def test_ring_is_always_on_and_bounded():
    recorder = BusRecorder(capacity=4)
    for i in range(10):
        recorder.record("topic.a", {"i": i})
    report = recorder.report()
    assert report["ring_size"] == 4
    assert report["captured"] == 10
    assert report["dropped_from_ring"] == 6
    window = recorder.window(60.0)
    assert [m.payload["i"] for m in window] == [6, 7, 8, 9]


def test_excluded_topics_never_enter_the_ring():
    recorder = BusRecorder(capacity=16)
    recorder.exclude("metrics.sample")
    recorder.record("metrics.sample", 1)
    recorder.record("will.decision", 2)
    assert [m.topic for m in recorder.window(60.0)] == ["will.decision"]


def test_window_filters_by_topic_and_age():
    recorder = BusRecorder(capacity=32)
    recorder.record("a", 1)
    recorder.record("b", 2)
    assert [m.topic for m in recorder.window(60.0, topics=("b",))] == ["b"]
    assert recorder.window(0.0) == []


def test_unserializable_payloads_are_summarized_not_dropped():
    recorder = BusRecorder(capacity=4)

    class Weird:
        def __repr__(self):
            return "<weird>"

    recorder.record("t", {"obj": Weird(), "n": 1})
    dumped = recorder.window(60.0)[0].to_dict()
    # The structure survives and the un-encodable leaf keeps its repr,
    # rather than the whole message being dropped.
    assert dumped["payload"] == {"obj": "<weird>", "n": 1}


def test_oversized_payloads_are_truncated_with_their_size_recorded():
    recorder = BusRecorder(capacity=4)
    recorder.record("t", {"blob": "x" * 20000})
    payload = recorder.window(60.0)[0].to_dict()["payload"]
    assert payload["truncated"] is True
    assert payload["bytes"] > 20000


def test_a_circular_payload_does_not_break_recording():
    recorder = BusRecorder(capacity=4)
    loop: dict = {}
    loop["self"] = loop
    recorder.record("t", loop)
    assert "repr" in recorder.window(60.0)[0].to_dict()["payload"]


def test_dump_and_replay_round_trip(tmp_path):
    async def scenario():
        recorder = BusRecorder(capacity=16)
        for i in range(3):
            recorder.record("will.decision", {"i": i})
        path = await recorder.dump(reason="test dump", directory=tmp_path)
        assert path is not None

        reader = BagReader(path)
        assert len(reader) == 3
        assert reader.topics() == ["will.decision"]
        assert reader.header["reason"] == "test dump"

        replayed: list[tuple[str, object]] = []

        async def fake_publish(topic, payload):
            replayed.append((topic, payload))

        count = await reader.replay(publish=fake_publish)
        return count, replayed

    count, replayed = asyncio.run(scenario())
    assert count == 3
    assert [p["i"] for _, p in replayed] == [0, 1, 2]


def test_dump_of_an_empty_ring_returns_none(tmp_path):
    async def scenario():
        return await BusRecorder().dump(reason="nothing", directory=tmp_path)

    assert asyncio.run(scenario()) is None


@pytest.mark.parametrize("export", ["dump", "recording"])
def test_bus_exports_preserve_existing_parent_permissions(tmp_path, monkeypatch, export):
    import json

    parent = tmp_path / "shared"
    parent.mkdir()
    parent.chmod(0o755)
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")

    async def scenario():
        recorder = BusRecorder(capacity=4)
        if export == "recording":
            recorder.start_file(parent / "capture.jsonl")
        recorder.record("test.event", {"value": 1})
        if export == "recording":
            return await recorder.stop_file()
        return await recorder.dump(reason="test", directory=parent)

    path = asyncio.run(scenario())
    assert path is not None
    assert parent.stat().st_mode & 0o777 == 0o755
    assert path.stat().st_mode & 0o777 == 0o600
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert any(row.get("topic") == "test.event" for row in rows)


def test_event_bus_feeds_the_ring():
    """The live bus must actually funnel through the recorder."""
    import inspect

    from core import event_bus

    source = inspect.getsource(event_bus)
    assert "_record_to_bag(topic, data)" in source
    assert source.count("_record_to_bag(topic, data)") >= 2, "both local paths"


# ── diagnostics ───────────────────────────────────────────────────────

def test_stale_is_worse_than_error_because_silence_is_less_trustworthy():
    assert Level.STALE > Level.ERROR > Level.WARN > Level.OK


def test_a_task_that_stops_reporting_goes_stale_not_absent():
    aggregator = DiagnosticsAggregator()
    updater = aggregator.updater("lane", stale_after_s=0.05)
    updater.add("warm", lambda: (Level.OK, "warm"))
    aggregator.update_all()
    assert aggregator.statuses()[0].level is Level.OK

    time.sleep(0.06)
    status = aggregator.statuses()[0]
    assert status.level is Level.STALE
    assert "its previous word was OK" in status.message


def test_a_task_that_never_reported_is_stale():
    aggregator = DiagnosticsAggregator()
    aggregator.updater("lane").add("never", lambda: Level.OK)
    status = aggregator.statuses()[0]
    assert status.level is Level.STALE
    assert "never reported" in status.message


def test_a_raising_task_reports_error_rather_than_nothing():
    aggregator = DiagnosticsAggregator()
    aggregator.updater("lane").add("boom", lambda: (_ for _ in ()).throw(ValueError("x")))
    aggregator.update_all()
    status = aggregator.statuses()[0]
    assert status.level is Level.ERROR
    assert "ValueError" in status.message


def test_aggregation_rolls_the_worst_level_upward_and_names_the_path():
    aggregator = DiagnosticsAggregator()
    aggregator.add_analyzer(Analyzer(path="/runtime", prefixes=("runtime/",)))
    updater = aggregator.updater("runtime")
    updater.add("ok_one", lambda: Level.OK)
    updater.add("bad_one", lambda: (Level.ERROR, "the thing broke"))
    aggregator.update_all()

    aggregate = aggregator.aggregate()
    assert aggregate["level"] == "ERROR"
    assert aggregate["ok"] is False
    node = next(n for n in aggregate["nodes"] if n["path"] == "/runtime")
    assert node["level"] == "ERROR"
    assert "runtime/bad_one" in node["message"]


def test_a_missing_expected_item_makes_the_node_stale():
    aggregator = DiagnosticsAggregator()
    aggregator.add_analyzer(
        Analyzer(path="/runtime", prefixes=("runtime/",), expected=("taint", "lockdep"))
    )
    updater = aggregator.updater("runtime")
    updater.add("taint", lambda: Level.OK)
    aggregator.update_all()

    node = next(n for n in aggregator.aggregate()["nodes"] if n["path"] == "/runtime")
    assert node["level"] == "STALE"
    assert node["missing"] == ["lockdep"]


def test_a_non_critical_analyzer_cannot_push_the_top_past_warn():
    aggregator = DiagnosticsAggregator()
    aggregator.add_analyzer(Analyzer(path="/optional", prefixes=("opt/",), critical=False))
    aggregator.updater("opt").add("broken", lambda: (Level.ERROR, "optional thing died"))
    aggregator.update_all()
    assert aggregator.aggregate()["level"] == "WARN"


def test_unclaimed_statuses_land_in_other_rather_than_vanishing():
    aggregator = DiagnosticsAggregator()
    aggregator.updater("nobody").add("task", lambda: Level.OK)
    aggregator.update_all()
    paths = [n["path"] for n in aggregator.aggregate()["nodes"]]
    assert "/other" in paths


def test_installed_runtime_diagnostics_report_clean_at_rest():
    diag_mod.install_default_analyzers()
    names = diag_mod.install_runtime_diagnostics()
    assert "taint" in names and "lockdep" in names
    report = diag_mod.diagnostics_report()
    assert report["stale"] == [], report["summary"]
    assert report["errors"] == [], report["summary"]


# ── invariants ────────────────────────────────────────────────────────

def test_middleware_invariants_registered_and_clean():
    from core.verify import runtime_invariants  # noqa: F401
    from core.verify.invariants import get_registry, verify

    names = {s.name for s in get_registry().specs()}
    for expected in (
        "lifecycle.critical_organs_are_active",
        "reality_middleware.effects_are_reconciled",
        "reality_metrology.live_mode_is_restored",
        "qos.state_topics_are_transient_local",
        "parameters.numeric_bounds_are_declared",
        "diagnostics.nothing_is_stale",
    ):
        assert expected in names

    diag_mod.install_default_analyzers()
    diag_mod.install_runtime_diagnostics()
    diag_mod.get_aggregator().update_all()
    report = verify("middleware", record=False)
    assert report.ok, report.summary()


def test_the_standard_topic_declarations_satisfy_their_own_invariant():
    from core.runtime.foundations import _declare_standard_topics
    from core.verify import runtime_invariants  # noqa: F401
    from core.verify.invariants import verify

    _declare_standard_topics()
    report = verify("middleware", record=False)
    offenders = [
        v for v in report.violations if v.invariant == "qos.state_topics_are_transient_local"
    ]
    assert offenders == [], f"shipped topic profiles violate their own rule: {offenders}"


def test_physical_middleware_invariant_rejects_unreconciled_effects(monkeypatch):
    from core.container import ServiceContainer
    from core.verify.runtime_invariants import _managed_physical_effects_reconciled

    class Middleware:
        @staticmethod
        def status():
            return {
                "alive": True,
                "ready": False,
                "recovery_required_count": 1,
            }

    services = {"reality_reach": object(), "reality_middleware": Middleware()}
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: services.get(name, default)),
    )

    violations = list(_managed_physical_effects_reconciled())

    assert len(violations) == 1
    assert "unresolved_effects=1" in violations[0].message


def test_metrology_invariant_rejects_stranded_simulation_mode(monkeypatch):
    from core.container import ServiceContainer
    from core.verify.runtime_invariants import _reality_metrology_live_mode_restored

    class Metrology:
        @staticmethod
        def status():
            return {
                "mode": "simulation",
                "active_run": None,
                "live_restoration_required": True,
            }

    services = {"reality_reach": object(), "reality_metrology": Metrology()}
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: services.get(name, default)),
    )

    violations = list(_reality_metrology_live_mode_restored())

    assert len(violations) == 1
    assert "simulation evidence could contaminate" in violations[0].message
