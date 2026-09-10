"""Contract tests for the flight-software disciplines.

core/fsw/{telemetry_dictionary,restart_protection,rate_groups,assertions,
command_dispatch,health_checker}.py.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.fsw import assertions as assert_mod
from core.fsw import command_dispatch as cmd_mod
from core.fsw import health_checker as health_mod
from core.fsw import rate_groups as rate_mod
from core.fsw import restart_protection as restart_mod
from core.fsw import telemetry_dictionary as tlm_mod
from core.fsw.assertions import Response, fw_assert
from core.fsw.command_dispatch import (
    ArgSpec,
    ArgType,
    CommandDispatcher,
    CommandSequence,
    CommandSpec,
    FailurePolicy,
    Step,
)
from core.fsw.health_checker import HealthChecker
from core.fsw.rate_groups import RateGroup
from core.fsw.restart_protection import (
    ALARM_NO_CORE_SETS,
    ALARM_RESTART_LOOP,
    Priority,
    RestartProtection,
)
from core.fsw.telemetry_dictionary import (
    ChannelType,
    EventSeverity,
    Limits,
    LimitState,
    TelemetryDictionary,
    channel,
)
from core.runtime import taint as taint_mod


@pytest.fixture(autouse=True)
def _clean():
    mods = (tlm_mod, restart_mod, rate_mod, assert_mod, cmd_mod, health_mod, taint_mod)
    for mod in mods:
        for name in dir(mod):
            if name.startswith("reset_") and name.endswith("_for_test"):
                getattr(mod, name)()
    yield
    for mod in mods:
        for name in dir(mod):
            if name.startswith("reset_") and name.endswith("_for_test"):
                getattr(mod, name)()


# ── telemetry dictionary ──────────────────────────────────────────────

def test_limits_evaluate_to_the_right_state():
    limits = Limits(yellow_low=20, yellow_high=80, red_low=10, red_high=90)
    assert limits.evaluate(50) is LimitState.NOMINAL
    assert limits.evaluate(15) is LimitState.YELLOW_LOW
    assert limits.evaluate(85) is LimitState.YELLOW_HIGH
    assert limits.evaluate(5) is LimitState.RED_LOW
    assert limits.evaluate(95) is LimitState.RED_HIGH
    # Red wins over yellow when both apply.
    assert limits.evaluate(10) is LimitState.RED_LOW


def test_incoherent_limits_are_refused_at_declaration():
    with pytest.raises(ValueError, match="incoherent limits"):
        channel(1, "bad", owner="t", yellow_low=10, red_low=20)
    with pytest.raises(ValueError, match="incoherent limits"):
        channel(2, "also_bad", owner="t", red_low=90, red_high=10)


def test_a_limit_crossing_is_a_transition_event_not_a_repeated_alarm():
    telemetry = TelemetryDictionary()
    telemetry.declare_channel(
        tlm_mod.ChannelSpec(
            identifier=1,
            name="mem",
            type=ChannelType.FLOAT,
            unit="fraction",
            description="memory",
            owner="t",
            limits=Limits(red_low=0.1),
        )
    )
    assert telemetry.write("mem", 0.5) is LimitState.NOMINAL
    assert telemetry.write("mem", 0.05) is LimitState.RED_LOW
    assert telemetry.write("mem", 0.04) is LimitState.RED_LOW
    assert telemetry.write("mem", 0.03) is LimitState.RED_LOW

    transitions = [e for e in telemetry.events(limit=100) if e.name == "channel_limit_transition"]
    # One event for entering red, not one per sample.
    assert len(transitions) == 1
    assert telemetry.write("mem", 0.9) is LimitState.NOMINAL
    transitions = [e for e in telemetry.events(limit=100) if e.name == "channel_limit_transition"]
    assert len(transitions) == 2, "leaving a violation is also a transition"


def test_a_channel_that_stops_reporting_goes_stale_not_nominal():
    telemetry = TelemetryDictionary()
    telemetry.declare_channel(
        tlm_mod.ChannelSpec(
            identifier=1,
            name="quiet",
            type=ChannelType.FLOAT,
            unit="",
            description="",
            owner="t",
            stale_after_s=0.05,
        )
    )
    telemetry.write("quiet", 1.0)
    assert telemetry.state("quiet") is LimitState.NOMINAL
    time.sleep(0.06)
    assert telemetry.state("quiet") is LimitState.STALE


def test_a_never_written_channel_is_stale():
    telemetry = TelemetryDictionary()
    telemetry.declare_channel(
        tlm_mod.ChannelSpec(
            identifier=1, name="never", type=ChannelType.FLOAT, unit="", description="", owner="t"
        )
    )
    assert telemetry.state("never") is LimitState.STALE
    assert "never" in telemetry.report()["silent_channels"]


def test_channel_ids_are_a_contract_and_cannot_be_reused():
    telemetry = TelemetryDictionary()
    spec = tlm_mod.ChannelSpec(
        identifier=7, name="a", type=ChannelType.FLOAT, unit="", description="", owner="t"
    )
    telemetry.declare_channel(spec)
    with pytest.raises(ValueError, match="already"):
        telemetry.declare_channel(
            tlm_mod.ChannelSpec(
                identifier=7, name="b", type=ChannelType.FLOAT, unit="", description="", owner="t"
            )
        )


def test_events_render_from_a_format_string_and_stay_countable():
    telemetry = TelemetryDictionary()
    telemetry.declare_event(
        tlm_mod.EventSpec(
            identifier=100,
            name="lane_swapped",
            severity=EventSeverity.ACTIVITY_HI,
            format_string="lane {lane} swapped to {model} in {ms}ms",
            description="",
            owner="t",
        )
    )
    event = telemetry.emit("lane_swapped", lane=1, model="32b", ms=430)
    assert event.text == "lane 1 swapped to 32b in 430ms"
    # The args survive separately, which is what makes events diffable.
    assert event.args == {"lane": 1, "model": "32b", "ms": 430}


def test_severity_filters_the_event_log():
    telemetry = TelemetryDictionary()
    telemetry.emit("chatter", severity=EventSeverity.DIAGNOSTIC)
    telemetry.emit("problem", severity=EventSeverity.WARNING_HI)
    names = [e.name for e in telemetry.events(min_severity=EventSeverity.WARNING_LO)]
    assert names == ["problem"]


def test_a_bad_format_string_still_produces_an_event():
    telemetry = TelemetryDictionary()
    telemetry.declare_event(
        tlm_mod.EventSpec(
            identifier=1,
            name="broken",
            severity=EventSeverity.DIAGNOSTIC,
            format_string="needs {missing}",
            description="",
            owner="t",
        )
    )
    event = telemetry.emit("broken", present=1)
    assert "format error" in event.text
    assert event.args == {"present": 1}


def test_violations_report_names_the_owner_and_duration():
    telemetry = TelemetryDictionary()
    telemetry.declare_channel(
        tlm_mod.ChannelSpec(
            identifier=1,
            name="hot",
            type=ChannelType.FLOAT,
            unit="C",
            description="",
            owner="core/thermal.py",
            limits=Limits(red_high=90),
        )
    )
    telemetry.write("hot", 95)
    violation = telemetry.violations()[0]
    assert violation["state"] == "red_high"
    assert violation["owner"] == "core/thermal.py"
    assert violation["unit"] == "C"


def test_the_dictionary_is_an_openmct_shaped_tree():
    telemetry = TelemetryDictionary()
    telemetry.declare_channel(
        tlm_mod.ChannelSpec(
            identifier=1,
            name="mem",
            type=ChannelType.FLOAT,
            unit="fraction",
            description="memory available",
            owner="t",
            group="resources",
            limits=Limits(red_low=0.1),
        )
    )
    objects = telemetry.domain_objects()
    root = next(o for o in objects if o["identifier"]["key"] == "root")
    assert root["composition"] == [{"namespace": "aura.telemetry", "key": "group:resources"}]
    point = next(o for o in objects if o["identifier"]["key"] == "mem")
    values = point["telemetry"]["values"]
    assert any(v["hints"].get("domain") for v in values)
    assert any(v.get("limits", {}).get("red_low") == 0.1 for v in values)


def test_the_standard_channel_set_declares_cleanly():
    from core.runtime.foundations import _declare_standard_telemetry

    channels, events = _declare_standard_telemetry()
    assert "memory.available_fraction" in channels
    assert "program_alarm" in events
    dictionary = tlm_mod.get_telemetry().dictionary()
    ids = [c["id"] for c in dictionary["channels"]]
    assert len(ids) == len(set(ids)), "channel ids must be unique"


# ── restart protection ────────────────────────────────────────────────

def test_overload_sheds_from_the_bottom_and_keeps_the_essential_loop():
    protection = RestartProtection(core_sets=4)
    protection.declare("kernel_tick", priority=Priority.ESSENTIAL)
    protection.declare("response", priority=Priority.INTERACTIVE)
    protection.declare("consolidation", priority=Priority.ROUTINE)
    protection.declare("curiosity", priority=Priority.BACKGROUND)
    for name in ("kernel_tick", "response", "consolidation", "curiosity"):
        protection.begin(name)

    alarm = protection.overload(needed=2)
    assert alarm.code == ALARM_NO_CORE_SETS
    # Bottom-up: background first, then routine.
    assert list(alarm.shed) == ["curiosity", "consolidation"]
    assert "kernel_tick" in alarm.kept
    assert "response" in alarm.kept


def test_essential_work_runs_even_with_no_core_sets_left():
    protection = RestartProtection(core_sets=1)
    protection.declare("hog", priority=Priority.ROUTINE)
    protection.declare("kernel_tick", priority=Priority.ESSENTIAL)
    assert protection.begin("hog") is True
    assert protection.begin("kernel_tick") is True, "guidance runs regardless"
    protection.declare("optional", priority=Priority.ROUTINE)
    assert protection.begin("optional") is False


def test_phase_tables_let_a_restart_resume_rather_than_repeat():
    protection = RestartProtection()
    resumed_from: list[str] = []
    protection.declare(
        "consolidation",
        priority=Priority.INTERACTIVE,
        phases=("select", "summarize", "embed", "write"),
        resume=lambda phase: resumed_from.append(phase.name if phase else "start"),
    )
    protection.begin("consolidation")
    protection.enter_phase("consolidation", "embed", batch=7)

    report = protection.restart(reason="test")
    assert resumed_from == ["embed"], "resuming from the start would lose the work"
    assert "consolidation" in report["resumed"]


def test_restart_discards_below_the_threshold_and_keeps_above():
    protection = RestartProtection()
    protection.declare("background_thing", priority=Priority.BACKGROUND)
    protection.declare("routine_thing", priority=Priority.ROUTINE)
    protection.declare("interactive_thing", priority=Priority.INTERACTIVE)
    for name in ("background_thing", "routine_thing", "interactive_thing"):
        protection.begin(name)

    report = protection.restart(keep_above=Priority.ROUTINE)
    assert report["discarded"] == ["background_thing"]
    assert set(report["resumed"]) == {"routine_thing", "interactive_thing"}


def test_a_group_that_cannot_resume_is_reported_lost():
    protection = RestartProtection()
    protection.declare(
        "fragile",
        priority=Priority.INTERACTIVE,
        resume=lambda phase: (_ for _ in ()).throw(RuntimeError("state gone")),
    )
    protection.begin("fragile")
    report = protection.restart()
    assert report["lost"] == ["fragile"]
    codes = [a["code"] for a in protection.report()["recent_alarms"]]
    assert restart_mod.ALARM_PHASE_LOST in codes


def test_a_restart_loop_is_detected_and_taints():
    protection = RestartProtection()
    for _ in range(5):
        protection.restart(reason="loop")
    codes = [a["code"] for a in protection.report()["recent_alarms"]]
    assert ALARM_RESTART_LOOP in codes
    assert taint_mod.is_tainted(taint_mod.TaintFlag.CRASHED_ORGAN)


def test_the_standard_group_set_declares_the_essentials():
    names = restart_mod.install_standard_groups()
    assert "kernel_tick" in names
    report = restart_mod.restart_report()
    assert "kernel_tick" in report["essential"]
    assert "unified_will" in report["essential"]
    assert "curiosity" not in report["essential"]


# ── rate groups ───────────────────────────────────────────────────────

def test_a_rate_group_runs_members_in_declared_order():
    order: list[str] = []
    group = RateGroup("test", 1.0)
    group.add("late", lambda: order.append("late"), order=90)
    group.add("early", lambda: order.append("early"), order=10)
    asyncio.run(group.run_cycle())
    assert order == ["early", "late"]


def test_a_cycle_that_overruns_its_period_is_a_measured_slip():
    group = RateGroup("slow", 0.02)
    group.add("hog", lambda: time.sleep(0.05))
    record = asyncio.run(group.run_cycle())
    assert record.slipped is True
    assert record.late_by_s > 0
    assert record.slowest == "hog"
    assert group.slips == 1


def test_the_slowest_member_is_identified_not_merely_suspected():
    group = RateGroup("mixed", 0.05)
    group.add("fast", lambda: None)
    group.add("slow", lambda: time.sleep(0.08))
    record = asyncio.run(group.run_cycle())
    assert record.slowest == "slow"
    report = group.report()
    assert "slow" in report["over_budget"]


def test_a_failing_member_does_not_stop_the_group():
    ran: list[str] = []
    group = RateGroup("resilient", 1.0)
    group.add("boom", lambda: (_ for _ in ()).throw(RuntimeError("x")), order=10)
    group.add("after", lambda: ran.append("after"), order=20)
    asyncio.run(group.run_cycle())
    assert ran == ["after"]
    assert group.report()["members"][0]["failures"] == 1


def test_consecutive_slips_reset_when_a_cycle_makes_its_period():
    group = RateGroup("recovering", 0.02)
    slow = group.add("member", lambda: time.sleep(0.05))
    asyncio.run(group.run_cycle())
    assert group.consecutive_slips == 1
    slow.fn = lambda: None
    asyncio.run(group.run_cycle())
    assert group.consecutive_slips == 0


def test_sustained_slipping_escalates_to_the_overload_response():
    restart_mod.install_standard_groups()
    group = RateGroup("overloaded", 0.001)
    group.add("hog", lambda: time.sleep(0.01))
    for _ in range(5):
        asyncio.run(group.run_cycle())
    alarms = restart_mod.restart_report()["recent_alarms"]
    assert any(a["code"] == ALARM_NO_CORE_SETS for a in alarms)


def test_the_period_is_a_period_not_a_gap():
    """A loop that sleeps AFTER working drifts; a rate group does not."""
    group = RateGroup("scheduled", 0.03)
    starts: list[float] = []
    group.add("member", lambda: (starts.append(time.monotonic()), time.sleep(0.01))[0])

    async def scenario():
        await group.start()
        await asyncio.sleep(0.13)
        await group.stop()

    asyncio.run(scenario())
    assert len(starts) >= 3
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    # Each start is ~one period after the last, not period + work time.
    assert all(0.02 < gap < 0.045 for gap in gaps), gaps


# ── assertions ────────────────────────────────────────────────────────

def test_a_passing_assertion_costs_nothing_and_records_nothing():
    assert fw_assert(True, "fine") is True
    assert assert_mod.assertions_clean() is True


def test_a_failure_records_the_site_the_args_and_taints():
    assert fw_assert(False, "lane owner changed", expected="a", observed="b") is False
    report = assert_mod.assertions_report()
    assert report["clean"] is False
    record = report["records"][0]
    assert record["condition"] == "lane owner changed"
    assert record["args"] == {"expected": "a", "observed": "b"}
    assert record["line"] > 0
    assert "test_flight_software_discipline" in record["file"]
    assert taint_mod.is_tainted(taint_mod.TaintFlag.ASSERTION)


def test_raise_response_raises_after_recording():
    with pytest.raises(assert_mod.AssertionFailure, match="bad state"):
        fw_assert(False, "bad state", response=Response.RAISE)
    assert assert_mod.assertions_report()["distinct_sites"] == 1


def test_restart_response_requests_a_controlled_restart(monkeypatch):
    import core.runtime.shutdown_coordinator as sc

    calls: list[str] = []
    monkeypatch.setattr(sc, "request_shutdown", lambda reason="": calls.append(reason) or {})
    fw_assert(False, "state is a guess", response=Response.RESTART)
    assert calls and "assertion_failed" in calls[0]


def test_repeated_failures_at_one_site_dedupe_but_count():
    for _ in range(4):
        fw_assert(False, "same site")
    report = assert_mod.assertions_report()
    assert report["distinct_sites"] == 1
    assert report["total_failures"] == 4
    assert report["records"][0]["count"] == 4


def test_assertions_always_run_regardless_of_optimisation():
    """python -O removes `assert`; fw_assert is a function call."""
    import inspect

    source = inspect.getsource(assert_mod.fw_assert)
    assert "if condition:" in source, "must be a real branch, not a bare assert"


# ── command dispatch ──────────────────────────────────────────────────

def _dispatcher_with_setter() -> tuple[CommandDispatcher, dict]:
    dispatcher = CommandDispatcher()
    state = {"lanes": 1}

    def set_lanes(lanes: int) -> bool:
        state["lanes"] = lanes
        return True

    dispatcher.declare(
        CommandSpec(
            opcode=0x10,
            name="set_lanes",
            description="set lane count",
            owner="t",
            args=(ArgSpec("lanes", ArgType.INT, minimum=1, maximum=4),),
            handler=set_lanes,
        )
    )
    return dispatcher, state


def test_arguments_are_validated_against_the_dictionary():
    dispatcher, state = _dispatcher_with_setter()
    result = asyncio.run(dispatcher.dispatch("set_lanes", lanes=3))
    assert result.ok is True and state["lanes"] == 3

    result = asyncio.run(dispatcher.dispatch("set_lanes", lanes=99))
    assert result.ok is False and "above the maximum" in result.error
    assert state["lanes"] == 3, "a rejected command must not have run"


def test_unknown_arguments_and_missing_required_ones_are_rejected():
    dispatcher, _ = _dispatcher_with_setter()
    assert "not an argument" in asyncio.run(
        dispatcher.dispatch("set_lanes", lanes=1, extra=2)
    ).error
    assert "required" in asyncio.run(dispatcher.dispatch("set_lanes")).error


def test_opcodes_cannot_be_reused():
    dispatcher, _ = _dispatcher_with_setter()
    with pytest.raises(ValueError, match="already"):
        dispatcher.declare(
            CommandSpec(opcode=0x10, name="other", description="", owner="t", handler=lambda: True)
        )


def test_a_sequence_is_validated_before_any_of_it_runs():
    dispatcher, state = _dispatcher_with_setter()
    sequence = CommandSequence(
        name="plan",
        steps=(
            Step(command="set_lanes", args={"lanes": 2}),
            Step(command="set_lanes", args={"lanes": 99}),  # invalid
        ),
    )
    problems = dispatcher.validate_sequence(sequence)
    assert problems and "step 2" in problems[0]

    outcome = asyncio.run(dispatcher.run_sequence(sequence))
    assert outcome["ok"] is False
    assert outcome["validated"] is False
    assert state["lanes"] == 1, "step 1 must not run when step 2 is malformed"


def test_a_sequence_aborts_at_a_failing_step_by_default():
    dispatcher = CommandDispatcher()
    ran: list[str] = []
    dispatcher.declare(
        CommandSpec(
            opcode=1,
            name="ok",
            description="",
            owner="t",
            handler=lambda: ran.append("ok") or True,
        )
    )
    dispatcher.declare(
        CommandSpec(opcode=2, name="fails", description="", owner="t", handler=lambda: False)
    )
    sequence = CommandSequence(
        name="s",
        steps=(Step(command="ok"), Step(command="fails"), Step(command="ok")),
    )
    outcome = asyncio.run(dispatcher.run_sequence(sequence))
    assert outcome["aborted_at_step"] == 2
    assert outcome["completed_steps"] == 2
    assert ran == ["ok"]


def test_continue_policy_runs_past_a_failure():
    dispatcher = CommandDispatcher()
    dispatcher.declare(
        CommandSpec(opcode=1, name="fails", description="", owner="t", handler=lambda: False)
    )
    dispatcher.declare(
        CommandSpec(opcode=2, name="ok", description="", owner="t", handler=lambda: True)
    )
    sequence = CommandSequence(
        name="s",
        steps=(
            Step(command="fails", on_failure=FailurePolicy.CONTINUE),
            Step(command="ok"),
        ),
    )
    outcome = asyncio.run(dispatcher.run_sequence(sequence))
    assert outcome["completed_steps"] == 2
    assert outcome["ok"] is False, "a completed sequence with a failed step is not ok"


def test_retry_policy_retries_the_declared_number_of_times():
    dispatcher = CommandDispatcher()
    attempts = {"n": 0}

    def flaky() -> bool:
        attempts["n"] += 1
        return attempts["n"] >= 3

    dispatcher.declare(
        CommandSpec(opcode=1, name="flaky", description="", owner="t", handler=flaky)
    )
    sequence = CommandSequence(
        name="s",
        steps=(Step(command="flaky", on_failure=FailurePolicy.RETRY, retries=3),),
    )
    outcome = asyncio.run(dispatcher.run_sequence(sequence))
    assert outcome["ok"] is True
    assert attempts["n"] == 3


def test_a_retry_policy_with_no_retries_is_a_validation_error():
    dispatcher, _ = _dispatcher_with_setter()
    sequence = CommandSequence(
        name="s",
        steps=(Step(command="set_lanes", args={"lanes": 1}, on_failure=FailurePolicy.RETRY),),
    )
    assert any("no retries" in p for p in dispatcher.validate_sequence(sequence))


def test_consequential_commands_need_the_sequence_to_say_so():
    dispatcher = CommandDispatcher()
    dispatcher.declare(
        CommandSpec(
            opcode=1,
            name="wipe",
            description="",
            owner="t",
            handler=lambda: True,
            consequential=True,
        )
    )
    unsafe = CommandSequence(name="s", steps=(Step(command="wipe"),))
    assert any("consequential" in p for p in dispatcher.validate_sequence(unsafe))

    declared = CommandSequence(
        name="s", steps=(Step(command="wipe"),), allows_consequential=True
    )
    assert dispatcher.validate_sequence(declared) == []


def test_dry_run_validates_without_executing():
    dispatcher, state = _dispatcher_with_setter()
    sequence = CommandSequence(name="s", steps=(Step(command="set_lanes", args={"lanes": 4}),))
    outcome = asyncio.run(dispatcher.run_sequence(sequence, dry_run=True))
    assert outcome["ok"] is True and outcome["dry_run"] is True
    assert state["lanes"] == 1


def test_the_installed_runtime_commands_all_have_handlers():
    names = cmd_mod.install_runtime_commands()
    assert "set_parameter" in names
    for entry in cmd_mod.get_dispatcher().dictionary()["commands"]:
        assert entry["has_handler"], entry["name"]


# ── health checker ────────────────────────────────────────────────────

def test_an_unresponsive_component_is_declared_only_after_the_threshold():
    checker = HealthChecker()
    alive = {"ok": True}
    declared: list[str] = []
    checker.watch(
        "lane",
        lambda: alive["ok"],
        miss_threshold=3,
        on_unresponsive=declared.append,
    )
    asyncio.run(checker.run_round())
    assert checker.report()["responsive"] == ["lane"]

    alive["ok"] = False
    asyncio.run(checker.run_round())
    asyncio.run(checker.run_round())
    assert declared == [], "one or two missed pings under load is normal"
    asyncio.run(checker.run_round())
    assert declared == ["lane"]
    assert checker.report()["unresponsive"] == ["lane"]


def test_slow_and_unresponsive_are_different_states():
    checker = HealthChecker()

    async def slow_ping() -> bool:
        await asyncio.sleep(0.05)
        return True

    checker.watch("slowpoke", slow_ping, timeout_s=0.2)
    asyncio.run(checker.run_round())
    # It answered, within the timeout: responsive.
    assert checker.report()["responsive"] == ["slowpoke"]

    checker.unwatch("slowpoke")
    checker.watch("timeout_er", slow_ping, timeout_s=0.01, miss_threshold=1)
    asyncio.run(checker.run_round())
    assert checker.report()["unresponsive"] == ["timeout_er"]


def test_recovery_is_announced():
    checker = HealthChecker()
    alive = {"ok": False}
    recovered: list[str] = []
    checker.watch("flaky", lambda: alive["ok"], miss_threshold=1, on_recovered=recovered.append)
    asyncio.run(checker.run_round())
    assert checker.report()["unresponsive"] == ["flaky"]
    alive["ok"] = True
    asyncio.run(checker.run_round())
    assert recovered == ["flaky"]
    assert checker.report()["responsive"] == ["flaky"]


def test_a_critical_component_going_unresponsive_taints():
    checker = HealthChecker()
    checker.watch("spine", lambda: False, miss_threshold=1, critical=True)
    asyncio.run(checker.run_round())
    assert checker.report()["critical_unresponsive"] == ["spine"]
    assert taint_mod.is_tainted(taint_mod.TaintFlag.CRASHED_ORGAN)


def test_a_raising_ping_counts_as_a_miss():
    checker = HealthChecker()
    checker.watch("raiser", lambda: (_ for _ in ()).throw(RuntimeError("x")), miss_threshold=1)
    asyncio.run(checker.run_round())
    assert checker.report()["unresponsive"] == ["raiser"]


def test_installed_runtime_pings_all_answer():
    names = health_mod.install_runtime_pings()
    assert "event_bus" in names
    asyncio.run(health_mod.get_health_checker().run_round())
    report = health_mod.health_checker_report()
    assert report["unresponsive"] == [], report


# ── invariants ────────────────────────────────────────────────────────

def test_flight_software_invariants_registered_and_clean():
    from core.runtime.foundations import _declare_standard_telemetry, _sample_standard_telemetry
    from core.verify import runtime_invariants  # noqa: F401
    from core.verify.invariants import get_registry, verify

    names = {s.name for s in get_registry().specs()}
    for expected in (
        "telemetry.limits_are_coherent",
        "restart.essential_work_is_declared",
        "rate_groups.are_not_slipping",
        "assertions.none_have_failed",
        "commands.declared_commands_have_handlers",
    ):
        assert expected in names

    _declare_standard_telemetry()
    restart_mod.install_standard_groups()
    cmd_mod.install_runtime_commands()
    health_mod.install_runtime_pings()
    _sample_standard_telemetry()

    report = verify("flight_software", record=False)
    assert report.ok, report.summary()


def test_periodic_sync_reads_run_off_the_event_loop():
    import threading

    from core.runtime.foundations import _run_periodic_read_off_loop

    async def exercise() -> tuple[list[int], bool]:
        main_thread = threading.get_ident()
        worker_threads: list[int] = []

        def slow_read() -> None:
            worker_threads.append(threading.get_ident())
            time.sleep(0.08)

        task = asyncio.create_task(_run_periodic_read_off_loop(slow_read))
        await asyncio.sleep(0.01)
        loop_remained_live = not task.done()
        await task
        return worker_threads, loop_remained_live and worker_threads[0] != main_thread

    worker_threads, off_loop = asyncio.run(exercise())
    assert worker_threads
    assert off_loop
