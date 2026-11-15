"""CP126 ``core/fictional_ai_synthesis.py`` — six engines, fifteen findings.

The module borrows six fictional AIs as design motifs. The motif is fine.
What CP126 found is that in four of the six the motif had drifted into the
measurement: a fill ratio was reported as melancholia, two hardcoded call
sites grew a number described as progress toward integrated personhood, a
keyword scan wrote social tension straight into kernel cognition, and a
resilience core reported failures it never tried to repair.

Each test below names the property, not the wording of the fix.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from core.fictional_ai_synthesis import (
    CognitiveHealthMonitor,
    CortanaPhase,
    DistributedResilienceCore,
    ProactiveAnticipationEngine,
    ProgressiveAutonomySystem,
    SocialModelingEngine,
    TemporalDilationScheduler,
)


# ── 6113643d: host sampling is not done on the event loop ───────────────────


@pytest.mark.asyncio
async def test_host_sampling_runs_off_the_event_loop():
    """statvfs and sysctl every two minutes, forever, on the loop."""
    engine = ProactiveAnticipationEngine()
    loop_thread = asyncio.get_running_loop()._thread_id  # type: ignore[attr-defined]
    seen: list[int] = []

    import threading

    def _probe():
        seen.append(threading.get_ident())
        return {"cpu_percent": 1.0, "memory_percent": 2.0,
                "memory_available_gb": 3.0, "disk_percent": 4.0}

    engine._sample_host_blocking = _probe  # type: ignore[method-assign]
    state = await engine._sample_system_state()
    assert state["cpu_percent"] == 1.0
    assert seen and seen[0] != loop_thread, (
        "the host sample ran on the event loop thread"
    )


# ── d1252922: a failed delivery does not spend the budget ───────────────────


def _her_words_form(monkeypatch):
    """What was noticed reaches the person only as her own turn on it (F06)."""

    async def her_turn_on(_heading, _reading, **_k):
        return "Something of hers."

    monkeypatch.setattr("core.brain.her_unprompted_turn.her_turn_on", her_turn_on)


@pytest.mark.asyncio
async def test_a_dropped_initiation_does_not_consume_a_daily_slot(monkeypatch):
    """Failed notifications were suppressing the ones that would land."""
    from core.container import ServiceContainer

    engine = ProactiveAnticipationEngine()
    engine._last_user_activity = 0.0  # idle
    monkeypatch.setattr(
        ServiceContainer, "get", staticmethod(lambda *a, **k: None)
    )

    delivered = await engine._fire_initiation("nobody is listening")
    assert delivered is False
    assert engine._daily_initiation_count == 0, (
        "a slot out of the daily cap was spent on a message with no output path"
    )


@pytest.mark.asyncio
async def test_a_delivered_initiation_does_consume_a_slot(monkeypatch):
    from core.container import ServiceContainer

    class _Orch:
        @staticmethod
        async def emit_spontaneous_message(_text, **_k):
            return True

    engine = ProactiveAnticipationEngine(orchestrator=_Orch())
    engine._last_user_activity = 0.0
    monkeypatch.setattr(
        ServiceContainer, "get", staticmethod(lambda *a, **k: None)
    )
    _her_words_form(monkeypatch)

    assert await engine._fire_initiation("this one lands") is True
    assert engine._daily_initiation_count == 1


@pytest.mark.asyncio
async def test_a_raising_output_path_gives_the_slot_back(monkeypatch):
    from core.container import ServiceContainer

    class _Orch:
        @staticmethod
        async def emit_spontaneous_message(_text, **_k):
            raise RuntimeError("output lane is down")

    engine = ProactiveAnticipationEngine(orchestrator=_Orch())
    engine._last_user_activity = 0.0
    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda *a, **k: None))
    _her_words_form(monkeypatch)

    assert await engine._fire_initiation("this one raises") is False
    assert engine._daily_initiation_count == 0


# ── 36b05560: a commitment has an author ────────────────────────────────────


def test_auras_own_open_thread_is_not_filed_as_the_users():
    """"I'll look into that" became "you mentioned something yesterday"."""
    engine = ProactiveAnticipationEngine()
    engine.record_activity(
        user_input="what is the capital of France",
        response="I'll dig into the regional history later",
    )
    assert len(engine._unresolved_topics) == 1
    topic = engine._unresolved_topics[0]
    assert topic["author"] == "aura", (
        "Aura's own unfinished sentence was attributed to the user"
    )
    assert "regional history" in topic["topic"], (
        "the topic recorded was the user's question, not what Aura actually said"
    )


def test_a_user_reminder_request_is_kept_apart_from_an_inference():
    engine = ProactiveAnticipationEngine()
    engine.record_activity(user_input="remind me to call the bank", response="sure")
    assert engine._unresolved_topics[0]["author"] == "user_request"


def test_the_reminder_says_whose_thread_it_was():
    engine = ProactiveAnticipationEngine()
    now = 1_000_000.0
    aura = engine._the_thread_as_read({"topic": "the regional history", "author": "aura", "timestamp": now - 90_000}, now)
    user = engine._the_thread_as_read({"topic": "call the bank", "author": "user_request", "timestamp": now}, now)
    inferred = engine._the_thread_as_read({"topic": "the tax thing", "author": "user", "timestamp": now}, now)

    assert "her own words" in aura and '"the regional history"' in aura and "25 hours ago" in aura
    assert "asking to be reminded" in user and "her own" not in user
    # The inferred one must not assert that a promise was made.
    assert "an inference, not a request" in inferred


def test_an_ordinary_turn_records_no_commitment():
    engine = ProactiveAnticipationEngine()
    engine.record_activity(user_input="what time is it", response="It is four o'clock.")
    assert list(engine._unresolved_topics) == []


# ── 14de312d: an ungraded turn is not evidence ──────────────────────────────


def test_an_ungraded_turn_does_not_move_the_coherence_score(tmp_path):
    monitor = CognitiveHealthMonitor(persist_path=str(tmp_path / "c.json"))
    for _ in range(200):
        monitor.record_turn(
            context_tokens=1000,
            max_tokens=8192,
            response_quality=None,
            identity_markers_present=None,
            topics_in_play=1,
            resolved_topics=1,
        )
    assert monitor._coherence_score == 0.0, (
        "200 ungraded turns produced a coherence score, which is how two "
        "hardcoded call sites grew a number reported as personhood"
    )
    assert monitor._measured_turns == 0
    assert monitor.coherence_is_reportable() is False


def test_a_graded_turn_does_move_it(tmp_path):
    """The gate is a floor, not a wall."""
    monitor = CognitiveHealthMonitor(persist_path=str(tmp_path / "c.json"))
    for _ in range(40):
        monitor.record_turn(
            context_tokens=100,
            max_tokens=8192,
            response_quality=0.9,
            identity_markers_present=True,
            topics_in_play=1,
            resolved_topics=1,
        )
    assert monitor._measured_turns == 40
    assert monitor._coherence_score > 0.0
    assert monitor.coherence_is_reportable() is True


def test_the_metastable_phase_needs_measured_turns(tmp_path):
    monitor = CognitiveHealthMonitor(persist_path=str(tmp_path / "c.json"))
    monitor._coherence_score = 0.99  # high, and resting on nothing
    monitor._measured_turns = 0
    snapshot = monitor.record_turn(
        context_tokens=100, max_tokens=8192, topics_in_play=1, resolved_topics=1
    )
    assert snapshot.phase is not CortanaPhase.METASTABLE


def test_an_ungraded_snapshot_says_coherence_is_unknown(tmp_path):
    monitor = CognitiveHealthMonitor(persist_path=str(tmp_path / "c.json"))
    snapshot = monitor.record_turn(context_tokens=100, max_tokens=8192)
    assert snapshot.identity_coherence is None, (
        "a default stood in for a measurement, and the default became evidence"
    )


def test_the_live_call_sites_no_longer_assert_a_grade():
    """The fabrication had two sources, and both are call sites."""
    import ast
    import pathlib

    for path, func in (
        ("core/cognition/cognitive_integration_layer.py", "record_turn"),
        ("core/orchestrator/mixins/context_streaming.py", "record_turn"),
    ):
        tree = ast.parse(pathlib.Path(path).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "attr", "") != func:
                continue
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            for name in ("response_quality", "identity_markers_present"):
                if name not in kwargs:
                    continue
                value = kwargs[name]
                assert isinstance(value, ast.Constant) and value.value is None, (
                    f"{path} still asserts {name} to Cortana without measuring it"
                )


# ── 3bb1d409: the trajectory survives a restart ─────────────────────────────


def test_cognitive_health_survives_a_restart(tmp_path):
    path = tmp_path / "c.json"
    first = CognitiveHealthMonitor(persist_path=str(path))
    for _ in range(30):
        first.record_turn(
            context_tokens=100,
            max_tokens=8192,
            response_quality=0.9,
            identity_markers_present=True,
            topics_in_play=2,
            resolved_topics=1,
        )
    first._save_state()

    second = CognitiveHealthMonitor(persist_path=str(path))
    assert second._measured_turns == first._measured_turns
    assert second._coherence_score == pytest.approx(first._coherence_score)
    assert second._total_turns == first._total_turns


def test_a_corrupt_health_journal_does_not_refuse_construction(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{ not json")
    monitor = CognitiveHealthMonitor(persist_path=str(path))
    assert monitor._total_turns == 0


# ── f6b47140: a fill ratio is not a mood ────────────────────────────────────


def test_the_prompt_never_tells_the_model_it_is_melancholic(tmp_path):
    monitor = CognitiveHealthMonitor(persist_path=str(tmp_path / "c.json"))
    monitor.record_turn(context_tokens=10, max_tokens=8192, topics_in_play=0)
    assert monitor._phase is CortanaPhase.MELANCHOLIA  # the internal stage still exists

    injection = monitor.get_system_prompt_injection()
    lowered = injection.lower()
    for word in ("melancholia", "melancholy", "apathetic", "jealousy", "anger"):
        assert word not in lowered, (
            f"the prompt asserted an affective condition ({word}) from a fill ratio"
        )
    assert "not measured" in lowered


def test_the_prompt_reports_coherence_only_once_it_is_measured(tmp_path):
    monitor = CognitiveHealthMonitor(persist_path=str(tmp_path / "c.json"))
    monitor.record_turn(context_tokens=4000, max_tokens=8192, topics_in_play=2)
    assert "not measured" in monitor.get_system_prompt_injection().lower()

    for _ in range(CognitiveHealthMonitor.MIN_MEASURED_TURNS_TO_REPORT):
        monitor.record_turn(
            context_tokens=4000,
            max_tokens=8192,
            response_quality=0.9,
            identity_markers_present=True,
            topics_in_play=2,
            resolved_topics=2,
        )
    reported = monitor.get_system_prompt_injection().lower()
    assert "graded turns" in reported and "not measured" not in reported


# ── a05a35dd: authority is resolved, not claimed ────────────────────────────


def test_a_governance_claim_is_not_governance(tmp_path):
    engine = ProgressiveAutonomySystem(persist_path=str(tmp_path / "t.json"))
    allowed, _ = engine.can_do("wipe_disk", risk_level="critical", governed=True)
    assert allowed is False, (
        "the flag that unlocked a critical action was supplied by the caller "
        "asking permission"
    )


def test_real_governance_does_clear_a_critical_action(tmp_path):
    from core.governance_context import local_internal_governed_scope

    engine = ProgressiveAutonomySystem(persist_path=str(tmp_path / "t.json"))
    with local_internal_governed_scope("test.edi_real_scope"):
        allowed, _ = engine.can_do("wipe_disk", risk_level="critical")
    assert allowed is True


def test_an_unreadable_prohibition_surface_drops_the_authorization(tmp_path, monkeypatch):
    """A directive store that cannot be read is not permission."""
    import core.governance.standing_directives as directives

    def _boom():
        raise RuntimeError("directives unreadable")

    monkeypatch.setattr(directives, "get_standing_directives", _boom)
    engine = ProgressiveAutonomySystem(persist_path=str(tmp_path / "t.json"))
    _governed, authorized = engine._resolve_authority(
        "wipe_disk", claimed_governed=False, claimed_user_authorized=True
    )
    assert authorized is False


# ── ad5752a2: a cognition modifier has an owner ─────────────────────────────


def test_a_modifier_write_records_who_wrote_it():
    from core.cognition.state_modifiers import (
        MODIFIER_PROVENANCE_KEY,
        modifier_owner,
        set_modifier,
    )

    modifiers: dict = {}
    assert set_modifier(modifiers, "social_tension", 0.4, owner="ava") is True
    assert modifiers["social_tension"] == 0.4
    assert modifier_owner(modifiers, "social_tension") == "ava"
    assert modifiers[MODIFIER_PROVENANCE_KEY]["social_tension"]["revision"] == 1


def test_a_second_owner_cannot_silently_take_a_modifier():
    from core.cognition.state_modifiers import set_modifier

    modifiers: dict = {}
    set_modifier(modifiers, "threat_level", 0.1, owner="immune_system")
    landed = set_modifier(modifiers, "threat_level", 0.9, owner="ava")
    assert landed is False
    assert modifiers["threat_level"] == 0.1, (
        "one owner overwrote another's belief about the same key, and the "
        "loser had no way to find out"
    )


def test_an_owner_may_update_its_own_reading():
    from core.cognition.state_modifiers import set_modifier

    modifiers: dict = {}
    set_modifier(modifiers, "social_tension", 0.1, owner="ava")
    assert set_modifier(modifiers, "social_tension", 0.8, owner="ava") is True
    assert modifiers["social_tension"] == 0.8


def test_ava_writes_modifiers_through_the_owned_path():
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("core/fictional/ava.py").read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        value = node.value
        if (
            isinstance(value, ast.Attribute)
            and value.attr == "modifiers"
            and isinstance(node.ctx, ast.Store)
        ):
            raise AssertionError(
                f"line {node.lineno}: a cognition modifier is still assigned "
                "directly, with no owner and no conflict detection"
            )


# ── b261f498 / 9f828005: the social model is a person's, and says so ────────


def test_the_social_model_is_partitioned_by_person(tmp_path, monkeypatch):
    from core.container import ServiceContainer

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda *a, **k: None))
    a = SocialModelingEngine(persist_path=str(tmp_path / "a.json"), user_id="bryan")
    b = SocialModelingEngine(persist_path=str(tmp_path / "b.json"), user_id="someone else")
    assert a.user_id != b.user_id
    assert a.persist_path != b.persist_path

    default = SocialModelingEngine.__new__(SocialModelingEngine)
    assert "/" not in default._resolve_user_id("../../etc/passwd")


def test_a_persisted_social_model_is_redacted_and_attributed(tmp_path, monkeypatch):
    from core.container import ServiceContainer

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda *a, **k: None))
    engine = SocialModelingEngine(persist_path=str(tmp_path / "m.json"), user_id="bryan")
    engine.model.personal_disclosures = [
        "my key is sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcd"
    ]
    payload = engine._persistable()
    flat = json.dumps(payload)
    assert "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcd" not in flat
    assert payload["user_id"] == "bryan"
    assert payload["schema"] == SocialModelingEngine.MODEL_SCHEMA


def test_a_heuristic_reading_is_labelled_one(tmp_path, monkeypatch):
    from core.container import ServiceContainer

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda *a, **k: None))
    engine = SocialModelingEngine(persist_path=str(tmp_path / "m.json"), user_id="bryan")

    engine.model.total_interactions = 2
    early = engine.get_context_injection()
    assert "not enough observations" in early, (
        "three messages produced numbers presented to the model as beliefs "
        "about a person"
    )

    engine.model.total_interactions = 50
    later = engine.get_context_injection().lower()
    assert "heuristic" in later and "inferred" in later
    assert "not stated by the person" in later


# ── 2b26af7d: a resilience core attempts a repair ───────────────────────────


def test_an_unhealthy_subsystem_is_asked_to_repair_itself(monkeypatch):
    from core.container import ServiceContainer

    repaired: list[str] = []

    class _Service:
        @staticmethod
        def recover():
            repaired.append("called")

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, **k: _Service() if name == "memory_facade" else None),
    )
    core = DistributedResilienceCore()
    core.register_subsystem("memory_facade")
    for _ in range(core._failure_threshold):
        core._report_failure("memory_facade", "probe failed")

    assert repaired == ["called"], (
        "a subsystem went unhealthy and nothing was asked to repair it"
    )


def test_a_subsystem_with_no_repair_method_records_the_request(monkeypatch):
    from core.container import ServiceContainer

    monkeypatch.setattr(
        ServiceContainer, "get", staticmethod(lambda name, **k: object())
    )
    core = DistributedResilienceCore()
    core.register_subsystem("orchestrator")
    for _ in range(core._failure_threshold):
        core._report_failure("orchestrator", "probe failed")

    pending = core.pending_repairs()
    assert "orchestrator" in pending
    assert pending["orchestrator"]["reason"] == "no_repair_method_published"


def test_repair_is_rate_limited(monkeypatch):
    from core.container import ServiceContainer

    calls: list[str] = []

    class _Service:
        @staticmethod
        def restart():
            calls.append("x")

    monkeypatch.setattr(
        ServiceContainer, "get", staticmethod(lambda *a, **k: _Service())
    )
    core = DistributedResilienceCore()
    core.register_subsystem("server")
    assert core._request_recovery("server", "down") is True
    assert core._request_recovery("server", "down") is False
    assert len(calls) == 1, "repair retried without a floor is a hot loop"


def test_a_recovered_subsystem_clears_its_repair_request(monkeypatch):
    from core.container import ServiceContainer

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda *a, **k: object()))
    core = DistributedResilienceCore()
    core.register_subsystem("orchestrator")
    for _ in range(core._failure_threshold):
        core._report_failure("orchestrator", "probe failed")
    assert core.pending_repairs()

    core._report_success("orchestrator")
    assert "orchestrator" not in core.pending_repairs()


# ── 20d4fbd3 / be9ce637: the idle loop survives, and keeps looking ──────────


@pytest.mark.asyncio
async def test_a_raising_cycle_does_not_end_the_idle_loop(monkeypatch):
    """Imports and service lookups used to escape the loop body entirely."""
    scheduler = TemporalDilationScheduler()
    calls: list[int] = []

    async def _cycle(_brain=None):
        calls.append(1)
        if len(calls) < 3:
            raise ImportError("a service module went missing")
        scheduler._is_running = False

    scheduler._idle_cycle = _cycle  # type: ignore[method-assign]
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    await scheduler.run_idle_loop()
    assert len(calls) == 3, "the loop ended on the first raising cycle"


@pytest.mark.asyncio
async def test_the_idle_loop_finds_a_brain_that_arrives_late(monkeypatch):
    """The starter gave the brain thirty seconds and then never looked again."""
    from core.container import ServiceContainer

    class _Orch:
        brain = None
        status = None
        _last_user_interaction_time = 0.0

    orch = _Orch()
    scheduler = TemporalDilationScheduler(orchestrator=orch)

    monkeypatch.setattr(
        ServiceContainer, "get", staticmethod(lambda *a, **k: None)
    )
    await scheduler._idle_cycle()
    assert scheduler._brain is None

    orch.brain = object()
    await scheduler._idle_cycle()
    assert scheduler._brain is orch.brain, (
        "a brain that appeared after the first cycle was never picked up"
    )


async def _no_sleep(_delay, *_a, **_k):
    return None


def test_the_deferred_starter_no_longer_gives_up():
    import inspect

    from core.fictional_ai_synthesis import register_all_fictional_engines

    source = inspect.getsource(register_all_fictional_engines)
    assert "brain never became available" not in source, (
        "the starter still abandons the idle loop when the brain is slow to boot"
    )


# ── a793b8cc: synchronous cognition calls get a thread and a deadline ───────


def test_the_idle_cycle_offloads_its_synchronous_calls():
    import ast
    import inspect

    from core.fictional_ai_synthesis import TemporalDilationScheduler as T

    tree = ast.parse(inspect.getsource(T._idle_cycle).lstrip())
    direct: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", "")
        if name in {"snapshot", "forge_fast"}:
            # These must be reached through to_thread, so the call node
            # here should be the ARGUMENT of to_thread rather than a call.
            direct.append(name)
    assert not direct, (
        f"{direct} are still called directly from the coroutine; blocking work "
        "on the loop is what this finding is about"
    )


# ── 9eb5b7c6: one owner per service ─────────────────────────────────────────


def test_a_short_name_is_an_alias_not_a_second_owner():
    import ast
    import pathlib

    source = pathlib.Path("core/fictional/registry.py").read_text()
    tree = ast.parse(source)
    registrations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "register_instance":
            if node.args and isinstance(node.args[0], ast.Constant):
                registrations.append(str(node.args[0].value))
    duplicates = {n for n in registrations if registrations.count(n) > 1}
    assert not duplicates, f"{duplicates} registered more than once"
    assert "register_alias" in source, (
        "the short names are still full instance registrations, so the "
        "container holds two owners for one lifecycle"
    )
