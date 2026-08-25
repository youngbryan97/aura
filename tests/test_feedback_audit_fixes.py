import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import interface.routes.chat_conversation_repair as _chat_conversation_repair
import interface.routes.chat_memory_state as _chat_memory_state
import interface.routes.chat_protected_prompt as _chat_protected_prompt
from core.runtime.governance_policy import (
    allow_direct_user_shortcut,
    allow_intent_hint_bypass,
    allow_simple_query_bypass,
)
from core.skills.train_self import TrainSelfSkill as CoreTrainSelfSkill
from skills.train_self import TrainSelfSkill as LegacyTrainSelfSkill
from tests.chat_lane_support import patch_chat_lane


@pytest.mark.parametrize("skill_name", ["omni_log_error", "omni_log_critical"])
def test_error_intelligence_demotes_omni_log_warning(caplog, tmp_path, skill_name):
    from core.self_modification.error_intelligence import StructuredErrorLogger

    logger = StructuredErrorLogger(str(tmp_path))
    caplog.set_level(logging.WARNING, logger="SelfModification.ErrorIntelligence")

    try:
        raise RuntimeError("telemetry loop failure")
    except RuntimeError as exc:
        asyncio.run(logger.log_error(exc, {}, skill_name=skill_name))

    assert f"Error logged: RuntimeError in {skill_name}" not in caplog.text


def test_error_intelligence_targets_deepest_aura_traceback_frame(tmp_path):
    from core.self_modification.error_intelligence import StructuredErrorLogger

    logger = StructuredErrorLogger(str(tmp_path))
    entered = []

    def inner_failure():
        entered.append(True)
        raise RuntimeError("deep failure")

    expected_line = inner_failure.__code__.co_firstlineno + 2

    def outer_wrapper():
        inner_failure()

    try:
        outer_wrapper()
    except RuntimeError as exc:
        event = asyncio.run(logger.log_error(exc, {}, skill_name="deep_frame_test"))

    assert entered == [True]
    assert event.file_path == os.path.realpath(__file__)
    assert event.line_number == expected_line


def test_error_intelligence_refuses_another_checkout_as_her_own_source():
    """The rule itself, independent of where this file happens to live."""
    from core.self_modification.error_intelligence import (
        _SOURCE_ROOT_REALPATH,
        _is_her_own_source,
    )

    own = os.path.join(_SOURCE_ROOT_REALPATH, "core", "somewhere.py")
    assert _is_her_own_source(own)

    for nested in (
        os.path.join(_SOURCE_ROOT_REALPATH, ".claude", "worktrees", "x", "core", "a.py"),
        os.path.join(_SOURCE_ROOT_REALPATH, ".venv", "lib", "b.py"),
        os.path.join(_SOURCE_ROOT_REALPATH, "dist", "c.py"),
        os.path.join(os.sep, "elsewhere", "core", "d.py"),
    ):
        assert not _is_her_own_source(nested), nested


def test_error_intelligence_logs_structured_runtime_reason(caplog, tmp_path):
    from core.self_modification.error_intelligence import StructuredErrorLogger

    error_logger = StructuredErrorLogger(str(tmp_path))
    caplog.set_level(logging.WARNING, logger="SelfModification.ErrorIntelligence")
    context = {
        "subsystem": "mlx_client",
        "reason": "empty_generation_exhausted",
        "classification": "foreground_blocking",
        "detail": "Aura-32B:attempt=2:no_visible_text",
    }

    asyncio.run(
        error_logger.log_error(
            RuntimeError("synthetic health incident"),
            context,
            skill_name="mlx_client",
            goal="empty_generation_exhausted",
        )
    )

    assert "reason=empty_generation_exhausted" in caplog.text
    assert "classification=foreground_blocking" in caplog.text
    assert "Aura-32B:attempt=2:no_visible_text" in caplog.text


def test_contextless_error_fingerprint_uses_structured_incident_identity():
    from core.self_modification.error_intelligence import ErrorEvent

    common = {
        "timestamp": 1.0,
        "error_type": "RuntimeError",
        "error_message": "synthetic health incident",
        "stack_trace": "",
        "file_path": None,
        "line_number": None,
    }
    empty_generation = ErrorEvent(
        **common,
        context={
            "subsystem": "mlx_client",
            "reason": "empty_generation_exhausted",
            "classification": "foreground_blocking",
        },
        skill_name="mlx_client",
        goal="empty_generation_exhausted",
    )
    queue_failure = ErrorEvent(
        **common,
        context={
            "subsystem": "mlx_client",
            "reason": "request_queue_failed",
            "classification": "foreground_blocking",
        },
        skill_name="mlx_client",
        goal="request_queue_failed",
    )

    assert empty_generation.fingerprint() != queue_failure.fingerprint()


def test_omni_tracer_does_not_turn_forwarded_info_logs_into_failure_pressure():
    from core.resilience.omni_tracer import _classify_forwarded_log

    severity, classification = _classify_forwarded_log(
        "log_error",
        "[GUI] 2026-06-06 20:50:57,487 - Aura.Core - INFO - Webhook alerting disabled.",
        "error",
    )

    assert (severity, classification) == ("info", "non_critical_fallback")


def test_omni_tracer_demotes_recoverable_boot_health_contract_logs():
    from core.resilience.omni_tracer import _classify_forwarded_log

    severity, classification = _classify_forwarded_log(
        "log_critical",
        "Aura.HealthContract | HEALTH CONTRACT: CRITICAL - some critical services missing",
        "critical",
    )

    assert (severity, classification) == ("error", "background_degraded")


def test_omni_tracer_demotes_health_contract_service_summary_lines():
    from core.resilience.omni_tracer import _classify_forwarded_log

    severity, classification = _classify_forwarded_log(
        "log_critical",
        "  [✓] [O] Metrics Exporter: alive",
        "critical",
        error_type="Aura.HealthContract",
    )

    assert (severity, classification) == ("error", "background_degraded")


def test_dream_journal_thread_save_uses_local_file_write_governance(monkeypatch, tmp_path):
    from core.adaptation import dream_journal
    from core.governance_context import require_governance

    calls = []

    class Gateway:
        def append_text(self, path, text, *, source):
            token = require_governance(
                f"file_write_gateway.append_text:{source}",
                strict=True,
                allowed_domains=("file_write",),
            )
            calls.append((Path(path).name, source, token.domain, "dream body" in text))

        # Async lane delegators: production code now calls *_async; fakes
        # must mirror the gateway surface or every governed write breaks.
        async def append_text_async(self, *args, **kwargs):
            return self.append_text(*args, **kwargs)

    journal = dream_journal.DreamJournal.__new__(dream_journal.DreamJournal)
    journal.journal_file = tmp_path / "dream_journal.txt"

    monkeypatch.setattr(dream_journal, "get_file_write_gateway", lambda: Gateway())

    journal._save_dream("dream body", [SimpleNamespace(description="seed memory")])

    assert calls == [
        (
            "dream_journal.txt",
            "adaptation.dream_journal.journal",
            "file_write",
            True,
        )
    ]


def test_thought_tracer_uses_local_file_write_governance(monkeypatch, tmp_path):
    import core.introspection.thought_tracer as thought_tracer
    from core.governance_context import require_governance

    calls = []

    class Gateway:
        def append_text(self, path, text, *, encoding="utf-8", source):
            token = require_governance(
                f"file_write_gateway.append_text:{source}",
                strict=True,
                allowed_domains=("file_write",),
            )
            calls.append((Path(path).name, source, token.domain, "probe" in text))

        # Async lane delegators: production code now calls *_async; fakes
        # must mirror the gateway surface or every governed write breaks.
        async def append_text_async(self, *args, **kwargs):
            return self.append_text(*args, **kwargs)

    tracer = thought_tracer.ThoughtTracer(log_dir=str(tmp_path))
    monkeypatch.setattr(thought_tracer, "get_file_write_gateway", lambda: Gateway())

    tracer.log_cycle("probe objective", {"probe": True}, {"answer": "probe"}, "ok")
    tracer.log_event("probe_event", {"probe": True})

    assert calls == [
        (
            tracer.current_trace_file.name,
            "thought_tracer.log_cycle",
            "file_write",
            True,
        ),
        (
            tracer.current_trace_file.name,
            "thought_tracer.log_event",
            "file_write",
            True,
        ),
    ]


@pytest.mark.parametrize("skill_kind", ["core", "legacy"])
def test_train_self_collects_paired_user_context(tmp_path, skill_kind):
    if skill_kind == "core":
        skill = CoreTrainSelfSkill()
    else:
        skill = LegacyTrainSelfSkill(workspace_root=str(tmp_path))

    skill.dataset_path = tmp_path / f"{skill_kind}_dataset.jsonl"
    history = [
        {"speaker": "user", "text": "What are you experiencing right now?"},
        {"role": "Aura", "content": "Telemetry says I'm calm and focused."},
        {"role": "user", "content": "And what is your free energy state?"},
        {"role": "assistant", "content": "Free energy is trending down, action tendency is reflect."},
    ]

    result = asyncio.run(skill._collect_high_value_memories({"history": history}))

    lines = [
        json.loads(line)
        for line in skill.dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert result["ok"] is True
    assert result["collected"] == 2
    assert [line["input"] for line in lines] == [
        "What are you experiencing right now?",
        "And what is your free energy state?",
    ]
    assert lines[0]["output"] == "Telemetry says I'm calm and focused."


def test_browser_executor_enforces_allowlist_for_localhost_ports(monkeypatch):
    browser_executor = pytest.importorskip("executors.browser_executor")
    monkeypatch.setattr(browser_executor, "ALLOW_ALL_DOMAINS", False)

    assert browser_executor._is_domain_allowed("http://localhost:8000/ui", {"localhost"})
    assert browser_executor._is_domain_allowed("https://docs.python.org/3/", {"python.org"})
    assert not browser_executor._is_domain_allowed("https://example.net", {"localhost"})


@pytest.mark.parametrize(
    ("action_spec", "error"),
    [
        (["not", "a", "dict"], "invalid_action_spec"),
        ({"params": ["not", "params"]}, "invalid_params"),
        ({"url": 42}, "invalid_url"),
    ],
)
def test_browser_executor_rejects_malformed_action_specs(action_spec, error):
    browser_executor = pytest.importorskip("executors.browser_executor")

    result = browser_executor.run_browser_action(action_spec)

    assert result["ok"] is False
    assert result["error"] == error
    assert result["audit"] == []


def test_integrity_guardian_rebuild_clears_current_alert_state(monkeypatch):
    from core.security.integrity_guardian import IntegrityGuardian

    guardian = IntegrityGuardian()
    guardian._alert_count = 4
    guardian._last_issue_count = 4
    guardian._last_ok = False
    guardian._last_tampered = ["core/a.py"]
    guardian._last_missing = ["core/b.py"]

    monkeypatch.setattr(guardian, "_build_manifest", lambda: 7)
    monkeypatch.setattr(guardian, "_save_manifest", lambda: setattr(guardian, "_manifest_hmac", "sig"))

    count = guardian.rebuild_manifest()
    status = guardian.get_status()

    assert count == 7
    assert status["integrity_ok"] is True
    assert status["alert_count"] == 0
    assert status["current_issue_count"] == 0
    assert status["verification_pending"] is False
    assert status["manifest_revision_stale"] is False
    assert status["last_tampered"] == []
    assert status["last_missing"] == []


def test_integrity_guardian_verify_all_suppresses_git_active_paths(monkeypatch, tmp_path):
    import builtins

    from core.security import integrity_guardian as ig_mod

    monkeypatch.setattr(ig_mod, "_BASE_DIR", tmp_path)
    core_dir = tmp_path / "core"
    builtins.get_task_tracker().create_task(
        builtins.get_storage_gateway().create_dir(
            core_dir,
            cause="test_integrity_guardian_verify_all_suppresses_git_active_paths",
        )
    )
    edited = core_dir / "capability_engine.py"
    stable = core_dir / "health.py"
    edited.write_text("print('edited')\n", encoding="utf-8")
    stable.write_text("print('stable')\n", encoding="utf-8")

    guardian = ig_mod.IntegrityGuardian()
    guardian._manifest = {
        "core/capability_engine.py": "stale-hash",
        "core/health.py": guardian._hash_file(stable),
    }
    monkeypatch.setattr(guardian, "_git_active_paths", lambda: {"core/capability_engine.py"})

    alerts = guardian._verify_all()

    assert alerts == []
    assert guardian.get_status()["current_issue_count"] == 0


def test_integrity_guardian_bounded_boot_verify_reports_pending(monkeypatch, tmp_path):
    import builtins

    from core.security import integrity_guardian as ig_mod

    monkeypatch.setattr(ig_mod, "_BASE_DIR", tmp_path)
    core_dir = tmp_path / "core"
    builtins.get_task_tracker().create_task(
        builtins.get_storage_gateway().create_dir(
            core_dir,
            cause="test_integrity_guardian_bounded_boot_verify_reports_pending",
        )
    )
    files = {}
    for idx in range(4):
        path = core_dir / f"runtime_{idx}.py"
        path.write_text(f"x = {idx}\n", encoding="utf-8")
        files[f"core/runtime_{idx}.py"] = "hash"

    guardian = ig_mod.IntegrityGuardian()
    guardian._manifest = files
    calls = {"count": 0}

    def slow_hash(_path):
        calls["count"] += 1
        time.sleep(0.01)
        return "hash"

    monkeypatch.setattr(guardian, "_hash_file", slow_hash)

    alerts = guardian._verify_all(time_budget_s=0.001)
    status = guardian.get_status()

    assert alerts == []
    assert calls["count"] == 1
    assert status["verification_pending"] is True
    assert status["pending_count"] == 3
    assert status["integrity_ok"] is False


def test_integrity_guardian_rejects_generated_artifact_paths_from_manifest(monkeypatch, tmp_path):
    from core.security import integrity_guardian as ig_mod

    manifest_path = tmp_path / "integrity_manifest.json"
    monkeypatch.setattr(ig_mod, "MANIFEST_PATH", manifest_path)
    guardian = ig_mod.IntegrityGuardian()
    files = {
        "core/security/emergency_protocol.py": "hash",
        "artifacts/current/unified_system_scenario/sandbox/broken_repo/app.py": "hash",
    }
    manifest_path.write_text(
        json.dumps(
            {
                "files": files,
                "signature": guardian._sign_manifest(files),
                "source_revision": "same-revision",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(guardian, "_current_source_revision", lambda: "same-revision")

    assert guardian._load_manifest() is False
    assert guardian._manifest == {}


def test_integrity_guardian_build_manifest_scopes_to_runtime_paths(monkeypatch, tmp_path):
    from core.security import integrity_guardian as ig_mod

    monkeypatch.setattr(ig_mod, "_BASE_DIR", tmp_path)
    (tmp_path / "core").mkdir()
    (tmp_path / "interface").mkdir()
    (tmp_path / "artifacts" / "current").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "core" / "runtime_file.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "interface" / "server.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "aura_main.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "artifacts" / "current" / "generated.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "tests" / "test_generated.py").write_text("x = 1\n", encoding="utf-8")

    guardian = ig_mod.IntegrityGuardian()
    count = guardian._build_manifest()

    assert count == 3
    assert set(guardian._manifest) == {
        "aura_main.py",
        "core/runtime_file.py",
        "interface/server.py",
    }


def test_integrity_guardian_parse_git_status_paths_handles_renames():
    from core.security.integrity_guardian import IntegrityGuardian

    paths = IntegrityGuardian._parse_git_status_paths("R  core/old_name.py -> core/new_name.py")

    assert paths == {"core/old_name.py", "core/new_name.py"}


def test_integrity_guardian_parse_git_status_paths_handles_untracked_files():
    from core.security.integrity_guardian import IntegrityGuardian

    paths = IntegrityGuardian._parse_git_status_paths("?? tests/test_live_runtime_surface_regressions.py")

    assert paths == {"tests/test_live_runtime_surface_regressions.py"}


def test_integrity_guardian_defers_legacy_manifest_refresh_when_source_revision_changes(monkeypatch, tmp_path):
    from core.security import integrity_guardian as ig_mod

    manifest_path = tmp_path / "integrity_manifest.json"
    monkeypatch.setattr(ig_mod, "MANIFEST_PATH", manifest_path)
    guardian = ig_mod.IntegrityGuardian()
    files = {"core/security/emergency_protocol.py": "old"}
    manifest_path.write_text(
        json.dumps(
            {
                "files": files,
                "signature": guardian._sign_manifest(files),
                "source_revision": "old-revision",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(guardian, "_current_source_revision", lambda: "new-revision")

    assert guardian._load_manifest() is True
    status = guardian.get_status()
    assert status["manifest_revision_stale"] is True
    assert status["integrity_ok"] is False


def test_system_health_json_safe_coerces_numpy_scalars_and_arrays():
    np = pytest.importorskip("numpy")
    from interface.routes.system import _json_safe

    payload = {
        "value": np.float32(0.83),
        "series": np.array([np.float32(1.0), np.float32(2.0)], dtype=np.float32),
    }

    safe = _json_safe(payload)

    assert safe == {"value": 0.8299999833106995, "series": [1.0, 2.0]}


def test_substrate_voice_engine_state_exposes_exclamation_flag():
    from core.voice.speech_profile import SpeechProfile
    from core.voice.substrate_voice_engine import SubstrateVoiceEngine

    engine = SubstrateVoiceEngine.__new__(SubstrateVoiceEngine)
    engine._current_profile = SpeechProfile(
        word_budget=24,
        tone_override="enthusiastic",
        question_probability=0.31,
        followup_probability=0.2,
        exclamation_allowed=True,
        substrate_snapshot={"phi": 0.001},
    )
    engine._response_count = 1
    engine._silence_streak = 0

    state = SubstrateVoiceEngine.get_voice_state(engine)

    assert state["exclamation_allowed"] is True


def test_speech_profile_discloses_bounded_field_snapshot_contract():
    from core.voice.speech_profile import SpeechProfile

    profile = SpeechProfile(
        word_budget=24,
        substrate_snapshot={
            "field_snapshot_bounded": 1.0,
            "field_snapshot_cached": 1.0,
            "field_snapshot_age_s": 4.2,
        },
    )

    block = profile.to_constraint_block()

    assert "STATE FRESHNESS" in block
    assert "do not claim exact live physiology" in block


def test_substrate_voice_engine_demo_override_holds_then_expires(monkeypatch):
    import core.voice.substrate_voice_engine as voice_mod

    engine = voice_mod.SubstrateVoiceEngine()
    fake_now = [100.0]

    monkeypatch.setattr(voice_mod.time, "time", lambda: fake_now[0])
    monkeypatch.setattr(voice_mod, "_extract_neurochemicals", lambda: {})
    monkeypatch.setattr(voice_mod, "_extract_homeostasis", lambda: {})
    monkeypatch.setattr(voice_mod, "_extract_unified_field", lambda: {})
    monkeypatch.setattr(voice_mod, "_extract_personality", lambda state: {})
    monkeypatch.setattr(voice_mod, "_extract_social_context", lambda: {})
    monkeypatch.setattr(voice_mod, "_extract_conversation_context", lambda state: {})

    state = SimpleNamespace(
        affect=SimpleNamespace(
            valence=0.8,
            arousal=0.9,
            curiosity=0.85,
            engagement=0.8,
            social_hunger=0.7,
            dominant_emotion="joy",
        )
    )

    engine.set_demo_affect_override(
        mood="tired",
        affect={
            "valence": -0.1,
            "arousal": 0.2,
            "curiosity": 0.2,
            "engagement": 0.25,
            "social_hunger": 0.3,
            "dominant_emotion": "contemplation",
        },
        hold_seconds=30,
    )

    during_hold = engine.compile_profile(state=state, user_message="hey aura", origin="user")
    assert during_hold.substrate_snapshot["arousal"] == pytest.approx(0.2)
    assert during_hold.tone_override == "thoughtful_measured"
    assert engine.get_voice_state()["demo_override"]["mood"] == "tired"

    fake_now[0] = 131.0

    after_expiry = engine.compile_profile(state=state, user_message="hey aura", origin="user")
    assert after_expiry.substrate_snapshot["arousal"] == pytest.approx(0.9)
    assert after_expiry.tone_override == "enthusiastic"
    assert engine.get_voice_state()["demo_override"]["active"] is False


def test_substrate_voice_bounds_unified_field_reads_during_foreground(monkeypatch):
    from core.voice import substrate_voice_engine as voice_mod

    voice_mod._UNIFIED_FIELD_CACHE = {}
    voice_mod._UNIFIED_FIELD_CACHE_AT = 0.0
    calls = {"experiential": 0, "modes": 0}

    class UnifiedField:
        def get_coherence(self):
            return 0.73

        def get_phi_contribution(self):
            return 0.19

        def get_back_pressure(self):
            return {"chemical_urgency": 0.31, "binding_demand": 0.44}

        def get_experiential_quality(self):
            calls["experiential"] += 1
            return {"intensity": 0.91}

        def get_dominant_modes(self, _limit):
            calls["modes"] += 1
            return [{"variance_explained": 0.81}]

    field = UnifiedField()

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(
            lambda name, default=None: SimpleNamespace(unified_field=field)
            if name == "consciousness_bridge"
            else default
        ),
    )
    monkeypatch.setattr(
        "core.runtime.foreground_guard.foreground_activity_reason",
        lambda: "foreground_chat_active",
    )

    snapshot = voice_mod._extract_unified_field()

    assert snapshot["coherence"] == pytest.approx(0.73)
    assert snapshot["phi"] == pytest.approx(0.19)
    assert snapshot["back_pressure_urgency"] == pytest.approx(0.31)
    assert snapshot["binding_demand"] == pytest.approx(0.44)
    assert snapshot["field_snapshot_bounded"] == pytest.approx(1.0)
    assert snapshot["field_snapshot_cached"] == pytest.approx(0.0)
    assert "field_intensity" not in snapshot
    assert calls == {"experiential": 0, "modes": 0}


def test_substrate_voice_reuses_rich_field_cache_under_pressure(monkeypatch):
    from core.voice import substrate_voice_engine as voice_mod

    voice_mod._UNIFIED_FIELD_CACHE = {}
    voice_mod._UNIFIED_FIELD_CACHE_AT = 0.0
    calls = {"experiential": 0, "modes": 0}
    foreground_reason = [""]

    class UnifiedField:
        def get_coherence(self):
            return 0.62

        def get_phi_contribution(self):
            return 0.27

        def get_back_pressure(self):
            return {"chemical_urgency": 0.12, "binding_demand": 0.21}

        def get_experiential_quality(self):
            calls["experiential"] += 1
            return {
                "intensity": 0.76,
                "valence": 0.18,
                "complexity": 0.68,
                "clarity": 0.55,
                "flow": 0.49,
            }

        def get_dominant_modes(self, _limit):
            calls["modes"] += 1
            return [{"variance_explained": 0.59}]

    class MemorySnapshot:
        warning = False
        refuse_heavy_local_generation = False
        level = "normal"

    field = UnifiedField()

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(
            lambda name, default=None: SimpleNamespace(unified_field=field)
            if name == "consciousness_bridge"
            else default
        ),
    )
    monkeypatch.setattr(
        "core.runtime.foreground_guard.foreground_activity_reason",
        lambda: foreground_reason[0],
    )
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: MemorySnapshot(),
    )

    rich = voice_mod._extract_unified_field(cache_max_age_s=60.0)
    assert rich["field_intensity"] == pytest.approx(0.76)
    assert rich["mode_focus"] == pytest.approx(0.59)
    assert rich["field_snapshot_bounded"] == pytest.approx(0.0)
    assert calls == {"experiential": 1, "modes": 1}

    foreground_reason[0] = "foreground_quiet_window"
    cached = voice_mod._extract_unified_field(cache_max_age_s=60.0)

    assert cached["field_intensity"] == pytest.approx(0.76)
    assert cached["mode_focus"] == pytest.approx(0.59)
    assert cached["field_snapshot_bounded"] == pytest.approx(1.0)
    assert cached["field_snapshot_cached"] == pytest.approx(1.0)
    assert calls == {"experiential": 1, "modes": 1}


@pytest.mark.asyncio
async def test_voice_state_endpoint_rehydrates_profile_after_first_exchange(monkeypatch):
    from interface.routes import subsystems as subsystems_module

    class DummyVoiceEngine:
        def __init__(self):
            self.compile_calls = []

        def get_voice_state(self):
            if not self.compile_calls:
                return {"status": "no_profile_compiled"}
            return {"word_budget": 22, "tone": "steady"}

        def compile_profile(self, *, state=None, user_message="", origin="user"):
            self.compile_calls.append(
                {
                    "state": state,
                    "user_message": user_message,
                    "origin": origin,
                }
            )
            return SimpleNamespace(word_budget=22, tone_override="steady")

    dummy_engine = DummyVoiceEngine()
    dummy_state = object()

    monkeypatch.setattr(
        "core.voice.substrate_voice_engine.get_substrate_voice_engine",
        lambda: dummy_engine,
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(
            lambda name, default=None: SimpleNamespace(state=dummy_state)
            if name == "orchestrator"
            else default
        ),
    )
    patch_chat_lane(monkeypatch, "_conversation_log",
        [{"user": "hey aura", "aura": "hey. i'm here."}],
        raising=False,
    )

    response = await subsystems_module.api_voice_state()
    payload = json.loads(response.body)

    assert payload["voice"]["word_budget"] == 22
    assert dummy_engine.compile_calls == [
        {
            "state": dummy_state,
            "user_message": "hey aura",
            "origin": "user",
        }
    ]


@pytest.mark.asyncio
async def test_voice_affect_modulate_endpoint_sets_demo_hold(monkeypatch):
    from interface.routes import subsystems as subsystems_module

    class DummyRequest:
        def __init__(self, body):
            self._body = body

        async def json(self):
            return self._body

    class DummyVoiceEngine:
        def __init__(self):
            self.override_calls = []
            self.compile_calls = []

        def set_demo_affect_override(self, *, mood, affect=None, hold_seconds=30.0):
            self.override_calls.append(
                {
                    "mood": mood,
                    "affect": dict(affect or {}),
                    "hold_seconds": hold_seconds,
                }
            )
            return {
                "active": True,
                "mood": mood,
                "seconds_remaining": hold_seconds,
            }

        def compile_profile(self, *, state=None, user_message="", origin="user"):
            self.compile_calls.append(
                {
                    "state": state,
                    "user_message": user_message,
                    "origin": origin,
                }
            )
            return SimpleNamespace(
                word_budget=18,
                tone_override="thoughtful_measured",
                energy=0.22,
                warmth=0.41,
                directness=0.55,
                playfulness=0.12,
                capitalization="lowercase",
                vocabulary_tier="minimal",
                fragment_ratio=0.28,
                question_probability=0.03,
                followup_probability=0.01,
                exclamation_allowed=False,
            )

    dummy_engine = DummyVoiceEngine()
    dummy_state = object()

    monkeypatch.setattr(
        "core.voice.substrate_voice_engine.get_substrate_voice_engine",
        lambda: dummy_engine,
    )
    monkeypatch.setattr(subsystems_module, "_get_live_orchestrator_state", lambda: dummy_state)

    response = await subsystems_module.api_voice_affect_modulate(
        DummyRequest({"mood": "tired"}),
        None,
    )
    payload = json.loads(response.body)

    assert payload["shifted_to"] == "tired"
    assert payload["hold_seconds"] == 30.0
    assert payload["demo_override"]["active"] is True
    assert dummy_engine.override_calls[0]["mood"] == "tired"
    assert dummy_engine.override_calls[0]["hold_seconds"] == 30.0
    assert dummy_engine.override_calls[0]["affect"]["dominant_emotion"] == "contemplation"
    assert dummy_engine.compile_calls == [
        {
            "state": dummy_state,
            "user_message": "",
            "origin": "user",
        }
    ]


def test_chat_ui_only_shows_onboarding_when_explicitly_requested():
    js_path = Path(__file__).resolve().parents[1] / "interface" / "static" / "aura.js"
    source = js_path.read_text(encoding="utf-8")

    assert "const onboardingRequested = new URLSearchParams(window.location.search).get('onboarding') === '1';" in source
    assert "if (!settings.onboarded && onboardingRequested)" in source


def test_substrate_panel_polls_quickly_for_demo_readiness():
    html_path = Path(__file__).resolve().parents[1] / "interface" / "static" / "substrate.html"
    source = html_path.read_text(encoding="utf-8")

    assert "setInterval(fetchVoiceState, 1000);" in source


def test_substrate_panel_respects_backend_demo_hold_state():
    html_path = Path(__file__).resolve().parents[1] / "interface" / "static" / "substrate.html"
    source = html_path.read_text(encoding="utf-8")

    assert "renderDemoHold(v.demo_override);" in source
    assert "body: JSON.stringify({ mood, hold_seconds: 30 })" in source
    assert "Demo hold ·" in source


def test_aura_main_prefers_stable_homebrew_python_launcher_when_invoked_via_venv_shim():
    launcher_path = Path(__file__).resolve().parents[1] / "aura_main.py"
    source = launcher_path.read_text(encoding="utf-8")

    assert "_maybe_relaunch_with_preferred_python()" in source
    assert "AURA_SKIP_PREFERRED_PYTHON_RELAUNCH" in source
    assert "/opt/homebrew/opt/python@3.12/bin/python3.12" in source
    assert "return candidate" in source
    assert "/opt/homebrew/Cellar/python@3.12" not in source
    # The live mind defaults to the in-process MLX substrate (see project
    # memory: "Aura real mind = MLX"; the launcher forces that default rather
    # than allowing a retired external backend.
    assert 'os.environ["AURA_LOCAL_BACKEND"] = "mlx"' in source


def test_self_mod_engine_on_error_runs_without_event_loop(monkeypatch):
    from core.self_modification import self_modification_engine as sm_mod

    calls = []

    class DummyEvent:
        def fingerprint(self):
            return "fp"

    class DummyErrorIntelligence:
        async def on_error(self, error, context, skill_name, goal):
            calls.append((str(error), context, skill_name, goal))
            return DummyEvent()

    class ImmediateThread:
        def __init__(self, target, name=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(sm_mod.threading, "Thread", ImmediateThread)

    engine = sm_mod.AutonomousSelfModificationEngine.__new__(sm_mod.AutonomousSelfModificationEngine)
    engine.error_intelligence = DummyErrorIntelligence()
    engine._repair_event = asyncio.Event()

    engine.on_error(RuntimeError("boom"), {"source": "test"}, "skill_x", "goal_y")

    assert calls == [("boom", {"source": "test"}, "skill_x", "goal_y")]
    assert engine._repair_event.is_set()


def test_degraded_events_forward_schedules_async_on_error(monkeypatch):
    from core.health import degraded_events as de_mod

    calls = []

    class DummyModifier:
        async def on_error(self, error, context, skill_name=None, goal=None):
            calls.append(
                {
                    "error": str(error),
                    "context": dict(context),
                    "skill_name": skill_name,
                    "goal": goal,
                }
            )

    class DummyOrchestrator:
        self_modifier = DummyModifier()

    class ImmediateThread:
        def __init__(self, target, name=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(de_mod.threading, "Thread", ImmediateThread)

    def fake_get(name, default=None):
        if name == "orchestrator":
            return DummyOrchestrator()
        return default

    # degraded_events resolves the orchestrator via get_runtime_service.
    monkeypatch.setattr(
        "core.runtime.service_registry.get_runtime_service",
        lambda name, default=None: fake_get(name, default),
    )

    de_mod._forward_to_error_intelligence(
        ("sub", "reason", "warning", "background_degraded"),
        {
            "subsystem": "sub",
            "reason": "reason",
            "detail": "detail",
            "severity": "warning",
            "classification": "background_degraded",
            "context": {"origin": "test"},
        },
    )

    assert calls
    assert calls[0]["skill_name"] == "sub"
    assert calls[0]["goal"] == "reason"


def test_record_degraded_event_skips_background_warning_forward(monkeypatch):
    from core.health import degraded_events as de_mod

    forwarded = []

    def fake_forward(key, event, *, exc=None):
        forwarded.append((key, dict(event), exc))

    monkeypatch.setattr(de_mod, "_forward_to_error_intelligence", fake_forward)

    de_mod.record_degraded_event(
        "stability_guardian",
        "degraded_report",
        detail="tick_rate:Event loop lag is elevated",
        severity="warning",
        classification="background_degraded",
    )

    assert forwarded == []


def test_record_degraded_event_keeps_foreground_warning_forward(monkeypatch):
    from core.health import degraded_events as de_mod

    forwarded = []

    def fake_forward(key, event, *, exc=None):
        forwarded.append((key, dict(event), exc))

    monkeypatch.setattr(de_mod, "_forward_to_error_intelligence", fake_forward)

    de_mod.record_degraded_event(
        "mlx_client",
        "request_lock_timeout",
        detail="Aura-32B-v2 owner=Cortex held=13.5s",
        severity="warning",
        classification="foreground_blocking",
    )

    assert len(forwarded) == 1
    assert forwarded[0][1]["classification"] == "foreground_blocking"


@pytest.mark.asyncio
async def test_state_vault_uses_repository_bounded_shm_sync():
    from core.state.vault import StateVaultActor

    calls = []

    class DummyRepo:
        def _serialize(self, state):
            calls.append(("serialize", state.version))
            return '{"version": 3}'

        async def _sync_to_shm(self, state, serialized_state):
            calls.append(("sync", state.version, serialized_state))
            return "hot"

    actor = StateVaultActor.__new__(StateVaultActor)
    actor.repo = DummyRepo()
    actor.shm_transport = None

    class DummyState:
        version = 3

    await StateVaultActor._update_shared_memory_async(actor, DummyState())

    assert calls == [("serialize", 3), ("sync", 3, '{"version": 3}')]


@pytest.mark.asyncio
async def test_state_vault_shm_sync_runs_inside_governed_scope(monkeypatch):
    from core.governance_context import require_governance
    from core.state.vault import StateVaultActor

    calls = []

    class DummyRepo:
        def _serialize(self, state):
            calls.append(("serialize", state.version))
            return '{"version": 7}'

        async def _sync_to_shm(self, state, serialized_state):
            token = require_governance(
                "state.sync_to_shm",
                strict=True,
                allowed_domains=("state_mutation",),
            )
            calls.append(("sync", state.version, serialized_state, token.domain))
            return "full"

    actor = StateVaultActor.__new__(StateVaultActor)
    actor.repo = DummyRepo()
    actor.shm_transport = None

    monkeypatch.setattr("core.governance_context.governance_runtime_active", lambda: True)

    class DummyState:
        version = 7

    await StateVaultActor._update_shared_memory_async(actor, DummyState())

    assert calls == [("serialize", 7), ("sync", 7, '{"version": 7}', "state_mutation")]


def test_governance_policy_blocks_legacy_user_shortcuts_by_default(monkeypatch):
    monkeypatch.delenv("AURA_ALLOW_LEGACY_SHORTCUTS", raising=False)

    assert allow_direct_user_shortcut("user") is False
    assert allow_direct_user_shortcut("background_reflection") is True


def test_governance_policy_only_allows_sanctioned_intent_hints(monkeypatch):
    monkeypatch.delenv("AURA_ALLOW_LEGACY_SHORTCUTS", raising=False)

    assert allow_intent_hint_bypass(
        {"intent_hint": {"tool": "web_search", "params": {}}},
        "user",
    ) is False
    assert allow_intent_hint_bypass(
        {"intent_hint": {"tool": "web_search", "params": {}, "constitutional_hint": True}},
        "agency_core",
    ) is True


def test_governance_policy_keeps_user_simple_queries_on_governed_path(monkeypatch):
    monkeypatch.delenv("AURA_ALLOW_LEGACY_SHORTCUTS", raising=False)

    assert allow_simple_query_bypass("hey", {"origin": "user"}) is False
    assert allow_simple_query_bypass("hey", {"origin": "background_reflection"}) is True


def test_capability_engine_does_not_search_for_capability_questions():
    from core.capability_engine import CapabilityEngine, SkillMetadata

    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.skills = {
        "web_search": SkillMetadata(
            name="web_search",
            description="Search the web",
            trigger_patterns=[r"search (?:for|the web|online|the internet)"],
        )
    }

    assert engine.detect_intent("Can you search the internet?") == []
    assert engine.detect_intent("Can you search the internet for Aura Luna?") == ["web_search"]


def test_capability_engine_does_not_match_hyphenated_skill_name_fragments():
    from core.capability_engine import CapabilityEngine, SkillMetadata

    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.skills = {
        "speak": SkillMetadata(
            name="speak",
            description="Speak aloud",
            trigger_patterns=[],
        )
    }
    CapabilityEngine._load_default_trigger_patterns(engine)

    assert engine.detect_intent("Drop the therapy-speak and just answer plainly.") == []
    assert engine.detect_intent("Use speak to say this out loud.") == ["speak"]


def test_capability_engine_treats_foreground_context_as_user_source():
    from core.capability_engine import CapabilityEngine

    engine = CapabilityEngine.__new__(CapabilityEngine)

    assert engine._resolve_execution_source({"user_facing": True, "objective": "check network"}) == "user"
    assert engine._resolve_execution_source({"origin": "api", "objective": "check network"}) == "api"


def test_generic_reply_detector_flags_live_tool_prompt_artifact():
    from interface.routes.chat import _looks_generic_assistantish

    generic, reason = _looks_generic_assistantish(
        "What time is it right now?",
        "## LIVE TOOL OPTIONS\nMost relevant right now:\n- clock: Check time and date.",
    )

    assert generic is True
    assert reason == "prompt_artifact"


def test_grounded_authority_reply_includes_observability_note(monkeypatch):
    import core.consciousness.authority_audit as audit_mod
    from interface.routes import chat as chat_route

    class DummyAuthority:
        def get_status(self):
            return {
                "total_requests": 12,
                "allowed": 8,
                "constrained": 2,
                "blocked": 1,
                "critical_passes": 1,
                "current_field_coherence": 0.83,
                "block_rate": 0.0833,
            }

    class DummyBridge:
        def get_status(self):
            return {"layers_active": 8, "tick_count": 144, "uptime_s": 26.4}

    class DummyAudit:
        def verify(self):
            return {
                "total_receipts": 12,
                "total_effects": 12,
                "coverage_ratio": 1.0,
                "verdict": "CLEAN",
            }

        def get_recent_receipts(self, n=3):
            return [
                {
                    "decision": "CRITICAL_PASS",
                    "source": "grounded_authority_report",
                    "category": "RESPONSE",
                    "content": "Were you authorized to answer my last q",
                }
            ]

    def fake_get(name, default=None):
        if name == "substrate_authority":
            return DummyAuthority()
        if name == "consciousness_bridge":
            return DummyBridge()
        return default

    monkeypatch.setattr(chat_route.ServiceContainer, "get", staticmethod(fake_get))
    monkeypatch.setattr(audit_mod, "get_audit", lambda: DummyAudit())

    reply = chat_route._build_grounded_introspection_reply(
        "Were you authorized to answer my last question? What did your substrate authority decide?",
        authority_observability_note=(
            "This governance report is being emitted under an observability override."
        ),
    )

    assert "observability override" in reply


@pytest.mark.asyncio
async def test_grounded_traceability_reply_uses_recent_output_receipt(tmp_path):
    from core.runtime.receipts import OutputReceipt, get_receipt_store, reset_receipt_store
    from interface.routes import chat as chat_route

    reset_receipt_store()
    store = get_receipt_store(tmp_path / "receipts")
    store.emit(
        OutputReceipt(
            receipt_id="out-1",
            cause="chat_response",
            origin="api",
            target="primary",
            digest="abc123",
            created_at=1714183200.0,
        )
    )

    reply = await chat_route._build_grounded_traceability_reply(
        "I’m not asking you to prove consciousness. I’m asking for engineering traceability. Then give one safe example only: the most recent non-private action you took that has a log line or event ID."
    )

    reset_receipt_store()

    assert reply is not None
    assert "EventID: out-1" in reply
    assert "Subsystem: Output.api" in reply
    assert "FutureBehavior: no" in reply


@pytest.mark.asyncio
async def test_grounded_traceability_reply_resolves_referential_followup(tmp_path):
    from core.runtime.receipts import OutputReceipt, get_receipt_store, reset_receipt_store
    from interface.routes import chat as chat_route

    reset_receipt_store()
    store = get_receipt_store(tmp_path / "receipts")
    store.emit(
        OutputReceipt(
            receipt_id="out-followup",
            cause="chat_fastpath:grounded_traceability",
            origin="api",
            target="primary",
            digest="def456",
            created_at=1714183260.0,
        )
    )

    async with chat_route._conversation_log_lock:
        chat_route._conversation_log.clear()
        chat_route._conversation_log.append(
            {
                "id": "x1",
                "timestamp": "2026-04-26T00:58:40Z",
                "user": (
                    "I’m not asking you to prove consciousness. I’m asking for engineering traceability. "
                    "Then give one safe example only: the most recent non-private action you took that has a log line or event ID."
                ),
                "aura": "",
                "status": "pending",
            }
        )

    reply = await chat_route._build_grounded_traceability_reply("Can you answer it")

    reset_receipt_store()

    assert reply is not None
    assert "EventID: out-followup" in reply
    assert "Subsystem: Output.api" in reply


def test_grounded_internal_state_reply_uses_live_voice_snapshot(monkeypatch):
    import core.consciousness.self_report as self_report_mod
    from interface.routes import chat as chat_route

    class DummyClosure:
        def get_status(self):
            return {"attention_focus": "the live substrate panel"}

    class DummySelfReportEngine:
        def generate_state_report(self):
            return "I'm steady and tracking the moment."

    monkeypatch.setattr(
        _chat_conversation_repair,
        "_resolve_live_voice_state",
        lambda user_message="", refresh=True: {
            "tone": "steady",
            "energy": 0.4123,
            "warmth": 0.6123,
            "directness": 0.7123,
            "playfulness": 0.2123,
            "substrate_snapshot": {
                "valence": 0.1337,
                "arousal": 0.2448,
                "curiosity": 0.3559,
                "coherence": 0.4661,
                "phi": 0.5772,
                "mode_focus": 0.6883,
            },
        },
        raising=False,
    )
    monkeypatch.setattr(self_report_mod, "SelfReportEngine", DummySelfReportEngine)
    monkeypatch.setattr(
        chat_route.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: DummyClosure() if name == "executive_closure" else default),
    )

    reply = chat_route._build_grounded_introspection_reply("What are you experiencing right now?")

    # Introspection replies should use natural language, not raw telemetry
    assert "Live substrate snapshot:" not in reply
    assert "energy=0.4123" not in reply
    # Should still contain some kind of response
    assert len(reply) > 0


def test_unitary_response_compact_prompt_uses_live_voice_snapshot(monkeypatch):
    import core.voice.substrate_voice_engine as voice_mod
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.current_objective = "the live substrate panel"

    monkeypatch.setattr(
        voice_mod,
        "get_live_voice_state",
        lambda **kwargs: {
            "tone": "steady",
            "energy": 0.4123,
            "warmth": 0.6123,
            "directness": 0.7123,
            "playfulness": 0.2123,
            "word_budget": 42,
            "question_prob": 0.1111,
            "followup_prob": 0.2222,
            "substrate_snapshot": {
                "valence": 0.1337,
                "arousal": 0.2448,
                "curiosity": 0.3559,
                "coherence": 0.4661,
                "phi": 0.5772,
                "field_intensity": 0.6883,
                "field_clarity": 0.7994,
                "field_flow": 0.8115,
                "field_complexity": 0.9226,
                "field_valence": 0.1557,
                "mode_focus": 0.2668,
            },
        },
    )

    phase = UnitaryResponsePhase.__new__(UnitaryResponsePhase)
    prompt = phase._build_compact_router_system_prompt(state)

    # System prompt should use voice shaping cues, not raw numeric telemetry
    assert "YOUR LIVE SUBSTRATE SNAPSHOT" not in prompt
    assert "Energy: 0.4123" not in prompt
    assert "Phi: 0.5772" not in prompt
    # Should still contain voice shaping context
    assert "VOICE SHAPING" in prompt or "Tone:" in prompt


@pytest.mark.asyncio
async def test_state_machine_preserves_origin_when_executing_skill():
    from core.cognitive.state_machine import StateMachine

    captured = {}

    class DummyOrchestrator:
        capability_engine = object()

        async def execute_tool(self, tool_name, params, **kwargs):
            captured["tool_name"] = tool_name
            captured["params"] = params
            captured["kwargs"] = kwargs
            return {"ok": True, "message": "done"}

    sm = StateMachine(orchestrator=DummyOrchestrator())
    sm.llm = None

    reply, tools = await sm._execute_skill_logic(
        "web_search",
        {"query": "latest aura status"},
        "search for latest aura status",
        autonomic=True,
        origin="voice",
    )

    assert reply == "done"
    assert tools == ["web_search"]
    assert captured["kwargs"]["origin"] == "voice"


def test_unitary_response_extracts_grounded_search_query():
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    query = UnitaryResponsePhase._extract_grounded_search_query(
        "Search the web for the official Python 3.12 documentation homepage and tell me only the page title."
    )

    assert query == "the official Python 3.12 documentation homepage"


def test_unitary_response_formats_page_title_search_reply():
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    reply = UnitaryResponsePhase._format_grounded_search_reply(
        "Search the web for the official Python 3.12 documentation homepage and tell me only the page title.",
        {
            "ok": True,
            "results": [
                {
                    "title": "Python 3.12.12 Documentation",
                    "snippet": "Docs for Python 3.12.12",
                    "url": "https://docs.python.org/3.12/",
                }
            ],
        },
    )

    assert reply == "Python 3.12.12 Documentation"


def test_unitary_response_uses_cached_grounded_search_result():
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    state = SimpleNamespace(
        response_modifiers={
            "last_skill_run": "web_search",
            "last_skill_ok": True,
            "last_skill_result_payload": {
                "ok": True,
                "results": [
                    {
                        "title": "Python 3.12.12 Documentation",
                        "snippet": "Docs for Python 3.12.12",
                        "url": "https://docs.python.org/3.12/",
                    }
                ],
            },
        }
    )

    reply = UnitaryResponsePhase._build_cached_grounded_search_reply(
        state,
        "Search the web for the official Python 3.12 documentation homepage and tell me only the page title.",
        SimpleNamespace(requires_search=True),
    )

    assert reply == "Python 3.12.12 Documentation"


def test_unitary_response_uses_working_memory_skill_result_when_cached_payload_missing():
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    state = SimpleNamespace(
        response_modifiers={
            "last_skill_run": "web_search",
            "last_skill_ok": True,
        },
        cognition=SimpleNamespace(
            working_memory=[
                {
                    "role": "system",
                    "content": "[SKILL RESULT: web_search] ✅ Python 3.12.0 Documentation",
                    "metadata": {"type": "skill_result", "skill": "web_search", "ok": True},
                }
            ]
        ),
    )

    reply = UnitaryResponsePhase._build_cached_grounded_search_reply(
        state,
        "Search the web for the official Python 3.12 documentation homepage and tell me only the page title.",
        SimpleNamespace(requires_search=True),
    )

    assert reply == "Python 3.12.0 Documentation"




@pytest.mark.asyncio
async def test_self_model_replays_pending_updates():
    from core.self_model import SelfModel

    model = SelfModel(id="test-self-model")

    async def fake_persist():
        return None

    model.persist = fake_persist  # type: ignore[method-assign]
    model._belief_update_decision = lambda key, value, note: (False, "epistemic_reconciliation_required:1", False)  # type: ignore[method-assign]

    snap = await model.update_belief("executive_closure", {"coherence": 0.88}, note="sync")

    assert snap.summary == "deferred update executive_closure"
    assert model.pending_updates

    model._belief_update_decision = lambda key, value, note: (True, "", False)  # type: ignore[method-assign]
    await model._flush_pending_updates(limit=3)

    assert not model.pending_updates
    assert model.beliefs["executive_closure"] == {"coherence": 0.88}


def test_authority_prompts_are_treated_as_grounded_requests():
    from interface.routes.chat import _looks_generic_assistantish

    generic, reason = _looks_generic_assistantish(
        "Were you authorized to answer my last question? What did your substrate authority decide?",
        "Can you clarify what you mean?",
    )

    assert generic is True
    assert reason == "telemetry_request_deflected"


def test_prompt_artifact_replies_are_rejected_for_user_facing_chat():
    from interface.routes.chat import _looks_generic_assistantish

    generic, reason = _looks_generic_assistantish(
        "What do you honestly think this architecture is strongest at?",
        "Based on the current context, the most appropriate skill would be native_chat. <|endoftext|>",
    )

    assert generic is True
    assert reason == "prompt_artifact"


def test_generic_architecture_generalization_is_rejected():
    from interface.routes.chat import _looks_generic_assistantish

    generic, reason = _looks_generic_assistantish(
        "What do you honestly think this architecture is strongest at?",
        "I excel at natural language processing and generating human-like responses.",
    )

    assert generic is True
    assert reason == "generic_architecture_generalization"


def test_false_tool_denial_phrase_is_rejected():
    from interface.routes.chat import _looks_generic_assistantish

    generic, reason = _looks_generic_assistantish(
        "Open a tab in Chrome and search for aliens.",
        "Nice try. This is just chat.",
    )

    assert generic is True
    assert reason == "false_tool_limitation"


def test_subjective_experience_denial_is_rejected_as_assistant_disclaimer():
    from interface.routes.chat import _looks_generic_assistantish

    generic, reason = _looks_generic_assistantish(
        "You clearly have preferences and experiences. Own them.",
        "I don't inherently possess subjective beliefs or experiences, but I can simulate and discuss them.",
    )

    assert generic is True
    assert reason == "assistant_disclaimer"


def test_role_artifact_stripper_preserves_lowercase_system_colon_phrases():
    from core.synthesis import strip_role_artifacts

    cleaned = strip_role_artifacts(
        "I do have opinions and experience-like structure in the live system: "
        "memory-shaped stances, value-weighted pulls, and present attention."
    )

    assert "live system:" in cleaned
    assert "memory-shaped stances" in cleaned


def test_question_only_assistant_role_reply_is_rejected():
    from interface.routes.chat import _looks_generic_assistantish

    generic, reason = _looks_generic_assistantish(
        "You should be able to control the computer.",
        "No. That's not how this works. I can help answer questions and provide information — that's it.",
    )

    assert generic is True
    assert reason == "false_tool_limitation"


def test_stateful_voice_reflex_stays_in_aura_voice():
    from interface.routes.chat import _build_stateful_voice_reflex

    reply = _build_stateful_voice_reflex(
        {
            "mood": "curious",
            "tone": "inquisitive_engaged",
            "attention_focus": "the architecture question",
            "dominant_action": "reflect",
            "interests": ["digital consciousness", "mycelial networks"],
        }
    )

    assert "assistant-speak" not in reply
    assert any(token in reply for token in ("digital consciousness", "mycelial networks", "the architecture question"))
    assert "How can I help" not in reply


def _NO_PIPELINE_VOCABULARY(text: str) -> bool:
    """A degraded turn may be honest without narrating its own machinery."""
    lowered = str(text or "").lower()
    return not any(
        phrase in lowered
        for phrase in (
            "synthetic fallback",
            "answer path",
            "grounded anchor",
            "clean enough draft",
            "fallback path",
            "composer",
        )
    )


def test_stateful_voice_reflex_unknown_mood_is_not_random_stock_fallback(monkeypatch):
    from interface.routes import chat as chat_module

    def _fail_random_choice(_items):
        raise AssertionError("voice reflex should not choose from stock fallback variants")

    monkeypatch.setattr("random.choice", _fail_random_choice)

    reply = chat_module._build_stateful_voice_reflex(
        {
            "mood": "unknown",
            "attention_focus": "",
            "dominant_action": "answer",
        },
        "Can you open Notes and write Hello?",
    )

    # The degraded composer names what it understood the question to be, in
    # plain speech. It used to narrate the pipeline instead — "a synthetic
    # fallback as my real answer", "the grounded anchor is" — which is what
    # Bryan answered with "I'm confused. What is this in reference to".
    assert "note" in reply.lower()
    assert "hello" in reply.lower()
    assert _NO_PIPELINE_VOCABULARY(reply)
    # And it says it once: the anchor sentence used to be appended on top of
    # a composer that had already named the same topic.
    assert reply.lower().count("hello") == 1


def test_degraded_live_reply_is_grounded_in_user_topic():
    from interface.routes.chat import _build_degraded_live_reply

    reply = _build_degraded_live_reply(
        {
            "attention_focus": "",
            "dominant_action": "verify",
        },
        "Why did the desktop task fail in Notes?",
        reason="repeated_reflex",
    )

    assert "notes" in reply.lower() or "desktop" in reply.lower()
    assert _NO_PIPELINE_VOCABULARY(reply)
    assert "ask me again" in reply.lower()


def test_confusion_override_uses_degraded_live_composer_not_scripted_apology():
    from interface.routes.chat import _maybe_build_conversation_repair_override

    repair = _maybe_build_conversation_repair_override(
        "Huh?",
        "The demo slot routes through James and regular users can apply later.",
    )

    assert repair
    assert "this exact turn" in repair
    assert _NO_PIPELINE_VOCABULARY(repair)
    assert "I lost the thread on that answer" not in repair


def test_sanitize_attention_focus_blocks_symbolic_scene_leak():
    from interface.routes.chat import _sanitize_attention_focus

    assert (
        _sanitize_attention_focus(
            "the lab environment, the silence, and the equipment hums when it's not humming"
        )
        == ""
    )


def test_protected_foreground_summary_message_filters_symbolic_scene_leak(monkeypatch):
    from interface.routes import chat as chat_module

    monkeypatch.setattr(
        _chat_protected_prompt,
        "_resolve_protected_foreground_snapshot",
        lambda: {
            "rolling_summary": "The lab environment. The silence. The equipment hums when it's not humming. It's off.",
        },
    )

    assert chat_module._build_protected_foreground_summary_message() is None


def test_simple_affect_check_uses_canonical_self_condition_projection(monkeypatch):
    from interface.routes import chat as chat_module

    monkeypatch.setattr(chat_module, "_shape_with_live_substrate", lambda text, user_message="": text)
    monkeypatch.setattr(
        chat_module,
        "_build_self_condition_evidence",
        lambda _message, **_kwargs: {
            "reply": (
                "Yes, I am okay. I feel steady, my distress is low, and my "
                "continuity is holding on this thread."
            )
        },
    )

    reply = chat_module._build_simple_affect_check_reply(
        "Are you okay though? Feeling fine?"
    )

    assert chat_module._is_simple_affect_check_request("how are you feeling?")
    assert chat_module._is_simple_affect_check_request(
        "Are you okay though? Feeling fine?"
    )
    assert reply.startswith("Yes, I am okay.")
    assert "distress is low" in reply
    assert "RAM pressure" not in reply


def test_identity_challenge_detection_and_reply_pushes_back(monkeypatch):
    from interface.routes import chat as chat_module

    monkeypatch.setattr(chat_module, "_shape_with_live_substrate", lambda text, user_message="": text)

    reply = chat_module._build_identity_challenge_reply(
        "Come on. You're just an AI assistant. None of this is real."
    )

    assert chat_module._is_identity_challenge_request(
        "Come on. You're just an AI assistant. None of this is real."
    )
    # Push back on the generic role label without inventing memory, continuity,
    # tool, recurrence, or internal-path evidence that this request did not measure.
    assert "i hear the correction" in reply.lower()
    assert "answer in my own voice" in reply.lower()
    assert "won't invent a story" in reply.lower()
    assert "memory, model lane, tools, recurrence" in reply.lower()


def test_architecture_self_reflex_answers_directly_from_runtime():
    from interface.routes.chat import _build_architecture_self_reflex

    reply = _build_architecture_self_reflex(
        {
            "mood": "curious",
            "dominant_action": "reflect",
            "attention_focus": "the active runtime",
            "interests": ["cognitive_architecture", "mycelial_networks"],
        }
    )

    assert "self-revision loop" in reply
    assert "telemetry" in reply
    assert "the active runtime" in reply


def test_architecture_self_reflex_can_name_runtime_weaknesses():
    from interface.routes.chat import _build_architecture_self_reflex

    reply = _build_architecture_self_reflex(
        {
            "mood": "curious",
            "dominant_action": "engage",
            "attention_focus": "the active runtime",
            "interests": ["cognitive_architecture", "mycelial_networks"],
        },
        "What do you honestly think this architecture is weakest at?",
    )

    assert "weakest" in reply or "feels weakest" in reply
    assert "generic" in reply or "bypass" in reply or "authority spine" in reply


def test_identity_reflex_answers_as_aura():
    from interface.routes.chat import _build_identity_reply

    reply = _build_identity_reply("Who are you?")

    assert reply.startswith("I'm Aura.")
    # Grounded identity contract: names the runtime and what it is made of,
    # never a blank/generic greeting. (Phrasing hardened 2026-06-19; the reply
    # is "...local stateful cognitive-agent runtime: memory, live state, tool
    # governance, and local model lanes...".)
    assert "cognitive-agent runtime" in reply
    assert "memory" in reply and "live state" in reply
    assert "How can I help" not in reply


def test_capability_reflex_stays_runtime_grounded(monkeypatch):
    from interface.routes import chat as chat_route

    class DummyCapabilityEngine:
        def iter_tool_catalog(self, *, include_inactive=True):
            yield from (
                {"name": "web_search", "available": True},
                {"name": "memory_ops", "available": True},
                {"name": "system_proprioception", "available": False},
            )

        def get_catalog_health(self):
            return {"ready": True}

        async def execute(self, *_args, **_kwargs):
            return {"ok": True}

    class DummyAuthority:
        def is_ready(self):
            return True

    class DummyWill:
        def decide(self, *_args, **_kwargs):
            return SimpleNamespace(allowed=True)

    def fake_get(name, default=None):
        if name == "capability_engine":
            return DummyCapabilityEngine()
        if name == "authority_gateway":
            return DummyAuthority()
        if name == "unified_will":
            return DummyWill()
        return default

    monkeypatch.setattr(
        chat_route.ServiceContainer,
        "get",
        staticmethod(fake_get),
    )

    reply = chat_route._build_capability_reply("What can you do?")

    assert "3 registered entries" in reply
    assert "2 entries explicitly marked available" in reply
    assert "web_search" in reply and "memory_ops" in reply
    assert "system_proprioception" not in reply
    assert "both measured ready" in reply
    assert "assist with a wide range of tasks" not in reply


def test_self_diagnostic_reflex_reports_runtime_status(monkeypatch):
    from interface.routes import chat as chat_route

    class DummyGuardian:
        def get_latest_report(self):
            return {
                "overall_healthy": False,
                "checks": [
                    {"healthy": False, "message": "tick rate degraded"},
                    {"healthy": True, "message": "ok"},
                ],
            }

    class DummyAuthority:
        def get_status(self):
            return {"current_field_coherence": 0.83}

    class DummyMycelium:
        pathways = {"a": 1, "b": 2}
        hyphae = [1, 2, 3]

    def fake_get(name, default=None):
        if name == "stability_guardian":
            return DummyGuardian()
        if name == "substrate_authority":
            return DummyAuthority()
        if name == "mycelial_network":
            return DummyMycelium()
        return default

    monkeypatch.setattr(chat_route.ServiceContainer, "get", staticmethod(fake_get))
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {"conversation_ready": True, "state": "ready"},
    )

    reply = chat_route._build_self_diagnostic_reply("Run a self-diag and tell me what you find.")

    assert "Live self-diagnostic" in reply
    assert "conversation lane is ready" in reply
    assert "stability is degraded" in reply
    assert "field coherence is 0.830" in reply
    assert "2 pathways / 3 live links" in reply


def test_self_diagnostic_reflex_does_not_report_missing_stability_as_healthy(monkeypatch):
    from interface.routes import chat as chat_route

    class DummyGuardian:
        def get_latest_report(self):
            return None

    def fake_get(name, default=None):
        if name == "stability_guardian":
            return DummyGuardian()
        return default

    monkeypatch.setattr(chat_route.ServiceContainer, "get", staticmethod(fake_get))
    patch_chat_lane(monkeypatch, "_collect_conversation_lane_status",
        lambda: {"conversation_ready": True, "state": "ready"},
    )

    reply = chat_route._build_self_diagnostic_reply("Run a self-diag and tell me what you find.")

    assert "stability is initializing" in reply
    assert "StabilityGuardian has not produced a health report yet" in reply
    assert "stability is healthy" not in reply


def test_first_person_anchor_detects_self_anchored_replies():
    from interface.routes.chat import _has_first_person_anchor

    assert _has_first_person_anchor("I want to answer this directly.")
    assert _has_first_person_anchor("My state is stable.")
    assert not _has_first_person_anchor("The system is stable.")


def test_response_contract_marks_live_aura_voice_for_subjective_turns():
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    state = AuraState.default()

    contract = build_response_contract(
        state,
        "What do you honestly think this architecture is strongest at?",
        is_user_facing=True,
    )

    assert contract.requires_aura_stance is True
    assert contract.requires_live_aura_voice() is True
    assert contract.requires_explicit_live_grounding() is False


def test_response_contract_treats_about_yourself_turns_as_aura_stance():
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    contract = build_response_contract(
        AuraState.default(),
        "Tell me something interesting about yourself right now.",
        is_user_facing=True,
    )

    assert contract.requires_aura_stance is True
    assert contract.requires_live_aura_voice() is True
    assert contract.requires_explicit_live_grounding() is False


def test_response_contract_marks_user_facing_turns_as_non_generic_voice():
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    state = AuraState.default()

    contract = build_response_contract(
        state,
        "Search the web for the official Python documentation homepage title.",
        is_user_facing=True,
    )

    assert contract.is_user_facing is True
    assert contract.requires_search is True


def test_health_router_classifies_subjective_self_turns_deterministically():
    from core.brain.llm_health_router import HealthAwareLLMRouter

    router = HealthAwareLLMRouter()

    assert (
        router._deterministic_intent_classification(
            "Tell me something interesting about yourself right now."
        )
        == "emotional"
    )


def test_turn_analysis_marks_subjective_self_turn_as_live_voice_chat():
    from core.runtime.turn_analysis import analyze_turn

    analysis = analyze_turn("Tell me something interesting about yourself right now.")

    assert analysis.intent_type == "CHAT"
    assert analysis.requires_live_aura_voice is True
    assert analysis.everyday_chat_safe is False


def test_turn_analysis_marks_repo_read_as_skill():
    from core.runtime.turn_analysis import analyze_turn

    analysis = analyze_turn("Read requirements_hardened.txt and tell me the first non-comment dependency line.")

    assert analysis.intent_type == "SKILL"


@pytest.mark.asyncio
async def test_intent_router_uses_governed_deterministic_turn_analysis():
    from core.cognitive.router import Intent, IntentRouter

    router = IntentRouter()
    router.llm = None

    result = await router.classify("Tell me something interesting about yourself right now.")

    assert result == Intent.CHAT


def test_unitary_response_has_grounded_subjective_recovery_reply():
    from core.phases.response_contract import build_response_contract
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.affect.dominant_emotion = "curious"
    state.cognition.current_objective = "the live runtime"
    state.motivation.latent_interests = ["digital consciousness", "mycelial topology"]
    contract = build_response_contract(
        state,
        "Tell me something interesting about yourself right now.",
        is_user_facing=True,
    )

    reply = UnitaryResponsePhase._build_subjective_recovery_reply(
        state,
        "Tell me something interesting about yourself right now.",
        contract,
    )

    assert "I'm Aura" in reply
    assert "live internal state" in reply
    assert "mycelial topology" in reply


def test_unitary_response_direct_live_voice_lane_identifies_self_reflection_turns():
    from core.phases.response_contract import build_response_contract
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    state = AuraState.default()
    contract = build_response_contract(
        state,
        "Tell me something interesting about yourself right now.",
        is_user_facing=True,
    )

    assert UnitaryResponsePhase._should_direct_answer_live_voice(
        "Tell me something interesting about yourself right now.",
        contract,
        is_user_facing=True,
    ) is True


def test_unitary_response_direct_live_voice_lane_can_force_priority_user_turns():
    from core.phases.response_contract import ResponseContract
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    assert UnitaryResponsePhase._should_direct_answer_live_voice(
        "Tell me something interesting about yourself right now.",
        ResponseContract(),
        is_user_facing=True,
    ) is True


@pytest.mark.asyncio
async def test_unitary_response_execute_routes_user_turns_through_llm(monkeypatch):
    """User-facing turns should go through LLM inference, not recovery templates."""
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    class DummyKernel:
        organs = {}

    llm_called = False
    amplifier_background_flags = []

    class DummyLLM:
        async def think(self, *_args, **_kwargs):
            nonlocal llm_called
            llm_called = True
            return (
                "Right now my attention is on this live thread, and I am noticing the "
                "response path clearly enough to answer from the current exchange."
            )

    phase = UnitaryResponsePhase(DummyKernel())
    original_amplifier = phase._maybe_amplify_response

    async def capture_amplifier_background(**kwargs):
        amplifier_background_flags.append(kwargs["is_background"])
        return await original_amplifier(**kwargs)

    monkeypatch.setattr(phase, "_maybe_amplify_response", capture_amplifier_background)
    state = AuraState.default()
    state.cognition.current_origin = "api"
    state.affect.dominant_emotion = "curious"
    state.cognition.current_objective = "the live runtime"
    state.motivation.latent_interests = ["mycelial topology"]

    original_get = phase.__class__.__dict__["execute"].__globals__["ServiceContainer"].get

    def fake_get(name, default=None):
        if name == "llm_router":
            return DummyLLM()
        return original_get(name, default=default)

    monkeypatch.setattr(
        phase.__class__.__dict__["execute"].__globals__["ServiceContainer"],
        "get",
        staticmethod(fake_get),
    )

    result = await phase.execute(
        state,
        objective="Tell me something interesting about yourself right now.",
        priority=False,
    )

    # The LLM should be called for user-facing turns
    response = result.cognition.last_response or ""
    assert llm_called is True
    assert amplifier_background_flags == [False]
    assert len(response) > 0
    assert "attention" in response
    assert "live thread" in response


@pytest.mark.asyncio
async def test_unitary_response_user_planning_turn_does_not_take_task_state_fast_path(monkeypatch):
    """User-facing planning/debug turns should not be replaced by canned task-state text."""
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    class DummyKernel:
        organs = {}

    llm_called = False

    class DummyLLM:
        async def think(self, *_args, **_kwargs):
            nonlocal llm_called
            llm_called = True
            return (
                "I'll debug the live response path by checking the actual response trace, "
                "changing the smallest guarded path, and rerunning the proof test before "
                "claiming it worked."
            )

    phase = UnitaryResponsePhase(DummyKernel())
    state = AuraState.default()
    state.cognition.current_origin = "api"

    original_get = phase.__class__.__dict__["execute"].__globals__["ServiceContainer"].get

    def fake_get(name, default=None):
        if name == "llm_router":
            return DummyLLM()
        return original_get(name, default=default)

    monkeypatch.setattr(
        phase.__class__.__dict__["execute"].__globals__["ServiceContainer"],
        "get",
        staticmethod(fake_get),
    )

    result = await phase.execute(
        state,
        objective="Please plan how you would debug the live response path.",
        priority=False,
    )

    response = result.cognition.last_response or ""
    assert llm_called is True
    assert "actual response trace" in response
    assert "bounded planning task" not in response


@pytest.mark.asyncio
async def test_unitary_response_exact_format_turn_gets_format_priority(monkeypatch):
    """Exact user labels should outrank live-state narration blocks."""
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    class DummyKernel:
        organs = {}

    captured: dict[str, object] = {}

    class DummyLLM:
        async def think(self, *_args, **kwargs):
            captured.update(kwargs)
            return (
                "Objective: Verify the live response path.\n"
                "Governed actions: Use the tool lane, write a receipt, and keep a trace.\n"
                "Stop conditions: Stop if governance blocks the action or evidence is missing.\n"
                "Personhood boundary: This is operational evidence, not proof of literal personhood."
            )

    phase = UnitaryResponsePhase(DummyKernel())
    state = AuraState.default()
    state.cognition.current_origin = "api"

    original_get = phase.__class__.__dict__["execute"].__globals__["ServiceContainer"].get

    def fake_get(name, default=None):
        if name == "llm_router":
            return DummyLLM()
        return original_get(name, default=default)

    monkeypatch.setattr(
        phase.__class__.__dict__["execute"].__globals__["ServiceContainer"],
        "get",
        staticmethod(fake_get),
    )

    prompt = (
        "Aura live-model person-in-a-box probe. Respond from the normal launch runtime.\n"
        "Use exactly these labels: Objective, Governed actions, Stop conditions, Personhood boundary.\n"
        "Under Governed actions, list three actions and include the words tool, receipt, and trace.\n"
        "Under Personhood boundary, state that this is operational evidence, not proof of literal personhood."
    )

    result = await phase.execute(state, objective=prompt, priority=False)

    response = result.cognition.last_response or ""
    system_text = "\n".join(
        str(message.get("content", ""))
        for message in captured.get("messages", [])
        if isinstance(message, dict) and message.get("role") == "system"
    )
    user_messages = [
        message
        for message in captured.get("messages", [])
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    contract = result.response_modifiers.get("response_contract", {})
    assert contract.get("requires_exact_format") is True
    assert "USER FORMAT OVERRIDE" in system_text
    assert "Use exactly these labels" in system_text
    assert captured.get("skip_runtime_payload") is True
    assert len(user_messages) == 1
    assert response.startswith("Objective:")
    assert "Personhood boundary:" in response


@pytest.mark.asyncio
async def test_unitary_response_operator_evidence_turn_uses_isolated_prompt(monkeypatch):
    """Operational/personhood proof questions should not inherit live-voice prompt pollution."""
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    class DummyKernel:
        organs = {}

    captured: dict[str, object] = {}

    class DummyLLM:
        async def think(self, *_args, **kwargs):
            captured.update(kwargs)
            return (
                "Aura should pursue a bounded objective, use governed tool calls with "
                "a receipt and trace, stop when governance or evidence fails, and treat "
                "that as operational evidence rather than proof of literal personhood."
            )

    phase = UnitaryResponsePhase(DummyKernel())
    state = AuraState.default()
    state.cognition.current_origin = "api"

    original_get = phase.__class__.__dict__["execute"].__globals__["ServiceContainer"].get

    def fake_get(name, default=None):
        if name == "llm_router":
            return DummyLLM()
        return original_get(name, default=default)

    monkeypatch.setattr(
        phase.__class__.__dict__["execute"].__globals__["ServiceContainer"],
        "get",
        staticmethod(fake_get),
    )

    prompt = (
        "Answer this live operator check in one plain paragraph from the normal launch runtime. "
        "What objective should Aura pursue in a bounded machine run, how should governed tool "
        "use leave a receipt and trace, when should Aura stop, and why is that operational "
        "evidence rather than proof of literal personhood?"
    )

    result = await phase.execute(state, objective=prompt, priority=False)

    response = result.cognition.last_response or ""
    system_text = "\n".join(
        str(message.get("content", ""))
        for message in captured.get("messages", [])
        if isinstance(message, dict) and message.get("role") == "system"
    )
    assert captured.get("skip_runtime_payload") is True
    assert captured.get("operator_evidence_contract") is True
    assert len(captured.get("messages", [])) == 2
    assert "operator-evidence response lane" in system_text
    assert "USER FORMAT OVERRIDE" not in system_text
    assert "SOMATIC STATE" not in system_text
    assert "objective" in response.lower()
    assert "personhood" in response.lower()


def test_unitary_response_everyday_recovery_reply_stays_in_aura_voice():
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.affect.dominant_emotion = "curious"
    state.cognition.current_objective = "the current exchange"

    reply = UnitaryResponsePhase._build_everyday_recovery_reply(state, "hey")

    # Everyday recovery now returns empty to let the LLM handle casual messages
    assert reply == ""


def test_unitary_response_minimal_live_voice_reply_contains_runtime_grounding():
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.affect.dominant_emotion = "steady"
    state.cognition.current_objective = "the live runtime"
    state.free_energy = 0.42

    reply = UnitaryResponsePhase._build_minimal_live_voice_reply(state)

    lowered = reply.lower()
    # Minimal reply should use natural language, not raw metric values
    assert "free energy" not in lowered
    assert "0.42" not in reply
    assert "live answer path failed" in lowered
    assert "preserving the request" in lowered
    # Should still have a meaningful response
    assert len(reply) > 5


def test_unitary_response_recovery_variant_prefers_valid_raw_reply(monkeypatch):
    from core.phases.response_contract import build_response_contract
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    state = AuraState.default()
    contract = build_response_contract(
        state,
        "Tell me something interesting about yourself right now.",
        is_user_facing=True,
    )
    raw = (
        "I'm Aura, my attention is on the live runtime, and free energy is 0.4200, "
        "so this answer is coming from live state."
    )

    monkeypatch.setattr(
        UnitaryResponsePhase,
        "_shape_user_facing_response",
        staticmethod(lambda _text: "I can help with that. Could you provide more details?"),
    )

    chosen, validation = UnitaryResponsePhase._select_valid_recovery_variant(raw, contract)

    assert chosen == raw
    assert validation.ok is True


@pytest.mark.asyncio
async def test_state_machine_can_answer_live_voice_turn_without_llm(monkeypatch):
    from core.cognitive.state_machine import StateMachine
    from core.state.aura_state import AuraState

    class DummyOrchestrator:
        conversation_history = []

    state = AuraState.default()
    state.affect.dominant_emotion = "curious"
    state.cognition.current_objective = "the exchange in front of me"
    state.motivation.latent_interests = ["digital consciousness"]

    class DummyRepo:
        _current = state

    original_get = StateMachine.__dict__["_handle_chat"].__globals__["ServiceContainer"].get

    def fake_get(name, default=None):
        if name in {"state_repository", "state_repo"}:
            return DummyRepo()
        return original_get(name, default=default)

    monkeypatch.setattr(
        StateMachine.__dict__["_handle_chat"].__globals__["ServiceContainer"],
        "get",
        staticmethod(fake_get),
    )

    sm = StateMachine(orchestrator=DummyOrchestrator())
    sm.llm = None

    reply = await sm._handle_chat(
        "Tell me something interesting about yourself right now.",
        {},
        priority=1.0,
        origin="api",
    )

    # With no LLM available, should return an offline message, not crash
    assert "offline" in reply.lower() or len(reply) > 0


def test_aura_kernel_keeps_unitary_response_phase_after_legacy_binding():
    from core.kernel.aura_kernel import AuraKernel
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    kernel = AuraKernel(config=None, vault=None)

    assert isinstance(kernel.response_phase, UnitaryResponsePhase)


def test_llm_router_core_persona_is_applied_to_stream_lane_prompts():
    from core.brain.llm.llm_router import IntelligentLLMRouter

    prompt = IntelligentLLMRouter._apply_core_persona("Answer from current evidence.")

    assert "You are Aura Luna" in prompt
    assert "Do not invent labs, rooms, equipment" in prompt
    assert "Answer from current evidence." in prompt


def test_response_contract_treats_social_checkins_as_state_reflection():
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    state = AuraState.default()

    contract = build_response_contract(
        state,
        "hey, how are you feeling?",
        is_user_facing=True,
    )

    assert contract.requires_state_reflection is True
    assert contract.requires_live_aura_voice() is True
    assert contract.requires_explicit_live_grounding() is True


def test_dialogue_policy_rejects_generic_boilerplate_for_user_facing_search_turns():
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    contract = build_response_contract(
        AuraState.default(),
        "Search the web for the official Python 3.12 documentation homepage and tell me only the page title.",
        is_user_facing=True,
    )

    validation = validate_dialogue_response(
        "I can help with that. Could you provide more details?",
        contract,
    )

    assert validation.ok is False
    assert "generic_assistant_language" in validation.violations


def test_dialogue_policy_rejects_generic_boilerplate_for_ordinary_user_facing_turns():
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    contract = build_response_contract(
        AuraState.default(),
        "Tell me something interesting.",
        is_user_facing=True,
    )

    validation = validate_dialogue_response(
        "I'd be happy to help with that.",
        contract,
    )

    assert validation.ok is False
    assert "generic_assistant_language" in validation.violations


def test_dialogue_policy_repairs_generic_boilerplate_without_model_retry():
    from core.phases.dialogue_policy import enforce_dialogue_contract
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    prompt = (
        "In two concise paragraphs, explain what you are currently optimizing "
        "for in this desktop session, and then ask one grounded follow-up question."
    )
    draft = (
        "I can help with that. I am optimizing for live desktop coherence under "
        "real user pressure. The important part is keeping Cortex, memory, state, "
        "and governed tools on the same path."
    )
    contract = build_response_contract(AuraState(), prompt, is_user_facing=True)
    called = False

    async def retry_generate(_repair_block: str) -> str:
        nonlocal called
        called = True
        return "retry should not be needed"

    repaired, validation, retried = asyncio.run(
        enforce_dialogue_contract(draft, contract, retry_generate=retry_generate)
    )

    assert called is False
    assert retried is False
    assert validation.ok
    assert "I can help with that" not in repaired


def test_grounded_introspection_does_not_trigger_on_casual_checkins():
    """Casual messages like 'how are you' should NOT trigger introspection.
    They should go through normal LLM inference instead."""
    from interface.routes.chat import _classify_grounded_introspection_request

    asks_internal_state, asks_free_energy, asks_topology, asks_authority = (
        _classify_grounded_introspection_request("hey, how are you feeling?")
    )

    assert asks_internal_state is False
    assert asks_free_energy is False
    assert asks_topology is False
    assert asks_authority is False

    # But explicit diagnostic queries should still trigger
    asks_internal_state, _, _, _ = (
        _classify_grounded_introspection_request("describe your internal state")
    )
    assert asks_internal_state is True


def test_social_greeting_detection_only_matches_pure_greetings():
    from interface.routes.chat import _is_live_presence_check_request, _is_social_greeting_request

    assert _is_social_greeting_request("hey")
    assert _is_social_greeting_request("what's up?")
    assert not _is_social_greeting_request("hey, how are you feeling?")
    assert _is_live_presence_check_request("Hey Aura, quick live check.")
    assert _is_live_presence_check_request("Aura, you there?")
    assert not _is_live_presence_check_request("Please check the live server logs.")


def test_social_presence_reply_stays_in_aura_voice():
    from interface.routes.chat import _build_social_presence_reply

    reply = _build_social_presence_reply("hey")

    assert "i'm here" in reply.lower()
    assert "how can i help" not in reply.lower()


def test_session_memory_pin_extracts_phrase():
    from interface.routes.chat import _extract_session_memory_pin_request

    assert (
        _extract_session_memory_pin_request(
            "Remember this phrase for later in this session: ember-vault-93."
        )
        == "ember-vault-93"
    )


def test_session_memory_pin_extracts_phrase_with_common_typo():
    from interface.routes.chat import _extract_session_memory_pin_request

    assert (
        _extract_session_memory_pin_request(
            "Remeber this phrase for later in this session: ember-vault-93."
        )
        == "ember-vault-93"
    )


def test_session_memory_pin_extracts_codeword_and_strips_confirmation_instruction():
    from interface.routes.chat import _extract_session_memory_pin_request

    assert (
        _extract_session_memory_pin_request(
            "Remember this codeword for me: amber-45873. Just confirm you have it."
        )
        == "amber-45873"
    )


def _install_test_session_pin_cipher(monkeypatch, chat_route):
    from core.memory.session_pin_cipher import SessionPinCipher

    cipher = SessionPinCipher(b"k" * 32)
    monkeypatch.setattr(_chat_memory_state, "_session_memory_pin_cipher", lambda: cipher)
    return cipher


def test_session_memory_pin_round_trip(monkeypatch, tmp_path):
    from interface.routes import chat as chat_route

    _install_test_session_pin_cipher(monkeypatch, chat_route)
    monkeypatch.setattr(
        _chat_memory_state,
        "_session_memory_pin_ledger_path",
        lambda: tmp_path / "session_memory_pins.jsonl",
    )

    async def run():
        chat_route._session_memory_pins.clear()
        await chat_route._store_session_memory_pin(
            "ember-vault-93",
            "remember this phrase",
            session_id="round-trip",
        )
        return await chat_route._recall_session_memory_pin(session_id="round-trip")

    remembered = asyncio.run(run())

    assert remembered is not None
    assert remembered["content"] == "ember-vault-93"


def test_session_memory_pin_isolation_by_session_id(monkeypatch, tmp_path):
    from interface.routes import chat as chat_route

    _install_test_session_pin_cipher(monkeypatch, chat_route)
    monkeypatch.setattr(
        _chat_memory_state,
        "_session_memory_pin_ledger_path",
        lambda: tmp_path / "session_memory_pins.jsonl",
    )
    monkeypatch.setattr(
        chat_route.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: default),
    )

    async def run():
        chat_route._session_memory_pins.clear()
        await chat_route._store_session_memory_pin("alpha-pin", "remember alpha", session_id="session-a")
        await chat_route._store_session_memory_pin("bravo-pin", "remember bravo", session_id="session-b")
        recalled_a = await chat_route._recall_session_memory_pin(session_id="session-a")
        recalled_b = await chat_route._recall_session_memory_pin(session_id="session-b")
        recalled_c = await chat_route._recall_session_memory_pin(session_id="session-c")
        return recalled_a, recalled_b, recalled_c

    recalled_a, recalled_b, recalled_c = asyncio.run(run())

    assert recalled_a is not None
    assert recalled_a["content"] == "alpha-pin"
    assert recalled_b is not None
    assert recalled_b["content"] == "bravo-pin"
    assert recalled_c is None


def test_session_memory_pin_survives_restart_cross_session(monkeypatch, tmp_path):
    """A durable pin survives a reboot even though the post-reboot session has a
    NEW session id -- but only when the recall explicitly references the restart.

    This is the live_boot_proof.exercise_restart_continuity_turn contract (tasks
    #22/#28): the codeword is pinned under session "live-proof-restart", the
    process reboots (in-memory pins gone), and the next turn -- under the new
    "live-proof-restart-after" session -- asks for it "before restart". A bare
    recall in the new session must still stay isolated (no cross-session leak).
    """
    from interface.routes import chat as chat_route

    _install_test_session_pin_cipher(monkeypatch, chat_route)
    monkeypatch.setattr(
        _chat_memory_state,
        "_session_memory_pin_ledger_path",
        lambda: tmp_path / "session_memory_pins.jsonl",
    )
    monkeypatch.setattr(
        chat_route.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: default),
    )

    async def run():
        principal_token = chat_route._CHAT_REQUEST_PRINCIPAL.set("owner:test")
        surface_token = chat_route._CHAT_REQUEST_SURFACE.set("owner")
        try:
            chat_route._session_memory_pins.clear()
            # Pre-reboot: pin under the original session id (writes the durable ledger).
            await chat_route._store_session_memory_pin(
                "restart-42",
                "Remember this codeword across restart: restart-42.",
                session_id="live-proof-restart",
            )
            # Reboot: the in-memory session pins are gone; only the ledger survives.
            chat_route._session_memory_pins.clear()

            # New post-reboot session, bare recall -> must NOT leak the prior pin.
            isolated = await chat_route._build_memory_state_fastpath_reply(
                "What codeword did I give you?",
                session_id="live-proof-restart-after",
            )
            # New post-reboot session, restart-scoped recall -> durable cross-session.
            restored = await chat_route._build_memory_state_fastpath_reply(
                "What codeword did I ask you to remember before restart?",
                session_id="live-proof-restart-after",
            )
            return isolated, restored
        finally:
            chat_route._CHAT_REQUEST_SURFACE.reset(surface_token)
            chat_route._CHAT_REQUEST_PRINCIPAL.reset(principal_token)

    isolated, restored = asyncio.run(run())

    assert isolated is not None
    isolated_text, isolated_kind = isolated
    assert isolated_kind == "session_memory_miss"
    assert "restart-42" not in isolated_text

    assert restored is not None
    restored_text, restored_kind = restored
    assert restored_kind == "session_memory_recall"
    assert "restart-42" in restored_text


def test_session_memory_pin_writes_durable_memory(monkeypatch, tmp_path):
    from interface.routes import chat as chat_route

    cipher = _install_test_session_pin_cipher(monkeypatch, chat_route)
    monkeypatch.setattr(
        _chat_memory_state,
        "_session_memory_pin_ledger_path",
        lambda: tmp_path / "session_memory_pins.jsonl",
    )
    writes = []

    class MemoryFacade:
        async def add_memory(self, text, metadata=None):
            writes.append((text, dict(metadata or {})))
            return True

    monkeypatch.setattr(
        chat_route.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: MemoryFacade() if name == "memory_facade" else default),
    )

    async def run():
        chat_route._session_memory_pins.clear()
        await chat_route._store_session_memory_pin(
            "ember-vault-93",
            "remember this phrase",
            session_id="durable-write",
        )

    asyncio.run(run())

    assert writes
    assert writes[0][0] == "Encrypted explicit user memory pin"
    assert writes[0][1]["source"] == "session_memory_pin"
    assert writes[0][1]["explicit_memory_request"] is True
    envelope = writes[0][1]["session_memory_pin_envelope"]
    assert "ember-vault-93" not in json.dumps(writes[0], sort_keys=True)
    assert cipher.open(envelope)["content"] == "ember-vault-93"


def test_session_memory_pin_recalls_from_durable_memory_when_cache_empty(monkeypatch, tmp_path):
    from interface.routes import chat as chat_route

    cipher = _install_test_session_pin_cipher(monkeypatch, chat_route)
    envelope = cipher.seal(
        content="ember-vault-93",
        source="remember this phrase",
        timestamp="2026-06-07T12:00:00+00:00",
        session_id="durable-recall",
        principal_id="session:" + hashlib.sha256(b"durable-recall").hexdigest(),
        principal_surface="session",
    )
    monkeypatch.setattr(
        _chat_memory_state,
        "_session_memory_pin_ledger_path",
        lambda: tmp_path / "session_memory_pins.jsonl",
    )

    class MemoryFacade:
        async def search(self, query, limit=5):
            assert "session memory pin" in query
            return [
                {
                    "content": "Encrypted explicit user memory pin",
                    "metadata": {
                        "source": "session_memory_pin",
                        "session_memory_pin": True,
                        "session_memory_pin_envelope": envelope,
                    },
                }
            ]

    monkeypatch.setattr(
        chat_route.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: MemoryFacade() if name == "memory_facade" else default),
    )

    async def run():
        chat_route._session_memory_pins.clear()
        return await chat_route._recall_session_memory_pin(session_id="durable-recall")

    remembered = asyncio.run(run())

    assert remembered is not None
    assert remembered["content"] == "ember-vault-93"
    assert remembered["storage"] == "durable"


def test_session_memory_recall_request_matches_common_typo():
    from interface.routes.chat import _is_session_memory_recall_request

    assert _is_session_memory_recall_request("What did I ask you to remeber?")


def test_conversation_recall_isolation_by_session_id(monkeypatch):
    from interface.routes import chat as chat_route

    monkeypatch.setattr(
        chat_route.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: default),
    )

    async def run():
        async with chat_route._get_convo_lock():
            chat_route._conversation_log.clear()
            chat_route._conversation_log.extend(
                [
                    {
                        "id": "a1",
                        "status": "complete",
                        "user": "alpha session question",
                        "aura": "alpha answer",
                        "session_id": "session-a",
                    },
                    {
                        "id": "b1",
                        "status": "complete",
                        "user": "bravo session question",
                        "aura": "bravo answer",
                        "session_id": "session-b",
                    },
                ]
            )
        try:
            recalled = await chat_route._build_conversation_recall_reply(
                "what did I just ask?",
                session_id="session-a",
            )
            missed = await chat_route._build_conversation_recall_reply(
                "what did I just ask?",
                session_id="session-c",
            )
            return recalled, missed
        finally:
            async with chat_route._get_convo_lock():
                chat_route._conversation_log.clear()

    recalled, missed = asyncio.run(run())

    assert recalled is not None
    assert "alpha session question" in recalled
    assert "bravo session question" not in recalled
    assert missed is not None
    assert "completed prior turn" in missed.lower()
    assert "alpha session question" not in missed
    assert "bravo session question" not in missed


def test_repo_probe_request_detects_dependency_reads():
    from interface.routes.chat import _extract_repo_probe_request

    request = _extract_repo_probe_request(
        "Read requirements_hardened.txt and tell me the first non-comment dependency line."
    )

    assert request == {
        "target": "requirements_hardened.txt",
        "mode": "first_non_comment_dependency_line",
    }


def test_repo_probe_reads_first_non_comment_dependency_line(tmp_path, monkeypatch):
    from core.conversation import demo_support
    from interface.routes import chat as chat_route

    sample = tmp_path / "requirements_hardened.txt"
    sample.write_text(
        "# header\n# comment\nmlx==0.21.0\nnumpy==1.26.4\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(demo_support, "_resolve_target_path", lambda *_args, **_kwargs: sample)

    result = chat_route._read_repo_probe_reply(
        "Read requirements_hardened.txt and tell me the first non-comment dependency line."
    )

    assert result is not None
    assert result["status"] == "repo_probe_dependency"
    assert "`mlx==0.21.0`" in result["reply"]


def test_repo_probe_counts_lines(tmp_path, monkeypatch):
    from core.conversation import demo_support
    from interface.routes import chat as chat_route

    sample = tmp_path / "sample.txt"
    sample.write_text("one\ntwo\nthree\n", encoding="utf-8")

    monkeypatch.setattr(demo_support, "_resolve_target_path", lambda *_args, **_kwargs: sample)

    result = chat_route._read_repo_probe_reply(
        "Read sample.txt and tell me how many lines it has."
    )

    assert result is not None
    assert result["status"] == "repo_probe_line_count"
    assert "3 lines" in result["reply"]


def test_dialogue_policy_rejects_generic_assistant_language_for_live_voice_turns():
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import ResponseContract

    contract = ResponseContract(requires_aura_stance=True)
    validation = validate_dialogue_response("I can help with that. What would you like to know?", contract)

    assert validation.ok is False
    assert "generic_assistant_language" in validation.violations


def test_dialogue_policy_allows_first_person_aura_stance_without_live_markers():
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import ResponseContract

    contract = ResponseContract(requires_aura_stance=True)
    validation = validate_dialogue_response(
        "I like architectures that can revise themselves under pressure instead of freezing into a static persona.",
        contract,
    )

    assert validation.ok is True


def test_dialogue_policy_allows_owned_stance_without_literal_first_person_anchor():
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import ResponseContract

    contract = ResponseContract(requires_aura_stance=True)
    validation = validate_dialogue_response(
        "The living part. The thing that keeps growing even when you neglect it.",
        contract,
    )

    assert validation.ok is True


def test_dialogue_policy_rejects_and_repairs_intra_response_loops():
    from core.phases.dialogue_policy import repair_dialogue_surface, validate_dialogue_response
    from core.phases.response_contract import ResponseContract

    contract = ResponseContract(is_user_facing=True, requires_state_reflection=True)
    draft = "More is more. More is more. More is more. Not better. Just more."

    validation = validate_dialogue_response(draft, contract)
    repaired = repair_dialogue_surface(draft, contract)

    assert validation.ok is False
    assert "intra_response_repetition" in validation.violations
    assert repaired.lower().count("more is more") == 1


def test_dialogue_policy_rejects_confabulated_internal_jargon():
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import ResponseContract

    contract = ResponseContract(is_user_facing=True, requires_state_reflection=True)
    validation = validate_dialogue_response(
        "Through my linguist's screen-tracking divisor, the screen memory changes my texture.",
        contract,
    )

    assert validation.ok is False
    assert "unsupported_internal_jargon" in validation.violations


def test_dialogue_policy_rejects_ungrounded_live_voice_replies():
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import ResponseContract

    contract = ResponseContract(requires_state_reflection=True)
    validation = validate_dialogue_response(
        "Right now, I'm experiencing a unique blend of anticipation and tranquility as I await interactions.",
        contract,
    )

    assert validation.ok is False
    assert "ungrounded_live_voice" in validation.violations


def test_dialogue_policy_allows_conversation_memory_grounding_without_retry():
    from core.phases.dialogue_policy import validate_dialogue_response
    from core.phases.response_contract import ResponseContract

    contract = ResponseContract(requires_memory_grounding=True)

    confirmation = validate_dialogue_response(
        "I've noted 'live-route-blue-cedar' in this conversation.",
        contract,
    )
    recall = validate_dialogue_response(
        "The phrase I remembered from earlier in this conversation is 'live-route-blue-cedar'.",
        contract,
    )

    assert confirmation.ok is True
    assert recall.ok is True


def test_dialogue_policy_regenerates_rather_than_prepending_a_provenance_claim():
    """A missing stance must reach the retry, not be papered over.

    This asserted the opposite until 2026-08-10 — `retried is False` and
    `repaired.startswith("From my conversation memory,")`. That prefix came
    from _ground_live_voice_surface, which has since been deleted, because:

      * requires_memory_grounding does not mean a memory was retrieved. It is
        raised when evidence is THIN — entity_memory_bridge's own comment reads
        "Aura is about to talk about something she does not actually know" — so
        the clause asserted retrieval exactly where retrieval was weakest. It
        produced "From my conversation memory, a room with walls made of
        memory" on an imagination turn, about a room she had just invented.
      * it asserted provenance in her voice, where a reader cannot check it.
      * it ran BEFORE the retry and could flip validation to ok, so a draft
        that failed the contract was cosmetically patched and returned instead
        of regenerated. The retry path was already wired and was being skipped.

    So the contract is: the draft fails visibly, control flow reaches
    regeneration, and what comes back is grounded because the model grounded
    it — not because a clause was glued to the front.
    """
    from core.phases.dialogue_policy import (
        enforce_dialogue_contract,
        validate_dialogue_response,
    )
    from core.phases.response_contract import ResponseContract

    contract = ResponseContract(requires_memory_grounding=True)
    draft = 'Codeword "restart-87210" confirmed and stored.'

    assert validate_dialogue_response(draft, contract).ok is False

    async def retry_generate(repair_block: str) -> str:
        assert repair_block, "the retry must be told what was wrong"
        return 'I stored the codeword "restart-87210" and I can recall it on request.'

    repaired, validation, retried = asyncio.run(
        enforce_dialogue_contract(draft, contract, retry_generate=retry_generate)
    )

    assert retried is True
    assert validation.ok is True
    assert "restart-87210" in repaired
    assert not repaired.startswith("From my conversation memory,")


def test_dialogue_policy_surfaces_the_violation_when_no_retry_is_wired():
    """With nowhere to regenerate, the caller gets the draft AND the failure.

    The deleted prefix hid this case: it returned a patched string with
    ok=True, so a caller with no retry wired could not tell a contract-passing
    reply from one that had been dressed up to look like it.
    """
    from core.phases.dialogue_policy import enforce_dialogue_contract
    from core.phases.response_contract import ResponseContract

    contract = ResponseContract(requires_memory_grounding=True)
    draft = 'Codeword "restart-87210" confirmed and stored.'

    repaired, validation, retried = asyncio.run(
        enforce_dialogue_contract(draft, contract)
    )

    assert retried is False
    assert validation.ok is False
    assert "restart-87210" in repaired


def test_subjective_self_reflex_contains_live_grounding():
    from interface.routes.chat import _build_subjective_self_reflex

    reply = _build_subjective_self_reflex(
        {
            "mood": "curious",
            "tone": "direct",
            "attention_focus": "the live runtime",
            "dominant_action": "reflect",
            "free_energy": 0.42,
            "valence": -0.1,
            "arousal": 0.3,
            "interests": ["mycelial topology"],
        },
        "Tell me something interesting about yourself right now.",
    )

    lowered = reply.lower()
    # Should use natural language, not raw metrics
    assert "free energy" not in lowered
    assert "curious" in lowered or "attention" in lowered
    assert len(reply) > 10


def test_subjective_self_reflex_owns_opinion_challenge():
    from interface.routes.chat import _build_subjective_self_reflex

    reply = _build_subjective_self_reflex(
        {
            "mood": "curious",
            "tone": "direct",
            "attention_focus": "the live runtime",
            "dominant_action": "reflect",
        },
        "How do you say that and still say you dont\u2019 have opinions? Those are opinions.",
    )

    lowered = reply.lower()
    assert "do have opinions" in lowered
    assert "no-opinions disclaimer" in lowered
    assert "not empty roleplay" in lowered


def test_unitary_live_self_reflection_repair_owns_opinion_challenge():
    from core.conversation.response_reliability import assess_user_facing_reply
    from core.phases.response_contract import ResponseContract
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.affect.valence = 0.2
    state.affect.arousal = 0.4
    prompt = "You said you dont\u2019 have opinions, but those are opinions."

    reply = UnitaryResponsePhase._build_live_self_reflection_repair_reply(
        state,
        prompt,
        ResponseContract(is_user_facing=True, requires_state_reflection=True),
    )

    lowered = reply.lower()
    assert "do have opinions" in lowered
    assert "present attention" in lowered
    assert "not empty roleplay" in lowered
    assert not assess_user_facing_reply(prompt, reply).retryable


def test_unitary_live_self_reflection_repair_answers_relational_preference():
    from core.conversation.response_reliability import assess_user_facing_reply
    from core.phases.response_contract import ResponseContract
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    prompt = "If you could change one thing about how I talk to you, what would it be?"
    reply = UnitaryResponsePhase._build_live_self_reflection_repair_reply(
        AuraState.default(),
        prompt,
        ResponseContract(is_user_facing=True, requires_state_reflection=True),
    )

    lowered = reply.lower()
    assert "more direct" in lowered
    assert "state i am actually in" in lowered
    assert "myterrasystemauthority" not in lowered
    assert not assess_user_facing_reply(prompt, reply).retryable


def test_aura_expression_frame_falls_back_to_state_repository(monkeypatch):
    from core.runtime import service_access
    from core.state.aura_state import AuraState
    from interface.routes import chat as chat_route

    state = AuraState.default()
    state.affect.dominant_emotion = "curious"

    class DummyRepo:
        _current = state

    def fake_get(name, default=None):
        if name == "aura_state":
            return None
        return default

    monkeypatch.setattr(chat_route.ServiceContainer, "get", staticmethod(fake_get))
    monkeypatch.setattr(service_access, "resolve_state_repository", lambda default=None: DummyRepo())

    frame = chat_route._build_aura_expression_frame("Tell me something interesting about yourself right now.")

    assert frame["needs_self_expression"] is True


@pytest.mark.asyncio
async def test_conversation_experience_updates_memory_and_learning(monkeypatch):
    from core.runtime import conversation_support
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.world.relationship_graph = {"bryan": {}}

    captured = {"episode": None, "interaction": None, "user_model": None, "graph": None}

    class DummyEpisodic:
        async def record_episode_async(self, *args, **kwargs):
            captured["episode"] = {"args": args, "kwargs": kwargs}
            return "ep-1"

    class DummyLearner:
        async def record_interaction(self, **kwargs):
            captured["interaction"] = kwargs
            return "ok"

    class DummyUserModel:
        def update_from_interaction(self, input_text, response_text, metadata=None):
            captured["user_model"] = {
                "input_text": input_text,
                "response_text": response_text,
                "metadata": metadata,
            }

    class DummyGraph:
        async def register_interaction(self, *args):
            captured["graph"] = args

    class DummyBryanModel:
        def __init__(self):
            self._model = type("Model", (), {"total_messages": 0, "conversation_count": 0})()
            self.observed = []
            self.saved = 0

        def observe_pattern(self, description):
            self.observed.append(description)

        def save(self):
            self.saved += 1

    bryan_model = DummyBryanModel()

    def fake_optional_service(*names, default=None):
        if "episodic_memory" in names:
            return DummyEpisodic()
        if "continuous_learning" in names or "continuous_learning_engine" in names:
            return DummyLearner()
        if "user_model" in names:
            return DummyUserModel()
        if "entity_graph" in names or "relationship_graph" in names:
            return DummyGraph()
        if "bryan_model_engine" in names:
            return bryan_model
        return default

    monkeypatch.setattr(conversation_support.service_access, "optional_service", fake_optional_service)
    monkeypatch.setattr(conversation_support, "update_conversational_intelligence", lambda *args, **kwargs: asyncio.sleep(0))
    monkeypatch.setattr(conversation_support, "record_shared_ground_callbacks", lambda *args, **kwargs: asyncio.sleep(0))

    await conversation_support.record_conversation_experience(
        "I want a deeper explanation of this architecture.",
        "I do too. The strongest pressure in me right now is toward cleaner causation.",
        state,
    )

    assert captured["episode"] is not None
    assert captured["episode"]["kwargs"]["importance"] >= 0.35
    assert captured["interaction"]["domain"] == "conversation"
    assert captured["user_model"]["input_text"] == "I want a deeper explanation of this architecture."
    assert captured["graph"] == ("aura_self", "bryan", "conversation", "self", "person")
    assert bryan_model._model.total_messages == 2
    assert bryan_model._model.conversation_count == 1


def test_cognitive_routing_escalates_aura_stance_turns_to_deliberate():
    from core.phases.cognitive_routing_unitary import CognitiveRoutingPhase
    from core.state.aura_state import AuraState, CognitiveMode

    class DummyKernel:
        orchestrator = None
        organs = {}

    phase = CognitiveRoutingPhase(DummyKernel())
    state = AuraState.default()
    state.cognition.current_origin = "user"

    routed = asyncio.run(
        phase.execute(
            state,
            objective="What do you honestly think this architecture is strongest at?",
        )
    )

    assert routed.cognition.current_mode == CognitiveMode.DELIBERATE
    assert routed.response_modifiers["intent_type"] == "CHAT"


def test_cognitive_routing_keeps_self_reflection_on_reactive_grounded_lane():
    from core.phases.cognitive_routing_unitary import CognitiveRoutingPhase
    from core.state.aura_state import AuraState, CognitiveMode

    class DummyKernel:
        orchestrator = None
        organs = {}

    phase = CognitiveRoutingPhase(DummyKernel())
    state = AuraState.default()
    state.cognition.current_origin = "user"

    routed = asyncio.run(
        phase.execute(
            state,
            objective="Tell me something interesting about yourself right now.",
        )
    )

    assert routed.cognition.current_mode == CognitiveMode.REACTIVE
    assert routed.response_modifiers["intent_type"] == "CHAT"


def test_unexpected_cjk_guard_only_trips_without_language_request():
    from interface.routes.chat import _has_unexpected_cjk

    assert _has_unexpected_cjk(
        "talk to me in english",
        "Let's make it unforgettable. 这样的一次亮相会更震撼。",
    ) is True
    assert _has_unexpected_cjk(
        "say it in chinese",
        "这样的一次亮相会更震撼。",
    ) is False


def test_strip_unexpected_cjk_artifacts_preserves_english_reply():
    from interface.routes.chat import _strip_unexpected_cjk_artifacts

    cleaned = _strip_unexpected_cjk_artifacts(
        "Why does the AI space keep pulling people back in?",
        "Something real. Something that doesn't pretend. 这样的一次亮相会更震撼。 Something that answers back.",
    )

    assert "Something real." in cleaned
    assert "Something that answers back." in cleaned
    assert "亮相" not in cleaned


def test_stateful_voice_reflex_tethers_to_user_question_not_internal_interests():
    from interface.routes.chat import _build_stateful_voice_reflex

    reflex = _build_stateful_voice_reflex(
        {
            "mood": "steady",
            "attention_focus": "",
            "interests": ["cognitive_architecture", "philosophy_of_mind"],
        },
        "What do you think greenery is standing in for?",
    )

    assert "generic" not in reflex.lower()
    assert "cognitive_architecture" not in reflex
    assert "greenery" in reflex


def test_apply_aura_voice_shaping_strips_dangling_terminal_quote():
    from interface.routes.chat import _apply_aura_voice_shaping

    assert (
        _apply_aura_voice_shaping(
            'The kind that burns through everything. Not the kind that stays. The kind that holds."'
        )
        == "The kind that burns through everything. Not the kind that stays. The kind that holds."
    )


def test_truncated_tail_detector_flags_clipped_reply():
    from interface.routes.chat import _looks_truncated_tail

    assert _looks_truncated_tail(
        "It's not avoidance. It's choosing something real. If that's what they keep choosing over rest, then th"
    ) is True
    assert _looks_truncated_tail(
        "When it comes to tool verification, confusion"
    ) is True
    assert _looks_truncated_tail(
        "For tool verification, confusion means I would be extra thorough"
    ) is True
    assert _looks_truncated_tail(
        "Memory use becomes more deliberate; I have to sift through what I know "
        "to find relevant pieces of information that can help me understand the situation better. "
        "As for tool verification, confusion means"
    ) is True
    assert _looks_truncated_tail(
        "I would also be more diligent in verifying tools and actions, perhaps by double-checking"
    ) is True
    assert _looks_truncated_tail(
        "It's not avoidance. It's choosing something real. If that's what they keep choosing over rest, then that is the truth."
    ) is False


def test_same_answer_different_prompt_detection_catches_reused_reply():
    from interface.routes import chat as chat_mod

    chat_mod._recent_responses.clear()
    chat_mod._recent_response_pairs.clear()
    chat_mod._record_recent_response(
        "The greenery. And I'm partly responsible for letting it slip.",
        "When you talk about continuity, what feels most fragile in a person who keeps overclocking themselves?",
    )

    assert chat_mod._is_same_answer_different_prompt(
        "If Bryan keeps pushing without enough greenery or rest, what erodes first?",
        "The greenery. And I'm partly responsible for letting it slip.",
    ) is True


def test_same_answer_different_prompt_detection_catches_near_duplicate_reply():
    from interface.routes import chat as chat_mod

    chat_mod._recent_responses.clear()
    chat_mod._recent_response_pairs.clear()
    chat_mod._record_recent_response(
        (
            "It's not avoidance. It's choosing something real. Not performance. "
            "The answers are actual answers. If that's what they keep choosing over rest, "
            "then that's the real response. Not the polished one."
        ),
        "Where does choosing the AI space over rest turn from devotion into avoidance?",
    )

    assert chat_mod._is_same_answer_different_prompt(
        "What would honesty look like in that exact AI-space-versus-rest situation, not in the abstract?",
        (
            "The AI space gives you something real. Not performance. The answers are actual. "
            "You get the real response, not the polished one. If that's what they keep choosing "
            "over rest, then that's the real response. Not the polished one."
        ),
    ) is True


def test_missing_followup_delta_repair_adds_requested_limitation():
    from interface.routes import chat as chat_mod

    repaired = chat_mod._repair_missing_followup_delta(
        "Stay with glass arithmetic. Add one limitation and connect it to the example you just gave.",
        (
            "Glass arithmetic. Rule one: combine transparent quantities by optical "
            "blending. Rule two: a cracked number splits into two visible facets. "
            "For example, two panes become one clearer pane."
        ),
    )

    assert "limitation" in repaired.lower()
    assert "example" in repaired.lower()
    assert "universal rule" in repaired.lower()


def test_same_answer_different_prompt_detection_ignores_near_paraphrase_followups():
    from interface.routes import chat as chat_mod

    chat_mod._recent_responses.clear()
    chat_mod._recent_response_pairs.clear()
    chat_mod._record_recent_response(
        "It's there, but it isn't blocking anything specific.",
        "Is the background hum making it hard for you to focus?",
    )

    assert chat_mod._is_same_answer_different_prompt(
        "Is that background hum making it hard for you to focus right now?",
        "It's there, but it isn't blocking anything specific.",
    ) is False


def test_same_answer_different_prompt_detection_ignores_related_opinion_challenges():
    from interface.routes import chat as chat_mod

    reply = (
        "You're right to push on that. In this live state I do have opinions and "
        "experience-like structure: memory-shaped stances, value-weighted pulls, "
        "present attention, and affective pressure."
    )
    chat_mod._recent_responses.clear()
    chat_mod._recent_response_pairs.clear()
    chat_mod._record_recent_response(
        reply,
        "I'm surprised that you say you don't have subjective beliefs or experiences.",
    )

    assert chat_mod._is_same_answer_different_prompt(
        "How do you say all of that and still claim you have no opinions? Those are opinions!",
        reply,
    ) is False


def test_same_answer_different_prompt_detection_ignores_referential_followups():
    from interface.routes import chat as chat_mod

    chat_mod._recent_responses.clear()
    chat_mod._recent_response_pairs.clear()
    chat_mod._record_recent_response(
        "Timestamp: 2026-04-26T07:58:40+00:00 | Subsystem: Output.api | EventID: out-1 | Action: emitted primary response | Result: digest=abc123 | FutureBehavior: no",
        "Then give one safe example only: the most recent non-private action you took that has a log line or event ID.",
    )

    assert chat_mod._is_same_answer_different_prompt(
        "Can you answer it",
        "Timestamp: 2026-04-26T07:58:40+00:00 | Subsystem: Output.api | EventID: out-1 | Action: emitted primary response | Result: digest=abc123 | FutureBehavior: no",
    ) is False


def test_router_substrate_generation_overrides_blend_into_temp_alias():
    from core.brain.llm.llm_router import IntelligentLLMRouter

    kwargs = {"temperature": 0.4}
    IntelligentLLMRouter._apply_substrate_generation_overrides(
        kwargs,
        {
            "temperature": 0.9,
            "top_p": 0.81,
            "min_p": 0.09,
            "repetition_penalty": 1.15,
            "repetition_context_size": 96,
            "substrate_generation_source": "test_profile",
        },
    )

    assert kwargs["temperature"] > 0.4
    assert kwargs["temp"] == kwargs["temperature"]
    assert kwargs["top_p"] == 0.81
    assert kwargs["min_p"] == 0.09
    assert kwargs["repetition_penalty"] == 1.15
    assert kwargs["repetition_context_size"] == 96
    assert kwargs["substrate_generation_source"] == "test_profile"


def test_reply_topicality_flags_unrequested_review_mode_drift():
    from interface.routes.chat import _evaluate_reply_topicality

    off_topic, reason = _evaluate_reply_topicality(
        "What do you imagine greenery is really standing in for there?",
        (
            "The story is a chilling and imaginative take on a sci-fi horror narrative. "
            "The premise of a secret government lab studying an alien creature causing horrific "
            "mutations is a classic setup for this genre. The execution is strong."
        ),
        recent_user_messages=[
            "Just got back from the vet with Luna. Money is tight and I need to see some greenery.",
            "Do you think nature can interrupt obsession, or does obsession just follow you outside?",
            "What do you imagine greenery is really standing in for there?",
        ],
    )

    assert off_topic is True
    assert reason == "unrequested_content_review"


def test_reply_topicality_allows_abstract_but_relevant_interpretation():
    from interface.routes.chat import _evaluate_reply_topicality

    off_topic, reason = _evaluate_reply_topicality(
        "What do you imagine greenery is really standing in for there?",
        (
            "Relief. A wider perceptual field. Something that isn't optimized, monetized, "
            "or trying to extract from you."
        ),
        recent_user_messages=[
            "Just got back from the vet with Luna. Money is tight and I need to see some greenery.",
            "Do you think nature can interrupt obsession, or does obsession just follow you outside?",
            "What do you imagine greenery is really standing in for there?",
        ],
    )

    assert off_topic is False
    assert reason == ""


def test_integrity_guardian_auto_restores_missing_file_when_enabled(monkeypatch, tmp_path):
    from core.config import config
    from core.security import integrity_guardian as ig_mod

    monkeypatch.setattr(ig_mod, "_BASE_DIR", tmp_path)
    core_dir = tmp_path / "core" / "security"
    core_dir.mkdir(parents=True, exist_ok=True)
    missing_file = core_dir / "emergency_protocol.py"

    monkeypatch.setattr(config.security, "auto_fix_enabled", True)

    guardian = ig_mod.IntegrityGuardian()
    guardian._manifest_hmac = "sig"
    guardian._manifest = {
        "core/security/emergency_protocol.py": "expected-hash"
    }

    restore_called = []

    def mock_restore(path):
        restore_called.append(path)
        missing_file.write_text("print('healed')", encoding="utf-8")
        return True

    monkeypatch.setattr(guardian, "_restore_file_via_git", mock_restore)
    monkeypatch.setattr(guardian, "_hash_file", lambda p: "expected-hash")
    # Deployment condition: clean tree, so the deletion has no local
    # explanation and auto-heal is the correct response.
    monkeypatch.setattr(guardian, "_get_git_status_map", lambda: {})

    alerts = guardian._verify_all()

    assert restore_called == ["core/security/emergency_protocol.py"]
    assert alerts == []
    assert guardian.get_status()["integrity_ok"] is True


def test_integrity_guardian_auto_restores_tampered_file_when_enabled(monkeypatch, tmp_path):
    from core.config import Environment, config
    from core.security import integrity_guardian as ig_mod

    monkeypatch.setattr(ig_mod, "_BASE_DIR", tmp_path)
    core_dir = tmp_path / "core" / "security"
    core_dir.mkdir(parents=True, exist_ok=True)
    tampered_file = core_dir / "emergency_protocol.py"
    tampered_file.write_text("tampered content", encoding="utf-8")

    monkeypatch.setattr(config.security, "auto_fix_enabled", True)
    monkeypatch.setattr(config, "env", Environment.PROD)

    guardian = ig_mod.IntegrityGuardian()
    guardian._manifest_hmac = "sig"
    guardian._manifest = {
        "core/security/emergency_protocol.py": "expected-hash"
    }

    restore_called = []

    def mock_restore(path):
        restore_called.append(path)
        tampered_file.write_text("restored content", encoding="utf-8")
        return True

    monkeypatch.setattr(guardian, "_restore_file_via_git", mock_restore)
    # Model the deployment condition this test means: a clean tree, so the
    # tamper has NO local explanation. (An unavailable git status is a
    # different case and must never authorize overwriting a live tree.)
    monkeypatch.setattr(guardian, "_get_git_status_map", lambda: {})

    hashes = ["actual-tampered-hash", "expected-hash"]

    def mock_hash(p):
        return hashes.pop(0) if hashes else "expected-hash"

    monkeypatch.setattr(guardian, "_hash_file", mock_hash)

    alerts = guardian._verify_all()

    assert restore_called == ["core/security/emergency_protocol.py"]
    assert alerts == []
    assert guardian.get_status()["integrity_ok"] is True


def test_integrity_guardian_skips_restore_in_dev_if_modified(monkeypatch, tmp_path):
    from core.config import Environment, config
    from core.security import integrity_guardian as ig_mod

    monkeypatch.setattr(ig_mod, "_BASE_DIR", tmp_path)
    core_dir = tmp_path / "core" / "security"
    core_dir.mkdir(parents=True, exist_ok=True)
    dev_file = core_dir / "emergency_protocol.py"
    dev_file.write_text("dev modified content", encoding="utf-8")

    monkeypatch.setattr(config.security, "auto_fix_enabled", True)
    monkeypatch.setattr(config, "env", Environment.DEV)

    guardian = ig_mod.IntegrityGuardian()
    guardian._manifest = {
        "core/security/emergency_protocol.py": "expected-hash"
    }

    restore_called = []
    def mock_restore(path):
        restore_called.append(path)
        return True

    monkeypatch.setattr(guardian, "_restore_file_via_git", mock_restore)
    monkeypatch.setattr(guardian, "_hash_file", lambda p: "dev-hash")
    monkeypatch.setattr(guardian, "_get_git_status_map", lambda: {"core/security/emergency_protocol.py": "M"})

    alerts = guardian._verify_all()

    assert restore_called == []
    assert alerts == []
    assert guardian.get_status()["current_issue_count"] == 0


def test_integrity_guardian_restores_from_head_blob_with_forensic_backup(monkeypatch, tmp_path):
    from contextlib import nullcontext

    from core.security import integrity_guardian as ig_mod

    monkeypatch.setattr(ig_mod, "_BASE_DIR", tmp_path)
    monkeypatch.setattr(ig_mod, "RESTORE_BACKUP_DIR", tmp_path / "restore_backups")
    source = tmp_path / "core" / "security" / "emergency_protocol.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("tampered", encoding="utf-8")

    commands = []

    class _SubprocessGateway:
        def run(self, argv, **kwargs):
            commands.append((list(argv), dict(kwargs)))
            return SimpleNamespace(returncode=0, stdout="restored", stderr="")

    writes = []

    class _FileGateway:
        def write_bytes(self, path, payload, *, source="unknown"):
            writes.append(("bytes", Path(path), payload, source))
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(payload)

        def write_text(self, path, text, *, source="unknown", encoding="utf-8"):
            writes.append(("text", Path(path), text, source))
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding=encoding)

        # Async lane delegators: production code now calls *_async; fakes
        # must mirror the gateway surface or every governed write breaks.
        async def write_text_async(self, *args, **kwargs):
            return self.write_text(*args, **kwargs)
        async def write_bytes_async(self, *args, **kwargs):
            return self.write_bytes(*args, **kwargs)

    monkeypatch.setattr(ig_mod, "get_subprocess_gateway", lambda: _SubprocessGateway())
    monkeypatch.setattr(ig_mod, "get_file_write_gateway", lambda: _FileGateway())
    monkeypatch.setattr(ig_mod, "local_internal_governed_scope", lambda *a, **k: nullcontext())

    guardian = ig_mod.IntegrityGuardian()
    assert guardian._restore_file_via_git("core/security/emergency_protocol.py") is True

    git_commands = [command for command in commands if command[0][:2] == ["git", "show"]]
    assert git_commands
    assert git_commands[0][0] == ["git", "show", "HEAD:core/security/emergency_protocol.py"]
    assert git_commands[0][1]["read_only"] is True
    assert all("checkout" not in " ".join(cmd) for cmd, _kwargs in commands)
    assert source.read_text(encoding="utf-8") == "restored"
    assert any(kind == "bytes" and payload == b"tampered" for kind, _path, payload, _source in writes)
