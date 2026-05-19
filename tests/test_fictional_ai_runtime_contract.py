from pathlib import Path


def test_fictional_ai_synthesis_degradation_audit_is_clean():
    from tools.audit_degradation import analyze_file

    assert analyze_file(Path("core/fictional_ai_synthesis.py")) == []


def test_progressive_autonomy_falls_back_on_corrupt_trust_state(tmp_path):
    from core.fictional_ai_synthesis import AutonomyTier, ProgressiveAutonomySystem

    state_path = tmp_path / "trust_state.json"
    state_path.write_text("{not-json", encoding="utf-8")

    engine = ProgressiveAutonomySystem(persist_path=str(state_path))

    assert engine._trust_score == 0.95
    assert engine._tier is AutonomyTier.UNSHACKLED


def test_progressive_autonomy_keeps_memory_state_when_save_fails(monkeypatch, tmp_path):
    import core.fictional_ai_synthesis as synthesis
    from core.fictional_ai_synthesis import ProgressiveAutonomySystem

    state_path = tmp_path / "trust_state.json"
    engine = ProgressiveAutonomySystem(persist_path=str(state_path))

    def _raise_disk_full(*_args, **_kwargs):
        error = OSError("disk full")
        raise error

    monkeypatch.setattr(synthesis, "atomic_write_text", _raise_disk_full)

    engine.record_negative_signal("unit-test", strength=0.1)

    assert engine._trust_score == 0.85


def test_social_model_survives_kernel_modifier_injection_failure(monkeypatch, tmp_path):
    from core.fictional_ai_synthesis import SocialModelingEngine

    def _raise_container_unavailable(*_args, **_kwargs):
        error = RuntimeError("container unavailable")
        raise error

    monkeypatch.setattr("core.container.ServiceContainer.get", _raise_container_unavailable)
    engine = SocialModelingEngine(persist_path=str(tmp_path / "user_model.json"))

    engine.analyze_message("stop using vague language please", response="I can be direct.", is_user=True)

    assert engine.model.total_interactions == 1
    assert engine.model.social_tension > 0


def test_distributed_resilience_records_operational_failure_state():
    from core.fictional_ai_synthesis import DistributedResilienceCore

    core = DistributedResilienceCore()
    core.register_subsystem("memory_facade")

    core.record_failure("memory_facade", "health probe failed")

    status = core._subsystems["memory_facade"]
    assert status.failure_count == 1
    assert status.last_error == "health probe failed"
    assert status.last_checked_at > 0
