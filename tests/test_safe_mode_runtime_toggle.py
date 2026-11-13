"""tests/test_safe_mode_runtime_toggle.py
=============================================
The user-facing "Safe mode" toggle must actually do something. It was a dead
setting: boot always applied full mode and nothing read safety.safe_mode. These
tests lock in the live runtime toggle and the settings → runtime bridge.
"""
from __future__ import annotations

from core.runtime.safe_mode import is_safe_mode, runtime_feature_enabled, set_safe_mode


class _Kernel:
    def __init__(self, volition: int):
        self.volition_level = volition


class _Orch:
    def __init__(self, volition: int = 3):
        self.kernel = _Kernel(volition)
        self.conversation_history: list = []


def test_set_safe_mode_disables_self_directed_behavior():
    orch = _Orch(volition=3)
    out = set_safe_mode(orch, True)
    assert out["safe_mode"] is True
    assert is_safe_mode(orch) is True
    # Consumers read these via runtime_feature_enabled on their next tick.
    assert runtime_feature_enabled(orch, "self_modification") is False
    assert runtime_feature_enabled(orch, "persona_evolution") is False
    assert runtime_feature_enabled(orch, "dream_cycle") is False
    assert runtime_feature_enabled(orch, "memory_consolidation") is False


def test_full_mode_restores_capabilities_at_high_volition():
    orch = _Orch(volition=3)
    set_safe_mode(orch, True)
    out = set_safe_mode(orch, False)
    assert out["safe_mode"] is False
    assert is_safe_mode(orch) is False
    # Volition 3 in full mode re-enables self-modification.
    assert runtime_feature_enabled(orch, "self_modification") is True
    assert runtime_feature_enabled(orch, "persona_evolution") is True


def test_safe_mode_holds_even_at_high_volition():
    # Safe mode must override volition — a "paused" kill switch.
    orch = _Orch(volition=3)
    set_safe_mode(orch, True)
    assert runtime_feature_enabled(orch, "self_modification") is False


def test_toggle_is_idempotent():
    orch = _Orch(volition=1)
    set_safe_mode(orch, True)
    set_safe_mode(orch, True)
    assert is_safe_mode(orch) is True
    assert runtime_feature_enabled(orch, "self_modification") is False


def test_is_safe_mode_defaults_false_on_unpatched_orch():
    assert is_safe_mode(_Orch()) is False


def _fresh_store(monkeypatch):
    """A fresh settings store wired with the runtime-mode bridge, fully isolated.

    Uses a UNIQUE temp dir per call so store.set() persistence in one test can
    never pollute another (an earlier bug: tests shared a 'nope.json' file).
    """
    import tempfile
    from pathlib import Path

    import interface.routes.settings as settings

    unique = Path(tempfile.mkdtemp(prefix="aura_settings_")) / "runtime.json"
    monkeypatch.setattr(settings, "_STORE", None)
    monkeypatch.setattr(settings, "_SETTINGS_PATH", unique)
    return settings, settings.get_settings()


def test_default_settings_are_full_aura(monkeypatch):
    """Out of the box, Aura runs at FULL capability — not restricted."""
    settings, store = _fresh_store(monkeypatch)
    assert store.get("safety.safe_mode") is False
    assert store.get("autonomy.level") == "full"
    assert settings._runtime_should_restrict(store) is False


def test_settings_bridge_applies_safe_mode_to_live_orchestrator(monkeypatch):
    """Changing safety.safe_mode in the store must apply to the live runtime."""
    import core.container as container

    settings, store = _fresh_store(monkeypatch)
    orch = _Orch(volition=3)
    monkeypatch.setattr(container.ServiceContainer, "get", staticmethod(
        lambda name, default=None: orch if name == "orchestrator" else default
    ))
    store.set("safety.safe_mode", True)
    assert is_safe_mode(orch) is True
    store.set("safety.safe_mode", False)
    assert is_safe_mode(orch) is False


def test_autonomy_level_is_not_a_runtime_kill_switch(monkeypatch):
    """Only explicit emergency containment may restrict the live runtime."""
    settings, store = _fresh_store(monkeypatch)

    assert store.get("autonomy.level") == "full"
    assert settings._runtime_should_restrict(store) is False
    assert "autonomy.level" not in settings._RUNTIME_MODE_KEYS


def test_safe_mode_holds_even_if_autonomy_not_paused(monkeypatch):
    import core.container as container

    _settings, store = _fresh_store(monkeypatch)
    orch = _Orch(volition=3)
    monkeypatch.setattr(container.ServiceContainer, "get", staticmethod(
        lambda name, default=None: orch if name == "orchestrator" else default
    ))
    store.set("safety.safe_mode", True)
    assert store.get("autonomy.level") == "full"
    assert is_safe_mode(orch) is True


def test_settings_bridge_ignores_other_keys(monkeypatch):
    import core.container as container
    import interface.routes.settings as settings

    orch = _Orch(volition=3)
    set_safe_mode(orch, True)
    monkeypatch.setattr(container.ServiceContainer, "get", staticmethod(
        lambda name, default=None: orch if name == "orchestrator" else default
    ))
    settings._apply_runtime_mode_from_settings("theme.mode", "auto", "dark")
    assert is_safe_mode(orch) is True


def test_settings_store_registers_runtime_mode_subscriber(monkeypatch):
    settings, store = _fresh_store(monkeypatch)
    assert settings._apply_runtime_mode_from_settings in {
        subscriber.callback for subscriber in store._subscribers
    }


def test_her_kernel_is_read_from_the_container_when_the_orchestrator_holds_none(caplog):
    """Her orchestrator has no kernel attribute; the boot line printed volition=0 while her kernel ran at 3."""
    import logging

    from core.container import ServiceContainer
    from core.runtime.safe_mode import apply_orchestrator_patches, runtime_mode_value

    class _Bare:
        conversation_history: list = []

    ServiceContainer.register_instance("aura_kernel", _Kernel(3))
    try:
        orch = _Bare()
        with caplog.at_level(logging.INFO):
            apply_orchestrator_patches(orch)
        assert "volition=3" in caplog.text
        assert runtime_mode_value(orch, "self_preservation") is True
    finally:
        ServiceContainer.unregister("aura_kernel")


def test_before_her_kernel_exists_the_line_says_so(caplog):
    import logging

    from core.runtime.safe_mode import apply_orchestrator_patches

    class _Bare:
        conversation_history: list = []

    with caplog.at_level(logging.INFO):
        apply_orchestrator_patches(_Bare())
    assert "volition=not yet set" in caplog.text
