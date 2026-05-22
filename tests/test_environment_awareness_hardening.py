from __future__ import annotations

import asyncio

from core import environment_awareness as env_module
from core.environment_awareness import UserIdentityManager


def test_environment_command_probe_blocks_non_allowlisted_command(monkeypatch):
    recorded: list[tuple[str, str, dict[str, object]]] = []

    monkeypatch.setattr(
        env_module,
        "record_degradation",
        lambda module, exc, **kwargs: recorded.append((module, type(exc).__name__, kwargs)),
    )

    result = asyncio.run(env_module._run_command(["python", "--version"]))

    assert result == ""
    assert recorded
    assert recorded[0][0] == "environment_awareness"
    assert recorded[0][1] == "PermissionError"
    assert recorded[0][2]["receipt_required"] is True
    assert "non-allowlisted" in str(recorded[0][2]["action"])


def test_user_identity_manager_corrupt_store_fails_soft(monkeypatch, tmp_path):
    recorded: list[tuple[str, str, dict[str, object]]] = []
    data_path = tmp_path / "user_sessions.json"
    data_path.write_text("{bad-json", encoding="utf-8")

    monkeypatch.setattr(env_module, "_data_path", lambda _filename: data_path)
    monkeypatch.setattr(
        env_module,
        "record_degradation",
        lambda module, exc, **kwargs: recorded.append((module, type(exc).__name__, kwargs)),
    )

    manager = UserIdentityManager()

    assert manager._known_fingerprints == {}
    assert recorded
    assert recorded[0][0] == "environment_awareness"
    assert recorded[0][2]["receipt_required"] is True
    assert "fingerprint store" in str(recorded[0][2]["action"])


def test_user_identity_manager_persists_under_configured_data_path(monkeypatch, tmp_path):
    data_path = tmp_path / "user_sessions.json"
    monkeypatch.setattr(env_module, "_data_path", lambda _filename: data_path)

    manager = UserIdentityManager()
    manager.register_identity("abc123", "Bryan")

    assert data_path.exists()
    assert "Bryan" in data_path.read_text(encoding="utf-8")
