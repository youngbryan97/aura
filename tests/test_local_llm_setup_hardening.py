from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.brain.llm import local_llm_setup
from core.brain.llm.local_llm_setup import OllamaManager


def test_ensure_installed_uses_bounded_version_check(monkeypatch):
    calls = []
    monkeypatch.setattr(local_llm_setup.shutil, "which", lambda _name: "/usr/local/bin/ollama")

    def _run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout="ollama version 1.0")

    monkeypatch.setattr(local_llm_setup.subprocess, "run", _run)

    assert OllamaManager().ensure_installed() is True
    assert calls == [
        (
            ["ollama", "--version"],
            {"check": True, "capture_output": True, "timeout": local_llm_setup._VERSION_TIMEOUT_S},
        )
    ]


def test_ensure_model_uses_bounded_list_and_pull(monkeypatch):
    calls = []

    def _run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd == ["ollama", "list"]:
            return SimpleNamespace(stdout="other-model")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(local_llm_setup.subprocess, "run", _run)

    manager = OllamaManager(model_name="aura-test")
    assert manager.ensure_model() is True
    assert calls[0] == (
        ["ollama", "list"],
        {
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": local_llm_setup._LIST_TIMEOUT_S,
        },
    )
    assert calls[1] == (
        ["ollama", "pull", "aura-test"],
        {"check": True, "timeout": local_llm_setup._PULL_TIMEOUT_S},
    )


@pytest.mark.asyncio
async def test_start_cleans_up_process_when_readiness_never_arrives(monkeypatch):
    class FakeProcess:
        returncode = None

        def __init__(self):
            self.terminated = False
            self.killed = False

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.killed = True
            self.returncode = -9

        async def wait(self):
            return self.returncode

    process = FakeProcess()

    async def _spawn(*_args, **_kwargs):
        return process

    monkeypatch.setattr(local_llm_setup.asyncio, "create_subprocess_exec", _spawn)
    monkeypatch.setattr(local_llm_setup.asyncio, "sleep", lambda *_args, **_kwargs: _noop())

    manager = OllamaManager(model_name="aura-test")
    checks = {"count": 0}

    async def _not_running():
        checks["count"] += 1
        return False

    manager.is_running = _not_running

    assert await manager.start() is False
    assert process.terminated is True
    assert process.killed is False
    assert checks["count"] == local_llm_setup._SERVE_READY_ATTEMPTS + 1


@pytest.mark.asyncio
async def test_start_cleans_up_process_after_spawn_failure(monkeypatch):
    async def _raise(*_args, **_kwargs):
        reason = "spawn failed"
        raise OSError(reason)

    monkeypatch.setattr(local_llm_setup.asyncio, "create_subprocess_exec", _raise)

    manager = OllamaManager(model_name="aura-test")
    manager.is_running = _false

    assert await manager.start() is False


async def _noop():
    return None


async def _false():
    return False
