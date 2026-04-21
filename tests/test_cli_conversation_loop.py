import builtins

import pytest


@pytest.mark.asyncio
async def test_conversation_loop_reuses_supplied_orchestrator(monkeypatch):
    from core import main as main_module

    class _StubOrchestrator:
        def __init__(self):
            self.start_calls = 0
            self.processed = []
            self._conversation_loop_heartbeat_started = False

        async def start(self):
            self.start_calls += 1

        async def run(self):
            return None

        async def _process_message(self, message):
            self.processed.append(message)
            return {"ok": True}

    orchestrator = _StubOrchestrator()
    tracked = []

    monkeypatch.setattr(
        "core.service_registration.register_all_services",
        lambda: (_ for _ in ()).throw(AssertionError("should not register services for a supplied orchestrator")),
    )
    monkeypatch.setattr(
        "core.orchestrator.create_orchestrator",
        lambda: (_ for _ in ()).throw(AssertionError("should not create a second orchestrator")),
    )
    monkeypatch.setattr(
        "core.utils.task_tracker.fire_and_track",
        lambda coro, name=None: (tracked.append(name), coro.close()),
    )
    monkeypatch.setattr(builtins, "input", lambda prompt="": "exit")
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: None)

    await main_module.conversation_loop(orchestrator=orchestrator)

    assert orchestrator.start_calls == 0
    assert tracked == ["OrchestratorMainLoop"]
    assert orchestrator.processed == ["System Command: ENTER_REM_SLEEP"]
