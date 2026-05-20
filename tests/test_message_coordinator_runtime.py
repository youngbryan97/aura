import asyncio
from types import SimpleNamespace

from core.coordinators import message_coordinator as coordinator_module
from core.coordinators.message_coordinator import MessageCoordinator


class Hooks:
    async def trigger(self, *_args, **_kwargs):
        return None


class IntentRouter:
    async def classify(self, _message, _context):
        return {"kind": "chat"}


class FailingStateMachine:
    async def execute(self, _intent, _message, _context):
        self.calls = getattr(self, "calls", 0) + 1
        raise RuntimeError("state machine offline")


class TaskTracker:
    def __init__(self):
        self.created = []

    def create_task(self, coro, name=None):
        task = asyncio.create_task(coro, name=name)
        self.created.append((name, task))
        return task


def _orch(state_machine=None):
    return SimpleNamespace(
        hooks=Hooks(),
        _current_thought_task=None,
        status=SimpleNamespace(is_processing=False),
        intent_router=IntentRouter(),
        state_machine=state_machine or FailingStateMachine(),
        conversation_history=[],
        AI_ROLE="Aura",
        reply_queue=asyncio.Queue(),
    )


def test_message_coordinator_returns_reply_on_state_machine_failure(monkeypatch):
    async def scenario():
        tracker = TaskTracker()
        monkeypatch.setattr(coordinator_module, "task_tracker", tracker)
        orch = _orch()
        coord = MessageCoordinator(orch)

        await coord.handle_incoming_message("hello", origin="user")
        await orch._current_thought_task

        reply = orch.reply_queue.get_nowait()
        assert "recoverable message-routing fault" in reply
        assert orch.status.is_processing is False
        assert tracker.created[0][0] == "message_coordinator.execute_and_reply"

    asyncio.run(scenario())


def test_message_coordinator_task_schedule_failure_replies_and_closes(monkeypatch):
    class RejectingTracker:
        def create_task(self, coro, name=None):
            self.name = name
            raise RuntimeError("task tracker offline")

    async def scenario():
        tracker = RejectingTracker()
        monkeypatch.setattr(coordinator_module, "task_tracker", tracker)
        orch = _orch()
        coord = MessageCoordinator(orch)

        await coord.handle_incoming_message("hello", origin="user")

        reply = orch.reply_queue.get_nowait()
        assert "recoverable message-routing fault" in reply
        assert tracker.name == "message_coordinator.execute_and_reply"
        assert orch.status.is_processing is False

    asyncio.run(scenario())
