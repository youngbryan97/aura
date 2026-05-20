import pytest

from core.conversation_loop import AutonomousConversationLoop
from core.runtime.errors import get_degradation_tracker


class _Transcript:
    def get_context_window(self, n=50):
        return []


class _Planner:
    def __init__(self, result=None, error: BaseException | None = None):
        self.result = result
        self.error = error

    async def decompose(self, _goal_text):
        if self.error:
            raise self.error
        return self.result


class _Executor:
    def __init__(self):
        self.calls = []

    async def execute(self, goal, _context):
        self.calls.append(goal)
        if goal["tool"] == "broken":
            raise RuntimeError("tool transport failed")
        return {"ok": True, "summary": f"ran {goal['tool']}", "skill": goal["tool"]}


class _Drives:
    def __init__(self):
        self.punished = []
        self.satisfied = []

    def punish(self, name, amount):
        self.punished.append((name, amount))

    def satisfy(self, name, amount):
        self.satisfied.append((name, amount))


class _Memory:
    def __init__(self):
        self.added = []

    async def add(self, **kwargs):
        self.added.append(kwargs)


class _Brain:
    pass


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    get_degradation_tracker().reset()
    monkeypatch.setattr("core.conversation_loop.get_transcript", lambda: _Transcript())
    yield
    get_degradation_tracker().reset()


def _loop(*, planner=None, executor=None, drives=None, memory=None):
    return AutonomousConversationLoop(
        planner=planner or _Planner({"tool_calls": []}),
        executor=executor or _Executor(),
        drive_system=drives or _Drives(),
        memory=memory or _Memory(),
        brain=_Brain(),
    )


@pytest.mark.asyncio
async def test_execute_plan_records_failed_tool_and_continues_remaining_steps():
    executor = _Executor()
    loop = _loop(executor=executor)

    results = await loop._execute_plan(
        {
            "tool_calls": [
                {"tool": "broken", "objective": "first"},
                {"tool": "working", "objective": "second"},
            ]
        }
    )

    assert [result["ok"] for result in results] == [False, True]
    assert [call["tool"] for call in executor.calls] == ["broken", "working"]
    assert loop.stats["failed_executions"] == 1
    assert loop.stats["successful_executions"] == 1
    assert loop.stats["recovered_tool_failures"] == 1
    last = get_degradation_tracker().recent(subsystem="conversation_loop")[-1]
    assert last.action == "recorded failed tool result and continued remaining plan steps"


@pytest.mark.asyncio
async def test_autonomous_goal_failure_punishes_competence_and_returns_result():
    drives = _Drives()
    loop = _loop(planner=_Planner(error=RuntimeError("planner offline")), drives=drives)

    result = await loop._execute_autonomous_goal("learn something useful")

    assert result["ok"] is False
    assert result["error"] == "RuntimeError"
    assert drives.punished == [("competence", 15.0)]
    assert loop.get_status()["autonomous_failure_streak"] == 1
    last = get_degradation_tracker().recent(subsystem="conversation_loop")[-1]
    assert (
        last.action
        == "punished competence and abandoned autonomous goal after recoverable execution failure"
    )


@pytest.mark.asyncio
async def test_execute_plan_invalid_shape_returns_empty_results_with_receipt():
    loop = _loop()

    assert await loop._execute_plan({"tool_calls": "not-a-list"}) == []
    assert loop.get_status()["plan_failure_streak"] == 1
    last = get_degradation_tracker().recent(subsystem="conversation_loop")[-1]
    assert (
        last.action == "returned empty execution results because plan tool_calls had invalid shape"
    )
