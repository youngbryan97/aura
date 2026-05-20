from pathlib import Path

import pytest
from pydantic import BaseModel

from core.brain.llm import structured_llm as structured_module
from core.brain.llm.structured_llm import StructuredLLM
from tools.audit_degradation import analyze_file


class _TaskModel(BaseModel):
    action: str
    priority: int


class _MetadataRouter:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def generate_with_metadata(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def test_structured_llm_degradation_audit_is_clean():
    assert analyze_file(Path("core/brain/llm/structured_llm.py")) == []


@pytest.mark.asyncio
async def test_validation_telemetry_failure_does_not_block_schema_retry(monkeypatch):
    router = _MetadataRouter(
        {"text": '{"action": "test", "priority": "high"}'},
        {"text": '{"action": "test", "priority": 10}'},
    )
    monkeypatch.setattr(
        structured_module,
        "record_degraded_event",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("telemetry down")),
    )

    result = await StructuredLLM(_TaskModel, max_retries=2, llm_router=router).generate("Return a task.")

    assert result == _TaskModel(action="test", priority=10)
    assert len(router.calls) == 2
    assert router.calls[1]["prefer_tier"] == "primary"


@pytest.mark.asyncio
async def test_router_technical_failure_escalates_and_recovers():
    router = _MetadataRouter(
        RuntimeError("local lane down"),
        {"text": '{"action": "recover", "priority": 2}'},
    )

    result = await StructuredLLM(_TaskModel, max_retries=2, llm_router=router).generate("Return a task.")

    assert result == _TaskModel(action="recover", priority=2)
    assert len(router.calls) == 2
    assert router.calls[1]["prefer_tier"] == "primary"


@pytest.mark.asyncio
async def test_background_policy_failure_defers_instead_of_running_router(monkeypatch):
    import core.runtime.background_policy as background_policy

    router = _MetadataRouter({"text": '{"action": "should-not-run", "priority": 1}'})

    def fail_policy(*_args, **_kwargs):
        raise RuntimeError("policy unavailable")

    monkeypatch.setattr(background_policy, "background_activity_reason", fail_policy)

    structured = StructuredLLM(_TaskModel, max_retries=2, llm_router=router)
    result = await structured.generate("Return a task.")

    assert result is None
    assert structured.last_defer_reason == "background_policy_unavailable"
    assert router.calls == []
