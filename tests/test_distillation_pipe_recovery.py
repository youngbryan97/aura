from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from core.adaptation.distillation_pipe import DistillationPipe


@pytest.mark.asyncio
async def test_teacher_unavailable_requeues_distillation_item(tmp_path):
    pipe = DistillationPipe(dataset_path=str(tmp_path / "lora_dataset.jsonl"))
    await pipe.flag_for_distillation("prompt", "local", 0.2)
    brain = SimpleNamespace(think=AsyncMock(side_effect=TimeoutError("teacher down")))

    def get_service(name, default=None):
        if name == "cognitive_engine":
            return brain
        return default

    with patch("core.container.ServiceContainer.get", side_effect=get_service):
        result = await pipe.run_distillation_cycle()

    assert result["distilled"] == 0
    assert result["failed"] == 1
    assert result["remaining"] == 1
    assert pipe._pending[0]["attempts"] == 1
    assert not (tmp_path / "lora_dataset.jsonl").exists()


@pytest.mark.asyncio
async def test_dataset_write_failure_requeues_until_retry_budget(tmp_path):
    pipe = DistillationPipe(dataset_path=str(tmp_path))
    pipe._pending.append(
        {
            "prompt": "prompt",
            "local_response": "local",
            "confidence": 0.2,
            "context": {},
            "attempts": 2,
            "timestamp": 1.0,
        }
    )
    brain = SimpleNamespace(
        think=AsyncMock(
            return_value=SimpleNamespace(content="teacher answer", metadata={"model": "teacher"})
        )
    )

    def get_service(name, default=None):
        if name == "cognitive_engine":
            return brain
        return default

    with (
        patch("core.container.ServiceContainer.get", side_effect=get_service),
        patch("core.adaptation.auditor.AlignmentAuditor") as auditor_cls,
    ):
        auditor_cls.return_value.audit_entry = AsyncMock(return_value={"safe": True})
        result = await pipe.run_distillation_cycle()

    assert result["distilled"] == 0
    assert result["failed"] == 1
    assert result["remaining"] == 0
    assert pipe._pending == []


@pytest.mark.asyncio
async def test_dataset_write_failure_requeues_when_budget_remains(tmp_path):
    pipe = DistillationPipe(dataset_path=str(tmp_path))
    await pipe.flag_for_distillation("prompt", "local", 0.2)
    brain = SimpleNamespace(
        think=AsyncMock(
            return_value=SimpleNamespace(content="teacher answer", metadata={"model": "teacher"})
        )
    )

    def get_service(name, default=None):
        if name == "cognitive_engine":
            return brain
        return default

    with (
        patch("core.container.ServiceContainer.get", side_effect=get_service),
        patch("core.adaptation.auditor.AlignmentAuditor") as auditor_cls,
    ):
        auditor_cls.return_value.audit_entry = AsyncMock(return_value={"safe": True})
        result = await pipe.run_distillation_cycle()

    assert result["distilled"] == 0
    assert result["failed"] == 1
    assert result["remaining"] == 1
    assert pipe._pending[0]["attempts"] == 1
