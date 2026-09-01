from __future__ import annotations

import logging

import pytest

from core.capability_engine import CapabilityEngine


class _ReportingSkill:
    def __init__(self, *, retryable: bool) -> None:
        self.calls = 0
        self.retryable = retryable

    async def safe_execute(self, _params, _context):
        self.calls += 1
        return {
            "ok": False,
            "error": "network access is forbidden by this local policy",
            "retryable": self.retryable,
        }


def _bare_engine() -> CapabilityEngine:
    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.timeout = 5.0
    engine.max_retries = 3
    engine.retry_delay = 0.0
    engine.logger = logging.getLogger("test.structured_retry")
    return engine


@pytest.mark.asyncio
async def test_typed_terminal_failure_beats_transient_sounding_prose() -> None:
    engine = _bare_engine()
    skill = _ReportingSkill(retryable=False)

    result = await engine._execute_with_retry(skill, "typed", {}, {})

    assert skill.calls == 1
    assert result["retries"] == 0


@pytest.mark.asyncio
async def test_typed_transient_failure_beats_terminal_sounding_prose() -> None:
    engine = _bare_engine()
    skill = _ReportingSkill(retryable=True)

    result = await engine._execute_with_retry(skill, "typed", {}, {})

    assert skill.calls == 3
    assert result["retries"] == 2
