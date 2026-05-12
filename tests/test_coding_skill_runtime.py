from types import SimpleNamespace

import pytest

from core.skills.coding_skill import CodingSkill


@pytest.mark.asyncio
async def test_coding_skill_uses_foreground_reasoning_contract(monkeypatch):
    calls = []

    class _Brain:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            return "def add(a, b):\n    return a + b"

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: _Brain() if name == "cognitive_engine" else default),
    )

    skill = CodingSkill()
    result = await skill.execute(
        {"params": {"task": "Write add(a, b).", "language": "python"}},
        {"origin": "api", "deep_handoff": True},
    )

    assert result["ok"] is True
    assert "def add" in result["code"]
    assert result["note"] == "Generated through foreground coding reasoning"
    assert calls[0]["origin"] == "api"
    assert calls[0]["purpose"] == "coding"
    assert calls[0]["prefer_tier"] == "primary"
    assert calls[0]["deep_handoff"] is True
    assert calls[0]["max_tokens"] == 4096
