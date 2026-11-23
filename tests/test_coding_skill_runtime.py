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


@pytest.mark.asyncio
async def test_coding_skill_rejects_empty_draft(monkeypatch):
    """An empty model draft must NOT report success with no code — the exact
    'technically true but useless' failure (empty_cognitive_engine_reply is a
    real live event under load)."""

    class _EmptyBrain:
        async def generate(self, **kwargs):
            return "   \n  "

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: _EmptyBrain() if name == "cognitive_engine" else default),
    )

    skill = CodingSkill()
    result = await skill.execute(
        {"params": {"task": "Write add(a, b).", "language": "python"}},
        {},
    )

    assert result["ok"] is False
    assert "no code" in result["error"].lower()


@pytest.mark.asyncio
async def test_a_draft_that_misses_its_examples_is_mended_against_them(monkeypatch):
    """A draft that fails examples someone else supplied was given back as a failure with no second try. It is now
    mended against those examples by the repair outside code gets (repairing_python.repaired_source), and a draft that
    then passes them comes back as code that does, with the edits that made it so."""

    class _Brain:
        async def generate(self, **kwargs):
            return "def area(width, height):\n    return width * height * 0\n"

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: _Brain() if name == "cognitive_engine" else default),
    )

    async def no_model(source, seen_wrong, **kwargs):
        return []

    monkeypatch.setattr("core.self_modification.edits_her_model_proposes.edits_her_model_proposes", no_model)
    result = await CodingSkill().execute(
        {"params": {"task": "Write area(width, height).", "language": "python",
                    "checks": [{"name": "a 3 by 4 room", "function": "area", "args": [3, 4], "expected": 12}]}},
        {},
    )
    assert result["ok"] is True, result
    assert "* 0" not in result["code"] and result["verification"]["verified"] is True
    assert result["repaired"] and "mended" in result["note"]
