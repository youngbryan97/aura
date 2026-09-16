"""The self-repair generator asks the router for code; it does not hand the
kernel an objective.

LIVE, 2026-09-15: through the cognitive engine the repair prompt became the
kernel's objective — "obj: You are fixing a bug in your own code." — and the
kernel answered it as a conversation, "I would handle this as a bounded...",
the same sentence every tick: 230 loop detections in one session.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.self_modification import code_repair


class _Generator:
    calls: list[tuple[str, dict]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def generate_async(self, prompt, context):
        _Generator.calls.append((prompt, dict(context)))
        return "def fixed():\n    return 1\n"


@pytest.mark.asyncio
async def test_the_fix_comes_from_the_code_generator(monkeypatch):
    import core.brain.llm.code_generator as cg

    monkeypatch.setattr(cg, "LLMCodeGenerator", _Generator)
    _Generator.calls.clear()

    class _Brain:
        async def think(self, *a, **k):
            raise AssertionError("the kernel must not be handed a repair prompt as an objective")

    fixer = code_repair.CodeFixGenerator(_Brain(), str(Path(__file__).resolve().parent.parent))
    code = await fixer._generate_fix_code(
        "core/example.py",
        12,
        {"start_line": 10, "end_line": 14, "buggy_section": "def f():\n    return 0\n", "ast_summary": {}},
        {"root_cause": "off by one", "explanation": "x", "potential_fix": "y"},
    )
    assert code == "def fixed():\n    return 1"
    (prompt, context), = _Generator.calls
    assert context["origin"] == "code_repair"
    assert context["is_background"] is True
    assert context["module_path"] == "core/example.py"


def test_the_repair_module_never_thinks_through_the_brain():
    source = (Path(code_repair.__file__)).read_text(encoding="utf-8")
    assert "self.brain.think(" not in source
