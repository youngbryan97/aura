"""A fix nobody could write yet is waiting, not failed.

The router hands back nothing when it puts a background request off — the
cortex busy with her conversation, a lane still warming — and code repair read
that as "Fix generation failed": a hundred and ten warnings in one session for
a pipeline that was only waiting its turn.
"""

from __future__ import annotations

import pytest

from core.self_modification.code_repair import CodeFixGenerator


@pytest.mark.asyncio
async def test_an_empty_answer_that_was_put_off_says_so(monkeypatch, tmp_path):
    from core.brain.llm import deferral_record
    from core.container import ServiceContainer

    class _Router:
        async def think(self, **_kwargs):
            deferral_record.record_deferral(origin="code_repair", reason="foreground_busy")
            return ""

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda name, default=None: _Router()))
    generator = CodeFixGenerator(None, str(tmp_path))
    generator.last_deferred = ""
    got = await generator._generate_fix_code(
        "x.py", 1, {"buggy_section": "x = 1\n", "before": "", "after": "", "buggy_line": "x = 1\n", "start_line": 1, "end_line": 1, "full_file": "x = 1\n"},
        {"root_cause": "r", "potential_fix": "p"},
    )
    assert got is None
    assert "foreground_busy" in generator.last_deferred


@pytest.mark.asyncio
async def test_an_empty_answer_nobody_put_off_is_still_a_failure(monkeypatch, tmp_path):
    from core.container import ServiceContainer

    class _Router:
        async def think(self, **_kwargs):
            return ""

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda name, default=None: _Router()))
    generator = CodeFixGenerator(None, str(tmp_path))
    generator.last_deferred = ""
    await generator._generate_fix_code(
        "x.py", 1, {"buggy_section": "x = 1\n", "before": "", "after": "", "buggy_line": "x = 1\n", "start_line": 1, "end_line": 1, "full_file": "x = 1\n"},
        {"root_cause": "r", "potential_fix": "p"},
    )
    assert generator.last_deferred == ""
