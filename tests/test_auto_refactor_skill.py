from __future__ import annotations

import pytest

from core.skills.auto_refactor import AutoRefactorParams, AutoRefactorSkill


@pytest.mark.asyncio
async def test_auto_refactor_scan_runs_off_event_loop(monkeypatch):
    skill = AutoRefactorSkill()
    observed = {}

    async def fake_to_thread(fn, *args, **kwargs):
        observed["fn"] = fn
        observed["args"] = args
        observed["kwargs"] = kwargs
        return [
            {
                "file": "core/example.py",
                "line": 12,
                "type": "complexity",
                "message": "Function 'demo' is too long (88 lines).",
            }
        ]

    monkeypatch.setattr("core.skills.auto_refactor.asyncio.to_thread", fake_to_thread)

    result = await skill.execute(AutoRefactorParams(path="."), context={})

    assert observed["fn"] == skill._scan_codebase
    assert observed["args"] == (".",)
    assert result["ok"] is True
    assert result["issues_found"] == 1
    assert result["top_issues"][0]["file"] == "core/example.py"
