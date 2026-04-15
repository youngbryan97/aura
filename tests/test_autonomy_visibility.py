from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.autonomous_initiative_loop import AutonomousInitiativeLoop
from core.orchestrator.mixins.output_formatter import OutputFormatterMixin


def test_emit_thought_stream_falls_back_to_thought_emitter(monkeypatch):
    emitter = SimpleNamespace(emit=MagicMock())
    monkeypatch.setattr("core.thought_stream.get_emitter", lambda: emitter)

    formatter = OutputFormatterMixin()
    formatter._emit_thought_stream("Mind wandering through loose threads.")

    emitter.emit.assert_called_once_with(
        "Autonomous Thought",
        "Mind wandering through loose threads.",
        level="info",
        category="Autonomy",
    )


@pytest.mark.asyncio
async def test_self_development_cycle_runs_scan_tests_and_proposal(monkeypatch):
    capability_engine = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                {
                    "ok": True,
                    "issues_found": 1,
                    "top_issues": [
                        {
                            "file": "core/example.py",
                            "message": "Function 'foo' is too long (88 lines).",
                        }
                    ],
                },
                {
                    "ok": False,
                    "error": "1 generated sandbox test failed",
                },
                {
                    "ok": True,
                    "proposal_path": "/tmp/evolution/proposal.md",
                },
            ]
        )
    )
    monkeypatch.setattr(
        "core.autonomous_initiative_loop.optional_service",
        lambda name, default=None: capability_engine if name == "capability_engine" else default,
    )

    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace(cognitive_engine=object()))
    emitted: list[tuple[str, str, str]] = []
    loop._emit_feed = lambda title, content, *, category: emitted.append((title, content, category))

    await loop._run_self_development_cycle()

    calls = capability_engine.execute.await_args_list
    assert [call.args[0] for call in calls] == ["auto_refactor", "test_generator", "self_evolution"]
    assert any("sandbox tests" in content.lower() for _, content, _ in emitted)
    assert any("proposal" in content.lower() or "saved to" in content.lower() for _, content, _ in emitted)
