"""Held back is not broken, in the repair loop and the organs that feed it.

LIVE 2026-09-16, one hour of the stream: "Fix generation or sandbox testing
failed: Fix generation failed" fourteen times, each a model held for a
foreground turn; a git probe lost to a loaded disk recorded a warning every
pulse, and the self-repair engine went looking for the bug behind it — at
the gateway's own governance refusal, which it then proposed to fix. Every
child the gateway killed said "timed out after 10.0 seconds", whichever bound
it had hit.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_a_killed_child_says_which_bound_it_hit() -> None:
    from core.runtime.subprocess_gateway import WorkBoundExpired

    exc = WorkBoundExpired(["git", "status"], 10.0, reason="wedged: no CPU progress for 10.0s at 0.1s of CPU")

    assert isinstance(exc, subprocess.TimeoutExpired)
    assert "timed out after 10.0 seconds" in str(exc)
    assert "no CPU progress" in str(exc)


def test_a_lost_git_probe_is_a_degradation_only_when_it_persists(monkeypatch) -> None:
    from core.soma import source_body as sb

    recorded: list[str] = []
    monkeypatch.setattr(sb, "record_degradation", lambda *a, **k: recorded.append(k.get("action", "")))

    class _Gateway:
        def run(self, *a, **k):
            raise subprocess.TimeoutExpired(["git"], 10.0)

    body = sb.SourceBodyAwareness.__new__(sb.SourceBodyAwareness)
    body._subprocess_gateway = _Gateway()
    body.source_root = ROOT
    body._git_available = None

    for _ in range(sb._GIT_PROBE_FAILURES_BEFORE_DEGRADED - 1):
        assert body._git("status") == (1, "")
    assert recorded == [], "a probe lost to a loaded disk is not yet a degradation"
    body._git("status")
    assert len(recorded) == 1 and "pulses running" in recorded[0]


def test_the_repair_loop_reports_a_deferral_as_a_deferral() -> None:
    repair = (ROOT / "core/self_modification/code_repair.py").read_text(encoding="utf-8")
    assert '"error": f"deferred: {deferred}"' in repair
    engine = (ROOT / "core/self_modification/self_modification_engine.py").read_text(encoding="utf-8")
    held = engine.index('if reason.startswith("deferred:"):')
    warned = engine.index('logger.warning("Fix generation or sandbox testing failed: %s", reason)')
    assert held < warned, "a deferral must be read before it can be called a failure"


def test_the_stabilizer_repairs_against_the_persons_question() -> None:
    source = (ROOT / "interface/routes/chat_reply_repair.py").read_text(encoding="utf-8")
    at = source.index('origin="api_stabilizer"')
    assert "visible_user_message=user_message" in source[at : at + 900]
