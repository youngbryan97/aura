""""Done" has to survive a stat().

LIVE 2026-08-18. "make a file on my desktop called aura-live-check.txt with
one sentence about what you are doing right now" was answered in 11 seconds
with:

    Done — the desktop steps completed and their effects verified.

No file existed. The skill had "completed" in 2ms, which is not enough time to
write anything, and the bridge that exists to stop exactly this accepted it.

Two holes, both in the same function:

  * steps_completed was read only far enough to check it was an integer, so a
    task that completed none of its steps still passed;
  * a file receipt states "path=X;bytes=N" — a claim anyone can check — and
    the bridge took the executor's word for it, having already established
    that it must not take the executor's word for ok=True.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from interface.routes.chat_desktop_objective import _verified_desktop_task_result


def _result(receipt: dict, *, requested: int = 1, completed: int = 1) -> dict:
    return {
        "ok": True,
        "steps_requested": requested,
        "steps_completed": completed,
        "receipts": [receipt],
    }


@pytest.fixture
def written(tmp_path) -> Path:
    target = tmp_path / "note.txt"
    target.write_text("hello", encoding="utf-8")
    return target


def test_a_truthful_file_receipt_verifies(written: Path) -> None:
    ok, reason = _verified_desktop_task_result(
        _result({
            "ok": True,
            "effect_verified": True,
            "effect_evidence": f"path={written};bytes=5;sha256={'a' * 64}",
        })
    )

    assert ok is True and reason == "verified"


def test_a_file_that_is_not_there_is_not_a_verified_effect() -> None:
    ok, reason = _verified_desktop_task_result(
        _result({
            "ok": True,
            "effect_verified": True,
            "effect_evidence": "path=/tmp/aura-does-not-exist-95731.txt;bytes=5",
        })
    )

    assert ok is False
    assert "not_on_disk" in reason


def test_a_size_that_does_not_match_is_not_a_verified_effect(written: Path) -> None:
    ok, reason = _verified_desktop_task_result(
        _result({
            "ok": True,
            "effect_verified": True,
            "effect_evidence": f"path={written};bytes=999",
        })
    )

    assert ok is False
    assert "size_does_not_match" in reason


def test_completing_no_steps_is_not_completion() -> None:
    ok, reason = _verified_desktop_task_result(
        _result(
            {"ok": True, "effect_verified": True, "effect_evidence": "did a thing"},
            completed=0,
        )
    )

    assert ok is False
    assert reason.startswith("no_steps_completed")


def test_a_non_critical_step_may_still_fail() -> None:
    """Partial completion is by design; completing NOTHING is not."""
    ok, reason = _verified_desktop_task_result(
        {
            "ok": True,
            "steps_requested": 2,
            "steps_completed": 1,
            "receipts": [
                {"ok": True, "effect_verified": True, "effect_evidence": "one"},
                {"ok": False, "critical": False, "effect_evidence": "timed out"},
            ],
        }
    )

    assert ok is True and reason == "verified"


def test_a_non_file_effect_is_still_accepted_on_its_receipt() -> None:
    """The disk check applies to claims that name a path, and only those."""
    ok, reason = _verified_desktop_task_result(
        _result({
            "ok": True,
            "effect_verified": True,
            "effect_evidence": "clipboard now holds BUILD-42",
        })
    )

    assert ok is True and reason == "verified"


def test_a_path_nothing_could_have_written_does_not_verify() -> None:
    """A claim naming an impossible path is a claim about no file."""
    ok, reason = _verified_desktop_task_result(
        _result({
            "ok": True,
            "effect_verified": True,
            "effect_evidence": "path=\x00bad;bytes=5",
        })
    )

    assert ok is False
    assert "not_on_disk" in reason


def test_an_unstatable_path_leaves_the_receipt_standing(monkeypatch) -> None:
    """Failing to look is not evidence of absence.

    The executor already read the file back and hashed it. When this bridge
    cannot stat the path at all — a permission error, a vanished mount — it
    has disproved nothing, and refusing the claim would tell the person their
    completed task might not have happened.
    """
    import interface.routes.chat_desktop_objective as module

    def _refuse(self):
        raise PermissionError("no access")

    monkeypatch.setattr(module.Path, "is_file", _refuse)

    ok, _reason = _verified_desktop_task_result(
        _result({
            "ok": True,
            "effect_verified": True,
            "effect_evidence": "path=/tmp/whatever.txt;bytes=5",
        })
    )

    assert ok is True
