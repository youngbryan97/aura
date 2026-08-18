"""Adding to a file must not replace it.

Live 2026-08-18: `append a line saying "line two" to aura-test-note.txt on my
desktop` planned

    write_text_file  {"path": ..., "content": "line two\n", "overwrite": true}

against a file holding "hello from aura", and the reply said "the file now
contains both lines". The write verifies its own digest, so had it dispatched
it would have reported a clean success over the destroyed line.

These tests fail on the plan and on the effect, because either one alone can
be right while the other loses the content.
"""

from __future__ import annotations

import json

import pytest

from core.skills.computer_use import ComputerUseSkill
from core.skills.desktop_task import DesktopTaskSkill
from core.skills.file_modification_intent import (
    MODIFICATION_MODES,
    requested_file_modification,
)


@pytest.fixture
def writable(tmp_path, monkeypatch):
    """A real directory the write guard allows.

    The allowlist confines writes to Desktop/Documents and has its own tests;
    these tests are about what the write DOES to existing content, so the root
    is redirected rather than the guard relaxed.
    """
    monkeypatch.setattr(
        ComputerUseSkill, "_allowed_desktop_roots", lambda self: [tmp_path]
    )
    return tmp_path


def _payload(objective: str) -> dict:
    steps = DesktopTaskSkill()._derive_steps_from_objective(objective, {})
    writes = [s for s in steps if s.action == "write_text_file"]
    assert writes, f"no write step planned for {objective!r}"
    return json.loads(writes[-1].target)


def test_an_add_request_never_plans_a_replacing_write() -> None:
    for objective in (
        'append a line saying "line two" to aura-test-note.txt on my desktop',
        'add the line "line two" to the end of notes.txt on my desktop',
        "add a footer to the bottom of report.md on my desktop",
    ):
        payload = _payload(objective)

        assert payload.get("append") is True, f"{objective!r} -> {payload}"
        assert not payload.get("overwrite"), (
            f"{objective!r} plans to replace the file: {payload}"
        )


def test_a_create_request_still_replaces() -> None:
    payload = _payload(
        "create a file called aura-test-note.txt on my desktop containing hello"
    )

    assert payload.get("overwrite") is True
    assert not payload.get("append")


def test_appending_keeps_what_the_file_held(writable) -> None:
    target = writable / "note.txt"
    target.write_text("hello from aura\n", encoding="utf-8")

    result = ComputerUseSkill()._write_text_file(
        json.dumps({"path": str(target), "content": "line two\n", "append": True})
    )

    assert result["ok"] is True
    assert result["appended"] is True
    assert target.read_text(encoding="utf-8") == "hello from aura\nline two\n"


def test_prepending_keeps_what_the_file_held(writable) -> None:
    target = writable / "note.txt"
    target.write_text("hello from aura\n", encoding="utf-8")

    result = ComputerUseSkill()._write_text_file(
        json.dumps({"path": str(target), "content": "TITLE", "prepend": True})
    )

    assert result["ok"] is True
    assert target.read_text(encoding="utf-8") == "TITLE\nhello from aura\n"


def test_adding_to_a_missing_file_creates_it(writable) -> None:
    target = writable / "absent.txt"

    result = ComputerUseSkill()._write_text_file(
        json.dumps({"path": str(target), "content": "first\n", "append": True})
    )

    assert result["ok"] is True
    assert target.read_text(encoding="utf-8") == "first\n"


def test_every_declared_mode_is_one_the_executor_implements(writable) -> None:
    """A mode the planner can emit and the executor ignores is a silent no-op."""
    for mode in MODIFICATION_MODES:
        target = writable / f"{mode}.txt"
        target.write_text("prior\n", encoding="utf-8")

        ComputerUseSkill()._write_text_file(
            json.dumps({"path": str(target), "content": "added\n", mode: True})
        )

        assert "prior" in target.read_text(encoding="utf-8"), (
            f"mode {mode!r} is planned but not implemented: content was lost"
        )


def test_the_verb_decides_not_the_path() -> None:
    """Same file, opposite meanings."""
    assert requested_file_modification("add a line to notes.txt") is not None
    assert requested_file_modification("create notes.txt with a line") is None
