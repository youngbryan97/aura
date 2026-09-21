"""Naming a place to read is the authorisation to read it.

LIVE, 2026-08-27: "docs and source are at <path>. Read it, then actually use
it" was refused with "Access denied: path resolves outside workspace", for a
directory in the person's own sentence.

`diagnose_repo` may already be aimed at any directory somebody names — a
capability this tree gained precisely because nothing could look at a project a
person pointed at. This is the same rule for the same reason, kept narrow:
reading only, and only a path that appears in what the person actually typed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.skills.file_operation import (
    _READING_ACTIONS,
    _the_person_named_this_path,
)


def test_writing_is_not_a_reading_action() -> None:
    """Naming a place to read is not asking for it to be changed."""
    for action in ("write", "append", "delete", "move", "copy", "patch"):
        assert action not in _READING_ACTIONS
    for action in ("read", "list", "exists"):
        assert action in _READING_ACTIONS


def test_a_path_nobody_typed_is_not_authorised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "core.conversation.session_scope.current_user_question",
        lambda: "what is the capital of Peru",
        raising=False,
    )
    assert _the_person_named_this_path("/etc/shadow") is False


def test_a_path_in_the_persons_own_words_is_authorised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "core.conversation.session_scope.current_user_question",
        lambda: f"docs and source are at {tmp_path}. Read it, then use it.",
        raising=False,
    )
    assert _the_person_named_this_path(str(tmp_path)) is True


def test_the_authorisation_is_literal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sibling the person did not name is not covered by one they did."""
    named = tmp_path / "named"
    named.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setattr(
        "core.conversation.session_scope.current_user_question",
        lambda: f"look at {named}",
        raising=False,
    )
    assert _the_person_named_this_path(str(named)) is True
    assert _the_person_named_this_path(str(other)) is False
    assert _the_person_named_this_path(str(tmp_path.parent)) is False


def test_nothing_said_authorises_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "core.conversation.session_scope.current_user_question", lambda: "", raising=False
    )
    assert _the_person_named_this_path("/tmp") is False
    assert _the_person_named_this_path("") is False


def test_a_symlinked_form_of_the_named_path_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """They may type the path that resolves to the one being opened."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        os.symlink(real, link)
    except (OSError, NotImplementedError):  # pragma: no cover - platform without symlinks
        pytest.skip("symlinks unavailable")
    monkeypatch.setattr(
        "core.conversation.session_scope.current_user_question",
        lambda: f"read {link}",
        raising=False,
    )
    assert _the_person_named_this_path(str(real)) is True


def test_a_parent_of_a_named_path_is_not_authorised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming a place does not authorise everything above it.

    Testing whether the text appeared in the sentence authorised every PARENT
    of a named path, because a parent is a prefix of it — so naming anything at
    all authorised "/". Caught by this file before it ran anywhere.
    """
    named = tmp_path / "project"
    named.mkdir()
    monkeypatch.setattr(
        "core.conversation.session_scope.current_user_question",
        lambda: f"the project is at {named}",
        raising=False,
    )
    assert _the_person_named_this_path(str(named)) is True
    assert _the_person_named_this_path(str(tmp_path)) is False
    assert _the_person_named_this_path("/") is False
    assert _the_person_named_this_path(str(Path.home())) is False


def test_what_is_inside_a_named_directory_is_authorised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """"Read the project at <dir>" covers the files in it."""
    named = tmp_path / "project"
    named.mkdir()
    (named / "lib.py").write_text("x = 1\n")
    monkeypatch.setattr(
        "core.conversation.session_scope.current_user_question",
        lambda: f"docs and source are at {named}",
        raising=False,
    )
    assert _the_person_named_this_path(str(named / "lib.py")) is True


def test_a_read_honours_the_range_it_was_asked_for(tmp_path: Path) -> None:
    """LIVE 2026-09-21: `start_line`/`end_line` were declared for 'patch' and
    ignored by 'read', so a model asking for lines 7-14 to get past a
    truncated whole-file read got the same truncated whole file back, twice,
    and concluded the section was empty."""
    import asyncio

    from core.skills.file_operation import FileOperationSkill

    (tmp_path / "rules.md").write_text(
        "\n".join(f"line {n}" for n in range(1, 21)) + "\n", encoding="utf-8"
    )
    skill = FileOperationSkill()
    skill.root_dir = str(tmp_path.resolve())

    def read(**extra):
        return asyncio.run(
            skill.execute(
                {"action": "read", "path": "rules.md", **extra},
                context={"origin": "unit_test"},
            )
        )

    whole = read()
    assert whole["ok"] and whole["content"].count("\n") == 20
    assert "lines" not in whole

    part = read(start_line=7, end_line=9)
    assert part["ok"]
    assert part["content"] == "0007: line 7\n0008: line 8\n0009: line 9\n"
    assert part["lines"] == {"from": 7, "to": 9, "of": 20}

    tail = read(start_line=18)
    assert tail["content"].startswith("0018: ") and tail["content"].count("\n") == 3

    past_the_end = read(start_line=30, end_line=40)
    assert past_the_end["ok"] and past_the_end["content"] == ""
    assert "no lines in that range" in past_the_end["note"]
