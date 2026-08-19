""""Count X in DIR and write it to FILE" has to find DIR.

LIVE 2026-08-18: "count the python files in core/runtime and write the number
into a file called aura-report.md on my desktop" answered "216 .py files" —
correct — and wrote no file at all. Half the request was silently dropped, and
the reply read as complete.

A directory has no extension, so the path extractor could not see one: the
only path it found was the DESTINATION. With no source, either the read was
skipped and the file got composed filler, or — worse, on a neighbouring
phrasing — the destination itself was taken as the directory to read.

Existence on disk is what qualifies a token as a directory, not its shape, so
"docs" counts as readily as "core/runtime".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.skills.desktop_task import DesktopTaskSkill

REPO_ROOT = Path(__file__).resolve().parents[1]


def _plan(objective: str):
    return DesktopTaskSkill()._derive_steps_from_objective(objective, {})


def test_the_source_directory_is_read_before_the_number_is_written() -> None:
    steps = _plan(
        "count the python files in core/runtime and write the number into a "
        "file called aura-report.md on my desktop"
    )

    assert [step.action for step in steps] == ["list_directory", "write_text_file"]
    read = json.loads(steps[0].target)
    assert read["path"] == "core/runtime"


def test_the_written_number_comes_from_the_read() -> None:
    """Composing the number here would be the whole defect again."""
    steps = _plan(
        "count the python files in core/runtime and write the number into a "
        "file called aura-report.md on my desktop"
    )
    written = json.loads(steps[-1].target)

    assert "{{last.result.count}}" in written["content"]
    assert written["path"].endswith("aura-report.md")


def test_a_kind_word_is_as_specific_as_an_extension() -> None:
    """"python files" must not list every file in the directory."""
    steps = _plan(
        "count the python files in core/runtime and write the number into "
        "aura-report.md on my desktop"
    )

    assert json.loads(steps[0].target)["pattern"] == "*.py"


def test_a_single_segment_directory_is_found() -> None:
    steps = _plan(
        "count the markdown files in docs and write the total to notes.txt on my desktop"
    )

    assert [step.action for step in steps] == ["list_directory", "write_text_file"]
    read = json.loads(steps[0].target)
    assert read["path"] == "docs"
    assert read["pattern"] == "*.md"


def test_the_destination_is_never_read_as_the_source() -> None:
    steps = _plan(
        "how many python files are in core/runtime? write the answer to notes.txt on my desktop"
    )
    reads = [step for step in steps if step.action == "list_directory"]

    assert reads, "no directory read planned"
    assert "notes.txt" not in json.loads(reads[0].target)["path"]


def test_a_plain_write_plans_no_read() -> None:
    steps = _plan("write a haiku to a file called poem.txt on my desktop")

    assert [step.action for step in steps] == ["write_text_file"]


@pytest.mark.parametrize("name", ["core/runtime", "docs"])
def test_the_named_directories_actually_exist(name: str) -> None:
    """The rule is existence, so the fixtures have to be real."""
    assert (REPO_ROOT / name).is_dir()
