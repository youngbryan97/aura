"""A tree is the tree, and what is in the files is countable too.

LIVE 2026-08-18: "estimate how many characters are in your own source tree.
show your method."

    There are about 300k non-blank lines of code across ~1500 files in my
    source tree. A rough average line length is around 80 characters, so
    that's about 24 million characters total.

The tree holds 6,352 files and 2.3M lines — every figure out by roughly four
times. The method was sound and the inputs were guesses, and each input is one
walk away.

Worse, the counter that existed answered "how many python files are in your
source tree?" with 12: the files sitting directly in the repository root. A
confidently wrong number is worse than an estimate, because nothing about it
looks uncertain.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.conversation.filesystem_check import (
    _UNCOUNTED_DIRS,
    requested_filesystem_count,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _source_files() -> list[Path]:
    return [
        path
        for path in REPO_ROOT.rglob("*.py")
        if not set(path.parts) & set(_UNCOUNTED_DIRS)
    ]


def test_a_source_tree_question_walks_the_tree() -> None:
    counted = requested_filesystem_count("how many python files are in your source tree?")

    assert counted is not None
    assert counted.recursive is True
    assert counted.count == len(_source_files())


def test_characters_are_counted_from_the_files() -> None:
    counted = requested_filesystem_count(
        "how many characters are in your own source tree?"
    )
    expected = sum(path.stat().st_size for path in _source_files())

    assert counted is not None
    assert counted.measure == "characters"
    assert counted.count == expected


def test_lines_are_counted_from_the_files() -> None:
    counted = requested_filesystem_count(
        "how many lines of code are in your source tree?"
    )
    expected = 0
    for path in _source_files():
        with path.open("rb") as handle:
            expected += sum(1 for _ in handle)

    assert counted is not None
    assert counted.measure == "lines"
    assert counted.count == expected


def test_one_directory_is_still_one_directory() -> None:
    """Widening the tree case must not turn every count into a whole-tree walk."""
    counted = requested_filesystem_count("how many python files are in core/runtime?")
    expected = len(sorted((REPO_ROOT / "core" / "runtime").glob("*.py")))

    assert counted is not None
    assert counted.recursive is False
    assert counted.count == expected


def test_a_source_question_does_not_count_model_weights() -> None:
    """"Characters in your source tree" is about source.

    Counting every byte under the repository returned 196 billion characters,
    having swept model weights, databases and captured artifacts into an
    answer about code.
    """
    counted = requested_filesystem_count(
        "how many characters are in your own source tree?"
    )

    assert counted is not None
    assert counted.suffix == ".py"
    assert counted.count < 1_000_000_000


@pytest.mark.parametrize(
    "question",
    ["how are you?", "what is 2 + 2", "tell me about python"],
)
def test_an_ordinary_turn_asks_for_no_count(question: str) -> None:
    assert requested_filesystem_count(question) is None
