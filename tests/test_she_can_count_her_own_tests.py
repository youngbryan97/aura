""""How many test files do you have" is a fair question with one answer.

LIVE, 2026-08-18. Asked for two counts and told not to guess either, the
filesystem checker resolved "how many python files are in core/agency" exactly
and returned nothing at all for "how many test files do you have" — so half
the question fell back to the model, against a real answer of 2444.

The counting pattern needs a preposition and a path ("... in core/agency"),
and this shape has neither. It also names a KIND the suffix table does not
know, which is refused on purpose: an unrecognised qualifier would silently
become "all files" and answer a different question.

Both are right in general and wrong here, because the place is implied by the
kind. Her tests live in tests/ and her docs in docs/. Naming that is what
turns an ambiguous question into a determinate one, rather than widening the
qualifier rule and answering something else.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core.conversation.filesystem_check import requested_filesystem_count

ROOT = Path(__file__).resolve().parents[1]


def _actual(directory: str, suffix: str) -> int:
    return len([p for p in (ROOT / directory).iterdir() if p.is_file() and p.name.endswith(suffix)])


@pytest.mark.parametrize(
    "question",
    [
        "how many test files do you have",
        "how many tests do you have?",
        "count the test files",
        "number of test files",
        "how many specs do you have",
    ],
)
def test_her_tests_are_countable_however_it_is_asked(question):
    counted = requested_filesystem_count(question)
    assert counted is not None, question
    assert counted.exists is True
    assert counted.suffix == ".py"
    assert counted.count == _actual("tests", ".py")


def test_her_docs_are_countable_too():
    counted = requested_filesystem_count("how many docs do you have")
    assert counted is not None
    assert counted.suffix == ".md"
    assert counted.count == _actual("docs", ".md")


def test_a_named_path_still_wins():
    """The explicit form must be untouched by the implied-place shortcut."""
    counted = requested_filesystem_count("how many python files are in core/agency")
    assert counted is not None
    assert counted.path.endswith("core/agency")
    assert counted.count == _actual("core/agency", ".py")


def test_a_word_that_merely_starts_with_test_is_not_a_test_file():
    """"testimonials" must not be read as "test"."""
    assert requested_filesystem_count("how many testimonials do you have") is None


def test_an_unrecognised_qualifier_is_still_refused():
    """The deliberate refusal that protects against answering a different
    question has to survive: only kinds with a known home are answered."""
    assert requested_filesystem_count("how many config files are in core") is None


def test_the_counts_match_the_shell():
    """Against `ls`, so the number is checked outside this module's own logic."""
    counted = requested_filesystem_count("how many test files do you have")
    shell = subprocess.run(
        "ls tests/*.py | wc -l", shell=True, cwd=ROOT, capture_output=True, text=True, timeout=60
    )
    assert counted is not None
    assert counted.count == int(shell.stdout.strip())
