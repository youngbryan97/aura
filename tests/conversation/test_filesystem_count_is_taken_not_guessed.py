"""A count of files on this disk is not a thing to estimate.

LIVE 2026-08-17: "count the .py files in core/introspection and tell me the
number" was answered "There are 3.py files in the core/introspection
directory." There are ten. Asked again with "use your tools and give me the
exact number", the answer was still 3 — and the log for that turn shows no tool
ran at all. The number came out of the model both times.

The runtime can take the count exactly, in microseconds, so a generated guess
was competing with a fact that was available the whole time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.conversation.filesystem_check import requested_filesystem_count

REPO = Path(__file__).resolve().parents[2]


def test_the_live_question_returns_the_real_count() -> None:
    truth = len(list((REPO / "core" / "introspection").glob("*.py")))
    result = requested_filesystem_count(
        "count the .py files in core/introspection and tell me the number"
    )

    assert result is not None
    assert result.count == truth
    assert result.count != 3, "the fabricated answer must not be reproducible"


@pytest.mark.parametrize(
    "question",
    [
        "how many python files are in the core/introspection folder?",
        "how many .py files are in core/introspection",
        "number of python files in core/introspection",
        # LIVE: this exact phrasing went unparsed and was answered "I don't
        # have file system access", while "are in" was answered exactly. Asking
        # the same thing a different way must not change what she can do.
        "how many python files live in core/introspection?",
        "how many python files do we have in core/introspection",
        "how many python files sit inside core/introspection",
    ],
)
def test_phrasings_all_reach_the_same_count(question: str) -> None:
    truth = len(list((REPO / "core" / "introspection").glob("*.py")))

    result = requested_filesystem_count(question)

    assert result is not None and result.count == truth


def test_counting_every_file_is_not_the_same_as_counting_python_files() -> None:
    everything = requested_filesystem_count("how many files are in core/introspection")
    python_only = requested_filesystem_count(
        "how many python files are in core/introspection"
    )

    assert everything is not None and python_only is not None
    assert everything.count >= python_only.count


# ── it must not answer questions about the rest of the machine ──────────────

@pytest.mark.parametrize(
    "question",
    [
        "how many files are in /etc",
        "how many files are in ../../../etc",
        "how many files are in ~/Documents",
    ],
)
def test_paths_outside_her_roots_are_declined(question: str) -> None:
    """pathlib treats `root / '/etc'` as `/etc`, which silently escapes."""
    assert requested_filesystem_count(question) is None


def test_a_missing_directory_is_named_not_counted_as_zero() -> None:
    """'There are 0 files' and 'that is not there' are different answers."""
    result = requested_filesystem_count("how many files are in core/nope_not_real")

    assert result is not None
    assert result.exists is False


def test_an_unrecognised_qualifier_is_declined() -> None:
    """'how many test files' must not silently become 'how many files'."""
    assert requested_filesystem_count("how many test files are in core") is None


@pytest.mark.parametrize(
    "question",
    ["what is the weather", "hey", "", "how are you doing"],
)
def test_unrelated_questions_are_ignored(question: str) -> None:
    assert requested_filesystem_count(question) is None


# ── the measured count is SERVED, not requested ──────────────────────────────
#
# The re-answer pass injected the real count into the context and asked the
# model to answer again from it. It did that, logged that it did it, and the
# model returned "There are 3 Python files" for the third time. Asking is still
# prompting; response_generation already says "evidence informs, it does not
# enforce."

from interface.routes.chat import _serve_measured_filesystem_count as serve


def test_a_contradicted_count_is_replaced_with_the_measured_one() -> None:
    truth = len(list((REPO / "core" / "introspection").glob("*.py")))

    out = str(serve(
        "count the .py files in core/introspection",
        "There are 3 Python files in the core/introspection folder.",
    ))

    assert str(truth) in out
    assert "3 Python files" not in out


def test_a_correct_count_leaves_her_wording_alone() -> None:
    truth = len(list((REPO / "core" / "introspection").glob("*.py")))
    original = f"There are {truth} Python files there, which is fewer than I expected."

    assert serve("count the .py files in core/introspection", original) == original


def test_the_served_answer_shows_what_was_counted() -> None:
    """A bare number invites the same doubt the guess did."""
    out = str(serve("count the .py files in core/introspection", "There are 3."))

    assert "self_evidence.py" in out


def test_a_missing_directory_is_reported_as_missing() -> None:
    out = str(serve("how many files are in core/nope_not_real", "There are 4 files."))

    assert "no directory" in out.lower()


def test_unrelated_replies_are_untouched() -> None:
    assert serve("how are you", "I am fine.") == "I am fine."


def test_a_path_outside_her_roots_is_untouched() -> None:
    original = "There are 56 files in /etc."

    assert serve("how many files are in /etc", original) == original


# ── reading a file she was asked to read ─────────────────────────────────────
#
# LIVE 2026-08-17: "read the file CONTRIBUTING.md and tell me the first rule it
# states" was answered "I don't have a clean grounded answer on that yet." The
# file is in the repo root and five registered skills can read it. The
# capability was never the problem — nothing executed.

from core.conversation.filesystem_check import requested_file_read


@pytest.mark.parametrize(
    "question",
    [
        "read the file CONTRIBUTING.md and tell me the first rule it states",
        "what does CONTRIBUTING.md say about tests?",
        "open CONTRIBUTING.md",
        "tell me about CONTRIBUTING.md",
    ],
)
def test_named_files_are_actually_read(question: str) -> None:
    read = requested_file_read(question)

    assert read is not None and read.exists
    assert read.text.strip()


def test_the_content_is_the_real_file() -> None:
    read = requested_file_read("open CONTRIBUTING.md")
    on_disk = (REPO / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert read is not None
    assert on_disk.startswith(read.text[:200])


def test_a_missing_file_is_reported_not_invented() -> None:
    read = requested_file_read("read the file NOPE_NOT_REAL.md")

    assert read is not None and read.exists is False


@pytest.mark.parametrize(
    "question",
    ["read /etc/passwd", "open ../../../etc/hosts", "read ~/.ssh/id_rsa"],
)
def test_files_outside_her_roots_are_refused(question: str) -> None:
    assert requested_file_read(question) is None


@pytest.mark.parametrize("question", ["how are you", "what is 2+2", ""])
def test_messages_naming_no_file_are_ignored(question: str) -> None:
    assert requested_file_read(question) is None


def test_a_large_file_is_bounded() -> None:
    from core.conversation.filesystem_check import READ_CHAR_BUDGET

    read = requested_file_read("open interface/routes/chat.py")

    assert read is not None
    assert len(read.text) <= READ_CHAR_BUDGET
    assert read.truncated is True
