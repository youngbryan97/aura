""""your source tree" is a reference, not a path.

LIVE, 2026-08-18. Asked to "count how many files are in your own source tree",
she answered:

    There is no directory at /Users/bryan/.aura/live-source/your, so there is
    nothing to count there.

The count pattern's path group is `[\\w./\\-]+`, which a pronoun satisfies
perfectly, so "your" was captured as the path. `_resolve` then did what it is
designed to do for a name it cannot find — compose the path anyway so the
caller can report it missing — and the result was a confident statement about
a directory nobody had mentioned. "my downloads" and "our repo" fail the same
way.

Worse, the question was answerable. She HAS a source tree, and it is the first
root this module already trusts.
"""

from __future__ import annotations

import pytest

from core.conversation.filesystem_check import requested_filesystem_count


def test_the_question_that_invented_a_directory_is_answered():
    counted = requested_filesystem_count("count how many files are in your own source tree")
    assert counted is not None
    assert counted.exists is True
    assert not counted.path.endswith("/your")
    assert counted.count > 0


def test_her_source_tree_is_countable_by_kind():
    counted = requested_filesystem_count("how many python files are in your source tree?")
    assert counted is not None
    assert counted.suffix == ".py"
    assert counted.exists is True


@pytest.mark.parametrize(
    "question",
    [
        "how many files are in my downloads",
        "how many files are in our repo folder",
        "how many files are in their directory",
        "how many files are in this folder",
    ],
)
def test_a_pronoun_that_resolves_to_nothing_is_declined(question):
    """Declining is right; inventing a path and reporting it missing is not."""
    counted = requested_filesystem_count(question)
    if counted is not None:
        # If it resolved at all, it must be to something real — never to a
        # composed path ending in the pronoun itself.
        assert counted.exists is True
        for pronoun in ("/my", "/our", "/their", "/this", "/your"):
            assert not counted.path.endswith(pronoun), counted.path


def test_named_directories_still_work():
    """The ordinary case must be untouched."""
    counted = requested_filesystem_count("how many python files are in core/introspection")
    assert counted is not None
    assert counted.exists is True
    assert counted.path.endswith("core/introspection")
    assert counted.suffix == ".py"


def test_a_missing_named_directory_is_still_reported_missing():
    """The invent-the-path tail exists for a reason and must survive."""
    counted = requested_filesystem_count("how many files are in core/definitely_not_here")
    assert counted is not None
    assert counted.exists is False


def test_an_absolute_path_is_still_refused():
    assert requested_filesystem_count("how many files are in /etc") is None
