"""One question gets one reading.

LIVE 2026-08-19: "read me the first line of CONTRIBUTING.md" returned:

    I couldn't get to an answer I'd stand behind on that one, and I won't send
    you a thinner one and pass it off as the real thing.

The log shows everything before that worked. The grounding channel took the
reading — "1 reading(s): file you were asked about" — and the model answered
in 61 characters. Then the turn was classified as a question about her own
CODE as well, a source-evidence brief was attached beside the file, the
provenance corrector required the reply to cite
core/memory/associative_entity_memory.py, that correction failed its
authorship proof, and the turn died.

At the production margin the semantic router scored even "read notes.txt on
my desktop" as a question about her own source. A file on someone's Desktop is
not her implementation, and a request to read a named file already has the
reading it needs.

Asking FOR her code still routes to the source brief. This only stops a second
and different reading being bolted onto a turn that already has the right one.
"""

from __future__ import annotations

import pytest

from interface.routes.chat import _turn_may_concern_own_source


@pytest.mark.parametrize(
    "message",
    [
        "read me the first line of CONTRIBUTING.md",
        "read notes.txt on my desktop",
        "what does CONTRIBUTING.md say about tests?",
        "open README.md and tell me what it says",
    ],
)
def test_reading_a_named_file_is_not_a_question_about_her_code(message: str) -> None:
    assert not _turn_may_concern_own_source(message)


@pytest.mark.parametrize(
    "message",
    [
        "show me your code for the write gateway",
        "show me the actual source of your lock ordering",
        "paste your own implementation of the file write gateway",
    ],
)
def test_asking_for_her_code_still_reaches_the_source(message: str) -> None:
    assert _turn_may_concern_own_source(message)


def test_the_file_reader_is_what_decides() -> None:
    """The rule is "the file reader already claims it", not a phrase list."""
    from core.conversation.filesystem_check import requested_file_read

    assert requested_file_read("read me the first line of CONTRIBUTING.md") is not None
    assert requested_file_read("show me your code for the write gateway") is None
