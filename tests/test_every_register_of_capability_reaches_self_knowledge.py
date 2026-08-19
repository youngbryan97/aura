"""What she can do and what she knows she can do are the same set.

Self-knowledge was built from the skill registry alone. Deterministic readers
answer turns without being skills, so "can you reverse a string for me" — a
thing she does exactly, in one line, without the model — was answered
"nothing in the capability registry matches the words in this question".

The gap was structural rather than a missing entry, so the test is structural
too: every register of capability must reach the lexicon, and a capability she
demonstrably HAS must be one she can find.
"""

from __future__ import annotations

import pytest

from core.conversation.turn_ownership import registered_readers
from core.self.capability_lexicon import capabilities_named_in, capability_status_block
from core.self.capability_sources import all_capabilities, registered_sources


def test_more_than_one_register_feeds_self_knowledge():
    assert len(registered_sources()) >= 2


def test_every_deterministic_reader_is_a_capability_she_knows_about():
    known = all_capabilities()
    for reader in registered_readers():
        assert reader.name in known, f"{reader.name} answers turns but is not declared"
        assert known[reader.name].description.strip()


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("can you reverse a string for me?", "text_operation"),
        ("can you count the letters in a word?", "text_operation"),
        ("are you able to do arithmetic exactly?", "arithmetic"),
        ("can you compute a greatest common divisor?", "arithmetic"),
        ("can you read a file on my disk?", "file_read"),
    ],
)
def test_a_capability_she_has_is_one_she_can_find(question: str, expected: str):
    found = [mention.skill for mention in capabilities_named_in(question)]
    assert expected in found, f"{question!r} found {found}"


@pytest.mark.parametrize(
    "question",
    [
        "can you help me think about this?",
        "what do you make of all that?",
    ],
)
def test_an_open_question_still_names_nothing(question: str):
    """The bar has to stay somewhere: matching everything is not knowing."""
    assert capabilities_named_in(question) == ()


def test_the_status_block_reports_a_reader_it_found():
    block = capability_status_block("can you reverse a string for me?")

    assert "text_operation" in block
    assert "registered and enabled" in block


def test_the_reader_vocabulary_is_derived_not_written():
    """A description typed by hand is a second copy of the truth.

    Each reader's vocabulary comes from the forms it implements, so a form
    added tomorrow is a capability she knows about tomorrow.
    """
    from core.conversation.computable_text import TEXT_FORMS, capability_vocabulary

    vocabulary = " ".join(capability_vocabulary()).lower()
    for form in TEXT_FORMS:
        assert form.name.replace("_", " ") in vocabulary
