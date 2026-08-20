"""A path, a URL and a UUID name something; their characters are not words.

Twice on 2026-08-19 a matcher read meaning out of an opaque identifier:

* "there's a python project at /private/tmp/claude-501/-Users-bryan--aura-
  live-source/7a6cdc9e-da7f-47f7-8c38-.../ledger ... work out why" was
  answered "30." — the arithmetic matcher found "7-8" inside the UUID
  `47f7-8c38`, and "work out" satisfied its intent gate.
* The same path routed a request to debug a Python file into the
  browser-dialogue skill for talking to another AI, because the directory is
  called `claude-501`. The person got "I routed the Claude conversation
  through the governed web_interlocutor skill ... observed 0/8 turns."

Both matchers were correct about the strings they saw. Neither could tell that
those characters belonged to an address the person pasted rather than
something they said.
"""

from __future__ import annotations

import pytest

from core.intent.opaque_spans import without_opaque_spans

REPO_PATH = (
    "/private/tmp/claude-501/-Users-bryan--aura-live-source/"
    "7a6cdc9e-da7f-47f7-8c38-8cfadf95a75e/scratchpad/ledger/ledger/accounts.py"
)


def test_a_product_name_inside_a_path_is_not_a_mention():
    stripped = without_opaque_spans(f"now read {REPO_PATH} - which line is wrong?")
    assert "claude" not in stripped.lower()
    assert "which line is wrong" in stripped


def test_a_product_named_out_loud_still_survives():
    """The matcher must keep working for what it was built for."""
    assert "claude.ai" in without_opaque_spans("go talk to claude.ai about fusion")
    assert "Claude" in without_opaque_spans("what do you think about Claude?")


@pytest.mark.parametrize(
    "address",
    [
        "https://example.com/a/b?x=1",
        "/private/tmp/claude-501/x/y.py",
        "~/Documents/report.md",
        "core/config.py",
        "7a6cdc9e-da7f-47f7-8c38-8cfadf95a75e",
    ],
)
def test_every_shape_of_address_is_removed(address: str):
    assert address not in without_opaque_spans(f"look at {address} please")


def test_ordinary_sentences_are_untouched():
    for sentence in (
        "what is 7919 * 6367?",
        "how are you feeling today",
        "the ratio was 16:9 and it looked wrong",
    ):
        assert without_opaque_spans(sentence) == sentence


def test_words_either_side_of_an_address_do_not_fuse():
    """Replacing with nothing would invent a word nobody wrote."""
    assert "read and" not in without_opaque_spans("read /tmp/a/b.py and stop")
    assert "read" in without_opaque_spans("read /tmp/a/b.py and stop")


def test_the_live_misroute_is_gone():
    from interface.routes.chat_capability_inventory import (
        _looks_like_web_interlocutor_execution_request as claims_it,
    )

    assert not claims_it(f"now read {REPO_PATH} - which line is wrong?")
    assert claims_it("go talk to claude.ai and ask it about fusion")


def test_arithmetic_is_also_immune_to_identifiers():
    """The other half of the same class, guarded in its own matcher."""
    from core.conversation.arithmetic_check import requested_arithmetic_result

    assert requested_arithmetic_result(f"read {REPO_PATH} and work out why") is None
    assert requested_arithmetic_result("what is 7919 * 6367?") == 7919 * 6367
