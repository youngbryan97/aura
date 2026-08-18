"""Asking what she would CHOOSE is not asking what she has.

Live 2026-08-18. "If I gave you 10 minutes of unsupervised compute, what would
you actually do with it?" was answered in 1.2 seconds with the registry: 74
entries, five category headings, a sketch of a five-step workflow, and a
closing note that she was not opening anything. The question was about what
she would choose, and choice is the one thing an inventory cannot express.

It matched on "actually do" — a marker added for "What can you actually do on
this computer right now?", where the same two words ask what is possible. CAN
asks the inventory; WOULD asks the will.

A hypothetical that names the inventory outright is still an inventory
question, so "if you had an hour, which of your tools would you reach for"
must keep working.
"""

from __future__ import annotations

import pytest

from interface.routes.chat_preflight import (
    _is_explicit_capability_inventory_request as asks_for_inventory,
)


@pytest.mark.parametrize(
    "message",
    [
        "if I gave you 10 minutes of unsupervised compute, what would you actually do with it?",
        "what would you do with a free afternoon?",
        "suppose nobody was watching, what would you actually do?",
        "imagine you had a whole day to yourself. what would you do?",
        "if you could work on anything, what would you pick?",
    ],
)
def test_a_question_about_choice_is_not_answered_with_a_catalogue(
    message: str,
) -> None:
    assert not asks_for_inventory(message)


@pytest.mark.parametrize(
    "message",
    [
        "what tools do you have?",
        "What can you actually do on this computer right now?",
        "what are your capabilities?",
        "list your tools",
        "what can you do on my computer",
        # A hypothetical that really does ask the inventory.
        "if you had an hour, which of your tools would you reach for?",
    ],
)
def test_a_real_inventory_question_still_gets_the_registry(message: str) -> None:
    assert asks_for_inventory(message)


def test_the_auxiliary_is_what_separates_them() -> None:
    """Same verb, opposite questions."""
    assert asks_for_inventory("what can you actually do on this machine?")
    assert not asks_for_inventory("what would you actually do on this machine?")
