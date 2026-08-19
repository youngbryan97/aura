"""A new deterministic reader reaches the chat route by declaring itself."""

from __future__ import annotations

import pytest

from core.conversation import turn_ownership
from core.conversation.turn_ownership import (
    another_reader_owns_this_turn,
    owning_readers,
    reader_owns,
    register_reader,
    registered_readers,
)


def test_every_declared_reader_resolves_to_a_real_callable():
    """A reader naming a function that does not exist claims nothing, silently."""
    for reader in registered_readers():
        module = pytest.importorskip(reader.module)
        fn = getattr(module, reader.function, None)
        assert callable(fn), f"{reader.name} names {reader.module}.{reader.function}"
        assert reader.answers.strip(), f"{reader.name} does not say what it answers"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("what is 12 * 12", "arithmetic"),
        ("reverse the word stressed", "text_operation"),
        ("read me the first line of CONTRIBUTING.md", "file_read"),
    ],
)
def test_each_reader_owns_the_turns_it_answers(message: str, expected: str):
    assert expected in owning_readers(message)
    assert reader_owns(message, expected)


def test_a_turn_no_reader_answers_is_owned_by_nobody():
    assert owning_readers("what do you think about consciousness") == ()
    assert not another_reader_owns_this_turn("what do you think about consciousness")


def test_the_owning_reader_is_not_another_reader():
    """Excluding itself is how a reader asks whether to stand down."""
    assert not another_reader_owns_this_turn("what is 12 * 12", excluding="arithmetic")


def test_a_newly_declared_reader_is_seen_without_editing_the_chat_route(monkeypatch):
    """The generality claim, checked: declaration is the only wiring step."""
    from interface.routes.chat import _another_reader_owns_this_turn

    message = "convert 4 furlongs to metres"
    assert not _another_reader_owns_this_turn(message)

    monkeypatch.setitem(
        turn_ownership._READERS,
        "unit_conversion",
        turn_ownership.Reader(
            name="unit_conversion",
            module="core.conversation.turn_ownership",
            function="_probe_reader_for_tests",
            answers="a unit conversion",
        ),
    )
    monkeypatch.setattr(
        turn_ownership,
        "_probe_reader_for_tests",
        lambda text: 804.672 if "furlong" in text else None,
        raising=False,
    )

    assert _another_reader_owns_this_turn(message)
