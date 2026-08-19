"""Which deterministic reader already answers this turn.

Several readers can compute an exact answer without the model: the filesystem
reader, arithmetic, text operations. When one of them owns a turn, the general
machinery around it must not staple a second, different reading beside the
answer the person asked for — that is how "read me the first line of
CONTRIBUTING.md" ended up carrying a brief about her own source code and then
failed a provenance check over a file it had read correctly.

The list of readers used to live inline in the chat route, so every new reader
was a chat-route edit that nobody would remember to make. Readers declare
themselves here instead, and everything downstream asks the registry.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

__all__ = [
    "Reader",
    "register_reader",
    "registered_readers",
    "owning_readers",
    "reader_owns",
    "another_reader_owns_this_turn",
]

_IMPORT_ERRORS = (ImportError, AttributeError, RuntimeError, TypeError, ValueError)


@dataclass(frozen=True)
class Reader:
    """One deterministic reader and the test for whether it owns a turn."""

    name: str
    module: str
    function: str
    answers: str

    def vocabulary(self) -> tuple[str, ...]:
        """The words this reader answers to, if its module publishes them.

        A reader that defines `capability_vocabulary()` derives it from the
        forms it actually implements, so self-knowledge tracks the code
        instead of a sentence someone wrote once.
        """
        try:
            fn = getattr(
                importlib.import_module(self.module), "capability_vocabulary", None
            )
            if fn is None:
                return ()
            return tuple(str(word) for word in fn())
        except _IMPORT_ERRORS:
            return ()

    def claims(self, text: str) -> bool:
        """Whether this reader produces an answer for this turn."""
        try:
            fn: Callable[[str], object] = getattr(
                importlib.import_module(self.module), self.function
            )
        except _IMPORT_ERRORS:
            return False
        try:
            return fn(text) is not None
        except _IMPORT_ERRORS:
            return False


_READERS: dict[str, Reader] = {}


def register_reader(
    name: str, module: str, function: str, answers: str
) -> Reader:
    """Declare a reader. `function(text)` returns None when it has no answer."""
    reader = Reader(name=name, module=module, function=function, answers=answers)
    _READERS[reader.name] = reader
    return reader


def registered_readers() -> tuple[Reader, ...]:
    """Every declared reader, in declaration order."""
    return tuple(_READERS.values())


def owning_readers(text: str) -> tuple[str, ...]:
    """Names of the readers that answer this turn."""
    message = str(text or "").strip()
    if not message:
        return ()
    return tuple(r.name for r in _READERS.values() if r.claims(message))


def reader_owns(text: str, name: str) -> bool:
    """Whether one named reader answers this turn."""
    message = str(text or "").strip()
    reader = _READERS.get(name)
    if not message or reader is None:
        return False
    return reader.claims(message)


def another_reader_owns_this_turn(text: str, *, excluding: str | None = None) -> bool:
    """Whether some reader other than `excluding` already answers this turn."""
    return any(name != excluding for name in owning_readers(text))


register_reader(
    "file_read",
    "core.conversation.filesystem_check",
    "requested_file_read",
    answers="a named file on disk, read and excerpted",
)
register_reader(
    "arithmetic",
    "core.conversation.arithmetic_check",
    "requested_arithmetic_result",
    answers="an arithmetic expression, evaluated",
)
register_reader(
    "text_operation",
    "core.conversation.computable_text",
    "computed_text_answer",
    answers="reversing, counting, sorting or testing a given string",
)
register_reader(
    "statistics",
    "core.conversation.computable_statistics",
    "computed_statistic",
    answers="a statistic with a closed form — a Wilson interval, a mean, a "
    "median, a standard deviation, a percentage",
)
