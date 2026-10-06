"""Requests whose answer needs exact counting or indexing in a long list.

G05's pilot (2026-10-06) found her free answers already right on five-step
arithmetic: thirteen-digit products worked out in her reasoning, correct
with or without the program reader's evidence. A reader cannot add to an
answer the model already gets, so G05 needs requests where free decoding is
known to slip and an exact reader does not: how often a value occurs among
40 to 64 entries (64 is the most the value algebra holds), or which entry
sits at a selector deep in the list.

Every request is two steps in phrasings the reader was trained on ("count
how often 17 occurs in [...]", "select the item at selector 57 in [...]",
then one arithmetic step), so what is measured is whether the reading
reaches her answer, not whether the reader transfers. Split ``test``.
"""

from __future__ import annotations

import random
from typing import Final

from core.learning.semantic_g04_transfer_corpus import (
    _CONSUMED_PHRASES,
    _SCALAR_OPS,
    _Request,
    _scalar_continuation,
    _then_scaffold,
)
from core.learning.semantic_program_corpus import SemanticProgramExample
from core.learning.semantic_program_ir import MAX_SEMANTIC_SEQUENCE_ITEMS

__all__ = ["G05_LONG_SEQUENCE_CORPUS_KIND", "build_g05_long_sequence_corpus"]

G05_LONG_SEQUENCE_CORPUS_KIND: Final = "g05_long_sequence_v1"
_SHAPES: Final = ("count_of", "at")


def _long_request(rng: random.Random, shape: str) -> _Request:
    request = _Request(random.Random(rng.getrandbits(64)))
    inner = request.rng
    length = inner.randint(40, MAX_SEMANTIC_SEQUENCE_ITEMS)
    items = [inner.randint(1, 60) for _ in range(length)]
    if shape == "count_of":
        wanted = inner.randint(1, 60)
        items = [item if item != wanted else item % 60 + 1 for item in items]
        for position in inner.sample(range(length), inner.randint(3, 12)):
            items[position] = wanted
        previous = request.step("count_of", request.sequence(items), request.fixed(wanted))
    else:
        selector = inner.randint(length // 3, length - 1)
        previous = request.step("at", request.sequence(items), request.fixed(selector))
    _scalar_continuation(request, inner.choice(_SCALAR_OPS), previous)
    return request


def build_g05_long_sequence_corpus(
    *, seed: int, tasks: int
) -> tuple[SemanticProgramExample, ...]:
    """``tasks`` requests, counting and indexing in turn; deterministic in ``seed``."""
    if tasks < 1:
        raise ValueError("the corpus needs at least one task")
    rng = random.Random(seed)
    rows: list[SemanticProgramExample] = []
    seen: set[str] = set()
    sample = 0
    while len(rows) < tasks:
        sample += 1
        shape = _SHAPES[len(rows) % len(_SHAPES)]
        request = _long_request(rng, shape)
        _then_scaffold(request, _CONSUMED_PHRASES)
        example = request.example("long_sequence", shape, sample)
        if example.source_text not in seen:
            seen.add(example.source_text)
            rows.append(example)
    return tuple(rows)
