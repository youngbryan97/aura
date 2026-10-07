"""G05's long-list requests need exact counting or indexing, in phrasings the reader knows."""

from __future__ import annotations

from core.learning.semantic_g05_long_sequence_corpus import build_g05_long_sequence_corpus
from core.learning.semantic_program_ir import MAX_SEMANTIC_SEQUENCE_ITEMS


def test_each_request_counts_or_indexes_in_a_long_list() -> None:
    rows = build_g05_long_sequence_corpus(seed=11, tasks=40)
    assert len(rows) == 40 and len({row.source_text for row in rows}) == 40
    shapes = {row.construction_id.split(":")[1] for row in rows}
    assert shapes == {"count_of", "at"}
    for row in rows:
        first = row.instructions[0].instruction
        sequence = row.inputs[first.args[0]]
        assert 40 <= len(sequence) <= MAX_SEMANTIC_SEQUENCE_ITEMS
        if first.op == "count_of":
            assert 3 <= sequence.count(row.inputs[first.args[1]]) <= 12
        else:
            assert row.inputs[first.args[1]] >= len(sequence) // 3
        assert isinstance(row.program.run(row.inputs), int)


def test_it_uses_only_consumed_phrasings() -> None:
    for row in build_g05_long_sequence_corpus(seed=3, tasks=10):
        text = row.source_text
        assert text.startswith("First, ") and text.endswith(" Return the integer result.")
        assert "count how often" in text or "select the item at selector" in text


def test_the_same_seed_gives_the_same_requests() -> None:
    first = build_g05_long_sequence_corpus(seed=5, tasks=6)
    assert [r.source_text for r in first] == [r.source_text for r in build_g05_long_sequence_corpus(seed=5, tasks=6)]
