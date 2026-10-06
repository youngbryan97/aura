"""G04's fresh strata each differ from what was consumed in the way they claim."""

from __future__ import annotations

import pytest

from core.learning.semantic_g04_transfer_corpus import (
    G04_STRATA,
    build_g04_transfer_corpus,
    novelty_signatures,
)

_SEQUENCE = {"at", "count_of"}


@pytest.fixture(scope="module")
def strata():
    return build_g04_transfer_corpus(seed=20261006, tasks_per_stratum=48)


def test_every_stratum_is_full_fresh_and_correct(strata) -> None:
    assert tuple(strata) == G04_STRATA
    texts = set()
    for rows in strata.values():
        assert len(rows) == 48
        for row in rows:
            assert row.split == "test"
            texts.add(row.source_text)
            answer = row.program.run(row.inputs)
            assert isinstance(answer, int) and answer >= 1, row.source_text
    assert len(texts) == 4 * 48


def test_the_same_seed_gives_the_same_requests() -> None:
    first = build_g04_transfer_corpus(seed=5, tasks_per_stratum=8)
    again = build_g04_transfer_corpus(seed=5, tasks_per_stratum=8)
    other = build_g04_transfer_corpus(seed=6, tasks_per_stratum=8)
    assert [r.source_text for rows in first.values() for r in rows] == [
        r.source_text for rows in again.values() for r in rows
    ]
    assert first["depth"][0].source_text != other["depth"][0].source_text


def test_each_input_is_said_once_and_no_two_scalars_are_equal(strata) -> None:
    for rows in strata.values():
        for row in rows:
            scalars = [value for value in row.inputs if isinstance(value, int)]
            assert len(scalars) == len(set(scalars)), row.source_text
            for value, span in zip(row.inputs, row.input_spans, strict=True):
                said = row.source_text[span.start:span.end]
                expected = (
                    "[" + ", ".join(map(str, value)) + "]" if isinstance(value, tuple) else str(value)
                )
                assert said == expected


def test_vocabulary_says_each_operation_in_words_nothing_consumed_has(strata) -> None:
    signatures = novelty_signatures("vocabulary")
    for row in strata["vocabulary"]:
        starts = [item.operation_span.start for item in row.instructions]
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else len(row.source_text)
            phrase = row.source_text[start:end].lower()
            assert any(sig in phrase for sig in signatures), (row.source_text, phrase)
        for consumed in ("add ", "subtract", "multiply", "divide", "count how often", "select the item"):
            assert consumed not in row.source_text.lower(), row.source_text


def test_construction_uses_a_scaffold_nothing_consumed_has(strata) -> None:
    signatures = novelty_signatures("construction")
    for row in strata["construction"]:
        assert any(sig in row.source_text.lower() for sig in signatures), row.source_text
    frames = {row.construction_id for row in strata["construction"]}
    assert len(frames) == 4


def test_depth_goes_past_every_consumed_program(strata) -> None:
    depths = {len(row.instructions) for row in strata["depth"]}
    assert depths == {6, 7}


def test_family_gives_a_sequence_operation_a_computed_argument(strata) -> None:
    for row in strata["family"]:
        n_inputs = len(row.inputs)
        assert any(
            item.instruction.op in _SEQUENCE
            and any(argument >= n_inputs for argument in item.instruction.args)
            for item in row.instructions
        ), row.source_text


def test_the_materializer_builds_the_strata_from_a_feature_config() -> None:
    from core.learning.semantic_program_feature_materialization import (
        FAMILY_FEATURE_CONFIG_SCHEMA,
        SemanticFeatureConfig,
        build_semantic_program_corpus_for_config,
    )

    config = SemanticFeatureConfig(
        seed=11, examples_per_operation_pair=4, max_examples=16,
        corpus_kind="g04_transfer_v1", schema=FAMILY_FEATURE_CONFIG_SCHEMA,
    )
    built = build_semantic_program_corpus_for_config(config)
    assert len(built) == 16
    assert {row.construction_id.split(":")[0] for row in built} == {
        f"g04_{name}" for name in G04_STRATA
    }
