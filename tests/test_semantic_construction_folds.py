"""Architecture selection cannot split construction or counterfactual lineage."""

from types import SimpleNamespace

import pytest

from core.learning.semantic_construction_folds import (
    construction_folds,
    utterance_construction_folds,
)


def row(source, construction, contrast="", split="train"):
    return SimpleNamespace(ir=SimpleNamespace(source_text_sha256=source),
                           construction_id=construction, contrast_id=contrast, split=split)


def test_transitive_lineage_and_construction_closure():
    rows = [row("a", "first"), row("b", "first"), row("c", "second", "b"),
            row("d", "third", "c"), row("e", "fourth"), row("f", "fifth")]
    result = construction_folds(rows)
    assignment = result["assignments"]
    assert len({assignment[x] for x in "abcd"}) == 1
    assert result["independent_groups"] == 3
    assert result == construction_folds(tuple(reversed(rows)))


def test_shared_contrast_parent_stays_together():
    result = construction_folds([row("a", "one", "parent"), row("b", "two", "parent"),
                                 row("c", "three"), row("d", "four")])
    assert result["assignments"]["a"] == result["assignments"]["b"]


@pytest.mark.parametrize("split", ["validation", "test"])
def test_nontraining_examples_are_rejected(split):
    with pytest.raises(ValueError, match="training"):
        construction_folds([row("a", "one", split=split)])


def test_insufficient_independent_groups_do_not_get_split_to_fill_folds():
    with pytest.raises(ValueError, match="independent"):
        construction_folds([row("a", "one"), row("b", "one"), row("c", "two")])


def test_duplicate_sources_are_not_independent_evidence():
    with pytest.raises(ValueError, match="repeated"):
        construction_folds([row("a", "one"), row("a", "two")])


def test_utterance_folds_share_semantics_but_hold_each_wording_construction():
    examples = [row(f"{family}-{index}-{variant}", f"{family}:wording-{index}",
                    contrast=f"{family}-shared")
                for family in ("first", "second") for index in range(3)
                for variant in range(2)]
    folds = utterance_construction_folds(examples)
    assert folds == utterance_construction_folds(tuple(reversed(examples)))
    assert folds["contrast_lineages_crossing_folds"] == 2
    assert folds["independent_semantic_transfer_claim"] is False
    for family in ("first", "second"):
        assert {folds["assignments"][f"{family}-{index}-0"] for index in range(3)} == {0, 1, 2}
        for index in range(3):
            assert folds["assignments"][f"{family}-{index}-0"] == folds["assignments"][
                f"{family}-{index}-1"]


def test_utterance_folds_require_per_family_coverage():
    with pytest.raises(ValueError, match="one construction per fold"):
        utterance_construction_folds([row("a", "family:one"), row("b", "family:two")])
    with pytest.raises(ValueError, match="family provenance"):
        utterance_construction_folds([row(str(index), f"wording-{index}") for index in range(3)])
