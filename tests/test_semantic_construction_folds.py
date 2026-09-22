"""Architecture selection cannot split construction or counterfactual lineage."""

from types import SimpleNamespace

import pytest

from core.learning.semantic_construction_folds import construction_folds


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
