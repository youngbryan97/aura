import hashlib
import json
from types import SimpleNamespace

import pytest

from tools.refit_semantic_argument_proposals import verify_source_splits


def example(split, identity):
    return SimpleNamespace(split=split, ir=SimpleNamespace(source_text_sha256=identity))


def receipt():
    return {
        f"{prefix}_{suffix}": value
        for prefix, ids in (("training", ["a", "b"]), ("validation", ["c"]))
        for suffix, value in (
            ("example_count", len(ids)),
            ("example_ids_sha256", hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode("ascii")
            ).hexdigest()),
        )
    }


def test_exact_source_ignores_order_and_test_examples():
    verify_source_splits(
        [example("train", "b"), example("test", "unused"),
         example("validation", "c"), example("train", "a")], receipt()
    )


@pytest.mark.parametrize("ids", [["a"], ["a", "x"], ["a", "a"], []])
def test_missing_substituted_or_duplicated_training_is_rejected(ids):
    with pytest.raises(ValueError, match="train source cohort"):
        verify_source_splits(
            [*(example("train", x) for x in ids), example("validation", "c")], receipt()
        )


def test_changed_validation_is_rejected():
    with pytest.raises(ValueError, match="validation source cohort"):
        verify_source_splits(
            [example("train", "a"), example("train", "b"), example("validation", "x")],
            receipt(),
        )
