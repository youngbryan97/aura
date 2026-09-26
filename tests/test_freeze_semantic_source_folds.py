from types import SimpleNamespace

import pytest

from core.learning.semantic_program_campaign import _sha
from tools.freeze_semantic_source_folds import source_folds


def _example(identity, construction, split):
    return SimpleNamespace(ir=SimpleNamespace(source_text_sha256=identity),
                           construction_id=construction, contrast_id=None, split=split)


def test_source_folds_exclude_withheld_rows_and_replay_deterministically():
    examples = [_example("a", "one", "train"), _example("b", "two", "train"),
                _example("c", "three", "train"), _example("v", "held", "validation")]
    folds = source_folds(examples)
    assert set(folds["assignments"]) == {"a", "b", "c"}
    assert folds["validation_used"] is False and folds["test_used"] is False
    assert folds["receipt_sha256"] == _sha({key: value for key, value in folds.items()
                                             if key != "receipt_sha256"})
    assert folds == source_folds(tuple(reversed(examples)))


def test_source_folds_require_train_and_withheld_population():
    train = [_example(str(index), str(index), "train") for index in range(3)]
    with pytest.raises(ValueError, match="train and withheld"):
        source_folds(train)
    with pytest.raises(ValueError, match="train and withheld"):
        source_folds([_example("v", "held", "validation")])
