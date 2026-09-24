"""A diagnostic replay cannot silently import train rows or change identity."""

from types import SimpleNamespace

import pytest

from tools.audit_semantic_cohort import select_diagnostic_examples


def _item(source, split):
    return SimpleNamespace(ir=SimpleNamespace(source_text_sha256=source), split=split)


def test_select_only_declared_validation_rows_in_declared_order():
    rows = [_item("train", "train"), _item("a", "validation"), _item("b", "validation")]
    assert [item.ir.source_text_sha256 for item in select_diagnostic_examples(rows, ["b", "a"])] == ["b", "a"]
    assert len(select_diagnostic_examples(rows, [])) == 3
    assert [item.ir.source_text_sha256 for item in select_diagnostic_examples(
        rows, [], validation_only=True)] == ["a", "b"]


@pytest.mark.parametrize("ids", [["train"], ["missing"], ["a", "a"]])
def test_reject_train_unknown_and_duplicate_diagnostic_ids(ids):
    rows = [_item("train", "train"), _item("a", "validation")]
    with pytest.raises(ValueError):
        select_diagnostic_examples(rows, ids)
