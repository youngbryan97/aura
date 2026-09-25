import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.compare_semantic_feature_cohorts import _cohort, compare_bundles


def _bundle(manifest, *, source="a", split="train", token="t", hidden="h", construction="c", contrast=None):
    metadata = {"source_text_sha256": source, "split": split, "token_ids_sha256": token,
                "hidden_states_sha256": hidden, "construction_id": construction, "contrast_id": contrast}
    return SimpleNamespace(manifest={"manifest_sha256": manifest}, examples=(SimpleNamespace(metadata=metadata),))


def test_lineage_change_does_not_impersonate_feature_change():
    result = compare_bundles("source", _bundle("old"),
                             _bundle("new", construction="rebound", contrast="parent"))
    assert result["lineage_changes"] == ["a"]
    assert result["hidden_changes"] == result["token_changes"] == []


@pytest.mark.parametrize("field,value", [("source", "b"), ("split", "validation"),
                                         ("token", "other"), ("hidden", "other")])
def test_measured_changes_are_reported(field, value):
    result = compare_bundles("source", _bundle("old"), _bundle("new", **{field: value}))
    expected = {"source": "added_sources", "split": "split_changes",
                "token": "token_changes", "hidden": "hidden_changes"}[field]
    assert result[expected]


def test_duplicate_source_refused():
    old = _bundle("old")
    old.examples += old.examples
    with pytest.raises(ValueError, match="repeats"):
        compare_bundles("source", old, _bundle("new"))


def test_cohort_argument_is_structured():
    assert _cohort("one=/old,/new") == ("one", Path("/old"), Path("/new"))
    with pytest.raises(argparse.ArgumentTypeError):
        _cohort("one=/old")
