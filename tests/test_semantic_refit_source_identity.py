import hashlib
import json
from types import SimpleNamespace

import pytest

from tools.refit_semantic_argument_proposals import (
    source_input_order_policy,
    verify_source_splits,
)


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


def test_source_input_order_policy_requires_matching_candidate_and_report():
    legacy = {"schema": "aura.compositional_source_training.v1"}
    source_order = {
        "schema": "aura.compositional_source_training.v2",
        "input_order_policy": "source_token_order_v1",
    }
    old_model = SimpleNamespace(training_receipt={})
    new_model = SimpleNamespace(training_receipt={
        "input_order_policy": "source_token_order_v1",
    })
    assert source_input_order_policy(old_model, legacy) is None
    assert source_input_order_policy(new_model, source_order) == "source_token_order_v1"
    for model, report in (
        (old_model, source_order),
        (new_model, legacy),
        (new_model, {**source_order, "input_order_policy": "other"}),
        (old_model, {"schema": "unknown"}),
    ):
        with pytest.raises(ValueError, match="source input order|unsupported"):
            source_input_order_policy(model, report)


def test_standalone_refit_isolates_state_before_loading_core(tmp_path, monkeypatch):
    import os
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    monkeypatch.delenv('AURA_LOG_DIR', raising=False)
    monkeypatch.delenv('AURA_STATE_ROOT', raising=False)
    configure_refit_environment(tmp_path / 'candidate.json')
    assert os.environ['AURA_LOG_DIR'] == str(tmp_path / 'logs')
    assert os.environ['AURA_STATE_ROOT'] == str(tmp_path / 'state')
    configure_refit_environment(tmp_path / 'another' / 'candidate.json')
    assert os.environ['AURA_STATE_ROOT'] == str(tmp_path / 'state')


def test_recovered_comparison_requires_exact_saved_fit_parent():
    from tools.refit_semantic_argument_proposals import verify_fit_start

    candidate = SimpleNamespace(training_receipt={
        "joint_graph_refit": {"parent_transducer_receipt_sha256": "parent"}})
    verify_fit_start(candidate, SimpleNamespace(receipt_sha256="parent"))
    with pytest.raises(ValueError, match="saved fit parent"):
        verify_fit_start(candidate, SimpleNamespace(receipt_sha256="different"))
    with pytest.raises(ValueError, match="saved fit parent"):
        verify_fit_start(SimpleNamespace(training_receipt={}), SimpleNamespace(receipt_sha256="parent"))
