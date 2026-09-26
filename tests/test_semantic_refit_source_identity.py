import hashlib
import json
from types import SimpleNamespace

import pytest

from tools.refit_semantic_argument_proposals import (
    source_bundle_arguments,
    source_input_order_policy,
    verify_candidate_report_identity,
    verify_source_report_identity,
    verify_source_splits,
)
from core.learning.semantic_program_campaign import _sha as semantic_sha


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


def test_source_bundles_accept_separate_roots_only_for_exact_cohort(tmp_path):
    for name in ("arithmetic", "cataphoric", "counterfactual", "other"):
        (tmp_path / name).mkdir()
    report = {"representation_compatibility": {
        "source_feature_manifest_sha256s": {
            "arithmetic": "a", "cataphoric": "b", "counterfactual": "c"}}}
    expected = ["arithmetic=" + str(tmp_path / "arithmetic"),
                "cataphoric=" + str(tmp_path / "cataphoric"),
                "counterfactual=" + str(tmp_path / "counterfactual")]
    assert source_bundle_arguments(report, feature_root=tmp_path) == expected
    explicit = ["counterfactual=" + str(tmp_path / "other"), expected[1], expected[0]]
    assert source_bundle_arguments(report, bundles=explicit) == [
        expected[0], expected[1], "counterfactual=" + str(tmp_path / "other")]
    with pytest.raises(ValueError, match="choose one"):
        source_bundle_arguments(report)
    with pytest.raises(ValueError, match="choose one"):
        source_bundle_arguments(report, feature_root=tmp_path, bundles=explicit)
    with pytest.raises(ValueError, match="cohort"):
        source_bundle_arguments(report, bundles=[expected[0], expected[1],
                                "unrelated=" + str(tmp_path / "other")])


def test_rebound_candidate_report_binds_model_cohort_and_parent_lineage():
    model = SimpleNamespace(receipt_sha256="rebound", training_receipt={
        "input_order_policy": "source_token_order_v1",
        "source_order_identity_rebind": {
            "parent_transducer_receipt_sha256": "parent",
            "parent_validation_receipt_sha256": "validation"}})
    source_body = {"schema": "aura.compositional_source_training.v2",
                   "input_order_policy": "source_token_order_v1",
                   "transducer_receipt_sha256": "rebound"}
    source_report = {**source_body, "report_sha256": semantic_sha(source_body)}
    body = {"schema": "aura.semantic_source_order_identity_rebind.v1",
            "candidate": "rebound", "source_report_sha256": source_report["report_sha256"],
            "parent_candidate_receipt_sha256": "parent",
            "parent_validation_receipt_sha256": "validation",
            "coefficients_changed": False, "serving_authority": False}
    report = {**body, "receipt_sha256": semantic_sha(body)}
    verify_candidate_report_identity(report, model, source_report)
    for changed_report, changed_model, changed_source in (
        ({**report, "candidate": "other"}, model, source_report),
        ({**report, "receipt_sha256": "bad"}, model, source_report),
        ({**report, "source_report_sha256": "other",
          "receipt_sha256": semantic_sha({**body, "source_report_sha256": "other"})},
         model, source_report),
        (report, SimpleNamespace(receipt_sha256="rebound", training_receipt={
            "input_order_policy": "source_token_order_v1",
            "source_order_identity_rebind": {
                "parent_transducer_receipt_sha256": "other",
                "parent_validation_receipt_sha256": "validation"}}), source_report),
        (report, model, {**source_report, "report_sha256": "other"}),
    ):
        with pytest.raises(ValueError, match="report identity"):
            verify_candidate_report_identity(changed_report, changed_model, changed_source)


def test_legacy_candidate_report_must_still_have_intact_receipt():
    model = SimpleNamespace(receipt_sha256="model")
    source_body = {"schema": "aura.compositional_source_training.v1",
                   "transducer_receipt_sha256": "model"}
    source_report = {**source_body, "report_sha256": semantic_sha(source_body)}
    body = {"schema": "aura.semantic_cohort_diagnosis.v2", "candidate": "model"}
    report = {**body, "receipt_sha256": semantic_sha(body)}
    verify_candidate_report_identity(report, model, source_report)
    with pytest.raises(ValueError, match="candidate report identity"):
        verify_candidate_report_identity({**report, "candidate": "other"}, model, source_report)
    with pytest.raises(ValueError, match="source training report identity"):
        verify_source_report_identity({**source_report, "transducer_receipt_sha256": "other"}, model)


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
