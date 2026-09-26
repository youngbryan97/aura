from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.learning import semantic_program_compositional_campaign as campaign
from core.learning.semantic_program_basis import SemanticRepresentationCompatibilityError
from tests.test_semantic_program_basis import _basis, _manifest, _sha
from tests.test_semantic_program_shared_transducer import _examples, _grounding
from tools.rebind_semantic_source_order_identity import (
    _sha as rebind_sha,
    rebind_source_order_identity,
)


@pytest.fixture
def source_bundles(monkeypatch):
    basis = _basis(boot="a" * 32, pid=111)
    examples = tuple(replace(item, ir=replace(item.ir, model_basis_receipt_sha256=_sha(basis)))
                     for item in _examples())
    bundles = {
        name: SimpleNamespace(manifest=_manifest(basis, manifest_hash=sha * 64),
            examples=tuple(item for item in examples if len(item.ir.instructions) == steps))
        for name, sha, steps in (("linear", "1", 2), ("fork", "2", 3))
    }
    extra = replace(examples[0], ir=replace(examples[0].ir, source_text_sha256=_sha({"augmented": True})))
    bundles["counterfactual"] = SimpleNamespace(manifest=_manifest(basis, manifest_hash="3" * 64), examples=(extra,))

    def convert(bundle, *, required_splits):
        assert required_splits == frozenset({"train"})
        return bundle.examples

    monkeypatch.setattr(campaign, "training_examples_from_feature_bundle", convert)
    return bundles


def test_training_only_augmentation_preserves_validation_and_excludes_test(source_bundles):
    selected, report = campaign.prepare_compositional_source_training(source_bundles)
    assert len(selected) == 21
    assert report["training_example_count"] == 17
    assert report["validation_example_count"] == 4
    assert report["cohort_split_counts"]["counterfactual"] == {"train": 1, "validation": 0, "test": 0}
    assert {item.split for item in selected} == {"train", "validation"}
    assert report["test_examples_available_to_fit"] == 0
    without = {key: value for key, value in source_bundles.items() if key != "counterfactual"}
    _base, before = campaign.prepare_compositional_source_training(without)
    assert before["validation_example_ids_sha256"] == report["validation_example_ids_sha256"]


def test_source_order_training_is_versioned_and_keeps_split_identity(source_bundles):
    original, original_plan = campaign.prepare_compositional_source_training(source_bundles)
    reordered, plan = campaign.prepare_compositional_source_training(
        source_bundles, source_order_inputs=True)
    assert plan["schema"] == "aura.compositional_source_training_plan.v2"
    assert plan["input_order_policy"] == "source_token_order_v1"
    assert plan["validation_example_ids_sha256"] == original_plan["validation_example_ids_sha256"]
    assert len(reordered) == len(original)
    assert all(item.ir.input_spans == tuple(sorted(item.ir.input_spans,
        key=lambda span: (span.start, span.end))) for item in reordered)


def test_source_order_fit_binds_policy_to_model_identity(source_bundles):
    result = campaign.fit_compositional_source_campaign(
        source_bundles, input_grounding=_grounding(), source_order_inputs=True)
    assert result.report["schema"] == "aura.compositional_source_training.v2"
    assert result.model.training_receipt["input_order_policy"] == "source_token_order_v1"
    assert result.report["transducer_receipt_sha256"] == result.model.receipt_sha256


def test_old_source_order_fit_can_rebind_identity_with_matched_evidence(source_bundles):
    fitted = campaign.fit_compositional_source_campaign(
        source_bundles, input_grounding=_grounding(), source_order_inputs=True)
    body = {key: value for key, value in fitted.model.training_receipt.items()
            if key not in {"receipt_sha256", "input_order_policy"}}
    old_model = replace(fitted.model, training_receipt={
        **body, "receipt_sha256": rebind_sha(body)})
    report_body = {key: value for key, value in fitted.report.items()
                   if key != "report_sha256"}
    report_body["transducer_receipt_sha256"] = old_model.receipt_sha256
    old_report = {**report_body, "report_sha256": rebind_sha(report_body)}
    _, plan = campaign.prepare_compositional_source_training(
        source_bundles, source_order_inputs=True)
    validation_body = {
        "schema": "aura.semantic_source_order_regrade.v1",
        "candidate_receipt_sha256": old_model.receipt_sha256,
        "source_input_order": {"candidates": {"frozen": {
            "transducer_receipt_sha256": old_model.receipt_sha256}}},
        "source_order_plan_sha256": plan["report_sha256"],
        "validation_ids_sha256": plan["validation_example_ids_sha256"],
        "serving_authority": False,
    }
    validation = {**validation_body, "receipt_sha256": rebind_sha(validation_body)}
    rebound, new_report, candidate_report = rebind_source_order_identity(
        old_model, old_report, plan, validation)
    assert rebound.training_receipt["input_order_policy"] == "source_token_order_v1"
    assert rebound.training_receipt["source_order_identity_rebind"][
        "parent_transducer_receipt_sha256"] == old_model.receipt_sha256
    assert new_report["transducer_receipt_sha256"] == rebound.receipt_sha256
    assert candidate_report["candidate"] == rebound.receipt_sha256
    assert candidate_report["serving_authority"] is False
    assert candidate_report["receipt_sha256"] == rebind_sha({
        key: value for key, value in candidate_report.items() if key != "receipt_sha256"})
    assert {key: value for key, value in rebound.to_dict().items()
            if key != "training_receipt"} == {key: value for key, value in
                                       old_model.to_dict().items() if key != "training_receipt"}
    with pytest.raises(ValueError, match="matched fit"):
        rebind_source_order_identity(old_model, old_report, plan,
                                     {**validation, "validation_ids_sha256": "wrong"})
    with pytest.raises(ValueError, match="matched fit"):
        rebind_source_order_identity(old_model, old_report,
                                     {**plan, "input_order_policy": "wrong"}, validation)
    with pytest.raises(ValueError, match="matched fit"):
        rebind_source_order_identity(rebound, new_report, plan, validation)


def test_duplicate_across_training_and_validation_is_rejected(source_bundles):
    item = source_bundles["linear"].examples[0]
    source_bundles["counterfactual"].examples = (replace(item, split="validation"),)
    with pytest.raises(ValueError, match="repeat"):
        campaign.prepare_compositional_source_training(source_bundles)


def test_changed_neural_representation_cannot_enter_source_fit(source_bundles):
    basis = _basis(boot="b" * 32, pid=222, source="f" * 64)
    source_bundles["counterfactual"].manifest = _manifest(basis, manifest_hash="4" * 64)
    with pytest.raises(SemanticRepresentationCompatibilityError, match="neural functions differ"):
        campaign.prepare_compositional_source_training(source_bundles)


def test_no_validation_is_not_a_train_only_promotion(source_bundles):
    for bundle in source_bundles.values():
        bundle.examples = tuple(item for item in bundle.examples if item.split == "train")
    with pytest.raises(ValueError, match="train and validation"):
        campaign.prepare_compositional_source_training(source_bundles)


def test_real_fitter_receives_no_held_out_examples_and_issues_new_coefficients(source_bundles, monkeypatch):
    original = campaign.fit_compositional_semantic_program_transducer
    calls, progress = [], []

    def checked_fit(examples, **kwargs):
        assert all(item.split != "test" for item in examples)
        calls.append(len(examples))
        return original(examples, **kwargs)

    monkeypatch.setattr(campaign, "fit_compositional_semantic_program_transducer", checked_fit)
    result = campaign.fit_compositional_source_campaign(source_bundles, input_grounding=_grounding(), progress=progress.append)
    assert calls == [21]
    assert progress[0]["stage"] == "source_fit_start"
    assert result.report["fit_complete"] and not result.report["evaluation_complete"]
    assert result.report["inherited_coefficients"] is False
    assert result.report["serving_authority"] is False
    assert result.model.training_receipt["training_example_count"] == 17
    assert result.model.training_receipt["definition_selection_policy"] == "joint_graph_v1"
    assert result.model.training_receipt["relation_score_strategy"] == "categorical_log_margin_v1"
    assert result.model.training_receipt["argument_literal_boundaries"] == "atomic_v1"


def test_wrong_tokenizer_is_rejected_before_fit(source_bundles):
    for bundle in source_bundles.values():
        bundle.examples = tuple(replace(item, tokenizer_identity_sha256="f" * 64) for item in bundle.examples)
    with pytest.raises(ValueError, match="bases or geometries differ"):
        campaign.fit_compositional_source_campaign(source_bundles, input_grounding=_grounding())
