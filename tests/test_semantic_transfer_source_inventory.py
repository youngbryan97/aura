"""Transfer exclusions come from retained source evidence, not family defaults."""

import copy
import hashlib
import json

import pytest

from core.learning import semantic_program_feature_materialization as features
from core.learning.semantic_program_corpus_natural import (
    build_semantic_program_natural_request_corpus,
)
from core.learning.semantic_program_natural_transfer import (
    build_bound_semantic_source_inventory,
    build_natural_request_transfer_preflight,
    procedure_schema_signature,
)
from tests.test_semantic_feature_reacquisition import acquired as acquired
from tests.test_semantic_program_feature_materialization import _sha


def _evidence(acquired):
    _, source, _, _ = acquired
    manifest = json.loads((source / "manifest.json").read_text())
    _, examples = features.rebuild_semantic_feature_selection(manifest)
    receipt = {"receipt_sha256": "a" * 64}
    for split, prefix in (("train", "training"), ("validation", "validation")):
        ids = sorted(
            hashlib.sha256(item.source_text.encode()).hexdigest()
            for item in examples
            if item.split == split
        )
        receipt[prefix + "_example_count"] = len(ids)
        receipt[prefix + "_example_ids_sha256"] = _sha(ids)
    campaign = {
        "fit_families": ["new_family"],
        "transducer_receipt_sha256": receipt["receipt_sha256"],
        "representation_compatibility": {
            "source_feature_manifest_sha256s": {"new_family": manifest["manifest_sha256"]}
        },
    }
    campaign["report_sha256"] = _sha(campaign)
    return {"new_family": manifest}, campaign, receipt, examples


def test_exact_manifest_selection_works_without_family_table_or_hidden_arrays(
    acquired, monkeypatch
):
    manifests, campaign, receipt, examples = _evidence(acquired)
    monkeypatch.setattr(
        features,
        "load_semantic_feature_record",
        lambda *_a, **_k: pytest.fail("loaded hidden arrays"),
    )
    result = build_bound_semantic_source_inventory(
        source_manifests=manifests, source_campaign=campaign, training_receipt=receipt
    )
    assert result["example_count"] == 36
    assert result["schema_sha256s"] == sorted(
        {procedure_schema_signature(item) for item in examples}
    )
    assert result["hidden_state_arrays_loaded"] is False
    assert (
        result["sources"]["new_family"]["corpus_sha256"] == manifests["new_family"]["corpus_sha256"]
    )
    assert result["inventory_sha256"] == _sha(
        {k: v for k, v in result.items() if k != "inventory_sha256"}
    )


@pytest.mark.parametrize(
    "change",
    ["missing_family", "extra_family", "manifest", "membership", "count", "report", "model"],
)
def test_inventory_rejects_unbound_or_changed_source_evidence(acquired, change):
    manifests, campaign, receipt, _ = _evidence(acquired)
    if change == "missing_family":
        manifests.clear()
    elif change == "extra_family":
        manifests["extra"] = manifests["new_family"]
    elif change == "manifest":
        manifests["new_family"]["manifest_sha256"] = "0" * 64
    elif change == "membership":
        receipt["training_example_ids_sha256"] = "0" * 64
    elif change == "count":
        receipt["training_example_count"] += 1
    elif change == "model":
        receipt["receipt_sha256"] = "b" * 64
    else:
        campaign["report_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        build_bound_semantic_source_inventory(
            source_manifests=manifests, source_campaign=campaign, training_receipt=receipt
        )


def test_current_source_fit_report_can_derive_families_from_bound_manifests(acquired):
    manifests, campaign, receipt, _ = _evidence(acquired)
    del campaign["fit_families"]
    campaign["report_sha256"] = _sha({k: v for k, v in campaign.items() if k != "report_sha256"})
    result = build_bound_semantic_source_inventory(
        source_manifests=manifests, source_campaign=campaign, training_receipt=receipt
    )
    assert set(result["sources"]) == {"new_family"}


def test_preflight_uses_bound_source_and_binds_frozen_evidence(acquired):
    manifests, campaign, receipt, examples = _evidence(acquired)
    source_schemas = {procedure_schema_signature(item) for item in examples}
    target = tuple(
        item
        for item in build_semantic_program_natural_request_corpus()
        if procedure_schema_signature(item) not in source_schemas
    )
    assert target
    receipt["primitive_support"] = sorted(
        {ins.instruction.op for item in target for ins in item.instructions}
    )
    campaign["transducer_receipt_sha256"] = receipt["receipt_sha256"]
    campaign["report_sha256"] = _sha({k: v for k, v in campaign.items() if k != "report_sha256"})
    verification = {
        "fit_families": ["new_family"],
        "verified": True,
        "serving_authority": False,
        "transducer_receipt_sha256": receipt["receipt_sha256"],
        "source_campaign_report_sha256": campaign["report_sha256"],
    }
    verification["verification_sha256"] = _sha(verification)
    arguments = dict(
        examples=target,
        transducer={"training_receipt": receipt},
        source_campaign=campaign,
        frozen_verification=verification,
        source_manifests=manifests,
    )
    result = build_natural_request_transfer_preflight(**arguments)
    assert result["source_inventory_verified"] and result["structural_preflight_verified"]
    assert not result["blockers"] and result["serving_authority"] is False
    changed = copy.deepcopy(verification)
    changed["transducer_receipt_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="evidence identity"):
        build_natural_request_transfer_preflight(**{**arguments, "frozen_verification": changed})
    changed["verification_sha256"] = _sha(
        {k: v for k, v in changed.items() if k != "verification_sha256"}
    )
    with pytest.raises(ValueError, match="frozen verified model"):
        build_natural_request_transfer_preflight(**{**arguments, "frozen_verification": changed})
    changed = copy.deepcopy(verification)
    changed["source_campaign_report_sha256"] = "c" * 64
    changed["verification_sha256"] = _sha(
        {k: v for k, v in changed.items() if k != "verification_sha256"}
    )
    with pytest.raises(ValueError, match="frozen verified model"):
        build_natural_request_transfer_preflight(**{**arguments, "frozen_verification": changed})


def test_rebuilt_program_drift_cannot_preserve_an_old_corpus_claim(acquired, monkeypatch):
    manifests, campaign, receipt, _ = _evidence(acquired)
    identity = features._example_public_identity

    def changed(example):
        result = identity(example)
        result["program"]["instructions"][0]["op"] = "changed_primitive"
        return result

    monkeypatch.setattr(features, "_example_public_identity", changed)
    with pytest.raises(features.SemanticFeatureMaterializationError, match="corpus hash"):
        build_bound_semantic_source_inventory(
            source_manifests=manifests, source_campaign=campaign, training_receipt=receipt
        )
