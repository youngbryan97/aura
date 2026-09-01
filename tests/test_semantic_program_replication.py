from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest

from core.learning.semantic_program_feature_materialization import (
    LoadedSemanticFeatureBundle,
    LoadedSemanticFeatureExample,
)
from core.learning.semantic_program_replication import (
    SEMANTIC_PROGRAM_REPLICATION_SOURCES,
    FrozenTrainingCohort,
    SemanticProgramReplicationError,
    _exact_one_sided_pair,
    evaluate_frozen_semantic_replication,
    load_reconstructed_semantic_cohort,
)
from core.learning.semantic_program_transducer import fit_semantic_program_transducer
from tests.test_semantic_program_evaluation import _battery
from tests.test_semantic_program_transducer import _example


def _sha_character(character: str) -> str:
    return character * 64


def _worker_basis(*, boot: str, pid: int, source: str = "7" * 64):
    return {
        "schema": "aura.latent_cortex.worker_identity.v3",
        "worker_boot_id": boot,
        "worker_pid": pid,
        "worker_model_path": "/models/aura-27b",
        "worker_model_parameter_count": 27_000_000_000,
        "worker_model_stored_parameter_element_count": 4_200_000_000,
        "worker_model_parameter_count_basis": "stored_tensor_elements",
        "worker_source_sha256": source,
        "worker_affective_steering_active": False,
        "worker_affective_steering_alpha": 0.0,
        "worker_action_capture_identity": {"worker_boot_id": boot, "worker_pid": pid},
        "worker_recurrent_adapter_activation": {"active": False},
        "worker_adapters": [],
        "worker_adapter_stack_sha256": "8" * 64,
        "worker_tokenizer": {"tokenizer.json": "9" * 64},
        "worker_runtime_tokenizer": {"vocab_size": 248044},
        "worker_quantization": {"bits": 4, "group_size": 64},
        "worker_stack_identity_gaps": [],
    }


def _manifest(basis, *, character: str):
    return {
        "manifest_sha256": _sha_character(character),
        "model_bases": [
            {"sha256": _canonical_sha(basis), "receipt": basis},
        ],
    }


def _with_basis(examples, basis):
    basis_hash = _canonical_sha(basis)
    return tuple(
        replace(
            item,
            ir=replace(item.ir, model_basis_receipt_sha256=basis_hash),
        )
        for item in examples
    )


def _bundle(examples, *, basis) -> LoadedSemanticFeatureBundle:
    rows = tuple(
        LoadedSemanticFeatureExample(
            metadata={
                "example_id": f"fresh-{index}",
                "gold_ir": example.ir.to_dict(),
                "split": example.split,
                "construction_id": example.construction_id,
                "topology_id": example.topology_id,
                "inputs": list(example.public_inputs),
            },
            token_ids=example.hidden_states[:, 0].astype("<i4"),
            hidden_states=example.hidden_states,
            payload_sha256=_sha_character("d"),
        )
        for index, example in enumerate(examples)
    )
    return LoadedSemanticFeatureBundle(
        manifest={
            **_manifest(basis, character="b"),
            "config_sha256": _sha_character("c"),
            "example_count": len(rows),
        },
        examples=rows,
    )


def _sources() -> dict[str, str]:
    return {path: _sha_character("a") for path in SEMANTIC_PROGRAM_REPLICATION_SOURCES}


def _replication_battery():
    return [
        *_battery(),
        *[
            _example(
                first,
                second,
                topology,
                split="validation",
                order=(1, 3, 5, 7, 0, 2, 4, 6, 8),
            )
            for topology in range(4)
            for first in ("add", "sub", "mul", "idiv")
            for second in ("add", "sub", "mul", "idiv")
        ],
    ]


def _canonical_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def test_frozen_replication_uses_matched_controls_without_refitting(monkeypatch) -> None:
    training_basis = _worker_basis(boot="1" * 32, pid=101)
    replication_basis = _worker_basis(boot="2" * 32, pid=202)
    training_examples = _with_basis(_replication_battery(), training_basis)
    examples = _with_basis(_replication_battery(), replication_basis)
    model = fit_semantic_program_transducer(training_examples)
    payload = model.to_dict()
    original = copy.deepcopy(payload)
    bundle = _bundle(examples, basis=replication_basis)

    def forbidden_fit(*_args, **_kwargs):
        raise AssertionError("frozen replication must not fit a model")

    monkeypatch.setattr(
        "core.learning.semantic_program_transducer.fit_semantic_program_transducer",
        forbidden_fit,
    )
    monkeypatch.setattr(
        "core.learning.semantic_program_campaign.fit_semantic_program_transducer",
        forbidden_fit,
    )
    report = evaluate_frozen_semantic_replication(
        bundle,
        trained_model_payload=payload,
        training_cohort=FrozenTrainingCohort(
            feature_manifest_sha256=_sha_character("e"),
            example_ids=("old-a", "old-b"),
        ),
        training_manifest=_manifest(training_basis, character="e"),
        source_sha256s=_sources(),
    )

    assert payload == original
    assert report["fresh_cohort"] is True
    assert report["fitting_calls"] == report["refitting_calls"] == 0
    assert report["trained_model_unchanged"] is True
    assert report["representation_compatibility"]["hidden_states_changed"] is False
    assert report["serving_authority"] is False
    assert set(report["arms"]) == {
        f"{arm}:{split}"
        for arm in ("treatment", "hidden_token_shuffle", "coefficient_lesion")
        for split in ("train", "validation", "test")
    }
    assert len(report["paired_exact_tests"]) == 16
    assert report["report_sha256"]


def test_frozen_replication_refuses_training_overlap_and_basis_mismatch() -> None:
    training_basis = _worker_basis(boot="1" * 32, pid=101)
    replication_basis = _worker_basis(boot="2" * 32, pid=202)
    examples = _with_basis(_replication_battery(), training_basis)
    model = fit_semantic_program_transducer(examples)
    bundle = _bundle(
        _with_basis(_replication_battery(), replication_basis),
        basis=replication_basis,
    )
    with pytest.raises(SemanticProgramReplicationError, match="overlaps"):
        evaluate_frozen_semantic_replication(
            bundle,
            trained_model_payload=model.to_dict(),
            training_cohort=FrozenTrainingCohort(
                feature_manifest_sha256=_sha_character("e"),
                example_ids=("fresh-0",),
            ),
            training_manifest=_manifest(training_basis, character="e"),
            source_sha256s=_sources(),
        )

    drifted_basis = _worker_basis(boot="2" * 32, pid=202, source="f" * 64)
    drifted_bundle = _bundle(
        _with_basis(_replication_battery(), drifted_basis),
        basis=drifted_basis,
    )
    with pytest.raises(RuntimeError, match="neural function differs"):
        evaluate_frozen_semantic_replication(
            drifted_bundle,
            trained_model_payload=model.to_dict(),
            training_cohort=FrozenTrainingCohort(
                feature_manifest_sha256=_sha_character("e"),
                example_ids=("old",),
            ),
            training_manifest=_manifest(training_basis, character="e"),
            source_sha256s=_sources(),
        )


def test_exact_pair_reports_reduced_rational_probability() -> None:
    treatment = [
        {"source_text_sha256": str(index), "program_exact": True}
        for index in range(4)
    ]
    control = [
        {"source_text_sha256": str(index), "program_exact": False}
        for index in range(4)
    ]
    assert _exact_one_sided_pair(treatment, control, metric="program_exact") == {
        "metric": "program_exact",
        "treatment_only": 4,
        "control_only": 0,
        "discordant": 4,
        "one_sided_exact_p_numerator": 1,
        "one_sided_exact_p_denominator": 16,
        "one_sided_exact_p": 0.0625,
    }


def test_bundle_loader_delegates_to_canonical_reconstruction(monkeypatch, tmp_path) -> None:
    basis = _worker_basis(boot="2" * 32, pid=202)
    expected = _bundle(_with_basis(_replication_battery(), basis), basis=basis)
    monkeypatch.setattr(
        "core.learning.semantic_program_replication.load_standard_semantic_feature_bundle",
        lambda path: expected if path == tmp_path else pytest.fail("wrong bundle path"),
    )
    assert load_reconstructed_semantic_cohort(tmp_path) is expected


def test_source_binding_refuses_incomplete_hash_set() -> None:
    basis = _worker_basis(boot="1" * 32, pid=101)
    examples = _with_basis(_replication_battery(), basis)
    model = fit_semantic_program_transducer(examples)
    sources = _sources()
    sources.pop(next(iter(sources)))
    with pytest.raises(SemanticProgramReplicationError, match="source identity"):
        evaluate_frozen_semantic_replication(
            _bundle(examples, basis=basis),
            trained_model_payload=model.to_dict(),
            training_cohort=FrozenTrainingCohort(
                feature_manifest_sha256=_sha_character("e"),
                example_ids=("old",),
            ),
            training_manifest=_manifest(basis, character="e"),
            source_sha256s=sources,
        )
