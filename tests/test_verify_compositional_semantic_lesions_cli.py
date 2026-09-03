from __future__ import annotations

from dataclasses import replace

import numpy as np

from tests.test_semantic_program_basis import _basis, _manifest, _model_for_basis
from tools.verify_compositional_semantic_lesions import (
    _bind_compatibility_to_report,
    _bind_family_examples,
    _sha,
)


def _fresh_examples(examples, basis):
    return tuple(
        replace(
            item,
            ir=replace(
                item.ir,
                model_basis_receipt_sha256=_manifest(
                    basis,
                    manifest_hash="f" * 64,
                )["model_bases"][0]["sha256"],
            ),
        )
        for item in examples
    )


def test_cross_session_lesions_bind_to_the_frozen_transducer_basis() -> None:
    training_basis = _basis(boot="a" * 32, pid=111)
    fresh_basis = _basis(boot="b" * 32, pid=222)
    model, training_examples = _model_for_basis(training_basis)
    fresh_examples = _fresh_examples(training_examples, fresh_basis)
    before = tuple(np.array(item.hidden_states, copy=True) for item in fresh_examples)

    selected, compatibility = _bind_family_examples(
        model=model,
        family="sequence",
        examples_by_family={"sequence": fresh_examples},
        manifests={
            "sequence": _manifest(fresh_basis, manifest_hash="2" * 64),
        },
        training_manifest=_manifest(training_basis, manifest_hash="1" * 64),
    )

    assert compatibility["training_session_basis_sha256"] == model.model_basis_sha256
    assert all(
        item.ir.model_basis_receipt_sha256 == model.model_basis_sha256
        for item in selected
    )
    assert all(
        np.array_equal(item.hidden_states, expected)
        for item, expected in zip(selected, before, strict=True)
    )


def test_same_session_multifamily_binding_keeps_family_selection() -> None:
    arithmetic_basis = _basis(boot="a" * 32, pid=111)
    sequence_basis = _basis(boot="b" * 32, pid=222)
    model, arithmetic = _model_for_basis(arithmetic_basis)
    sequence = _fresh_examples(arithmetic, sequence_basis)

    selected, compatibility = _bind_family_examples(
        model=model,
        family="sequence",
        examples_by_family={"arithmetic": arithmetic, "sequence": sequence},
        manifests={
            "arithmetic": _manifest(arithmetic_basis, manifest_hash="1" * 64),
            "sequence": _manifest(sequence_basis, manifest_hash="2" * 64),
        },
        training_manifest=None,
    )

    assert len(selected) == len(sequence)
    assert all(item.construction_id.startswith("sequence:") for item in selected)
    assert {
        item.ir.model_basis_receipt_sha256 for item in selected
    } == {compatibility["target_training_session_basis_sha256"]}


def test_cross_session_compatibility_is_hashed_into_the_report() -> None:
    original = {
        "schema": "aura.semantic_program_compositional_lesions.v1",
        "arms": {"treatment": {}},
        "report_sha256": "0" * 64,
    }
    compatibility = {
        "schema": "aura.semantic_representation_compatibility.v1",
        "receipt_sha256": "1" * 64,
    }

    bound = _bind_compatibility_to_report(
        original,
        compatibility=compatibility,
    )

    assert bound["representation_compatibility"] == compatibility
    assert bound["report_sha256"] == _sha(
        {key: value for key, value in bound.items() if key != "report_sha256"}
    )
    assert bound["report_sha256"] != original["report_sha256"]
