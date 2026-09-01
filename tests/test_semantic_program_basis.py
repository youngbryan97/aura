from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from core.learning.semantic_program_basis import (
    SemanticRepresentationCompatibilityError,
    bind_examples_to_compatible_training_session,
    establish_semantic_representation_compatibility,
)
from core.learning.semantic_program_transducer import fit_semantic_program_transducer
from tests.test_semantic_program_transducer import _training


def _sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _basis(*, boot: str, pid: int, source: str = "c" * 64):
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
        "worker_adapter_stack_sha256": "d" * 64,
        "worker_tokenizer": {"tokenizer.json": "e" * 64},
        "worker_runtime_tokenizer": {"vocab_size": 248044},
        "worker_quantization": {"bits": 4, "group_size": 64},
        "worker_stack_identity_gaps": [],
    }


def _manifest(basis, *, manifest_hash: str):
    return {
        "manifest_sha256": manifest_hash,
        "model_bases": [{"sha256": _sha(basis), "receipt": basis}],
    }


def _model_for_basis(basis):
    examples = [
        copy.deepcopy(item) for item in _training()
    ]
    rebound = []
    basis_hash = _sha(basis)
    for item in examples:
        rebound.append(
            replace(
                item,
                ir=replace(item.ir, model_basis_receipt_sha256=basis_hash),
            )
        )
    return fit_semantic_program_transducer(rebound), tuple(rebound)


def test_restart_compatibility_preserves_arrays_and_coefficients() -> None:
    training_basis = _basis(boot="a" * 32, pid=111)
    fresh_basis = _basis(boot="b" * 32, pid=222)
    model, training = _model_for_basis(training_basis)
    fresh = tuple(
        replace(
            item,
            ir=replace(
                item.ir,
                model_basis_receipt_sha256=_sha(fresh_basis),
            ),
        )
        for item in training
    )
    before = [np.array(item.hidden_states, copy=True) for item in fresh]
    compatibility = establish_semantic_representation_compatibility(
        model=model,
        training_manifest=_manifest(training_basis, manifest_hash="1" * 64),
        replication_manifest=_manifest(fresh_basis, manifest_hash="2" * 64),
    )

    bound = bind_examples_to_compatible_training_session(
        fresh,
        compatibility=compatibility,
    )

    assert compatibility["coefficients_changed"] is False
    assert compatibility["hidden_states_changed"] is False
    assert all(
        item.ir.model_basis_receipt_sha256 == model.model_basis_sha256
        for item in bound
    )
    assert all(
        np.array_equal(item.hidden_states, expected)
        for item, expected in zip(bound, before, strict=True)
    )


def test_compatibility_refuses_source_or_adapter_drift() -> None:
    training_basis = _basis(boot="a" * 32, pid=111)
    model, _ = _model_for_basis(training_basis)
    for drifted in (
        _basis(boot="b" * 32, pid=222, source="f" * 64),
        {**_basis(boot="b" * 32, pid=222), "worker_adapter_stack_sha256": "f" * 64},
    ):
        with pytest.raises(
            SemanticRepresentationCompatibilityError,
            match="neural function differs",
        ):
            establish_semantic_representation_compatibility(
                model=model,
                training_manifest=_manifest(training_basis, manifest_hash="1" * 64),
                replication_manifest=_manifest(drifted, manifest_hash="2" * 64),
            )


def test_compatibility_refuses_incomplete_identity() -> None:
    training_basis = _basis(boot="a" * 32, pid=111)
    model, _ = _model_for_basis(training_basis)
    fresh_basis = _basis(boot="b" * 32, pid=222)
    fresh_basis["worker_stack_identity_gaps"] = ["tokenizer:unavailable"]

    with pytest.raises(
        SemanticRepresentationCompatibilityError,
        match="basis is incomplete",
    ):
        establish_semantic_representation_compatibility(
            model=model,
            training_manifest=_manifest(training_basis, manifest_hash="1" * 64),
            replication_manifest=_manifest(fresh_basis, manifest_hash="2" * 64),
        )
