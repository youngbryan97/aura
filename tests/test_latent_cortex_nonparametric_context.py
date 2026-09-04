"""One-shot token memory becomes immutable, receipted recurrent evidence."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from core.brain import nonparametric_memory, nonparametric_worker
from core.brain.latent_cortex_service import LatentCortexService
from core.brain.llm.latent_cortex.nonparametric_context import (
    retrieve_observation,
    validate_receipt,
)
from core.brain.llm.latent_cortex.resource_accounting import (
    ModelComputeProfile,
    ResourceLedger,
    build_information_receipt,
    policy_sha256,
)
from core.brain.nonparametric_memory import NonParametricMemory
from tests.nonparametric_support import TEST_PRINCIPAL, entry_provenance


class _Tokenizer:
    def decode(self, tokens):
        return {7: " remembered"}.get(int(tokens[0]), "")


@pytest.fixture
def active_store(tmp_path, monkeypatch):
    store = NonParametricMemory(dim=4, path=tmp_path / "one-shot")
    store.add(np.array([1.0, 0.0, 0.0, 0.0]), 7, "remembered", provenance=entry_provenance())
    monkeypatch.setattr(nonparametric_worker, "foreground_enabled", lambda: True)
    monkeypatch.setattr(
        nonparametric_memory,
        "get_nonparametric_memory",
        lambda dim=0: store if int(dim or 4) == 4 else None,
    )
    return store


def test_exact_prompt_tail_is_admitted_as_non_authoritative_evidence(active_store):
    observation, receipt = retrieve_observation(
        np.array([1.0, 0.0, 0.0, 0.0]), _Tokenizer(), principal=TEST_PRINCIPAL
    )

    assert validate_receipt(receipt) == receipt
    assert receipt["status"] == "admitted"
    assert receipt["applied"] is True
    assert receipt["similarity"] >= receipt["similarity_gate"]
    assert receipt["resource_accounting"]["entries_examined"] == 1
    assert receipt["resource_accounting"]["identity_scan_bytes"] > 0
    assert observation is not None
    assert observation["context_role"] == "evidence_observation"
    assert observation["instruction_authority"] is False
    assert observation["evidence_kind"] == "one_shot_nonparametric_memory"
    assert observation["content_sha256"] == receipt["observation_sha256"]
    assert (
        observation["content_sha256"]
        == hashlib.sha256(observation["text"].encode("utf-8")).hexdigest()
    )


def test_mlx_bfloat_hidden_is_admitted_without_a_pep3118_failure(active_store):
    import mlx.core as mx

    hidden = mx.array([1.0, 0.0, 0.0, 0.0], dtype=mx.bfloat16)
    mx.eval(hidden)

    observation, receipt = retrieve_observation(
        hidden,
        _Tokenizer(),
        principal=TEST_PRINCIPAL,
    )

    assert observation is not None
    assert validate_receipt(receipt)["status"] == "admitted"


def test_unrelated_prompt_tail_fails_closed_at_similarity_gate(active_store):
    observation, receipt = retrieve_observation(
        np.array([0.0, 1.0, 0.0, 0.0]), _Tokenizer(), principal=TEST_PRINCIPAL
    )

    assert observation is None
    assert validate_receipt(receipt)["status"] == "below_similarity_gate"
    assert receipt["applied"] is False
    assert receipt["observation_sha256"] == ""


def test_disabled_store_is_not_loaded_or_queried(active_store, monkeypatch):
    monkeypatch.setattr(nonparametric_worker, "foreground_enabled", lambda: False)
    observation, receipt = retrieve_observation(np.ones(4), _Tokenizer(), principal=TEST_PRINCIPAL)

    assert observation is None
    assert validate_receipt(receipt)["status"] == "store_unavailable"
    assert receipt["source_identity"] == {}
    assert receipt["resource_accounting"]["entries_examined"] == 0


def test_receipt_rejects_verdict_and_resource_tampering(active_store):
    _observation, receipt = retrieve_observation(
        np.array([1.0, 0.0, 0.0, 0.0]), _Tokenizer(), principal=TEST_PRINCIPAL
    )
    tampered = copy.deepcopy(receipt)
    tampered["resource_accounting"]["entries_examined"] += 1

    with pytest.raises(ValueError):
        validate_receipt(tampered)


def test_service_binds_admitted_receipt_to_exact_immutable_slot(active_store):
    observation, one_shot = retrieve_observation(
        np.array([1.0, 0.0, 0.0, 0.0]), _Tokenizer(), principal=TEST_PRINCIPAL
    )
    assert observation is not None
    accounting = one_shot["resource_accounting"]
    ledger = ResourceLedger(
        ModelComputeProfile(
            model_type="one-shot-test",
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=32,
            head_dim=2,
        )
    )
    ledger.charge(
        "nonparametric_memory_retrieval",
        tensor_element_reads=accounting["tensor_element_reads"],
        tensor_element_writes=accounting["tensor_element_writes"],
        tensor_scalar_ops=accounting["tensor_scalar_ops"],
        host_scalar_ops=accounting["host_scalar_ops"],
    )
    source_identity = one_shot["source_identity"]
    information = build_information_receipt(
        sources=[
            {
                "source_id": "one_shot_nonparametric_memory",
                "kind": "local_nonparametric_memory_store",
                "content_sha256": source_identity["content_sha256"],
                "byte_count": source_identity["source_bytes"],
                "token_count": 0,
            },
            {
                "source_id": "cognitive_context:0:one_shot_memory",
                "kind": "typed_cognitive_context",
                "content_sha256": one_shot["observation_sha256"],
                "byte_count": len(observation["text"].encode()),
                "token_count": 1,
            },
        ],
        policies={
            "nonparametric_memory": policy_sha256(
                {
                    "policy": "context_only_prompt_tail_recall_v1",
                    "active_source_receipt_sha256": source_identity["receipt_sha256"],
                }
            )
        },
    )
    config = {"n_slots": 9, "n_branches": 1}
    receipt = {
        "n_slots": 9,
        "n_branches": 1,
        "nonparametric_memory": one_shot,
        "budget": {
            "resource_accounting": ledger.to_receipt(),
            "information_accounting": information,
        },
        "cognitive_slots": [
            {
                "slot": 1,
                "context_index": 0,
                "source": "one_shot_memory",
                "knowledge_class": "one_shot_nonparametric_memory",
                "instruction_authority": False,
                "text_sha256": one_shot["observation_sha256"],
            }
        ],
    }

    errors = LatentCortexService._receipt_contract_errors(receipt, config)
    assert "nonparametric_memory_binding_unproven" not in errors

    tampered = copy.deepcopy(receipt)
    tampered["cognitive_slots"][0]["text_sha256"] = "0" * 64
    errors = LatentCortexService._receipt_contract_errors(tampered, config)
    assert "nonparametric_memory_binding_unproven" in errors


def test_an_unscoped_recurrent_step_gets_no_clue(active_store):
    """A turn that cannot say whose memory this is reads nobody's."""
    observation, receipt = retrieve_observation(
        np.array([1.0, 0.0, 0.0, 0.0]),
        _Tokenizer(),
    )

    assert observation is None
    assert validate_receipt(receipt)["status"] == "no_principal"
    assert receipt["applied"] is False
    assert receipt["observation_sha256"] == ""


def test_another_principal_gets_no_clue_from_this_one(active_store):
    """The entry belongs to TEST_PRINCIPAL; a different turn cannot see it."""
    observation, receipt = retrieve_observation(
        np.array([1.0, 0.0, 0.0, 0.0]),
        _Tokenizer(),
        principal="somebody_else",
    )

    assert observation is None
    assert validate_receipt(receipt)["status"] != "admitted"
