"""An episode must not imply evidence it never gathered.

Four findings, one root: the facade has no trust domain separate from the
worker it checks.

e1c09324 — identity fields are checked for format and internal consistency,
never recomputed from trusted local state.
9b110bc5 — params_unchanged cannot prove which effective model answered.
fc19e25e — the "independent" verifier replay recomputes decisions from the
same receipt it is verifying.
265b0fae — a successful episode ran no vanilla control or held-out test, so
it establishes accepted EXECUTION, not superior reasoning.

Closing these properly needs a signed attestation and an out-of-band
verifier. Until then the claims are stated at their true strength.
"""
from __future__ import annotations

import pytest

from core.brain.latent_cortex_service import LatentCortexService


@pytest.fixture()
def disclosure():
    return LatentCortexService._attestation_disclosure({})


def test_the_trust_domain_is_named_honestly(disclosure):
    assert disclosure["trust_domain"] == "worker_self_reported"
    assert disclosure["independently_verified"] is False


def test_it_states_what_it_does_prove(disclosure):
    proves = " ".join(disclosure["proves"])

    assert "internally consistent" in proves
    assert "request payload" in proves
    assert "allocation" in proves


@pytest.mark.parametrize(
    "claim",
    [
        "trusted local state",
        "effective model",
        "separate trust domain",
        "vanilla or frontier baseline",
    ],
)
def test_each_unearned_claim_is_disclaimed(disclosure, claim):
    assert any(claim in item for item in disclosure["does_not_prove"])


def test_reasoning_gain_is_not_claimed(disclosure):
    """265b0fae: ok_episodes proves accepted execution, not better thinking."""
    assert disclosure["reasoning_gain_established"] is False


def test_verifier_replay_is_not_called_independent(disclosure):
    """fc19e25e: it recomputes from the receipt it is verifying."""
    assert disclosure["verifier_replay_independent"] is False


def test_model_identity_is_incomplete_without_adapter_state():
    """9b110bc5: params_unchanged alone cannot identify the effective model."""
    without = LatentCortexService._attestation_disclosure({"params_unchanged": True})

    assert without["model_identity_complete"] is False


def test_model_identity_is_complete_when_adapter_and_tokenizer_are_bound():
    with_state = LatentCortexService._attestation_disclosure(
        {"active_adapters": [], "tokenizer_sha256": "a" * 64}
    )

    assert with_state["model_identity_complete"] is True


def test_it_names_what_independence_would_require(disclosure):
    """A disclosure that only says 'no' is not actionable."""
    required = " ".join(disclosure["required_for_independence"])

    assert "signed" in required
    assert "out-of-band verifier" in required
    assert "vanilla control" in required


def test_a_non_mapping_receipt_is_safe():
    assert LatentCortexService._attestation_disclosure(None)["independently_verified"] is False
    assert LatentCortexService._attestation_disclosure("x")["model_identity_complete"] is False


def test_every_episode_carries_the_disclosure():
    """`deep_reason` attaches it, directly or through what it calls.

    The method-size sweep moved the line into `_deep_reason_part_10`, so
    reading `deep_reason` alone found a shell that calls names while every
    episode still carried its disclosure.
    """
    from core.brain import latent_cortex_service as module
    from tests.source_contract import function_with_its_helpers

    source = function_with_its_helpers(
        module, "LatentCortexService.deep_reason", depth=2
    )
    assert 'result["attestation"] = self._attestation_disclosure(' in source
