from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.organism.model_validation import (
    _resident_semantic_neural_composition_decode_certificate_holds,
    get_suite,
    install_runtime_validation,
)
from tools.verify_semantic_neural_composition_decode_canary import _sha, verify

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = (
    REPO_ROOT
    / "artifacts/closeout/latent_cortex/typed_composition_decode_canary_20260831"
)
RESULT = ARTIFACT_ROOT / "result.json"
JOURNAL = ARTIFACT_ROOT / "result.json.journal.jsonl"


def test_independent_verifier_reconstructs_resident_composition_result() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))

    verification = verify(payload, journal_path=JOURNAL)

    assert verification["verified"] is True
    assert verification["independent_exact_by_arm"] == {
        "ordinary_base": 0,
        "matched_wire_base": 0,
        "treatment": 8,
        "additive_lesion": 0,
        "multiplicative_lesion": 0,
        "matched_wrong_state": 0,
    }
    assert verification["paired_one_sided_exact_p"] == 0.00390625
    assert verification["journal_identity"]["decode_count"] == 48


def test_verifier_rejects_a_resealed_response_mutation() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    payload["rows"][0]["response"] += " mutated"
    payload["rows"][0]["response_sha256"] = hashlib.sha256(
        payload["rows"][0]["response"].encode()
    ).hexdigest()
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    payload["receipt_sha256"] = _sha(body)

    with pytest.raises(RuntimeError, match="independent row replay differs"):
        verify(payload, journal_path=JOURNAL)


def test_resident_composition_claim_is_bound_to_verified_evidence() -> None:
    install_runtime_validation()
    claims = {claim.test: claim for claim in get_suite().claims()}

    assert _resident_semantic_neural_composition_decode_certificate_holds() is True
    claim = claims["resident_semantic_composition_decode_is_causally_verified"]
    assert claim.evidence.value == "measured_synthetic"
    assert "not hidden-state internalization" in claim.evidence_note
    assert "open-domain reasoning gain" in claim.evidence_note
