from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.organism.model_validation import (
    _induced_neural_procedure_decode_certificate_holds,
    get_suite,
    install_runtime_validation,
)
from tools.run_induced_neural_procedure_canary import task_set
from tools.run_induced_neural_procedure_decode_canary import (
    ARMS,
    _arm_order,
    _induction_basis,
    _states,
)
from tools.verify_induced_neural_procedure_decode_canary import _sha, verify

REPO_ROOT = Path(__file__).resolve().parents[1]
BASIS = (
    REPO_ROOT
    / "artifacts/closeout/latent_cortex/"
    "induced_neural_procedure_canary_20260831"
)
DECODE_ARTIFACT = (
    REPO_ROOT
    / "artifacts/closeout/latent_cortex/"
    "induced_neural_procedure_decode_canary_20260831"
)
RESULT = DECODE_ARTIFACT / "result.json"
JOURNAL = DECODE_ARTIFACT / "journal.jsonl"


def test_verified_basis_reconstructs_the_frozen_induced_program() -> None:
    identity, program = _induction_basis(BASIS)

    assert identity["program_sha"] == program.sha()
    assert identity["support_count"] == 16
    assert program.describe() == "idiv(add(in0, in1), in2)"


def test_state_arms_preserve_treatment_and_disrupt_causal_controls() -> None:
    _identity, program = _induction_basis(BASIS)
    tasks = task_set(8, seed=2026084802)

    for task in tasks:
        workflow, states = _states(program, tuple(int(value) for value in task.inputs))
        treatment = states["treatment"]
        lesion = states["coefficient_lesion"]
        wrong_input = states["matched_wrong_input"]

        assert workflow
        assert treatment is not None
        assert tuple(treatment.semantic_result.values()) == (task.output,)
        assert lesion is None or lesion.semantic_result != treatment.semantic_result
        assert wrong_input is not None
        assert wrong_input.semantic_result != treatment.semantic_result
        assert wrong_input.objective_sha256 != treatment.objective_sha256


def test_arm_order_is_complete_deterministic_and_counterbalanced() -> None:
    first = _arm_order("task-a")
    second = _arm_order("task-a")

    assert first == second
    assert set(first) == set(ARMS)
    assert len(first) == len(ARMS)
    assert len({_arm_order(f"task-{index}") for index in range(32)}) > 1


def test_independent_verifier_reconstructs_resident_induced_result() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))

    verification = verify(payload, journal_path=JOURNAL)

    assert verification["verified"] is True
    assert verification["independent_exact_by_arm"] == {
        "ordinary_base": 1,
        "matched_wire_base": 1,
        "treatment": 8,
        "coefficient_lesion": 1,
        "matched_wrong_input": 0,
        "matched_wrong_state": 0,
    }
    assert verification["paired_one_sided_exact_p"] == 0.0078125
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


def test_resident_induced_claim_is_bound_to_verified_evidence() -> None:
    install_runtime_validation()
    claims = {claim.test: claim for claim in get_suite().claims()}

    assert _induced_neural_procedure_decode_certificate_holds() is True
    claim = claims["induced_neural_procedure_reaches_resident_decode"]
    assert claim.evidence.value == "measured_synthetic"
    assert "not natural-language compilation" in claim.evidence_note
    assert "open-domain reasoning" in claim.evidence_note
