from __future__ import annotations

import hashlib
import random

from core.learning.semantic_neural_composition import render_public_typed_workflow
from tools.run_semantic_neural_composition_canary import (
    _reference,
    _task_document,
    run_canary,
)
from tools.verify_semantic_neural_composition_canary import verify


def test_composition_canary_tasks_are_deterministic_and_answer_blind() -> None:
    first = _task_document(random.Random(31))
    second = _task_document(random.Random(31))
    prompt = render_public_typed_workflow(first)

    assert first == second
    assert "expected" not in prompt
    assert "answer" not in prompt.lower()
    assert _reference(first) == {"r0": 2, "r1": 3, "r2": 0, "r3": 1, "s0": 2}


def test_composition_canary_preflight_is_lesion_dependent() -> None:
    report = run_canary(seed=2026083101, task_count=24)

    assert report["passed"] is True
    assert report["verdict"] == "SUPPORTED_OPERATION_COMPOSITION"
    assert report["counts"] == {
        "treatment_exact": 24,
        "additive_lesion_disrupted": 24,
        "multiplicative_lesion_disrupted": 24,
        "wrong_operand_disrupted": 24,
    }
    assert all(
        row["public_prompt_sha256"]
        == hashlib.sha256(row["public_prompt"].encode()).hexdigest()
        for row in report["rows"]
    )
    assert verify(report)["verified"] is True


def test_composition_canary_verifier_rejects_row_tampering() -> None:
    report = run_canary(seed=2026083102, task_count=24)
    report["rows"][0]["treatment_exact"] = False

    try:
        verify(report)
    except ValueError as error:
        assert str(error) == "composition canary receipt is invalid"
    else:  # pragma: no cover - proof failure is the assertion.
        raise AssertionError("tampered composition report was accepted")


def test_verified_composition_result_is_registered_with_its_boundary() -> None:
    from core.organism.model_validation import (
        _semantic_neural_composition_certificate_holds,
        get_suite,
        install_runtime_validation,
    )

    install_runtime_validation()
    claims = {
        claim.test: claim
        for claim in get_suite().claims()
    }

    assert _semantic_neural_composition_certificate_holds() is True
    claim = claims["semantic_neural_operations_recombine_beyond_family_templates"]
    assert claim.evidence.value == "measured_synthetic"
    assert "not natural-language transfer" in claim.evidence_note
    assert "resident decoded-answer superiority" in claim.evidence_note
