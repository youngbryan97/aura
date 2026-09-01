"""The frozen semantic replication claim stays bound to its certificate."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = (
    ROOT
    / "docs/evidence/semantic_program_27b_frozen_replication_2026-09-01.json"
)
PUBLIC_PAGES = (ROOT / "README.md", ROOT / "docs/RECURSIVE_LATENT_CORTEX.md")
BINDING = (
    ROOT / "docs/evidence/semantic_program_27b_source_binding_2026-09-01.json"
)


def _assert_measured_commit_sources(certificate: dict[str, object]) -> None:
    binding = json.loads(BINDING.read_text(encoding="utf-8"))
    commit = binding["source_commit"]

    assert binding["serving_authority"] is False
    assert (
        hashlib.sha256(CERTIFICATE.read_bytes()).hexdigest()
        == binding["certificates"][CERTIFICATE.name]
    )
    for relative, expected in certificate["source_sha256s"].items():
        payload = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(payload).hexdigest() == expected


def test_fresh_cohort_claim_matches_source_bound_certificate() -> None:
    certificate = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    body = {
        key: value
        for key, value in certificate.items()
        if key != "verification_sha256"
    }
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")

    assert certificate["verified"] is True
    assert certificate["frozen_replay_exact"] is True
    assert certificate["held_out_treatment_answer_exact"] == 114
    assert certificate["held_out_hidden_shuffle_answer_exact"] == 10
    assert certificate["held_out_coefficient_lesion_answer_exact"] == 0
    assert certificate["held_out_total"] == 256
    assert certificate["expected_answers_available_to_training"] is False
    assert certificate["serving_authority"] is False
    assert certificate["verification_sha256"] == hashlib.sha256(encoded).hexdigest()
    _assert_measured_commit_sources(certificate)

    for page in PUBLIC_PAGES:
        text = page.read_text(encoding="utf-8")
        for figure in (
            "114/256",
            "hidden-state shuffle 10/256",
            "coefficient lesion 0/256",
        ):
            assert figure in text, (page, figure)


def test_fresh_cohort_claim_is_registered_with_its_boundary() -> None:
    from core.organism.model_validation import (
        _semantic_program_27b_replication_certificate_holds,
        get_suite,
        install_runtime_validation,
    )

    install_runtime_validation()
    claims = {claim.test: claim for claim in get_suite().claims()}

    assert _semantic_program_27b_replication_certificate_holds() is True
    claim = claims["frozen_semantic_programs_transfer_to_fresh_cohort"]
    assert claim.evidence.value == "measured_synthetic"
    assert "not serving authority" in claim.evidence_note
    assert "broad-domain gain" in claim.evidence_note
