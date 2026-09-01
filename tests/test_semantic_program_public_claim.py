"""The public semantic-transfer claim must remain bound to its certificate."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = (
    ROOT / "docs/evidence/semantic_program_27b_reverification_2026-09-01.json"
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


def test_public_semantic_program_claim_matches_source_bound_certificate() -> None:
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
    assert certificate["held_out_treatment_answer_exact"] == 134
    assert certificate["held_out_treatment_program_exact"] == 133
    assert certificate["held_out_total"] == 256
    assert certificate["verification_sha256"] == hashlib.sha256(encoded).hexdigest()
    _assert_measured_commit_sources(certificate)
    for page in PUBLIC_PAGES:
        text = page.read_text(encoding="utf-8")
        for figure in (
            "134/256",
            "133/256",
            "hidden-state shuffle 14/256",
            "coefficient lesion 0/256",
            "label permutation 4/256",
        ):
            assert figure in text, (page, figure)


def test_verified_semantic_program_claim_is_registered_with_its_boundary() -> None:
    from core.organism.model_validation import (
        _semantic_program_27b_certificate_holds,
        get_suite,
        install_runtime_validation,
    )

    install_runtime_validation()
    claims = {claim.test: claim for claim in get_suite().claims()}

    assert _semantic_program_27b_certificate_holds() is True
    claim = claims["resident_semantic_programs_execute_exact_answers"]
    assert claim.evidence.value == "measured_synthetic"
    assert "not unrestricted serving" in claim.evidence_note
    assert "frontier reasoning" in claim.evidence_note
