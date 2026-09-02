"""The semantic-floor composition claim stays bound to measured source."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = (
    ROOT / "docs/evidence/semantic_program_27b_floor_equivalence_2026-09-01.json"
)
BINDING = (
    ROOT / "docs/evidence/semantic_program_27b_floor_source_binding_2026-09-01.json"
)
PUBLIC_PAGES = (ROOT / "README.md", ROOT / "docs/RECURSIVE_LATENT_CORTEX.md")


def test_floor_claim_matches_immutable_source_and_public_numbers() -> None:
    certificate = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    binding = json.loads(BINDING.read_text(encoding="utf-8"))
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
    assert certificate["accepted"] == certificate["agreements"] == 368
    assert certificate["value_agreements"] == 366
    assert certificate["refusal_agreements"] == 2
    assert certificate["primitive_coverage"]["complete"] is True
    assert certificate["fit_or_refit_calls"] == 0
    assert certificate["expected_answers_available"] is False
    assert certificate["serving_authority"] is False
    assert certificate["verification_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert binding["serving_authority"] is False
    assert (
        hashlib.sha256(CERTIFICATE.read_bytes()).hexdigest()
        == binding["certificates"][CERTIFICATE.name]
    )
    for relative, expected in certificate["source_sha256s"].items():
        payload = subprocess.run(
            ["git", "show", f"{binding['source_commit']}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(payload).hexdigest() == expected

    for page in PUBLIC_PAGES:
        text = page.read_text(encoding="utf-8")
        for figure in ("368/368", "366", "two", "20/20"):
            assert figure in text, (page, figure)


def test_floor_claim_is_registered_with_its_boundary() -> None:
    from core.organism.model_validation import (
        _semantic_program_27b_floor_certificate_holds,
        get_suite,
        install_runtime_validation,
    )

    install_runtime_validation()
    claims = {claim.test: claim for claim in get_suite().claims()}

    assert _semantic_program_27b_floor_certificate_holds() is True
    claim = claims["shared_semantic_programs_execute_on_universal_floor"]
    assert claim.evidence.value == "measured_synthetic"
    assert "not unseen-schema induction" in claim.evidence_note
    assert "not unseen-schema induction" not in claim.statement
