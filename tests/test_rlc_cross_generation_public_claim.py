"""Public RLC claims must stay bound to both frozen adjudications."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADJUDICATIONS = {
    "32B": ROOT
    / "artifacts/closeout/latent_cortex/cp566_resident_mixed_multidomain_replication/adjudication.json",
    "27B": ROOT
    / "artifacts/migration/27b/recovery/cp1003-semantic-canary/adjudication.json",
}
PUBLIC_PAGES = (
    ROOT / "README.md",
    ROOT / "docs/RECURSIVE_LATENT_CORTEX.md",
    ROOT / "docs/INTRINSIC_RECURRENCE.md",
)


def test_public_cross_generation_claim_matches_frozen_adjudications() -> None:
    evidence = {
        model: json.loads(path.read_text(encoding="utf-8"))
        for model, path in ADJUDICATIONS.items()
    }

    assert evidence["32B"]["verdict"] == "BOUNDED_WOW_SIGNAL"
    assert evidence["32B"]["independent_exact_by_arm"] == {
        "coefficient_lesion": 5,
        "matched_wire_base": 7,
        "matched_wrong_state": 0,
        "ordinary_base": 16,
        "treatment": 60,
    }
    assert evidence["27B"]["verdict"] == "BOUNDED_WOW_SIGNAL"
    assert evidence["27B"]["independent_exact_by_arm"] == {
        "coefficient_lesion": 4,
        "matched_wire_base": 6,
        "matched_wrong_state": 0,
        "ordinary_base": 0,
        "treatment": 60,
    }
    assert evidence["32B"]["limitations"] == evidence["27B"]["limitations"]

    for page in PUBLIC_PAGES:
        text = page.read_text(encoding="utf-8")
        assert "60/60" in text, page
        assert "16/60" in text, page
        assert "0/60" in text, page
        assert "8.67 × 10⁻¹⁹" in text, page
        assert "not" in text.lower() and "frontier" in text.lower(), page


def test_public_pages_keep_failed_replication_separate_from_development() -> None:
    baseline = json.loads(
        (ROOT / "docs/evidence/G01_RLC_BASELINE_2026-09-08.json").read_text()
    )
    measurements = baseline["measurements"]
    failed = measurements["v16_natural"]
    development = measurements["v19_development"]
    assert failed["verdict"] == "FAIL_ABSOLUTE_CAPABILITY_FLOOR"
    assert development["fresh_replication"] is False
    for page in PUBLIC_PAGES:
        text = page.read_text(encoding="utf-8")
        assert f"{failed['answer_exact']}/{failed['total']}" in text, page
        assert f"{failed['minimum_answer_exact']}/{failed['total']}" in text, page
        assert f"{development['answer_exact']}/{development['total']}" in text, page
        links = re.findall(r"\]\(([^)]+G01_RLC_BASELINE_2026-09-08\.md)\)", text)
        assert links, page
        assert all((page.parent / link).resolve().is_file() for link in links), page


def test_public_pages_do_not_attribute_cp566_to_backbone_depth() -> None:
    # These were the specific false attributions corrected by G13.
    retired_claims = (
        "That is where the gain came from.",
        "The dividend did come from training the checkpoint to use recurrence",
        "That prediction held. Recurrence-native training is where the dividend came",
        "A different mechanism: the answer's own token stream re-enters the middle block",
        "It runs in the live serving path today.",
    )
    for page in PUBLIC_PAGES:
        text = page.read_text(encoding="utf-8")
        for claim in retired_claims:
            assert claim not in text, (page, claim)
