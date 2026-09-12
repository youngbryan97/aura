"""Keep the RLC lineages and their different claim boundaries separate."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from core.learning.compositional_semantic_qualification import (
    _verify_mechanism,
    _verify_ordinary,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "artifacts/rlc/semantic_program_27b_frozen_path_v1"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_small_frozen_backbone_campaign_remains_negative():
    path = ROOT / "artifacts/current/latent_campaign_1p5b_run2.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "065b2546817bdb054afe2dcad39405b536438f13b757b67f5e745b78192d12a6"
    )
    report = _read(path)
    assert "Qwen2.5-1.5B-Instruct-4bit" in report["model"]
    assert report["frontier_claim_eligible"] is False
    arms = report["results"]["ablations"]["arms"]
    totals = {
        name: sum(cell["successes"] for cell in cells.values())
        for name, cells in arms.items()
    }
    assert totals.pop("vanilla") == 21
    assert len(totals) == 7
    assert min(totals.values()) == 7
    assert max(totals.values()) == 13
    assert all(sum(cell["n"] for cell in cells.values()) == 72 for cells in arms.values())


def test_new_composition_counts_recompute_with_existing_qualification():
    activation = _read(PACKAGE / "activation.json")
    prereg = _read(ROOT / activation["evidence"]["preregistration"]["path"])
    mechanism = _read(PACKAGE / "mechanism_result.json")
    treatment = _verify_mechanism(
        prereg,
        mechanism,
        transducer_receipt_sha256=activation["transducer"]["receipt_sha256"],
    )
    assert treatment["total"] == 48
    assert treatment["answer_exact"] == 21
    assert sum(row["program_exact"] for row in treatment["rows"]) == 19
    for name, expected in (("ordinary_primary", 1), ("ordinary_sensitivity", 2)):
        result = _verify_ordinary(
            _read(PACKAGE / f"{name}.json"),
            treatment_rows=treatment["rows"],
            mechanism_result_sha256=mechanism["result_sha256"],
            descriptor_sha256=activation["model"]["descriptor_sha256"],
        )
        assert result["ordinary_answer_exact"] == expected
        assert result["paired_exact_test"]["control_only"] == 0
    assert activation["mode"] == "shadow"
    assert activation["serving_authority"] is False
    assert activation["composition_policy"]["shadow_result_is_observation_only"] is True


def test_new_composition_failures_are_preserved_for_development():
    rows = _read(PACKAGE / "mechanism_result.json")["arms"]["frozen_transducer"]["rows"]
    failures = Counter(
        row["refusal"] or "wrong_program" for row in rows if not row["answer_exact"]
    )
    assert failures == {"typed_argument_chart_empty": 23, "wrong_program": 4}
    assert sum(row["answer_exact"] and not row["program_exact"] for row in rows) == 2


def test_depth_reports_do_not_supply_a_resident_model_identity():
    # These are dated observations, not an identity contract for future runs.
    for loops, correct in ((1, 21), (2, 2)):
        path = ROOT / f"artifacts/recurrent_depth/arm_loops{loops}.json"
        report = _read(path)
        responses = path.with_suffix("").with_suffix(".responses.jsonl")
        assert hashlib.sha256(responses.read_bytes()).hexdigest() == report["responses_sha256"]
        rows = [json.loads(line) for line in responses.read_text().splitlines()]
        assert len(rows) == report["result"]["total"] == 40
        assert sum(row["correct"] for row in rows) == report["result"]["correct"] == correct
        assert report["recurrent_loops"] == loops
        assert "model_identity" not in report
        assert "model_descriptor_sha256" not in report
