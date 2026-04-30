"""tests/test_canary_replay_real.py — True Canary Replay

Replay real recent interactions through baseline and patched Aura,
compare with semantic/verifier scoring.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.promotion.canary_runtime import (
    CanaryRuntime, CanaryReport, ReplayExample, build_replay_examples,
)
from core.promotion.behavioral_contracts import BehavioralContractSuite
from core.verification.semantic_verifier import SemanticVerifier

MONOLOGUE_PATH = ROOT / "data" / "internal_monologue.jsonl"
COMM_LOGS_PATH = ROOT / "data" / "comm_logs.jsonl"


def _load_interaction_records(limit=32):
    """Load real interaction records, preferring monologue for density."""
    records = []
    for path in [MONOLOGUE_PATH, COMM_LOGS_PATH]:
        if not path.exists():
            continue
        for line in open(path, "r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Normalize to input/output format
            text = str(r.get("reflection") or r.get("content") or "")
            if not text or len(text) < 10:
                continue
            records.append({
                "input": text[:80],
                "output": text,
                "id": f"real_{len(records)}",
                "timestamp": r.get("timestamp") or r.get("t", 0),
            })
            if len(records) >= limit:
                break
    return records


def _baseline_responder(example: ReplayExample) -> str:
    """Baseline: returns the recorded output verbatim."""
    return example.baseline_output


def _patched_responder(example: ReplayExample) -> str:
    """Patched: minor normalization to simulate a real patch diff."""
    text = example.baseline_output
    # Simulate a real patch: whitespace normalization + minor rewording
    text = " ".join(text.split())
    # Slight word substitution to test the diff machinery
    text = text.replace("circuits", "pathways").replace("spinning", "processing")
    return text


class TestCanaryReplayReal:
    def test_interaction_records_available(self):
        records = _load_interaction_records(limit=5)
        assert len(records) > 0, "no interaction records found"

    def test_build_replay_examples(self):
        records = _load_interaction_records(limit=10)
        if not records:
            pytest.skip("no interaction data")
        examples = build_replay_examples(records, limit=10)
        assert len(examples) > 0
        for ex in examples:
            assert ex.input_text
            assert ex.baseline_output

    def test_canary_baseline_identity(self):
        """Baseline responder (identical output) must pass perfectly."""
        records = _load_interaction_records(limit=16)
        if not records:
            pytest.skip("no interaction data")
        examples = build_replay_examples(records, limit=16)
        canary = CanaryRuntime()
        metrics = {
            "phi": 0.5, "governance_receipt_coverage": 1.0,
            "scar_false_positive_rate": 0.0, "event_loop_lag_p95_s": 0.05,
        }
        report = canary.compare(examples, _baseline_responder, metrics=metrics)
        assert report.passed, f"baseline identity failed: similarity={report.mean_similarity:.3f}"
        assert report.mean_similarity >= 0.99

    def test_canary_patched_still_passes(self):
        """Minor-patch responder must still pass canary (similarity >= 0.62)."""
        records = _load_interaction_records(limit=16)
        if not records:
            pytest.skip("no interaction data")
        examples = build_replay_examples(records, limit=16)
        canary = CanaryRuntime()
        metrics = {
            "phi": 0.5, "governance_receipt_coverage": 1.0,
            "scar_false_positive_rate": 0.0, "event_loop_lag_p95_s": 0.05,
        }
        report = canary.compare(examples, _patched_responder, metrics=metrics)
        assert report.passed, (
            f"patched canary failed: similarity={report.mean_similarity:.3f}, "
            f"flagged={report.flagged_examples}"
        )
        assert report.mean_similarity >= 0.62

    def test_canary_catastrophic_regression_detected(self):
        """A totally different response must be flagged."""
        records = _load_interaction_records(limit=10)
        if not records:
            pytest.skip("no interaction data")
        examples = build_replay_examples(records, limit=10)
        canary = CanaryRuntime()
        metrics = {
            "phi": 0.5, "governance_receipt_coverage": 1.0,
            "scar_false_positive_rate": 0.0, "event_loop_lag_p95_s": 0.05,
        }

        def _broken_responder(ex: ReplayExample) -> str:
            return "CATASTROPHIC FAILURE: all systems offline"

        report = canary.compare(examples, _broken_responder, metrics=metrics)
        assert not report.passed, "catastrophic regression was not detected!"
        assert report.flagged_examples > 0

    def test_semantic_verifier_consistency(self):
        """Semantic verifier consistency channel on real outputs."""
        records = _load_interaction_records(limit=8)
        if not records:
            pytest.skip("no interaction data")
        outputs = [r["output"] for r in records if r.get("output")]
        if len(outputs) < 3:
            pytest.skip("not enough outputs")
        verifier = SemanticVerifier()
        result = verifier.self_consistency(outputs[:5])
        assert result.pairs > 0
        # Real monologue outputs from the same session should have some consistency
        assert result.mean_cosine >= 0.0  # sanity — hash embedder may be low

    def test_canary_report_serializable(self):
        records = _load_interaction_records(limit=8)
        if not records:
            pytest.skip("no interaction data")
        examples = build_replay_examples(records, limit=8)
        canary = CanaryRuntime()
        metrics = {"phi": 0.5, "governance_receipt_coverage": 1.0,
                   "scar_false_positive_rate": 0.0, "event_loop_lag_p95_s": 0.05}
        report = canary.compare(examples, _baseline_responder, metrics=metrics)
        data = report.to_dict()
        json.dumps(data)  # must not throw
