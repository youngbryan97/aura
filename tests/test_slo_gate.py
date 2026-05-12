"""Tests for the SLO comparator and baseline format."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from slo.check import compare
from slo.measure import measure_doctor_bundle_p95_ms


def _baseline(slos, hard_limits=None):
    return {
        "schema_version": 1,
        "slos": slos,
        "hard_limits": hard_limits or {},
    }


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------
def test_compare_passes_when_within_tolerance():
    baseline = _baseline(
        {"x_p95_ms": {"value": 100.0, "unit": "ms", "tolerance_pct": 50}}
    )
    measured = {"x_p95_ms": {"value": 110.0, "unit": "ms"}}
    report = compare(baseline, measured)
    assert report["ok"] is True
    assert report["results"][0]["ok"] is True


def test_compare_passes_at_tolerance_boundary():
    baseline = _baseline(
        {"x_p95_ms": {"value": 100.0, "unit": "ms", "tolerance_pct": 50}}
    )
    measured = {"x_p95_ms": {"value": 150.0, "unit": "ms"}}
    report = compare(baseline, measured)
    assert report["ok"] is True


# ---------------------------------------------------------------------------
# soft regression
# ---------------------------------------------------------------------------
def test_compare_fails_when_soft_regressed():
    baseline = _baseline(
        {"x_p95_ms": {"value": 100.0, "unit": "ms", "tolerance_pct": 20}}
    )
    measured = {"x_p95_ms": {"value": 150.0, "unit": "ms"}}
    report = compare(baseline, measured)
    assert report["ok"] is False
    assert "soft regression" in report["results"][0]["reason"]


# ---------------------------------------------------------------------------
# hard limit
# ---------------------------------------------------------------------------
def test_compare_fails_when_hard_limit_exceeded():
    baseline = _baseline(
        {"x_p95_ms": {"value": 1000.0, "unit": "ms", "tolerance_pct": 100}},
        hard_limits={"x_p95_ms": 1500.0},
    )
    measured = {"x_p95_ms": {"value": 1700.0, "unit": "ms"}}
    report = compare(baseline, measured)
    assert report["ok"] is False
    assert "hard limit exceeded" in report["results"][0]["reason"]


# ---------------------------------------------------------------------------
# correctness scores
# ---------------------------------------------------------------------------
def test_correctness_score_uses_hard_limit_only():
    baseline = _baseline(
        {"brier": {"value": 0.0, "unit": "score", "tolerance_pct": 50}},
        hard_limits={"brier": 0.001},
    )
    # Slightly worse correctness still under hard limit -> ok
    measured_ok = {"brier": {"value": 0.0005, "unit": "score"}}
    assert compare(baseline, measured_ok)["ok"] is True

    # Above hard limit -> fail
    measured_bad = {"brier": {"value": 0.01, "unit": "score"}}
    bad = compare(baseline, measured_bad)
    assert bad["ok"] is False
    assert "correctness regressed" in bad["results"][0]["reason"]


# ---------------------------------------------------------------------------
# missing data
# ---------------------------------------------------------------------------
def test_missing_measurement_fails():
    baseline = _baseline(
        {"x_p95_ms": {"value": 100.0, "unit": "ms", "tolerance_pct": 50}}
    )
    report = compare(baseline, measured={})
    assert report["ok"] is False
    assert report["results"][0]["reason"] == "measurement missing"


# ---------------------------------------------------------------------------
# baseline file shape
# ---------------------------------------------------------------------------
def test_repository_baseline_has_required_shape():
    """Sanity check on the committed baseline so a malformed file is caught."""
    baseline_path = Path(__file__).resolve().parent.parent / "slo" / "baseline.json"
    assert baseline_path.exists(), "slo/baseline.json must be committed"
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert payload.get("schema_version") == 1
    assert "slos" in payload and isinstance(payload["slos"], dict)
    assert "hard_limits" in payload
    for name, slo in payload["slos"].items():
        assert "value" in slo, f"{name} missing 'value'"
        assert "unit" in slo, f"{name} missing 'unit'"
        assert "tolerance_pct" in slo, f"{name} missing 'tolerance_pct'"


def test_repository_baseline_covers_known_slos():
    baseline_path = Path(__file__).resolve().parent.parent / "slo" / "baseline.json"
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    expected = {
        "audit_chain_append_p95_ms",
        "audit_chain_verify_per_entry_us",
        "prediction_ledger_register_p95_ms",
        "prediction_ledger_brier_correctness",
        "mutation_eval_passed_p95_ms",
        "doctor_bundle_p95_ms",
    }
    assert set(payload["slos"].keys()) == expected


def test_doctor_bundle_measurement_uses_isolated_receipt_store(monkeypatch):
    import core.runtime.diagnostics_bundle as diagnostics_bundle
    import core.runtime.receipts as receipts

    resets = []
    store_roots = []
    bundle_calls = []

    def fake_reset_receipt_store():
        resets.append("reset")

    def fake_get_receipt_store(root=None):
        store_roots.append(Path(root) if root is not None else None)
        return object()

    def fake_build_bundle(*, output_path=None, workspace=None):
        bundle_calls.append((output_path, workspace))
        return {"ok": True}

    monkeypatch.setattr(receipts, "reset_receipt_store", fake_reset_receipt_store)
    monkeypatch.setattr(receipts, "get_receipt_store", fake_get_receipt_store)
    monkeypatch.setattr(diagnostics_bundle, "build_bundle", fake_build_bundle)

    measured = measure_doctor_bundle_p95_ms(samples=1, warmup=1)

    assert measured >= 0.0
    assert len(bundle_calls) == 2
    assert len(resets) == 2
    assert len(store_roots) == 1
    assert store_roots[0] is not None
    assert store_roots[0].name == "receipts"
