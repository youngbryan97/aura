"""SelfObject — snapshot, calibration, bias detection."""
from __future__ import annotations

from core.identity.self_object import get_self


def test_snapshot_returns_consistent_continuity_hash():
    a = get_self().snapshot()
    b = get_self().snapshot()
    # The continuity hash is driven by self-relevant fields; it should be
    # stable when sampled twice in quick succession.
    assert a.continuity_hash == b.continuity_hash


def test_calibrate_perfect_report_yields_score_one():
    snap = get_self().snapshot().as_dict()
    report = {k: v for k, v in snap.items() if not isinstance(v, (list, dict))}
    result = get_self().calibrate(report)
    assert result["score"] >= 0.95


def test_introspect_returns_focus_field():
    out = get_self().introspect("current_action")
    assert out["focus"] == "current_action"
    assert "viability" in out


def test_debug_bias_returns_list():
    biases = get_self().debug_bias()
    assert isinstance(biases, list)
