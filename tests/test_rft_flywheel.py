"""RFT flywheel — verifier-clean derivations become a training dataset, gated.

The loop only ever compounds verified signal, and never launches a Cortex train
beside the live instance: the gate must fail-closed on insufficient data or a
failed preflight.
"""
from __future__ import annotations

import json
from pathlib import Path

import training.rft_flywheel as flywheel


def _row(prompt, chosen, confidence=1.0):
    return {"prompt": prompt, "chosen": chosen, "rejected": "wrong", "confidence": confidence}


class _Harness:
    def __init__(self, rows, pending=0):
        self._rows = rows
        self._pending = pending

    def export_dpo_rows(self, *, limit=1000):
        return self._rows[:limit]

    def pending_count(self):
        return self._pending


def test_gather_filters_confidence_and_dedups(monkeypatch):
    rows = [
        _row("2+2?", "It is 4.", 0.9),
        _row("2+2?", "It is 4.", 0.9),          # exact dup — dropped
        _row("cap of France?", "Paris.", 0.2),  # below floor — dropped
        _row("3*3?", "Nine.", 0.8),
    ]
    monkeypatch.setattr(
        flywheel, "get_verifiable_preference_harness", lambda: _Harness(rows), raising=False
    )
    monkeypatch.setattr(
        "core.learning.verifiable_preference_harness.get_verifiable_preference_harness",
        lambda: _Harness(rows),
    )
    out = flywheel.gather_verified_rows()
    prompts = [m["messages"][1]["content"] for m in out]
    assert prompts == ["2+2?", "3*3?"]
    for m in out:
        assert m["messages"][0]["role"] == "system"
        assert m["messages"][2]["role"] == "assistant"


def test_build_dataset_writes_split_and_manifest(monkeypatch, tmp_path):
    rows = [_row(f"q{i}?", f"a{i}", 0.9) for i in range(40)]
    monkeypatch.setattr(
        "core.learning.verifiable_preference_harness.get_verifiable_preference_harness",
        lambda: _Harness(rows),
    )
    manifest = flywheel.build_dataset(output_dir=tmp_path)
    assert manifest["total_rows"] == 40
    assert manifest["train_rows"] + manifest["valid_rows"] == 40
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "valid.jsonl").exists()
    # Every written row is valid chat-format JSON.
    for line in (tmp_path / "train.jsonl").read_text().splitlines():
        obj = json.loads(line)
        assert [m["role"] for m in obj["messages"]] == ["system", "user", "assistant"]


def test_gate_fails_closed_without_enough_data(monkeypatch):
    rows = [_row(f"q{i}?", f"a{i}", 0.9) for i in range(5)]  # < _MIN_ROWS
    monkeypatch.setattr(
        "core.learning.verifiable_preference_harness.get_verifiable_preference_harness",
        lambda: _Harness(rows, pending=5),
    )
    monkeypatch.setattr(
        flywheel, "training_preflight", lambda **_k: {"passed": True}, raising=False
    )
    import training.train_and_fuse as tf

    monkeypatch.setattr(
        tf,
        "training_preflight",
        lambda **_k: (_ for _ in ()).throw(AssertionError("preflight should not run")),
    )
    monkeypatch.setattr(
        tf,
        "get_default_base_model",
        lambda: (_ for _ in ()).throw(AssertionError("model should not resolve")),
    )
    gate = flywheel.flywheel_gate()
    assert gate["enough_data"] is False
    assert gate["preflight"]["reason"] == "insufficient_verified_rows"
    assert gate["ready"] is False


def test_gate_ready_only_when_data_and_preflight_pass(monkeypatch):
    rows = [_row(f"q{i}?", f"a{i}", 0.9) for i in range(flywheel._MIN_ROWS + 5)]
    monkeypatch.setattr(
        "core.learning.verifiable_preference_harness.get_verifiable_preference_harness",
        lambda: _Harness(rows, pending=100),
    )
    import training.train_and_fuse as tf

    monkeypatch.setattr(tf, "training_preflight", lambda **_k: {"passed": True})
    monkeypatch.setattr(tf, "get_default_base_model", lambda: Path("/models/cortex"))
    gate = flywheel.flywheel_gate()
    assert gate["enough_data"] is True
    assert gate["preflight_passed"] is True
    assert gate["ready"] is True

    # Preflight failure closes the gate even with plenty of data.
    monkeypatch.setattr(
        tf, "training_preflight", lambda **_k: {"passed": False, "reason": "live_running"}
    )
    assert flywheel.flywheel_gate()["ready"] is False
