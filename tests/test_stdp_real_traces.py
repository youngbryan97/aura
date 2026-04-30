"""tests/test_stdp_real_traces.py — Real Operational STDP Validation

Train on actual subsystem/tool/user-environment traces, not synthetic only.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.consciousness.stdp_learning import STDPLearningEngine
from core.consciousness.stdp_external_validation import STDPExternalValidator

TRACE_SOURCES = [
    ROOT / ".aura_runtime" / "data" / "unified_action_log.jsonl",
    ROOT / "data" / "comm_logs.jsonl",
    ROOT / "data" / "internal_monologue.jsonl",
]
N_NEURONS = 16

def _load_jsonl(path, limit=500):
    records = []
    if not path.exists(): return records
    for line in open(path, "r", encoding="utf-8"):
        line = line.strip()
        if not line: continue
        try: records.append(json.loads(line))
        except json.JSONDecodeError: continue
        if len(records) >= limit: break
    return records

def _record_to_signal(record, n=N_NEURONS):
    vec = np.zeros(n, dtype=np.float32)
    ts = float(record.get("t") or record.get("timestamp") or 0.0)
    if ts > 0:
        phase = (ts % 3600) / 3600.0
        vec[0] = 0.5 + 0.45 * np.sin(2 * np.pi * phase)
        vec[1] = 0.5 + 0.45 * np.cos(2 * np.pi * phase)
    action = str(record.get("action") or record.get("direction") or "")
    vec[2] = 0.85 if "tool" in action.lower() else (0.65 if "outbound" in action.lower() else 0.35)
    vec[3] = np.clip(hash(str(record.get("source") or record.get("target") or "")) % 1000 / 1000.0, 0.1, 0.9)
    vec[4] = 0.8 if str(record.get("gate", "")) == "approved" else 0.3
    pad = record.get("pad_state", {})
    if isinstance(pad, dict):
        vec[5], vec[6], vec[7] = float(pad.get("P", 0.5)), float(pad.get("A", 0.5)), float(pad.get("D", 0.5))
    content = str(record.get("content") or record.get("reflection") or "")
    words = content.lower().split() if content else []
    vec[8] = min(len(words) / 30.0, 1.0)
    vec[9] = len(set(words)) / max(len(words), 1) if words else 0.5
    vec[10] = sum(1 for w in words if w in {"error","fail","bug","crash"}) / max(len(words), 1)
    vec[11] = sum(1 for w in words if w in {"ok","success","done","ready"}) / max(len(words), 1)
    vec[12:16] = 0.5
    return np.clip(vec, 0.0, 1.0)

def _load_real_sequences():
    all_records = []
    for p in TRACE_SOURCES:
        all_records.extend(_load_jsonl(p, limit=500))
    if len(all_records) < 20: return None
    all_records.sort(key=lambda r: float(r.get("t") or r.get("timestamp") or 0.0))
    return np.stack([_record_to_signal(r) for r in all_records])


class TestSTDPRealTraces:
    def test_trace_sources_exist(self):
        found = sum(1 for p in TRACE_SOURCES if p.exists() and p.stat().st_size > 0)
        assert found > 0, f"no trace data found"

    def test_trace_parsing(self):
        for path in TRACE_SOURCES:
            for record in _load_jsonl(path, limit=5):
                sig = _record_to_signal(record)
                assert sig.shape == (N_NEURONS,)
                assert np.all(sig >= 0.0) and np.all(sig <= 1.0)

    def test_real_sequence_length(self):
        seq = _load_real_sequences()
        if seq is None: pytest.skip("insufficient trace data")
        assert seq.shape[0] >= 20
        assert seq.shape[1] == N_NEURONS

    def test_real_stdp_external_beats_controls(self):
        seq = _load_real_sequences()
        if seq is None: pytest.skip("insufficient trace data")
        split = seq.shape[0] // 2
        if split < 10: pytest.skip("not enough data")
        validator = STDPExternalValidator(n_neurons=N_NEURONS, seed=7)
        groups = (
            validator._run_group("external_real", seq[:split], seq[split:], train_mode="external"),
            validator._run_group("self_generated", seq[:split], seq[split:], train_mode="self"),
            validator._run_group("frozen_matrix", seq[:split], seq[split:], train_mode="frozen"),
            validator._run_group("shuffled_env", seq[:split], seq[split:], train_mode="shuffled"),
        )
        external = groups[0]
        for ctrl in groups[1:]:
            margin = ctrl.heldout_mse - external.heldout_mse
            assert margin > 0.0, f"real traces MSE ({external.heldout_mse:.4f}) didn't beat {ctrl.group} ({ctrl.heldout_mse:.4f})"

    def test_instability_budget(self):
        seq = _load_real_sequences()
        if seq is None: pytest.skip("insufficient trace data")
        split = seq.shape[0] // 2
        if split < 10: pytest.skip("not enough data")
        validator = STDPExternalValidator(n_neurons=N_NEURONS, seed=7)
        ext = validator._run_group("external_real", seq[:split], seq[split:], train_mode="external")
        assert ext.instability <= 0.30, f"instability {ext.instability:.4f} > 0.30"

    def test_synthetic_validator_still_passes(self):
        report = STDPExternalValidator(n_neurons=N_NEURONS, seed=7).run(steps=96)
        assert report.passed, f"synthetic STDP failed: margins={report.margins}"

class TestSTDPEngine:
    def test_engine_instantiates(self):
        engine = STDPLearningEngine(n_neurons=N_NEURONS)
        assert engine.get_status()["total_updates"] == 0

    def test_engine_records_and_rewards(self):
        engine = STDPLearningEngine(n_neurons=N_NEURONS)
        spike = np.random.default_rng(0).uniform(0, 1, N_NEURONS).astype(np.float32)
        engine.record_spikes(spike, t=0.0)
        dw = engine.deliver_reward(surprise=0.5, prediction_error=0.3)
        assert dw.shape == (N_NEURONS, N_NEURONS)
