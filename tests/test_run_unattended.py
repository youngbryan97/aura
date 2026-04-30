"""Tests for training/run_unattended.py — exercises real behavior
without invoking mlx_lm or any heavy training step."""
from __future__ import annotations

import importlib
import json
import signal
import sys
from pathlib import Path

import pytest

TRAINING_DIR = Path(__file__).resolve().parent.parent / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


@pytest.fixture
def orch(tmp_path, monkeypatch):
    """Fresh import per test, ADAPTER_DIR redirected to tmp_path."""
    sys.modules.pop("run_unattended", None)
    mod = importlib.import_module("run_unattended")
    adapter_dir = tmp_path / "adapters" / "aura-personality"
    adapter_dir.mkdir(parents=True)
    monkeypatch.setattr(mod, "ADAPTER_DIR", adapter_dir)
    monkeypatch.setattr(mod, "STATE_FILE", adapter_dir / "training_state.json")
    return mod


def _make_ckpt(adapter_dir: Path, n: int) -> Path:
    p = adapter_dir / f"{n:07d}_adapters.safetensors"
    p.write_bytes(b"\x00")
    return p


def test_state_recorded_after_synthetic_checkpoint(orch):
    started_at = "2026-04-30T00:00:00+0000"
    state = orch.update_state(started_at=started_at)
    assert state["last_iter"] == 0
    assert state["last_checkpoint_path"] is None
    assert state["started_at"] == started_at
    assert orch.STATE_FILE.exists()

    _make_ckpt(orch.ADAPTER_DIR, 250)
    state2 = orch.update_state(started_at=started_at)
    assert state2["last_iter"] == 250
    assert state2["last_checkpoint_path"].endswith("0000250_adapters.safetensors")
    # started_at sticks across re-spawns.
    assert state2["started_at"] == started_at
    assert json.loads(orch.STATE_FILE.read_text())["last_iter"] == 250


def test_resume_detection_picks_latest_checkpoint(orch):
    assert orch.has_partial_run() is False
    _make_ckpt(orch.ADAPTER_DIR, 250)
    _make_ckpt(orch.ADAPTER_DIR, 1000)
    _make_ckpt(orch.ADAPTER_DIR, 750)
    assert orch.has_partial_run() is True
    ckpt, n = orch.latest_checkpoint()
    assert n == 1000
    assert ckpt.name == "0001000_adapters.safetensors"


def test_clean_shutdown_writes_final_snapshot(orch):
    started_at = "2026-04-30T01:02:03+0000"
    orch._install_signal_handlers(started_at)
    _make_ckpt(orch.ADAPTER_DIR, 500)

    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler)
    handler(signal.SIGTERM, None)

    persisted = json.loads(orch.STATE_FILE.read_text())
    assert persisted["phase"] == "signal_exit"
    assert persisted["last_signal"] == int(signal.SIGTERM)
    assert persisted["last_iter"] == 500
    assert persisted["last_heartbeat"]
    assert orch._shutdown.is_set()


def test_dryrun_short_circuits_clean(orch, capsys):
    rc = orch.main(["--skip-train", "--skip-dataset", "--tag", "dryrun-test"])
    assert rc == 0
    persisted = json.loads(orch.STATE_FILE.read_text())
    assert persisted["phase"] == "dryrun_done"
    assert "dryrun mode" in capsys.readouterr().out


def test_is_dryrun_requires_all_three_flags(orch):
    assert orch.is_dryrun(orch.parse_args(
        ["--skip-train", "--skip-dataset", "--tag", "dryrun-x"])) is True
    assert orch.is_dryrun(orch.parse_args(
        ["--skip-train", "--tag", "dryrun-x"])) is False
    assert orch.is_dryrun(orch.parse_args(
        ["--skip-train", "--skip-dataset", "--tag", "real-run"])) is False
