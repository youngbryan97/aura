"""A spawn backoff ends when its seconds have passed on the machine, however her clock moves.

LIVE 2026-10-30, seed 101's reports run at aeefdca31: at 18:56 her cortex's spawn was refused once, and the refusal set
a ten-second backoff stamped with `time.time`. In a run `time.time` is her clock (core/runtime/wall_clock.py), and
every arm restores its anchor and rewinds it, so the ten seconds never passed: every later attempt logged that the
backoff stood, every arm waited 900 s and ended in the failure floor, and the run had to be resumed in a fresh process.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

import core.brain.llm.mlx_client as mlx_client
import core.brain.llm.mlx_client_worker_lifecycle as lifecycle
from core.brain.llm.mlx_client import MLXLocalClient

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
TEST_MODEL = "/models/Qwen2.5-7B-Instruct-4bit"


def test_every_backoff_stamp_and_read_uses_the_machines_clock():
    offenders = []
    for path in sorted((ROOT / "core/brain/llm").glob("mlx_client*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.Compare, ast.If)):
                continue
            text = ast.unparse(node.test if isinstance(node, ast.If) else node)
            if "_spawn_backoff_until" in text and "time.time()" in text:
                offenders.append(f"{path.name}:{node.lineno}: {text[:90]}")
    assert offenders == []


def test_a_backoff_set_while_her_clock_rewinds_still_ends(monkeypatch):
    machine = [1000.0]
    for module in (mlx_client, lifecycle):
        monkeypatch.setattr(module, "monotonic_time", lambda: machine[0])
    her_clock = [5.0]
    monkeypatch.setattr(time, "time", lambda: her_clock[0])

    client = MLXLocalClient(model_path=TEST_MODEL)
    monkeypatch.setattr(client, "_set_lane_state", lambda *a, **k: None)
    monkeypatch.setattr(client, "_record_degraded_event", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_record_mlx_degradation", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(mlx_client, "_record_mlx_degradation", lambda *a, **k: None)
    client._ensure_worker_alive_inner_bug_fix_exponential(
        0, "memory_pressure_refused_worker_spawn", RuntimeError("refused"), foreground_request=False
    )
    assert client._spawn_backoff_cause == "spawn_failure"
    assert client._spawn_backoff_until == pytest.approx(1010.0)

    her_clock[0] = 4.0  # a restore rewinds her clock
    machine[0] = 1011.0  # eleven seconds pass on the machine
    assert machine[0] >= client._spawn_backoff_until
