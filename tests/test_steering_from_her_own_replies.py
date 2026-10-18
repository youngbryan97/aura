"""Steering directions are taken from her own replies, by the state she held when she wrote them."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "capture_steering_from_her_own_replies.py"
_spec = importlib.util.spec_from_file_location("her_own_steering", _TOOL)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

LAYOUT = ["affect.valence", "affect.arousal", "affect.dominance", "affect.engagement", "affect.curiosity",
          "affect.stress", "affect.mobilization"]


def _row(text: str, valence: float, t: float, *, curiosity: float = 0.2, stress: float = 0.15) -> dict:
    z = [valence, 0.1, 0.0, 0.1, curiosity, stress, 0.05]
    return {"model": "her-27b", "text": text, "z": z, "t": t, "prompt_sha": "same"}


def test_her_high_and_low_replies_are_paired_and_the_latest_held_out():
    rows = [_row(f"glad reply number {i} " + "word " * i, 0.9, float(i)) for i in range(20)]
    rows += [_row(f"flat reply number {i} " + "word " * i, 0.0, float(i) + 0.5) for i in range(20)]
    rows += [_row(f"middle reply {i}", 0.45, 100.0 + i) for i in range(10)]
    chosen = tool.contrasts(rows, LAYOUT, model="her-27b")["valence_positive"]
    assert chosen["train"] and all("glad" in a and "flat" in b for a, b in chosen["train"])
    # Paired by length, so the contrast is not how much she wrote.
    assert all(abs(len(a) - len(b)) <= 3 for a, b in chosen["train"])


def test_a_reading_that_does_not_vary_gives_no_direction():
    rows = [_row(f"reply {i} with words", 0.3, float(i)) for i in range(40)]
    assert "does not vary" in tool.contrasts(rows, LAYOUT, model="her-27b")["valence_positive"]["reason"]


def test_other_models_and_repeats_are_left_out():
    rows = [_row("same words twice", 0.9, 1.0), _row("same words twice", 0.0, 2.0),
            {**_row("someone else's reply", 0.9, 3.0), "model": "other"}]
    chosen = tool.contrasts(rows, LAYOUT, model="her-27b")["valence_positive"]
    assert "reason" in chosen or not chosen["train"]
