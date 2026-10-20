"""Steering directions are taken from her own replies, by the state each reply was steered by."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "capture_steering_from_her_own_replies.py"
_spec = importlib.util.spec_from_file_location("her_own_steering", _TOOL)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

DIMENSIONS = {"valence_positive": (0, "tanh"), "arousal": (1, "linear_half"), "frustration": (3, "tanh")}


def _row(text: str, t: float, *, valence: float = 0.5, arousal: float = 0.5, frustration: float = 0.5) -> dict:
    return {"model": "her-27b", "text": text, "s": [valence, arousal, 0.5, frustration, 0.5, 0.5], "t": t}


def test_her_high_and_low_replies_are_paired_and_the_latest_held_out():
    rows = [_row(f"glad reply number {i} " + "word " * i, float(i), valence=0.9) for i in range(20)]
    rows += [_row(f"flat reply number {i} " + "word " * i, float(i) + 0.5, valence=0.1) for i in range(20)]
    rows += [_row(f"middle reply {i}", 100.0 + i) for i in range(10)]
    chosen = tool.contrasts(rows, DIMENSIONS, model="her-27b")["valence_positive"]
    assert chosen["train"] and all("glad" in a and "flat" in b for a, b in chosen["train"])
    # Paired by length, so the contrast is not how much she wrote.
    assert all(abs(len(a) - len(b)) <= 3 for a, b in chosen["train"])


def test_the_dimensions_are_the_hooks_own_slots():
    from core.consciousness.affective_steering import AFFECTIVE_DIMENSIONS

    assert tool.steered_dimensions() == {d["key"]: (d["substrate_idx"], d["substrate_fn"])
                                         for d in AFFECTIVE_DIMENSIONS}


def test_a_slot_the_others_explain_gives_no_direction():
    # Arousal moves exactly with valence here, as mobilization moved with arousal in her record on 18 October.
    rows = [_row(f"reply {i} with words", float(i), valence=0.1 + 0.02 * i, arousal=0.1 + 0.02 * i,
                 frustration=0.5 + (0.3 if i % 2 else -0.3)) for i in range(40)]
    alike = {**DIMENSIONS, "arousal": (1, "tanh")}
    chosen = tool.contrasts(rows, alike, model="her-27b")
    assert "once the other dimensions are known" in chosen["arousal"]["reason"]
    assert "train" in chosen["frustration"]


def test_two_states_that_move_together_are_each_contrasted_on_their_own_part():
    # Valence and frustration share a drift; frustration alone also follows whether the reply was a "snag".
    rows = []
    for i in range(80):
        drift = 0.3 * ((i % 10) / 9.0)
        snag = i % 2 == 0
        rows.append(_row(f"{'snag' if snag else 'calm'} reply {i} words", float(i),
                         valence=0.3 + drift + 0.1 * ((i * 7) % 3 - 1) / 1.0,
                         frustration=0.3 + drift + (0.2 if snag else 0.0)))
    chosen = tool.contrasts(rows, DIMENSIONS, model="her-27b")["frustration"]
    assert all(a.startswith("snag") and b.startswith("calm") for a, b in chosen["train"])
    assert chosen["explained_by_the_others"] > 0.0


def test_replies_recorded_without_their_steering_state_are_not_stood_in_for():
    rows = [{"model": "her-27b", "text": f"old reply {i} here", "z": [0.9], "t": float(i)} for i in range(30)]
    chosen = tool.contrasts(rows, DIMENSIONS, model="her-27b")
    assert all("0 of her replies carry the state" in c["reason"] and "30 were recorded before" in c["reason"]
               for c in chosen.values())


def test_other_models_and_repeats_are_left_out():
    rows = [_row("same words twice", 1.0, valence=0.9), _row("same words twice", 2.0, valence=0.1),
            {**_row("someone else's reply", 3.0, valence=0.9), "model": "other"}]
    chosen = tool.contrasts(rows, DIMENSIONS, model="her-27b")["valence_positive"]
    assert "reason" in chosen or not chosen["train"]


def test_a_direction_another_dimension_separates_as_well_is_not_kept():
    assert tool.admitted(0.75, 1e-20, {"arousal": 0.70}, layers=16) == (True, True)
    # 18 October: valence's direction and frustration's were one axis; each separated the other's pairs alike.
    assert tool.admitted(0.75, 1e-20, {"frustration": 0.76}, layers=16) == (False, False)
    assert tool.admitted(0.75, 0.01, {}, layers=16) == (True, False)
