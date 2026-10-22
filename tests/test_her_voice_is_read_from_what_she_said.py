"""The fifth criterion is read from replies she actually gave.

The fusion certificate holds only when her stored replies are no less likely
under the steering than without it. No tool scored that until 21 October 2026,
so every certificate measured since the criterion was written was refused for
want of a reading, and the reports ground could not test whether her state
reaches her words through the forward pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.consciousness import fusion_probe
from tools.measure_fusion_channel import her_replies

MODEL = "Aura-Qwen3.8-27B-persona-test"


def _pairs(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _row(text: str, **overrides) -> dict:
    row = {"model": MODEL, "lane": "foreground_chat", "s": [0.1, -0.2, 0.3], "text": text}
    row.update(overrides)
    return row


SPOKEN = "The wrens fledged this morning and the nest is quiet now, which I keep noticing."


def test_only_her_spoken_replies_with_this_checkpoint_are_read(tmp_path):
    path = _pairs(tmp_path / "pairs.jsonl", [
        _row(SPOKEN),
        _row(SPOKEN, model="some-other-checkpoint"),
        _row(SPOKEN, lane="background_reflection"),
        _row(SPOKEN, s=[]),
        _row("Too short to carry a register."),
        _row('{"actions": [{"index": 0, "type": "click"}], "why": "pressed it four times already"}'),
    ])
    found = her_replies(path, MODEL, 10, seed=1)
    assert found == [(SPOKEN, [0.1, -0.2, 0.3])]


def test_the_draw_is_seeded_and_bounded(tmp_path):
    rows = [_row(f"{SPOKEN} Reply number {index}.") for index in range(20)]
    path = _pairs(tmp_path / "pairs.jsonl", rows)
    assert her_replies(path, MODEL, 5, seed=3) == her_replies(path, MODEL, 5, seed=3)
    assert len(her_replies(path, MODEL, 5, seed=3)) == 5
    assert her_replies(tmp_path / "absent.jsonl", MODEL, 5, seed=3) == []


class _Hook:
    def __init__(self) -> None:
        self.updates: list[np.ndarray] = []

    def update_substrate_vector(self, vector, *, source: str) -> None:
        self.updates.append(np.asarray(vector))


class _Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return [len(word) for word in text.split()]


def test_the_reading_is_steered_minus_plain_per_token(monkeypatch):
    alphas: list[float] = []
    # Steered paths score -1 a token and plain ones -3, so the reading is +2.
    monkeypatch.setattr(fusion_probe, "chat_ids", lambda tokenizer, prompt: [0])
    monkeypatch.setattr(
        fusion_probe, "_log_likelihood",
        lambda model, prompt, tokens: (-1.0 if alphas[-1] else -3.0) * len(tokens),
    )
    hook = _Hook()
    delta, scored = fusion_probe.voice_reading(
        None, _Tokenizer(), [hook], alphas.append, 0.2,
        [(SPOKEN, [0.5, 0.5]), ("one", [0.5]), (SPOKEN, [float("nan")])],
    )
    assert scored == 1
    assert delta == 2.0
    # The hooks were set to the state she said it in, and the steering ends off.
    assert np.allclose(hook.updates[-1], [0.5, 0.5])
    assert alphas[-1] == 0.0


def test_nothing_to_read_is_no_reading():
    assert fusion_probe.voice_reading(None, _Tokenizer(), [], lambda alpha: None, 0.2, []) == (None, 0)
