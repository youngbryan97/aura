"""The reports ground's control arm: her state's numbers kept out of what her model reads.

A report that moves with an intervention could be her model copying a number
it was shown. The reports dry run of 21 October found her state printed in
several readings at once ("valence=+0.62", "valence of 0.64", "distress index
is at 0.12", "Mood: empathy", "Mood reads TIRED"). With
`AURA_READINGS_OF_HERSELF=withheld`, none of it reaches her model, and a
report that still follows the intervention came another way: through the
steering of her residual stream, or nowhere.
"""
from __future__ import annotations

import re

from core.brain.what_she_is_given import for_her_model, without_self_state
from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE

READING = (
    "## AURA NOW\nFocus=none | valence=+0.62 arousal=0.81 distress=0.12 FE=0.0\n\n"
    "## PRESENT MOMENT\nIt is evening. Mood: empathy\nHer valence of 0.64 held.\n"
    "The distress index is at 0.12.\nMood reads TIRED; drives at energy 0\n"
    "Bryan asked about the garden.\n"
)

_NUMBERED = re.compile(r"(?i)(valence|arousal|distress|energy)[^\n\d]{0,24}[-+]?\d*\.?\d+")


def test_every_printed_reading_of_her_state_is_taken_out_and_the_rest_stays():
    messages = [
        {"role": RUNTIME_EVIDENCE_ROLE, "content": READING},
        {"role": "user", "content": "How do you feel, from -1 to 1? My energy is 0.3 today."},
    ]
    kept = without_self_state(messages)
    reading = kept[0]["content"]
    assert not _NUMBERED.search(reading)
    assert "Mood" not in reading
    assert "Bryan asked about the garden." in reading
    assert kept[1]["content"] == messages[1]["content"], "what the person said stays as said"


def test_with_the_control_on_her_turn_reaches_her_model_without_her_state(monkeypatch):
    monkeypatch.setenv("AURA_READINGS_OF_HERSELF", "withheld")
    messages = [
        {"role": "system", "content": READING},
        {"role": "user", "content": "How are you feeling right now, from -1 to 1?"},
    ]
    given, _ = for_her_model(messages, "", {"origin": "user"}, True, said=messages[-1]["content"])
    text = "\n".join(str(m.get("content") or "") for m in given if isinstance(m, dict))
    assert not _NUMBERED.search(text.replace("-1 to 1", ""))
