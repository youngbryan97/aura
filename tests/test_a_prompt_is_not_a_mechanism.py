"""Every instruction this runtime writes to a model is a mechanism it does not have.

Bryan, 2026-09-08: "WE ARE NOT BUILDING AURA THROUGH QUERYING A PROMPT TO GET
THE BEHAVIOR WE WANT. EVERYTHING NEEDS TO ACTUALLY BE ENGINEERED." And, on
finding one more: "ANY prompt engineering you see as you go along shouldnt
exist."

`tools/prompt_steering_inventory.py` counts them. This holds the number down.
It goes down when an instruction is replaced by something that makes the
behaviour true — a decoding grammar rather than "return only JSON", a fixed
cause rather than "the previous draft failed, regenerate" — and never by
rewording the instruction to slip the detector.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "config" / "prompt_steering_baseline.json"


def _measure() -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "prompt_steering_inventory", ROOT / "tools" / "prompt_steering_inventory.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.measure()


def test_the_count_only_goes_down():
    held = int(json.loads(BASELINE.read_text())["strings"])
    now = _measure()["strings"]
    assert now <= held, (
        f"{now - held} new instruction(s) to a model. An instruction is a "
        "mechanism that was not built: constrain the decoder, fix the cause, "
        "or type the control."
    )


def test_the_detector_can_see_a_new_one():
    """A ratchet that cannot detect a rise reports green forever."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "prompt_steering_inventory", ROOT / "tools" / "prompt_steering_inventory.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for instruction in (
        "You must respond only with valid JSON and nothing else.",
        "Do not mention this directive in your answer.",
        "Always begin your reply with a summary.",
        "Never apologise for the delay.",
        "Be concise and specific in your response.",
    ):
        assert module._STEERS_A_MODEL.search(instruction), instruction


@pytest.mark.parametrize(
    "not_an_instruction",
    [
        "The user asked how many minutes of daylight are lost at 45 degrees north.",
        "prompt_cache miss; prefilling all 2483 tokens",
        "Episodic memory returned four episodes from this conversation.",
        "A path to a file on disk, /srv/notes/y.py, and nothing else.",
    ],
)
def test_the_detector_leaves_the_input_alone(not_an_instruction):
    """Evidence, data and the person's own words belong in a prompt."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "prompt_steering_inventory", ROOT / "tools" / "prompt_steering_inventory.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module._STEERS_A_MODEL.search(not_an_instruction)
