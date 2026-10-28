"""The battery's positive control clears the differentiation line, on seeds it was not tuned on.

`beats_every_null` needs the recurrent reference to pass the conjunction the
nulls fail, so that the battery is shown able to say yes. On seed 89, 28
October, it read a normalised effective dimension of 0.4002 against a bar below
0.40 and the line failed with no null passing: across seeds 1 to 200 the
reference failed the differentiation line on 62. Its domains are now recorded
wider than the null family's (`REFERENCE_WIDTH`, core/subject/nulls.py), still
driven by one shock each.
"""
from __future__ import annotations

import pytest

import core.subject.nulls as nulls
from core.subject.battery import THRESHOLDS
from core.subject.differentiation import effective_dimension
from core.subject.nulls import ARCHITECTURES, architecture, toy_recording
from core.subject.state import DOMAINS

#: Seeds the width was not chosen on: it was read off seeds 1 to 200.
HELD_OUT = tuple(range(201, 225))


def _reading(seed: int):
    return effective_dimension(toy_recording(architecture("recurrent", seed=seed), steps=2500, seed=seed))


@pytest.mark.parametrize("seed", HELD_OUT)
def test_the_reference_is_differentiated_on_a_seed_it_was_not_tuned_on(seed):
    reading = _reading(seed)
    assert reading.normalised < THRESHOLDS["d_eff_normalised"], reading.normalised
    assert reading.top_share < THRESHOLDS["component_share"], reading.top_share


def test_at_the_null_family_width_seed_89_fails_as_its_campaign_did(monkeypatch):
    monkeypatch.setattr(nulls, "REFERENCE_WIDTH", 4)
    monkeypatch.setattr(nulls, "_REFERENCES", {})
    assert _reading(89).normalised == pytest.approx(0.4002, abs=5e-5)


def test_only_the_reference_is_built_wider():
    assert set(architecture("recurrent", seed=3).widths.values()) == {nulls.REFERENCE_WIDTH}
    for name in ARCHITECTURES:
        if name != "recurrent":
            assert set(architecture(name, seed=3).widths.values()) == {4}, name
    # A caller that asks for widths gets them, the reference included.
    asked = {key: 3 for key in DOMAINS}
    assert architecture("recurrent", seed=3, widths=asked).widths == asked
