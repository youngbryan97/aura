"""A way on that the screen reading got a letter or two wrong is still known for one.

LIVE 2026-10-11 a lander's crash screen said OUCH! and TRY AGAIN. The button was read as "TRY ABAIN", "UUTRY-AGAIN"
and "try againfu m" on three looks; none was known for a way back, and each time she pressed keys at the screen for
over a minute before clicking it.
"""
from __future__ import annotations

import pytest

from core.language.a_way_on import how_much_it_leads_on, label_is_a_way_on, nearly_says
from core.skills.screen_pursuit_bearings import restart_controls

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("read", ["TRY ABAIN", "UUTRY-AGAIN", "try againfu m", "TRY AGAlN", "Try Again!"])
def test_a_garbled_try_again_is_a_way_back(read: str) -> None:
    assert nearly_says(read) == "try again"
    assert how_much_it_leads_on(read) > 1.0
    assert restart_controls({"layout": [{"text": read}]})


@pytest.mark.parametrize("read", ["OUCH!", "LANDINGS: 0", "SUCCESS RATE: 0%", "CRASHES: 3", "Trampoline", "Play the piano", "Training"])
def test_other_words_are_not_taken_for_one(read: str) -> None:
    assert nearly_says(read) == ""
    assert not restart_controls({"layout": [{"text": read}]})


@pytest.mark.parametrize('label', ['BTART', 'NEX7', 'PL4Y', 'RESUMEe', 'STAR', 'BEGlN', 'ACCEFT', 'TRY ABAIN'])
def test_whole_damaged_control_labels_have_explicit_ocr_provenance(label):
    assert not label_is_a_way_on(label)
    assert label_is_a_way_on(label, from_ocr=True)


@pytest.mark.parametrize('label', ['Display', 'Play the piano', 'START OVER', 'BTART:', 'SCORE 120',
                                  'Time left: 12', 'NEXT ROOM IS DARK', 'FIRE', 'LEFT', 'UP', 'METER', 'XXSTART'])
def test_ocr_tolerance_keeps_whole_label_readout_and_destructive_reset_boundaries(label):
    assert not label_is_a_way_on(label, from_ocr=True)


def test_ocr_tolerance_requires_one_unambiguous_expected_label(monkeypatch):
    import core.language.a_way_on as module

    monkeypatch.setattr(module, '_ONE_WORD', ('start', 'smart'))
    assert not module.label_is_a_way_on('SCART', from_ocr=True)
