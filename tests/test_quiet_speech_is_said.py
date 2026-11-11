"""U02: speech her voice gate heard and her loudness gate dropped is said in her log, not only at debug.

LIVE 2026-11-10: "Hey Aura, what is two plus two?" played through the laptop's speakers at volume 45 crossed her
voice gate and fell under the -35 dB utterance gate, and her log said nothing; at 70 it was transcribed (-30.4 dB).
"She does not answer when I talk" had no line to read. A quiet stretch with no voice in it stays at debug, since
every second and a half of silence passes the same gate.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from core.senses.voice_engine import SAMPLE_RATE, SovereignVoiceEngine

pytestmark = pytest.mark.unit


def _quiet(seconds: float, level: float) -> bytes:
    samples = (np.sin(np.linspace(0, 2 * np.pi * 220 * seconds, int(SAMPLE_RATE * seconds))) * level * 32767)
    return samples.astype(np.int16).tobytes()


@pytest.mark.parametrize("voiced, level", [(True, logging.INFO), (False, logging.DEBUG)])
def test_speech_under_the_loudness_gate_is_said_when_it_was_speech(caplog, voiced, level):
    engine = object.__new__(SovereignVoiceEngine)
    with caplog.at_level(logging.DEBUG, logger="Aura.VoiceEngine"):
        engine._process_transcript(_quiet(2.0, 0.015), voiced=voiced)
    said = [r for r in caplog.records if "under the -35.0 dB gate" in r.getMessage()]
    assert said and said[0].levelno == level
    assert ("of speech" in said[0].getMessage()) is voiced
