"""Resonant mask treatments that preserve speech timing and pitch."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def treat_voice(samples: np.ndarray, sample_rate: int, treatment: str = "clean") -> np.ndarray:
    """Process a mono clause without changing its length or sample rate."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("Voice treatment requires finite mono audio")
    if treatment == "clean" or not audio.size:
        return audio.copy()
    if treatment not in {"veil", "veil_deep"}:
        raise ValueError(f"Unknown voice treatment: {treatment}")
    if sample_rate < 16000:
        raise ValueError("Voice treatment requires at least 16 kHz")

    # Keep consonants above the telephone band. Short reflections supply the
    # enclosure; no pitch shifting or noise layer covers the underlying voice.
    deep = treatment == "veil_deep"
    upper = min(6200 if deep else 7200, sample_rate * .45)
    filtered = sosfilt(butter(2, [85, upper], btype="bandpass", fs=sample_rate, output="sos"), audio)
    chest = sosfilt(butter(1, [190, 460], btype="bandpass", fs=sample_rate, output="sos"), filtered)
    body = filtered + .16 * chest
    if deep:
        presence = sosfilt(butter(1, [850, 2200], btype="bandpass", fs=sample_rate, output="sos"), filtered)
        body += .18 * presence
    wet = body.copy()
    amount = .17 if treatment == "veil" else .29
    for delay_ms, gain in ((2.6, amount), (5.3, -amount * .65), (8.1, amount * .3)):
        delay = max(1, round(sample_rate * delay_ms / 1000))
        wet[delay:] += gain * body[:-delay]
    if deep:
        # A small parallel ring component supplies the electronic texture.
        # The original carrier remains dominant, so vowels keep their pitch.
        phase = np.arange(audio.size, dtype=np.float32) * (2 * np.pi * 52 / sample_rate)
        wet *= .86 + .14 * np.cos(phase)
    wet = .82 * np.tanh(wet * 1.35)
    dry = .08 if deep else .26
    mixed = dry * audio + (1 - dry) * wet
    # The same transfer curve applies to the whole clause; there is no
    # per-block normalisation that could pump at streaming boundaries.
    return np.clip(mixed, -.97, .97).astype(np.float32)
