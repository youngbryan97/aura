"""Which words a speaker stressed, measured against the rest of what they said.

Paralinguistics read an utterance as a whole, so "I did not say that" said with
the weight on "not" and said evenly were the same reading. These pin the
per-word measure on synthetic speech whose answer is known.
"""

from __future__ import annotations

import numpy as np

from core.voice.duplex.paralinguistics import stressed_words

RATE = 16_000


def _speech(words: list[tuple[str, float, float, float]]) -> tuple[np.ndarray, list[tuple[str, float, float]]]:
    """Each word a tone: (text, seconds, hertz, amplitude), with a short gap after."""
    pieces: list[np.ndarray] = []
    spans: list[tuple[str, float, float]] = []
    clock = 0.0
    for text, seconds, hertz, amplitude in words:
        t = np.arange(int(RATE * seconds)) / RATE
        pieces.append((amplitude * np.sin(2 * np.pi * hertz * t)).astype(np.float32))
        spans.append((text, clock, clock + seconds))
        clock += seconds
        gap = np.zeros(int(RATE * 0.08), dtype=np.float32)
        pieces.append(gap)
        clock += 0.08
    return np.concatenate(pieces), spans


def test_the_word_said_higher_louder_and_longer_is_the_one_leaned_on() -> None:
    signal, spans = _speech(
        [
            ("i", 0.20, 150, 0.2),
            ("did", 0.25, 150, 0.2),
            ("not", 0.60, 230, 0.6),
            ("say", 0.25, 150, 0.2),
            ("that", 0.25, 150, 0.2),
        ]
    )
    assert stressed_words(signal, RATE, spans) == ("not",)


def test_an_utterance_said_evenly_leans_on_nothing() -> None:
    signal, spans = _speech([(w, 0.25, 150, 0.2) for w in ("we", "can", "talk", "about", "it")])
    assert stressed_words(signal, RATE, spans) == ()


def test_too_few_words_give_nothing_to_compare_against() -> None:
    signal, spans = _speech([("no", 0.2, 150, 0.2), ("never", 0.6, 230, 0.6)])
    assert stressed_words(signal, RATE, spans) == ()


def test_no_word_timings_mean_no_reading() -> None:
    signal, _ = _speech([(w, 0.25, 150, 0.2) for w in ("one", "two", "three")])
    assert stressed_words(signal, RATE, []) == ()


def test_the_words_come_back_in_the_order_they_were_said() -> None:
    signal, spans = _speech(
        [
            ("you", 0.55, 230, 0.6),
            ("said", 0.20, 150, 0.2),
            ("it", 0.20, 150, 0.2),
            ("was", 0.20, 150, 0.2),
            ("fine", 0.20, 150, 0.2),
            ("and", 0.20, 150, 0.2),
            ("never", 0.55, 230, 0.6),
        ]
    )
    assert stressed_words(signal, RATE, spans) == ("you", "never")
