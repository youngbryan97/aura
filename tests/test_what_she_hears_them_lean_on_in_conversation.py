"""The words a speaker leaned on, read in live conversation.

`stressed_words` needs to know when each word was said, and recognition used
to hand back text alone, so the reader could not fire outside its own tests.
These pin the path: each backend's timings become (word, start, end), the
final decode keeps them, and the delivery reading her mind is given names the
words that stood out.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from core.voice.duplex.config import AsrConfig
from core.voice.duplex.paralinguistics import DeliveryReading
from core.voice.duplex.streaming_asr import (
    StreamingAsr,
    Transcript,
    _parakeet_words,
    _whisper_words,
    looks_hallucinated,
)


def _token(text: str, start: float, duration: float) -> SimpleNamespace:
    return SimpleNamespace(text=text, start=start, duration=duration, end=start + duration)


def test_parakeet_pieces_join_into_the_words_they_spell() -> None:
    tokens = [
        _token(" I", 0.00, 0.10),
        _token(" did", 0.12, 0.15),
        _token(" n", 0.30, 0.10),
        _token("ot", 0.40, 0.25),
        _token(" say", 0.70, 0.20),
    ]
    words = _parakeet_words(tokens)
    assert [w for w, _, _ in words] == ["I", "did", "not", "say"]
    assert [t for _, s, e in words for t in (s, e)] == pytest.approx([0.00, 0.10, 0.12, 0.27, 0.30, 0.65, 0.70, 0.90])


def test_a_first_piece_without_a_space_still_starts_a_word() -> None:
    words = _parakeet_words([_token("so", 0.0, 0.2), _token(" what", 0.3, 0.2)])
    assert [w for w, _, _ in words] == ["so", "what"]
    assert [t for _, s, e in words for t in (s, e)] == pytest.approx([0.0, 0.2, 0.3, 0.5])


def test_whisper_words_are_read_from_dicts_and_from_objects() -> None:
    as_dicts = [{"words": [{"word": " you", "start": 0.0, "end": 0.4}, {"word": " never", "start": 0.5, "end": 1.0}]}]
    as_objects = [SimpleNamespace(words=[SimpleNamespace(word=" you", start=0.0, end=0.4)])]
    assert _whisper_words(as_dicts) == (("you", 0.0, 0.4), ("never", 0.5, 1.0))
    assert _whisper_words(as_objects) == (("you", 0.0, 0.4),)


def test_a_word_without_a_time_is_left_out_rather_than_guessed() -> None:
    segments = [{"words": [{"word": " fine", "start": None, "end": 0.4}, {"word": " ok", "start": 0.5, "end": 0.7}]}]
    assert _whisper_words(segments) == (("ok", 0.5, 0.7),)


def test_the_mlx_backend_asks_for_timings_only_on_the_decode_that_needs_them(monkeypatch) -> None:
    from core.runtime import model_lane_control
    from core.voice.duplex.streaming_asr import _WhisperBackend

    class Lease:
        def set_preemptible(self, _value):
            return True

        def release(self, *, reason):
            return True

    monkeypatch.setattr(model_lane_control, "acquire_synchronous_in_process_model_lane", lambda **_kw: Lease())
    asked: list[bool] = []

    class FakeMlx:
        @staticmethod
        def transcribe(_audio, **kwargs):
            asked.append(bool(kwargs.get("word_timestamps")))
            return {
                "text": " you never",
                "segments": [{"words": [{"word": " you", "start": 0.0, "end": 0.4}, {"word": " never", "start": 0.5, "end": 1.0}]}],
            }

    backend = _WhisperBackend(AsrConfig(partial_model="small", final_model="large"))
    backend._impl = "mlx"
    backend._mlx = FakeMlx()
    try:
        audio = np.zeros(16, dtype=np.float32)
        assert backend.transcribe(audio, "small") == " you never"
        assert backend.transcribe_words(audio, "large") == (" you never", (("you", 0.0, 0.4), ("never", 0.5, 1.0)))
    finally:
        backend.shutdown()
    assert asked == [False, True]


class _TimedBackend:
    available = True

    def __init__(self, text: str, words: tuple[tuple[str, float, float], ...]) -> None:
        self.text = text
        self.words = words

    def transcribe(self, _audio, _repo):
        return self.text

    def transcribe_words(self, _audio, _repo):
        return self.text, self.words

    def warm(self, _repo):
        return True

    def shutdown(self):
        return None


def test_the_final_transcript_keeps_when_each_word_was_said() -> None:
    words = (("we", 0.0, 0.2), ("talked", 0.3, 0.7), ("about", 0.8, 1.0), ("this", 1.1, 1.4))
    asr = StreamingAsr(AsrConfig(), backend=_TimedBackend("we talked about this", words), owns_backend=False)
    final = asyncio.run(asr.finalize(np.ones(24_000, dtype=np.float32) * 0.1))
    assert final.is_final
    assert final.words == words


def test_a_transcript_thrown_away_as_hallucination_keeps_no_words() -> None:
    asr = StreamingAsr(
        AsrConfig(),
        backend=_TimedBackend("Thank you for watching.", (("Thank", 0.0, 0.2),)),
        owns_backend=False,
    )
    assert looks_hallucinated("Thank you for watching.")
    final = asyncio.run(asr.finalize(np.zeros(24_000, dtype=np.float32)))
    assert final.stable == ""
    assert final.words == ()


def test_a_backend_that_reports_no_timings_still_decodes() -> None:
    class TextOnly:
        available = True

        def transcribe(self, _audio, _repo):
            return "just the words"

        def shutdown(self):
            return None

    asr = StreamingAsr(AsrConfig(), backend=TextOnly(), owns_backend=False)
    final = asyncio.run(asr.finalize(np.ones(16_000, dtype=np.float32) * 0.1))
    assert final.stable == "just the words"
    assert final.words == ()


def test_her_mind_is_told_which_words_they_leaned_on() -> None:
    reading = DeliveryReading(stressed=("never",))
    assert reading.notable
    assert "never" in reading.as_context()


def test_nothing_leaned_on_and_nothing_notable_says_nothing() -> None:
    assert DeliveryReading().as_context() == ""


def test_the_session_reads_stress_from_the_words_the_final_decode_kept() -> None:
    from core.voice.duplex.session import DuplexVoiceSession

    rate = 16_000
    pieces, spans, clock = [], [], 0.0
    for text, seconds, hertz, amplitude in (
        ("i", 0.20, 150, 0.2),
        ("did", 0.25, 150, 0.2),
        ("not", 0.60, 230, 0.6),
        ("say", 0.25, 150, 0.2),
        ("that", 0.25, 150, 0.2),
    ):
        t = np.arange(int(rate * seconds)) / rate
        pieces.append((amplitude * np.sin(2 * np.pi * hertz * t)).astype(np.float32))
        spans.append((text, clock, clock + seconds))
        clock += seconds + 0.08
        pieces.append(np.zeros(int(rate * 0.08), dtype=np.float32))
    audio = np.concatenate(pieces)

    session = DuplexVoiceSession.__new__(DuplexVoiceSession)
    from core.voice.duplex.paralinguistics import SpeakerBaseline

    session._speaker_baseline = SpeakerBaseline()
    session._delivery = DeliveryReading()
    transcript = Transcript(stable="i did not say that", is_final=True, words=tuple(spans))
    session._read_delivery(audio, transcript.stable, transcript.words)
    assert session._delivery.stressed == ("not",)
    assert "not" in session._delivery.as_context()
