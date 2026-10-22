"""A local speech upgrade stays inside the existing model lifetime."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from core.voice.duplex.config import TtsConfig
from core.voice.duplex.pocket_speech import PocketSpeech
from core.voice.duplex.prosody import ProsodySpec
from core.voice.duplex.tts_stream import CancellationToken, StreamingTts


def test_unprepared_or_remote_model_never_downloads_during_a_call(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "pocket_tts", SimpleNamespace(
        TTSModel=SimpleNamespace(load_model=lambda **kw: calls.append(kw))))
    engine = PocketSpeech(tmp_path)
    assert not engine.load()
    (tmp_path / "alba.safetensors").touch()
    (tmp_path / "config.yaml").write_text(
        "weights_path: hf://remote/model\nflow_lm:\n  lookup_table:\n    tokenizer_path: https://remote/tokenizer\n")
    assert not engine.load()
    assert calls == []


def test_each_clause_copies_the_prepared_speaker_state(tmp_path):
    calls = []

    class Audio:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.ones(100, dtype=np.float32)

    def generate(state, text, **kwargs):
        calls.append((state, text, kwargs))
        return Audio()

    engine = PocketSpeech(tmp_path)
    engine._model = SimpleNamespace(generate_audio=generate, sample_rate=24000)
    state = object()
    engine._voices = {"alba": state}
    for text in ("First.", "Second."):
        samples, rate = engine.synthesize(text, ProsodySpec(voice="af_heart"))
        assert len(samples) == 100 and rate == 24000
    assert [row[1] for row in calls] == ["First.", "Second."]
    assert all(row[0] is state and row[2] == {"copy_state": True} for row in calls)


@pytest.mark.asyncio
async def test_upgrade_uses_the_admitted_lane_and_releases_its_model(tmp_path, monkeypatch):
    import core.voice.duplex.tts_stream as module

    class Engine:
        name = "pocket"
        voices = {"alba"}

        def __init__(self, directory):
            assert directory == tmp_path

        def load(self):
            return True

        def synthesize(self, text, spec):
            return np.ones(100, dtype=np.float32), 24000

    released = []
    lease = SimpleNamespace(set_preemptible=lambda value: True, release=lambda **kw: released.append(kw))
    tts = StreamingTts(TtsConfig(pocket_directory=tmp_path))
    def admit():
        tts._lane_lease = lease
        return lease
    monkeypatch.setattr(tts, "_acquire_model_lane", admit)
    monkeypatch.setattr(module, "PocketSpeech", Engine)
    assert await tts.ensure_loaded()
    assert tts.engine_name == "pocket"
    assert tts.available_voices() == ["alba"]
    assert tts.status()["model_lane_owned"]
    result = await tts.synthesize("Hello.", ProsodySpec(voice="alba"), CancellationToken())
    assert result is not None and result.engine == "pocket"
    tts.shutdown()
    assert tts._state.pocket is None
    assert not tts.status()["model_lane_owned"]
    assert released


def test_a_loaded_upgrade_without_its_lane_is_a_violation(tmp_path):
    from core.voice.duplex.tts_stream import loaded_speech_models_have_an_owner

    tts = StreamingTts(TtsConfig(pocket_directory=tmp_path))
    tts._state.pocket = object()
    tts._state.loaded = True
    try:
        assert any(v.subject == f"voice-tts:{id(tts)}" for v in loaded_speech_models_have_an_owner())
    finally:
        tts.shutdown()
    assert not any(v.subject == f"voice-tts:{id(tts)}" for v in loaded_speech_models_have_an_owner())
