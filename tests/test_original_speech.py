"""Original speech keeps its reference and worker inside the admitted lifetime."""
import asyncio
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.voice.duplex.config import TtsConfig
from core.voice.duplex.original_speech import OriginalSpeech, SpeechWorkerLease
from core.voice.duplex.prosody import ProsodySpec
from core.voice.duplex.tts_stream import CancellationToken, StreamingTts


def test_changed_reference_never_starts_a_model_worker(tmp_path, monkeypatch):
    reference = tmp_path / 'reference.wav'
    reference.write_bytes(b'changed')
    (tmp_path / 'model.safetensors').touch()
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'model_directory':str(tmp_path),'reference_audio':str(reference),
                                    'reference_sha256':hashlib.sha256(b'original').hexdigest()}))
    spawn = []
    monkeypatch.setattr('core.voice.duplex.original_speech.subprocess.Popen', lambda *a, **kw: spawn.append(kw))
    assert not OriginalSpeech(manifest, Path(__file__)).load()
    assert not spawn


def test_worker_is_reaped_before_memory_ownership_is_released():
    order = []
    lease = SpeechWorkerLease(SimpleNamespace(release=lambda **kw: order.append('release')),
                              SimpleNamespace(close=lambda: order.append('reap')))
    lease.release(reason='test')
    assert order == ['reap', 'release']


@pytest.mark.asyncio
async def test_shutdown_during_model_load_keeps_ownership_until_native_completion(monkeypatch):
    import core.voice.duplex.tts_stream as module
    entered, finish = threading.Event(), threading.Event()
    order = []

    class Engine:
        name = 'qwen_original'
        voices = {'aura_velvet'}

        def __init__(self, *args): pass
        def load(self):
            entered.set()
            assert finish.wait(3)
            return True
        def close(self): order.append('reap')

    monkeypatch.setattr(module, 'OriginalSpeech', Engine)
    tts = StreamingTts(TtsConfig(engine='qwen_original'))
    lease = SimpleNamespace(set_preemptible=lambda _:True, release=lambda **kw:order.append('release'))
    def admit():
        tts._lane_lease = lease
        return lease
    monkeypatch.setattr(tts, '_acquire_model_lane', admit)
    task = asyncio.create_task(tts.ensure_loaded())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        tts.shutdown()
        assert order == []
        finish.set()
        for _ in range(100):
            if order: break
            await asyncio.sleep(.01)
        assert order == ['reap', 'release']
        assert tts._state.pocket is None
    finally:
        finish.set()
        tts.shutdown()


@pytest.mark.asyncio
async def test_original_audio_reaches_the_production_mask(monkeypatch):
    tts = StreamingTts(TtsConfig(treatment='veil'))
    samples = np.ones(2400, dtype=np.float32) * .2
    tts._state.pocket = SimpleNamespace(name='qwen_original', synthesize=lambda *a:(samples,24000))
    tts._state.loaded = True
    tts._accepting_synthesis = True
    monkeypatch.setattr(tts,'ensure_loaded',lambda: asyncio.sleep(0,result=True))
    try:
        result = await tts.synthesize('A clear sentence.',ProsodySpec(voice='aura_velvet'),CancellationToken())
        assert result.engine == 'qwen_original'
        assert result.samples.shape == samples.shape
        assert not np.array_equal(result.samples,samples)
    finally: tts.shutdown()
