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


def test_timed_out_audio_cannot_become_the_next_sentence(monkeypatch):
    engine = OriginalSpeech(Path('/unused'), Path(__file__))
    writes, closed = [], []
    engine.process = SimpleNamespace(stdin=SimpleNamespace(write=writes.append))
    def timeout(_):
        raise TimeoutError('no response')
    def close():
        closed.append(True)
        engine.process = None
    monkeypatch.setattr(engine, '_receive', timeout)
    monkeypatch.setattr(engine, 'close', close)
    with pytest.raises(TimeoutError):
        engine.synthesize('First sentence.', ProsodySpec(voice='aura_kiwi'))
    with pytest.raises(RuntimeError, match='not loaded'):
        engine.synthesize('Next sentence.', ProsodySpec(voice='aura_kiwi'))
    assert len(writes) == 1 and closed == [True]


@pytest.mark.parametrize('engine', ['qwen_original', 'voxcpm2'])
@pytest.mark.parametrize('shards', ['complete', 'missing', 'traversal'])
def test_sharded_reference_model_is_complete_before_worker_start(tmp_path, monkeypatch, engine, shards):
    reference = tmp_path / 'reference.wav'
    reference.write_bytes(b'prepared')
    (tmp_path / 'part1.safetensors').touch()
    if shards == 'complete':
        (tmp_path / 'part2.safetensors').touch()
    second = '../part2.safetensors' if shards == 'traversal' else 'part2.safetensors'
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {'first.weight': 'part1.safetensors', 'second.weight': second},
    }))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({
        'engine': engine, 'voice_id': 'aura_kiwi', 'model_directory': str(tmp_path),
        'reference_audio': str(reference),
        'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
    }))
    spawned = []
    monkeypatch.setattr('core.voice.duplex.original_speech.subprocess.Popen',
                        lambda *a, **kw: spawned.append(a) or SimpleNamespace())
    speech = OriginalSpeech(manifest, Path(__file__), engine)
    monkeypatch.setattr(speech, '_receive', lambda _: {'ready': True})
    assert speech.load() is (shards == 'complete')
    assert bool(spawned) is (shards == 'complete')
    assert speech.name == engine


def test_voxcpm_uses_the_continuation_api_from_the_selected_audition():
    from tools.voice.original_speech_worker import generation_options

    options = generation_options({
        'engine': 'voxcpm2', 'reference_audio': '/local/kiwi.wav',
        'reference_text': 'The reference sentence.',
        'inference_timesteps': 10, 'cfg_value': 2.,
    })
    assert options == {
        'prompt_audio': '/local/kiwi.wav', 'prompt_text': 'The reference sentence. ',
        'inference_timesteps': 10, 'cfg_value': 2., 'max_tokens': 300,
    }
    # VoxCPM2 ignores ref_text. Passing it would silently choose another mode.
    assert 'ref_text' not in options and 'ref_audio' not in options


def test_faster_voxcpm_keeps_the_acoustic_modules_at_full_precision():
    from tools.voice.original_speech_worker import prepare_model

    solver = object()
    compiled = object()
    model = SimpleNamespace(
        base_lm=object(), residual_lm=object(), audio_vae=object(),
        feat_decoder=SimpleNamespace(solve_euler=solver), parameters=lambda: 'weights',
    )
    acoustic = model.audio_vae, model.feat_decoder
    quantized, evaluated = [], []
    def compile_method(fn):
        assert fn is solver
        return compiled
    mx = SimpleNamespace(compile=compile_method, eval=evaluated.append, clear_cache=lambda: None)
    nn = SimpleNamespace(quantize=lambda module, **kw: quantized.append((module, kw)))
    prepare_model(model, {'engine': 'voxcpm2', 'language_model_bits': 8, 'compile_solver': True}, mx, nn)
    assert quantized == [(model.base_lm, {'group_size': 64, 'bits': 8}),
                         (model.residual_lm, {'group_size': 64, 'bits': 8})]
    assert (model.audio_vae, model.feat_decoder) == acoustic
    assert model.feat_decoder.solve_euler is compiled
    assert evaluated == ['weights']


@pytest.mark.parametrize('engine,budget,path', [
    ('voxcpm2', 10., 'voice-duplex/tts/voxcpm2'),
    ('qwen_original', 8., 'voice-duplex/tts/qwen-original'),
])
def test_prepared_worker_reserves_its_model_family_and_measured_memory(monkeypatch, engine, budget, path):
    admitted = []
    lease = SimpleNamespace(release=lambda **kw: True)
    monkeypatch.setattr('core.runtime.model_lane_control.acquire_synchronous_in_process_model_lane',
                        lambda **kw: admitted.append(kw) or lease)
    tts = StreamingTts(TtsConfig(engine=engine))
    try:
        assert tts._acquire_model_lane() is lease
        assert admitted[0]['request_gb'] == budget
        assert admitted[0]['model_path'] == path
    finally:
        tts.shutdown()


@pytest.mark.parametrize('voice_id', ['aura_kiwi', None, 'Kiwi', 'aura_' + 'x' * 65])
def test_prepared_profile_advertises_its_verified_voice_identifier(tmp_path, monkeypatch, voice_id):
    reference = tmp_path / 'reference.wav'
    reference.write_bytes(b'prepared local voice')
    (tmp_path / 'model.safetensors').touch()
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({
        'model_directory': str(tmp_path), 'reference_audio': str(reference),
        'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
        'voice_id': voice_id,
    }))
    spawned = []
    monkeypatch.setattr('core.voice.duplex.original_speech.subprocess.Popen',
                        lambda *args, **kwargs: spawned.append(args) or SimpleNamespace())
    engine = OriginalSpeech(manifest, Path(__file__))
    monkeypatch.setattr(engine, '_receive', lambda timeout: {'ready': True})
    assert engine.load() is (voice_id == 'aura_kiwi')
    assert bool(spawned) is (voice_id == 'aura_kiwi')
    if spawned:
        assert engine.voices == {'aura_kiwi'}


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
