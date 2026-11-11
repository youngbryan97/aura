"""Her voice as it forms: a sentence sent in pieces is the sentence sent whole, and her first sound waits for nothing.

The worker side (tools/voice/her_speech_as_it_forms.py) is proved against her real model by
tools/voice/prove_speech_as_it_forms.py: stitched against whole at 140.3, 84.6, 124.2 and 83.3 dB on 11 November
2026, one per sentence. These tests hold the parts that do not need the model: the stitching against a causal decoder,
the filter against resample_poly, the engine's stop, and how her runtime plays a clause that arrives in pieces.
"""
import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import resample_poly

from core.voice.duplex.config import TtsConfig
from core.voice.duplex.original_speech import OriginalSpeech
from core.voice.duplex.prosody import ProsodySpec
from core.voice.duplex.tts_stream import CancellationToken, StreamingTts
from tools.voice.her_speech_as_it_forms import (
    Downsampler,
    Level,
    SpeechAsItForms,
    receptive_patches,
)

PATCH, FRAME = 4, 6  # latent frames per patch, samples per latent frame
LOOKBACK = 2  # patches the fake decoder below looks back


class _FakeMx:
    float32 = np.float32

    @staticmethod
    def concatenate(parts, axis):
        return np.concatenate(parts, axis=axis)


def _causal_decode(window):
    """A causal decoder: each output sample reads its own latent frame and the frames LOOKBACK patches before it."""
    frames = window[0, :, 0]  # (frames,)
    reach = LOOKBACK * PATCH - 1
    padded = np.concatenate([np.zeros(reach), frames])
    weights = np.linspace(1.0, 0.2, reach + 1)
    out = np.array([np.dot(padded[i:i + reach + 1][::-1], weights) for i in range(len(frames))])
    return np.repeat(out, FRAME)[None].astype(np.float32)


def _model():
    return SimpleNamespace(patch_size=PATCH, sample_rate=48000, audio_vae=SimpleNamespace(decode_chunk_size=FRAME))


def test_a_sentence_sent_in_pieces_is_the_sentence_decoded_whole():
    rng = np.random.default_rng(7)
    patches = [rng.normal(size=(1, PATCH, 1)).astype(np.float32) for _ in range(13)]
    sent = []
    clock = iter(np.arange(0, 1000, 1.0))  # every patch is due, so every window is as short as it can be
    sentence = SpeechAsItForms(model=_model(), decode=_causal_decode, mx=_FakeMx, prompt_context=[], lookback=LOOKBACK,
                               out_rate=48000, speed=1.0, level=Level(0.0), decode_s_per_patch=0.0,
                               send=lambda audio, more, timing=None: sent.append((audio, more)), clock=lambda: next(clock))
    for patch in patches:
        sentence.patch(patch)
    sentence.finish()
    whole = _causal_decode(np.concatenate(patches, axis=1)).flatten()
    stitched = np.concatenate([audio for audio, _ in sent])
    assert np.allclose(stitched, whole, atol=1e-6)
    assert [more for _, more in sent][-1] is False and all(more for _, more in sent[:-1])
    assert len(sent[0][0]) == PATCH * FRAME  # her first sound is her first patch
    assert len(sent) == len(patches) + 1  # one window per patch, and the sentence's end


def test_a_window_that_cannot_see_far_enough_back_is_not_the_same_audio():
    """The check above would pass for any decoder that only reads its own patch; this one must fail it."""
    rng = np.random.default_rng(7)
    patches = [rng.normal(size=(1, PATCH, 1)).astype(np.float32) for _ in range(9)]
    sent = []
    clock = iter(np.arange(0, 1000, 1.0))  # each patch takes longer than the one before it plays: every one is due
    sentence = SpeechAsItForms(model=_model(), decode=_causal_decode, mx=_FakeMx, prompt_context=[], lookback=1,
                               out_rate=48000, speed=1.0, level=Level(0.0), decode_s_per_patch=0.0,
                               send=lambda audio, more, timing=None: sent.append(audio), clock=lambda: next(clock))
    for patch in patches:
        sentence.patch(patch)
    sentence.finish()
    whole = _causal_decode(np.concatenate(patches, axis=1)).flatten()
    assert not np.allclose(np.concatenate(sent), whole, atol=1e-3)


def test_pieces_wait_while_the_listener_still_has_audio_to_hear():
    """Decoding is paid for when the audio already sent would otherwise run out, not on every patch."""
    patches = [np.ones((1, PATCH, 1), np.float32) for _ in range(12)]
    sent, now = [], [0.0]
    sentence = SpeechAsItForms(model=_model(), decode=_causal_decode, mx=_FakeMx, prompt_context=[], lookback=LOOKBACK,
                               out_rate=48000, speed=1.0, level=Level(0.0), decode_s_per_patch=0.0,
                               send=lambda audio, more, timing=None: sent.append(audio), clock=lambda: now[0])
    # Each patch plays for PATCH*FRAME / 48000 = 0.5 ms here and is made in 0.1 ms: the first goes at once, and the
    # next waits until what was sent is nearly heard.
    for patch in patches:
        sentence.patch(patch)
        now[0] += 0.0001
    sentence.finish()
    assert len(sent[0]) == PATCH * FRAME
    assert 2 < len(sent) < len(patches)


def test_receptive_field_comes_from_the_decoders_own_layers():
    conv = lambda pad: SimpleNamespace(pad_val=pad)  # noqa: E731
    unit = lambda: SimpleNamespace(conv1=conv(3 * 3), conv2=conv(0))  # noqa: E731 - kernel 7, dilation 3
    block = lambda stride: SimpleNamespace(  # noqa: E731
        conv_t=SimpleNamespace(stride=stride, weight=np.zeros((1, 2 * stride, 1))), res1=unit(), res2=unit(),
        res3=unit())
    decoder = SimpleNamespace(conv_in=conv(3), blocks=SimpleNamespace(layers=[block(8), block(6)]),
                              conv_out=conv(3))
    # 6 frames in; block 1: 1 frame + 3*18 samples at 8/frame; block 2: 1/8 frame + 3*18 at 48/frame; 6 at 48/frame.
    back = 6 + 1 + 54 / 8 + 1 / 8 + 54 / 48 + 6 / 48
    assert receptive_patches(decoder, 4) == int(np.ceil(back / 4))


def test_the_rate_filter_carried_across_pieces_is_resample_polys_own():
    rng = np.random.default_rng(3)
    audio = rng.normal(size=48000).astype(np.float32)
    down = Downsampler(2)
    pieces = [down.push(audio[i:i + 7680]) for i in range(0, len(audio), 7680)] + [down.finish()]
    streamed = np.concatenate(pieces)
    whole = resample_poly(audio, 1, 2).astype(np.float32)
    assert len(streamed) == len(whole)
    assert np.allclose(streamed[40:-40], whole[40:-40], atol=1e-5)


def test_her_level_is_what_her_finished_sentences_needed():
    level = Level(0.09)
    assert level.gain == 1.0
    level.learn(np.full(4800, 0.045, np.float32))
    level.learn(np.full(4800, 0.18, np.float32))
    assert level.gain == pytest.approx(np.sqrt(2.0 * 0.5))


def test_a_cut_off_sentence_stops_her_worker_and_still_reads_its_last_message(monkeypatch):
    engine = OriginalSpeech(Path('/unused'), Path(__file__))
    writes = []
    engine.process = SimpleNamespace(stdin=SimpleNamespace(write=writes.append))
    piece = base64.b64encode(np.ones(240, '<f4').tobytes()).decode()
    replies = iter([{'audio': piece, 'sample_rate': 24000, 'more': True},
                    {'audio': piece, 'sample_rate': 24000, 'more': True},
                    {'audio': '', 'sample_rate': 24000, 'more': False}])
    monkeypatch.setattr(engine, '_receive', lambda _timeout: next(replies))
    heard = []
    engine.synthesize_pieces('A sentence she will not finish.', ProsodySpec(voice='aura_kiwi'),
                             lambda audio, rate, more: heard.append((len(audio), more)),
                             stopped=lambda: len(heard) >= 1)
    assert json.loads(writes[0])['stream'] is True
    assert writes[1:] == [b'{"stop": true}\n']
    assert heard == [(240, True), (240, True), (0, False)]
    assert engine.process is not None  # the pipe stayed in step, so her worker stays loaded


def _streaming_tts(pieces, *, fail_after=None):
    tts = StreamingTts(TtsConfig(treatment='clean'))
    whole = []

    def synthesize_pieces(text, spec, deliver, stopped=None):
        for index, (audio, more) in enumerate(pieces):
            if fail_after is not None and index == fail_after:
                raise RuntimeError('worker died')
            deliver(audio, 24000, more)

    tts._state.pocket = SimpleNamespace(name='voxcpm2', streams=True, synthesize_pieces=synthesize_pieces,
                                        synthesize=lambda text, spec: whole.append(text) or (np.ones(480, np.float32),
                                                                                             24000))
    tts._state.loaded = True
    tts._accepting_synthesis = True
    return tts, whole


@pytest.mark.asyncio
async def test_a_clause_arrives_in_pieces_with_its_gain_and_its_pause_on_the_last(monkeypatch):
    pieces = [(np.full(480, .5, np.float32), True), (np.full(480, .5, np.float32), True),
              (np.full(240, .5, np.float32), False)]
    tts, _ = _streaming_tts(pieces)
    monkeypatch.setattr(tts, 'ensure_loaded', lambda: asyncio.sleep(0, result=True))
    spec = ProsodySpec(voice='aura_kiwi', gain=.5, trailing_pause_ms=10.0)
    try:
        results = [r async for r in tts.synthesize_as_it_forms('One clause.', spec, CancellationToken())]
    finally:
        tts.shutdown()
    assert [r.ends_clause for r in results] == [False, False, True]
    assert np.allclose(results[0].samples, .25)
    assert len(results[-1].samples) == 240 + 240  # 10 ms of pause at 24 kHz on the last piece only


@pytest.mark.asyncio
async def test_a_clause_that_failed_before_it_was_heard_is_spoken_whole(monkeypatch):
    tts, whole = _streaming_tts([(np.ones(480, np.float32), False)], fail_after=0)
    monkeypatch.setattr(tts, 'ensure_loaded', lambda: asyncio.sleep(0, result=True))
    try:
        results = [r async for r in tts.synthesize_as_it_forms('Said whole.', ProsodySpec(voice='aura_kiwi'),
                                                               CancellationToken())]
    finally:
        tts.shutdown()
    assert whole == ['Said whole.'] and [r.ends_clause for r in results] == [True]


def test_a_clause_in_pieces_is_one_clause_to_whoever_listens():
    """One caption and one record of words for the clause; no empty audio frame; her turn ends when the last piece
    has played, not when a piece that came late let the listener's buffer run dry."""
    from core.voice.duplex.session import DuplexVoiceSession, MindBridge
    from core.voice.duplex.tts_stream import SynthesisResult

    def piece(n, ends):
        return SynthesisResult(samples=np.ones(n, np.float32) * .1, sample_rate=24000, text='Hello there.',
                               synth_ms=1.0, engine='voxcpm2', ends_clause=ends)

    class PiecesTts:
        async def stream(self, chunks, _spec, _token):
            async for _ in chunks:
                for result in (piece(2400, False), piece(0, False), piece(2400, False), piece(1200, True)):
                    yield result

        def shutdown(self):
            return None

    async def exercise() -> None:
        events, frames, order = [], [], []
        session = None

        async def send_json(payload):
            events.append(dict(payload))
            if payload.get('type') == 'voice.state':
                order.append(('state', payload.get('state')))

        async def report(played_ms, delay):
            await asyncio.sleep(delay)
            order.append(('reported', played_ms))
            await session.handle_command({'command': 'playback', 'utterance_id': session._speaking.utterance_id,
                                          'played_ms': played_ms, 'drained': True})

        async def send_binary(payload):
            frames.append(len(payload))
            if len(frames) == 3:
                # A drain from the first piece's gap arrives after the last piece went out; the real one later.
                asyncio.get_running_loop().create_task(report(100, 0.0))
                asyncio.get_running_loop().create_task(report(250, 0.3))

        mind = MindBridge(session_id='as-it-forms')
        session = DuplexVoiceSession(session_id='as-it-forms', send_json=send_json, send_binary=send_binary, mind=mind)
        session._tts = PiecesTts()
        await session._speak_text('Hello there.', cause=None)
        record = mind.last_spoken
        assert [e['text'] for e in events if e.get('type') == 'voice.chunk'] == ['Hello there.']
        assert [n for n in frames if n > 8] == [4808, 4808, 2408]  # the empty piece sent nothing; 8 is the end mark
        assert record.spoken == 'Hello there.' and record.delivery_complete is True
        assert order.index(('reported', 250)) < order.index(('state', 'listening'))
        await session.close()

    asyncio.run(exercise())


def test_the_sentences_last_piece_says_how_it_was_made():
    """Her log says, sentence by sentence, whether her voice kept ahead of the listener (real_time at least 1)."""
    timings = []
    clock = iter(np.arange(0, 1000, 0.25))
    sentence = SpeechAsItForms(model=_model(), decode=_causal_decode, mx=_FakeMx, prompt_context=[], lookback=LOOKBACK,
                               out_rate=48000, speed=1.0, level=Level(0.0), decode_s_per_patch=0.0,
                               send=lambda audio, more, timing=None: timings.append(timing), clock=lambda: next(clock))
    for _ in range(6):
        sentence.patch(np.ones((1, PATCH, 1), np.float32))
    sentence.finish()
    assert all(timing is None for timing in timings[:-1])
    last = timings[-1]
    assert last["patches"] == 6 and last["audio_s"] == round(6 * PATCH * FRAME / 48000, 2)
    assert last["real_time"] < 1 and last["windows"] >= 1


def _played(pieces):
    """Underruns of pieces played as they arrive, as her window plays them: (arrival time, seconds of audio)."""
    plays_until, gaps = None, []
    for at, seconds in pieces:
        if plays_until is not None and at > plays_until + 1e-9:
            gaps.append(round(at - plays_until, 3))
        plays_until = max(plays_until or at, at) + seconds
    return gaps


def _a_sentence_at_her_pace(patches, made_s, decode_s_per_patch, *, carried=0.0, expected=0):
    """Her measured pace, 11 November 2026: a patch made in 0.129 s, decoded at 9 ms a patch, playing 0.16 s."""
    now = [0.0]
    model = SimpleNamespace(patch_size=PATCH, sample_rate=48000, audio_vae=SimpleNamespace(decode_chunk_size=1920))

    def decode(window):
        now[0] += decode_s_per_patch * window.shape[1] / PATCH
        return np.zeros((1, window.shape[1] * 1920), np.float32)

    pieces = []
    sentence = SpeechAsItForms(model=model, decode=decode, mx=_FakeMx, prompt_context=[], lookback=5, out_rate=48000,
                               speed=1.0, level=Level(0.0), decode_s_per_patch=decode_s_per_patch,
                               gen_s_per_patch=carried, expected_patches=expected,
                               send=lambda audio, more, timing=None: pieces.append((now[0], len(audio) / 48000)),
                               clock=lambda: now[0])
    for _ in range(patches):
        now[0] += made_s
        sentence.patch(np.zeros((1, PATCH, 1), np.float32))
    sentence.finish()
    return sentence, [piece for piece in pieces if piece[1] > 0]


def test_at_her_quiet_pace_a_sentence_plays_without_a_gap():
    """LIVE 11 November 2026: a decode for every patch made each one 0.18 s against 0.16 s of audio, and a long reply
    went silent 220 to 393 times. Windows sized to her pace keep ahead of the listener."""
    sentence, pieces = _a_sentence_at_her_pace(60, 0.129, 0.009, carried=0.129)
    assert sentence.window() == 3
    assert _played(pieces) == []
    assert sentence.decodes < 60 / 2
    assert pieces[0][0] < 4 * 0.129 + 0.2  # her first sound still comes within a few patches


def test_when_she_cannot_keep_up_the_silence_comes_before_the_sentence_not_inside_it():
    """Her GPU shared with her cortex's decoding: 0.37 s a patch, and no window keeps up. Knowing the sentence's length,
    she starts once the rest will be made before what she has runs out; not knowing it, at worst a gap per patch."""
    sentence, pieces = _a_sentence_at_her_pace(60, 0.37, 0.009, carried=0.37, expected=60)
    assert sentence.window() is None
    assert _played(pieces) == []
    _sentence, unknown = _a_sentence_at_her_pace(60, 0.37, 0.009, carried=0.37)
    assert len(_played(unknown)) <= 60
