"""Her voice as it forms: each stretch of speech decoded, resampled, levelled and paced while the rest is still made.

VoxCPM2 makes speech a patch at a time (four latent frames, 160 ms at 48 kHz), and mlx_audio decodes the patches only
when the sentence is done, so her first sound waited for her last: 8.8 s for a 5.4 s sentence on 11 November 2026, with
a test batch running beside her. Its audio decoder is causal. A window that starts as far back as the decoder can see
decodes to the same audio a whole decode gives: stitched against whole measured 77 dB, which is the same waveform,
and her first sound came 0.46 s after the request
(~/.aura/voice-lab/kiwi-reference/clean-2026-11-11/stream_decode_probe.py).

Everything after the decoder keeps its state across pieces, so a sentence sent in pieces is the sentence sent whole:
the 48 kHz to 24 kHz filter is resample_poly's own, run with its state carried; her pace is WSOLA, whose streamed
output equals its whole output sample for sample. Her loudness cannot wait for the whole sentence, so it is the gain
her own finished sentences have needed, learned as she speaks.

This runs in her speech environment (requirements/voice-original.txt), not her runtime's.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

# The ceiling the whole-sentence path levels to (original_speech_worker.py): no peak above it.
PEAK_CEILING = .85
# A sentence's first patches are made at her full diffusion steps whatever its body is made at: at 5 steps, Bryan heard
# "a little noise that isn't her voice at the beginning"; measured, it starts 80 ms in and is gone by 240 ms, inside
# the first two patches (11 November 2026, ~/.aura/voice-lab/kiwi-reference/steps-2026-11-11).
ONSET_PATCHES = 2


def causal_lookback(conv: Any) -> int:
    """How many earlier samples a causal convolution reads: its left padding (audio_vae.CausalConv1d)."""
    return 2 * int(getattr(conv, "pad_val", 0))


def receptive_patches(decoder: Any, patch_size: int) -> int:
    """How many earlier patches one decoded sample depends on, from the decoder's own layers.

    The decoder only looks back, so a window holding this many patches before the new ones decodes them exactly as
    a whole decode would. VoxCPM2's: 19.1 latent frames, so 5 patches of four (3 measured 67 dB, 6 measured 83 dB).
    """
    back, rate = 0.0, 1  # latent frames looked back; output samples per latent frame at this depth
    convs = getattr(decoder, "conv_in", None)
    for conv in getattr(convs, "layers", None) or [convs]:
        back += causal_lookback(conv) / rate
    for block in decoder.blocks.layers:
        stride = int(block.conv_t.stride)
        kernel = int(block.conv_t.weight.shape[1])  # MLX weights are (out, kernel, in)
        back += (math.ceil(kernel / stride) - 1) / rate  # a transposed step reads its own frame and the one before
        rate *= stride
        for unit in (block.res1, block.res2, block.res3):
            back += (causal_lookback(unit.conv1) + causal_lookback(unit.conv2)) / rate
    back += causal_lookback(decoder.conv_out) / rate
    return max(1, math.ceil(back / patch_size))


def next_body_steps(body: int, *, full: int, busy: int, real_time: float, busy_per_full: float | None) -> int:
    """The steps her next sentence's body is made at (Bryan, 11 November 2026: adaptive). Full while her voice keeps
    ahead of the listener; busy for the next sentence once one at full fell behind; full again once one at busy ran
    fast enough that full would have kept ahead too (its real time scaled by what busy steps cost against full)."""
    if body == full and real_time < 1:
        return busy
    if body == busy and busy_per_full and real_time * busy_per_full >= 1:
        return full
    return body


class Downsampler:
    """resample_poly(x, 1, down)'s filter, run with its state carried, so pieces join without a seam."""

    def __init__(self, down: int) -> None:
        from scipy.signal import firwin

        self.down = int(down)
        half = 10 * self.down  # resample_poly's own half length for this ratio
        self._taps = firwin(2 * half + 1, 1.0 / self.down, window=("kaiser", 5.0)) if self.down > 1 else None
        self._state = np.zeros(2 * half) if self.down > 1 else None
        self._delay = half
        self._seen = 0

    def push(self, audio: np.ndarray) -> np.ndarray:
        if self._taps is None or not len(audio):
            return np.asarray(audio, dtype=np.float32)
        from scipy.signal import lfilter

        filtered, self._state = lfilter(self._taps, 1.0, np.asarray(audio, dtype=np.float64), zi=self._state)
        index = np.arange(self._seen, self._seen + len(filtered))
        self._seen += len(filtered)
        keep = (index >= self._delay) & ((index - self._delay) % self.down == 0)
        return filtered[keep].astype(np.float32)

    def finish(self) -> np.ndarray:
        """What the filter still holds: the last input samples' outputs."""
        return self.push(np.zeros(self._delay)) if self._taps is not None else np.zeros(0, np.float32)


class Pace:
    """Her speaking pace, applied as the audio arrives (WSOLA). At 1.0 it passes audio through untouched."""

    def __init__(self, speed: float) -> None:
        self.speed = max(.85, min(1.15, float(speed)))  # the whole-sentence path's own bounds
        self._tsm = None
        self.held_back = 0  # output samples WSOLA keeps until more input follows: its frame
        if abs(self.speed - 1.) > .01:
            import audiotsm

            self._tsm = audiotsm.wsola(1, speed=self.speed)
            self.held_back = int(self._tsm._frame_length)

    def _drain(self, reader: Any) -> np.ndarray:
        from audiotsm.io.array import ArrayWriter

        writer, finished = ArrayWriter(1), False
        while not (finished and reader.empty):
            self._tsm.read_from(reader)
            _, finished = self._tsm.write_to(writer)
        return writer.data[0].astype(np.float32)

    def push(self, audio: np.ndarray) -> np.ndarray:
        if self._tsm is None or not len(audio):
            return audio
        from audiotsm.io.array import ArrayReader

        return self._drain(ArrayReader(np.asarray(audio, dtype=np.float32)[None]))

    def finish(self) -> np.ndarray:
        if self._tsm is None:
            return np.zeros(0, np.float32)
        from audiotsm.io.array import ArrayReader, ArrayWriter

        # WSOLA drops its last two frames' worth of input unless silence follows it; measured 11 November 2026, the
        # last 80 ms of every sentence went missing without it.
        tail = self._drain(ArrayReader(np.zeros((1, 2 * self._tsm._frame_length), dtype=np.float32)))
        writer, finished = ArrayWriter(1), False
        while not finished:
            _, finished = self._tsm.flush_to(writer)
        return np.concatenate([tail, writer.data[0].astype(np.float32)])


class Level:
    """The gain her sentences need to reach her audition's loudness, learned from the sentences she has finished."""

    def __init__(self, target_rms: float) -> None:
        self.target = float(target_rms)
        self._log_gains: list[float] = []

    @property
    def gain(self) -> float:
        return float(math.exp(sum(self._log_gains) / len(self._log_gains))) if self._log_gains else 1.0

    def learn(self, sentence: np.ndarray) -> None:
        """The gain the whole-sentence path would have given this sentence."""
        if not self.target or not len(sentence):
            return
        rms = float(np.sqrt(np.mean(np.square(sentence, dtype=np.float64))))
        peak = float(np.max(np.abs(sentence)))
        self._log_gains.append(math.log(min(self.target / max(rms, 1e-8), PEAK_CEILING / max(peak, 1e-8))))


class CutOffError(Exception):
    """The listener cut her off: the rest of the sentence is not made."""


class SpeechAsItForms:
    """One sentence: patches in as the model makes them, audio out whenever the listener would otherwise run out."""

    def __init__(self, *, model: Any, decode: Callable[[Any], Any], mx: Any, prompt_context: list[Any],
                 lookback: int, out_rate: int, speed: float, level: Level, decode_s_per_patch: float,
                 send: Callable[[np.ndarray, bool], None], clock: Callable[[], float] = time.perf_counter,
                 should_stop: Callable[[], bool] = lambda: False, gen_s_per_patch: float = 0.0,
                 expected_patches: int = 0) -> None:
        self.model, self._decode, self.mx = model, decode, mx
        self._history = list(prompt_context)[-lookback:] if lookback else []
        self._lookback = lookback
        self._pending: list[Any] = []
        self._patch_samples = model.patch_size * model.audio_vae.decode_chunk_size
        self._rate = int(model.sample_rate)
        self._down = Downsampler(self._rate // out_rate) if out_rate != self._rate else Downsampler(1)
        self._out_rate = out_rate
        self._pace = Pace(speed)
        self._level = level
        self._gain = level.gain if level.target else 1.0
        self._sentence: list[np.ndarray] = []
        self._send = send
        self._clock = clock
        self._should_stop = should_stop
        self.stopped = False
        self.decode_s_per_patch = float(decode_s_per_patch)
        self.gen_s_per_patch = float(gen_s_per_patch)  # making one patch, decoding apart; carried between sentences
        self.expected_patches = int(expected_patches)  # from the text, at her own patches per character
        self._patch_audio_s = self._patch_samples / self._rate / max(self._pace.speed, 1e-6)
        self._held_back_s = self._pace.held_back / out_rate  # the first piece plays that much shorter
        self._decoding_since_patch = 0.0
        self._gen_deviation_s = 0.0  # how far a patch's making strays from her pace (Jacobson's estimator, as TCP's)
        self._slowest_patch_s = 0.0
        self._last_patch_at: float | None = None
        self._first_patch_at: float | None = None
        self._plays_until: float | None = None  # when the audio already sent runs out, if it plays as it arrives
        self.first_sent_at: float | None = None
        self.started = clock()
        self.decodes = 0
        self.real_time = 0.0
        self.body_steps: int | None = None
        self.patches = 0
        self.decode_s = 0.0
        self.sent_s = 0.0

    def patch(self, feat: Any) -> None:
        """A patch the model just made, as generate collects it: (batch, patch, latent)."""
        if self._should_stop():
            raise CutOffError
        now = self._clock()
        if self._last_patch_at is not None:
            self._slowest_patch_s = max(self._slowest_patch_s, now - self._last_patch_at)
            made = max(0.0, now - self._last_patch_at - self._decoding_since_patch)
            # Her own recent pace and how much it strays, weighted to the last few (Jacobson's estimator).
            if not self.gen_s_per_patch:
                self.gen_s_per_patch, self._gen_deviation_s = made, made / 2
            else:
                self._gen_deviation_s = 0.75 * self._gen_deviation_s + 0.25 * abs(made - self.gen_s_per_patch)
                self.gen_s_per_patch = 0.875 * self.gen_s_per_patch + 0.125 * made
        self._decoding_since_patch = 0.0
        self._last_patch_at = now
        if self._first_patch_at is None:
            self._first_patch_at = now
        self.patches += 1
        self._pending.append(feat)
        if self._due(now):
            self._flush(last=False)

    def falls_behind(self) -> bool:
        """Whether this sentence's patches, since its first, have come slower than they play."""
        if self._first_patch_at is None or self._last_patch_at is None or self.patches < 2:
            return False
        return (self._last_patch_at - self._first_patch_at) / (self.patches - 1) > self._patch_audio_s

    def window(self) -> int | None:
        """The fewest new patches a window can hold and still play at least as long as making the next one takes.

        A window of n patches costs n patches made plus a decode of n and the lookback before them, and plays for n
        patches. None when no window can keep up (her GPU is shared and slow): then each goes out when the listener
        has run dry. LIVE 11 November 2026, a decode for every patch made each one 0.18 s against 0.16 s of audio.
        """
        spare = self._patch_audio_s - self.gen_s_per_patch - self.decode_s_per_patch
        if not self.gen_s_per_patch:
            return 1
        if spare <= 0:
            return None
        return max(1, math.ceil(self.decode_s_per_patch * self._lookback / spare))

    def _due(self, now: float) -> bool:
        pending, window = len(self._pending), self.window()
        # The window is sized on her average pace; the deadline on her pace plus four times how much it strays, as a
        # retransmission timeout is: one slow patch is not a gap, and one spike does not hold back every sentence.
        # (LIVE 11 November 2026, her slowest patch as the deadline held one sentence's first sound 9.99 s.)
        expected = self.gen_s_per_patch + 4 * self._gen_deviation_s
        making = (expected + self.decode_s_per_patch) * (window or pending) + self.decode_s_per_patch * self._lookback
        if self._plays_until is None:
            # Her first sound, once it plays as long as making the next window takes: sooner, and a gap follows it.
            if window is None:
                return self._the_rest_is_made_in_time(-self._held_back_s)
            return pending * self._patch_audio_s - self._held_back_s >= making
        left = self._plays_until - now
        if window is None:
            return self._the_rest_is_made_in_time(left)
        return pending >= window and left < making

    def _the_rest_is_made_in_time(self, left: float) -> bool:
        """When she cannot keep up, what is sent waits until it will play for as long as making the rest takes, so the
        silence comes before her sentence and not as a stumble inside it. Without a length to expect, it goes when
        the listener has run dry."""
        pending = len(self._pending)
        if not self.expected_patches:
            return left <= 0
        rest = max(0, self.expected_patches - self.patches)
        return max(0.0, left) + pending * self._patch_audio_s >= rest * (self.gen_s_per_patch + self.decode_s_per_patch)

    def _flush(self, *, last: bool) -> None:
        audio = np.zeros(0, np.float32)
        if self._pending:
            context = self._history[-self._lookback:] if self._lookback else []
            window = self.mx.concatenate(context + self._pending, axis=1)
            started = self._clock()
            decoded = np.asarray(self._decode(window).flatten().astype(self.mx.float32))
            spent = self._clock() - started
            self.decode_s += spent
            self._decoding_since_patch += spent
            per = spent / (len(context) + len(self._pending))
            self.decode_s_per_patch = per if not self.decode_s_per_patch else 0.7 * self.decode_s_per_patch + 0.3 * per
            self.decodes += 1
            skip = len(context) * self._patch_samples
            # A decode returns a few samples past its last patch; only the sentence's last keeps them, as whole does.
            audio = decoded[skip:] if last else decoded[skip:skip + len(self._pending) * self._patch_samples]
            self._history = (self._history + self._pending)[-max(self._lookback, 1):]
            self._pending = []
        audio = self._down.push(audio)
        if last:
            audio = np.concatenate([audio, self._down.finish()])
        self._sentence.append(audio)
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if self._level.target and peak * self._gain > PEAK_CEILING:
            self._gain = PEAK_CEILING / peak  # never louder than the whole path allows; it only comes down
        out = self._pace.push(audio * self._gain)
        if last:
            out = np.concatenate([out, self._pace.finish()])
        now = self._clock()
        if len(out) or last:
            self.sent_s += len(out) / self._out_rate
            if last:
                self._send(out.astype(np.float32), False, self.timing(now))
            else:
                self._send(out.astype(np.float32), True)
            if self.first_sent_at is None and len(out):
                self.first_sent_at = now
            self._plays_until = max(self._plays_until or now, now) + len(out) / self._out_rate

    def timing(self, now: float) -> dict[str, Any]:
        """How the sentence was made, for her log: whether her voice kept ahead of the listener, and what it cost."""
        made_s = now - self.started
        self.real_time = self.sent_s / made_s if made_s > 0 else 0.0
        return {"patches": self.patches, "audio_s": round(self.sent_s, 2), "made_s": round(made_s, 2),
                "first_sound_s": round((self.first_sent_at or now) - self.started, 2),
                "slowest_patch_s": round(self._slowest_patch_s, 3), "decode_s": round(self.decode_s, 2),
                "windows": self.decodes, "real_time": round(self.sent_s / made_s, 2) if made_s > 0 else 0.0,
                "gen_s_per_patch": round(self.gen_s_per_patch, 3), "decode_s_per_patch": round(self.decode_s_per_patch, 4),
                "window": self.window(), "body_steps": self.body_steps}

    def finish(self) -> None:
        self._flush(last=True)
        self._level.learn(np.concatenate(self._sentence) if self._sentence else np.zeros(0, np.float32))

    def stop(self) -> None:
        """End the sentence where it was cut: nothing more is decoded, and the listener's end of it is told."""
        self.stopped = True
        self._pending = []
        self._send(np.zeros(0, np.float32), False)


def speak_as_it_forms(model: Any, text: str, options: dict[str, Any], *, mx: Any, prompt_context: list[Any],
                      lookback: int, out_rate: int, speed: float, level: Level, decode_s_per_patch: float,
                      send: Callable[[np.ndarray, bool], None],
                      should_stop: Callable[[], bool] = lambda: False,
                      gen_s_per_patch: float = 0.0, expected_patches: int = 0, body_steps: int | None = None,
                      onset_steps: int | None = None, busy_steps: int | None = None,
                      clock: Callable[[], float] = time.perf_counter) -> SpeechAsItForms:
    """Generate ``text`` and send its audio in pieces as it forms. Returns the sentence's record (timings)."""
    decode = model.audio_vae.decode
    sentence = SpeechAsItForms(model=model, decode=decode, mx=mx, prompt_context=prompt_context, lookback=lookback,
                               out_rate=out_rate, speed=speed, level=level, decode_s_per_patch=decode_s_per_patch,
                               send=send, should_stop=should_stop, gen_s_per_patch=gen_s_per_patch,
                               expected_patches=expected_patches, clock=clock)
    sample = model.feat_decoder.sample
    sentence.body_steps = body_steps

    def sampled(*args: Any, **kwargs: Any) -> Any:
        if body_steps and onset_steps and "n_timesteps" in kwargs:
            if (busy_steps and sentence.body_steps == onset_steps and sentence.patches >= ONSET_PATCHES
                    and sentence.falls_behind()):
                # LIVE 11 November 2026, a sentence at her full steps ran at 0.88 of real time and its listener heard
                # ten gaps inside it, the next sentence going to her busy steps only after. Bryan heard 10 steps then
                # 6 as clean as 10 throughout, so a sentence falling behind goes to her busy steps where it is.
                sentence.body_steps = busy_steps
            kwargs["n_timesteps"] = onset_steps if sentence.patches < ONSET_PATCHES else sentence.body_steps
        feat = sample(*args, **kwargs)
        mx.eval(feat)
        sentence.patch(feat.transpose(0, 2, 1))
        return feat

    # generate's own decode of the whole sentence is not needed: its audio has already gone out.
    model.feat_decoder.sample = sampled
    model.audio_vae.decode = lambda *a, **k: mx.zeros((1, 1))
    try:
        for result in model.generate(text, **options):
            mx.eval(result.audio)
    except CutOffError:
        sentence.stop()
        return sentence
    finally:
        model.feat_decoder.sample = sample
        model.audio_vae.decode = decode
    sentence.finish()
    return sentence


def prompt_patches(model: Any, prompt_audio: str) -> list[Any]:
    """Her reference clip's patches, as generate encodes them for a continuation: the context of her first sound."""
    feats = model._encode_wav(prompt_audio, padding_mode="left")  # (T, P, D)
    return [feats[i][None] for i in range(feats.shape[0])]
