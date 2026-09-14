"""core/voice/duplex/paralinguistics.py — Hearing *how* it was said.

Everything upstream of this module throws away the most informative part of
speech. ASR returns words; the pitch, the speed, the hesitation, the fact
that someone sounded tired or clipped or delighted — all of it dies at the
transcription boundary. Aura then answers the words with prosody compiled
purely from her *own* internal state, which is why a pipeline agent can be
warm at exactly the moment you needed it to be careful.

This is the reason ChatGPT's voice mode feels alive, more than its latency:
it is speech-to-speech, so tone survives the round trip in both directions.
We cannot make a 32B text model speech-native, but the perceptual half is
recoverable from audio we are already buffering for the ASR, at a cost of a
few milliseconds of numpy.

Two things come out of it:

  * **Context.** Descriptors go to her mind as real observations, so "you
    sound rushed" is something she actually knows rather than infers from
    word choice.
  * **Convergence.** Her delivery moves partway toward the user's. Speaking
    quietly and slowly to someone who is speaking quietly and slowly is most
    of what rapport is made of.

Two design rules keep it honest:

  1. **Baseline-relative, never absolute.** A 190 Hz median means nothing on
     its own — it depends on the speaker and the microphone. Only deviation
     from *this speaker's* running baseline is reported, so the system says
     "quieter than usual" rather than inventing a mood from a number.
  2. **Report only what is notable.** Handing her a paralinguistic readout
     every single turn trains her to ignore it. Silence is the default.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("Aura.Voice.Paralinguistics")

# Human speech F0 essentially lives here. Searching wider invites
# octave errors on creaky voice, which is common at the end of statements.
F0_MIN_HZ = 70.0
F0_MAX_HZ = 350.0

_FRAME_MS = 40.0
_HOP_MS = 20.0


def _frame(signal: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if signal.size < frame_len:
        return np.zeros((0, frame_len), dtype=np.float32)
    n = 1 + (signal.size - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return signal[idx]


def estimate_f0(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Per-frame fundamental frequency; NaN where unvoiced.

    Normalised autocorrelation rather than a neural pitch tracker: it is a
    few milliseconds for a whole utterance, has no weights to load, and is
    accurate enough for "is this rising or falling", which is all we ask of
    it. A dedicated tracker would buy precision this module has no use for.
    """
    if signal.size == 0:
        return np.zeros(0, dtype=np.float32)

    frame_len = int(sample_rate * _FRAME_MS / 1000.0)
    hop = int(sample_rate * _HOP_MS / 1000.0)
    frames = _frame(signal.astype(np.float32), frame_len, hop)
    if frames.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    min_lag = max(2, int(sample_rate / F0_MAX_HZ))
    max_lag = min(frame_len - 1, int(sample_rate / F0_MIN_HZ))
    if max_lag <= min_lag:
        return np.full(frames.shape[0], np.nan, dtype=np.float32)

    # Remove DC per frame: an offset puts a large spurious peak at lag 0
    # and biases every correlation that follows.
    frames = frames - frames.mean(axis=1, keepdims=True)
    energy = np.sqrt((frames**2).mean(axis=1))

    out = np.full(frames.shape[0], np.nan, dtype=np.float32)
    # Frames quieter than this are silence or breath; correlating them
    # produces confident nonsense.
    voiced_floor = max(1e-4, float(np.median(energy)) * 0.35)

    for i, frame in enumerate(frames):
        if energy[i] < voiced_floor:
            continue
        corr = np.correlate(frame, frame, mode="full")[frame_len - 1:]
        if corr[0] <= 0:
            continue
        corr = corr / corr[0]
        window = corr[min_lag:max_lag]
        if window.size == 0:
            continue
        peak = int(np.argmax(window)) + min_lag
        # A weak peak means no periodicity — unvoiced, not a low pitch.
        if corr[peak] < 0.3:
            continue
        out[i] = sample_rate / float(peak)

    return out


@dataclass(slots=True)
class VoiceSignature:
    """Measured delivery of one utterance. All values are raw, not judged."""

    f0_median_hz: float = 0.0
    f0_range_semitones: float = 0.0
    final_slope_semitones: float = 0.0  # + rising, - falling
    energy_rms: float = 0.0
    speaking_rate_wps: float = 0.0      # words per second of voiced audio
    pause_ratio: float = 0.0            # fraction of the utterance that is silence
    longest_pause_s: float = 0.0
    voiced_ratio: float = 0.0
    duration_s: float = 0.0

    @property
    def valid(self) -> bool:
        return self.duration_s > 0.25 and self.voiced_ratio > 0.1


def analyze(signal: np.ndarray, sample_rate: int, *, word_count: int = 0) -> VoiceSignature:
    """Measure one utterance. Pure function, no state."""
    sig = VoiceSignature()
    if signal.size == 0:
        return sig

    sig.duration_s = signal.size / float(sample_rate)
    sig.energy_rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))

    f0 = estimate_f0(signal, sample_rate)
    voiced = f0[~np.isnan(f0)] if f0.size else np.zeros(0, dtype=np.float32)
    sig.voiced_ratio = float(voiced.size / f0.size) if f0.size else 0.0

    if voiced.size >= 3:
        sig.f0_median_hz = float(np.median(voiced))
        # Semitones, not hertz: pitch is perceived logarithmically, so a
        # 20 Hz move means something very different high up than low down.
        lo = float(np.percentile(voiced, 10))
        hi = float(np.percentile(voiced, 90))
        if lo > 0:
            sig.f0_range_semitones = float(12.0 * np.log2(max(hi, 1e-6) / lo))
        # Final contour decides question-vs-statement more reliably than
        # punctuation the ASR guessed at.
        tail = voiced[-max(3, voiced.size // 5):]
        if tail.size >= 3 and tail[0] > 0:
            sig.final_slope_semitones = float(
                12.0 * np.log2(max(float(np.mean(tail[-2:])), 1e-6) / max(float(tail[0]), 1e-6))
            )

    # Pause structure from a short-time energy envelope.
    hop = max(1, int(sample_rate * 0.02))
    env = np.array(
        [float(np.sqrt(np.mean(np.square(signal[i:i + hop], dtype=np.float64))))
         for i in range(0, signal.size, hop)],
        dtype=np.float32,
    )
    if env.size:
        # Two floors, whichever is higher. The percentile floor adapts to the
        # speaker, but on a pause-heavy utterance the 20th percentile *is*
        # pause level, so it silently stops detecting the very thing we want.
        # The peak-relative floor keeps working in exactly that case.
        floor = max(
            1e-4,
            float(np.percentile(env, 20)) * 1.6,
            float(env.max()) * 0.08,
        )
        quiet = env < floor
        # Silence before the first word and after the last is not a pause —
        # it is the VAD preroll and the trailing hangover. Counting it made
        # a *fast* utterance register as hesitant, because the synthesiser's
        # trailing silence looked like a long mid-sentence stall.
        voiced_idx = np.flatnonzero(~quiet)
        if voiced_idx.size:
            quiet = quiet[voiced_idx[0]: voiced_idx[-1] + 1]
        else:
            quiet = np.zeros(0, dtype=bool)

        sig.pause_ratio = float(np.mean(quiet)) if quiet.size else 0.0
        # Longest run of quiet frames strictly between the first and last word.
        longest = run = 0
        for q in quiet:
            run = run + 1 if q else 0
            longest = max(longest, run)
        sig.longest_pause_s = longest * 0.02

    speech_s = sig.duration_s * max(0.05, 1.0 - sig.pause_ratio)
    if word_count > 0 and speech_s > 0:
        sig.speaking_rate_wps = word_count / speech_s

    return sig


@dataclass
class SpeakerBaseline:
    """Running per-speaker norm.

    Absolute pitch and loudness are properties of a person and a microphone,
    not of a mood. Without a baseline this module would confidently report
    that a naturally quiet person is "subdued" on every single turn.
    """

    _f0: list[float] = field(default_factory=list)
    _energy: list[float] = field(default_factory=list)
    _rate: list[float] = field(default_factory=list)
    _max_samples: int = 40

    def observe(self, sig: VoiceSignature) -> None:
        if not sig.valid:
            return
        if sig.f0_median_hz > 0:
            self._push(self._f0, sig.f0_median_hz)
        if sig.energy_rms > 0:
            self._push(self._energy, sig.energy_rms)
        if sig.speaking_rate_wps > 0:
            self._push(self._rate, sig.speaking_rate_wps)

    def _push(self, store: list[float], value: float) -> None:
        store.append(value)
        if len(store) > self._max_samples:
            store.pop(0)

    @property
    def ready(self) -> bool:
        # Under three samples the "baseline" is just the last thing they
        # said, and every comparison against it is noise.
        return len(self._energy) >= 3

    def _z(self, store: list[float], value: float, min_relative: float) -> float:
        """Deviation in standard deviations, gated by perceptible magnitude.

        A z-score alone is not enough. Early in a session the baseline holds
        a handful of similar samples, its standard deviation is tiny, and a
        difference far too small to hear scores 3+ sigma — so she announces
        that someone is "quieter than usual" on a turn that sounded
        identical. Requiring the change to *also* be a real proportional
        move kills that false positive without weakening genuine detection.
        """
        if len(store) < 3 or value <= 0:
            return 0.0
        arr = np.asarray(store, dtype=np.float64)
        mean = float(arr.mean())
        spread = float(arr.std())
        if mean <= 0:
            return 0.0
        delta = value - mean
        relative = abs(delta) / mean
        if relative < min_relative:
            return 0.0
        # A perfectly steady baseline has zero variance, but that must make a
        # large departure more visible, not mathematically invisible. The
        # perceptibility floor supplies a conservative scale until natural
        # variance appears.
        scale = max(spread, mean * min_relative / 2.0, 1e-9)
        return float(delta / scale)

    # Perceptibility floors, one per dimension, because just-noticeable
    # difference is not the same across them: roughly 1 dB of loudness
    # (~12% amplitude), ~5% of speaking rate, and well under a semitone of
    # pitch (~4%). A single shared threshold either spams on loudness or
    # goes deaf to tempo.
    _REL_F0 = 0.04
    _REL_ENERGY = 0.12
    _REL_RATE = 0.06

    def f0_z(self, sig: VoiceSignature) -> float:
        return self._z(self._f0, sig.f0_median_hz, self._REL_F0)

    def energy_z(self, sig: VoiceSignature) -> float:
        return self._z(self._energy, sig.energy_rms, self._REL_ENERGY)

    def rate_z(self, sig: VoiceSignature) -> float:
        return self._z(self._rate, sig.speaking_rate_wps, self._REL_RATE)


def stressed_words(
    signal: np.ndarray,
    sample_rate: int,
    words: list[tuple[str, float, float]] | tuple[tuple[str, float, float], ...],
) -> tuple[str, ...]:
    """The words this utterance leaned on, measured against its own other words.

    Stress is how a speaker marks the word that carries the point: "I did not
    say that" and "I did not say that" are different sentences. A word is
    leaned on when its pitch peak, its loudness and its length per letter stand
    out from the rest of the same utterance, so the comparison is with this
    speaker at this moment rather than with anyone's norm. Each measure is a
    z-score across the utterance's words, a word's prominence is the mean of
    the scores it has, and the words past the module's own notability line are
    returned in the order they were said. Fewer than three words give nothing to
    compare against, and an utterance said evenly stresses nothing.

    `words` are (text, start, end) in seconds from the start of `signal`.
    """
    spans = [
        (str(text).strip(), float(start), float(end))
        for text, start, end in words
        if str(text).strip() and float(end) > float(start)
    ]
    if len(spans) < 3 or signal.size == 0 or sample_rate <= 0:
        return ()

    f0 = estimate_f0(signal, sample_rate)
    frame_len = int(sample_rate * _FRAME_MS / 1000.0)
    hop = max(1, int(sample_rate * _HOP_MS / 1000.0))
    centres = (np.arange(f0.size) * hop + frame_len / 2.0) / float(sample_rate)
    voiced = f0[~np.isnan(f0)]
    median_f0 = float(np.median(voiced)) if voiced.size else 0.0

    lengths: list[float] = []
    energies: list[float] = []
    peaks: list[float] = []
    for text, start, end in spans:
        letters = max(1, sum(ch.isalnum() for ch in text))
        lengths.append((end - start) / letters)
        a = max(0, int(start * sample_rate))
        b = min(signal.size, int(end * sample_rate))
        chunk = signal[a:b].astype(np.float64)
        energies.append(float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0)
        inside = f0[(centres >= start) & (centres <= end)] if f0.size else np.zeros(0)
        inside = inside[~np.isnan(inside)]
        if inside.size and median_f0 > 0:
            peaks.append(float(12.0 * np.log2(float(np.max(inside)) / median_f0)))
        else:
            peaks.append(float("nan"))

    def zscores(values: list[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        good = ~np.isnan(arr)
        out = np.full(arr.shape, np.nan)
        if good.sum() < 3:
            return out
        spread = float(np.std(arr[good]))
        if spread <= 0.0:
            out[good] = 0.0
            return out
        out[good] = (arr[good] - float(np.mean(arr[good]))) / spread
        return out

    scores = np.vstack([zscores(lengths), zscores(energies), zscores(peaks)])
    counted = (~np.isnan(scores)).sum(axis=0)
    prominence = np.where(counted > 0, np.nansum(scores, axis=0) / np.maximum(counted, 1), 0.0)
    return tuple(text for (text, _, _), value in zip(spans, prominence, strict=True) if value >= _NOTABLE_Z)


@dataclass(slots=True)
class DeliveryReading:
    """Interpreted delivery — what she is told, and what shapes her voice."""

    descriptors: tuple[str, ...] = ()
    # Normalised drivers for prosodic convergence, each roughly -1..1.
    energy_z: float = 0.0
    rate_z: float = 0.0
    pitch_z: float = 0.0
    rising_final: bool = False
    hesitant: bool = False
    #: The words they leaned on, from `stressed_words`, in the order said.
    stressed: tuple[str, ...] = ()

    @property
    def notable(self) -> bool:
        return bool(self.descriptors or self.stressed)

    def as_context(self) -> str:
        """A short observation for her mind, or "" when nothing stands out."""
        parts: list[str] = []
        if self.descriptors:
            parts.append(f"[you can hear that the user sounds {', '.join(self.descriptors)}]")
        if self.stressed:
            parts.append(f"[you can hear them lean on: {', '.join(self.stressed)}]")
        return " ".join(parts)


# Deviation past this many standard deviations is worth mentioning. Lower
# and every turn carries a readout, which trains her to ignore it.
_NOTABLE_Z = 1.15


def interpret(sig: VoiceSignature, baseline: SpeakerBaseline) -> DeliveryReading:
    """Turn raw measurements into something sayable, relative to baseline."""
    reading = DeliveryReading()
    if not sig.valid:
        return reading

    reading.energy_z = baseline.energy_z(sig)
    reading.rate_z = baseline.rate_z(sig)
    reading.pitch_z = baseline.f0_z(sig)
    reading.rising_final = sig.final_slope_semitones > 1.5

    descriptors: list[str] = []

    # Hesitation is structural, not relative, so it needs no baseline: a long
    # internal silence is someone choosing their words, whoever they are.
    # Either signal is sufficient — requiring both missed real hesitation in
    # utterances that were mostly speech with one long stall in the middle.
    reading.hesitant = sig.longest_pause_s >= 0.35 or (
        sig.pause_ratio > 0.28 and sig.longest_pause_s >= 0.22
    )
    if reading.hesitant:
        descriptors.append("hesitant")

    if baseline.ready:
        if reading.rate_z >= _NOTABLE_Z:
            descriptors.append("rushed")
        elif reading.rate_z <= -_NOTABLE_Z:
            descriptors.append("slower than usual")

        if reading.energy_z >= _NOTABLE_Z:
            descriptors.append("louder and more emphatic")
        elif reading.energy_z <= -_NOTABLE_Z:
            descriptors.append("quieter than usual")

        # Pitch only reads as affect alongside energy. Raised pitch with
        # normal energy is usually just a question.
        if reading.pitch_z >= _NOTABLE_Z and reading.energy_z > 0:
            descriptors.append("keyed up")
        elif reading.pitch_z <= -_NOTABLE_Z and reading.energy_z < 0:
            descriptors.append("flat, possibly tired")

    reading.descriptors = tuple(descriptors)
    return reading


# How far her delivery moves toward the user's. Full mirroring is mimicry
# and reads as mockery; none at all is the flat-affect problem this module
# exists to fix. Partial convergence is what people actually do.
CONVERGENCE = 0.3


def convergence_factors(reading: DeliveryReading) -> tuple[float, float]:
    """Multipliers ``(speed, gain)`` for her prosody, from the user's delivery."""
    if not reading.notable and abs(reading.rate_z) < 0.5 and abs(reading.energy_z) < 0.5:
        return 1.0, 1.0

    def clamp(z: float) -> float:
        return max(-2.0, min(2.0, z))

    # ±2 sd of user deviation maps to at most ±12% of her own delivery.
    speed = 1.0 + CONVERGENCE * clamp(reading.rate_z) * 0.20
    gain = 1.0 + CONVERGENCE * clamp(reading.energy_z) * 0.18

    # Someone speaking hesitantly is thinking; matching their pace gives
    # them room instead of hurrying them.
    if reading.hesitant:
        speed *= 0.96

    return max(0.85, min(1.15, speed)), max(0.8, min(1.2, gain))
