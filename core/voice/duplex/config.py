"""core/voice/duplex/config.py — Tunables for the full-duplex voice lane.

Every number here is a latency or naturalness budget that was measured on
this host, not guessed. The comments record what each one buys so a future
change is an informed trade rather than a nudge.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _voice_names() -> tuple[str, ...]:
    """The names she answers to, as a declared flag rather than a raw read.

    Declared because this one is operator-facing in a way the latency budgets
    above are not — it is the difference between an assistant that answers to
    its name and one that does not — and because a raw ``os.environ.get`` is
    untyped and invisible to `flag_report()`, which is how this repo grew
    several hundred knobs nobody could enumerate.
    """
    from core.runtime.flags import FlagKind, declare

    flag = declare(
        "AURA_VOICE_NAMES",
        kind=FlagKind.STRING,
        default="aura",
        description="Comma-separated names Aura answers to on an open microphone",
        owner="core.voice.duplex.config",
    )
    names = tuple(
        part.strip() for part in str(flag.value() or "").split(",") if part.strip()
    )
    return names or ("aura",)


# ── Audio format ──────────────────────────────────────────────────────────
# 16 kHz mono is what Silero and Whisper both want; resampling once at the
# browser edge is cheaper than resampling every frame server-side.
CAPTURE_RATE = 16_000
# Kokoro emits 24 kHz. We keep it — downsampling would throw away the top
# octave that makes the voice sound like a person instead of a phone.
OUTPUT_RATE = 24_000

# Silero's ONNX graph is compiled for exactly 512 samples at 16 kHz (32 ms).
VAD_FRAME_SAMPLES = 512
VAD_FRAME_MS = VAD_FRAME_SAMPLES / CAPTURE_RATE * 1000.0


@dataclass(slots=True)
class VadConfig:
    """Hysteresis around Silero's frame probability.

    Two thresholds, not one: a high bar to *start* believing there is speech
    and a lower bar to keep believing it. A single threshold chatters on
    every fricative and turns one sentence into four utterances.
    """

    speech_threshold: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_VAD_ON", 0.55)
    )
    silence_threshold: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_VAD_OFF", 0.35)
    )
    # Consecutive speech frames before onset is real. 2 frames = 64 ms, which
    # rejects key clicks and lip smacks without eating the first phoneme.
    onset_frames: int = field(
        default_factory=lambda: _env_int("AURA_VOICE_VAD_ONSET_FRAMES", 2)
    )
    # Speech shorter than this is a noise burst, not a turn.
    min_utterance_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_MIN_UTTERANCE_MS", 220.0)
    )


@dataclass(slots=True)
class AsrConfig:
    """One model, where there used to be two.

    The split existed to buy accuracy without paying for it on every window:
    partials only steer endpointing, backchannels and captions, so they ran
    on small.en (~72 ms/window), while the transcript her mind actually
    reasons over ran once on large-v3-turbo (~195 ms). Paying the accurate
    model's cost on every 500 ms partial would have contended with the
    resident 32B for the GPU for no accuracy that mattered.

    Parakeet TDT removes the trade. Measured on this host 12 Aug 2026, real
    speech with a known transcript, median of 5 warm runs on 12.4 s of audio:

        parakeet-tdt-0.6b-v3      166 ms
        whisper-small.en          193 ms   <- the model it replaces on PARTIALS
        whisper-large-v3-turbo    317 ms   <- the model it replaces on FINALS

    One Parakeet decode is cheaper than the incumbent PARTIAL and roughly
    half the incumbent FINAL, so both stages can run the same weights: one
    model-lane lease, one set of weights resident, and no accuracy sacrificed
    on partials. Parakeet also reports 6.32% English WER on the Open ASR
    Leaderboard against turbo's 7.83% (all three scored 0% on the clean local
    sample, which is too easy to discriminate — that comparison is cited, not
    measured here).

    Both entries stay configurable and stay separate: pointing them at
    different repos is still supported, and the Whisper fallback chain is
    untouched for hosts without parakeet-mlx installed.
    """

    partial_model: str = field(
        default_factory=lambda: os.environ.get(
            "AURA_VOICE_ASR_PARTIAL", "mlx-community/parakeet-tdt-0.6b-v3"
        )
    )
    final_model: str = field(
        default_factory=lambda: os.environ.get(
            "AURA_VOICE_ASR_FINAL", "mlx-community/parakeet-tdt-0.6b-v3"
        )
    )
    # How often a partial decode runs while the user is speaking.
    partial_interval_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_PARTIAL_INTERVAL_MS", 480.0)
    )
    # Don't bother decoding less than this — Whisper hallucinates on scraps.
    min_decode_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_MIN_DECODE_MS", 700.0)
    )
    # Hard ceiling on a single utterance's audio buffer (memory + decode cost).
    max_utterance_s: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_MAX_UTTERANCE_S", 60.0)
    )
    language: str = "en"

    # Speculative final decode.
    #
    # The endpointer spends 340–1500 ms *deliberately waiting* to see whether
    # the user resumes. That wait is dead time in which the machine is idle,
    # and the final decode (~320 ms) then runs afterwards in series. Starting
    # it during the wait overlaps the two, so by the time the endpoint
    # confirms, the transcript her mind needs already exists.
    #
    # The decode is a pure function of the audio — no side effects — so a
    # wasted one costs only some CPU. It is discarded the instant speech
    # resumes, because then it describes a sentence that is not over.
    speculative_final: bool = field(
        default_factory=lambda: _env_flag("AURA_VOICE_SPECULATIVE_ASR", True)
    )
    # Far enough into the pause that this is probably a real turn end, early
    # enough that the decode finishes before the endpoint fires.
    speculate_after_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_SPECULATE_AFTER_MS", 130.0)
    )


@dataclass(slots=True)
class EndpointConfig:
    """Adaptive turn-end detection — the thing that makes it feel human.

    A fixed silence timeout is the single biggest tell of a robot: too short
    and it interrupts you the moment you pause to think, too long and every
    reply feels sluggish. So the wait is a function of whether what you just
    said *sounds finished*.
    """

    # Sounds complete: falling intonation, terminal punctuation, full clause.
    complete_silence_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_EP_COMPLETE_MS", 340.0)
    )
    # No strong signal either way.
    neutral_silence_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_EP_NEUTRAL_MS", 620.0)
    )
    # Ends on a conjunction/preposition — they are mid-thought, wait.
    incomplete_silence_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_EP_INCOMPLETE_MS", 1100.0)
    )
    # Ends on an audible filler ("uh…", "so…") — they are actively thinking.
    thinking_silence_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_EP_THINKING_MS", 1500.0)
    )
    # Absolute ceiling; a turn always ends eventually.
    max_silence_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_EP_MAX_MS", 2000.0)
    )


@dataclass(slots=True)
class BackchannelConfig:
    """"mhm" while they talk.

    Overlapping acknowledgement is what separates a listener from a recorder.
    It is also the easiest thing in this whole system to overdo — an agent
    that says "mhm" every two seconds is worse than one that says nothing.
    """

    enabled: bool = field(
        default_factory=lambda: _env_flag("AURA_VOICE_BACKCHANNEL", True)
    )
    # They must have held the floor this long before acknowledgement is apt.
    min_floor_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_BC_MIN_FLOOR_MS", 3200.0)
    )
    # Never twice inside this window.
    cooldown_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_BC_COOLDOWN_MS", 5000.0)
    )
    # A micro-pause is a prosodic boundary — the natural slot for "mhm".
    # Below the floor it is just inter-word silence; above it, an endpoint.
    pause_min_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_BC_PAUSE_MIN_MS", 140.0)
    )
    pause_max_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_BC_PAUSE_MAX_MS", 420.0)
    )
    # Even at a perfect slot, stay quiet sometimes. Humans do.
    fire_probability: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_BC_PROBABILITY", 0.55)
    )
    # Backchannels ride under the user's voice, so they are quiet.
    gain: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_BC_GAIN", 0.34)
    )


@dataclass(slots=True)
class FillerConfig:
    """Thinking sounds, tied to the real reason she is slow.

    These are not decoration. Aura's mind is a governed 32B pipeline; some
    turns genuinely take seconds. A filler is an honest status report in the
    register of speech — and critically it asserts nothing about the answer,
    so it can never be wrong about content.
    """

    enabled: bool = field(default_factory=lambda: _env_flag("AURA_VOICE_FILLERS", True))
    # Under this, silence reads as "instant" and a filler would be noise.
    first_delay_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_FILLER_FIRST_MS", 380.0)
    )
    # A short breath/"uh" has covered the gap; now say something contentful
    # about what is actually taking the time.
    second_delay_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_FILLER_SECOND_MS", 1900.0)
    )
    # Long enough that silence would read as a crash.
    third_delay_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_FILLER_THIRD_MS", 6500.0)
    )


@dataclass(slots=True)
class TtsConfig:
    """Prepared Pocket speech, Kokoro fallback, optional licensed XTTS clone."""

    engine: str = field(default_factory=lambda: os.environ.get("AURA_VOICE_TTS_ENGINE", "auto"))
    pocket_directory: Path = field(default_factory=lambda: Path(os.environ.get(
        "AURA_VOICE_POCKET_DIR", str(Path.home() / ".aura/live-source/data/voice_models/pocket"))))

    voice: str = field(
        default_factory=lambda: os.environ.get("AURA_VOICE_KOKORO_VOICE", "af_heart")
    )
    speed: float = field(default_factory=lambda: _env_float("AURA_VOICE_SPEED", 1.0))
    model_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "AURA_VOICE_KOKORO_MODEL",
                str(Path.home() / ".aura/live-source/data/voice_models/kokoro/kokoro-v1.0.onnx"),
            )
        )
    )
    voices_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "AURA_VOICE_KOKORO_VOICES",
                str(Path.home() / ".aura/live-source/data/voice_models/kokoro/voices-v1.0.bin"),
            )
        )
    )
    # First spoken chunk is deliberately tiny: time-to-first-audio dominates
    # perceived latency, and a 3-word opener sounds like a person starting to
    # answer. Later chunks grow because longer context = better prosody.
    #
    # Measured on this host, synthesis cost tracks output duration closely:
    #   "Yeah."                      140 ms
    #   "Yeah, I think so."          193 ms
    #   "Okay, so the short answer"  251 ms
    # so shrinking the opener from 48 to 24 characters buys ~110 ms of
    # perceived latency for free. Below about 20 the opener stops carrying
    # enough prosody and the reply sounds like it starts on a stutter.
    first_chunk_max_chars: int = field(
        default_factory=lambda: _env_int("AURA_VOICE_FIRST_CHUNK_CHARS", 24)
    )
    chunk_max_chars: int = field(
        default_factory=lambda: _env_int("AURA_VOICE_CHUNK_CHARS", 180)
    )
    # Synthesis threads. Kokoro is CPU/ONNX so this does not touch the GPU
    # the resident 32B is holding.
    workers: int = field(default_factory=lambda: _env_int("AURA_VOICE_TTS_WORKERS", 2))

    # Zero-shot voice cloning from a reference clip (XTTS-v2). Off unless a
    # clip is configured, because it costs seconds of latency per reply —
    # see _ClonedVoiceEngine for the trade.
    clone_reference: str = field(
        default_factory=lambda: os.environ.get("AURA_VOICE_CLONE_REFERENCE", "")
    )
    prefer_clone: bool = field(
        default_factory=lambda: _env_flag("AURA_VOICE_PREFER_CLONE", False)
    )


@dataclass(slots=True)
class BargeInConfig:
    """Interruption handling.

    Two things matter. First, stop fast — anything over ~200 ms and the user
    talks over themselves. Second, and more subtly: remember only what she
    *actually said out loud*. If her memory holds a paragraph the user never
    heard, every later reference to it is a hallucination from the user's
    point of view.
    """

    enabled: bool = field(default_factory=lambda: _env_flag("AURA_VOICE_BARGE_IN", True))
    # Sustained speech probability over this window cancels playback. Short
    # enough to feel instant, long enough to ignore a cough or a door.
    trigger_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_BARGE_TRIGGER_MS", 128.0)
    )
    # Barge-in needs more confidence than ordinary onset: a false positive
    # cuts her off mid-word, which is worse than a slightly late stop.
    threshold: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_BARGE_THRESHOLD", 0.72)
    )
    # Grace period after she starts speaking. Without it, the tail of the
    # user's own question (or residual echo) instantly kills the reply.
    grace_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_BARGE_GRACE_MS", 320.0)
    )


@dataclass(slots=True)
class AmbientConfig:
    """Listening without being switched on first.

    The point of ambient mode is that you walk up and talk. The cost is that
    an open microphone hears everything in the room, so the decision of what
    to *answer* moves out of the wake word and into a gate that weighs several
    signals (see ``addressivity.py``).

    ``enabled`` here governs only that gate. Whether the microphone opens at
    all is the user's runtime setting and their browser permission — two
    consents this module never overrides.
    """

    enabled: bool = field(
        default_factory=lambda: _env_flag("AURA_VOICE_AMBIENT", True)
    )
    # Names she answers to. Her own name is the strongest single signal there
    # is, and it stays the reliable way to get her attention in a noisy room.
    names: tuple[str, ...] = field(default_factory=lambda: _voice_names())
    # How long after she stops speaking a bare reply is still obviously hers
    # to take. This is the term that makes the second and third thing you say
    # need no ceremony, which is most of what a wake word costs you.
    open_floor_s: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_OPEN_FLOOR_S", 18.0)
    )
    # Words required before a cold open — no name, no open floor — is treated
    # as a request rather than as overheard speech.
    min_cold_open_words: int = field(
        default_factory=lambda: _env_int("AURA_VOICE_COLD_OPEN_WORDS", 3)
    )


@dataclass(slots=True)
class DuplexConfig:
    vad: VadConfig = field(default_factory=VadConfig)
    ambient: AmbientConfig = field(default_factory=AmbientConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    backchannel: BackchannelConfig = field(default_factory=BackchannelConfig)
    filler: FillerConfig = field(default_factory=FillerConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    barge_in: BargeInConfig = field(default_factory=BargeInConfig)

    # How long a cognition turn may run before the lane gives up and says so.
    cognition_timeout_s: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_COGNITION_TIMEOUT_S", 120.0)
    )

    # Target length for a spoken reply, in words.
    #
    # This used to be the largest latency lever in the lane: the governed turn
    # returned one finished string, so nothing could be spoken until the last
    # token was decoded and time-to-first-audio was proportional to *total*
    # reply length. Capping the words was the only way to make her answer
    # sooner — which is why spoken answers came out shallower than the same
    # question typed.
    #
    # Clause streaming removed that job entirely (see governed_stream.py and
    # mind_bridge.respond_streaming): the first clause is spoken while the
    # rest is still being produced, so length no longer buys latency.
    #
    # What is left is the honest reason, and it is about the medium rather
    # than the machine. Nobody says a 150-word paragraph on a phone call; they
    # answer, then wait to see if you want more, and reading a written-length
    # reply aloud is the thing that makes voice assistants exhausting. So this
    # is now a shape hint the model may exceed when the answer needs it — see
    # `MindBridge._reply_budget_for`, which sizes it per question — rather
    # than a ceiling protecting a latency budget.
    spoken_reply_words: int = field(
        default_factory=lambda: _env_int("AURA_VOICE_REPLY_WORDS", 45)
    )

    @classmethod
    def load(cls) -> DuplexConfig:
        return cls()
