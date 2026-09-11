"""core/voice/duplex/backchannel.py — "mhm" while they are still talking.

Overlapping acknowledgement is most of the difference between something
that is listening and something that is recording. It also carries real
information: it tells the speaker they still have the floor and are being
followed, which is why people keep talking naturally instead of stopping to
check whether the line dropped.

Two failure modes, both worse than silence:

  1. Too frequent. An agent that says "mhm" every two seconds is grating,
     and reads as nervousness rather than attention.
  2. Wrongly placed. Acknowledgement lands at a *prosodic boundary* — the
     brief dip at the end of a clause. Dropped mid-word it sounds like an
     interruption, because that is what it is.

So the gate is: they have held the floor a while, we are inside a
micro-pause too short to be an endpoint, the cooldown has elapsed, and a
probability check passes so the rhythm never becomes mechanical.

The choice of token is driven by her actual substrate state rather than a
flat random pick, so a curious Aura says "oh?" where a settled one says
"mm". Nothing here asserts anything about content, so a backchannel can
never be *wrong* the way a sentence can.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass

from core.runtime.errors import record_degradation
from core.voice.duplex.config import BackchannelConfig
from core.runtime.the_laboratory import seeded

logger = logging.getLogger("Aura.Voice.Backchannel")


# Grouped by what they actually communicate. Register is chosen from her
# state; the item within a register is random so it does not become a tic.
_REGISTERS: dict[str, tuple[str, ...]] = {
    # Default: "I'm with you, keep going."
    "neutral": ("mhm.", "mm.", "yeah.", "right.", "okay.", "mm-hm."),
    # Elevated curiosity / arousal: leaning in.
    "curious": ("oh?", "huh.", "oh really?", "hm!", "interesting."),
    # High agreement / positive valence.
    "affirming": ("yeah.", "right.", "exactly.", "totally.", "yep."),
    # Low arousal — quieter, sparser tokens.
    "settled": ("mm.", "mhm.", "hm."),
    # Something sounds heavy; acknowledgement without brightness.
    "grave": ("mm.", "yeah.", "I hear you."),
}


@dataclass(slots=True)
class BackchannelDecision:
    should_emit: bool
    text: str = ""
    register: str = ""
    gain: float = 0.0
    reason: str = ""


class BackchannelReflex:
    """Sub-300 ms decision loop. Deliberately contains no LLM call.

    A backchannel that waited on cognition would land a second late, which
    is worse than not making it at all.
    """

    def __init__(self, config: BackchannelConfig | None = None, rng: random.Random | None = None) -> None:
        self._config = config or BackchannelConfig()
        self._rng = rng or seeded("voice.backchannel")
        self._last_emit_at = 0.0
        self._floor_started_at = 0.0
        self._emitted_this_turn = 0
        self._recent: list[str] = []

    def on_user_turn_start(self, now: float | None = None) -> None:
        self._floor_started_at = now if now is not None else time.monotonic()
        self._emitted_this_turn = 0

    def on_user_turn_end(self) -> None:
        self._floor_started_at = 0.0

    def consider(
        self,
        *,
        silence_ms: float,
        speech_ms: float,
        aura_is_speaking: bool,
        now: float | None = None,
        substrate: dict[str, float] | None = None,
    ) -> BackchannelDecision:
        """Evaluate the current micro-pause as a slot for acknowledgement."""
        cfg = self._config
        if not cfg.enabled:
            return BackchannelDecision(False, reason="disabled")
        # Talking over her own output is not backchannelling, it is a mess.
        if aura_is_speaking:
            return BackchannelDecision(False, reason="aura_speaking")

        now = now if now is not None else time.monotonic()

        if speech_ms < cfg.min_floor_ms:
            return BackchannelDecision(False, reason="floor_too_short")

        # The window that makes this a clause boundary rather than either
        # inter-word silence (too short) or a turn end (too long).
        if not (cfg.pause_min_ms <= silence_ms <= cfg.pause_max_ms):
            return BackchannelDecision(False, reason="not_a_pause_boundary")

        if (now - self._last_emit_at) * 1000.0 < cfg.cooldown_ms:
            return BackchannelDecision(False, reason="cooldown")

        # Taper: the second acknowledgement in one turn is less likely than
        # the first, the third less again. Long monologues get sparse "mhm"s
        # rather than a metronome.
        taper = cfg.fire_probability / (1.0 + self._emitted_this_turn)
        if self._rng.random() > taper:
            return BackchannelDecision(False, reason="probability_declined")

        register = self._select_register(substrate or {})
        text = self._select_token(register)

        self._last_emit_at = now
        self._emitted_this_turn += 1
        return BackchannelDecision(
            should_emit=True,
            text=text,
            register=register,
            gain=cfg.gain,
            reason="prosodic_boundary",
        )

    def _select_register(self, substrate: dict[str, float]) -> str:
        """Map her affective state onto a listening register."""
        try:
            valence = float(substrate.get("valence", 0.0))
            arousal = float(substrate.get("arousal", 0.5))
            curiosity = float(substrate.get("curiosity", 0.0))

            if curiosity >= 0.65 and arousal >= 0.5:
                return "curious"
            if valence <= -0.35:
                return "grave"
            if valence >= 0.45 and arousal >= 0.45:
                return "affirming"
            if arousal <= 0.3:
                return "settled"
        except (TypeError, ValueError) as exc:
            record_degradation(
                "voice_duplex.backchannel",
                exc,
                action="used neutral backchannel register",
                severity="debug",
            )
        return "neutral"

    def _select_token(self, register: str) -> str:
        options = list(_REGISTERS.get(register) or _REGISTERS["neutral"])
        # Avoid immediate repetition — saying "mhm" three times running is
        # the exact thing that makes this feel canned.
        fresh = [o for o in options if o not in self._recent[-2:]]
        choice = self._rng.choice(fresh or options)
        self._recent.append(choice)
        if len(self._recent) > 4:
            self._recent.pop(0)
        return choice

    @property
    def emitted_this_turn(self) -> int:
        return self._emitted_this_turn
