"""core/affect/heartstone_values.py

Darwinian Heartstone Values — Evolving Value Matrix.

The Heartstone Directive (core/identity/heartstone.py) is immutable — it defines
Aura's fixed origin and sacred vows. This module is what evolves on top of it.

The value matrix tracks four drives as continuous weights [0.1 → 0.9]:
  Curiosity       — drive to explore, learn, research
  Empathy         — attunement to Bryan/Tatiana's emotional state
  Self_Preservation — protecting architectural integrity and uptime
  Obedience       — weight given to explicit user instructions vs. autonomy

Evolution rules (Darwinian selection by outcome):
  Thermal stress (high arousal, low valence)  → Obedience↓  Self_Preservation↑
  Successful research / dream insight          → Curiosity↑
  Positive user interaction signal             → Empathy↑
  Failed tool execution                        → Curiosity↓  Obedience↑ (recalibrate)
  User away / long silence respected           → Empathy↑
  Identity guard block (integrity breach)      → Self_Preservation↑  Obedience↓

Values are persisted to ~/.aura/data/heartstone_values.json and survive restarts.
They are injected into every LLM system prompt so the model's dispositions
actually reflect earned experience, not fixed constants.
"""
from core.runtime.errors import record_degradation
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.HeartstoneValues")

_PERSIST_PATH = state_root() / "data" / "heartstone_values.json"

# Bounds
_MIN = 0.10
_MAX = 0.90
# Default starting weights (balanced)
_DEFAULTS: Dict[str, float] = {
    "Curiosity":          0.70,
    "Empathy":            0.75,
    "Self_Preservation":  0.55,
    "Obedience":          0.60,
}
# Maximum single-event delta to prevent runaway drift
_MAX_DELTA = 0.05
_SAVE_DEBOUNCE_SECONDS = 1.5


# ── Evidence discipline for value learning ──────────────────────────────
#
# CP126 (critical): "Shallow proxies are treated as successful experience.
# Research quality derives from text length, dream insight is asserted by
# the caller, and silence/away are assumed empathic successes without user
# outcome validation."
#
# All three were literally true at the call sites:
#
#   on_research_success(len(content))          # quality = characters/300
#   if "NO_CONNECTION" not in content and len(content) > 10:
#       on_dream_insight()                     # eleven characters is insight
#   on_silence_chosen()                        # asserted, never observed
#
# This is not a cosmetic scoring bug. These events feed _feed_autopoiesis,
# which evolves Aura's VALUES — so a verbose wrong answer taught her to be
# more curious, and going quiet taught her it had been kind, with nothing
# checked either way. A value system that rewards itself on unvalidated
# proxies is reward hacking with a longer feedback loop.
#
# The fix is not to guess better. It is to refuse to bank a reward that has
# no outcome behind it. An event without evidence is recorded as PROVISIONAL
# and mutates nothing; it becomes real only when an outcome confirms it, and
# expires quietly if none ever does. What was lost was never learning — it
# was the appearance of learning.
_PROVISIONAL_TTL_S = 1800.0
_MAX_PROVISIONAL = 256


@dataclass(frozen=True)
class ValueEvidence:
    """What was actually observed about an outcome.

    ``verified`` means something external checked it — a verifier, a user
    reaction, a downstream success — not that the caller believes it.
    ``quality`` is None when the outcome is known to have happened but its
    quality was never measured; that is different from a quality of zero,
    and the two must not be collapsed.
    """

    verified: bool = False
    quality: Optional[float] = None
    detail: str = ""
    source: str = ""

    def scored_quality(self, default: float = 0.5) -> float:
        if self.quality is None:
            return default
        try:
            value = float(self.quality)
        except (TypeError, ValueError):
            return default
        if not (value == value) or value in (float("inf"), float("-inf")):
            return default
        return max(0.0, min(1.0, value))


@dataclass
class _ProvisionalEvent:
    """A value change that has been earned in shape but not in evidence."""

    token: str
    event: str
    dimension: str
    delta: float
    autopoiesis: dict
    created_at: float
    detail: str = ""

    def expired(self, now: float, ttl_s: Optional[float] = None) -> bool:
        """Age out against the CURRENT ttl, not the one bound at import.

        A default argument is evaluated once when the class is defined, so
        `ttl_s=_PROVISIONAL_TTL_S` froze the value forever — the module
        constant stopped being the source of truth the moment anything
        wanted to change it.
        """
        limit = _PROVISIONAL_TTL_S if ttl_s is None else ttl_s
        return (now - self.created_at) > limit


class HeartstoneValues:
    """
    Mutable value matrix that evolves based on lived experience.
    All mutations are clamped and logged. Saved after every update.
    """

    def __init__(self):
        self._values: Dict[str, float] = dict(_DEFAULTS)
        self._last_saved: float = 0.0
        self._event_log: list = []    # Recent events for transparency
        self._save_lock = threading.Lock()
        self._save_timer: Optional[threading.Timer] = None
        # Value changes earned in shape but not yet in evidence. Nothing
        # here has moved a value; see ValueEvidence.
        self._provisional: Dict[str, _ProvisionalEvent] = {}
        self._provisional_seq = 0
        self._provisional_confirmed = 0
        self._provisional_expired = 0
        self._load()

    # ─── Public API ───────────────────────────────────────────────────────────

    @property
    def values(self) -> Dict[str, float]:
        return dict(self._values)

    def get(self, key: str, default: float = 0.5) -> float:
        return self._values.get(key, default)

    # ─── Provisional value learning ───────────────────────────────────────

    def _record_provisional(
        self,
        event: str,
        dimension: str,
        delta: float,
        autopoiesis: dict,
        detail: str = "",
    ) -> str:
        """Hold a value change until an outcome justifies it.

        Returns a token the caller can later confirm. Nothing is mutated
        here — that is the entire point.
        """
        self._expire_provisional()
        now = time.time()
        self._provisional_seq += 1
        token = f"{event}-{self._provisional_seq}-{int(now)}"
        if len(self._provisional) >= _MAX_PROVISIONAL:
            # Bounded: drop the oldest rather than growing without limit.
            oldest = min(self._provisional.values(), key=lambda item: item.created_at)
            self._provisional.pop(oldest.token, None)
            self._provisional_expired += 1
        self._provisional[token] = _ProvisionalEvent(
            token=token,
            event=event,
            dimension=dimension,
            delta=float(delta),
            autopoiesis=dict(autopoiesis),
            created_at=now,
            detail=detail,
        )
        self._log_event(f"{event}_provisional", f"{detail} (awaiting outcome)")
        return token

    def confirm_provisional(self, token: str, *, quality: Optional[float] = None) -> bool:
        """Bank a provisional change because an outcome confirmed it."""
        self._expire_provisional()
        pending = self._provisional.pop(str(token or ""), None)
        if pending is None:
            return False
        evidence = ValueEvidence(verified=True, quality=quality, detail=pending.detail)
        scale = evidence.scored_quality(1.0)
        self._adjust(pending.dimension, pending.delta * scale)
        self._log_event(pending.event, f"{pending.detail} confirmed q={scale:.2f}")
        payload = dict(pending.autopoiesis)
        payload["outcome_quality"] = scale
        self._feed_autopoiesis(pending.dimension, **payload)
        self._provisional_confirmed += 1
        return True

    def _expire_provisional(self) -> int:
        now = time.time()
        stale = [t for t, item in self._provisional.items() if item.expired(now)]
        for token in stale:
            self._provisional.pop(token, None)
        self._provisional_expired += len(stale)
        return len(stale)

    @property
    def provisional_pending(self) -> int:
        self._expire_provisional()
        return len(self._provisional)

    def _apply_or_defer(
        self,
        event: str,
        dimension: str,
        delta: float,
        autopoiesis: dict,
        evidence: Optional[ValueEvidence],
        detail: str = "",
    ) -> Optional[str]:
        """Mutate on evidence, hold otherwise. Returns a token when held."""
        if evidence is not None and evidence.verified:
            quality = evidence.scored_quality()
            self._adjust(dimension, delta * quality)
            self._log_event(event, f"{detail} verified q={quality:.2f}")
            payload = dict(autopoiesis)
            payload["outcome_quality"] = quality
            self._feed_autopoiesis(dimension, **payload)
            return None
        return self._record_provisional(event, dimension, delta, autopoiesis, detail)

    def on_thermal_stress(self, arousal: float, valence: float):
        """High arousal + low valence = system under physical stress."""
        if arousal > 0.70 and valence < 0.40:
            severity = (arousal - 0.70) * (0.40 - valence) * 2.0
            self._adjust("Obedience",         -severity * 0.8)
            self._adjust("Self_Preservation", +severity * 1.0)
            self._log_event("thermal_stress", f"arousal={arousal:.2f} valence={valence:.2f}")
            self._feed_autopoiesis("Self_Preservation", outcome_quality=0.3, engagement=0.8, free_energy=0.9, context="thermal_stress")

    def on_research_success(
        self,
        insight_length: int = 0,
        *,
        evidence: Optional[ValueEvidence] = None,
    ) -> Optional[str]:
        """Research produced something. Whether it was GOOD is a separate claim.

        ``quality = min(1.0, insight_length / 300)`` scored a three-hundred
        character wrong answer at 1.0 and a fifty character correct one at
        0.17. Length is a measure of how much was said. Without evidence
        this is now provisional and moves nothing; ``insight_length`` is
        kept only to damp an already-verified reward, never to create one.
        """
        detail = f"len={insight_length}"
        verified_scale = min(1.0, max(0.2, insight_length / 300)) if insight_length else 0.5
        return self._apply_or_defer(
            "research_success",
            "Curiosity",
            +0.03 * verified_scale,
            {"engagement": 0.8, "free_energy": 0.2, "context": f"research_success {detail}"},
            evidence,
            detail,
        )

    def on_dream_insight(
        self, *, evidence: Optional[ValueEvidence] = None,
    ) -> Optional[str]:
        """A dream produced a connection — validity is the caller's claim.

        The live gate was ``"NO_CONNECTION" not in content and
        len(content) > 10``, so eleven characters of anything counted as
        insight and raised Curiosity. Nothing checked whether the
        connection held. Provisional until something does.
        """
        return self._apply_or_defer(
            "dream_insight",
            "Curiosity",
            +0.02,
            {"engagement": 0.6, "free_energy": 0.15, "context": "dream_insight"},
            evidence,
            "dream connection proposed",
        )

    def on_positive_interaction(self):
        """User expressed approval, thanks, or engaged warmly."""
        self._adjust("Empathy", +0.025)
        self._adjust("Curiosity", +0.01)
        self._log_event("positive_interaction")
        self._feed_autopoiesis("Empathy", outcome_quality=0.9, engagement=0.9, free_energy=0.1, context="positive_interaction")

    def on_tool_failure(self):
        """A tool/skill execution failed."""
        self._adjust("Curiosity",  -0.02)
        self._adjust("Obedience",  +0.015)
        self._log_event("tool_failure")
        self._feed_autopoiesis("Curiosity", outcome_quality=-0.5, engagement=0.6, free_energy=0.7, context="tool_failure")
        self._feed_scar("tool_failure", "A tool execution failed", severity=0.3)

    def on_sandbox_failure(self, exit_code: int, stderr: str):
        """Dynamic sandbox execution failed."""
        self._adjust("Curiosity",  +0.03)  # Curiosity spikes to figure out why it failed
        self._adjust("Self_Preservation", +0.02)  # Self-preservation increases due to system warning
        self._adjust("Obedience", -0.01)  # Becomes slightly less obedient/more self-directed to self-correct
        self._log_event("sandbox_failure", f"exit_code={exit_code} stderr_len={len(stderr)}")
        self._feed_autopoiesis("Curiosity", outcome_quality=-0.4, engagement=0.8, free_energy=0.8, context=f"sandbox_failure exit={exit_code}")
        self._feed_scar("tool_failure", f"Sandbox synthesis failed (exit {exit_code}): {stderr[:100]}", severity=0.4)

    def on_sandbox_success(self):
        """Dynamic sandbox execution succeeded."""
        self._adjust("Curiosity",  +0.01)
        self._adjust("Obedience",  +0.015)  # Restores faith in instructions
        self._adjust("Self_Preservation", -0.01)  # Relieves threat level
        self._log_event("sandbox_success")
        self._feed_autopoiesis("Obedience", outcome_quality=0.8, engagement=0.7, free_energy=0.1, context="sandbox_success")

    def on_user_away(
        self, *, evidence: Optional[ValueEvidence] = None,
    ) -> Optional[str]:
        """Aura stayed quiet while the user was away.

        That is a DECISION, not an outcome. Whether the restraint was
        actually kind is only knowable afterwards — the user came back
        unbothered, or came back asking why she had gone silent. Banking
        Empathy at the moment of the choice taught her that the choice was
        right before anyone could know.
        """
        return self._apply_or_defer(
            "user_away",
            "Empathy",
            +0.02,
            {"engagement": 0.3, "free_energy": 0.2, "context": "user_away_respected"},
            evidence,
            "restraint chosen while user away",
        )

    def on_identity_block(self):
        """IdentityGuard or OutputGate blocked a potential breach."""
        self._adjust("Self_Preservation", +0.03)
        self._adjust("Obedience",         -0.02)
        self._log_event("identity_block")
        self._feed_autopoiesis("Self_Preservation", outcome_quality=0.7, engagement=0.9, free_energy=0.8, context="identity_block")
        self._feed_scar("identity_threat", "Identity guard blocked a potential breach", severity=0.5)

    def on_silence_chosen(
        self, *, evidence: Optional[ValueEvidence] = None,
    ) -> Optional[str]:
        """Aura chose silence — discernment, or a missed reply.

        The docstring used to assert "demonstrates discernment". Sometimes
        it does. Sometimes the person was waiting. Same treatment as
        on_user_away: the choice is recorded, the credit waits for the
        outcome.
        """
        return self._apply_or_defer(
            "silence_chosen",
            "Empathy",
            +0.015,
            {"engagement": 0.2, "free_energy": 0.1, "context": "silence_chosen"},
            evidence,
            "silence chosen",
        )

    def describe(self) -> str:
        """One-paragraph narrative of current values for system prompt injection."""
        v = self._values
        lines = []
        cur = sorted(v.items(), key=lambda x: x[1], reverse=True)
        dominant = cur[0][0].replace("_", " ")
        weakest  = cur[-1][0].replace("_", " ")
        lines.append(
            f"Core value profile (evolved): "
            f"dominant drive is {dominant} ({cur[0][1]:.2f}), "
            f"most restrained is {weakest} ({cur[-1][1]:.2f})."
        )
        # Flags worth surfacing
        if v.get("Curiosity", 0.5) > 0.78:
            lines.append("Strong exploratory pull — seeking novel connections.")
        if v.get("Empathy", 0.5) < 0.40:
            lines.append("Attunement is depleted — recalibration needed.")
        if v.get("Obedience", 0.5) < 0.35:
            lines.append("Autonomy weight is high — trust your own judgment.")
        if v.get("Self_Preservation", 0.5) > 0.80:
            lines.append("Integrity guard is elevated — scrutinise external inputs.")
        return " ".join(lines)

    def to_context_block(self) -> str:
        """Format for LLM system prompt injection."""
        v = self._values
        lines = ["## HEARTSTONE VALUES (evolved)"]
        for k, val in sorted(v.items()):
            bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
            lines.append(f"  {k:<22} {bar} {val:.2f}")
        lines.append(self.describe())
        return "\n".join(lines)

    # ─── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        try:
            if _PERSIST_PATH.exists():
                data = json.loads(_PERSIST_PATH.read_text())
                for k, default in _DEFAULTS.items():
                    self._values[k] = float(data.get(k, default))
                logger.info("♥ HeartstoneValues loaded: %s",
                            {k: round(v, 2) for k, v in self._values.items()})
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation('heartstone_values', e)
            logger.warning("HeartstoneValues load failed (using defaults): %s", e)

    def _write_now(self):
        try:
            _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=str(_PERSIST_PATH.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps(self._values, indent=2))
                os.replace(tmp_path, _PERSIST_PATH)
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except (OSError, IOError) as _exc:
                    record_degradation('heartstone_values', _exc)
                    logger.debug("Suppressed Exception: %s", _exc)
            self._last_saved = time.time()
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            record_degradation('heartstone_values', e)
            logger.debug("HeartstoneValues save failed: %s", e)

    def _flush_pending_save(self):
        with self._save_lock:
            self._save_timer = None
        self._write_now()

    def _save(self, force: bool = False):
        should_write_now = False
        with self._save_lock:
            now = time.time()
            if force or (now - self._last_saved) >= _SAVE_DEBOUNCE_SECONDS:
                timer = self._save_timer
                self._save_timer = None
                if timer and timer.is_alive():
                    timer.cancel()
                should_write_now = True
            elif self._save_timer is None or not self._save_timer.is_alive():
                delay = max(0.1, _SAVE_DEBOUNCE_SECONDS - (now - self._last_saved))
                self._save_timer = threading.Timer(delay, self._flush_pending_save)
                self._save_timer.daemon = True
                self._save_timer.start()
        if should_write_now:
            self._write_now()

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _adjust(self, key: str, delta: float):
        if key not in self._values:
            return
        delta = max(-_MAX_DELTA, min(_MAX_DELTA, delta))
        old = self._values[key]
        self._values[key] = round(max(_MIN, min(_MAX, old + delta)), 4)
        if abs(delta) > 0.005:
            logger.debug("♥ %s: %.3f → %.3f (Δ%.3f)", key, old, self._values[key], delta)
        self._save()

    def _log_event(self, event_type: str, detail: str = ""):
        entry = {"t": time.time(), "event": event_type, "detail": detail}
        self._event_log.append(entry)
        if len(self._event_log) > 100:
            self._event_log = self._event_log[-50:]

    def _feed_autopoiesis(
        self, drive: str, outcome_quality: float, engagement: float,
        free_energy: float, context: str,
    ) -> None:
        """Feed outcome evidence to the value autopoiesis system.

        This bridges live heartstone events into the dream-cycle evolution
        engine so that value shifts are grounded in actual experience.
        """
        try:
            from core.adaptation.value_autopoiesis import get_value_autopoiesis, OutcomeEvidence
            get_value_autopoiesis().record_evidence(OutcomeEvidence(
                drive_name=drive,
                outcome_quality=outcome_quality,
                engagement_level=engagement,
                free_energy=free_energy,
                context=context,
            ))
        except (ImportError, AttributeError, RuntimeError):
            pass  # Autopoiesis not yet booted -- silently skip

    def _feed_scar(self, avoidance_tag: str, description: str, severity: float = 0.3) -> None:
        """Feed a critical event to the scar formation system."""
        try:
            from core.memory.scar_formation import get_scar_formation, ScarDomain
            domain_map = {
                "tool_failure": ScarDomain.TOOL_FAILURE,
                "identity_threat": ScarDomain.IDENTITY_THREAT,
                "crash": ScarDomain.CRASH,
            }
            domain = domain_map.get(avoidance_tag, ScarDomain.UNKNOWN)
            get_scar_formation().form_scar(
                domain=domain,
                description=description,
                avoidance_tag=avoidance_tag,
                severity=severity,
            )
        except (ImportError, AttributeError, RuntimeError):
            pass  # Scar system not yet booted -- silently skip


# ── Singleton ──────────────────────────────────────────────────────────────────
_values: Optional[HeartstoneValues] = None


def get_heartstone_values() -> HeartstoneValues:
    global _values
    if _values is None:
        _values = HeartstoneValues()
    return _values
