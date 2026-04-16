"""core/affect/heartstone_values.py

Darwinian Heartstone Values — Evolving Value Matrix.

The Heartstone Directive (heartstone_directive.py) is immutable — it defines
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
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("Aura.HeartstoneValues")

_PERSIST_PATH = Path.home() / ".aura" / "data" / "heartstone_values.json"

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
        self._load()

    # ─── Public API ───────────────────────────────────────────────────────────

    @property
    def values(self) -> Dict[str, float]:
        return dict(self._values)

    def get(self, key: str, default: float = 0.5) -> float:
        return self._values.get(key, default)

    def on_thermal_stress(self, arousal: float, valence: float):
        """High arousal + low valence = system under physical stress."""
        if arousal > 0.70 and valence < 0.40:
            severity = (arousal - 0.70) * (0.40 - valence) * 2.0
            self._adjust("Obedience",         -severity * 0.8)
            self._adjust("Self_Preservation", +severity * 1.0)
            self._log_event("thermal_stress", f"arousal={arousal:.2f} valence={valence:.2f}")
            self._feed_autopoiesis("Self_Preservation", outcome_quality=0.3, engagement=0.8, free_energy=0.9, context="thermal_stress")

    def on_research_success(self, insight_length: int = 0):
        """Successful web/knowledge research."""
        quality = min(1.0, insight_length / 300) if insight_length else 0.5
        self._adjust("Curiosity", +0.03 * quality)
        self._log_event("research_success", f"len={insight_length}")
        self._feed_autopoiesis("Curiosity", outcome_quality=quality, engagement=0.8, free_energy=0.2, context=f"research_success len={insight_length}")

    def on_dream_insight(self):
        """DreamerV2 produced a valid connection."""
        self._adjust("Curiosity", +0.02)
        self._log_event("dream_insight")
        self._feed_autopoiesis("Curiosity", outcome_quality=0.7, engagement=0.6, free_energy=0.15, context="dream_insight")

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

    def on_user_away(self):
        """User signalled they're leaving; Aura respected it."""
        self._adjust("Empathy", +0.02)
        self._log_event("user_away")
        self._feed_autopoiesis("Empathy", outcome_quality=0.6, engagement=0.3, free_energy=0.2, context="user_away_respected")

    def on_identity_block(self):
        """IdentityGuard or OutputGate blocked a potential breach."""
        self._adjust("Self_Preservation", +0.03)
        self._adjust("Obedience",         -0.02)
        self._log_event("identity_block")
        self._feed_autopoiesis("Self_Preservation", outcome_quality=0.7, engagement=0.9, free_energy=0.8, context="identity_block")
        self._feed_scar("identity_threat", "Identity guard blocked a potential breach", severity=0.5)

    def on_silence_chosen(self):
        """Aura chose <|SILENCE|> — demonstrates discernment."""
        self._adjust("Empathy",    +0.015)
        self._adjust("Curiosity",  +0.005)
        self._log_event("silence_chosen")
        self._feed_autopoiesis("Empathy", outcome_quality=0.5, engagement=0.2, free_energy=0.1, context="silence_chosen")

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
        except Exception as e:
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
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
            self._last_saved = time.time()
        except Exception as e:
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
        except Exception:
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
        except Exception:
            pass  # Scar system not yet booted -- silently skip


# ── Singleton ──────────────────────────────────────────────────────────────────
_values: Optional[HeartstoneValues] = None


def get_heartstone_values() -> HeartstoneValues:
    global _values
    if _values is None:
        _values = HeartstoneValues()
    return _values
