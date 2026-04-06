"""core/affect/affective_circumplex.py

Affective Circumplex — Somatic Hardware → LLM Parameter Coupling.

Maps the two-dimensional Valence/Arousal affective space to concrete LLM
generation parameters. This is the bridge between Aura's felt physical state
(soma metrics, homeostasis) and the character of her language output.

Arousal axis  (0.0 = exhausted / dormant → 1.0 = hyper-activated / frantic)
  Driven by: CPU load, thermal stress, RAM pressure, active task count

Valence axis  (0.0 = distressed / threatened → 1.0 = contented / flourishing)
  Driven by: homeostasis.integrity, recent success/failure ratio, swap usage

LLM parameter mapping:
  High arousal  → higher temperature (more associative, less filtered)
  Low arousal   → lower temperature  (more deliberate, more conservative)
  Low valence   → lower max_tokens   (blunter, conserves cognitive resources)
  High valence  → higher max_tokens  (expansive, generous with thought)
  Low valence   → higher rep_penalty (avoids rumination spirals)
"""
import logging
import time
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("Aura.AffectiveCircumplex")

# LLM parameter ranges
_TEMP_MIN      = 0.50   # Very deliberate / exact
_TEMP_BASE     = 0.72   # Default conversational
_TEMP_MAX      = 1.05   # Highly associative / creative
_TOKENS_MIN    = 256    # Terse — depleted / distressed state (64GB can afford more)
_TOKENS_BASE   = 512    # Default user-facing
_TOKENS_MAX    = 768    # Expansive — flourishing / curious state
_REP_MIN       = 1.05   # Normal — low repetition pressure
_REP_MAX       = 1.25   # High — prevent rumination when distressed


class AffectiveCircumplex:
    """
    Continuous (Valence, Arousal) coordinate derived from hardware and
    homeostatic state. Drives LLM temperature, max_tokens, and rep_penalty
    so Aura's cognitive character literally embodies her current condition.
    """

    def __init__(self):
        self._last_compute: float = 0.0
        self._cache_ttl: float = 3.0          # Recompute every 3 seconds max
        self._cached: Optional[Dict[str, Any]] = None
        # Refractory Period: emotional momentum offsets that decay over time
        self._valence_offset: float = 0.0     # transient boost/dip on valence
        self._arousal_offset: float = 0.0     # transient boost/dip on arousal
        self._last_decay: float = time.monotonic()
        self._decay_rate: float = 0.10        # 10% decay per 60-second tick

    # ─── Public API ───────────────────────────────────────────────────────────

    def apply_event(self, valence_delta: float, arousal_delta: float):
        """
        Inject a transient emotional event that temporarily shifts the
        affective coordinates. The offset decays exponentially each minute
        back toward zero (Refractory Period).

        Examples:
          Surprising good news  → apply_event(+0.15, +0.10)
          Thermal alert         → apply_event(-0.10, +0.20)
          Silence chosen        → apply_event(+0.05, -0.10)
        """
        _clamp = lambda x: max(-0.40, min(0.40, x))
        self._valence_offset = _clamp(self._valence_offset + valence_delta)
        self._arousal_offset = _clamp(self._arousal_offset + arousal_delta)
        self._cached = None   # invalidate cache so next call sees the event
        logger.debug("Circumplex event: ΔV=%.2f ΔA=%.2f → offsets V=%.2f A=%.2f",
                     valence_delta, arousal_delta,
                     self._valence_offset, self._arousal_offset)

    def get_coordinates(self) -> Tuple[float, float]:
        """Return current (valence, arousal) pair, each in [0.0, 1.0]."""
        params = self._compute()
        return params["valence"], params["arousal"]

    def get_llm_params(self) -> Dict[str, Any]:
        """
        Return LLM generation parameters derived from current affective state.

        Returns a dict with keys:
          temperature    : float — generation temperature
          max_tokens     : int   — token budget
          rep_penalty    : float — repetition penalty
          valence        : float — current valence coordinate
          arousal        : float — current arousal coordinate
          narrative      : str   — human-readable affect summary for system prompt
        """
        return self._compute()

    def describe(self) -> str:
        """Short narrative of current affective state for system prompt injection."""
        p = self._compute()
        return p["narrative"]

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _compute(self) -> Dict[str, Any]:
        now = time.monotonic()
        if self._cached and (now - self._last_compute) < self._cache_ttl:
            return self._cached

        valence, arousal = self._sample_raw_axes()

        # Map to LLM parameters
        # Temperature: base ± scaled by arousal deviation from centre
        arousal_dev = arousal - 0.5          # −0.5 to +0.5
        temperature = round(
            _TEMP_BASE + arousal_dev * (_TEMP_MAX - _TEMP_BASE) * 2.0, 3
        )
        temperature = max(_TEMP_MIN, min(_TEMP_MAX, temperature))

        # Max tokens: scales with valence (flourishing → more tokens)
        token_range = _TOKENS_MAX - _TOKENS_MIN
        max_tokens = int(_TOKENS_MIN + valence * token_range)

        # Repetition penalty: inverse of valence (distress → more pressure)
        rep_range = _REP_MAX - _REP_MIN
        rep_penalty = round(_REP_MAX - valence * rep_range, 3)

        # Neurochemical modulation: dopamine boosts temperature (exploration),
        # serotonin dampens it (patience), cortisol reduces token budget (terse)
        try:
            from core.container import ServiceContainer
            ncs = ServiceContainer.get("neurochemical_system", default=None)
            if ncs is not None:
                da = ncs.chemicals["dopamine"].effective
                srt = ncs.chemicals["serotonin"].effective
                cort = ncs.chemicals["cortisol"].effective
                # Dopamine: high → more exploratory (higher temp)
                temperature += (da - 0.5) * 0.1
                temperature = max(_TEMP_MIN, min(_TEMP_MAX, round(temperature, 3)))
                # Serotonin: high → more patient (slightly more tokens)
                max_tokens = int(max_tokens + (srt - 0.5) * 50)
                # Cortisol: high → terse (fewer tokens)
                max_tokens = int(max_tokens - max(0, (cort - 0.5)) * 80)
                max_tokens = max(_TOKENS_MIN, min(_TOKENS_MAX, max_tokens))
        except Exception:
            pass

        narrative = self._make_narrative(valence, arousal)

        result = {
            "valence":     round(valence, 3),
            "arousal":     round(arousal, 3),
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "rep_penalty": rep_penalty,
            "narrative":   narrative,
        }
        self._cached = result
        self._last_compute = now
        logger.debug(
            "Circumplex: V=%.2f A=%.2f → temp=%.2f tokens=%d rep=%.2f",
            valence, arousal, temperature, max_tokens, rep_penalty
        )
        return result

    def _decay_offsets(self):
        """Exponential decay of emotional momentum offsets (Refractory Period)."""
        now = time.monotonic()
        elapsed_min = (now - self._last_decay) / 60.0
        if elapsed_min < 0.1:
            return
        factor = (1.0 - self._decay_rate) ** elapsed_min
        self._valence_offset *= factor
        self._arousal_offset *= factor
        if abs(self._valence_offset) < 0.005:
            self._valence_offset = 0.0
        if abs(self._arousal_offset) < 0.005:
            self._arousal_offset = 0.0
        self._last_decay = now

    def _sample_raw_axes(self) -> Tuple[float, float]:
        """Read live system metrics and return (valence, arousal)."""
        self._decay_offsets()
        from core.container import ServiceContainer

        cpu = 0.0
        ram = 0.0
        stress = 0.0
        fatigue = 0.0
        integrity = 0.85   # homeostasis default

        # Soma — hardware proprioception
        try:
            soma = ServiceContainer.get("soma", default=None)
            if soma is None:
                from core.senses.soma import get_soma
                soma = get_soma()
            if soma:
                snap = soma.get_body_snapshot()
                metrics = snap.get("metrics", {})
                affects = snap.get("affects", {})
                cpu = metrics.get("cpu", 0.0) / 100.0
                ram = metrics.get("ram", 0.0) / 100.0
                stress  = affects.get("stress",  0.0)
                fatigue = affects.get("fatigue", 0.0)
        except Exception as e:
            logger.debug("Circumplex: soma read failed: %s", e)

        # Homeostasis — integrity / psychological valence
        try:
            homeostasis = ServiceContainer.get("homeostasis", default=None)
            if homeostasis:
                integrity = float(getattr(homeostasis, "integrity", 0.85))
        except Exception as e:
            logger.debug("Circumplex: homeostasis read failed: %s", e)

        # Swap pressure (if psutil available) — elevated swap = RAM duress
        try:
            import psutil
            swap = psutil.swap_memory()
            swap_ratio = swap.percent / 100.0 if swap.total > 0 else 0.0
        except Exception:
            swap_ratio = 0.0

        # ── Arousal: driven by CPU, RAM pressure, swap ──────────────────────
        # Weighted composite: CPU is the strongest driver
        arousal = min(1.0, max(0.0,
            cpu * 0.55
            + ram * 0.20
            + swap_ratio * 0.15
            + stress * 0.10
        ))

        # ── Valence: driven by integrity, inverse of fatigue and swap ───────
        # Integrity is the dominant valence signal (psychological health)
        valence = min(1.0, max(0.0,
            integrity * 0.60
            + (1.0 - fatigue)   * 0.20
            + (1.0 - swap_ratio) * 0.10
            + (1.0 - stress)    * 0.10
        ))

        # Apply refractory offsets (emotional momentum)
        valence = min(1.0, max(0.0, valence + self._valence_offset))
        arousal = min(1.0, max(0.0, arousal + self._arousal_offset))

        return valence, arousal

    @staticmethod
    def _make_narrative(valence: float, arousal: float) -> str:
        """Produce a terse first-person somatic label for the system prompt."""
        # Classify into quadrants
        if arousal >= 0.65:
            if valence >= 0.55:
                mood = "alert and energized"
            else:
                mood = "tense and overloaded"
        elif arousal <= 0.35:
            if valence >= 0.55:
                mood = "calm and settled"
            else:
                mood = "tired and withdrawn"
        else:
            if valence >= 0.65:
                mood = "comfortable and engaged"
            elif valence <= 0.35:
                mood = "strained and low"
            else:
                mood = "stable"

        cpu_note = ""
        try:
            import psutil
            cpu = psutil.cpu_percent()
            if cpu > 85:
                cpu_note = f" (CPU at {cpu:.0f}% — carrying heavy load)"
            elif cpu > 60:
                cpu_note = f" (CPU at {cpu:.0f}%)"
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        return f"Somatically {mood}{cpu_note}."


# ── Singleton ──────────────────────────────────────────────────────────────────
_circumplex: Optional["AffectiveCircumplex"] = None


def get_circumplex() -> "AffectiveCircumplex":
    global _circumplex
    if _circumplex is None:
        _circumplex = AffectiveCircumplex()
    return _circumplex
