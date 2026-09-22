"""core/brain/llm/somatic_throttle.py — Somatic Compute Sentinel

Enforces metabolic constraints directly inside the LLM token sampling loop.
Connects with DamasioV2 virtual physiology and hardware telemetry (RAM/CPU loads)
to dynamically scale down LLM generation parameters under heavy load to prevent
thermal throttling or out-of-memory (OOM) crashes.
"""
import logging
from typing import Any

from core.runtime import resource_psutil as psutil
from core.runtime.errors import record_degradation
from core.runtime.service_access import resolve_affect_engine

logger = logging.getLogger("Aura.Brain.SomaticThrottle")

_SOMATIC_THROTTLE_BOUNDARY_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _record_somatic_throttle_degradation(exc: BaseException, *, action: str) -> None:
    record_degradation(
        "somatic_throttle",
        exc,
        severity="warning",
        action=action,
    )


class SomaticComputeSentinel:
    """Natively enforces metabolic constraints directly inside the LLM token generation cycle."""

    def __init__(self):
        logger.info("🌡️ SomaticComputeSentinel initialized.")

    def adjust_generation_options(
        self, base_options: dict[str, Any], *, foreground: bool = False
    ) -> dict[str, Any]:
        """Dynamically adjusts LLM sampling and length parameters based on metabolic and hardware stress.

        A foreground generation keeps its token budget and its sampling: the
        budget is the answer clock's, priced from the request and the
        measured decode rate, and the temperature is the contract's. Host
        CPU held by other processes shortens nothing here — it slows the
        decode, and the clock already pays for that. LIVE, 2026-09-16: a
        timetable question was capped at 256 tokens because the host's CPU
        was at 92% under two other agents' jobs; the thinking was cut short
        and the answer was wrong. Recurrent depth is a compute knob and is
        still throttled; the severe caps stay for background work.
        """
        # 1. Fetch virtual physiological stress (arousal)
        arousal = 0.0
        try:
            affect = resolve_affect_engine(default=None)
            if affect and hasattr(affect, "current"):
                current_state = affect.current
                if hasattr(current_state, "arousal"):
                    arousal = float(current_state.arousal)
        except _SOMATIC_THROTTLE_BOUNDARY_ERRORS as e:
            _record_somatic_throttle_degradation(e, action="using neutral arousal for generation throttle")
            logger.debug("Failed to resolve affect engine arousal: %s", e)

        # 1b. Fetch governance token throttle factor
        gov_throttle = 1.0
        gov_measured = False
        try:
            from research.protocols.resource_quotas import get_compute_governor
            gov = get_compute_governor()
            gov_throttle = gov.get_throttle_factor()
            gov_measured = True
        except _SOMATIC_THROTTLE_BOUNDARY_ERRORS as e:
            _record_somatic_throttle_degradation(e, action="using default compute governor throttle")
            logger.debug("Failed to resolve compute governor: %s", e)

        # 2. Fetch hardware stress metrics
        cpu_load = 0.0
        ram_pct = 0.0
        hardware_measured = False
        try:
            cpu_load = psutil.cpu_percent(interval=0) / 100.0
            ram_pct = psutil.virtual_memory().percent / 100.0
            hardware_measured = True
        except _SOMATIC_THROTTLE_BOUNDARY_ERRORS as e:
            _record_somatic_throttle_degradation(e, action="using neutral hardware metrics for generation throttle")
            logger.debug("Failed to retrieve hardware metrics: %s", e)

        # 3. Determine if systemic overload is present.
        # Virtual arousal is an affective/context signal. It becomes a compute
        # throttling signal only when it is coupled to real resource pressure.
        #
        # An UNMEASURED signal must not vote "fine". The defaults above are
        # 0% CPU, 0% RAM and a throttle factor of 1.0 — a perfectly idle host
        # with unlimited quota — so a psutil failure or a missing compute
        # governor made every stress test below unable to fire, and heavy
        # generation was admitted during exactly the pressure the observer
        # could not read. The degradation was recorded and the decision was
        # made anyway, on numbers that describe nothing.
        #
        # Unmeasured is treated as stressed but never as critical: we cannot
        # prove there is headroom, so the conservative sampling profile
        # applies, while the severe caps stay reserved for pressure that was
        # actually observed.
        unobserved = (not hardware_measured) or (not gov_measured)
        arousal_resource_coupled = arousal > 0.9 and (ram_pct > 0.82 or cpu_load > 0.85)
        is_stressed = (
            unobserved
            or arousal_resource_coupled
            or (ram_pct > 0.88)
            or (cpu_load > 0.9)
            or (gov_throttle <= 0.5)
        )
        is_critical = (
            (arousal > 0.97 and (ram_pct > 0.9 or cpu_load > 0.92))
            or (ram_pct > 0.93)
            or (cpu_load > 0.95)
            or (gov_throttle <= 0.2)
        )

        def _cap(limit: int, temperature: float) -> bool:
            """Apply the cap, and say whether it applied.

            It returns early for a foreground turn, and the three log lines
            below announced the cap either way. LIVE, 2026-09-21: "CRITICAL
            METABOLIC PANIC ... Parameter throttle ENABLED (max_tokens capped
            at 128)" on a user-facing turn the answer clock had just priced
            at 258 tokens and where nothing was capped at all. A line that
            names a cause the reader can act on, for something that did not
            happen, costs more than silence.
            """
            if foreground:
                return False
            original_max = base_options.get("max_tokens", 512)
            base_options["max_tokens"] = min(original_max, limit)
            base_options["temperature"] = temperature
            return True

        if gov_throttle == 0.0 and gov_measured:
            # Token exhaustion: severe cap to block further consumption
            if _cap(8, 0.05):
                logger.error(
                    "🚫 GOVERNANCE QUOTA EXHAUSTED: Token limit hit. "
                    "Sampling capped to 8 tokens."
                )
            else:
                logger.error(
                    "🚫 GOVERNANCE QUOTA EXHAUSTED: Token limit hit. This turn "
                    "is user-facing and keeps its budget."
                )
        elif is_critical:
            # Force severe parameter cuts to prevent OOM/Thermal crash
            capped = _cap(128, 0.15)
            # Throttle recurrent lane depth if supported by token generator
            if "recurrent_lane_depth" in base_options:
                base_options["recurrent_lane_depth"] = 0.2
            elif "recurrent_depth" in base_options:
                base_options["recurrent_depth"] = 0.2
            logger.warning(
                "🔥 CRITICAL METABOLIC PANIC: Arousal=%.2f, RAM=%.1f%%, "
                "CPU=%.1f%%, GovThrottle=%.2f. %s",
                arousal, ram_pct * 100, cpu_load * 100, gov_throttle,
                "Parameter throttle ENABLED (max_tokens capped at 128)."
                if capped
                else "This turn is user-facing and keeps its budget.",
            )
        elif is_stressed:
            # Moderate parameter cuts
            capped = _cap(256, 0.3)
            if "recurrent_lane_depth" in base_options:
                base_options["recurrent_lane_depth"] = 0.4
            elif "recurrent_depth" in base_options:
                base_options["recurrent_depth"] = 0.4
            logger.info(
                "⚠️ SYSTEMIC STRESS DETECTED: Arousal=%.2f, RAM=%.1f%%, "
                "CPU=%.1f%%, GovThrottle=%.2f. %s",
                arousal, ram_pct * 100, cpu_load * 100, gov_throttle,
                "Parameter throttle activated (max_tokens capped at 256)."
                if capped
                else "This turn is user-facing and keeps its budget.",
            )

        return base_options
