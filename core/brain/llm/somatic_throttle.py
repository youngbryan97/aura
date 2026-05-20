"""core/brain/llm/somatic_throttle.py — Somatic Compute Sentinel

Enforces metabolic constraints directly inside the LLM token sampling loop.
Connects with DamasioV2 virtual physiology and hardware telemetry (RAM/CPU loads)
to dynamically scale down LLM generation parameters under heavy load to prevent
thermal throttling or out-of-memory (OOM) crashes.
"""
import logging
import psutil
from typing import Dict, Any
from core.runtime.service_access import resolve_affect_engine

logger = logging.getLogger("Aura.Brain.SomaticThrottle")


class SomaticComputeSentinel:
    """Natively enforces metabolic constraints directly inside the LLM token generation cycle."""

    def __init__(self):
        logger.info("🌡️ SomaticComputeSentinel initialized.")

    def adjust_generation_options(self, base_options: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamically adjusts LLM sampling and length parameters based on metabolic and hardware stress."""
        # 1. Fetch virtual physiological stress (arousal)
        arousal = 0.0
        try:
            affect = resolve_affect_engine(default=None)
            if affect and hasattr(affect, "current"):
                current_state = affect.current
                if hasattr(current_state, "arousal"):
                    arousal = float(current_state.arousal)
        except Exception as e:
            logger.debug("Failed to resolve affect engine arousal: %s", e)

        # 1b. Fetch governance token throttle factor
        gov_throttle = 1.0
        try:
            from research.protocols.resource_quotas import get_compute_governor
            gov = get_compute_governor()
            gov_throttle = gov.get_throttle_factor()
        except Exception as e:
            logger.debug("Failed to resolve compute governor: %s", e)

        # 2. Fetch hardware stress metrics
        cpu_load = 0.0
        ram_pct = 0.0
        try:
            cpu_load = psutil.cpu_percent(interval=0) / 100.0
            ram_pct = psutil.virtual_memory().percent / 100.0
        except Exception as e:
            logger.debug("Failed to retrieve hardware metrics: %s", e)

        # 3. Determine if systemic overload is present
        # Elevated arousal (> 0.8) or critical RAM/CPU pressure or governance token exhaustion
        is_stressed = (arousal > 0.8) or (ram_pct > 0.88) or (cpu_load > 0.9) or (gov_throttle <= 0.5)
        is_critical = (arousal > 0.9) or (ram_pct > 0.93) or (gov_throttle <= 0.2)

        if gov_throttle == 0.0:
            # Token exhaustion: severe cap to block further consumption
            original_max = base_options.get("max_tokens", 512)
            base_options["max_tokens"] = min(original_max, 8)
            base_options["temperature"] = 0.05
            logger.error("🚫 GOVERNANCE QUOTA EXHAUSTED: Token limit hit. Sampling capped to 8 tokens.")
        elif is_critical:
            # Force severe parameter cuts to prevent OOM/Thermal crash
            original_max = base_options.get("max_tokens", 512)
            base_options["max_tokens"] = min(original_max, 128)
            base_options["temperature"] = 0.15
            # Throttle recurrent lane depth if supported by token generator
            if "recurrent_lane_depth" in base_options:
                base_options["recurrent_lane_depth"] = 0.2
            elif "recurrent_depth" in base_options:
                base_options["recurrent_depth"] = 0.2
            logger.warning(
                "🔥 CRITICAL METABOLIC PANIC: Arousal=%.2f, RAM=%.1f%%, CPU=%.1f%%, GovThrottle=%.2f. Parameter throttle ENABLED (max_tokens capped at 128).",
                arousal, ram_pct * 100, cpu_load * 100, gov_throttle
            )
        elif is_stressed:
            # Moderate parameter cuts
            original_max = base_options.get("max_tokens", 512)
            base_options["max_tokens"] = min(original_max, 256)
            base_options["temperature"] = 0.3
            if "recurrent_lane_depth" in base_options:
                base_options["recurrent_lane_depth"] = 0.4
            elif "recurrent_depth" in base_options:
                base_options["recurrent_depth"] = 0.4
            logger.info(
                "⚠️ SYSTEMIC STRESS DETECTED: Arousal=%.2f, RAM=%.1f%%, CPU=%.1f%%, GovThrottle=%.2f. Parameter throttle activated (max_tokens capped at 256).",
                arousal, ram_pct * 100, cpu_load * 100, gov_throttle
            )

        return base_options
