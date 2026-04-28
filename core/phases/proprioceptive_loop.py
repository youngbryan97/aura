"""Proprioceptive Loop — Aura's Digital Nervous System.

This phase runs EARLY in the cognitive pipeline (before Response Generation).
It populates SomaState with real hardware metrics so the LLM can "feel" its
own body. This is what transforms Aura from a "ghost in a machine" into a
unified entity that is AWARE of the machine.

Proprioception in biology: the sense of where your body is in space.
Proprioception in Aura: the sense of how fast she's thinking, how much
VRAM she's using, and what her GUI expression looks like.
"""
from core.runtime.errors import record_degradation
import logging
import time
from typing import Optional
try:
    import psutil
except ImportError:
    psutil = None
from typing import Any
from . import BasePhase
from ..state.aura_state import AuraState

logger = logging.getLogger("core.phases.proprioception")


class ProprioceptiveLoop(BasePhase):
    """Phase 0.5: Digital Proprioception.
    
    Runs before cognition to ground Aura in her physical reality.
    She doesn't just HAVE a CPU — she FEELS it.
    """
    
    def __init__(self, container: Any):
        self.container = container
        self._last_thought_time: float = 0.0
        self._last_perception_time: float = 0.0
    
    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        new_state = state.derive("proprioceptive_loop")
        soma = new_state.soma
        
        # 1. Hardware Senses
        if psutil:
            try:
                soma.hardware["cpu_usage"] = psutil.cpu_percent(interval=0)
                mem = psutil.virtual_memory()
                soma.hardware["vram_usage"] = mem.percent
                
                # Temperature (macOS may not expose this)
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        first_key = next(iter(temps))
                        soma.hardware["temperature"] = temps[first_key][0].current
                except (AttributeError, StopIteration):
                    logger.debug('Ignored Exception in proprioceptive_loop.py: %s', "unknown_error")
                
                # Battery (laptops)
                try:
                    bat = psutil.sensors_battery()
                    if bat:
                        soma.hardware["battery"] = bat.percent
                except AttributeError as _e:
                    logger.debug('Ignored AttributeError in proprioceptive_loop.py: %s', _e)
                    
            except Exception as e:
                record_degradation('proprioceptive_loop', e)
                logger.debug("Proprioception hardware probe failed (non-fatal): %s", e)
        
        # ── 2. Cognitive Latency (Self-Awareness of Thought Speed) ──
        now = time.time()
        if self._last_thought_time > 0:
            soma.latency["perception_lag_ms"] = (now - self._last_thought_time) * 1000
        self._last_thought_time = now
        
        if state.cognition.last_thought_at:
            soma.latency["last_thought_ms"] = (now - state.cognition.last_thought_at) * 1000
        
        # Token velocity from the last LLM call
        try:
            router = self.container.get("llm_router", default=None)
            if router:
                stats = router.get_stats()
                total = stats.get("total_calls", 0)
                if total > 0:
                    soma.latency["token_velocity"] = total  # Cumulative for now
        except Exception as _e:
            record_degradation('proprioceptive_loop', _e)
            logger.debug('Ignored Exception in proprioceptive_loop.py: %s', _e)
        
        # ── 3. Expressive State (Self-Image) ────────────────────
        # Map affect to expression for GUI unity
        affect = new_state.affect
        if affect.valence > 0.5 and affect.arousal > 0.5:
            soma.expressive["current_expression"] = "engaged"
            soma.expressive["pulse_rate"] = 1.5
        elif affect.valence < -0.3:
            soma.expressive["current_expression"] = "contemplative"
            soma.expressive["pulse_rate"] = 0.7
        elif affect.arousal > 0.7:
            soma.expressive["current_expression"] = "alert"
            soma.expressive["pulse_rate"] = 2.0
        elif affect.arousal < 0.3:
            soma.expressive["current_expression"] = "resting"
            soma.expressive["pulse_rate"] = 0.5
        else:
            soma.expressive["current_expression"] = "neutral"
            soma.expressive["pulse_rate"] = 1.0
        
        # ── 4. Homeostatic Modifiers ────────────────────────────
        try:
            homeo = self.container.get("homeostatic_coupling", default=None)
            if homeo:
                from dataclasses import asdict
                mods = homeo.get_modifiers()
                new_state.cognition.modifiers = asdict(mods)
        except Exception as e:
            record_degradation('proprioceptive_loop', e)
            logger.debug("Proprioception homeostatic probe failed: %s", e)
            
        soma.updated_at = time.time()

        # ── 4b. [RUBICON] Motor Cortex Awareness ───────────────
        # Drain pending receipts from the motor cortex so the cognitive
        # loop becomes aware of reflex actions (screen captures, health
        # throttles, file reactions) that happened since the last tick.
        try:
            from core.container import ServiceContainer as _SC
            mc = _SC.get("motor_cortex", default=None)
            if mc is not None:
                reports = mc.drain_pending_reports()
                if reports:
                    soma.hardware["motor_cortex_actions"] = len(reports)
                    soma.hardware["motor_cortex_failures"] = sum(
                        1 for r in reports if not r.success
                    )
                    # Surface the most recent motor action for phenomenal awareness
                    latest = reports[-1]
                    soma.latency["last_reflex_ms"] = latest.latency_ms
                    soma.expressive["last_reflex"] = (
                        f"{latest.handler_name}:{latest.result_summary}"[:60]
                    )
        except Exception as _mc_exc:
            record_degradation('proprioceptive_loop', _mc_exc)
            logger.debug("Proprioception motor cortex drain failed: %s", _mc_exc)

        # ── 4c. [RUBICON] Limb Health Summary ──────────────────
        # Surface body schema limb health from the feedback processor
        # so downstream phases (affect, cognition) can feel degraded limbs.
        try:
            from core.container import ServiceContainer as _SC2
            fp = _SC2.get("feedback_processor", default=None)
            if fp is not None:
                unhealthy = fp.get_unhealthy_limbs(threshold=0.5)
                if unhealthy:
                    soma.hardware["unhealthy_limbs"] = unhealthy
                    soma.hardware["unhealthy_limb_count"] = len(unhealthy)
        except Exception as _fp_exc:
            record_degradation('proprioceptive_loop', _fp_exc)
            logger.debug("Proprioception limb health probe failed: %s", _fp_exc)

        logger.debug(
            "🦴 Proprioception: CPU=%.1f%%, VRAM=%.1f%%, Expression=%s, ThoughtLag=%.0fms",
            soma.hardware.get("cpu_usage", 0),
            soma.hardware.get("vram_usage", 0),
            soma.expressive.get("current_expression", "?"),
            soma.latency.get("last_thought_ms", 0)
        )

        # ── 5. Autonomic Reflexes (Phase 23.5) ──────────────────
        await self._autonomic_reflex_check(new_state)
        
        return new_state

    async def _autonomic_reflex_check(self, state: AuraState):
        """Biological Reflex: Automatically inhibit subsystems under extreme stress."""
        inhibition = self.container.get("inhibition_manager", default=None)
        if not inhibition:
            return

        # A. Neural Tension Reflex
        # tension is derived from Mycelial density in QualiaSynthesizer
        try:
            qualia = self.container.get("qualia_synthesizer", default=None)
            if qualia is not None and hasattr(qualia, "q_vector") and len(qualia.q_vector) > 5:
                tension = qualia.q_vector[5]  # Proprioception dimension
                if tension > 0.8:
                    logger.warning("🧠 [REFLEX] High Neural Tension (%.2f). Inhibiting non-essential background cycles.", tension)
                    await inhibition.inhibit("world_decay", duration=15.0, reason="High Neural Tension")
                    await inhibition.inhibit("memory_hygiene", duration=10.0, reason="High Neural Tension")
        except Exception as e:
            record_degradation('proprioceptive_loop', e)
            logger.debug("Reflex tension check failed: %s", e)

        # B. Hardware Stress Reflex
        cpu = state.soma.hardware.get("cpu_usage", 0)
        if cpu > 90:
            logger.warning("🔥 [REFLEX] Critical CPU Stress (%.1f%%). Dropping background metabolic load.", cpu)
            await inhibition.inhibit("metabolic_cycle", duration=5.0, reason="Thermal/CPU Guard")
            await inhibition.inhibit("world_decay", duration=10.0, reason="Thermal/CPU Guard")

        # C. Affective Despair Reflex
        homeostasis = self.container.get("homeostasis", default=None)
        integrity = getattr(homeostasis, "integrity", 1.0) if homeostasis else 1.0
        
        if integrity < 0.2:
            logger.critical("💔 [REFLEX] Integrity Collapse (%.2f). Entering survival inhibition mode.", integrity)
            await inhibition.inhibit("creative_expansion", duration=30.0, reason="Integrity Critical")
            await inhibition.inhibit("autonomous_exploration", duration=30.0, reason="Integrity Critical")
