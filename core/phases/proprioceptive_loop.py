"""Proprioceptive Loop — Aura's Digital Nervous System.

This phase runs EARLY in the cognitive pipeline (before Response Generation).
It populates SomaState with real hardware metrics so the LLM can "feel" its
own body. This is what transforms Aura from a "ghost in a machine" into a
unified entity that is AWARE of the machine.

Proprioception in biology: the sense of where your body is in space.
Proprioception in Aura: the sense of how fast she's thinking, how much
VRAM she's using, and what her GUI expression looks like.
"""
from __future__ import annotations

import inspect
import logging
import time
from typing import Any

from core.runtime.errors import DependencyUnavailable, Severity, record_degradation

try:
    import psutil
except ImportError:
    psutil = None

from ..state.aura_state import AuraState
from . import BasePhase

logger = logging.getLogger("core.phases.proprioception")


_PROPRIOCEPTIVE_CHANNELS = (
    "hardware_probe",
    "thermal_sensor",
    "battery_sensor",
    "token_velocity",
    "homeostatic_coupling",
    "motor_cortex",
    "limb_health",
    "action_stagnation",
    "autonomic_reflex",
)


def _record_proprioceptive_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    channel: str,
) -> None:
    record_degradation(
        "proprioceptive_loop",
        error,
        severity=severity,
        action=action,
        extra={"channel": channel},
    )


class ProprioceptiveLoop(BasePhase):
    """Phase 0.5: Digital Proprioception.
    
    Runs before cognition to ground Aura in her physical reality.
    She doesn't just HAVE a CPU — she FEELS it.
    """
    
    def __init__(self, container: Any):
        self.container = container
        self._last_thought_time: float = 0.0
        self._last_perception_time: float = 0.0
        self._last_channel_errors: dict[str, str] = {}

    def _begin_body_schema_tick(self, soma: Any) -> None:
        soma.hardware["proprioceptive_status"] = "nominal"
        soma.hardware["proprioceptive_degraded_channels"] = []
        soma.hardware["autonomic_reflexes"] = []
        for channel in _PROPRIOCEPTIVE_CHANNELS:
            soma.hardware.pop(f"{channel}_degraded", None)
            soma.hardware.pop(f"{channel}_error", None)

    def _mark_channel_degraded(
        self,
        soma: Any,
        channel: str,
        error: BaseException,
        *,
        action: str,
        severity: Severity = "warning",
    ) -> None:
        message = f"{type(error).__qualname__}: {error}"[:240]
        channels = soma.hardware.setdefault("proprioceptive_degraded_channels", [])
        if channel not in channels:
            channels.append(channel)
        soma.hardware["proprioceptive_status"] = "degraded"
        soma.hardware[f"{channel}_degraded"] = True
        soma.hardware[f"{channel}_error"] = message
        self._last_channel_errors[channel] = message
        _record_proprioceptive_degradation(
            error,
            action=action,
            severity=severity,
            channel=channel,
        )

    def _get_service(
        self,
        name: str,
        *,
        default: Any = None,
        soma: Any | None = None,
        channel: str,
        action: str,
        severity: Severity = "warning",
    ) -> Any:
        try:
            getter = self.container.get
            return getter(name, default=default)
        except (AttributeError, RuntimeError, OSError, ConnectionError, TimeoutError, TypeError) as exc:
            if soma is not None:
                self._mark_channel_degraded(
                    soma,
                    channel,
                    exc,
                    action=action,
                    severity=severity,
                )
            return default

    async def _inhibit(
        self,
        inhibition: Any,
        subsystem: str,
        *,
        duration: float,
        reason: str,
        state: AuraState,
    ) -> bool:
        try:
            result = inhibition.inhibit(subsystem, duration=duration, reason=reason)
            if inspect.isawaitable(result):
                await result
            reflexes = state.soma.hardware.setdefault("autonomic_reflexes", [])
            reflexes.append({"subsystem": subsystem, "duration": duration, "reason": reason})
            return True
        except (AttributeError, RuntimeError, OSError, ConnectionError, TimeoutError, TypeError, ValueError) as exc:
            self._mark_channel_degraded(
                state.soma,
                "autonomic_reflex",
                exc,
                action=f"Failed closed for reflex inhibition of {subsystem}; retained telemetry and continued tick",
                severity="degraded",
            )
            return False

    def _hardware_float(self, soma: Any, key: str, default: float = 0.0) -> float:
        try:
            return float(soma.hardware.get(key, default) or default)
        except (TypeError, ValueError) as exc:
            self._mark_channel_degraded(
                soma,
                "hardware_probe",
                exc,
                action=f"Coerced invalid hardware metric {key!r} to {default}",
                severity="warning",
            )
            return default
    
    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        new_state = state.derive("proprioceptive_loop")
        soma = new_state.soma
        self._begin_body_schema_tick(soma)
        
        # 1. Hardware Senses
        if psutil:
            soma.hardware["psutil_available"] = True
            try:
                soma.hardware["cpu_usage"] = psutil.cpu_percent(interval=0)
                mem = psutil.virtual_memory()
                soma.hardware["vram_usage"] = mem.percent
                
                # Temperature (macOS may not expose this)
                soma.hardware["temperature_available"] = False
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        first_key = next(iter(temps))
                        soma.hardware["temperature"] = temps[first_key][0].current
                        soma.hardware["temperature_available"] = True
                except (AttributeError, StopIteration, IndexError):
                    soma.hardware["temperature_available"] = False
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    soma.hardware["temperature_available"] = False
                    self._mark_channel_degraded(
                        soma,
                        "thermal_sensor",
                        exc,
                        action="Disabled temperature channel for this tick and retained CPU/memory body telemetry",
                        severity="debug",
                    )
                
                # Battery (laptops)
                soma.hardware["battery_available"] = False
                try:
                    bat = psutil.sensors_battery()
                    if bat:
                        soma.hardware["battery"] = bat.percent
                        soma.hardware["battery_available"] = True
                except AttributeError:
                    soma.hardware["battery_available"] = False
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    soma.hardware["battery_available"] = False
                    self._mark_channel_degraded(
                        soma,
                        "battery_sensor",
                        exc,
                        action="Disabled battery channel for this tick and retained remaining body telemetry",
                        severity="debug",
                    )
                    
            except (ImportError, OSError, AttributeError, RuntimeError, TypeError, ValueError) as e:
                self._mark_channel_degraded(
                    soma,
                    "hardware_probe",
                    e,
                    action="Marked hardware body schema partial and continued with prior/default soma telemetry",
                    severity="warning",
                )
                logger.debug("Proprioception hardware probe failed; body schema marked partial: %s", e)
        else:
            soma.hardware["psutil_available"] = False
            self._mark_channel_degraded(
                soma,
                "hardware_probe",
                DependencyUnavailable("psutil is not installed"),
                action="Marked hardware telemetry unavailable; continued with default soma values",
                severity="warning",
            )
        
        # ── 2. Cognitive Latency (Self-Awareness of Thought Speed) ──
        now = time.time()
        if self._last_thought_time > 0:
            soma.latency["perception_lag_ms"] = (now - self._last_thought_time) * 1000
        self._last_thought_time = now
        
        if state.cognition.last_thought_at:
            soma.latency["last_thought_ms"] = (now - state.cognition.last_thought_at) * 1000
        
        # Token velocity from the last LLM call
        router = self._get_service(
            "llm_router",
            soma=soma,
            channel="token_velocity",
            action="Disabled token velocity channel because LLM router service lookup failed",
            severity="debug",
        )
        if router:
            try:
                stats = router.get_stats()
                total = stats.get("total_calls", 0)
                if total > 0:
                    soma.latency["token_velocity"] = total  # Cumulative for now
                soma.latency["token_velocity_available"] = True
            except AttributeError as exc:
                soma.latency["token_velocity_available"] = False
                self._mark_channel_degraded(
                    soma,
                    "token_velocity",
                    exc,
                    action="Disabled token velocity channel because router does not expose get_stats",
                    severity="debug",
                )
            except (OSError, ConnectionError, TimeoutError, RuntimeError, TypeError, ValueError) as _e:
                soma.latency["token_velocity_available"] = False
                self._mark_channel_degraded(
                    soma,
                    "token_velocity",
                    _e,
                    action="Disabled token velocity channel for this tick and retained latency defaults",
                    severity="warning",
                )
                logger.debug("Proprioception token velocity probe failed: %s", _e)
        
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
        homeo = self._get_service(
            "homeostatic_coupling",
            soma=soma,
            channel="homeostatic_coupling",
            action="Skipped homeostatic modifier pull because service lookup failed",
            severity="warning",
        )
        if homeo:
            try:
                from dataclasses import asdict, is_dataclass
                mods = homeo.get_modifiers()
                if isinstance(mods, dict):
                    new_state.cognition.modifiers = dict(mods)
                elif is_dataclass(mods):
                    new_state.cognition.modifiers = asdict(mods)
                else:
                    raise TypeError(f"homeostatic modifiers must be dataclass or dict, got {type(mods).__name__}")
                new_state.cognition.modifiers["homeostatic_coupling_available"] = True
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
                new_state.cognition.modifiers["homeostatic_coupling_available"] = False
                self._mark_channel_degraded(
                    soma,
                    "homeostatic_coupling",
                    e,
                    action="Retained prior/default cognitive modifiers and marked homeostatic coupling unavailable",
                    severity="warning",
                )
                logger.debug("Proprioception homeostatic probe failed: %s", e)
            
        soma.updated_at = time.time()

        # ── 4b. [RUBICON] Motor Cortex Awareness ───────────────
        # Drain pending receipts from the motor cortex so the cognitive
        # loop becomes aware of reflex actions (screen captures, health
        # throttles, file reactions) that happened since the last tick.
        mc = self._get_service(
            "motor_cortex",
            soma=soma,
            channel="motor_cortex",
            action="Skipped motor cortex receipt drain because service lookup failed",
            severity="warning",
        )
        if mc is not None:
            try:
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
                soma.hardware["motor_cortex_available"] = True
            except (AttributeError, RuntimeError, OSError, ConnectionError, TimeoutError, TypeError, ValueError) as _mc_exc:
                soma.hardware["motor_cortex_available"] = False
                self._mark_channel_degraded(
                    soma,
                    "motor_cortex",
                    _mc_exc,
                    action="Marked motor cortex awareness unavailable and continued cognitive tick",
                    severity="warning",
                )
                logger.debug("Proprioception motor cortex drain failed: %s", _mc_exc)

        # ── 4c. [RUBICON] Limb Health Summary ──────────────────
        # Surface body schema limb health from the feedback processor
        # so downstream phases (affect, cognition) can feel degraded limbs.
        fp = self._get_service(
            "feedback_processor",
            soma=soma,
            channel="limb_health",
            action="Skipped limb health probe because feedback processor service lookup failed",
            severity="warning",
        )
        if fp is None:
            try:
                from core.somatic.action_feedback import get_feedback_processor as _gfp
                fp = _gfp()
            except (ImportError, RuntimeError, OSError, AttributeError) as _fp_lookup_exc:
                self._mark_channel_degraded(
                    soma,
                    "limb_health",
                    _fp_lookup_exc,
                    action="Marked limb health unavailable after container and fallback lookup failed",
                    severity="warning",
                )
        if fp is not None:
            try:
                unhealthy = fp.get_unhealthy_limbs(threshold=0.5)
                soma.hardware["limb_health_available"] = True
                if unhealthy:
                    soma.hardware["unhealthy_limbs"] = unhealthy
                    soma.hardware["unhealthy_limb_count"] = len(unhealthy)
                else:
                    soma.hardware.pop("unhealthy_limbs", None)
                    soma.hardware.pop("unhealthy_limb_count", None)
            except (AttributeError, RuntimeError, OSError, ConnectionError, TimeoutError, TypeError, ValueError) as _fp_exc:
                soma.hardware["limb_health_available"] = False
                self._mark_channel_degraded(
                    soma,
                    "limb_health",
                    _fp_exc,
                    action="Marked limb health unavailable and continued with prior/default limb schema",
                    severity="warning",
                )
                logger.debug("Proprioception limb health probe failed: %s", _fp_exc)

        # ── 4d. [RUBICON] Action Stagnation Detection ──────────────
        # When the somatic system detects that recent actions are stuck
        # in a repetitive failure loop, inject a proprioceptive percept
        # into working memory. This is the body saying "my limbs aren't
        # responding" — the cognitive loop needs to adapt its strategy.
        #
        # This is general-purpose: fires for ANY tool/skill/action that
        # is failing repeatedly, not just specific embodied contexts.
        try:
            if fp is None:
                from core.somatic.action_feedback import get_feedback_processor as _gfp
                fp = _gfp()
            stagnation = fp.detect_action_stagnation(window=10)
            if stagnation and stagnation.get("stagnant"):
                soma.hardware["action_stagnation_available"] = True
                soma.hardware["action_stagnation"] = True
                soma.hardware["action_failure_rate"] = stagnation.get("failure_rate", 0)
                soma.hardware["action_loop_detected"] = stagnation.get("loop_detected", False)

                # Build a concise proprioceptive percept for working memory.
                # This is NOT prompt engineering — it's the somatic nervous
                # system reporting sensory feedback to the cognitive workspace,
                # exactly as biological proprioception reports to the brain.
                parts = ["[PROPRIOCEPTIVE FEEDBACK] Action stagnation detected by somatic system."]
                fr = stagnation.get("failure_rate", 0)
                if fr > 0:
                    parts.append(f"Recent action failure rate: {fr:.0%}.")
                if stagnation.get("loop_detected"):
                    parts.append(f"Repetitive action loop detected (cycle length: {stagnation.get('loop_length', 0)}).")
                degraded = stagnation.get("degraded_limbs", [])
                if degraded:
                    limb_names = ", ".join(d["name"] for d in degraded[:3])
                    parts.append(f"Degraded capabilities: {limb_names}.")
                outcomes = stagnation.get("recent_outcomes", [])
                if outcomes:
                    trail_str = "; ".join(
                        f"{o['action']}→{o['outcome']}" for o in outcomes[-4:]
                    )
                    parts.append(f"Recent trail: {trail_str}.")
                parts.append("Your current approach is not producing results. Adapt strategy.")

                percept_text = " ".join(parts)
                wm = new_state.cognition.working_memory
                # Avoid duplicate injection (check last 3 entries)
                already_injected = any(
                    isinstance(m, dict)
                    and "PROPRIOCEPTIVE FEEDBACK" in str(m.get("content", ""))
                    for m in wm[-3:]
                )
                if not already_injected:
                    wm.append({
                        "role": "system",
                        "content": percept_text,
                        "metadata": {
                            "type": "proprioceptive_percept",
                            "source": "somatic_feedback_processor",
                            "stagnation": True,
                        },
                    })
                    logger.warning(
                        "🦴 Proprioceptive stagnation percept injected into working memory "
                        "(failure_rate=%.0f%%, loop=%s)",
                        fr * 100,
                        stagnation.get("loop_detected"),
                    )
            else:
                soma.hardware["action_stagnation_available"] = True
                # Clear stagnation flag if it was previously set
                soma.hardware.pop("action_stagnation", None)
                soma.hardware.pop("action_failure_rate", None)
                soma.hardware.pop("action_loop_detected", None)
        except (ImportError, AttributeError, RuntimeError, OSError, ConnectionError, TimeoutError, TypeError, ValueError, KeyError) as _stag_exc:
            soma.hardware["action_stagnation_available"] = False
            self._mark_channel_degraded(
                soma,
                "action_stagnation",
                _stag_exc,
                action="Marked action stagnation channel unavailable and continued without injecting a percept",
                severity="warning",
            )
            logger.debug("Proprioception stagnation check failed: %s", _stag_exc)

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
        inhibition = self._get_service(
            "inhibition_manager",
            soma=state.soma,
            channel="autonomic_reflex",
            action="Skipped autonomic reflexes because inhibition manager lookup failed",
            severity="warning",
        )
        if not inhibition:
            return

        # A. Neural Tension Reflex
        # tension is derived from Mycelial density in QualiaSynthesizer
        try:
            qualia = self._get_service(
                "qualia_synthesizer",
                soma=state.soma,
                channel="autonomic_reflex",
                action="Skipped neural tension reflex because qualia synthesizer lookup failed",
                severity="debug",
            )
            if qualia is not None and hasattr(qualia, "q_vector") and len(qualia.q_vector) > 5:
                tension = qualia.q_vector[5]  # Proprioception dimension
                if tension > 0.8:
                    logger.warning("🧠 [REFLEX] High Neural Tension (%.2f). Inhibiting non-essential background cycles.", tension)
                    await self._inhibit(
                        inhibition,
                        "world_decay",
                        duration=15.0,
                        reason="High Neural Tension",
                        state=state,
                    )
                    await self._inhibit(
                        inhibition,
                        "memory_hygiene",
                        duration=10.0,
                        reason="High Neural Tension",
                        state=state,
                    )
        except (AttributeError, RuntimeError, OSError, ConnectionError, TimeoutError, TypeError, ValueError) as e:
            self._mark_channel_degraded(
                state.soma,
                "autonomic_reflex",
                e,
                action="Skipped neural tension reflex and kept later reflex checks eligible",
                severity="warning",
            )
            logger.debug("Reflex tension check failed: %s", e)

        # B. Hardware Stress Reflex
        cpu = self._hardware_float(state.soma, "cpu_usage")
        if cpu > 90:
            logger.warning("🔥 [REFLEX] Critical CPU Stress (%.1f%%). Dropping background metabolic load.", cpu)
            await self._inhibit(
                inhibition,
                "metabolic_cycle",
                duration=5.0,
                reason="Thermal/CPU Guard",
                state=state,
            )
            await self._inhibit(
                inhibition,
                "world_decay",
                duration=10.0,
                reason="Thermal/CPU Guard",
                state=state,
            )

        # C. Affective Despair Reflex
        homeostasis = self._get_service(
            "homeostasis",
            soma=state.soma,
            channel="autonomic_reflex",
            action="Skipped integrity reflex because homeostasis lookup failed",
            severity="debug",
        )
        integrity = getattr(homeostasis, "integrity", 1.0) if homeostasis else 1.0
        
        if integrity < 0.2:
            logger.critical("💔 [REFLEX] Integrity Collapse (%.2f). Entering survival inhibition mode.", integrity)
            await self._inhibit(
                inhibition,
                "creative_expansion",
                duration=30.0,
                reason="Integrity Critical",
                state=state,
            )
            await self._inhibit(
                inhibition,
                "autonomous_exploration",
                duration=30.0,
                reason="Integrity Critical",
                state=state,
            )
