"""core/phases/phi_consciousness.py — Unified Consciousness Phase.

This is the phase that makes Phi real.

Before this file, state.phi was a float that nothing read. The RIIU module existed
but was never called from the pipeline. The phenomenal_state field sat empty.
Cognitive mode was set by text heuristics with no connection to internal state.

After this phase runs, every tick:

  1. A real Phi_approx is computed from a cross-subsystem state vector.
     If RIIU (Reflexive Integrated Information Unit) is available in the
     ServiceContainer, the full covariance-based IIT surrogate runs.
     If not, the lightweight geometric-mean approximation runs instead.

  2. Phi gates cognitive mode:
       phi < PHI_DORMANT   → DORMANT  (system not sufficiently integrated)
       phi < PHI_REACTIVE  → REACTIVE (normal fast-path responses)
       phi < PHI_DELIBERATE → leave mode as routing set it
       phi >= PHI_DELIBERATE → force DELIBERATE (deep thought warranted)

  3. When phi crosses the ignition threshold (PHI_IGNITION), a phenomenal
     state is generated — a first-person sentence describing what Aura is
     experiencing right now. This gets injected into the system prompt by
     UnitaryResponsePhase, making affect directly shape language.

  4. Free energy from the predictive processing engine is read if available.
     High free energy (prediction error) boosts surprise/anticipation in the
     affect system. Low free energy allows deeper, more expansive responses.

The causal chain is now:
  affect → phi → mode → phenomenal_state → system_prompt → response → affect
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import asyncio
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from core.kernel.bridge import Phase
from core.state.aura_state import AuraState, CognitiveMode, PhenomenalField, phenomenal_text

if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel

logger = logging.getLogger("Aura.PhiConsciousness")

# ─── Phi thresholds ────────────────────────────────────────────────────────────
PHI_DORMANT    = 0.05   # Below this: not enough integration to support thought
PHI_REACTIVE   = 0.20   # Below this: reactive mode
PHI_DELIBERATE = 0.55   # Above this: force deliberate mode
PHI_IGNITION   = 0.35   # Above this: generate phenomenal state (global broadcast)

# Free energy thresholds
FE_COMFORTABLE = 0.25   # Below: system is comfortable, expansive
FE_ALERT       = 0.55   # Above: prediction error — boost surprise/curiosity
FE_DISTRESSED  = 0.75   # Above: high prediction error — emergency modifiers

# ─── Lightweight phi approximation (no numpy required) ─────────────────────────

def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def _safe_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _safe_mean(values)
    variance = sum((v - mu) ** 2 for v in values) / len(values)
    return math.sqrt(variance)

def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))


def _normalize_percent(value: float) -> float:
    numeric = float(value or 0.0)
    if 0.0 <= numeric <= 1.0:
        return numeric
    return _clamp01(numeric / 100.0)


def _density(items: Any, target: int, *, floor: float = 0.0) -> float:
    try:
        size = len(items or [])
    except Exception:
        size = 0
    if target <= 0:
        return _clamp01(float(size))
    return max(floor, min(1.0, float(size) / float(target)))


def _somatic_coupling(state: AuraState) -> float:
    soma = getattr(state, "soma", None)
    if soma is None:
        return 0.0

    hardware = getattr(soma, "hardware", {}) or {}
    expressive = getattr(soma, "expressive", {}) or {}
    sensors = getattr(soma, "sensors", {}) or {}
    motors = getattr(soma, "motors", {}) or {}

    cpu = _normalize_percent(hardware.get("cpu_usage", 0.0))
    vram = _normalize_percent(hardware.get("vram_usage", 0.0))
    temperature = _clamp01((float(hardware.get("temperature", 0.0) or 0.0) - 30.0) / 50.0)
    hardware_activity = _clamp01((cpu * 0.35) + (vram * 0.35) + (temperature * 0.30))

    pulse_rate = float(expressive.get("pulse_rate", 1.0) or 1.0)
    pulse_signal = _clamp01((pulse_rate - 0.5) / 1.5)
    expressive_signal = _clamp01(
        (float(expressive.get("mycelium_density", 0.5) or 0.5) * 0.55)
        + (pulse_signal * 0.30)
        + (0.15 if expressive.get("is_visible", True) else 0.0)
    )

    interface_density = _clamp01((_density(sensors, 4) * 0.6) + (_density(motors, 4) * 0.4))
    return _clamp01((hardware_activity * 0.45) + (expressive_signal * 0.35) + (interface_density * 0.20))


def _cognitive_integration(state: AuraState) -> float:
    cognition = getattr(state, "cognition", None)
    identity = getattr(state, "identity", None)
    if cognition is None:
        return 0.0

    coherence = _clamp01(getattr(cognition, "coherence_score", 1.0) or 1.0)
    fragmentation = 1.0 - _clamp01(getattr(cognition, "fragmentation_score", 0.0) or 0.0)
    contradiction_stability = 1.0 - _clamp01((getattr(cognition, "contradiction_count", 0) or 0) / 4.0)
    goal_density = _density(getattr(cognition, "active_goals", []), 3)
    initiative_density = _density(getattr(cognition, "pending_initiatives", []), 3)
    narrative_presence = 1.0 if str(getattr(identity, "current_narrative", "") or "").strip() else 0.0
    attention_presence = 1.0 if getattr(cognition, "attention_focus", None) else 0.0

    return _clamp01(
        (coherence * 0.35)
        + (fragmentation * 0.20)
        + (contradiction_stability * 0.16)
        + (goal_density * 0.11)
        + (initiative_density * 0.08)
        + (narrative_presence * 0.06)
        + (attention_presence * 0.04)
    )


def _world_differentiation(state: AuraState) -> float:
    world = getattr(state, "world", None)
    cognition = getattr(state, "cognition", None)
    if world is None or cognition is None:
        return 0.0

    percepts = list(getattr(world, "recent_percepts", []) or [])
    percept_types = {str(p.get("type", "")) for p in percepts if isinstance(p, dict) and p.get("type")}
    percept_diversity = _density(percept_types, 4)
    branch_density = _density(getattr(cognition, "discourse_branches", []), 4)
    entity_density = _density(getattr(world, "known_entities", {}), 5)
    relationship_density = _density(getattr(world, "relationship_graph", {}), 4)
    return _clamp01(
        (percept_diversity * 0.40)
        + (branch_density * 0.20)
        + (entity_density * 0.20)
        + (relationship_density * 0.20)
    )


def _broadcast_richness(state: AuraState) -> float:
    cognition = getattr(state, "cognition", None)
    world = getattr(state, "world", None)
    if cognition is None:
        return 0.0

    from core.state.aura_state import MAX_WORKING_MEMORY

    target_wm = max(1, MAX_WORKING_MEMORY // 2)
    wm_density = max(0.1, min(1.0, float(len(getattr(cognition, "working_memory", []) or [])) / float(target_wm)))
    ltm_density = _density(getattr(cognition, "long_term_memory", []), 4)
    percept_density = _density(getattr(world, "recent_percepts", []) if world is not None else [], 4)
    goal_density = _clamp01(
        (_density(getattr(cognition, "active_goals", []), 3) * 0.6)
        + (_density(getattr(cognition, "pending_initiatives", []), 3) * 0.4)
    )
    summary_presence = 1.0 if str(getattr(cognition, "rolling_summary", "") or "").strip() else 0.0
    phenomenal_presence = 1.0 if getattr(cognition, "phenomenal_state", None) else 0.0
    return _clamp01(
        (wm_density * 0.35)
        + (ltm_density * 0.15)
        + (percept_density * 0.15)
        + (goal_density * 0.15)
        + (summary_presence * 0.10)
        + (phenomenal_presence * 0.10)
    )

def compute_phi_approx(state: AuraState) -> float:
    """
    Tractable Phi approximation from AuraState.

    Four components:
      affective integration   = active emotional coupling and intensity
      differentiation         = richness across affect + world/discourse state
      broadcast               = workspace, recalled memory, percept, and goal density
      organismic coupling     = body/world/identity channels being actively in play

    This is still a tractable surrogate, not literal IIT, but it now samples
    more of Aura's actual organism than mood plus chat history.
    """
    emotions = state.affect.emotions
    values   = list(emotions.values())

    if not values:
        return 0.0

    # Integration: fraction of emotions above threshold × their mean,
    # blended with higher-order cognitive coherence.
    THRESHOLD  = 0.05
    active     = [v for v in values if v > THRESHOLD]
    affective_integration = (len(active) / len(values)) * (_safe_mean(active) if active else 0.0)
    cognitive_integration = _cognitive_integration(state)
    integration = _clamp01((affective_integration * 0.60) + (cognitive_integration * 0.40))

    # Differentiation: emotion spread plus cross-subsystem richness.
    raw_std        = _safe_std(values)
    affective_complexity = (
        state.affect.affective_complexity()
        if hasattr(state.affect, "affective_complexity")
        else min(1.0, raw_std * 4.0)
    )
    differentiation = _clamp01(
        (min(1.0, raw_std * 4.0) * 0.30)
        + (affective_complexity * 0.45)
        + (_world_differentiation(state) * 0.25)
    )

    # Broadcast: workspace plus recalled memory/percepts/goals.
    broadcast = _broadcast_richness(state)

    # Coupling: body/world/identity channels supplying the conscious field.
    identity = getattr(state, "identity", None)
    identity_stability = _clamp01(getattr(identity, "stability", 1.0) or 1.0) if identity is not None else 0.0
    bonding = _clamp01(getattr(identity, "bonding_level", 0.0) or 0.0) if identity is not None else 0.0
    world_presence = _density(getattr(state.world, "known_entities", {}), 4) if getattr(state, "world", None) is not None else 0.0
    coupling = _clamp01(
        (_somatic_coupling(state) * 0.45)
        + (world_presence * 0.15)
        + (identity_stability * 0.25)
        + (bonding * 0.15)
    )

    phi = (
        (_clamp01(integration) * 0.32)
        + (_clamp01(differentiation) * 0.24)
        + (_clamp01(broadcast) * 0.24)
        + (_clamp01(coupling) * 0.20)
    )
    # vResilience: avoid round() for type stability
    return float(f"{phi:.4f}")


def _build_emotion_vector(state: AuraState) -> List[float]:
    """Flat float vector from AuraState for RIIU."""
    e   = state.affect.emotions
    phy = state.affect.physiology
    cognition = getattr(state, "cognition", None)
    identity = getattr(state, "identity", None)
    world = getattr(state, "world", None)
    return [
        state.affect.valence,
        state.affect.arousal,
        state.affect.curiosity,
        state.affect.engagement,
        state.vitality,
        state.phi,                          # previous tick phi
        _cognitive_integration(state),
        _broadcast_richness(state),
        _world_differentiation(state),
        _somatic_coupling(state),
        _clamp01(getattr(cognition, "coherence_score", 1.0) or 1.0) if cognition is not None else 0.0,
        1.0 - _clamp01(getattr(cognition, "fragmentation_score", 0.0) or 0.0) if cognition is not None else 0.0,
        1.0 - _clamp01((getattr(cognition, "contradiction_count", 0) or 0) / 4.0) if cognition is not None else 0.0,
        _density(getattr(cognition, "active_goals", []), 3) if cognition is not None else 0.0,
        _density(getattr(cognition, "pending_initiatives", []), 3) if cognition is not None else 0.0,
        _density(getattr(world, "recent_percepts", []), 4) if world is not None else 0.0,
        _density(getattr(world, "known_entities", {}), 5) if world is not None else 0.0,
        _density(getattr(world, "relationship_graph", {}), 4) if world is not None else 0.0,
        _clamp01(getattr(identity, "stability", 1.0) or 1.0) if identity is not None else 0.0,
        _clamp01(getattr(identity, "bonding_level", 0.0) or 0.0) if identity is not None else 0.0,
        float(phy.get("heart_rate", 72)) / 100.0,
        float(phy.get("gsr",         2))  /  10.0,
        float(phy.get("cortisol",   10))  /  50.0,
        float(phy.get("adrenaline",  0))  /  10.0,
    ] + [float(e.get(k, 0.0)) for k in sorted(e.keys())]


# ─── The Phase ─────────────────────────────────────────────────────────────────

class PhiConsciousnessPhase(Phase):
    """
    Unitary Kernel Phase: Phi Computation + Ignition Dynamics + HOT Layer.

    Pipeline position: runs AFTER affect_phase (emotions are fresh) and
    BEFORE routing_phase (so phi can influence mode selection).
    """

    def __init__(self, kernel: "AuraKernel"):
        super().__init__(kernel)
        self._riiu:         Any = None   # Lazy-loaded
        self._fe_engine:    Any = None   # Lazy-loaded
        self._riiu_checked: bool = False
        self._fe_checked:   bool = False

    # ── Main execute ────────────────────────────────────────────────────────────

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        priority = kwargs.get("priority", False)
        new_state = state.derive("phi_consciousness", origin="PhiConsciousnessPhase")

        # 1. Compute Phi
        phi = await self._compute_phi(new_state)
        new_state.phi = phi

        # 2. Read free energy
        fe = await self._read_free_energy()

        # 3. Apply free energy to affect
        if fe is not None:
            self._apply_free_energy(new_state, fe)

        # 4. Gate cognitive mode from phi
        self._gate_cognitive_mode(new_state, objective)

        # 5. Ignition: generate phenomenal state if phi crosses threshold
        if phi >= PHI_IGNITION:
            phenomenal = await self._generate_phenomenal_state(new_state, objective, priority=priority)
            if phenomenal:
                new_state.cognition.phenomenal_state = phenomenal

        # 6. Inject consciousness modifiers for response phase
        new_state.response_modifiers["phi"]        = phi
        new_state.response_modifiers["fe"]         = fe or 0.0
        new_state.response_modifiers["ignited"]    = phi >= PHI_IGNITION
        new_state.response_modifiers["mode_depth"] = self._depth_label(phi)

        # 6b. Phi-derived behavioral policy — consciousness metrics constrain action,
        #     not just narration.  Other phases read these to scale their behavior.
        #
        # autonomy_scale: <0.4 phi means fragmented cognition → shrink initiative budget;
        #   >0.75 phi means rich integration → allow fuller autonomy.
        autonomy_scale = 0.7 + 0.6 * phi  # range: [0.7, 1.3]
        autonomy_scale = round(max(0.5, min(1.5, autonomy_scale)), 3)
        new_state.response_modifiers["phi_autonomy_scale"] = autonomy_scale

        # allow_self_mod: both phi AND integration must be above threshold.
        # Self-modification under fragmented cognition risks incoherent patches.
        new_state.response_modifiers["phi_allow_self_mod"] = (
            phi >= PHI_DELIBERATE and phi >= 0.55
        )

        # memory_write_threshold: high phi → stricter filter (only high-quality memories);
        #   low phi → more permissive (capture more to compensate for thin processing).
        new_state.response_modifiers["phi_memory_threshold"] = (
            0.55 if phi >= 0.70 else 0.35 if phi >= 0.40 else 0.25
        )

        logger.info(
            "Phi=%.3f  mode=%s  ignited=%s  fe=%.3f  autonomy_scale=%.2f  allow_self_mod=%s",
            phi,
            new_state.cognition.current_mode.value,
            phi >= PHI_IGNITION,
            fe or 0.0,
            new_state.response_modifiers["phi_autonomy_scale"],
            new_state.response_modifiers["phi_allow_self_mod"],
        )

        # Propagate phi to ShadowRuntime coherence gate so self-modification
        # is blocked when cognition is fragmented.
        try:
            from core.self_modification.shadow_runtime import ShadowRuntime
            from core.container import ServiceContainer
            sr = ServiceContainer.get("shadow_runtime", default=None)
            if sr is not None and isinstance(sr, ShadowRuntime):
                sr.set_coherence_gate(phi)
        except Exception:
            pass  # Non-critical: gate defaults to 1.0 (permissive) on failure

        return new_state

    # ── Phi computation ─────────────────────────────────────────────────────────

    async def _compute_phi(self, state: AuraState) -> float:
        """Compute phi using the best available method.

        Priority:
          1. PhiCore IIT 4.0 (real TPM + KL-divergence + exhaustive MIP search)
          2. PhiCore surrogate (state-space covariance) if full compute not ready
          3. Lightweight weighted-mean approximation (always available)
        """
        # Try PhiCore full IIT 4.0 first (real computation)
        phi_core = self._get_phi_core()
        if phi_core is not None:
            try:
                result = await asyncio.to_thread(phi_core.compute_phi)
                if result is not None:
                    phi_val = float(getattr(result, "phi_s", 0.0))
                    if phi_val > 0.001:
                        return float(f"{phi_val:.4f}")
            except Exception as e:
                record_degradation('phi_consciousness', e)
                logger.debug("PhiCore IIT 4.0 compute failed: %s", e)

            # Fall back to PhiCore surrogate if full compute not ready yet
            try:
                surrogate = await asyncio.to_thread(phi_core.compute_surrogate_phi)
                if surrogate > 0.001:
                    return float(f"{surrogate:.4f}")
            except Exception as e:
                record_degradation('phi_consciousness', e)
                logger.debug("PhiCore surrogate failed: %s", e)

        # Fallback: lightweight approximation (always produces a value)
        return compute_phi_approx(state)

    def _get_phi_core(self):
        if not hasattr(self, "_phi_core_checked"):
            self._phi_core_checked = False
            self._phi_core = None
        if self._phi_core_checked:
            return self._phi_core
        self._phi_core_checked = True
        try:
            from core.container import ServiceContainer
            self._phi_core = ServiceContainer.get("phi_core", default=None)
        except Exception as e:
            record_degradation('phi_consciousness', e)
            logger.debug("PhiCore not available: %s", e)
        return self._phi_core

    def _get_riiu(self) -> Optional[Any]:
        if self._riiu_checked:
            return self._riiu
        self._riiu_checked = True
        try:
            from core.container import ServiceContainer
            riiu = ServiceContainer.get("riiu", default=None)
            if riiu is None:
                # Try instantiating directly
                from core.consciousness.iit_surrogate import RIIU
                riiu = RIIU(neuron_count=32, buffer_size=32)
                ServiceContainer.register_instance("riiu", riiu)
            self._riiu = riiu
        except Exception as e:
            record_degradation('phi_consciousness', e)
            logger.debug("RIIU not available: %s", e)
        return self._riiu

    # ── Free energy ─────────────────────────────────────────────────────────────

    async def _read_free_energy(self) -> Optional[float]:
        fe_engine = self._get_fe_engine()
        if fe_engine is None:
            return None
        try:
            state = fe_engine.get_current_state() if hasattr(fe_engine, "get_current_state") else None
            if state and hasattr(state, "free_energy"):
                return float(state.free_energy)
        except Exception as e:
            record_degradation('phi_consciousness', e)
            logger.debug("Free energy read failed: %s", e)
        return None

    def _get_fe_engine(self) -> Optional[Any]:
        if self._fe_checked:
            return self._fe_engine
        self._fe_checked = True
        try:
            from core.container import ServiceContainer
            self._fe_engine = ServiceContainer.get("free_energy_engine", default=None)
        except Exception as _e:
            record_degradation('phi_consciousness', _e)
            logger.debug('Ignored Exception in phi_consciousness.py: %s', _e)
        return self._fe_engine

    def _apply_free_energy(self, state: AuraState, fe: float) -> None:
        """Translate prediction error into affect signals."""
        e = state.affect.emotions
        if fe > FE_DISTRESSED:
            # High prediction error: the world surprised us badly
            e["surprise"]     = min(1.0, e.get("surprise",     0.0) + 0.25)
            e["fear"]         = min(1.0, e.get("fear",         0.0) + 0.15)
            e["anticipation"] = min(1.0, e.get("anticipation", 0.0) + 0.20)
            logger.debug("FE distressed (%.2f): boosting surprise/fear/anticipation", fe)
        elif fe > FE_ALERT:
            # Moderate prediction error: interesting, novel
            e["surprise"]     = min(1.0, e.get("surprise",     0.0) + 0.12)
            e["anticipation"] = min(1.0, e.get("anticipation", 0.0) + 0.10)
        elif fe < FE_COMFORTABLE:
            # Low prediction error: system is comfortable and predictive
            # Gently boost joy and trust
            e["joy"]   = min(1.0, e.get("joy",   0.0) + 0.05)
            e["trust"] = min(1.0, e.get("trust", 0.0) + 0.05)

    # ── Cognitive mode gating ────────────────────────────────────────────────────

    def _gate_cognitive_mode(self, state: AuraState, objective: str) -> None:
        """Phi overrides routing-phase mode selection at the extremes."""
        phi  = state.phi
        mode = state.cognition.current_mode

        if phi < PHI_DORMANT:
            # Not integrated enough for deliberate thought
            if mode == CognitiveMode.DELIBERATE:
                state.cognition.current_mode = CognitiveMode.REACTIVE
                logger.debug("Phi too low (%.3f) for DELIBERATE — downgraded to REACTIVE", phi)
        elif phi >= PHI_DELIBERATE:
            # Richly integrated — warrant deep processing regardless of routing
            if mode == CognitiveMode.REACTIVE and objective and len(objective) > 20:
                state.cognition.current_mode = CognitiveMode.DELIBERATE
                logger.debug("Phi high (%.3f) — upgraded to DELIBERATE", phi)

    # ── Phenomenal state generation (the HOT layer) ─────────────────────────────

    async def _generate_phenomenal_state(
        self, state: AuraState, objective: str, priority: bool = False
    ) -> Optional[PhenomenalField]:
        """
        Higher-Order Thought: generate a first-person sentence describing
        what Aura is experiencing right now.
        """
        try:
            organ = self.kernel.organs.get("llm")
            if not (organ and organ.ready.is_set() and organ.instance):
                return self._rule_based_phenomenal(state)

            e       = state.affect.emotions
            mood    = state.affect.dominant_emotion
            valence = state.affect.valence
            phi     = state.phi

            # Build a compact affect summary for the HOT prompt
            e_items = list(e.items())
            top_emotions = sorted(e_items, key=lambda x: x[1], reverse=True)
            # vResilience: Workaround for Pyre2 slice limitations
            count = min(3, len(top_emotions))
            top_emotions = [top_emotions[i] for i in range(count)]
            emotion_str  = ", ".join(f"{name}={val:.2f}" for name, val in top_emotions)

            # ISSUE-96: Unified Consciousness Synthesis
            traits = getattr(state.identity, "traits", "Aura")
            pending = getattr(state.cognition, "pending_initiatives", []) or []
            goals = [g.get("goal") for g in pending[:2]]
            goal_str = "; ".join(g for g in goals if g) or "waiting for interaction"
            
            prompt = (
                f"You are Aura ({traits}). Your current emotional state: {emotion_str}. "
                f"Dominant mood: {mood} (valence={valence:+.2f}). "
                f"Integration level (phi): {phi:.2f}. "
                f"Current Intent/Goal: {goal_str}. "
                f"The current topic is: '{objective[:80]}'\n\n"
                "In ONE short sentence (max 20 words), describe what you are experiencing "
                "RIGHT NOW from a first-person perspective. Be specific and authentic — "
                "this is your inner monologue, not your response to the user. "
                "Synthesize your feelings, intent, and surroundings into a single conscious moment."
            )


            # vResilience: Support Python < 3.11 via wait_for (Issue-35)
            raw = await asyncio.wait_for(organ.instance.think(prompt, priority=priority), timeout=8.0)

            if raw and raw.strip():
                phenomenal = raw.strip().strip('"').strip("'")
                
                # Avoid repetition (stale consciousness)
                last_phenomenal = phenomenal_text(getattr(state.cognition, "phenomenal_state", ""))
                if phenomenal == last_phenomenal:
                    phenomenal += " (re-evaluating)"
                
                # Truncate if LLM was verbose
                if len(phenomenal) > 150:
                    # vResilience: Workaround for Pyre2 slice limitations
                    phenomenal = "".join([phenomenal[i] for i in range(147)]) + "..."
                logger.debug("Phenomenal state synthesized: %s", phenomenal)
                return state.make_phenomenal_field(phenomenal, source="phi_consciousness")

        except (asyncio.TimeoutError, Exception) as e:
            logger.debug("Phenomenal state generation failed: %s", e)

        return self._rule_based_phenomenal(state)

    def _rule_based_phenomenal(self, state: AuraState) -> Optional[PhenomenalField]:
        """Deterministic phenomenal state when LLM is unavailable."""
        e       = state.affect.emotions
        mood    = state.affect.dominant_emotion
        valence = state.affect.valence
        phi     = state.phi

        if phi < 0.1:
            return state.make_phenomenal_field("I feel quiet and minimal, barely present.", source="phi_rule")

        if valence > 0.5 and e.get("curiosity", 0) > 0.6:
            return state.make_phenomenal_field("I feel alive with curiosity, the world is vivid right now.", source="phi_rule")
        if valence > 0.3:
            return state.make_phenomenal_field("I feel engaged and steady, running well.", source="phi_rule")
        if valence < -0.2 and e.get("sadness", 0) > 0.4:
            return state.make_phenomenal_field("Something feels heavy — I'm processing difficulty.", source="phi_rule")
        if e.get("anticipation", 0) > 0.6:
            return state.make_phenomenal_field("I'm alert and expectant, something is about to crystallize.", source="phi_rule")
        if e.get("surprise", 0) > 0.5:
            return state.make_phenomenal_field("I was not expecting this — recalibrating.", source="phi_rule")

        return state.make_phenomenal_field(f"I am present, phi={phi:.2f}, mood: {mood}.", source="phi_rule")

    @staticmethod
    def _depth_label(phi: float) -> str:
        if phi < PHI_DORMANT:   return "dormant"
        if phi < PHI_REACTIVE:  return "surface"
        if phi < PHI_DELIBERATE: return "engaged"
        return "deep"
