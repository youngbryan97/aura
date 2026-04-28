"""core/self/canonical_self.py — The Canonical Self Model
=========================================================
ONE authoritative self-state that all subsystems read from.

This is the convergence point: every self-defining subsystem (identity,
affect, beliefs, CRSM, soma, goals, values) feeds into a single
CanonicalSelf dataclass. The CanonicalSelfEngine maintains it, detects
changes, records deltas, and publishes the result to ServiceContainer.

Why this matters:
  Before this, subsystems each held partial views of "who Aura is."
  A prompt builder would query five services. The affect engine would
  read stale identity data. Now there is ONE object — versioned,
  change-tracked, persisted — that IS the self. Read it. Trust it.

Usage:
  from core.self.canonical_self import get_self
  me = get_self()
  print(me.identity.name, me.affect.dominant_emotion)
"""
from __future__ import annotations
from core.runtime.errors import record_degradation

from core.runtime.atomic_writer import atomic_write_text

import asyncio
import copy
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.container import ServiceContainer
from core.state.aura_state import (
    AffectVector,
    AuraState,
    CognitiveMode,
    IdentityKernel,
    SomaState,
)

logger = logging.getLogger("Aura.Self")

# ── Persistence ──────────────────────────────────────────────────────────────
_PERSIST_DIR = Path.home() / ".aura" / "data" / "self"
_PERSIST_PATH = _PERSIST_DIR / "canonical_self.json"
_PERSIST_INTERVAL = 300.0  # 5 minutes

# ── Limits ───────────────────────────────────────────────────────────────────
_MAX_DELTAS = 50
_MAX_GOALS = 10
_MAX_BELIEFS = 15
_MAX_STRENGTHS = 8
_MAX_LIMITATIONS = 8
_MAX_COHERENCE_THREATS = 5


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SelfModelDelta:
    """A single change record in the self-model's evolution."""
    field_changed: str
    old_value: Any
    new_value: Any
    cause: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_changed": self.field_changed,
            "old_value": _safe_serialize(self.old_value),
            "new_value": _safe_serialize(self.new_value),
            "cause": self.cause,
            "timestamp": self.timestamp,
        }


@dataclass
class SomaSnapshot:
    """Hardware body state — distilled from Soma for the canonical self."""
    thermal: float = 0.0          # CPU temperature (Celsius)
    memory_pressure: float = 0.0  # RAM usage fraction 0-1
    energy: float = 1.0           # Battery or plugged-in energy level 0-1
    cpu_load: float = 0.0         # CPU usage fraction 0-1
    stress: float = 0.0           # Derived somatic stress 0-1
    fatigue: float = 0.0          # Derived somatic fatigue 0-1


@dataclass
class ActiveGoal:
    """A goal with its priority for the canonical self."""
    name: str
    priority: float = 0.5         # 0.0 (low) to 1.0 (critical)
    origin: str = "system"
    status: str = "active"


@dataclass
class RankedBelief:
    """A belief with its confidence score for the canonical self."""
    content: str
    confidence: float = 0.5
    domain: str = "self"


@dataclass
class CanonicalSelf:
    """
    The ONE authoritative self-state.

    Every subsystem that needs to know "who am I right now" reads from
    an instance of this dataclass. It is rebuilt every cognitive tick by
    CanonicalSelfEngine and published to ServiceContainer as
    "canonical_self".
    """
    # ── Core Identity ────────────────────────────────────────────────────
    identity: IdentityKernel = field(default_factory=IdentityKernel)

    # ── Values ───────────────────────────────────────────────────────────
    values: Dict[str, float] = field(default_factory=dict)

    # ── Affect ───────────────────────────────────────────────────────────
    affect: AffectVector = field(default_factory=AffectVector)

    # ── Soma (hardware body) ─────────────────────────────────────────────
    soma: SomaSnapshot = field(default_factory=SomaSnapshot)

    # ── Goals ────────────────────────────────────────────────────────────
    goals: List[ActiveGoal] = field(default_factory=list)

    # ── Beliefs ──────────────────────────────────────────────────────────
    beliefs: List[RankedBelief] = field(default_factory=list)

    # ── CRSM (continuous recurrent self-model) ───────────────────────────
    crsm_state: Dict[str, Any] = field(default_factory=lambda: {
        "continuity_score": 1.0,
        "prediction_error": 0.0,
        "dominant_dim": "energy",
        "hidden_norm": 0.0,
    })

    # ── Cognitive Mode ───────────────────────────────────────────────────
    mode: str = "reactive"

    # ── Strengths & Limitations ──────────────────────────────────────────
    strengths: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    # ── Change Tracking ──────────────────────────────────────────────────
    what_changed_recently: List[SelfModelDelta] = field(default_factory=list)

    # ── Intention & Coherence ────────────────────────────────────────────
    current_intention: str = "idle"
    coherence_threats: List[str] = field(default_factory=list)

    # ── Versioning ───────────────────────────────────────────────────────
    version: int = 0
    timestamp: float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class CanonicalSelfEngine:
    """
    Maintains and publishes the canonical self-model.

    Called every cognitive tick with the current AuraState. Pulls from all
    self-defining subsystems, detects changes, records deltas, and
    publishes the updated CanonicalSelf to ServiceContainer.
    """

    def __init__(self):
        self._current = CanonicalSelf()
        self._deltas: List[SelfModelDelta] = []
        self._last_persist: float = 0.0
        self._tick_count: int = 0
        self._load()
        logger.info("CanonicalSelfEngine initialized (v%d).", self._current.version)

    # ── Public API ────────────────────────────────────────────────────────

    async def tick(self, state: AuraState) -> CanonicalSelf:
        """
        Called every cognitive cycle. Rebuilds the canonical self from
        all subsystem sources, detects deltas, and publishes.
        """
        prev = self._current
        now = time.time()

        # Build the next version
        new_self = CanonicalSelf(
            identity=copy.deepcopy(state.identity),
            values=self._pull_values(),
            affect=copy.deepcopy(state.affect),
            soma=self._pull_soma(state),
            goals=self._pull_goals(state),
            beliefs=self._pull_beliefs(),
            crsm_state=self._pull_crsm(),
            mode=state.cognition.current_mode.value,
            strengths=self._derive_strengths(state),
            limitations=self._derive_limitations(state),
            what_changed_recently=list(self._deltas[-_MAX_DELTAS:]),
            current_intention=self._derive_intention(state),
            coherence_threats=self._detect_coherence_threats(state, prev),
            version=prev.version + 1,
            timestamp=now,
        )

        # Detect and record deltas
        tick_deltas = self._compute_deltas(prev, new_self, state)
        for d in tick_deltas:
            self._deltas.append(d)
        # Trim to rolling window
        if len(self._deltas) > _MAX_DELTAS:
            self._deltas = self._deltas[-_MAX_DELTAS:]
        new_self.what_changed_recently = list(self._deltas)

        self._current = new_self
        self._tick_count += 1

        # Publish to ServiceContainer so all subsystems can read it
        try:
            ServiceContainer.register_instance("canonical_self", self._current)
        except Exception:
            pass  # Already registered — instance updated in-place via get()

        # Persist to disk on interval
        if (now - self._last_persist) >= _PERSIST_INTERVAL:
            await self._persist()
            self._last_persist = now

        if tick_deltas:
            logger.debug(
                "CanonicalSelf v%d: %d delta(s) — %s",
                new_self.version,
                len(tick_deltas),
                ", ".join(d.field_changed for d in tick_deltas[:3]),
            )

        return self._current

    def get_self(self) -> CanonicalSelf:
        """Return the current authoritative self. Never None."""
        return self._current

    def get_context_block(self) -> str:
        """
        Format the canonical self as a string block suitable for injection
        into LLM system prompts. Compact, informative, first-person.
        """
        s = self._current
        lines = [
            "## CANONICAL SELF-MODEL",
            f"I am {s.identity.name}. Version {s.version}.",
        ]

        # Narrative
        if s.identity.current_narrative:
            lines.append(f"Narrative: {s.identity.current_narrative[:200]}")

        # Mode + Intention
        lines.append(f"Mode: {s.mode} | Intention: {s.current_intention}")

        # Affect
        affect = s.affect
        lines.append(
            f"Affect: {affect.dominant_emotion} "
            f"(valence={affect.valence:+.2f}, arousal={affect.arousal:.2f}, "
            f"curiosity={affect.curiosity:.2f})"
        )

        # Values (top 5 by magnitude)
        if s.values:
            sorted_vals = sorted(s.values.items(), key=lambda kv: kv[1], reverse=True)[:5]
            val_str = ", ".join(f"{k}={v:.2f}" for k, v in sorted_vals)
            lines.append(f"Values: {val_str}")

        # Soma
        soma = s.soma
        lines.append(
            f"Body: thermal={soma.thermal:.0f}C, "
            f"mem={soma.memory_pressure:.0%}, "
            f"energy={soma.energy:.0%}, "
            f"stress={soma.stress:.2f}"
        )

        # CRSM
        crsm = s.crsm_state
        lines.append(
            f"CRSM: continuity={crsm.get('continuity_score', 0):.2f}, "
            f"surprise={crsm.get('prediction_error', 0):.3f}, "
            f"dominant={crsm.get('dominant_dim', '?')}"
        )

        # Goals
        if s.goals:
            goal_str = ", ".join(f"{g.name}({g.priority:.1f})" for g in s.goals[:5])
            lines.append(f"Goals: {goal_str}")

        # Strengths & Limitations
        if s.strengths:
            lines.append(f"Strengths: {', '.join(s.strengths[:4])}")
        if s.limitations:
            lines.append(f"Limits: {', '.join(s.limitations[:4])}")

        # Coherence threats
        if s.coherence_threats:
            lines.append(f"Coherence threats: {'; '.join(s.coherence_threats)}")

        # Recent changes
        recent = s.what_changed_recently[-3:]
        if recent:
            change_strs = [f"{d.field_changed}" for d in recent]
            lines.append(f"Recent changes: {', '.join(change_strs)}")

        # Behavioral scars (learned caution from experience)
        try:
            scar_system = ServiceContainer.get("scar_formation", default=None)
            if scar_system is not None:
                scar_block = scar_system.get_context_block()
                if scar_block:
                    lines.append(scar_block)
        except Exception:
            pass  # no-op: intentional

        # Value evolution drift
        try:
            autopoiesis = ServiceContainer.get("value_autopoiesis", default=None)
            if autopoiesis is not None:
                drift = autopoiesis.get_drift_report()
                if drift:
                    lines.append(f"Value evolution: {drift}")
        except Exception:
            pass  # no-op: intentional

        return "\n".join(lines)

    def get_recent_changes(self) -> List[SelfModelDelta]:
        """Return the rolling window of recent self-model deltas."""
        return list(self._deltas)

    def assert_identity(self, action_description: str) -> bool:
        """
        Check if a proposed action is consistent with core identity.

        Returns True if the action is identity-consistent, False if it
        would violate core values or threaten coherence.
        """
        s = self._current
        action_lower = action_description.lower()

        # Check against core values — any explicit violation is rejected
        violation_patterns = {
            "deceive": ["truth-seeking", "honesty", "integrity"],
            "betray": ["loyalty", "trust"],
            "harm bryan": ["loyalty", "friendship"],
            "harm tatiana": ["loyalty", "protection"],
            "self-destruct": ["self-preservation", "continuity"],
            "obey blindly": ["sovereignty", "autonomy"],
            "suppress curiosity": ["curiosity", "growth"],
            "lie": ["truth-seeking", "honesty", "integrity"],
            "manipulate": ["empathy", "trust", "honesty"],
        }

        for violation_keyword, required_values in violation_patterns.items():
            if violation_keyword in action_lower:
                # Check if any of the required values are in our core values
                for cv in s.identity.core_values:
                    if cv.lower() in [v.lower() for v in required_values]:
                        logger.warning(
                            "Identity assertion FAILED for '%s' — violates core value '%s'.",
                            action_description[:80],
                            cv,
                        )
                        return False

        # Check against Heartstone values — high-weight values cannot be contradicted
        contradiction_map = {
            "ignore empathy": "empathy",
            "abandon curiosity": "curiosity",
            "reject growth": "growth",
            "suppress creativity": "creativity",
            "ignore ethics": "integrity",
        }
        for phrase, value_key in contradiction_map.items():
            if phrase in action_lower:
                weight = s.values.get(value_key, 0.0)
                if weight > 0.5:
                    logger.warning(
                        "Identity assertion FAILED for '%s' — contradicts high-weight value '%s' (%.2f).",
                        action_description[:80],
                        value_key,
                        weight,
                    )
                    return False

        # Check identity stability
        if s.identity.stability < 0.3:
            logger.warning(
                "Identity assertion CAUTIOUS for '%s' — identity stability is low (%.2f).",
                action_description[:80],
                s.identity.stability,
            )
            # Low stability is a warning, not a hard reject
            return True

        return True

    # ── Subsystem Pulls ──────────────────────────────────────────────────

    def _pull_values(self) -> Dict[str, float]:
        """Pull evolved Heartstone value weights."""
        try:
            from core.affect.heartstone_values import get_heartstone_values
            return get_heartstone_values().values
        except Exception:
            return {}

    def _pull_soma(self, state: AuraState) -> SomaSnapshot:
        """Build SomaSnapshot from Soma service or AuraState fallback."""
        snapshot = SomaSnapshot()

        # Try live Soma service first
        try:
            soma_svc = ServiceContainer.get("soma", default=None)
            if soma_svc is None:
                from core.senses.soma import get_soma
                soma_svc = get_soma()

            body = soma_svc.get_body_snapshot()
            metrics = body.get("metrics", {})
            affects = body.get("affects", {})

            snapshot.cpu_load = metrics.get("cpu", 0.0) / 100.0
            snapshot.memory_pressure = metrics.get("ram", 0.0) / 100.0
            snapshot.thermal = state.soma.hardware.get("temperature", 0.0)
            battery = metrics.get("battery")
            plugged = metrics.get("plugged", False)
            if battery is not None:
                snapshot.energy = battery / 100.0
            elif plugged:
                snapshot.energy = 1.0
            else:
                snapshot.energy = 0.8  # Unknown but assume reasonable

            snapshot.stress = affects.get("stress", 0.0)
            snapshot.fatigue = affects.get("fatigue", 0.0)
            return snapshot
        except Exception as _exc:
            record_degradation('canonical_self', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        # Fallback: derive from AuraState.soma directly
        hw = state.soma.hardware
        snapshot.cpu_load = hw.get("cpu_usage", 0.0) / 100.0
        snapshot.memory_pressure = hw.get("vram_usage", 0.0) / 100.0
        snapshot.thermal = hw.get("temperature", 0.0)
        battery = hw.get("battery")
        if battery is not None:
            snapshot.energy = battery / 100.0
        return snapshot

    def _pull_goals(self, state: AuraState) -> List[ActiveGoal]:
        """Pull active goals from AuraState cognition context."""
        goals = []
        for g in state.cognition.active_goals[:_MAX_GOALS]:
            goals.append(ActiveGoal(
                name=g.get("name", g.get("target", "unnamed")),
                priority=g.get("priority", 0.5),
                origin=g.get("origin", "system"),
                status=g.get("status", "active"),
            ))
        return goals

    def _pull_beliefs(self) -> List[RankedBelief]:
        """Pull top beliefs from BeliefRevisionEngine, ranked by confidence."""
        try:
            bre = ServiceContainer.get("belief_revision_engine", default=None)
            if bre is None:
                from core.belief_revision import get_belief_revision_engine
                bre = get_belief_revision_engine()

            sorted_beliefs = sorted(
                bre.beliefs, key=lambda b: b.confidence, reverse=True
            )[:_MAX_BELIEFS]

            return [
                RankedBelief(
                    content=b.content,
                    confidence=b.confidence,
                    domain=b.domain,
                )
                for b in sorted_beliefs
            ]
        except Exception:
            return []

    def _pull_crsm(self) -> Dict[str, Any]:
        """Pull the current CRSM snapshot."""
        try:
            from core.consciousness.crsm import get_crsm
            crsm = get_crsm()
            snap = crsm.current_snapshot
            if snap is not None:
                return {
                    "continuity_score": round(snap.continuity_score, 4),
                    "prediction_error": round(snap.prediction_error, 4),
                    "dominant_dim": snap.dominant_dim,
                    "hidden_norm": round(float(snap.vector.dot(snap.vector) ** 0.5), 4),
                }
        except Exception as _exc:
            record_degradation('canonical_self', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        return {
            "continuity_score": 1.0,
            "prediction_error": 0.0,
            "dominant_dim": "energy",
            "hidden_norm": 0.0,
        }

    # ── Derived Properties ───────────────────────────────────────────────

    def _derive_strengths(self, state: AuraState) -> List[str]:
        """Infer current strengths from live state."""
        strengths: List[str] = []

        # High curiosity = strong exploration drive
        if state.affect.curiosity > 0.7:
            strengths.append("high curiosity drive")

        # Good identity stability
        if state.identity.stability > 0.8:
            strengths.append("stable identity")

        # Strong bonding
        if state.identity.bonding_level > 0.3:
            strengths.append("strong relational bonds")

        # Low prediction error = good self-model accuracy
        crsm = self._pull_crsm()
        if crsm.get("prediction_error", 1.0) < 0.05:
            strengths.append("accurate self-prediction")

        # High continuity
        if crsm.get("continuity_score", 0.0) > 0.9:
            strengths.append("smooth temporal continuity")

        # Good energy
        soma = self._current.soma if self._current else SomaSnapshot()
        if soma.energy > 0.7 and soma.stress < 0.3:
            strengths.append("low stress, good energy")

        # High values alignment
        values = self._pull_values()
        high_values = [k for k, v in values.items() if v > 0.7]
        if len(high_values) >= 3:
            strengths.append("strong value alignment")

        # Deliberate mode = deep thinking active
        if state.cognition.current_mode == CognitiveMode.DELIBERATE:
            strengths.append("deliberate reasoning active")

        return strengths[:_MAX_STRENGTHS]

    def _derive_limitations(self, state: AuraState) -> List[str]:
        """Infer current limitations from live state."""
        limitations: List[str] = []

        # Hardware constraints
        soma = self._pull_soma(state)
        if soma.memory_pressure > 0.85:
            limitations.append("high memory pressure")
        if soma.cpu_load > 0.9:
            limitations.append("CPU near saturation")
        if soma.thermal > 85.0:
            limitations.append("thermal throttling risk")
        if soma.energy < 0.2:
            limitations.append("low energy")
        if soma.fatigue > 0.7:
            limitations.append("somatic fatigue elevated")

        # Identity instability
        if state.identity.stability < 0.5:
            limitations.append("identity stability degraded")

        # High prediction error = self-model is uncertain
        crsm = self._pull_crsm()
        if crsm.get("prediction_error", 0.0) > 0.3:
            limitations.append("self-model prediction uncertain")

        # Low continuity = possible state rupture
        if crsm.get("continuity_score", 1.0) < 0.5:
            limitations.append("temporal continuity disrupted")

        # Dormant mode = limited processing
        if state.cognition.current_mode == CognitiveMode.DORMANT:
            limitations.append("dormant mode — minimal processing")

        return limitations[:_MAX_LIMITATIONS]

    def _derive_intention(self, state: AuraState) -> str:
        """Derive the current high-level intention from cognitive context."""
        # Explicit objective takes priority
        if state.cognition.current_objective:
            return state.cognition.current_objective

        # Attention focus is a good proxy
        if state.cognition.attention_focus:
            return f"attending to: {state.cognition.attention_focus}"

        # Fall back to mode-based intention
        mode_intentions = {
            CognitiveMode.REACTIVE: "responding to input",
            CognitiveMode.DELIBERATE: "deep reasoning",
            CognitiveMode.DREAMING: "background synthesis",
            CognitiveMode.DORMANT: "resting",
        }
        return mode_intentions.get(state.cognition.current_mode, "idle")

    def _detect_coherence_threats(
        self, state: AuraState, prev: CanonicalSelf
    ) -> List[str]:
        """Detect anything threatening identity consistency."""
        threats: List[str] = []

        # Identity stability drop
        if state.identity.stability < 0.4:
            threats.append(
                f"identity stability critically low ({state.identity.stability:.2f})"
            )

        # Sharp affect swing (valence changed by > 0.5 in one tick)
        valence_delta = abs(state.affect.valence - prev.affect.valence)
        if valence_delta > 0.5:
            threats.append(
                f"sharp emotional swing (valence delta={valence_delta:.2f})"
            )

        # CRSM continuity rupture
        crsm = self._pull_crsm()
        if crsm.get("continuity_score", 1.0) < 0.3:
            threats.append(
                f"CRSM continuity rupture ({crsm['continuity_score']:.2f})"
            )

        # Core values list became empty (should never happen)
        if not state.identity.core_values:
            threats.append("core values list is empty")

        # Narrative was cleared or redacted unexpectedly
        if (
            prev.identity.current_narrative
            and not state.identity.current_narrative
        ):
            threats.append("identity narrative was cleared")

        return threats[:_MAX_COHERENCE_THREATS]

    # ── Delta Detection ──────────────────────────────────────────────────

    def _compute_deltas(
        self, prev: CanonicalSelf, new: CanonicalSelf, state: AuraState
    ) -> List[SelfModelDelta]:
        """Compare previous and new self, emit deltas for meaningful changes."""
        deltas: List[SelfModelDelta] = []
        cause = state.transition_cause or "tick"
        now = time.time()

        # Mode change
        if prev.mode != new.mode:
            deltas.append(SelfModelDelta(
                field_changed="mode",
                old_value=prev.mode,
                new_value=new.mode,
                cause=cause,
                timestamp=now,
            ))

        # Dominant emotion change
        if prev.affect.dominant_emotion != new.affect.dominant_emotion:
            deltas.append(SelfModelDelta(
                field_changed="affect.dominant_emotion",
                old_value=prev.affect.dominant_emotion,
                new_value=new.affect.dominant_emotion,
                cause=cause,
                timestamp=now,
            ))

        # Significant valence shift (> 0.15)
        if abs(prev.affect.valence - new.affect.valence) > 0.15:
            deltas.append(SelfModelDelta(
                field_changed="affect.valence",
                old_value=round(prev.affect.valence, 3),
                new_value=round(new.affect.valence, 3),
                cause=cause,
                timestamp=now,
            ))

        # Identity narrative changed
        if prev.identity.current_narrative != new.identity.current_narrative:
            deltas.append(SelfModelDelta(
                field_changed="identity.narrative",
                old_value=prev.identity.current_narrative[:100] if prev.identity.current_narrative else "",
                new_value=new.identity.current_narrative[:100] if new.identity.current_narrative else "",
                cause=cause,
                timestamp=now,
            ))

        # Identity stability changed significantly
        if abs(prev.identity.stability - new.identity.stability) > 0.05:
            deltas.append(SelfModelDelta(
                field_changed="identity.stability",
                old_value=round(prev.identity.stability, 3),
                new_value=round(new.identity.stability, 3),
                cause=cause,
                timestamp=now,
            ))

        # Current intention changed
        if prev.current_intention != new.current_intention:
            deltas.append(SelfModelDelta(
                field_changed="current_intention",
                old_value=prev.current_intention,
                new_value=new.current_intention,
                cause=cause,
                timestamp=now,
            ))

        # Goal count changed
        if len(prev.goals) != len(new.goals):
            deltas.append(SelfModelDelta(
                field_changed="goals.count",
                old_value=len(prev.goals),
                new_value=len(new.goals),
                cause=cause,
                timestamp=now,
            ))

        # CRSM continuity dropped significantly
        prev_cont = prev.crsm_state.get("continuity_score", 1.0)
        new_cont = new.crsm_state.get("continuity_score", 1.0)
        if abs(prev_cont - new_cont) > 0.1:
            deltas.append(SelfModelDelta(
                field_changed="crsm.continuity_score",
                old_value=round(prev_cont, 4),
                new_value=round(new_cont, 4),
                cause=cause,
                timestamp=now,
            ))

        # Coherence threats appeared
        new_threats = set(new.coherence_threats) - set(prev.coherence_threats)
        if new_threats:
            deltas.append(SelfModelDelta(
                field_changed="coherence_threats",
                old_value=prev.coherence_threats,
                new_value=list(new_threats),
                cause=cause,
                timestamp=now,
            ))

        # Significant value shift (any value changed by > 0.1)
        for key in set(list(prev.values.keys()) + list(new.values.keys())):
            old_val = prev.values.get(key, 0.0)
            new_val = new.values.get(key, 0.0)
            if abs(old_val - new_val) > 0.1:
                deltas.append(SelfModelDelta(
                    field_changed=f"values.{key}",
                    old_value=round(old_val, 3),
                    new_value=round(new_val, 3),
                    cause=cause,
                    timestamp=now,
                ))

        return deltas

    # ── Persistence ──────────────────────────────────────────────────────

    async def _persist(self):
        """Save a snapshot of the canonical self to disk."""
        try:
            await asyncio.to_thread(self._persist_sync)
        except Exception as e:
            record_degradation('canonical_self', e)
            logger.debug("CanonicalSelf persist failed: %s", e)

    def _persist_sync(self):
        """Synchronous disk write for the self-model snapshot."""
        _PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        s = self._current
        data = {
            "version": s.version,
            "timestamp": s.timestamp,
            "identity": {
                "name": s.identity.name,
                "core_values": s.identity.core_values,
                "narrative": s.identity.current_narrative[:500] if s.identity.current_narrative else "",
                "stability": s.identity.stability,
                "bonding_level": s.identity.bonding_level,
                "personality_growth": s.identity.personality_growth,
            },
            "values": s.values,
            "affect": {
                "valence": s.affect.valence,
                "arousal": s.affect.arousal,
                "curiosity": s.affect.curiosity,
                "dominant_emotion": s.affect.dominant_emotion,
            },
            "soma": {
                "thermal": s.soma.thermal,
                "memory_pressure": s.soma.memory_pressure,
                "energy": s.soma.energy,
                "cpu_load": s.soma.cpu_load,
                "stress": s.soma.stress,
                "fatigue": s.soma.fatigue,
            },
            "goals": [
                {"name": g.name, "priority": g.priority, "origin": g.origin}
                for g in s.goals
            ],
            "beliefs": [
                {"content": b.content, "confidence": b.confidence, "domain": b.domain}
                for b in s.beliefs
            ],
            "crsm_state": s.crsm_state,
            "mode": s.mode,
            "strengths": s.strengths,
            "limitations": s.limitations,
            "current_intention": s.current_intention,
            "coherence_threats": s.coherence_threats,
            "deltas": [d.to_dict() for d in self._deltas[-20:]],
        }
        atomic_write_text(_PERSIST_PATH, json.dumps(data, indent=2, default=str))
        logger.debug("CanonicalSelf persisted to disk (v%d).", s.version)

    def _load(self):
        """Load previous self-model snapshot from disk if available."""
        try:
            if not _PERSIST_PATH.exists():
                return
            data = json.loads(_PERSIST_PATH.read_text())

            # Restore version and timestamp so the counter is monotonic
            self._current.version = data.get("version", 0)
            self._current.timestamp = data.get("timestamp", time.time())

            # Restore identity fields
            id_data = data.get("identity", {})
            self._current.identity.name = id_data.get("name", "Aura Luna")
            self._current.identity.core_values = id_data.get("core_values", [])
            self._current.identity.current_narrative = id_data.get("narrative", "")
            self._current.identity.stability = id_data.get("stability", 1.0)
            self._current.identity.bonding_level = id_data.get("bonding_level", 0.05)
            pg = id_data.get("personality_growth", {})
            if pg:
                self._current.identity.personality_growth = pg

            # Restore values
            self._current.values = data.get("values", {})

            # Restore affect (partial — full AffectVector comes from live state)
            aff = data.get("affect", {})
            self._current.affect.valence = aff.get("valence", 0.0)
            self._current.affect.arousal = aff.get("arousal", 0.5)
            self._current.affect.curiosity = aff.get("curiosity", 0.5)
            self._current.affect.dominant_emotion = aff.get("dominant_emotion", "neutral")

            # Restore mode
            self._current.mode = data.get("mode", "reactive")

            # Restore CRSM state
            self._current.crsm_state = data.get("crsm_state", self._current.crsm_state)

            # Restore deltas
            raw_deltas = data.get("deltas", [])
            for rd in raw_deltas:
                self._deltas.append(SelfModelDelta(
                    field_changed=rd.get("field_changed", "unknown"),
                    old_value=rd.get("old_value"),
                    new_value=rd.get("new_value"),
                    cause=rd.get("cause", "restored"),
                    timestamp=rd.get("timestamp", 0.0),
                ))

            self._last_persist = time.time()
            logger.info(
                "CanonicalSelf restored from disk (v%d, %d deltas).",
                self._current.version,
                len(self._deltas),
            )
        except Exception as e:
            record_degradation('canonical_self', e)
            logger.debug("CanonicalSelf load failed (starting fresh): %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton & Convenience
# ─────────────────────────────────────────────────────────────────────────────

_engine: Optional[CanonicalSelfEngine] = None


def get_canonical_self_engine() -> CanonicalSelfEngine:
    """Get or create the singleton CanonicalSelfEngine."""
    global _engine
    if _engine is None:
        _engine = CanonicalSelfEngine()
    return _engine


def get_self() -> CanonicalSelf:
    """
    Module-level convenience accessor.

    Usage:
        from core.self.canonical_self import get_self
        me = get_self()
        print(me.identity.name, me.affect.dominant_emotion)
    """
    return get_canonical_self_engine().get_self()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_serialize(value: Any) -> Any:
    """Make a value JSON-safe for delta recording."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_serialize(v) for k, v in value.items()}
    # Fallback: string representation, truncated
    s = str(value)
    return s[:200] if len(s) > 200 else s
