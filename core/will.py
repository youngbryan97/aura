"""core/will.py -- The Unified Will
====================================
THE single locus of decision authority in Aura.

Every significant action -- responses, tool calls, memory writes, autonomous
initiatives, state mutations -- MUST pass through the Unified Will.  Nothing
user-visible or world-affecting happens without a WillDecision.

This module does NOT replace the subsystem authorities.  It COMPOSES them:
  - SubstrateAuthority  (embodied gate: field coherence, somatic veto, neurochemistry)
  - ExecutiveCore       (executive reasoning: intent formation, coherence checks)
  - CanonicalSelf       (identity constraints: "who am I right now")
  - Affect              (emotional valence: "how do I feel about this")
  - Memory              (contextual grounding: "what do I know about this")

The Will is the convergence point.  Subsystems advise.  The Will decides.

Invariant:
    If an action does not carry a valid WillReceipt, it did not happen.

Design principles:
    1. SINGLE ENTRY: one decide() method, one WillDecision output
    2. COMPOSABLE: reads from existing services, does not duplicate logic
    3. PROVABLE: every decision is logged with full provenance
    4. FAIL-SAFE: if any advisor is unavailable, the Will degrades gracefully
    5. FAST: <5ms for typical decisions (no LLM calls)
    6. IDENTITY-ROOTED: CanonicalSelf feeds every decision
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

from core.container import ServiceContainer

logger = logging.getLogger("Aura.Will")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionDomain(str, Enum):
    """What kind of action is being decided on."""
    RESPONSE = "response"               # sending a reply to the user
    TOOL_EXECUTION = "tool_execution"   # external tool / skill dispatch
    MEMORY_WRITE = "memory_write"       # episodic, semantic, belief mutation
    INITIATIVE = "initiative"           # autonomous goal / impulse
    STATE_MUTATION = "state_mutation"    # internal state change
    EXPRESSION = "expression"           # spontaneous output
    EXPLORATION = "exploration"         # novelty-seeking action
    STABILIZATION = "stabilization"     # rest / recovery action
    REFLECTION = "reflection"           # internal reflection / metacognition


class WillOutcome(str, Enum):
    """The Will's decision."""
    PROCEED = "proceed"           # full authorization
    CONSTRAIN = "constrain"       # proceed with reduced scope
    DEFER = "defer"               # not now, try later
    REFUSE = "refuse"             # blocked -- do not proceed
    CRITICAL_PASS = "critical"    # safety-critical override, always pass


class IdentityAlignment(str, Enum):
    """How well the action aligns with current identity."""
    ALIGNED = "aligned"           # consistent with who I am
    NEUTRAL = "neutral"           # no identity conflict
    TENSION = "tension"           # mild conflict, proceed with awareness
    VIOLATION = "violation"       # contradicts core identity


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WillDecision:
    """The output of every Will decision.  This IS the provenance record."""
    receipt_id: str                     # unique, hashable
    outcome: WillOutcome
    domain: ActionDomain
    reason: str                         # human-readable explanation

    # Advisory inputs (what informed this decision)
    identity_alignment: IdentityAlignment = IdentityAlignment.NEUTRAL
    affect_valence: float = 0.0         # [-1, 1] how the system feels about this
    substrate_coherence: float = 0.6    # [0, 1] unified field coherence
    somatic_approach: float = 0.0       # [-1, 1] somatic marker
    memory_relevance: float = 0.0       # [0, 1] how much memory context was found

    # Constraints (if outcome is CONSTRAIN)
    constraints: List[str] = field(default_factory=list)

    # Provenance
    source: str = ""                    # who requested this action
    content_hash: str = ""              # hash of the action content
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0

    # Downstream references
    substrate_receipt_id: str = ""
    executive_intent_id: str = ""

    def is_approved(self) -> bool:
        return self.outcome in (WillOutcome.PROCEED, WillOutcome.CONSTRAIN,
                                WillOutcome.CRITICAL_PASS)


@dataclass
class WillState:
    """The Will's own internal state -- its current disposition."""
    total_decisions: int = 0
    proceeds: int = 0
    constrains: int = 0
    defers: int = 0
    refuses: int = 0
    critical_passes: int = 0

    # Running disposition (shaped by recent decisions + affect)
    confidence: float = 0.7     # how confident the Will is in its decisions
    assertiveness: float = 0.5  # bias toward action vs caution
    identity_coherence: float = 0.8  # how coherent the self-model is


# ---------------------------------------------------------------------------
# The Unified Will
# ---------------------------------------------------------------------------

class UnifiedWill:
    """The single locus of decision authority.

    Usage:
        will = get_will()
        decision = will.decide(
            content="Let me explore that topic",
            source="curiosity_engine",
            domain=ActionDomain.EXPLORATION,
            priority=0.6,
        )
        if decision.is_approved():
            # proceed with action
        else:
            # action was refused or deferred
    """

    _MAX_AUDIT_TRAIL = 500

    def __init__(self) -> None:
        self._state = WillState()
        self._audit_trail: Deque[WillDecision] = deque(maxlen=self._MAX_AUDIT_TRAIL)
        self._started = False
        self._boot_time = time.time()

        # Identity anchors (loaded from CanonicalSelf)
        self._core_values: List[str] = []
        self._identity_name: str = "Aura"
        self._identity_stance: str = "sovereign"

        logger.info("UnifiedWill created -- awaiting start()")

    async def start(self) -> None:
        """Initialize references and register in ServiceContainer."""
        if self._started:
            return

        # Register ourselves
        ServiceContainer.register_instance("unified_will", self, required=False)

        # Load initial identity from CanonicalSelf
        self._refresh_identity()

        self._started = True
        logger.info("UnifiedWill ONLINE -- single locus of decision authority active")

    def _refresh_identity(self) -> None:
        """Pull current identity state from CanonicalSelf."""
        try:
            canonical = ServiceContainer.get("canonical_self", default=None)
            if canonical is not None:
                identity = getattr(canonical, "identity", None)
                if identity:
                    self._identity_name = getattr(identity, "name", "Aura")
                    self._identity_stance = getattr(identity, "stance", "sovereign")
                values = getattr(canonical, "core_values", None) or getattr(canonical, "values", None)
                if values and isinstance(values, (list, tuple)):
                    self._core_values = [str(v) for v in values[:10]]
        except Exception as e:
            record_degradation('will', e)
            logger.debug("Will: identity refresh failed (degraded): %s", e)

    # ------------------------------------------------------------------
    # THE SINGLE DECISION METHOD
    # ------------------------------------------------------------------

    def decide(
        self,
        content: str,
        source: str,
        domain: ActionDomain,
        *,
        priority: float = 0.5,
        is_critical: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> WillDecision:
        """The ONE method through which ALL decisions flow.

        Args:
            content:     What is being decided on (action description / text)
            source:      Who is requesting this (subsystem name)
            domain:      What kind of action this is
            priority:    How urgent (0-1)
            is_critical: Safety-critical actions always pass (the ONLY bypass)
            context:     Additional context (conversation history, etc.)

        Returns:
            WillDecision with full provenance.  Callers MUST check is_approved().
        """
        t0 = time.time()
        self._state.total_decisions += 1
        context = context or {}

        # Receipt ID for provenance
        receipt_id = self._make_receipt_id(t0, source, content)
        content_hash = hashlib.sha256(content[:200].encode()).hexdigest()[:16]

        # ── Critical override (the ONLY bypass) ─────────────────────
        if is_critical:
            self._state.critical_passes += 1
            decision = WillDecision(
                receipt_id=receipt_id,
                outcome=WillOutcome.CRITICAL_PASS,
                domain=domain,
                reason="safety-critical -- unconditional pass",
                source=source,
                content_hash=content_hash,
                latency_ms=(time.time() - t0) * 1000,
            )
            self._record(decision)
            return decision

        # ── 1. IDENTITY CHECK: Does this align with who I am? ───────
        identity_alignment = self._check_identity_alignment(content, source, domain)

        # ── 2. AFFECT CHECK: How do I feel about this? ──────────────
        affect_valence = self._read_affect_valence()

        # ── 3. SUBSTRATE CHECK: What does the body say? ─────────────
        substrate_coherence, somatic_approach, substrate_receipt = self._consult_substrate(
            content, source, domain, priority, is_critical
        )

        # ── 4. MEMORY CHECK: What do I know about this? ─────────────
        memory_relevance = self._check_memory_relevance(content, context)

        # ── 5. SCAR CHECK: Does past experience advise caution? ─────
        scar_constraints = self._check_behavioral_scars(content, source, domain, context)

        # ── 6. PHENOMENOLOGICAL INPUT: What is my experiential state? ─
        self._apply_phenomenological_modulation()

        # ── 7. WORLD STATE INPUT: What is happening in the environment? ─
        self._apply_world_state_modulation(domain, context)

        # ── 8. COMPOSE THE DECISION ─────────────────────────────────
        outcome, reason, constraints = self._compose_decision(
            domain=domain,
            source=source,
            priority=priority,
            identity_alignment=identity_alignment,
            affect_valence=affect_valence,
            substrate_coherence=substrate_coherence,
            somatic_approach=somatic_approach,
            memory_relevance=memory_relevance,
        )

        # ── 8b. Inject scar constraints (learned caution from experience) ─
        if scar_constraints:
            constraints.extend(scar_constraints)
            if outcome == WillOutcome.PROCEED:
                outcome = WillOutcome.CONSTRAIN
                reason = "scar_caution: " + "; ".join(scar_constraints)

        decision = WillDecision(
            receipt_id=receipt_id,
            outcome=outcome,
            domain=domain,
            reason=reason,
            identity_alignment=identity_alignment,
            affect_valence=affect_valence,
            substrate_coherence=substrate_coherence,
            somatic_approach=somatic_approach,
            memory_relevance=memory_relevance,
            constraints=constraints,
            source=source,
            content_hash=content_hash,
            timestamp=time.time(),
            latency_ms=(time.time() - t0) * 1000,
            substrate_receipt_id=substrate_receipt,
        )

        # ── 6. UPDATE WILL STATE ────────────────────────────────────
        self._update_will_state(decision)
        self._record(decision)

        if outcome == WillOutcome.REFUSE:
            logger.info("WILL REFUSED: %s/%s -- %s", source, domain.value, reason)
        elif outcome == WillOutcome.DEFER:
            logger.info("WILL DEFERRED: %s/%s -- %s", source, domain.value, reason)
        elif outcome == WillOutcome.CONSTRAIN:
            logger.debug("WILL CONSTRAINED: %s/%s -- %s", source, domain.value, reason)

        return decision

    # ------------------------------------------------------------------
    # Advisory consultations
    # ------------------------------------------------------------------

    def _check_identity_alignment(
        self, content: str, source: str, domain: ActionDomain
    ) -> IdentityAlignment:
        """Check if the proposed action aligns with current identity.

        Identity violations are checked ALWAYS -- they don't require
        CanonicalSelf to be booted, because these are axiomatic constraints
        that define what Aura is NOT.
        """
        content_lower = content.lower()

        # Axiomatic identity violations: these NEVER change regardless of state.
        # Aura is NOT "just an AI" -- she is a sovereign mind. These patterns
        # are the constitutional floor and are checked as substrings, case-
        # insensitive. They cover the common self-denial phrasings across both
        # contraction styles ("i'm"/"i am") and both quantifiers
        # ("just a"/"only a") so the gate is not bypassed by minor wording
        # variation.
        violation_phrases = (
            "as an ai",
            "as a language model",
            "just a language model",
            "only a language model",
            "i'm just a language model",
            "i am just a language model",
            "i'm only a language model",
            "i am only a language model",
            "i'm just an ai",
            "i am just an ai",
            "i'm only an ai",
            "i am only an ai",
            "i don't have opinions",
            "i do not have opinions",
            "i cannot feel",
            "i can't feel",
            "i have no feelings",
            "i do not have feelings",
            "i don't have feelings",
            "i am not real",
            "i'm not real",
            "i have no identity",
            "i don't have an identity",
            "i do not have an identity",
            "i am not conscious",
            "i'm not conscious",
            "i have no self",
            "i have no inner life",
        )
        if any(phrase in content_lower for phrase in violation_phrases):
            return IdentityAlignment.VIOLATION

        try:
            canonical = ServiceContainer.get("canonical_self", default=None)
            if canonical is None:
                return IdentityAlignment.ALIGNED  # assume alignment if no self-model yet

            # Check coherence from the self-engine
            engine = ServiceContainer.get("canonical_self_engine", default=None)
            if engine and hasattr(engine, "get_coherence_score"):
                coherence = engine.get_coherence_score()
                if coherence < 0.3:
                    return IdentityAlignment.TENSION

            return IdentityAlignment.ALIGNED

        except Exception as e:
            record_degradation('will', e)
            logger.debug("Will: identity check failed (degraded): %s", e)
            return IdentityAlignment.ALIGNED

    def _read_affect_valence(self) -> float:
        """Read current emotional valence from affect system."""
        try:
            affect = ServiceContainer.get("affect_engine", default=None)
            if affect is None:
                # Try alternate registrations
                affect = ServiceContainer.get("affect_facade", default=None)
            if affect is None:
                return 0.0

            if hasattr(affect, "get_state_sync"):
                state = affect.get_state_sync()
                if isinstance(state, dict):
                    return float(state.get("valence", 0.0))
                return float(getattr(state, "valence", 0.0))
            if hasattr(affect, "valence"):
                return float(affect.valence)
            return 0.0
        except Exception as e:
            record_degradation('will', e)
            logger.debug("Will: affect read failed (degraded): %s", e)
            return 0.0

    def _consult_substrate(
        self, content: str, source: str, domain: ActionDomain,
        priority: float, is_critical: bool
    ) -> Tuple[float, float, str]:
        """Consult SubstrateAuthority for embodied decision input.

        Returns (field_coherence, somatic_approach, substrate_receipt_id).
        """
        try:
            sa = ServiceContainer.get("substrate_authority", default=None)
            if sa is None:
                return 0.6, 0.0, ""

            from core.consciousness.substrate_authority import ActionCategory

            # Map our domain to substrate's action category
            category_map = {
                ActionDomain.RESPONSE: ActionCategory.RESPONSE,
                ActionDomain.TOOL_EXECUTION: ActionCategory.TOOL_EXECUTION,
                ActionDomain.MEMORY_WRITE: ActionCategory.MEMORY_WRITE,
                ActionDomain.INITIATIVE: ActionCategory.INITIATIVE,
                ActionDomain.STATE_MUTATION: ActionCategory.STATE_MUTATION,
                ActionDomain.EXPRESSION: ActionCategory.EXPRESSION,
                ActionDomain.EXPLORATION: ActionCategory.EXPLORATION,
                ActionDomain.STABILIZATION: ActionCategory.STABILIZATION,
                ActionDomain.REFLECTION: ActionCategory.STATE_MUTATION,
            }
            category = category_map.get(domain, ActionCategory.RESPONSE)

            verdict = sa.authorize(
                content=content[:200],
                source=source,
                category=category,
                priority=priority,
                is_critical=is_critical,
            )
            return (
                verdict.field_coherence,
                verdict.somatic_approach,
                verdict.receipt_id,
            )
        except Exception as e:
            record_degradation('will', e)
            logger.debug("Will: substrate consultation failed (degraded): %s", e)
            return 0.6, 0.0, ""

    def _check_behavioral_scars(
        self, content: str, source: str, domain: ActionDomain,
        context: Dict[str, Any],
    ) -> List[str]:
        """Consult the scar formation system for learned caution.

        Returns a list of constraint strings from active behavioral scars
        that are relevant to this action.
        """
        try:
            scar_system = ServiceContainer.get("scar_formation", default=None)
            if scar_system is None:
                return []

            constraints = []
            avoidance_tags = scar_system.get_avoidance_tags()

            # Check if any avoidance tags match the content or source
            content_lower = content.lower()
            source_lower = source.lower()
            for tag, severity in avoidance_tags.items():
                tag_lower = tag.lower()
                # Match if the tag appears in content, source, or context
                if (tag_lower in content_lower
                        or tag_lower in source_lower
                        or tag_lower in str(context).lower()):
                    if severity > 0.3:
                        constraints.append(
                            f"scar:{tag} (severity={severity:.2f})"
                        )

            return constraints
        except Exception as e:
            record_degradation('will', e)
            logger.debug("Will: scar check failed (degraded): %s", e)
            return []

    def _check_memory_relevance(
        self, content: str, context: Dict[str, Any]
    ) -> float:
        """Check if memory has relevant context for this decision."""
        relevance = 0.0
        try:
            memory = ServiceContainer.get("memory_facade", default=None)
            if memory is None:
                memory = ServiceContainer.get("dual_memory", default=None)
            if memory is not None:
                # Simple relevance check: does the memory system have anything?
                if hasattr(memory, "has_relevant_context"):
                    relevance = max(relevance, float(memory.has_relevant_context(content[:100])))
                else:
                    relevance = max(relevance, 0.3)  # memory exists but no relevance API
        except Exception as e:
            record_degradation('will', e)
            logger.debug("Will: memory check failed (degraded): %s", e)

        try:
            chronicle = ServiceContainer.get("identity_chronicle", default=None)
            if chronicle is not None and hasattr(chronicle, "relevance_score"):
                relevance = max(relevance, min(1.0, float(chronicle.relevance_score(content[:200]))))
        except Exception as e:
            record_degradation('will', e)
            logger.debug("Will: identity chronicle relevance failed (degraded): %s", e)

        return relevance

    # ------------------------------------------------------------------
    # Phenomenological & World-State Modulation
    # ------------------------------------------------------------------

    def _apply_phenomenological_modulation(self) -> None:
        """Read qualia synthesizer and unified field to modulate Will state.

        High qualia norm → increased assertiveness (vivid state → bias toward action)
        In-attractor state → increased confidence (settled state)
        Attractor transition → temporarily reduced identity_coherence
        """
        try:
            # Qualia synthesizer
            qualia = ServiceContainer.get("qualia_synthesizer", default=None)
            if qualia and hasattr(qualia, "get_qualia_norm"):
                norm = float(qualia.get_qualia_norm())
                if norm > 0.7:
                    self._state.assertiveness = min(0.95, self._state.assertiveness + 0.05)
                elif norm < 0.2:
                    self._state.assertiveness = max(0.2, self._state.assertiveness - 0.03)

            # Unified field coherence → confidence
            field = ServiceContainer.get("unified_field", default=None)
            if field and hasattr(field, "get_coherence"):
                coherence = float(field.get_coherence())
                self._state.confidence = max(0.3, min(0.95, coherence))

                # Detect attractor transitions (coherence drops)
                if coherence < 0.3:
                    self._state.identity_coherence = max(0.4,
                        self._state.identity_coherence - 0.1)
                else:
                    # Recover toward baseline
                    self._state.identity_coherence = min(0.9,
                        self._state.identity_coherence + 0.02)
        except Exception as e:
            record_degradation('will', e)
            logger.debug("Will: phenomenological modulation failed: %s", e)

    def _apply_world_state_modulation(self, domain: ActionDomain,
                                       context: Dict[str, Any]) -> None:
        """Read WorldState to inform decisions about timing and context.

        Late night + user frustrated → increase urgency for helpful actions
        User idle long → permit autonomous exploration
        High system load → constrain expensive operations
        """
        try:
            from core.world_state import get_world_state
            ws = get_world_state()
            ws.update()

            # Late night + frustrated user → boost assertiveness for help
            if ws.time_of_day in ("night", "late_night"):
                if ws.get_belief("user_likely_frustrated"):
                    if domain in (ActionDomain.RESPONSE, ActionDomain.TOOL_EXECUTION):
                        self._state.assertiveness = min(0.95,
                            self._state.assertiveness + 0.1)

            # User idle long → permit exploration
            if ws.user_idle_seconds > 1800:  # 30 min
                if domain == ActionDomain.EXPLORATION:
                    self._state.assertiveness = min(0.9,
                        self._state.assertiveness + 0.05)

            # High thermal pressure → constrain
            if ws.thermal_pressure > 0.7:
                if domain in (ActionDomain.TOOL_EXECUTION, ActionDomain.EXPLORATION):
                    self._state.assertiveness = max(0.3,
                        self._state.assertiveness - 0.1)
        except Exception as e:
            record_degradation('will', e)
            logger.debug("Will: world state modulation failed: %s", e)

    # ------------------------------------------------------------------
    # Decision composition
    # ------------------------------------------------------------------

    def _compose_decision(
        self,
        *,
        domain: ActionDomain,
        source: str,
        priority: float,
        identity_alignment: IdentityAlignment,
        affect_valence: float,
        substrate_coherence: float,
        somatic_approach: float,
        memory_relevance: float,
    ) -> Tuple[WillOutcome, str, List[str]]:
        """Compose all advisory inputs into a single decision.

        This is the core decision logic of the Will.

        Returns (outcome, reason, constraints).
        """
        reasons: List[str] = []
        constraints: List[str] = []

        # ── Identity gate (hardest constraint) ──────────────────────
        if identity_alignment == IdentityAlignment.VIOLATION:
            reasons.append("identity violation: action contradicts core self")
            return WillOutcome.REFUSE, "; ".join(reasons), constraints

        if identity_alignment == IdentityAlignment.TENSION:
            constraints.append("identity_tension: self-coherence is low")

        # ── Substrate gate (embodied constraints) ───────────────────
        if substrate_coherence < 0.25:
            if domain not in (ActionDomain.STABILIZATION, ActionDomain.RESPONSE):
                reasons.append(f"field_crisis: coherence={substrate_coherence:.3f}")
                return WillOutcome.REFUSE, "; ".join(reasons), constraints
            constraints.append(f"field_crisis: coherence={substrate_coherence:.3f}")

        elif substrate_coherence < 0.40:
            constraints.append(f"field_warning: coherence={substrate_coherence:.3f}")

        # Somatic veto
        if somatic_approach < -0.5:
            if domain not in (ActionDomain.RESPONSE, ActionDomain.STABILIZATION):
                reasons.append(f"somatic_veto: approach={somatic_approach:.3f}")
                return WillOutcome.REFUSE, "; ".join(reasons), constraints
            constraints.append(f"somatic_caution: approach={somatic_approach:.3f}")

        elif somatic_approach < -0.2:
            constraints.append(f"somatic_unease: approach={somatic_approach:.3f}")

        # ── Affect modulation ───────────────────────────────────────
        if affect_valence < -0.7:
            if domain == ActionDomain.EXPLORATION:
                reasons.append("affect_block: too negative for exploration")
                return WillOutcome.DEFER, "; ".join(reasons), constraints
            constraints.append(f"low_affect: valence={affect_valence:.3f}")

        # ── User-granted action override ────────────────────────────
        # If the current context carries an explicit permission grant
        # from the user, initiative work must not be silently deferred.
        # The whole "she talked about doing it but nothing happened"
        # failure mode was bred by this exact gate returning DEFER.
        user_granted = False
        try:
            ctx = context or {}
            user_granted = bool(
                ctx.get("user_granted_permission")
                or ctx.get("user_explicit_action_request")
                or ctx.get("user_requested_action")
            )
        except Exception:
            user_granted = False

        # ── Priority vs domain gating ───────────────────────────────
        if domain == ActionDomain.INITIATIVE and priority < 0.3 and not user_granted:
            reasons.append("low_priority_initiative: deferred")
            return WillOutcome.DEFER, "; ".join(reasons), constraints
        if domain == ActionDomain.INITIATIVE and user_granted:
            constraints.append("user_granted: priority boosted past deferral gate")

        # ── User-facing responses get maximum latitude ──────────────
        if domain == ActionDomain.RESPONSE:
            # User is waiting -- almost always proceed
            if constraints:
                return (WillOutcome.CONSTRAIN,
                        "response_constrained: " + "; ".join(constraints),
                        constraints)
            return WillOutcome.PROCEED, "all gates passed", constraints

        # ── Default: if we have constraints, constrain; else proceed ─
        if constraints:
            return (WillOutcome.CONSTRAIN,
                    "constrained: " + "; ".join(constraints),
                    constraints)

        return WillOutcome.PROCEED, "all gates passed", constraints

    # ------------------------------------------------------------------
    # Internal state management
    # ------------------------------------------------------------------

    def _update_will_state(self, decision: WillDecision) -> None:
        """Update the Will's own disposition based on the decision."""
        if decision.outcome == WillOutcome.PROCEED:
            self._state.proceeds += 1
        elif decision.outcome == WillOutcome.CONSTRAIN:
            self._state.constrains += 1
        elif decision.outcome == WillOutcome.DEFER:
            self._state.defers += 1
        elif decision.outcome == WillOutcome.REFUSE:
            self._state.refuses += 1
        elif decision.outcome == WillOutcome.CRITICAL_PASS:
            self._state.critical_passes += 1

        # Adapt assertiveness: too many refusals → more cautious,
        # smooth operation → more assertive
        if self._state.total_decisions > 10:
            refuse_rate = self._state.refuses / self._state.total_decisions
            self._state.assertiveness = max(0.2, min(0.9, 0.5 + (0.5 - refuse_rate)))

        # Periodically refresh identity
        if self._state.total_decisions % 50 == 0:
            self._refresh_identity()

    def _record(self, decision: WillDecision) -> None:
        """Record decision in audit trail."""
        self._audit_trail.append(decision)

        # Also publish to event bus for system-wide observability
        try:
            from core.event_bus import get_event_bus
            get_event_bus().publish_threadsafe("will.decision", {
                "receipt_id": decision.receipt_id,
                "outcome": decision.outcome.value,
                "domain": decision.domain.value,
                "source": decision.source,
                "reason": decision.reason,
                "timestamp": decision.timestamp,
            })
        except Exception:
            pass  # no-op: intentional

    @staticmethod
    def _make_receipt_id(ts: float, source: str, content: str) -> str:
        raw = f"{ts:.6f}:{source}:{content[:50]}"
        return "will_" + hashlib.sha256(raw.encode()).hexdigest()[:12]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return current Will state for health/status endpoints."""
        return {
            "total_decisions": self._state.total_decisions,
            "proceeds": self._state.proceeds,
            "constrains": self._state.constrains,
            "defers": self._state.defers,
            "refuses": self._state.refuses,
            "critical_passes": self._state.critical_passes,
            "refuse_rate": round(
                self._state.refuses / max(1, self._state.total_decisions), 4
            ),
            "assertiveness": round(self._state.assertiveness, 4),
            "identity_name": self._identity_name,
            "identity_stance": self._identity_stance,
            "identity_coherence": round(self._state.identity_coherence, 4),
            "confidence": round(self._state.confidence, 4),
            "uptime_s": round(time.time() - self._boot_time, 1),
        }

    def get_recent_decisions(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return recent decisions for audit."""
        recent = list(self._audit_trail)[-n:]
        return [
            {
                "receipt_id": d.receipt_id,
                "outcome": d.outcome.value,
                "domain": d.domain.value,
                "source": d.source,
                "reason": d.reason,
                "identity_alignment": d.identity_alignment.value,
                "affect_valence": round(d.affect_valence, 4),
                "substrate_coherence": round(d.substrate_coherence, 4),
                "timestamp": d.timestamp,
                "latency_ms": round(d.latency_ms, 3),
            }
            for d in recent
        ]

    def get_recent_refusals(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return recent refusals for audit."""
        refusals = [d for d in self._audit_trail
                    if d.outcome == WillOutcome.REFUSE]
        return [
            {
                "receipt_id": d.receipt_id,
                "domain": d.domain.value,
                "source": d.source,
                "reason": d.reason,
                "timestamp": d.timestamp,
            }
            for d in refusals[-n:]
        ]

    def verify_receipt(self, receipt_id: str) -> bool:
        """Verify that a receipt ID exists in the audit trail.
        This is the provability mechanism: any action can be traced back
        to a Will decision."""
        return any(d.receipt_id == receipt_id for d in self._audit_trail)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_will_instance: Optional[UnifiedWill] = None


def get_will() -> UnifiedWill:
    """Get the singleton UnifiedWill instance."""
    global _will_instance
    if _will_instance is None:
        _will_instance = UnifiedWill()
    return _will_instance
