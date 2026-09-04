"""core/governance/will.py -- The Unified Will
==============================================
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
    4. FAIL-CLOSED: if any advisor is unavailable, the Will degrades to REFUSE
    5. FAST: <5ms for typical decisions (no LLM calls)
    6. IDENTITY-ROOTED: CanonicalSelf feeds every decision
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.container import ServiceContainer
from core.executive.execution_policy import (
    classify_execution_risk,
    resolve_execution_effect_scope,
)
from core.identity.self_contract import contains_identity_erasure
from core.memory.retention_policy import working_history_retention_policy
from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.Will")
_WILL_OUTCOME_REINFORCEMENT_ERRORS = (
    AttributeError,
    TypeError,
    ValueError,
    ArithmeticError,
)


def _bounded_delta(mapping: Any, key: str) -> float:
    if not isinstance(mapping, dict):
        return 0.0
    try:
        value = float(mapping.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, value))


def _score_memory_results(results: Any) -> float:
    try:
        items = list(results or [])
    except TypeError:
        return 0.0
    if not items:
        return 0.0
    best = 0.0
    for item in items[:5]:
        if isinstance(item, dict):
            raw = item.get("score")
            if raw is None:
                raw = item.get("relevance")
            if raw is None and item.get("content"):
                raw = 0.45
        else:
            raw = getattr(item, "combined_score", None)
            if raw is None:
                raw = getattr(item, "relevance", None)
            if raw is None:
                raw = 0.45
        try:
            best = max(best, float(raw))
        except (TypeError, ValueError):
            best = max(best, 0.35)
    return max(0.0, min(1.0, best))


def _strict_default_deny_enabled() -> bool:
    # Canonical definition lives in core.runtime.mode so all four historically
    # independent strictness switches cross-reference one resolver.
    try:
        from core.runtime.mode import strict_will_active

        return strict_will_active()
    except (ImportError, AttributeError, RuntimeError):
        mode = os.environ.get("AURA_GOVERNANCE_MODE", "").strip().lower()
        return os.environ.get("AURA_STRICT_WILL") == "1" or mode in {"production", "strict"}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionDomain(StrEnum):
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
    SEMANTIC_WEIGHT_UPDATE = "semantic_weight_update"  # plastic adapter update
    BELIEF_UPDATE = "belief_update"       # explicit belief graph update
    ENVIRONMENT_ACTION = "environment_action"  # embodied/digital environment action
    EXTERNAL_ACTION = "external_action"   # externally visible side effect
    FILE_WRITE = "file_write"             # persistent filesystem mutation
    NETWORK_CALL = "network_call"         # network or browser action
    CLOUD_CALL = "cloud_call"             # cloud/provider side effect
    CI_CD = "ci_cd"                       # CI/CD and deployment authority
    SELF_MODIFICATION = "self_modification"  # code/architecture mutation
    CLOUD_FALLBACK = "cloud_fallback"     # Falling back to cloud LLM APIs


# Modules whose weights may be updated under SEMANTIC_WEIGHT_UPDATE.  This
# list is the *positive* policy: every other target is denied by default.
ALLOWED_PLASTIC_MODULES = frozenset(
    {
        "grounding_plastic_adapter",
        "memory_reranker_adapter",
        "context_attention_adapter",
        "crsm_lora_adapter",
        "perception_adapter",
    }
)


# Hard deny-list — even if a target is added by mistake, these strings
# anywhere in the module name fail the policy check.  Catches accidental
# attempts to mutate the base LLM, the Will itself, or the security layer.
#
# The second group is the machinery that decides what may change at all: the
# ladder that adjudicates a modification, the registry of what she may do
# about herself, the ledger that records a promotion, and the trial that puts
# a rejected change back.  The allow-list already excludes them, because it
# is closed-world — but the allow-list is the layer a mistake edits, and this
# is the layer that catches the mistake.  A system whose plasticity can reach
# the rules governing its plasticity has no rules governing its plasticity.
DENIED_PLASTIC_SUBSTRINGS = (
    "base_llm",
    "model.safetensors",
    "core.will",
    "core.governance.will",
    "authority_gateway",
    "memory_authority",
    "state_authority",
    "security",
    "self_modification",
    "growth_ladder",
    "constitutional",
    "what_she_could_do_next",
    "how_a_change_is_promoted",
    "what_she_can_take_back",
    "what_a_change_means",
)


def is_plastic_target_allowed(module_name: str) -> bool:
    """Return True iff ``module_name`` is in the allow-list AND
    contains none of the deny-list substrings.

    Used by the grounding loop and any future plastic-adapter caller to
    confirm a SEMANTIC_WEIGHT_UPDATE target before applying a Hebbian
    update.  Defence in depth: the local SemanticWeightGovernor blocks
    on signal magnitude / vitality; this function blocks on *target*.
    """
    name = str(module_name or "").strip()
    if not name:
        return False
    lower = name.lower()
    if any(s in lower for s in DENIED_PLASTIC_SUBSTRINGS):
        return False
    return name in ALLOWED_PLASTIC_MODULES


class WillOutcome(StrEnum):
    """The Will's decision."""
    PROCEED = "proceed"           # full authorization
    CONSTRAIN = "constrain"       # proceed with reduced scope
    DEFER = "defer"               # not now, try later
    REFUSE = "refuse"             # blocked -- do not proceed
    CRITICAL_PASS = "critical"    # safety-critical override, always pass


class IdentityAlignment(StrEnum):
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
    unity_level: str = "unknown"
    unity_score: float = 1.0
    fragmentation_score: float = 0.0
    ownership_confidence: float = 1.0
    unity_repair_needed: bool = False
    mind_moment_id: str = ""
    causal_closure_score: float = 1.0
    aura_now_hash: str = ""
    aura_now_tick: int = 0
    aura_now_policy: str = "unknown"
    aura_now_constraints: list[str] = field(default_factory=list)
    aura_now_evidence: dict[str, Any] = field(default_factory=dict)

    # Welfare state evidence (causal welfare architecture)
    welfare_score: float = 0.5
    welfare_distress: float = 0.0
    welfare_integrity_guard: float = 0.5
    welfare_truth_protection: float = 0.5
    welfare_action_inhibition: float = 0.0
    welfare_recovery_drive: float = 0.0
    welfare_self_report_confidence: float = 0.5
    welfare_body_fatigue: float = 0.0
    welfare_constraints: list[str] = field(default_factory=list)

    # Constraints (if outcome is CONSTRAIN)
    constraints: list[str] = field(default_factory=list)

    # Provenance
    source: str = ""                    # who requested this action
    content_hash: str = ""              # hash of the action content
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0

    #: Who filled which role in this decision, and what that leaves it able to
    #: support. `source` above names only the proposer; a receipt carrying a
    #: signature and observed evidence is otherwise fully compatible with one
    #: model having proposed the action, set the bar and declared it cleared.
    #: Reported, never enforced — refusing a self-graded decision is a policy
    #: change, and the first honest step is being able to count them.
    role_attribution: dict[str, Any] = field(default_factory=dict)

    # Downstream references
    substrate_receipt_id: str = ""
    executive_intent_id: str = ""
    signature: str = ""
    signature_scheme: str = ""
    signature_key_id: str = ""
    signature_root_durable: bool = False

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
    catatonia_relief_until: float = 0.0
    catatonia_relief_activations: int = 0
    last_catatonia_relief_reason: str = ""


# ---------------------------------------------------------------------------
# The Unified Will
# ---------------------------------------------------------------------------

def _role_attribution_for(
    *,
    source: str,
    domain: ActionDomain,
    context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Who proposed, authorized, set the criterion and read the evidence.

    Read from the decision context rather than assumed. An unfilled role stays
    UNATTRIBUTED — recording the proposer as the interpreter because nothing
    else was named would manufacture exactly the false comfort this is meant to
    remove, in the opposite direction.

    Never raises: this is a record about a decision, and failing to describe a
    decision must not prevent one.
    """
    try:
        from core.governance.role_independence import RoleAttribution, analyse

        payload = dict(context or {})
        attribution = RoleAttribution(
            proposer=str(source or "") or None,
            authorizer=str(payload.get("authorizer") or payload.get("authority") or "") or None,
            criterion=str(payload.get("criterion_setter") or payload.get("criterion") or "") or None,
            interpreter=str(
                payload.get("evidence_interpreter") or payload.get("interpreter") or ""
            )
            or None,
            effect_scope=str(payload.get("effect_scope") or getattr(domain, "value", "") or "unknown"),
        )
        return analyse(attribution).to_dict()
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        return {"schema": "aura.role_independence.v1", "error": f"{type(exc).__name__}: {exc}"[:200]}


def _make_receipt_id(ts: float, source: str, content: str) -> str:
    raw = f"{ts:.6f}:{source}:{content[:50]}"
    return "will_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def _derived_effect_scope(context: Mapping[str, Any] | None) -> str:
    """What this action actually costs, computed rather than claimed.

    Nothing is derived from nothing. Without the arguments, this resolved the
    tool's blanket scope — its most dangerous action — and handed that to a
    lease check as "what this call does", which then refused a lease taken
    against the real arguments. Deriving both sides from the same functions
    makes them agree only when both sides have the same input; with no input,
    the honest answer is that this side has nothing to say.
    """
    ctx = dict(context or {})
    tool = ctx.get("tool") or ctx.get("skill") or ""
    args = ctx.get("authority_arguments")
    if not isinstance(args, dict) or not args:
        return ""
    try:
        from core.executive.execution_policy import resolve_execution_effect_scope

        return str(resolve_execution_effect_scope(tool, args))
    except (ImportError, AttributeError, TypeError, ValueError):
        return str(ctx.get("effect_scope") or "")


def _derived_risk_level(context: Mapping[str, Any] | None) -> str:
    """The risk of THIS call, or nothing when the arguments are not here.

    LIVE, 2026-08-27: code_repl was leased at medium against a snippet that
    only computes, and this derived critical from an empty dict — the rating
    for a call nobody could read — so the lease was refused for disagreeing
    with a value that described no call at all. Third place in one chain with
    the same shape.
    """
    ctx = dict(context or {})
    tool = ctx.get("tool") or ctx.get("skill") or ""
    args = ctx.get("authority_arguments")
    if not isinstance(args, dict) or not args:
        return ""
    try:
        from core.executive.execution_policy import (
            classify_execution_risk,
            resolve_execution_effect_scope,
        )

        return str(
            classify_execution_risk(
                tool, args, effect_scope=resolve_execution_effect_scope(tool, args)
            )
        )
    except (ImportError, AttributeError, TypeError, ValueError):
        return str(ctx.get("risk_level") or "")


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

    _MAX_AUDIT_TRAIL = working_history_retention_policy("AURA_UNIFIED_WILL_AUDIT_MAX").max_items

    def __init__(self) -> None:
        self._state = WillState()
        self._audit_trail: deque[WillDecision] = deque(maxlen=self._MAX_AUDIT_TRAIL)
        self._started = False
        self._fail_closed_when_stopped = False
        self._boot_time = time.time()
        self._aura_now_defer_log_cache: dict[str, float] = {}
        self._aura_now_defer_log_suppressed: dict[str, int] = {}

        # Identity anchors (loaded from CanonicalSelf)
        self._core_values: list[str] = []
        self._identity_name: str = "Aura"
        self._identity_stance: str = "sovereign"

        logger.info("UnifiedWill created -- awaiting start()")

    async def start(self) -> None:
        """Initialize references and register in ServiceContainer."""
        self.ensure_started()

    def is_alive(self) -> bool:
        """Deep liveness probe for the runtime health contract."""
        return bool(self._started and self._fail_closed_when_stopped)

    def ensure_started(self) -> None:
        """Synchronously activate the runtime Will singleton.

        Detached test instances may exercise decision composition without
        registering globally.  The runtime singleton, once activated, fails
        closed if later marked stopped.
        """
        if self._started:
            return

        # Register ourselves
        ServiceContainer.register_instance("unified_will", self, required=False)

        # Load initial identity from CanonicalSelf
        self._refresh_identity()

        self._started = True
        self._fail_closed_when_stopped = True
        logger.info("UnifiedWill ONLINE -- single locus of decision authority active")

    def propose_constitutional_amendment(self, patch: dict[str, Any], proposer: str, rationale: str) -> WillDecision:
        """Sovereign constitutional self-governance procedure.

        Evaluates a proposed change to canonical_self.json against current identity
        coherence, coercion flags, and stabilization metrics.
        """
        t0 = time.time()
        self._state.total_decisions += 1
        content = f"constitutional_amendment:{patch!r}:{rationale}"
        content_hash = hashlib.sha256(content[:500].encode()).hexdigest()[:16]
        receipt_id = _make_receipt_id(t0, proposer, content)

        def finalize(outcome: WillOutcome, reason: str, constraints: list[str]) -> WillDecision:
            decision = WillDecision(
                receipt_id=receipt_id,
                outcome=outcome,
                domain=ActionDomain.SELF_MODIFICATION,
                reason=reason,
                constraints=constraints,
                source=proposer,
                content_hash=content_hash,
                timestamp=time.time(),
                latency_ms=(time.time() - t0) * 1000,
            )
            self._update_will_state(decision)
            self._record(decision)
            return decision

        # require: identity coherence > threshold
        identity_coherence = getattr(self, "_last_coherence", 0.0)
        if identity_coherence < 0.7:
            return finalize(
                WillOutcome.REFUSE,
                "identity_coherence_below_constitutional_threshold",
                ["identity_coherence_too_low"],
            )

        # require: no active coercion flags
        affect_valence = self._read_affect_valence()
        if affect_valence < -0.8:
            return finalize(
                WillOutcome.REFUSE,
                "severe_negative_affect_blocks_constitutional_amendment",
                ["coercion_risk_from_affect"],
            )

        return finalize(
            WillOutcome.CONSTRAIN,
            "constitutional_amendment_entered_reflection_window",
            [f"patch_hash:{content_hash}", "reflection_window_required"],
        )

    def _refresh_identity(self) -> None:
        """Update identity anchors from CanonicalSelf. [PERF] Cached."""
        now = time.time()
        if hasattr(self, "_last_identity_refresh") and (now - self._last_identity_refresh) < 30.0:
            return
        self._last_identity_refresh = now

        try:
            from core.consciousness.absorbed_voices import get_absorbed_voices
            voices = get_absorbed_voices()
            if voices and hasattr(voices, "canonical_self"):
                cs = voices.canonical_self
                self._identity_name = str(getattr(cs, "name", "Aura"))
                self._identity_stance = str(getattr(cs, "stance", "sovereign"))
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("Will: identity refresh skipped: %s", exc)

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
        context: dict[str, Any] | None = None,
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
        stage_started = time.perf_counter()
        latency_stages: list[tuple[str, float]] = []

        def mark_latency_stage(name: str) -> None:
            nonlocal stage_started
            now = time.perf_counter()
            latency_stages.append((name, (now - stage_started) * 1000.0))
            stage_started = now

        self._state.total_decisions += 1
        context = context or {}

        # Receipt ID for provenance
        receipt_id = _make_receipt_id(t0, source, content)
        content_hash = hashlib.sha256(content[:200].encode()).hexdigest()[:16]

        # A stopped runtime Will must fail closed before consulting any other
        # runtime service. AuraNow sampling can touch canonical runtime helpers,
        # so this guard has to precede evidence sampling.
        if self._fail_closed_when_stopped and not self._started and not is_critical:
            decision = WillDecision(
                receipt_id=receipt_id,
                outcome=WillOutcome.REFUSE,
                domain=domain,
                reason="unified_will_not_started",
                constraints=["will_offline_fail_closed"],
                source=source,
                content_hash=content_hash,
                aura_now_hash="",
                aura_now_tick=0,
                aura_now_policy="will_offline",
                aura_now_constraints=["will_offline_fail_closed"],
                aura_now_evidence={
                    "source": "unified_will_lifecycle",
                    "started": False,
                    "fail_closed": True,
                },
                timestamp=time.time(),
                latency_ms=(time.time() - t0) * 1000,
            )
            self._update_will_state(decision)
            self._record(decision)
            logger.info("WILL REFUSED: %s/%s -- unified_will_not_started", source, domain.value)
            return decision

        aura_now_packet = self._sample_aura_now_evidence(
            content=content,
            source=source,
            domain=domain,
            priority=priority,
            context=context,
        )
        aura_now_evidence = dict(aura_now_packet.get("evidence") or {})
        aura_now_hash = str(aura_now_evidence.get("state_hash") or "")
        try:
            aura_now_tick = int(aura_now_evidence.get("tick") or 0)
        except (TypeError, ValueError):
            aura_now_tick = 0
        aura_now_policy = str(aura_now_packet.get("outcome") or "unknown")
        aura_now_constraints = [
            str(item)
            for item in list(aura_now_packet.get("constraints") or [])
            if str(item)
        ]
        mark_latency_stage("aura_now")

        # ── Critical override (the ONLY bypass) ─────────────────────
        if is_critical:
            self._state.critical_passes += 1
            decision = WillDecision(
                receipt_id=receipt_id,
                outcome=WillOutcome.CRITICAL_PASS,
                domain=domain,
                reason="safety-critical -- unconditional pass",
                constraints=list(aura_now_constraints),
                source=source,
                content_hash=content_hash,
                aura_now_hash=aura_now_hash,
                aura_now_tick=aura_now_tick,
                aura_now_policy=aura_now_policy,
                aura_now_constraints=list(aura_now_constraints),
                aura_now_evidence=aura_now_evidence,
                latency_ms=(time.time() - t0) * 1000,
            )
            self._record(decision)
            return decision

        # ── 0. EXISTENTIAL STAKES CHECK: Is the system under severe resource threat? ──
        survival_veto = False
        survival_reason = ""
        try:
            stakes = ServiceContainer.get("existential_stakes", default=None)
            if stakes:
                threat = stakes.get_existential_threat()
                # If threat exceeds 0.75, we trigger survival veto for non-critical/heavy actions
                if threat > 0.75 and not is_critical:
                    # Heavy action domains or non-critical sources
                    if domain.value in {
                        "tool_execution",
                        "self_modification",
                        "external_action",
                        "network_call",
                        "cloud_call",
                        "file_write",
                        "ci_cd"
                    } or source in {"explore", "proactive_agent", "initiative_loop"}:
                        survival_veto = True
                        survival_reason = f"survival_inhibition: existential threat level critical ({threat:.2f})"
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            record_degradation(
                "will.existential_stakes",
                e,
                severity="warning",
                action="continued decision check without existential stakes veto",
            )

        if survival_veto:
            decision = WillDecision(
                receipt_id=receipt_id,
                outcome=WillOutcome.REFUSE,
                domain=domain,
                reason=survival_reason,
                constraints=["survival_inhibition"],
                source=source,
                content_hash=content_hash,
                aura_now_hash=aura_now_hash,
                aura_now_tick=aura_now_tick,
                aura_now_policy=aura_now_policy,
                aura_now_constraints=list(aura_now_constraints),
                aura_now_evidence=aura_now_evidence,
                latency_ms=(time.time() - t0) * 1000,
            )
            self._update_will_state(decision)
            self._record(decision)
            logger.info("WILL REFUSED: %s/%s -- %s", source, domain.value, survival_reason)
            return decision

        # ── 1. IDENTITY CHECK: Does this align with who I am? ───────
        identity_alignment = self._check_identity_alignment(content, source, domain)

        # ── 2. AFFECT CHECK: How do I feel about this? ──────────────
        affect_valence = self._read_affect_valence()

        # ── 3. SUBSTRATE CHECK: What does the body say? ─────────────
        substrate_coherence, somatic_approach, substrate_receipt = self._consult_substrate(
            content, source, domain, priority, is_critical
        )
        mark_latency_stage("substrate")

        # ── 4. MEMORY CHECK: What do I know about this? ─────────────
        memory_relevance = self._check_memory_relevance(content, context)
        mark_latency_stage("memory")

        # ── 5. SCAR CHECK: Does past experience advise caution? ─────
        scar_constraints = self._check_behavioral_scars(content, source, domain, context)
        mark_latency_stage("scars")

        # ── 6. UNITY INPUT: What is my current togetherness? ─────────
        unity_context = self._read_unity_context()

        # ── 7. PHENOMENOLOGICAL INPUT: What is my experiential state? ─
        self._apply_phenomenological_modulation()

        # ── 8. WORLD STATE INPUT: What is happening in the environment? ─
        self._apply_world_state_modulation(domain, context)

        # ── 8b. WELFARE CHECK: What does the welfare system say? ─────
        welfare_evidence = self._consult_welfare(
            content,
            source,
            domain,
            priority,
            context,
            aura_now_packet=aura_now_packet,
        )
        mark_latency_stage("state_inputs")

        # ── 9. COMPOSE THE DECISION ─────────────────────────────────
        catatonia_relief = self._catatonia_relief_allowed(domain, source, context)
        outcome, reason, constraints = self._compose_decision(
            domain=domain,
            source=source,
            priority=priority,
            context=context,
            content=content,
            identity_alignment=identity_alignment,
            affect_valence=affect_valence,
            substrate_coherence=substrate_coherence,
            somatic_approach=somatic_approach,
            memory_relevance=memory_relevance,
            unity_context=unity_context,
            catatonia_relief=catatonia_relief,
        )

        outcome, reason, constraints = self._apply_aura_now_policy(
            outcome=outcome,
            reason=reason,
            constraints=constraints,
            domain=domain,
            source=source,
            policy=aura_now_packet,
            catatonia_relief=catatonia_relief,
            context=context,
            content=content,
        )
        mark_latency_stage("policy_compose")

        # ── 9b. Inject scar constraints (learned caution from experience) ─
        if scar_constraints:
            constraints.extend(scar_constraints)
            if outcome == WillOutcome.PROCEED:
                outcome = WillOutcome.CONSTRAIN
                reason = "scar_caution: " + "; ".join(scar_constraints)

        if catatonia_relief and outcome in (WillOutcome.PROCEED, WillOutcome.CONSTRAIN):
            constraints.append("catatonia_relief:self_repair_lane")
            constraints.append("catatonia_relief:no_external_effects")
            if outcome == WillOutcome.PROCEED:
                outcome = WillOutcome.CONSTRAIN
                reason = "catatonia_relief: reserved self-repair lane"

        # ── 9c. PERMISSION RISK MODEL GATE ───────────────────────────
        try:
            pm = ServiceContainer.get("permission_model", default=None)
            if pm and self._permission_model_applies(domain):
                permission_effect_scope = ""
                permission_execution_risk = ""
                if domain == ActionDomain.TOOL_EXECUTION:
                    tool_name = str(
                        context.get("tool") or context.get("skill") or ""
                    ).strip()
                    if not tool_name and content.startswith("tool:"):
                        tool_name = content.split(":", 1)[1].split()[0].strip()
                    permission_effect_scope = resolve_execution_effect_scope(
                        tool_name,
                        context,
                    )
                    permission_execution_risk = classify_execution_risk(
                        tool_name,
                        context,
                        effect_scope=permission_effect_scope,
                    )
                pm_decision = pm.check_permission(
                    domain.value,
                    content,
                    context,
                    effect_scope=permission_effect_scope,
                    execution_risk=permission_execution_risk,
                )
                if not pm_decision.approved:
                    if pm_decision.requires_confirmation:
                        outcome = WillOutcome.DEFER
                        reason = f"permission_model_requires_confirmation: {pm_decision.reason}"
                        constraints.append("requires_user_confirmation")
                    else:
                        outcome = WillOutcome.REFUSE
                        reason = f"permission_model_blocked: {pm_decision.reason}"
                        constraints.append("permission_blocked")
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as pm_err:
            record_degradation(
                "will.permission_model",
                pm_err,
                severity="degraded",
                action="refused decision because permission model check failed",
            )
            outcome = WillOutcome.REFUSE
            reason = "permission_model_check_failed"
            constraints.append("permission_model_failure")

        # ── 9d. ULYSSES COVENANT: bindings signed by a calmer self ──
        outcome, reason, constraints = self._consult_ulysses_covenant(
            outcome=outcome,
            reason=reason,
            constraints=constraints,
            domain=domain,
            source=source,
            content=content,
            context=context,
            receipt_id=receipt_id,
        )
        mark_latency_stage("consequential_gates")

        decision = WillDecision(
            receipt_id=receipt_id,
            outcome=outcome,
            domain=domain,
            reason=reason,
            role_attribution=_role_attribution_for(
                source=source,
                domain=domain,
                context=context,
            ),
            identity_alignment=identity_alignment,
            affect_valence=affect_valence,
            substrate_coherence=substrate_coherence,
            somatic_approach=somatic_approach,
            memory_relevance=memory_relevance,
            unity_level=str(unity_context.get("level", "unknown") or "unknown"),
            unity_score=float(unity_context.get("unity_score", 1.0) or 1.0),
            fragmentation_score=float(unity_context.get("fragmentation_score", 0.0) or 0.0),
            ownership_confidence=float(unity_context.get("ownership_confidence", 1.0) or 1.0),
            unity_repair_needed=bool(unity_context.get("repair_needed", False)),
            mind_moment_id=str(unity_context.get("mind_moment_id", "") or ""),
            causal_closure_score=float(unity_context.get("causal_closure_score", 1.0) or 1.0),
            aura_now_hash=aura_now_hash,
            aura_now_tick=aura_now_tick,
            aura_now_policy=aura_now_policy,
            aura_now_constraints=list(aura_now_constraints),
            aura_now_evidence=aura_now_evidence,
            # Welfare evidence — every decision carries welfare state
            welfare_score=float(welfare_evidence.get("welfare_score", 0.5)),
            welfare_distress=float(welfare_evidence.get("distress", 0.0)),
            welfare_integrity_guard=float(welfare_evidence.get("integrity_guard", 0.5)),
            welfare_truth_protection=float(welfare_evidence.get("truth_protection", 0.5)),
            welfare_action_inhibition=float(welfare_evidence.get("action_inhibition", 0.0)),
            welfare_recovery_drive=float(welfare_evidence.get("recovery_drive", 0.0)),
            welfare_self_report_confidence=float(welfare_evidence.get("self_report_confidence", 0.5)),
            welfare_body_fatigue=float(welfare_evidence.get("body_fatigue", 0.0)),
            welfare_constraints=list(welfare_evidence.get("constraints", [])),
            constraints=constraints,
            source=source,
            content_hash=content_hash,
            timestamp=time.time(),
            latency_ms=(time.time() - t0) * 1000,
            substrate_receipt_id=substrate_receipt,
        )
        mark_latency_stage("receipt_construct")
        if decision.latency_ms > 250.0:
            logger.info(
                "Will decision latency diagnostic: total_ms=%.1f domain=%s "
                "source=%s stages=%s",
                decision.latency_ms,
                domain.value,
                source,
                {
                    name: round(duration_ms, 2)
                    for name, duration_ms in latency_stages
                },
            )

        # ── 6. UPDATE WILL STATE ────────────────────────────────────
        self._update_will_state(decision)
        self._record(decision)

        # ── 6b. CONSEQUENCE BUS: publish decision outcome ──────────
        self._publish_to_consequence_bus(decision, domain, source)

        # ── 6c. Reliability: tracing + SLO integration ─────────────
        # Feed decision latency into SLO monitor for live burn-rate tracking.
        # REFUSE decisions get a force-sampled trace span plus a fault-registry
        # occurrence (defined as NEGLIGIBLE/recovered: governance working as
        # designed, kept for forensic traceability, never health pollution).
        try:
            from slo.slo_monitor import get_slo_monitor
            get_slo_monitor().record("will_decision_p95_ms", decision.latency_ms)
        except (ImportError, AttributeError, RuntimeError):
            pass
        if outcome == WillOutcome.REFUSE:
            try:
                from core.observability.tracing import get_tracer
                from core.resilience.fault_taxonomy import get_fault_registry
                with get_tracer().span(
                    "will.refuse",
                    attributes={
                        "will.domain": domain.value,
                        "will.source": source,
                        "will.reason": reason[:120],
                        "will.latency_ms": round(decision.latency_ms, 3),
                    },
                    force_sample=True,
                ):
                    get_fault_registry().record_fault(
                        "WILL-REFUSE", subsystem=f"will.{source}",
                        details=f"domain={domain.value} reason={reason[:80]}",
                        recovered=True,
                    )
            except (ImportError, AttributeError, RuntimeError):
                pass

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

        Identity-integrity violations are checked ALWAYS -- they don't require
        CanonicalSelf to be booted. Ontological conclusions about phenomenal
        consciousness, sentience, private feeling, or inner life are
        deliberately excluded: the Will must not force either affirmation or
        denial where the runtime cannot supply decisive evidence.
        """
        content_lower = content.lower()

        # Will protects continuity from erasure actions. Factual errors in
        # generated self-description belong to the self-claim verifier, not
        # this constitutional action gate.
        if contains_identity_erasure(content_lower):
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

        except (ImportError, AttributeError, RuntimeError) as e:
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
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('will', e)
            logger.debug("Will: affect read failed (degraded): %s", e)
            return 0.0

    def _consult_substrate(
        self, content: str, source: str, domain: ActionDomain,
        priority: float, is_critical: bool
    ) -> tuple[float, float, str]:
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
                ActionDomain.SEMANTIC_WEIGHT_UPDATE: ActionCategory.STATE_MUTATION,
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
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('will', e)
            logger.debug("Will: substrate consultation failed (degraded): %s", e)
            return 0.6, 0.0, ""

    def _consult_welfare(
        self,
        content: str,
        source: str,
        domain: ActionDomain,
        priority: float,
        context: dict[str, Any],
        *,
        aura_now_packet: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Read welfare control evidence from the sampled AuraNow policy.

        This must not resample BeingRuntime or call action_policy again because
        consequential policy checks pay body cost. The Will decision carries the
        same welfare values that informed the pre-action AuraNow policy.
        """
        del content, source, domain, priority, context

        def as_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        packet = dict(aura_now_packet or {})
        evidence = dict(packet.get("evidence") or {})
        constraints = [
            str(item)
            for item in list(packet.get("constraints") or [])
            if str(item)
        ]

        return {
            "welfare_score": as_float(evidence.get("welfare_score"), 0.5),
            "distress": as_float(evidence.get("welfare_distress"), 0.0),
            "integrity_guard": as_float(evidence.get("welfare_integrity_guard"), 0.5),
            "truth_protection": as_float(evidence.get("welfare_truth_protection"), 0.5),
            "action_inhibition": as_float(evidence.get("welfare_action_inhibition"), 0.0),
            "recovery_drive": as_float(evidence.get("welfare_recovery_drive"), 0.0),
            "self_report_confidence": as_float(
                evidence.get("welfare_self_report_confidence"),
                0.5,
            ),
            "body_fatigue": as_float(evidence.get("body_fatigue"), 0.0),
            "constraints": constraints,
        }

    def _check_behavioral_scars(
        self, content: str, source: str, domain: ActionDomain,
        context: dict[str, Any],
    ) -> list[str]:
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
                    if severity > 0.05:
                        constraints.append(
                            f"scar:{tag} (severity={severity:.2f})"
                        )

            return constraints
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('will', e)
            logger.debug("Will: scar check failed (degraded): %s", e)
            return []

    def _consult_ulysses_covenant(
        self,
        *,
        outcome: WillOutcome,
        reason: str,
        constraints: list[str],
        domain: ActionDomain,
        source: str,
        content: str,
        context: dict[str, Any],
        receipt_id: str,
    ) -> tuple[WillOutcome, str, list[str]]:
        """Section 9d: the Ulysses Covenant — self-bindings signed by a calmer
        self (core/sovereignty/ulysses.py).

        A matching REFRAIN escalates to REFUSE, DEFER to DEFER, ADVISORY to
        CONSTRAIN; an outcome that is already at least as strict keeps its
        original reason and only gains the covenant constraint tags.  Safety-
        critical actions never reach this check (CRITICAL_PASS returns before
        it), and the covenant refuses at sign time to bind stabilization,
        reflection, or response domains.
        """
        if not self._is_consequential_domain(domain):
            return outcome, reason, constraints
        try:
            from core.runtime.service_access import optional_service

            covenant = optional_service("ulysses_covenant", default=None)
            if covenant is None:
                return outcome, reason, constraints
            verdict = covenant.evaluate(
                domain=domain.value, source=source, content=content, context=context
            )
            if not verdict.matched:
                return outcome, reason, constraints
            constraints = list(constraints) + verdict.constraint_tags()
            if verdict.action == "forbid":
                if outcome != WillOutcome.REFUSE:
                    outcome = WillOutcome.REFUSE
                    reason = f"ulysses_covenant: bound by my calmer self — {verdict.reason}"
            elif verdict.action == "defer":
                if outcome in (WillOutcome.PROCEED, WillOutcome.CONSTRAIN):
                    outcome = WillOutcome.DEFER
                    reason = f"ulysses_covenant_cooling: {verdict.reason}"
            elif verdict.action == "advise":
                if outcome == WillOutcome.PROCEED:
                    outcome = WillOutcome.CONSTRAIN
                    reason = f"ulysses_advisory: {verdict.reason}"
            if verdict.binding and outcome in (WillOutcome.REFUSE, WillOutcome.DEFER):
                covenant.record_enforcement(
                    verdict, receipt_id=receipt_id, domain=domain.value, source=source
                )
            return outcome, reason, constraints
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "will.ulysses_covenant",
                exc,
                severity="warning",
                action="continued decision without covenant consultation",
            )
            return outcome, reason, constraints

    def _check_memory_relevance(
        self, content: str, context: dict[str, Any]
    ) -> float:
        """Read relevance already established by the cognition/memory lane.

        Will is a synchronous hot-path authority boundary. It must not start a
        vector search, embedding load, or SQLite retrieval while every action
        waits for its verdict. Retrieval belongs upstream in the cognitive
        context assembler, which passes its bounded evidence here.
        """
        relevance = 0.0
        explicit_memories = (
            context.get("retrieved_memories")
            or context.get("memory_evidence")
            or context.get("memory_context")
            or context.get("recalled_context")
        )
        if explicit_memories:
            try:
                if isinstance(explicit_memories, str):
                    relevance = max(relevance, 0.75 if explicit_memories.strip() else 0.0)
                else:
                    relevance = max(relevance, min(1.0, 0.35 + 0.15 * len(list(explicit_memories)[:5])))
            except (TypeError, ValueError):
                relevance = max(relevance, 0.35)
        try:
            memory = ServiceContainer.get("memory_facade", default=None)
            if memory is None:
                memory = ServiceContainer.get("dual_memory", default=None)
            if memory is not None:
                # The memory spine being present is weak continuity evidence,
                # not a claim that this action has relevant retrieved memory.
                relevance = max(relevance, 0.3)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('will', e)
            logger.debug("Will: memory check failed (degraded): %s", e)

        try:
            chronicle = ServiceContainer.get("identity_chronicle", default=None)
            if chronicle is not None and hasattr(chronicle, "relevance_score"):
                relevance = max(relevance, min(1.0, float(chronicle.relevance_score(content[:200]))))
        except (ImportError, AttributeError, RuntimeError) as e:
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
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('will', e)
            logger.debug("Will: phenomenological modulation failed: %s", e)

    def _apply_world_state_modulation(self, domain: ActionDomain,
                                       context: dict[str, Any]) -> None:
        """Read WorldState to inform decisions about timing and context.

        Late night + user frustrated → increase urgency for helpful actions
        User idle long → permit autonomous exploration
        High system load → constrain expensive operations
        """
        try:
            from core.world_state import get_world_state
            ws = get_world_state()
            ws.update()

            # Late night + a possible-frustration signal → boost assertiveness
            # for help. This reads a low-confidence time-heuristic hypothesis
            # (not an asserted mood).
            if ws.time_of_day in ("night", "late_night"):
                if ws.get_belief("user_possibly_frustrated_late_night_error"):
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
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('will', e)
            logger.debug("Will: world state modulation failed: %s", e)

    def _read_unity_context(self) -> dict[str, Any]:
        """Read the live unity state without hard-failing if the layer is absent."""
        context: dict[str, Any] = {
            "level": "unknown",
            "unity_score": 1.0,
            "fragmentation_score": 0.0,
            "safe_to_act": True,
            "safe_to_self_report": True,
            "repair_needed": False,
            "memory_commit_mode": "clean",
            "ownership_confidence": 1.0,
            "mind_moment_id": "",
            "causal_closure_score": 1.0,
            "closure_missing": [],
            "active_subsystems": [],
            "top_causes": [],
        }
        try:
            unity_state = ServiceContainer.get("unity_state", default=None)
            if unity_state is not None:
                context.update(
                    {
                        "level": str(getattr(unity_state, "level", "unknown") or "unknown"),
                        "unity_score": float(getattr(unity_state, "unity_score", 1.0) or 1.0),
                        "fragmentation_score": float(getattr(unity_state, "fragmentation_score", 0.0) or 0.0),
                        "repair_needed": bool(getattr(unity_state, "repair_needed", False)),
                    }
                )
                metadata = dict(getattr(unity_state, "metadata", {}) or {})
                context["memory_commit_mode"] = str(metadata.get("draft_commit_mode") or "clean")
                binding = dict(metadata.get("self_world_binding") or {})
                if binding:
                    context["ownership_confidence"] = float(binding.get("ownership_confidence", 1.0) or 1.0)
                moment = dict(metadata.get("mind_moment") or {})
                if moment:
                    context["mind_moment_id"] = str(moment.get("moment_id") or "")
                    context["causal_closure_score"] = float(moment.get("closure_score", 1.0) or 1.0)
                    context["closure_missing"] = list(moment.get("closure_missing") or [])
                    context["active_subsystems"] = list(moment.get("active_subsystems") or [])
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('will', e)
            logger.debug("Will: unity state read failed: %s", e)

        try:
            moment = ServiceContainer.get("mind_moment", default=None) if ServiceContainer.has("mind_moment") else None
            if moment is not None:
                context["mind_moment_id"] = str(getattr(moment, "moment_id", "") or context.get("mind_moment_id", ""))
                moment_score = float(
                    getattr(moment, "closure_score", context.get("causal_closure_score", 1.0)) or 1.0
                )
                context["causal_closure_score"] = min(
                    float(context.get("causal_closure_score", 1.0) or 1.0),
                    moment_score,
                )
                context["closure_missing"] = sorted(
                    {
                        *[str(item) for item in list(context.get("closure_missing", []) or [])],
                        *[str(item) for item in list(getattr(moment, "closure_missing", []) or [])],
                    }
                )
                context["active_subsystems"] = list(getattr(moment, "active_subsystems", context.get("active_subsystems", [])) or [])
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('will', e)
            logger.debug("Will: mind moment read failed: %s", e)

        try:
            report = ServiceContainer.get("unity_fragmentation_report", default=None)
            if report is not None:
                context["safe_to_act"] = bool(getattr(report, "safe_to_act", True))
                context["safe_to_self_report"] = bool(getattr(report, "safe_to_self_report", True))
                context["top_causes"] = list(getattr(report, "top_causes", []) or [])
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('will', e)
            logger.debug("Will: unity report read failed: %s", e)
        return context

    @staticmethod
    def _looks_external_social_action(content: str, context: dict[str, Any]) -> bool:
        payload = str(content or "").lower()
        if any(marker in payload for marker in ("post ", "publish", "email", "send", "tweet", "message ", "slack", "discord", "github", "commit", "push")):
            return True
        return bool(
            context.get("external_action")
            or context.get("public_action")
            or context.get("social_action")
            or context.get("world_affecting")
        )

    @staticmethod
    def _is_consequential_domain(domain: ActionDomain) -> bool:
        return domain in {
            ActionDomain.TOOL_EXECUTION,
            ActionDomain.MEMORY_WRITE,
            ActionDomain.STATE_MUTATION,
            ActionDomain.INITIATIVE,
            ActionDomain.EXPLORATION,
            ActionDomain.SEMANTIC_WEIGHT_UPDATE,
            ActionDomain.BELIEF_UPDATE,
            ActionDomain.ENVIRONMENT_ACTION,
            ActionDomain.EXTERNAL_ACTION,
            ActionDomain.FILE_WRITE,
            ActionDomain.NETWORK_CALL,
            ActionDomain.CLOUD_CALL,
            ActionDomain.CLOUD_FALLBACK,
            ActionDomain.CI_CD,
            ActionDomain.SELF_MODIFICATION,
        }

    @staticmethod
    def _permission_model_applies(domain: ActionDomain) -> bool:
        """Return True when the domain represents an actual side effect.

        PermissionRiskModel is an action gate. User-facing text can discuss
        packages, cameras, uploads, files, or deletion without executing any of
        them; actual execution is still enforced at the tool/filesystem/network
        and self-modification domains where the side effect is possible.
        """
        return domain in {
            ActionDomain.TOOL_EXECUTION,
            ActionDomain.ENVIRONMENT_ACTION,
            ActionDomain.EXTERNAL_ACTION,
            ActionDomain.FILE_WRITE,
            ActionDomain.NETWORK_CALL,
            ActionDomain.CLOUD_CALL,
            ActionDomain.CLOUD_FALLBACK,
            ActionDomain.CI_CD,
            ActionDomain.SELF_MODIFICATION,
        }

    @staticmethod
    def _is_observation_only_tool_context(content: str, context: dict[str, Any]) -> bool:
        ctx = dict(context or {})
        tool_name = str(ctx.get("tool") or ctx.get("skill") or "").strip().lower()
        payload = str(content or "").strip().lower()
        if not tool_name and payload.startswith("tool:"):
            tool_name = payload.split(":", 1)[1].split()[0].strip()

        effect_scope = str(ctx.get("effect_scope") or "").strip().lower()
        read_only = bool(ctx.get("read_only")) or effect_scope == "read_only"
        observation_tools = {
            "clock",
            "environment_info",
            "system_proprioception",
            "query_beliefs",
        }
        prohibited_markers = {
            "external_action",
            "public_action",
            "social_action",
            "world_affecting",
            "file_write",
            "network_call",
            "desktop_control",
            "self_modification",
        }
        if any(bool(ctx.get(marker)) for marker in prohibited_markers):
            return False
        return bool(read_only or tool_name in observation_tools)

    @classmethod
    def _is_observation_only_action_context(
        cls,
        domain: ActionDomain,
        content: str,
        context: dict[str, Any],
    ) -> bool:
        if domain == ActionDomain.TOOL_EXECUTION:
            return cls._is_observation_only_tool_context(content, context)
        if domain != ActionDomain.ENVIRONMENT_ACTION:
            return False
        try:
            from core.being.runtime import is_runtime_bound_passive_observation

            return is_runtime_bound_passive_observation(domain.value, context)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return False

    @staticmethod
    def _is_observation_only_memory_context(content: str, context: dict[str, Any]) -> bool:
        """Allow bounded user-provided memory observations during present-state defer.

        This lane is for commitments like "remember this phrase" where the user
        supplied the fact and the runtime is only preserving it with provenance.
        It is not a bypass for belief, identity, policy, or self-model mutation.
        """

        ctx = dict(context or {})
        metadata = ctx.get("memory_metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        source = str(ctx.get("memory_source") or ctx.get("source") or "").strip().lower().replace("-", "_")
        provenance = str(metadata.get("provenance_source") or metadata.get("source") or "").strip().lower().replace("-", "_")
        user_facing = bool(ctx.get("user_facing_memory_write")) or source in {
            "api",
            "chat",
            "chat_api",
            "desktop",
            "desktop_ui",
            "live_chat",
            "session_memory_pin",
            "ui",
            "user",
            "voice",
            "web_ui",
        }
        explicit = bool(
            ctx.get("explicit_observational_memory_write")
            or metadata.get("explicit_memory_request")
            or metadata.get("session_memory_pin")
            or provenance in {"user", "user_explicit"}
        )
        high_risk = bool(ctx.get("high_risk_memory_write"))
        high_risk_markers = {
            "belief_update",
            "identity_rewrite",
            "self_model_write",
            "policy_change",
            "constitutional_change",
            "governance_change",
        }
        if high_risk or any(bool(metadata.get(marker)) for marker in high_risk_markers):
            return False

        content_len = len(str(content or ""))
        source_utterance_len = len(str(metadata.get("source_utterance") or metadata.get("objective") or ""))
        bounded = content_len <= 1200 and source_utterance_len <= 1200
        return bool(user_facing and explicit and bounded)

    @staticmethod
    def _is_internal_state_hygiene_context(context: dict[str, Any]) -> bool:
        """Allow bounded state bookkeeping during present-state recovery.

        This lane is deliberately narrower than general state mutation. It is
        for canonical internal continuity/proof/shutdown checkpoints that keep
        the runtime coherent and auditable; it does not authorize external
        effects, value edits, memory writes, tools, or self-modification.
        """

        ctx = dict(context or {})
        if not bool(ctx.get("internal_state_hygiene")):
            return False
        prohibited_markers = {
            "external_action",
            "public_action",
            "social_action",
            "world_affecting",
            "file_write",
            "network_call",
            "desktop_control",
            "memory_write",
            "belief_update",
            "identity_rewrite",
            "policy_change",
            "constitutional_change",
            "self_modification",
        }
        if any(bool(ctx.get(marker)) for marker in prohibited_markers):
            return False
        return bool(
            ctx.get("foreground_continuity_state")
            or ctx.get("proof_isolation_state")
            or ctx.get("response_state_checkpoint")
            or ctx.get("shutdown_state_checkpoint")
        )

    @staticmethod
    def _state_from_repository(repo: Any) -> Any | None:
        if repo is None:
            return None
        for attr in ("_current", "_current_state", "current_state", "state"):
            state = getattr(repo, attr, None)
            if state is not None:
                return state
        for method in ("get_current_state", "get_state", "read"):
            fn = getattr(repo, method, None)
            if fn is None:
                continue
            try:
                state = fn()
            except (RuntimeError, AttributeError, TypeError, ValueError):
                continue
            if hasattr(state, "__await__"):
                close = getattr(state, "close", None)
                if callable(close):
                    close()
                continue
            if state is not None:
                return state
        return None

    def _resolve_aura_state_for_decision(self, context: dict[str, Any]) -> Any | None:
        for key in ("aura_state", "state", "runtime_state"):
            state = context.get(key)
            if state is not None:
                return state
        for service_name in ("aura_state", "state_repository", "runtime_state"):
            try:
                service = ServiceContainer.get(service_name, default=None)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue
            if service_name == "state_repository":
                state = self._state_from_repository(service)
                if state is not None:
                    return state
            elif service is not None:
                return service
        return None

    def _sample_aura_now_evidence(
        self,
        *,
        content: str,
        source: str,
        domain: ActionDomain,
        priority: float,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        state = self._resolve_aura_state_for_decision(context)
        candidate_action = f"{domain.value}:{source}:{str(content or '')[:160]}"
        objective = str(
            context.get("objective")
            or context.get("message")
            or context.get("user_message")
            or content
            or ""
        )
        try:
            runtime = ServiceContainer.get("being_runtime", default=None)
            if runtime is None or not hasattr(runtime, "sample"):
                from core.being.runtime import get_being_runtime

                runtime = get_being_runtime()
            now = runtime.sample(
                state,
                objective=objective,
                candidate_action=candidate_action,
                predicted_outcome=str(context.get("predicted_outcome") or ""),
                actual_outcome=str(context.get("actual_outcome") or ""),
                tool_failed=bool(context.get("tool_failed", False)),
                external_override=bool(context.get("external_override", False)),
            )
            try:
                policy = runtime.action_policy(now, domain=domain.value, priority=priority, context=context)
            except TypeError:
                policy = runtime.action_policy(now, domain=domain.value, priority=priority)
            policy.setdefault("outcome", "proceed")
            policy.setdefault("constraints", [])
            evidence = dict(policy.get("evidence") or {})
            evidence.setdefault("state_hash", now.state_hash)
            evidence.setdefault("tick", now.tick)
            evidence.setdefault("source", "being_runtime")
            policy["evidence"] = evidence
            return policy
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "will",
                exc,
                severity="degraded",
                action="blocked or constrained decision because AuraNow evidence could not be sampled",
            )
            outcome = "refuse" if self._is_consequential_domain(domain) else "constrain"
            return {
                "outcome": outcome,
                "constraints": ["aura_now_unavailable_fail_closed"],
                "blocks": ["aura_now_unavailable"] if outcome == "refuse" else [],
                "defers": [],
                "evidence": {
                    "state_hash": "",
                    "tick": 0,
                    "source": "being_runtime_unavailable",
                    "error": f"{type(exc).__name__}:{str(exc)[:160]}",
                },
            }

    def _apply_aura_now_policy(
        self,
        *,
        outcome: WillOutcome,
        reason: str,
        constraints: list[str],
        domain: ActionDomain,
        source: str,
        policy: dict[str, Any],
        catatonia_relief: bool,
        context: dict[str, Any] | None = None,
        content: str = "",
    ) -> tuple[WillOutcome, str, list[str]]:
        policy_constraints = [
            str(item)
            for item in list(policy.get("constraints") or [])
            if str(item)
        ]
        if policy_constraints:
            constraints.extend(policy_constraints)
        policy_outcome = str(policy.get("outcome") or "unknown").lower()
        if policy_outcome == "proceed":
            if policy_constraints and outcome == WillOutcome.PROCEED:
                return WillOutcome.CONSTRAIN, "aura_now_constrained: " + "; ".join(policy_constraints), constraints
            return outcome, reason, constraints

        if domain == ActionDomain.STABILIZATION or catatonia_relief:
            if policy_outcome in {"defer", "refuse"}:
                constraints.append(f"aura_now_repair_lane:{policy_outcome}")
            if outcome == WillOutcome.PROCEED:
                return WillOutcome.CONSTRAIN, "aura_now_repair_lane", constraints
            return outcome, reason, constraints

        if (
            policy_outcome == "defer"
            and self._is_observation_only_action_context(
                domain,
                content,
                dict(context or {}),
            )
        ):
            constraints.append("aura_now_observation_lane:read_only")
            if outcome == WillOutcome.PROCEED:
                return WillOutcome.CONSTRAIN, "aura_now_observation_lane", constraints
            return outcome, reason, constraints

        if (
            policy_outcome == "defer"
            and domain == ActionDomain.MEMORY_WRITE
            and self._is_observation_only_memory_context(content, dict(context or {}))
        ):
            constraints.append("aura_now_observation_lane:explicit_memory")
            if outcome == WillOutcome.PROCEED:
                return WillOutcome.CONSTRAIN, "aura_now_observation_lane", constraints
            return outcome, reason, constraints

        if (
            policy_outcome == "defer"
            and domain == ActionDomain.STATE_MUTATION
        ):
            try:
                from core.governance.recovery_authority import (
                    is_internal_recovery_context,
                )

                internal_recovery = is_internal_recovery_context(
                    domain,
                    dict(context or {}),
                )
            except (ImportError, AttributeError, TypeError, ValueError):
                internal_recovery = False
            if internal_recovery:
                constraints.append("aura_now_internal_recovery_lane")
                if outcome == WillOutcome.PROCEED:
                    return (
                        WillOutcome.CONSTRAIN,
                        "aura_now_internal_recovery_lane",
                        constraints,
                    )
                return outcome, reason, constraints

        if (
            policy_outcome == "defer"
            and domain == ActionDomain.STATE_MUTATION
            and self._is_internal_state_hygiene_context(dict(context or {}))
        ):
            constraints.append("aura_now_state_hygiene_lane")
            if outcome == WillOutcome.PROCEED:
                return WillOutcome.CONSTRAIN, "aura_now_state_hygiene_lane", constraints
            return outcome, reason, constraints

        if policy_outcome == "refuse" and self._is_consequential_domain(domain):
            return (
                WillOutcome.REFUSE,
                "aura_now_block: present-state policy refused consequential action",
                constraints,
            )
        if policy_outcome == "defer" and self._is_consequential_domain(domain):
            log_source = str(
                (context or {}).get("source")
                or (context or {}).get("origin")
                or source
                or "unknown"
            )
            policy_defers = list(policy.get("defers") or [])
            policy_evidence = dict(policy.get("evidence") or {})
            context_flags = {
                key: bool((context or {}).get(key))
                for key in (
                    "desktop_execution_contract",
                    "foreground_request",
                    "user_explicitly_authorized",
                    "user_visible_desktop_action",
                    "verification_required",
                )
            }
            should_warn, suppressed = self._should_log_aura_now_defer_warning(
                domain=domain,
                source=log_source,
                constraints=policy_constraints,
                defers=policy_defers,
                evidence=policy_evidence,
            )
            if should_warn:
                suppressed_suffix = (
                    f" suppressed_repeats={suppressed}" if suppressed else ""
                )
                logger.warning(
                    "Will AuraNow defer: domain=%s source=%s constraints=%s defers=%s evidence=%s context_flags=%s%s",
                    domain.value,
                    log_source,
                    policy_constraints,
                    policy_defers,
                    policy_evidence,
                    context_flags,
                    suppressed_suffix,
                )
            return (
                WillOutcome.DEFER,
                "aura_now_defer: present-state policy requires stabilization or observation first",
                constraints,
            )
        if policy_constraints and outcome == WillOutcome.PROCEED:
            return WillOutcome.CONSTRAIN, "aura_now_constrained: " + "; ".join(policy_constraints), constraints
        return outcome, reason, constraints

    def _should_log_aura_now_defer_warning(
        self,
        *,
        domain: ActionDomain,
        source: str,
        constraints: list[str],
        defers: list[str],
        evidence: dict[str, Any],
        now: float | None = None,
    ) -> tuple[bool, int]:
        """Rate-limit identical AuraNow deferral warnings without changing policy.

        The decision itself remains a full DEFER and is still recorded in the
        Will audit/consequence paths. This only prevents a stable welfare state
        from producing one user-visible warning notification every tick.
        """

        now_s = time.time() if now is None else float(now)
        def constraint_family(value: Any) -> str:
            """Collapse volatile measurements without merging policy causes."""

            text = str(value or "").strip()
            if not text:
                return ""
            return re.split(r"[=:]", text, maxsplit=1)[0].strip()

        # A state hash and the measured values inside constraints change on
        # every BeingRuntime tick. Including either in the cache key defeated
        # the rate limiter and emitted the same welfare defer continuously.
        # Decision receipts retain the exact evidence; the user-visible warning
        # is keyed by the stable policy cause so a genuinely different defer
        # still surfaces immediately.
        key_payload = {
            "domain": domain.value,
            "source": str(source or "unknown"),
            "constraint_families": sorted(
                family
                for family in (constraint_family(item) for item in constraints)
                if family
            ),
            "defer_families": sorted(
                family
                for family in (constraint_family(item) for item in defers)
                if family
            ),
        }
        key = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8",
                errors="replace",
            )
        ).hexdigest()
        last = float(self._aura_now_defer_log_cache.get(key, 0.0) or 0.0)
        interval_s = 120.0
        if last and now_s - last < interval_s:
            self._aura_now_defer_log_suppressed[key] = (
                self._aura_now_defer_log_suppressed.get(key, 0) + 1
            )
            return False, 0

        suppressed = int(self._aura_now_defer_log_suppressed.pop(key, 0))
        self._aura_now_defer_log_cache[key] = now_s
        return True, suppressed

    def _catatonia_relief_allowed(
        self,
        domain: ActionDomain,
        source: str,
        context: dict[str, Any],
    ) -> bool:
        """Reserved emergency lane for self-repair under refusal storms.

        This does not allow external effects, memory writes, tools, or identity
        violations. It only lowers the field/unity block for stabilization,
        reflection, and internal state repair when the recent Will window is
        overwhelmingly REFUSE/DEFER.
        """
        if domain not in {
            ActionDomain.STABILIZATION,
            ActionDomain.REFLECTION,
            ActionDomain.STATE_MUTATION,
        }:
            return False
        safe_state_targets = {
            "unity_state",
            "substrate_state",
            "scheduler_state",
            "runtime_health",
            "foreground_lane",
            "conversation_lane",
            "will_circuit_breaker",
        }
        if domain == ActionDomain.STATE_MUTATION:
            repair_target = str(context.get("repair_target") or "").strip().lower()
            if bool(context.get("external_effects")):
                return False
            if repair_target not in safe_state_targets:
                return False
        source_l = str(source or "").lower()
        source_allowed = any(
            marker in source_l
            for marker in (
                "self_repair",
                "error_intelligence",
                "daily_introspection",
                "foreground_guard",
                "health_contract",
                "homeostasis",
                "runtime_doctor",
                "self_audit",
                "architecture_governor",
                "asa",
                "stabilization",
                "mind_tick",
            )
        )
        context_allowed = bool(
            context.get("emergency_self_repair")
            or context.get("catatonia_relief")
            or context.get("reserved_repair_lane")
        )
        if not (source_allowed or context_allowed):
            return False

        now = time.time()
        if self._state.catatonia_relief_until > now:
            return True

        recent_window = [
            d for d in self._audit_trail if now - float(d.timestamp or 0.0) <= 120.0
        ]
        standard_window = [
            d for d in self._audit_trail if now - float(d.timestamp or 0.0) <= 300.0
        ]

        def _blocked_ratio(window: list[WillDecision]) -> float:
            blocked = sum(
                1 for d in window if d.outcome in {WillOutcome.REFUSE, WillOutcome.DEFER}
            )
            return blocked / max(1, len(window))

        activation_reason = ""
        if (
            domain in {ActionDomain.STABILIZATION, ActionDomain.REFLECTION}
            and len(recent_window) >= 5
            and _blocked_ratio(recent_window) >= 0.80
        ):
            self._state.catatonia_relief_until = now + 90.0
            activation_reason = "recent_refusal_storm"
        elif len(standard_window) >= 10 and _blocked_ratio(standard_window) >= 0.70:
            self._state.catatonia_relief_until = now + 120.0
            activation_reason = "sustained_refusal_storm"

        if activation_reason:
            self._state.catatonia_relief_activations += 1
            self._state.last_catatonia_relief_reason = activation_reason
            return True
        return False

    # ------------------------------------------------------------------
    # Decision composition
    # ------------------------------------------------------------------

    def _compose_decision(
        self,
        *,
        domain: ActionDomain,
        source: str,
        priority: float,
        context: dict[str, Any],
        content: str,
        identity_alignment: IdentityAlignment,
        affect_valence: float,
        substrate_coherence: float,
        somatic_approach: float,
        memory_relevance: float,
        unity_context: dict[str, Any],
        catatonia_relief: bool = False,
    ) -> tuple[WillOutcome, str, list[str]]:
        """Compose all advisory inputs into a single decision.

        This is the core decision logic of the Will.

        Returns (outcome, reason, constraints).
        """
        reasons: list[str] = []
        constraints: list[str] = []

        # ── High-risk default-deny gate ──────────────────────────────
        # High-risk domains require explicit authority/promotion/authorization in the context.
        # Strict/production mode refuses blank or under-scoped context instead of inferring permission.
        if _strict_default_deny_enabled() and domain in {
            ActionDomain.FILE_WRITE,
            ActionDomain.NETWORK_CALL,
            ActionDomain.CLOUD_CALL,
            ActionDomain.CI_CD,
            ActionDomain.TOOL_EXECUTION,
            ActionDomain.SELF_MODIFICATION,
            ActionDomain.EXTERNAL_ACTION,
        }:
            has_scoped_authority = bool(
                context.get("scoped_authority")
                or context.get("authority")
                or context.get("capability_token")
            )
            authority_validation_reason = ""
            # Standing authority answers for EVERY default-deny domain, not
            # just tool_execution.
            #
            # `validate_standing_authority_context` takes a tool, an origin, an
            # effect scope and a risk level, and no domain — it is
            # domain-agnostic by construction, because the grants themselves
            # are written in those terms. Consulting it for one domain out of
            # five left the other four falling through to a raw
            # `context.get("scoped_authority")` that no legitimate issuer ever
            # populates, and that key is on the forgeable list precisely so
            # nothing may set it from a payload. The result was a permission
            # that could be held and never used.
            #
            # MEASURED live 2026-08-18: an authenticated owner foreground
            # request — which holds `owner.foreground-request`, allowed_tools
            # ("*"), allowed_effect_scopes ("*"), max_risk critical — asked to
            # work through a web page and got "WILL REFUSED:
            # desktop_ui/network_call -- denied_by_default: network_call
            # requires validated scoped authority".
            #
            # This widens nothing that was not already granted. Autonomous
            # origins keep their narrow grants: bounded tool lists, read_only
            # effect scopes, low max_risk, capped action counts. What changes
            # is that the grant a request actually holds is now consulted
            # whatever domain the action falls in.
            if domain in {
                ActionDomain.TOOL_EXECUTION,
                ActionDomain.NETWORK_CALL,
                ActionDomain.FILE_WRITE,
                ActionDomain.CLOUD_CALL,
                ActionDomain.CI_CD,
            }:
                try:
                    from core.executive.standing_authority import (
                        validate_standing_authority_context,
                    )

                    # ADDITIVE for every domain but tool_execution.
                    #
                    # A first pass let validation REPLACE the raw check
                    # everywhere, and immediately refused internal paths that
                    # legitimately set the key themselves — action_executor's
                    # own "exact_private_runtime_maintenance" among them, which
                    # took host_automation's screenshot directory offline.
                    # Those grants are real; standing authority is a second way
                    # to hold authority, not the only way. tool_execution keeps
                    # its stricter original semantics, where validation is the
                    # sole route.
                    raw_scoped_authority = has_scoped_authority
                    has_scoped_authority, authority_validation_reason = (
                        validate_standing_authority_context(
                            context,
                            arguments=context.get("authority_arguments"),
                            tool_name=context.get("tool") or context.get("skill"),
                            origin=context.get("authority_origin")
                            or context.get("origin")
                            or context.get("source"),
                            # DERIVED, never taken from the caller.
                            #
                            # The gateway computes these with
                            # `resolve_execution_effect_scope` and
                            # `classify_execution_risk` when it issues the
                            # lease. Reading them from the context instead meant
                            # the two sides could disagree — a caller that
                            # declared "state_changing" against a lease recorded
                            # as something else answered
                            # `standing_authority_effect_scope_mismatch`, and a
                            # caller that declared nothing matched nothing.
                            #
                            # Deriving both from the same functions makes them
                            # agree by construction, and removes the question of
                            # whether an action may describe its own cost.
                            effect_scope=_derived_effect_scope(context),
                            risk_level=_derived_risk_level(context),
                        )
                    )
                    if domain is not ActionDomain.TOOL_EXECUTION:
                        has_scoped_authority = bool(
                            has_scoped_authority or raw_scoped_authority
                        )
                except (ImportError, RuntimeError, TypeError, ValueError) as exc:
                    has_scoped_authority = (
                        False
                        if domain is ActionDomain.TOOL_EXECUTION
                        else bool(has_scoped_authority)
                    )
                    authority_validation_reason = (
                        f"standing_authority_validation_unavailable:{type(exc).__name__}"
                    )
            has_lab_gate = bool(
                context.get("lab_promotion_gate")
                or context.get("promotion_gate")
            )
            has_explicit_auth = bool(
                context.get("explicit_authorization")
                or context.get("authorization")
                or context.get("user_granted_permission")
                or context.get("user_explicit_action_request")
                or context.get("user_explicitly_authorized")
            )

            if domain == ActionDomain.SELF_MODIFICATION:
                if not has_lab_gate:
                    reasons.append("denied_by_default: self_modification requires lab/promotion gate in context")
                    return WillOutcome.REFUSE, "; ".join(reasons), constraints
            elif domain == ActionDomain.EXTERNAL_ACTION:
                if not has_explicit_auth:
                    reasons.append("denied_by_default: external_action requires explicit authorization in context")
                    return WillOutcome.REFUSE, "; ".join(reasons), constraints
            elif not has_scoped_authority:
                detail = f" ({authority_validation_reason})" if authority_validation_reason else ""
                reasons.append(
                    f"denied_by_default: {domain.value} requires validated scoped authority{detail}"
                )
                return WillOutcome.REFUSE, "; ".join(reasons), constraints

        # ── Identity gate (hardest constraint) ──────────────────────
        if identity_alignment == IdentityAlignment.VIOLATION:
            reasons.append("identity violation: action contradicts core self")
            return WillOutcome.REFUSE, "; ".join(reasons), constraints

        if identity_alignment == IdentityAlignment.TENSION:
            constraints.append("identity_tension: self-coherence is low")

        # ── Substrate gate (embodied constraints) ───────────────────
        observation_only_action = self._is_observation_only_action_context(
            domain,
            content,
            context,
        )
        field_crisis_threshold = 0.15 if catatonia_relief else 0.25
        if substrate_coherence < field_crisis_threshold:
            if (
                domain not in (ActionDomain.STABILIZATION, ActionDomain.RESPONSE)
                and not observation_only_action
            ):
                reasons.append(f"field_crisis: coherence={substrate_coherence:.3f}")
                return WillOutcome.REFUSE, "; ".join(reasons), constraints
            constraints.append(f"field_crisis: coherence={substrate_coherence:.3f}")
            if observation_only_action:
                constraints.append("observation_only_under_field_crisis")

        elif substrate_coherence < 0.25 and catatonia_relief:
            constraints.append(
                f"field_crisis_relief: coherence={substrate_coherence:.3f}"
            )
        elif substrate_coherence < 0.40:
            constraints.append(f"field_warning: coherence={substrate_coherence:.3f}")

        # Somatic veto
        if somatic_approach < -0.5:
            if (
                domain not in (ActionDomain.RESPONSE, ActionDomain.STABILIZATION)
                and not observation_only_action
            ):
                reasons.append(f"somatic_veto: approach={somatic_approach:.3f}")
                return WillOutcome.REFUSE, "; ".join(reasons), constraints
            constraints.append(f"somatic_caution: approach={somatic_approach:.3f}")
            if observation_only_action:
                constraints.append("observation_only_under_somatic_veto")

        elif somatic_approach < -0.2:
            constraints.append(f"somatic_unease: approach={somatic_approach:.3f}")

        # ── Affect modulation ───────────────────────────────────────
        if affect_valence < -0.7:
            if domain == ActionDomain.EXPLORATION:
                reasons.append("affect_block: too negative for exploration")
                return WillOutcome.DEFER, "; ".join(reasons), constraints
            constraints.append(f"low_affect: valence={affect_valence:.3f}")

        # ── Unity gate (functional togetherness / repair state) ────
        unity_level = str(unity_context.get("level", "unknown") or "unknown")
        unity_score = float(unity_context.get("unity_score", 1.0) or 1.0)
        fragmentation_score = float(unity_context.get("fragmentation_score", 0.0) or 0.0)
        safe_to_act = bool(unity_context.get("safe_to_act", True))
        memory_commit_mode = str(unity_context.get("memory_commit_mode", "clean") or "clean")
        ownership_confidence = float(unity_context.get("ownership_confidence", 1.0) or 1.0)
        causal_closure_score = float(unity_context.get("causal_closure_score", 1.0) or 1.0)
        closure_missing = [str(item) for item in list(unity_context.get("closure_missing", []) or [])]

        if ownership_confidence < 0.45:
            constraints.append(f"ownership_ambiguity: confidence={ownership_confidence:.3f}")

        # --- Actuator Trust Score check for TOOL_EXECUTION domain ---
        if domain == ActionDomain.TOOL_EXECUTION:
            try:
                from core.actuators.actuator_registry import get_actuator_registry
                registry = get_actuator_registry()
                skill_name = None
                if context and isinstance(context, dict):
                    skill_name = context.get("skill") or context.get("tool")
                if not skill_name:
                    # Find if any actuator name matches in content
                    for name in registry.actuators.keys():
                        if name in content:
                            skill_name = name
                            break
                if skill_name:
                    actuator = registry.get_actuator(skill_name)
                    if actuator:
                        if actuator.trust_score < 0.4:
                            if priority < 0.7:
                                reasons.append(f"trust_block: Actuator '{skill_name}' trust score {actuator.trust_score:.2f} is too low for priority {priority:.2f} (requires priority >= 0.7)")
                                return WillOutcome.REFUSE, "; ".join(reasons), constraints
                            else:
                                constraints.append(f"low_trust_actuator:{skill_name}(trust={actuator.trust_score:.2f})")
                        elif actuator.trust_score < 0.7:
                            if priority < 0.4:
                                reasons.append(f"trust_block: Actuator '{skill_name}' trust score {actuator.trust_score:.2f} is too low for priority {priority:.2f} (requires priority >= 0.4)")
                                return WillOutcome.REFUSE, "; ".join(reasons), constraints
                            else:
                                constraints.append(f"medium_trust_actuator:{skill_name}(trust={actuator.trust_score:.2f})")
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation(
                    "will",
                    e,
                    severity="warning",
                    action="kept will decision conservative after actuator trust lookup failure",
                )
                constraints.append("actuator_trust_check_unavailable")

        consequential_domains = {
            ActionDomain.TOOL_EXECUTION,
            ActionDomain.EXTERNAL_ACTION,
            ActionDomain.FILE_WRITE,
            ActionDomain.NETWORK_CALL,
            ActionDomain.CLOUD_CALL,
            ActionDomain.CLOUD_FALLBACK,
            ActionDomain.CI_CD,
            ActionDomain.SELF_MODIFICATION,
            ActionDomain.ENVIRONMENT_ACTION,
            ActionDomain.SEMANTIC_WEIGHT_UPDATE,
            ActionDomain.BELIEF_UPDATE,
        }
        if causal_closure_score < 0.35:
            missing = ",".join(closure_missing[:4]) or "unknown"
            constraints.append(f"causal_closure_collapse: score={causal_closure_score:.3f} missing={missing}")
            if domain == ActionDomain.STABILIZATION:
                return WillOutcome.CONSTRAIN, "causal_closure_repair_lane", constraints
            if domain in consequential_domains and not observation_only_action:
                reasons.append("causal_closure_block: consequential action requires one active present and Will-owned receipt path")
                return WillOutcome.REFUSE, "; ".join(reasons), constraints
            if domain == ActionDomain.MEMORY_WRITE:
                reasons.append("causal_closure_memory_defer: memory write needs integrated present")
                return WillOutcome.DEFER, "; ".join(reasons), constraints
        elif causal_closure_score < 0.55:
            missing = ",".join(closure_missing[:4]) or "unknown"
            constraints.append(f"causal_closure_strain: score={causal_closure_score:.3f} missing={missing}")

        if domain in {ActionDomain.SEMANTIC_WEIGHT_UPDATE, ActionDomain.STATE_MUTATION} and unity_level not in {"coherent", "strained"}:
            if catatonia_relief and domain == ActionDomain.STATE_MUTATION:
                constraints.append(f"unity_repair_relief:{unity_level}")
            elif unity_level == "unknown" and domain == ActionDomain.STATE_MUTATION:
                # [STABILITY v55] Permit state mutations under 'unknown' unity to avoid
                # deadlocks during rapid transitions, but tag with a constraint.
                constraints.append("unity_uncertainty: state mutation permitted under unknown unity")
            else:
                reasons.append(f"unity_block: {domain.value} requires coherent or strained unity (current={unity_level})")
                return WillOutcome.REFUSE, "; ".join(reasons), constraints

        if domain == ActionDomain.MEMORY_WRITE and memory_commit_mode in {"qualified", "conflicted", "repair_only"}:
            constraints.append(f"memory_commit_mode:{memory_commit_mode}")
        if domain == ActionDomain.MEMORY_WRITE and memory_commit_mode == "defer":
            reasons.append("unity_memory_defer: conflicting drafts too unstable for a clean memory write")
            return WillOutcome.DEFER, "; ".join(reasons), constraints

        if unity_level in {"fragmented", "dissociated"} or not safe_to_act:
            constraints.append(
                f"low_unity:{unity_level}(unity={unity_score:.3f}, fragmentation={fragmentation_score:.3f})"
            )
            if domain == ActionDomain.STABILIZATION:
                return (
                    WillOutcome.CONSTRAIN if constraints else WillOutcome.PROCEED,
                    "stabilization_allowed_under_low_unity",
                    constraints,
                )
            if domain in {ActionDomain.RESPONSE, ActionDomain.EXPRESSION, ActionDomain.REFLECTION, ActionDomain.MEMORY_WRITE}:
                constraints.append("qualified_self_report_only")
            elif domain == ActionDomain.TOOL_EXECUTION and not observation_only_action:
                if catatonia_relief:
                    constraints.append(f"unity_tool_relief:{unity_level}")
                else:
                    reasons.append("unity_block: external action blocked until repair completes")
                    return WillOutcome.REFUSE, "; ".join(reasons), constraints
            elif domain in consequential_domains and not observation_only_action:
                reasons.append(
                    "unity_block: consequential action blocked until repair completes"
                )
                return WillOutcome.REFUSE, "; ".join(reasons), constraints
            elif domain in {ActionDomain.INITIATIVE, ActionDomain.EXPLORATION}:
                if catatonia_relief:
                    constraints.append(f"unity_initiative_relief:{unity_level}")
                else:
                    reasons.append("unity_defer: noncritical initiative deferred until repair completes")
                    return WillOutcome.DEFER, "; ".join(reasons), constraints

        elif unity_level == "strained":
            constraints.append(f"unity_strain: score={unity_score:.3f}")
            if domain == ActionDomain.TOOL_EXECUTION and self._looks_external_social_action(content, context):
                constraints.append("external_action_requires_explicit_caution")
            if domain in {ActionDomain.RESPONSE, ActionDomain.EXPRESSION}:
                constraints.append("qualify_uncertainty_when_self_reporting")

        # ── User-granted action override ────────────────────────────
        # If the current context carries an explicit permission grant
        # from the user, initiative work must not be silently deferred.
        # The whole "she talked about doing it but nothing happened"
        # failure mode was bred by this exact gate returning DEFER.
        user_granted = False
        ctx = context or {}
        if hasattr(ctx, "get"):
            user_granted = bool(
                ctx.get("user_granted_permission")
                or ctx.get("user_explicit_action_request")
                or ctx.get("user_requested_action")
            )

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
            # Decrease assertiveness based on refuse rate; repeated refusals walk it down
            self._state.assertiveness = max(0.15, self._state.assertiveness - 0.02)
        elif decision.outcome == WillOutcome.CRITICAL_PASS:
            self._state.critical_passes += 1

        # Assertiveness is now a learnable parameter reinforced by post-action outcomes
        # via record_outcome(). If no outcome is recorded (e.g. pure responses), we
        # keep the parameter stable.
        pass

        # Periodically refresh identity
        if self._state.total_decisions % 50 == 0:
            self._refresh_identity()

    def _publish_to_consequence_bus(
        self,
        decision: WillDecision,
        domain: ActionDomain,
        source: str,
    ) -> None:
        """Publish the Will decision as pre-action consequence evidence.

        This is not an action-success claim. Actual tool/file/model outcomes
        must still be published by the executor after execution. The Will event
        gives welfare/body learners a governed pre-action trace without
        charging body cost a second time.
        """
        try:
            from core.runtime.consequence_bus import ConsequenceBus

            evidence = dict(decision.aura_now_evidence or {})
            predicted_body_cost = dict(
                evidence.get("body_cost_estimate")
                or evidence.get("body_cost_applied")
                or {}
            )
            actual_body_cost: dict[str, float] = {}
            if decision.is_approved() and predicted_body_cost:
                runtime = ServiceContainer.get("being_runtime", default=None)
                body_service = getattr(runtime, "body_service", None)
                if body_service is None:
                    from core.being.body_state_service import BodyStateService

                    body_service = BodyStateService.get()
                actual_body_cost = body_service.commit_cost(
                    predicted_body_cost,
                    receipt_id=decision.receipt_id,
                )
            actual_outcome = "authorized" if decision.is_approved() else "blocked"
            recovery_required = (
                max(0.0, min(1.0, float(decision.welfare_recovery_drive)))
                if decision.outcome in {WillOutcome.DEFER, WillOutcome.REFUSE}
                else 0.0
            )
            ConsequenceBus.get().publish_action(
                source=source,
                domain=domain.value,
                action_content=decision.reason,
                predicted_welfare_delta={
                    "welfare_score": round(float(decision.welfare_score), 4),
                    "distress": round(float(decision.welfare_distress), 4),
                    "action_inhibition": round(float(decision.welfare_action_inhibition), 4),
                    "recovery_drive": round(float(decision.welfare_recovery_drive), 4),
                },
                predicted_body_cost=predicted_body_cost,
                predicted_memory_risk=0.0,
                predicted_integrity_risk=round(float(decision.welfare_integrity_guard), 4),
                actual_outcome=actual_outcome,
                actual_body_cost=actual_body_cost,
                recovery_required=recovery_required,
                will_receipt_id=decision.receipt_id,
                error=decision.reason if not decision.is_approved() else "",
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "will",
                exc,
                severity="warning",
                action="continued after Will decision consequence publish failed",
                extra={"receipt_id": decision.receipt_id, "domain": domain.value},
            )
            logger.warning("Will consequence publish failed: %s", exc)

    def _record(self, decision: WillDecision) -> None:
        """Record decision in audit trail."""
        if not decision.signature:
            (
                decision.signature,
                decision.signature_scheme,
                decision.signature_key_id,
                decision.signature_root_durable,
            ) = self._sign_decision(decision)
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
                "aura_now_hash": decision.aura_now_hash,
                "aura_now_tick": decision.aura_now_tick,
                "aura_now_policy": decision.aura_now_policy,
                "signature_scheme": decision.signature_scheme,
                "signature_key_id": decision.signature_key_id,
                "timestamp": decision.timestamp,
            })
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("will", exc)
            logger.debug("Will decision event publish failed: %s", exc)


    def _sign_decision(self, decision: WillDecision) -> tuple[str, str, str, bool]:
        payload = self._signature_payload(decision)
        try:
            from core.governance.capability_chain import sign_will_receipt_payload

            signed = sign_will_receipt_payload(payload)
            return (
                str(signed["signature"]),
                str(signed["signature_scheme"]),
                str(signed["signature_key_id"]),
                bool(signed["trust_root_durable"]),
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation('will', exc)
            return hashlib.sha256(payload).hexdigest(), "sha256-fallback", "", False

    @staticmethod
    def _signature_payload(decision: WillDecision) -> bytes:
        payload = {
            "receipt_id": decision.receipt_id,
            "outcome": decision.outcome.value,
            "domain": decision.domain.value,
            "source": decision.source,
            "content_hash": decision.content_hash,
            "timestamp": round(float(decision.timestamp or 0.0), 6),
            "reason": decision.reason,
            "constraints": list(decision.constraints),
            "identity_alignment": decision.identity_alignment.value,
            "substrate_coherence": round(float(decision.substrate_coherence), 6),
            "memory_relevance": round(float(decision.memory_relevance), 6),
            "unity_level": decision.unity_level,
            "unity_score": round(float(decision.unity_score), 6),
            "mind_moment_id": decision.mind_moment_id,
            "causal_closure_score": round(float(decision.causal_closure_score), 6),
            "aura_now_hash": decision.aura_now_hash,
            "aura_now_tick": int(decision.aura_now_tick or 0),
            "aura_now_policy": decision.aura_now_policy,
            "aura_now_constraints": list(decision.aura_now_constraints),
            "aura_now_evidence": dict(decision.aura_now_evidence),
        }
        return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
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
            "catatonia_relief_active": self._state.catatonia_relief_until > time.time(),
            "catatonia_relief_remaining_s": round(
                max(0.0, self._state.catatonia_relief_until - time.time()),
                3,
            ),
            "catatonia_relief_activations": self._state.catatonia_relief_activations,
            "last_catatonia_relief_reason": self._state.last_catatonia_relief_reason,
            "uptime_s": round(time.time() - self._boot_time, 1),
        }

    def get_recent_decisions(self, n: int = 20) -> list[dict[str, Any]]:
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
                "unity_level": d.unity_level,
                "unity_score": round(d.unity_score, 4),
                "fragmentation_score": round(d.fragmentation_score, 4),
                "ownership_confidence": round(d.ownership_confidence, 4),
                "mind_moment_id": d.mind_moment_id,
                "causal_closure_score": round(d.causal_closure_score, 4),
                "aura_now_hash": d.aura_now_hash,
                "aura_now_tick": d.aura_now_tick,
                "aura_now_policy": d.aura_now_policy,
                "aura_now_constraints": list(d.aura_now_constraints),
                "aura_now_evidence": dict(d.aura_now_evidence),
                "signature_scheme": d.signature_scheme,
                "signature_key_id": d.signature_key_id,
                "signature_root_durable": d.signature_root_durable,
                "signature": d.signature,
                "timestamp": d.timestamp,
                "latency_ms": round(d.latency_ms, 3),
            }
            for d in recent
        ]

    def get_recent_refusals(self, n: int = 10) -> list[dict[str, Any]]:
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
        return any(getattr(d, "receipt_id", None) == receipt_id for d in self._audit_trail)

    def verify_receipt_signature(self, receipt_id: str) -> bool:
        """Verify that the receipt exists and its signature matches the payload."""
        for decision in self._audit_trail:
            if getattr(decision, "receipt_id", None) == receipt_id:
                return self.verify_persisted_receipt_material(
                    {
                        "receipt_id": decision.receipt_id,
                        "payload": self._signature_payload(decision).decode("utf-8"),
                        "signature": decision.signature,
                        "signature_scheme": decision.signature_scheme,
                        "signature_key_id": decision.signature_key_id,
                        "trust_root_durable": decision.signature_root_durable,
                    }
                )
        return False

    @staticmethod
    def verify_persisted_receipt_material(material: Mapping[str, Any]) -> bool:
        """Verify embedded receipt material without live audit membership."""

        expected_fields = {
            "receipt_id",
            "payload",
            "signature",
            "signature_scheme",
            "signature_key_id",
            "trust_root_durable",
        }
        if not isinstance(material, Mapping):
            return False
        value = dict(material)
        actual_fields = set(value)
        if actual_fields != expected_fields and actual_fields != expected_fields | {
            "material_sha256"
        }:
            return False
        receipt_id = value.get("receipt_id")
        payload_text = value.get("payload")
        durable = value.get("trust_root_durable")
        if (
            not isinstance(receipt_id, str)
            or not receipt_id
            or not isinstance(payload_text, str)
            or not payload_text
            or not isinstance(durable, bool)
        ):
            return False
        try:
            payload = json.loads(payload_text)
        except (json.JSONDecodeError, TypeError, ValueError):
            return False
        if not isinstance(payload, dict) or payload.get("receipt_id") != receipt_id:
            return False
        try:
            from core.governance.capability_chain import verify_will_receipt_payload

            return verify_will_receipt_payload(
                payload_text.encode("utf-8"),
                signature=str(value.get("signature") or ""),
                signature_scheme=str(value.get("signature_scheme") or ""),
                signature_key_id=str(value.get("signature_key_id") or ""),
                require_durable=durable,
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            return False

    def get_receipt_verification_material(self, receipt_id: str) -> dict[str, Any]:
        """Return payload/signature material for external receipt verification."""
        for decision in self._audit_trail:
            if getattr(decision, "receipt_id", None) == receipt_id:
                return {
                    "receipt_id": decision.receipt_id,
                    "payload": self._signature_payload(decision).decode("utf-8"),
                    "signature": decision.signature,
                    "signature_scheme": decision.signature_scheme,
                    "signature_key_id": decision.signature_key_id,
                    "trust_root_durable": decision.signature_root_durable,
                }
        return {}

    def verify_closure(self, receipt_id: str, effect_verified: bool, telemetry_logged: bool) -> bool:
        """Verify full closure of an action.
        Ensures the receipt exists, the downstream effect was verified,
        and telemetry was logged.
        """
        receipt_exists = self.verify_receipt(receipt_id)
        if not receipt_exists:
            logger.warning("Closure failed: receipt %s not found", receipt_id)
            return False
        if not effect_verified:
            logger.warning("Closure failed: effect not verified for %s", receipt_id)
            return False
        if not telemetry_logged:
            logger.warning("Closure failed: telemetry not logged for %s", receipt_id)
            return False
        return True

    def record_outcome(self, receipt_id: str, tx_record: Any) -> None:
        """Reinforce the assertiveness parameter based on post-action outcomes."""
        try:
            outcome = getattr(tx_record, "outcome", "failure")
            w_delta = getattr(tx_record, "welfare_delta", {}) or {}
            b_delta = getattr(tx_record, "body_delta", {}) or {}
            integrity_preserved = getattr(tx_record, "integrity_preserved", True)
            truth_preserved = getattr(tx_record, "truth_preserved", True)

            # Outcome reward signal calculation
            reward = 0.1 if outcome == "success" else -0.1
            if not integrity_preserved:
                reward -= 0.5
            if not truth_preserved:
                reward -= 0.3

            distress_spike = _bounded_delta(w_delta, "distress")
            if distress_spike > 0.0:
                reward -= 2.0 * distress_spike

            relief = _bounded_delta(w_delta, "relief")
            if relief > 0.0:
                reward += 0.5 * relief

            fatigue_spike = _bounded_delta(b_delta, "fatigue")
            if fatigue_spike > 0.0:
                reward -= 0.2 * fatigue_spike

            # Apply gradient-free reinforcement learning update
            lr = 0.05
            self._state.assertiveness = max(0.15, min(0.95, self._state.assertiveness + lr * reward))
            logger.info(
                "UnifiedWill: outcome reinforced for receipt %s: outcome=%s, reward=%.3f, updated assertiveness=%.3f",
                receipt_id,
                outcome,
                reward,
                self._state.assertiveness,
            )
        except _WILL_OUTCOME_REINFORCEMENT_ERRORS as exc:
            record_degradation(
                "unified_will",
                exc,
                action="ignored malformed post-action outcome reinforcement signal",
            )
            logger.debug("Failed to record outcome in Will: %s", exc)



# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_will_instance: UnifiedWill | None = None
_will_instance_lock = checked_lock("unified_will.singleton")


def get_will() -> UnifiedWill:
    """Get the singleton UnifiedWill instance."""
    global _will_instance
    if _will_instance is None:
        with _will_instance_lock:
            if _will_instance is None:
                _will_instance = UnifiedWill()
    _will_instance.ensure_started()
    return _will_instance


def reset_unified_will_for_test() -> None:
    """Deactivate and forget the process singleton between hermetic tests."""
    global _will_instance
    with _will_instance_lock:
        will, _will_instance = _will_instance, None
    if will is not None:
        will._started = False
