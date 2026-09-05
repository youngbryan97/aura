"""core/executive/executive_core.py — ZENITH ExecutiveCore v1.0
The single sovereign control plane for Aura.

Every meaningful operation — tool execution, response emission, state mutation,
memory commit, and background task spawn — must request approval from the
ExecutiveCore. This is NOT a coordinator; it is a governor.

Invariant:
    Nothing user-visible or world-affecting happens unless ExecutiveCore
    has assigned or approved an Intent.

Design:
    - Wraps and extends the existing ExecutiveAuthority (spontaneous message gate)
    - Adds approval gates for tools, state, memory, and background tasks
    - Integrates with BindingEngine for coherence-aware decisions
    - Integrates with CanonicalSelf for identity-aware decisions
    - Maintains a full audit ledger of all decisions
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.container import ServiceContainer
from core.executive.bounded_sandbox_policy import validate_idle_sandbox_probe_arguments
from core.executive.executive_ledger import ExecutiveLedger
from core.goals.goal_text import (
    first_actionable_goal_text,
    is_actionable_goal_text,
    is_intrinsic_goal_text,
)
from core.memory.retention_policy import working_history_retention_policy
from core.runtime.errors import record_degradation
from core.runtime.service_access import (
    resolve_canonical_self,
    resolve_canonical_self_engine,
    resolve_state_repository,
)
from core.state.aura_state import _is_speculative_autonomy_label, _normalize_goal_text

logger = logging.getLogger("Aura.Executive")


_DECISION_HISTORY_LIMIT = working_history_retention_policy(
    "AURA_EXECUTIVE_DECISION_HISTORY_MAX"
).max_items


_AUTONOMOUS_RESEARCH_SOURCE_ALIASES = frozenset(
    {
        "autonomous_research",
        "action_consequence_graph",
        "content_fetcher",
        "content_method_router",
        "curiosity",
        "curiosity_daemon",
        "curiosity_engine",
        "curiosity_explorer",
        "curiosity_scheduler",
        "empirical_observation",
        "external_evidence",
        "free_search",
        "grounded_search",
        "knowledge_curiosity_finding",
        "knowledge_research_finding",
        "learned_from_web",
        "local_corpus",
        "research",
        "research_cycle",
        "research_pipeline",
        "search_web",
        "tool_execution",
        "tool_result",
        "tool_result_evidence",
        "runtime_evidence",
        "web_learning",
        "web_retained",
        "web_search",
    }
)
_AUTONOMOUS_RESEARCH_SOURCE_PREFIXES = (
    "autonomous_research:",
    "action_consequence_graph:",
    "curiosity:",
    "curiosity_engine:",
    "empirical_observation:",
    "free_search:",
    "grounded_search:",
    "knowledge:curiosity",
    "knowledge:research",
    "learned_from_web:",
    "research:",
    "research_pipeline:",
    "search_web:",
    "tool_execution:",
    "tool_result:",
    "tool_result_evidence:",
    "runtime_evidence:",
    "web_learning:",
    "web_retained:",
    "web_search:",
)
_MAINTENANCE_SOURCE_ALIASES = frozenset(
    {
        "peer_mode",
        "repair_loop",
        "runtime_repair",
        "self_repair",
        "sovereign_self_modification",
        "system_maintenance",
    }
)
_MAINTENANCE_SOURCE_PREFIXES = (
    "peer_mode:",
    "repair_loop:",
    "runtime_repair:",
    "self_repair:",
    "sovereign_self_modification:",
    "system_maintenance:",
)


def _normalize_intent_source_label(source: str) -> str:
    return str(source or "").strip().lower().replace("-", "_")


def _is_autonomous_research_source(source: str) -> bool:
    normalized = _normalize_intent_source_label(source)
    if not normalized:
        return False
    normalized_token = normalized.replace(":", "_")
    return (
        normalized in _AUTONOMOUS_RESEARCH_SOURCE_ALIASES
        or normalized_token in _AUTONOMOUS_RESEARCH_SOURCE_ALIASES
        or normalized.startswith(_AUTONOMOUS_RESEARCH_SOURCE_PREFIXES)
    )


def _is_transient_conversation_memory_objective(text: str) -> bool:
    normalized = _normalize_goal_text(text).lower()
    if not normalized.startswith("remember this "):
        return False
    if not any(token in normalized for token in ("note", "phrase", "word", "token", "codeword", "detail")):
        return False
    return "later in this conversation" in normalized or ":" in normalized


def _coerce_intent_source(source: str) -> IntentSource:
    normalized = _normalize_intent_source_label(source)
    user_aliases = {
        "api",
        "api.skill.execute",
        "api_skill",
        "api_skill_execute",
        "admin",
        "chat",
        "chat_api",
        "desktop",
        "desktop_ui",
        "desktop_task",
        "voice",
        "voice_bridge",
        "voice_input",
        "gui",
        "live_chat",
        "live_skill_api",
        "ws",
        "websocket",
        "direct",
        "external",
        "frontend",
        "ui",
        "embodied",
        "embodied_motor_reflex",
        "embodied_sensory_feed",
        "reflex",
        "motor",
    }
    if normalized in user_aliases:
        return IntentSource.USER
    if normalized in _MAINTENANCE_SOURCE_ALIASES or normalized.startswith(_MAINTENANCE_SOURCE_PREFIXES):
        return IntentSource.MAINTENANCE
    if _is_autonomous_research_source(normalized):
        return IntentSource.AUTONOMOUS_RESEARCH
    for candidate in IntentSource:
        if candidate.value == normalized:
            return candidate
    return IntentSource.AUTONOMOUS


# ── Data Structures ──────────────────────────────────────────────────────────

class IntentSource(str, Enum):
    USER = "user"
    DRIVE = "drive"
    REFLECTION = "reflection"
    MAINTENANCE = "maintenance"
    SOCIAL = "social"
    AUTONOMOUS = "autonomous"
    # Self-initiated research outputs (curated content consumption, knowledge-gap
    # closure). Distinguished from generic AUTONOMOUS so Rule 7 epistemic
    # reconciliation can let provisional research writes through instead of
    # deferring them indefinitely. Consumers should commit research-derived
    # claims with provisional confidence; durable promotion happens via the
    # normal reconciliation pathway. See scoping/cortex-break-diagnosis.md.
    AUTONOMOUS_RESEARCH = "autonomous_research"
    SYSTEM = "system"
    BACKGROUND = "background"


class ActionType(str, Enum):
    RESPOND = "respond"
    TOOL_CALL = "tool_call"
    REFLECT = "reflect"
    UPDATE_BELIEF = "update_belief"
    WRITE_MEMORY = "write_memory"
    MUTATE_STATE = "mutate_state"
    SPAWN_TASK = "spawn_task"
    EMIT_MESSAGE = "emit_message"
    IDLE = "idle"


class DecisionOutcome(str, Enum):
    APPROVED = "approved"
    DEFERRED = "deferred"
    REJECTED = "rejected"
    DEGRADED = "degraded"  # approved but with constraints


@dataclass
class Intent:
    """A proposed action that requires executive approval."""
    intent_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    source: IntentSource = IntentSource.USER
    goal: str = ""
    action_type: ActionType = ActionType.RESPOND
    payload: dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5  # 0-1
    confidence: float = 0.5  # 0-1
    blocking: bool = False  # does this block other operations?
    requires_tool: bool = False
    requires_memory_commit: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class DecisionRecord:
    """Record of an executive decision."""
    intent_id: str
    outcome: DecisionOutcome
    reason: str
    coherence_at_decision: float = 1.0
    identity_check: bool = True  # did this pass identity assertion?
    constraints: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "coherence": self.coherence_at_decision,
            "identity_check": self.identity_check,
            "constraints": self.constraints,
            "timestamp": self.timestamp,
        }


# ── Policy Constants ─────────────────────────────────────────────────────────

# When coherence drops below this, only essential operations proceed
COHERENCE_LOCKDOWN_THRESHOLD = 0.30

# When coherence is below this, degrade non-essential operations
COHERENCE_DEGRADE_THRESHOLD = 0.50

# Maximum concurrent approved intents (raised from 8 — stale intents were
# filling the queue and blocking cognitive_cycle state mutations)
MAX_CONCURRENT_INTENTS = 24

# Operations that ALWAYS proceed (even in lockdown)
ESSENTIAL_ACTIONS = {
    ActionType.RESPOND,      # always respond to the user
    ActionType.IDLE,
}

# Operations that are blocked during lockdown
LOCKDOWN_BLOCKED = {
    ActionType.SPAWN_TASK,
    ActionType.EMIT_MESSAGE,
    ActionType.REFLECT,
}

# Tool names that are always safe (don't need extra scrutiny)
SAFE_TOOLS = {
    "clock", "environment_info", "system_proprioception",
    "personality_skill", "memory_ops", "query_beliefs",
}

TEMPORAL_SAFE_AUTONOMOUS_TOOLS = {
    "clock",
    "email_adapter",
    "environment_info",
    "query_beliefs",
    "reddit_adapter",
    "swarm_debate",
    "system_proprioception",
    "test_generator",
    "web_search",
    "sovereign_browser",
    "sovereign_network",
}

# Tool names that require identity assertion before execution
IDENTITY_SENSITIVE_TOOLS = {
    "self_evolution", "self_repair", "self_improvement",
    "auto_refactor", "train_self",
}

# Tools that are allowed to bypass lockdown for recovery
RECOVERY_AND_EVOLUTION_TOOLS = {
    "web_search", "sovereign_browser", "self_repair", "auto_refactor",
    "self_evolution", "train_self", "memory_ops", "query_beliefs",
    "system_proprioception", "process_supervisor",
}


# ── Executive Core ───────────────────────────────────────────────────────────

class ExecutiveCore:
    """The single sovereign control plane.

    All significant operations request approval. The executive decides
    based on coherence, identity, resource state, and current intent queue.
    """

    def __init__(self) -> None:
        self._active_intents: dict[str, Intent] = {}
        self._decision_history: deque[DecisionRecord] = deque(maxlen=_DECISION_HISTORY_LIMIT)
        self._approval_count: int = 0
        self._rejection_count: int = 0
        self._lock = asyncio.Lock()
        self._initialized = False
        self._ledger: ExecutiveLedger | None = None
        #: intent_id -> ontogeny episode, so a completion can grade the
        #: admission that allowed it. Bounded by the active-intent lifetime.
        self._ontogeny_episodes: dict[str, str] = {}
        logger.info("🏛️ ExecutiveCore initialized — sovereign control plane active.")

    # ── Core Approval API ────────────────────────────────────────────────

    async def request_approval(self, intent: Intent) -> DecisionRecord:
        """Request approval for an operation.

        This is the ONLY entry point for getting permission to act.
        Returns a DecisionRecord with outcome and constraints.
        """
        async with self._lock:
            return await self._evaluate(intent)

    def request_approval_sync(self, intent: Intent) -> DecisionRecord:
        """Synchronous approval for non-async contexts.

        Uses a more lenient policy since we can't check async services.
        """
        self._sweep_stale_intents_sync()
        return self._evaluate_sync(intent)

    # ── Convenience Methods ──────────────────────────────────────────────

    async def approve_tool(self, tool_name: str, args: dict[str, Any],
                           source: str = "unknown") -> tuple[bool, str, dict]:
        """Quick check: should this tool execution proceed?

        Returns (approved, reason, constraints).
        """
        intent, record = await self.prepare_tool_intent(tool_name, args, source=source)
        approved = record.outcome in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED)
        if approved:
            self.complete_intent(intent.intent_id, success=True)
        return (approved, record.reason, record.constraints)

    async def prepare_tool_intent(
        self,
        tool_name: str,
        args: dict[str, Any],
        source: str = "unknown",
    ) -> tuple[Intent, DecisionRecord]:
        """Build and evaluate a tool-call intent while preserving the intent id."""
        intent = Intent(
            source=_coerce_intent_source(source),
            goal=f"execute_tool:{tool_name}",
            action_type=ActionType.TOOL_CALL,
            payload={"tool_name": tool_name, "args": args},
            requires_tool=True,
        )

        # User-initiated tools always approved
        if _coerce_intent_source(source) == IntentSource.USER:
            intent.source = IntentSource.USER
            intent.priority = 0.9

        record = await self.request_approval(intent)
        return intent, record

    async def approve_emission(self, content: str, source: str = "unknown",
                               urgency: float = 0.5) -> tuple[bool, str]:
        """Quick check: should this spontaneous message be emitted?"""
        intent = Intent(
            source=IntentSource.SOCIAL if source == "proactive_presence" else IntentSource.AUTONOMOUS,
            goal=f"emit_message:{content[:40]}",
            action_type=ActionType.EMIT_MESSAGE,
            payload={"content": content, "source": source},
            priority=urgency,
        )
        record = await self.request_approval(intent)
        if record.outcome in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED):
            self.complete_intent(intent.intent_id, success=True)
        return (
            record.outcome in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED),
            record.reason,
        )

    async def approve_memory_write(self, memory_type: str, content: str,
                                    importance: float = 0.5,
                                    source: str = "unknown") -> tuple[bool, str]:
        """Quick check: should this memory be committed?"""
        intent = Intent(
            source=IntentSource.SYSTEM,
            goal=f"write_memory:{memory_type}",
            action_type=ActionType.WRITE_MEMORY,
            payload={"type": memory_type, "content": content[:200], "importance": importance},
            priority=importance,
            requires_memory_commit=True,
        )
        record = await self.request_approval(intent)
        if record.outcome in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED):
            self.complete_intent(intent.intent_id, success=True)
        return (
            record.outcome in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED),
            record.reason,
        )

    async def approve_state_mutation(self, origin: str, cause: str) -> tuple[bool, str]:
        """Quick check: should this state mutation proceed?"""
        intent = Intent(
            source=IntentSource.SYSTEM,
            goal=f"mutate_state:{origin}",
            action_type=ActionType.MUTATE_STATE,
            payload={"origin": origin, "cause": cause},
        )
        record = await self.request_approval(intent)
        if record.outcome in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED):
            self.complete_intent(intent.intent_id, success=True)
        return (
            record.outcome in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED),
            record.reason,
        )

    async def approve_background_task(self, task_name: str,
                                       source: str = "unknown") -> tuple[bool, str]:
        """Quick check: should this background task be spawned?"""
        intent = Intent(
            source=IntentSource.BACKGROUND,
            goal=f"spawn_task:{task_name}",
            action_type=ActionType.SPAWN_TASK,
            payload={"task_name": task_name, "source": source},
        )
        record = await self.request_approval(intent)
        if record.outcome in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED):
            self.complete_intent(intent.intent_id, success=True)
        return (
            record.outcome in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED),
            record.reason,
        )

    # ── Internal Evaluation ──────────────────────────────────────────────

    async def _sweep_stale_intents(self) -> None:
        """Evict intents older than 90s to prevent capacity lockout.

        Autonomous/system intents often have no explicit completion path,
        so they linger in _active_intents and eventually hit the capacity cap,
        blocking all non-essential state mutations.  TTL set to 90s: long enough
        for legitimate background ops (memory consolidation, episodic recall)
        but short enough to prevent queue exhaustion during cognitive bursts.
        """
        now = time.time()
        stale = [
            iid for iid, intent in self._active_intents.items()
            if (now - intent.timestamp) > 90.0
        ]
        for iid in stale:
            self._active_intents.pop(iid, None)
        if stale:
            logger.info("♻️ Executive: swept %d stale intents (TTL 90s)", len(stale))

    def _sweep_stale_intents_sync(self) -> None:
        now = time.time()
        stale = [
            iid for iid, intent in self._active_intents.items()
            if (now - intent.timestamp) > 90.0
        ]
        for iid in stale:
            self._active_intents.pop(iid, None)
        if stale:
            logger.info("♻️ Executive: swept %d stale intents (TTL 90s, sync)", len(stale))

    @staticmethod
    def _temporal_safe_autonomous_tool_constraints(
        intent: Intent,
    ) -> dict[str, Any] | None:
        if intent.action_type != ActionType.TOOL_CALL:
            return None
        tool_name = str(intent.payload.get("tool_name", "") or "").strip()
        if tool_name == "process_supervisor":
            return {"timeout_s": 45, "read_only": True}
        if tool_name == "subconscious_sandbox_probe":
            args = intent.payload.get("args", {}) or {}
            valid, _reason = validate_idle_sandbox_probe_arguments(args)
            if not valid:
                return None
            return {
                "timeout_s": 30,
                "sandboxed_compute": True,
                "network_access": False,
            }
        if tool_name in TEMPORAL_SAFE_AUTONOMOUS_TOOLS:
            args = intent.payload.get("args", {}) or {}
            mode = str(args.get("mode") or "").strip().lower()
            if tool_name == "email_adapter":
                if mode not in {"", "check", "read", "search"}:
                    return None
            if tool_name == "reddit_adapter":
                if mode not in {
                    "",
                    "browse",
                    "read_post",
                    "check_inbox",
                    "read_rules",
                    "check_shadowban",
                }:
                    return None
            return {"timeout_s": 45, "read_only": True}
        if tool_name == "auto_refactor":
            args = intent.payload.get("args", {}) or {}
            if not bool(args.get("run_tests")):
                return {"timeout_s": 45, "read_only": True}
            return None
        if tool_name == "self_evolution":
            args = intent.payload.get("args", {}) or {}
            action = str(args.get("action") or "propose").strip().lower()
            if action in {"", "propose"}:
                return {"timeout_s": 45, "read_only": True}
            return None
        if tool_name == "memory_ops":
            args = intent.payload.get("args", {}) or {}
            action = str(args.get("action") or args.get("mode") or "").strip().lower()
            if action in {"", "recall", "search", "query", "read"}:
                return {"timeout_s": 45, "read_only": True}
        return None

    @classmethod
    def _is_temporal_safe_autonomous_tool(cls, intent: Intent) -> bool:
        return cls._temporal_safe_autonomous_tool_constraints(intent) is not None

    async def _evaluate(self, intent: Intent) -> DecisionRecord:
        """Core evaluation logic. All approval paths converge here."""

        # Sweep stale intents to prevent capacity lockout
        await self._sweep_stale_intents()
        strict_runtime = self._strict_runtime_active()

        # Rule 1: User-facing operations ALWAYS proceed
        if intent.source == IntentSource.USER:
            return self._approve(intent, "user_facing")

        # Rule 2: Essential actions always proceed
        if intent.action_type in ESSENTIAL_ACTIONS:
            return self._approve(intent, "essential_action")

        # Rule 3: Identity integrity is not optional for self-shaping/autonomous operations.
        if (
            strict_runtime
            and intent.source != IntentSource.USER
            and intent.action_type in {
                ActionType.TOOL_CALL,
                ActionType.EMIT_MESSAGE,
                ActionType.SPAWN_TASK,
                ActionType.UPDATE_BELIEF,
                ActionType.WRITE_MEMORY,
            }
            and not self._identity_integrity_available()
        ):
            return self._reject(intent, "self_model_required")

        # Rule 4: Global failure identity. When the organism is degraded, non-essential
        # autonomous actions must feel that failure everywhere.
        failure_state = self._get_failure_state()
        if strict_runtime:
            is_recovery = (
                intent.action_type == ActionType.TOOL_CALL
                and intent.payload.get("tool_name") in RECOVERY_AND_EVOLUTION_TOOLS
            ) or intent.source == IntentSource.AUTONOMOUS_RESEARCH

            # RECOVERY EXCEPTION: If this is a recovery tool or research, and priority is > 0.4,
            # allow it to proceed even in high pressure lockdown.
            if is_recovery and intent.priority >= 0.4:
                # Proceed to Rule 5, effectively bypassing Rule 4 lockdown
                pass
            else:
                if (
                    int(failure_state.get("critical", 0) or 0) >= 1
                    and int(failure_state.get("count", 0) or 0) >= 3
                    and intent.source != IntentSource.USER
                ):
                    return self._reject(intent, f"unified_failure_lockdown_{failure_state['pressure']:.2f}")
                if failure_state["pressure"] >= 0.85:
                    return self._reject(intent, f"unified_failure_lockdown_{failure_state['pressure']:.2f}")
                if (
                    failure_state["pressure"] >= 0.45
                    and intent.action_type in LOCKDOWN_BLOCKED | {ActionType.TOOL_CALL, ActionType.UPDATE_BELIEF, ActionType.WRITE_MEMORY}
                    and intent.priority < 0.9
                ):
                    return self._defer(intent, f"failure_pressure_{failure_state['pressure']:.2f}")

        # Rule 5: Temporal identity lock. Existing commitments and unfinished work
        # constrain later background behavior until reconciled.
        temporal = self._get_temporal_identity_context()
        if (
            strict_runtime
            and temporal["obligation_pressure"] > 0.0
            and intent.source in {
                IntentSource.AUTONOMOUS,
                IntentSource.BACKGROUND,
                IntentSource.SOCIAL,
                IntentSource.DRIVE,
                IntentSource.REFLECTION,
            }
            # EMIT_MESSAGE is deliberately NOT here.
            #
            # This rule exists so unfinished work constrains STARTING MORE
            # WORK. Saying something to the person is not competing work: it
            # spawns nothing, holds nothing, and finishes when the sentence
            # ends. Deferring it because a background research goal is open is
            # a category error, and it produced a permanent gag.
            #
            # Measured live 2026-08-10: 44 proactive initiations generated, 44
            # suppressed, 0 ever spoken — `seconds_since_spoke: None` after
            # 1008 ambient ticks — every one deferred with
            # "temporal_obligation_active:Find the most obscure fact about
            # xenobiology concepts."
            #
            # That is a structural deadlock, not a tuning problem. The gate
            # closes on ANY nonzero obligation pressure, and the pressure is
            # cleared by finishing autonomous work that this same gate defers.
            # A goal that can only be discharged by autonomous action, blocking
            # all autonomous action, can never be discharged. It had been
            # holding since a stale goal list persisted in continuity.json.
            #
            # SPAWN_TASK, TOOL_CALL and REFLECT stay: those genuinely compete
            # with unfinished work, and deferring them is the rule doing its
            # job.
            and intent.action_type in {
                ActionType.SPAWN_TASK,
                ActionType.TOOL_CALL,
                ActionType.REFLECT,
            }
            and intent.priority < 0.85
        ):
            temporal_constraints = self._temporal_safe_autonomous_tool_constraints(intent)
            if temporal_constraints is not None:
                return self._degrade(
                    intent,
                    "temporal_safe_autonomous_tool",
                    1.0,
                    constraints=temporal_constraints,
                )
            return self._defer(
                intent,
                f"temporal_obligation_active:{temporal['anchor']}",
            )

        # Rule 6: Internal states are causally binding for autonomous behavior.
        internal_state = self._get_internal_state_constraints()
        if strict_runtime and intent.source in {
            IntentSource.AUTONOMOUS,
            IntentSource.BACKGROUND,
            IntentSource.SOCIAL,
            IntentSource.DRIVE,
            IntentSource.REFLECTION,
        }:
            # RECOVERY EXCEPTION for Rule 6
            is_recovery = (
                intent.action_type == ActionType.TOOL_CALL
                and intent.payload.get("tool_name") in RECOVERY_AND_EVOLUTION_TOOLS
            ) or intent.source == IntentSource.AUTONOMOUS_RESEARCH
            
            if is_recovery and intent.priority >= 0.4:
                # Bypass Rule 6 pressure checks
                pass
            else:
                if internal_state["identity_mismatch"] and intent.action_type in {
                    ActionType.SPAWN_TASK,
                    ActionType.EMIT_MESSAGE,
                    ActionType.TOOL_CALL,
                    ActionType.UPDATE_BELIEF,
                    ActionType.WRITE_MEMORY,
                    ActionType.MUTATE_STATE,
                }:
                    return self._reject(intent, "identity_continuity_mismatch")
                if internal_state["thermal_pressure"] >= 0.92: # Relaxed from 0.85
                    return self._defer(intent, f"internal_state_thermal_pressure:{internal_state['thermal_pressure']:.2f}")
                if internal_state["load_pressure"] >= 0.95: # Relaxed from 0.9
                    return self._defer(intent, f"internal_state_load_pressure:{internal_state['load_pressure']:.2f}")
                if internal_state["energy"] <= 0.10 and intent.priority < 0.85: # Relaxed from 0.15/0.95
                    return self._defer(intent, f"internal_state_energy_low:{internal_state['energy']:.2f}")
                if internal_state["distress"] >= 0.9 and intent.priority < 0.85: # Relaxed from 0.8/0.95
                    return self._defer(intent, f"internal_state_distress:{internal_state['distress']:.2f}")

        # Rule 7: Closed-loop epistemology.
        #
        # Original intent of this rule: prevent silent accumulation of durable
        # beliefs when contradictions are unresolved. That goal is correct.
        #
        # Refinement (2026-04-27): contested beliefs should INVITE research and
        # debate, not block all autonomous belief work. The split:
        #
        #   • IntentSource.AUTONOMOUS_RESEARCH writes are permitted through —
        #     this is exactly the activity that resolves contestation. They are
        #     auto-tagged provisional via payload["confidence_tier"] so the
        #     consumer commits at provisional confidence and queues for
        #     reconciliation rather than entering as durable beliefs.
        #
        #   • USER writes are permitted through (unchanged).
        #
        #   • Other autonomous sources writing UPDATE_BELIEF / WRITE_MEMORY
        #     while contested are still deferred — but with a side-effect:
        #     the contested belief is surfaced as a research-trigger so the
        #     research pipeline picks it up next cycle. Deferral is no longer
        #     a dead end; it queues the question.
        epistemic = self._get_epistemic_state()
        if strict_runtime and intent.action_type in {ActionType.UPDATE_BELIEF, ActionType.WRITE_MEMORY}:
            if intent.source == IntentSource.AUTONOMOUS_RESEARCH:
                # Mark as provisional so memory_facade / belief_graph store it
                # behind a reconciliation gate rather than as a durable claim.
                intent.payload.setdefault("confidence_tier", "provisional")
                intent.payload.setdefault("requires_reconciliation", True)
                # Fall through to subsequent rules.
            elif (
                epistemic["contested"] > 0
                and intent.source != IntentSource.USER
                and intent.priority < 0.9
                and self._intent_touches_contested_topic(intent, epistemic)
            ):
                # Surface contested topic to the research pipeline rather than
                # silently dropping the work. Best-effort: never block on this.
                self._surface_research_trigger(intent, epistemic)
                return self._defer(intent, f"epistemic_reconciliation_required:{epistemic['contested']}")

        # Rule 8: Check coherence
        coherence = await self._get_coherence()

        # Lockdown mode: only essential operations
        if coherence < COHERENCE_LOCKDOWN_THRESHOLD:
            if intent.action_type in LOCKDOWN_BLOCKED:
                if (
                    intent.action_type == ActionType.EMIT_MESSAGE
                    and intent.source == IntentSource.USER
                ):
                    # Silence toward the user is worse than constrained
                    # speech: a user-facing reply under lockdown degrades
                    # (short, slow, cautious) but is never muted outright.
                    return self._degrade(intent,
                        f"coherence_lockdown_user_speech_{coherence:.2f}",
                        coherence,
                        constraints={"max_tokens": 192, "timeout_s": 30})
                return self._reject(intent,
                    f"coherence_lockdown_{coherence:.2f}",
                    coherence)
            # Degrade non-blocked operations
            return self._degrade(intent,
                f"coherence_degraded_{coherence:.2f}",
                coherence,
                constraints={"max_tokens": 256, "timeout_s": 30})

        # Rule 9: Identity-sensitive tools require assertion
        if intent.action_type == ActionType.TOOL_CALL:
            tool_name = intent.payload.get("tool_name", "")
            if tool_name in IDENTITY_SENSITIVE_TOOLS:
                identity_ok = await self._check_identity(intent)
                if not identity_ok:
                    return self._reject(intent,
                        f"identity_assertion_failed:{tool_name}",
                        coherence)

        # Rule 10: Degrade mode for low coherence
        if coherence < COHERENCE_DEGRADE_THRESHOLD:
            return self._degrade(intent,
                f"coherence_caution_{coherence:.2f}",
                coherence,
                constraints={"max_tokens": 512})

        # Rule 11: Capacity check
        active_count = len(self._active_intents)
        if active_count >= MAX_CONCURRENT_INTENTS:
            is_recovery = (
                intent.action_type == ActionType.TOOL_CALL
                and intent.payload.get("tool_name") in RECOVERY_AND_EVOLUTION_TOOLS
            ) or intent.source == IntentSource.AUTONOMOUS_RESEARCH
            
            if not is_recovery and intent.priority < 0.8: # Recovery tools bypass capacity check
                return self._defer(intent,
                    f"capacity_full_{active_count}/{MAX_CONCURRENT_INTENTS}")

        # Rule 12: Default approve
        return self._approve(intent, "approved", coherence)

    def _evaluate_sync(self, intent: Intent) -> DecisionRecord:
        """Synchronous evaluation path for write-gated legacy callers."""
        strict_runtime = self._strict_runtime_active()

        if intent.source == IntentSource.USER:
            return self._approve(intent, "user_facing_fast_path")

        if intent.action_type in ESSENTIAL_ACTIONS:
            return self._approve(intent, "essential_action")

        if (
            strict_runtime
            and intent.source != IntentSource.USER
            and intent.action_type in {
                ActionType.TOOL_CALL,
                ActionType.EMIT_MESSAGE,
                ActionType.SPAWN_TASK,
                ActionType.UPDATE_BELIEF,
                ActionType.WRITE_MEMORY,
            }
            and not self._identity_integrity_available()
        ):
            return self._reject(intent, "self_model_required")

        failure_state = self._get_failure_state()
        if strict_runtime:
            is_recovery = (
                intent.action_type == ActionType.TOOL_CALL
                and intent.payload.get("tool_name") in RECOVERY_AND_EVOLUTION_TOOLS
            ) or intent.source == IntentSource.AUTONOMOUS_RESEARCH

            if is_recovery and intent.priority >= 0.4:
                # Bypass lockdown
                pass
            else:
                if failure_state["pressure"] >= 0.85:
                    return self._reject(intent, f"unified_failure_lockdown_{failure_state['pressure']:.2f}")
                if (
                    failure_state["pressure"] >= 0.45
                    and intent.action_type in LOCKDOWN_BLOCKED | {ActionType.TOOL_CALL, ActionType.UPDATE_BELIEF, ActionType.WRITE_MEMORY}
                    and intent.priority < 0.9
                ):
                    return self._defer(intent, f"failure_pressure_{failure_state['pressure']:.2f}")

        temporal = self._get_temporal_identity_context()
        if (
            strict_runtime
            and temporal["obligation_pressure"] > 0.0
            and intent.source in {
                IntentSource.AUTONOMOUS,
                IntentSource.BACKGROUND,
                IntentSource.SOCIAL,
                IntentSource.DRIVE,
                IntentSource.REFLECTION,
            }
            # EMIT_MESSAGE is deliberately NOT here.
            #
            # This rule exists so unfinished work constrains STARTING MORE
            # WORK. Saying something to the person is not competing work: it
            # spawns nothing, holds nothing, and finishes when the sentence
            # ends. Deferring it because a background research goal is open is
            # a category error, and it produced a permanent gag.
            #
            # Measured live 2026-08-10: 44 proactive initiations generated, 44
            # suppressed, 0 ever spoken — `seconds_since_spoke: None` after
            # 1008 ambient ticks — every one deferred with
            # "temporal_obligation_active:Find the most obscure fact about
            # xenobiology concepts."
            #
            # That is a structural deadlock, not a tuning problem. The gate
            # closes on ANY nonzero obligation pressure, and the pressure is
            # cleared by finishing autonomous work that this same gate defers.
            # A goal that can only be discharged by autonomous action, blocking
            # all autonomous action, can never be discharged. It had been
            # holding since a stale goal list persisted in continuity.json.
            #
            # SPAWN_TASK, TOOL_CALL and REFLECT stay: those genuinely compete
            # with unfinished work, and deferring them is the rule doing its
            # job.
            and intent.action_type in {
                ActionType.SPAWN_TASK,
                ActionType.TOOL_CALL,
                ActionType.REFLECT,
            }
            and intent.priority < 0.85
        ):
            temporal_constraints = self._temporal_safe_autonomous_tool_constraints(intent)
            if temporal_constraints is not None:
                return self._degrade(
                    intent,
                    "temporal_safe_autonomous_tool",
                    1.0,
                    constraints=temporal_constraints,
                )
            return self._defer(
                intent,
                f"temporal_obligation_active:{temporal['anchor']}",
            )

        internal_state = self._get_internal_state_constraints()
        if strict_runtime and intent.source in {
            IntentSource.AUTONOMOUS,
            IntentSource.BACKGROUND,
            IntentSource.SOCIAL,
            IntentSource.DRIVE,
            IntentSource.REFLECTION,
            IntentSource.SYSTEM,
        }:
            if internal_state["identity_mismatch"] and intent.action_type in {
                ActionType.SPAWN_TASK,
                ActionType.EMIT_MESSAGE,
                ActionType.TOOL_CALL,
                ActionType.UPDATE_BELIEF,
                ActionType.WRITE_MEMORY,
                ActionType.MUTATE_STATE,
            }:
                return self._reject(intent, "identity_continuity_mismatch")
            if internal_state["thermal_pressure"] >= 0.85:
                return self._defer(intent, f"internal_state_thermal_pressure:{internal_state['thermal_pressure']:.2f}")
            if internal_state["load_pressure"] >= 0.9:
                return self._defer(intent, f"internal_state_load_pressure:{internal_state['load_pressure']:.2f}")
            if internal_state["energy"] <= 0.15 and intent.priority < 0.95:
                return self._defer(intent, f"internal_state_energy_low:{internal_state['energy']:.2f}")
            if internal_state["distress"] >= 0.8 and intent.priority < 0.95:
                return self._defer(intent, f"internal_state_distress:{internal_state['distress']:.2f}")

        epistemic = self._get_epistemic_state()
        if strict_runtime and intent.action_type in {ActionType.UPDATE_BELIEF, ActionType.WRITE_MEMORY}:
            if intent.source == IntentSource.AUTONOMOUS_RESEARCH:
                intent.payload.setdefault("confidence_tier", "provisional")
                intent.payload.setdefault("requires_reconciliation", True)
            elif (
                epistemic["contested"] > 0
                and intent.source != IntentSource.USER
                and intent.priority < 0.9
                and self._intent_touches_contested_topic(intent, epistemic)
            ):
                self._surface_research_trigger(intent, epistemic)
                return self._defer(intent, f"epistemic_reconciliation_required:{epistemic['contested']}")

        coherence = self._get_coherence_sync()
        if coherence < COHERENCE_LOCKDOWN_THRESHOLD:
            if intent.action_type in LOCKDOWN_BLOCKED:
                return self._reject(intent, f"coherence_lockdown_{coherence:.2f}", coherence)
            return self._degrade(
                intent,
                f"coherence_degraded_{coherence:.2f}",
                coherence,
                constraints={"max_tokens": 256, "timeout_s": 30},
            )

        if intent.action_type == ActionType.TOOL_CALL:
            tool_name = intent.payload.get("tool_name", "")
            if tool_name in IDENTITY_SENSITIVE_TOOLS:
                identity_ok = self._check_identity_sync(intent)
                if not identity_ok:
                    return self._reject(intent, f"identity_assertion_failed:{tool_name}", coherence)

        if coherence < COHERENCE_DEGRADE_THRESHOLD:
            return self._degrade(
                intent,
                f"coherence_caution_{coherence:.2f}",
                coherence,
                constraints={"max_tokens": 512},
            )

        active_count = len(self._active_intents)
        if active_count >= MAX_CONCURRENT_INTENTS and intent.priority < 0.7:
            return self._defer(intent, f"capacity_full_{active_count}/{MAX_CONCURRENT_INTENTS}")

        return self._approve(intent, "sync_approved", coherence)

    def _approve(self, intent: Intent, reason: str,
                 coherence: float = 1.0) -> DecisionRecord:
        return self._commit(intent, DecisionOutcome.APPROVED, reason, coherence)

    def _reject(self, intent: Intent, reason: str,
                coherence: float = 0.0) -> DecisionRecord:
        return self._commit(intent, DecisionOutcome.REJECTED, reason, coherence)

    def _defer(self, intent: Intent, reason: str) -> DecisionRecord:
        return self._commit(intent, DecisionOutcome.DEFERRED, reason, 1.0)

    def _degrade(self, intent: Intent, reason: str,
                 coherence: float, constraints: dict = None) -> DecisionRecord:
        return self._commit(
            intent, DecisionOutcome.DEGRADED, reason, coherence, constraints=constraints or {}
        )

    def _commit(
        self,
        intent: Intent,
        outcome: DecisionOutcome,
        reason: str,
        coherence: float,
        constraints: dict | None = None,
    ) -> DecisionRecord:
        """The single point where an admission verdict becomes real.

        Every rule in ``_evaluate`` converges here, which is what lets the
        ontogeny organ see the decision, record it, and — once it has earned
        the right at this control point — differ from it. Sealed reasons
        (identity, coherence lockdown, governance) are recorded but never
        contested: see ``core/ontogeny/wiring.SEALED_REASONS``.

        When the organ is absent, broken, or unpromoted, this is exactly the
        behaviour the executive has always had.
        """
        # Gathered once and shared. These reach into other services, so
        # computing them separately for the organ and for the ledger doubled
        # the per-decision cost of a real-time path.
        context = self._decision_context()

        final = outcome
        verdict: dict[str, Any] | None = None
        try:
            from core.ontogeny.wiring import observe_admission

            chosen, verdict = observe_admission(
                incumbent_choice=outcome.value,
                reason=reason,
                intent_id=intent.intent_id,
                goal=intent.goal,
                source=intent.source.value,
                action_type=intent.action_type.value,
                features=self._ontogeny_features(intent, coherence, context),
                priority=intent.priority,
                blocking=intent.blocking,
            )
            if chosen != outcome.value:
                final = DecisionOutcome(chosen)
        except (ImportError, ValueError, RuntimeError, AttributeError, TypeError, KeyError) as exc:
            record_degradation("executive_core", exc, severity="debug",
                               action="ontogeny not consulted; executive rules stand")

        record = DecisionRecord(
            intent_id=intent.intent_id,
            outcome=final,
            reason=reason if final is outcome else f"{reason}|ontogeny:{final.value}",
            coherence_at_decision=coherence,
            identity_check=final is not DecisionOutcome.REJECTED,
            constraints=dict(constraints or {}),
        )
        if final in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED):
            self._active_intents[intent.intent_id] = intent
            self._approval_count += 1
        elif final is DecisionOutcome.REJECTED:
            self._rejection_count += 1
        self._decision_history.append(record)
        if verdict and verdict.get("episode_id"):
            self._ontogeny_episodes[intent.intent_id] = str(verdict["episode_id"])
        self._append_decision_event(intent, record, ontogeny=verdict, context=context)

        if final in (DecisionOutcome.REJECTED, DecisionOutcome.DEFERRED):
            self._record_failure_obligation(reason, intent)
        if final is DecisionOutcome.REJECTED:
            logger.warning("🚫 Executive REJECTED: %s (reason: %s, coherence: %.2f)",
                           intent.goal[:50], record.reason, coherence)
        elif final is DecisionOutcome.DEGRADED:
            logger.info("Executive constrained %s (constraints: %s)",
                        intent.goal[:50], record.constraints)
        return record

    def _decision_context(self) -> dict[str, Any]:
        """Temporal, epistemic and failure state, read once per decision."""
        return {
            "temporal": self._get_temporal_identity_context(),
            "epistemic": self._get_epistemic_state(),
            "failure": self._get_failure_state(),
        }

    def _ontogeny_features(
        self, intent: Intent, coherence: float, context: dict[str, Any]
    ) -> dict[str, float]:
        """The situation as the organ sees it — all of it already computed here."""
        from core.ontogeny.wiring import admission_features

        temporal = context["temporal"]
        epistemic = context["epistemic"]
        failure = context["failure"]
        return admission_features(
            priority=intent.priority,
            confidence=intent.confidence,
            coherence=coherence,
            failure_pressure=failure.get("pressure", 0.0),
            active_goals=temporal.get("active_goal_count", 0),
            beliefs_contested=epistemic.get("contested", 0),
            pending_initiatives=temporal.get("pending_count", 0),
            blocking=intent.blocking,
            requires_tool=intent.requires_tool,
            requires_memory_commit=intent.requires_memory_commit,
            identity_check=True,
            self_model_available=self._identity_integrity_available(),
            source=intent.source.value,
            action_type=intent.action_type.value,
        )

    # ── Intent Lifecycle ─────────────────────────────────────────────────

    def complete_intent(self, intent_id: str, success: bool = True) -> None:
        """Mark an intent as completed. Frees capacity."""
        intent = self._active_intents.pop(intent_id, None)
        episode_id = self._ontogeny_episodes.pop(intent_id, None)
        if episode_id and intent is not None:
            try:
                from core.ontogeny.wiring import note_admission_completion

                note_admission_completion(episode_id, success=success, goal=intent.goal)
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('executive_core', exc, severity="debug",
                                   action="ontogeny completion not recorded")
        if intent is not None:
            try:
                self._get_ledger().append(
                    {
                        "event": "intent_complete",
                        "intent_id": intent.intent_id,
                        "goal": intent.goal,
                        "source": intent.source.value,
                        "action_type": intent.action_type.value,
                        "success": bool(success),
                    }
                )
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('executive_core', exc)
                logger.debug("Executive ledger completion append failed: %s", exc)

    def get_active_intents(self) -> list[Intent]:
        return list(self._active_intents.values())

    # ── Integration with BindingEngine + CanonicalSelf ───────────────────

    async def _get_coherence(self) -> float:
        """Get current coherence from BindingEngine."""
        try:
            binding = ServiceContainer.get("binding_engine", default=None)
            if binding and hasattr(binding, "get_coherence"):
                return binding.get_coherence()
        except (ImportError, AttributeError, RuntimeError) as _exc:
            record_degradation('executive_core', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        return 0.75  # conservative default — allows normal ops but not risky ones

    def _get_coherence_sync(self) -> float:
        """Synchronous coherence check."""
        try:
            binding = ServiceContainer.get("binding_engine", default=None)
            if binding and hasattr(binding, "get_coherence"):
                return binding.get_coherence()
        except (ImportError, AttributeError, RuntimeError) as _exc:
            record_degradation('executive_core', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        return 0.75

    async def _check_identity(self, intent: Intent) -> bool:
        """Check if intent is consistent with identity."""
        try:
            self_engine = resolve_canonical_self_engine(default=None, autocreate=False)
            if self_engine and hasattr(self_engine, "assert_identity"):
                return self_engine.assert_identity(intent.goal)
        except (RuntimeError, AttributeError, TypeError) as _exc:
            record_degradation('executive_core', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        return self._identity_integrity_available()

    def _check_identity_sync(self, intent: Intent) -> bool:
        try:
            self_engine = resolve_canonical_self_engine(default=None, autocreate=False)
            if self_engine and hasattr(self_engine, "assert_identity"):
                return bool(self_engine.assert_identity(intent.goal))
        except (RuntimeError, AttributeError, TypeError) as _exc:
            record_degradation('executive_core', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        return self._identity_integrity_available()

    def _strict_runtime_active(self) -> bool:
        try:
            return (
                ServiceContainer.has("aura_kernel")
                or ServiceContainer.has("kernel_interface")
                or bool(getattr(ServiceContainer, "_registration_locked", False))
            )
        except (RuntimeError, AttributeError, TypeError):
            return False

    def _identity_integrity_available(self) -> bool:
        try:
            if resolve_canonical_self_engine(default=None, autocreate=False) is not None:
                return True
            if resolve_canonical_self(default=None, autocreate=False) is not None:
                return True
            if ServiceContainer.get("self_model", default=None) is not None:
                return True
        except (ImportError, AttributeError, RuntimeError) as _exc:
            record_degradation('executive_core', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        return not self._strict_runtime_active()

    def _get_temporal_identity_context(self) -> dict[str, Any]:
        current_objective = ""
        current_origin = ""
        objective_binding: dict[str, Any] = {}
        pending_count = 0
        active_goal_count = 0
        contradiction_count = 0
        commitments: list[str] = []
        anchor = "none"
        pending_anchor = ""
        active_goal_anchor = ""
        try:
            repo = resolve_state_repository(default=None)
            state = getattr(repo, "_current", None) if repo is not None else None
            cognition = getattr(state, "cognition", None) if state is not None else None
            current_objective = str(getattr(cognition, "current_objective", "") or "")
            current_origin = str(getattr(cognition, "current_origin", "") or "")
            modifiers = dict(getattr(cognition, "modifiers", {}) or {})
            objective_binding = dict(modifiers.get("current_objective_binding", {}) or {})
            pending_items = list(getattr(cognition, "pending_initiatives", []) or [])
            active_goal_items = list(getattr(cognition, "active_goals", []) or [])
            pending_count = len(pending_items)
            active_goal_count = len(active_goal_items)
            contradiction_count = int(getattr(cognition, "contradiction_count", 0) or 0)
            pending_anchor = first_actionable_goal_text(pending_items)
            active_goal_anchor = first_actionable_goal_text(active_goal_items)
        except (RuntimeError, AttributeError, TypeError) as _exc:
            record_degradation('executive_core', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        try:
            from core.continuity import get_continuity

            continuity = get_continuity()
            if getattr(continuity, "_record", None) is None:
                continuity.load()
            obligations = continuity.get_obligations()
            if not current_objective:
                current_objective = str(obligations.get("current_objective", "") or "")
            commitments = list(obligations.get("active_commitments", []) or [])
            if pending_count == 0:
                pending_items = list(obligations.get("pending_initiatives", []) or [])
                pending_count = len(pending_items)
                pending_anchor = first_actionable_goal_text(pending_items)
            if active_goal_count == 0:
                continuity_goals = list(obligations.get("active_goals", []) or [])
                active_goal_count = len(continuity_goals)
                active_goal_anchor = first_actionable_goal_text(continuity_goals)
            contradiction_count = max(
                contradiction_count,
                int(obligations.get("contradiction_count", 0) or 0),
            )
        except (ImportError, AttributeError, RuntimeError) as _exc:
            record_degradation('executive_core', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        try:
            goal_engine = ServiceContainer.get("goal_engine", default=None)
            if goal_engine and hasattr(goal_engine, "get_active_goals"):
                active_goal_items = list(
                    goal_engine.get_active_goals(limit=6, include_external=True, actionable_only=True) or []
                )
                if active_goal_items:
                    active_goal_count = max(active_goal_count, len(active_goal_items))
                    if not current_objective:
                        current_objective = first_actionable_goal_text(active_goal_items)
                    if not active_goal_anchor:
                        active_goal_anchor = first_actionable_goal_text(active_goal_items)
        except (ImportError, AttributeError, RuntimeError) as _exc:
            record_degradation('executive_core', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        current_objective = _normalize_goal_text(current_objective)
        if (
            _is_speculative_autonomy_label(current_objective)
            or is_intrinsic_goal_text(current_objective)
            or _is_transient_conversation_memory_objective(current_objective)
        ):
            current_objective = ""
        commitments = [
            text
            for text in (_normalize_goal_text(entry) for entry in commitments)
            if text and not _is_speculative_autonomy_label(text)
        ]
        actionable_commitments = [text for text in commitments if is_actionable_goal_text(text)]
        try:
            ttl_s = float(os.getenv("AURA_TEMPORAL_USER_OBJECTIVE_TTL_S", "300"))
        except (TypeError, ValueError):
            ttl_s = 300.0
        objective_source = str(objective_binding.get("source") or current_origin or "").strip().lower()
        objective_promoted_at = 0.0
        try:
            objective_promoted_at = float(objective_binding.get("promoted_at", 0.0) or 0.0)
        except (TypeError, ValueError):
            objective_promoted_at = 0.0
        objective_age_s = time.time() - objective_promoted_at if objective_promoted_at > 0.0 else None
        objective_matches_active_work = bool(
            current_objective
            and current_objective in {
                text
                for text in (
                    pending_anchor,
                    active_goal_anchor,
                    *(actionable_commitments[:3]),
                )
                if text
            }
        )
        source_is_user_like = (
            _coerce_intent_source(objective_source).value == IntentSource.USER.value
            if objective_source
            else False
        )
        if (
            current_objective
            and source_is_user_like
            and objective_age_s is not None
            and objective_age_s > ttl_s
            and not objective_matches_active_work
        ):
            current_objective = ""
        obligation_pressure = min(
            1.0,
            (float(pending_count) * 0.25) + (float(active_goal_count) * 0.2) + (float(len(commitments)) * 0.2),
        )
        anchor = (
            current_objective
            or pending_anchor
            or active_goal_anchor
            or (actionable_commitments[0] if actionable_commitments else "")
            or ("unresolved_runtime_obligations" if obligation_pressure > 0.0 else "none")
        )
        return {
            "current_objective": current_objective,
            "pending_count": pending_count,
            "active_goal_count": active_goal_count,
            "commitments": commitments[:5],
            "contradiction_count": contradiction_count,
            "obligation_pressure": round(obligation_pressure, 4),
            "anchor": str(anchor or "none")[:80],
        }

    def _get_epistemic_state(self) -> dict[str, Any]:
        try:
            from core.constitution import get_constitutional_core

            summary = get_constitutional_core().belief_authority.summary()
            return {
                # Fresh contests gate autonomy; aged-out contests (no
                # re-assertion within the freshness window) stop wedging
                # every autonomous write forever. Falls back to the raw
                # count when the authority predates freshness tracking.
                "contested": int(
                    summary.get("fresh_contested", summary.get("contested", 0)) or 0
                ),
                "contested_keys": list(summary.get("fresh_contested_keys") or []),
                "trusted": int(summary.get("trusted", 0) or 0),
                "coherence_score": float(summary.get("coherence_score", 1.0) or 1.0),
            }
        except (ImportError, AttributeError, RuntimeError):
            return {"contested": 0, "contested_keys": [], "trusted": 0, "coherence_score": 1.0}

    @staticmethod
    def _intent_touches_contested_topic(
        intent: Intent, epistemic: dict[str, Any]
    ) -> bool:
        """Whether this write is actually ABOUT something contested.

        A contest is evidence that one subject is unsettled. It is not evidence
        that everything is. The live 2026-07-25 hour deferred 71 autonomous
        knowledge writes on ``epistemic_reconciliation_required:2`` — two
        contested claims blocking every unrelated fact and concept she learned.
        The freshness window fixed contests that never aged out; it did not
        make the gate about relevance.

        Fails CLOSED: with contests present but no keys to compare against,
        the old global behaviour stands.
        """
        keys = [str(k).strip().lower() for k in epistemic.get("contested_keys") or []]
        keys = [k for k in keys if k]
        if not keys:
            return True

        haystack = " ".join(
            str(part).lower()
            for part in (
                intent.goal,
                intent.payload.get("topic", ""),
                intent.payload.get("key", ""),
                intent.payload.get("namespace", ""),
                intent.payload.get("content", ""),
                intent.payload.get("summary", ""),
            )
            if part
        )
        if not haystack.strip():
            return True  # nothing to judge relevance by — stay conservative

        for key in keys:
            namespace, _, subject = key.partition(":")
            for token in (key, subject, namespace):
                token = token.strip()
                if len(token) >= 4 and token in haystack:
                    return True
        return False

    def _surface_research_trigger(self, intent: Intent, epistemic: dict[str, Any]) -> None:
        """Best-effort: surface a deferred autonomous belief-update as a
        research trigger so the autonomy pipeline picks the contested topic
        up next cycle. Never throws — Rule 7 deferral must remain idempotent
        regardless of trigger emission success.
        """
        try:
            from core.autonomy.research_triggers import emit_research_trigger
            emit_research_trigger(
                topic=intent.goal or "contested_belief",
                source_intent_id=intent.intent_id,
                contested_count=int(epistemic.get("contested", 0)),
                payload_hint=intent.payload,
            )
        except (ImportError, AttributeError, RuntimeError):
            pass  # no-op: intentional

    def _get_failure_state(self) -> dict[str, Any]:
        try:
            from core.health.degraded_events import get_unified_failure_state

            return get_unified_failure_state(limit=25)
        except (ImportError, AttributeError, RuntimeError):
            return {"pressure": 0.0, "count": 0, "critical": 0, "errors": 0, "warnings": 0, "top_subsystems": []}

    def _get_internal_state_constraints(self) -> dict[str, float | bool]:
        energy = 1.0
        thermal_pressure = 0.0
        load_pressure = 0.0
        distress = 0.0
        identity_mismatch = False
        try:
            repo = resolve_state_repository(default=None)
            state = getattr(repo, "_current", None) if repo is not None else None
            cognition = getattr(state, "cognition", None) if state is not None else None
            soma = getattr(state, "soma", None) if state is not None else None
            body = getattr(state, "body", None) if state is not None else None
            affect = getattr(state, "affect", None) if state is not None else None
            motivation = getattr(state, "motivation", None) if state is not None else None

            raw_energy = getattr(soma, "energy", getattr(body, "energy", 1.0))
            if raw_energy is not None:
                energy = float(raw_energy)
                if energy > 1.0:
                    energy = max(0.0, min(1.0, energy / 100.0))
                else:
                    energy = max(0.0, min(1.0, energy))

            thermal_pressure = float(
                getattr(body, "thermal_pressure", getattr(soma, "thermal_pressure", 0.0)) or 0.0
            )
            load_pressure = float(getattr(cognition, "load_pressure", 0.0) or 0.0)

            valence = float(getattr(affect, "valence", 0.0) or 0.0)
            arousal = float(getattr(affect, "arousal", 0.0) or 0.0)
            drive_pressure = float(
                getattr(motivation, "pressure", getattr(motivation, "drive_pressure", 0.0)) or 0.0
            )
            distress = max(
                0.0,
                min(
                    1.0,
                    max(0.0, -valence) * 0.5 + max(0.0, arousal) * 0.25 + max(0.0, drive_pressure) * 0.25,
                ),
            )

            modifiers = dict(getattr(cognition, "modifiers", {}) or {})
            continuity = dict(modifiers.get("continuity_obligations", {}) or {})
            identity_mismatch = bool(continuity.get("identity_mismatch", False))
        except (OSError, ConnectionError, TimeoutError) as _exc:
            record_degradation('executive_core', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        return {
            "energy": energy,
            "thermal_pressure": max(0.0, min(1.0, thermal_pressure)),
            "load_pressure": max(0.0, min(1.0, load_pressure)),
            "distress": distress,
            "identity_mismatch": identity_mismatch,
        }

    def _record_failure_obligation(self, reason: str, intent: Intent) -> None:
        try:
            repo = resolve_state_repository(default=None)
            state = getattr(repo, "_current", None) if repo is not None else None
            cognition = getattr(state, "cognition", None) if state is not None else None
            if cognition is not None:
                modifiers = dict(getattr(cognition, "modifiers", {}) or {})
                failure_state = dict(modifiers.get("failure_obligations", {}) or {})
                failure_state["last_reason"] = str(reason or "")[:200]
                failure_state["last_goal"] = str(getattr(intent, "goal", "") or "")[:200]
                failure_state["last_source"] = getattr(intent.source, "value", str(intent.source))
                failure_state["last_at"] = time.time()
                failure_state["count"] = int(failure_state.get("count", 0) or 0) + 1
                modifiers["failure_obligations"] = failure_state
                cognition.modifiers = modifiers
        except (OSError, ConnectionError, TimeoutError) as _exc:
            record_degradation('executive_core', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        try:
            from core.continuity import get_continuity

            continuity = get_continuity()
            if getattr(continuity, "_record", None) is None:
                continuity.load()
            continuity.note_failure_obligation(reason, getattr(intent, "goal", ""))
        except (ImportError, AttributeError, RuntimeError) as _exc:
            record_degradation('executive_core', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

    # ── Observability ────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        return {
            "approved": self._approval_count,
            "rejected": self._rejection_count,
            "active_intents": len(self._active_intents),
            "recent_decisions": [d.to_dict() for d in list(self._decision_history)[-10:]],
        }

    def get_decision_history(self, n: int = 20) -> list[DecisionRecord]:
        return list(self._decision_history)[-n:]

    def get_rejection_rate(self) -> float:
        total = self._approval_count + self._rejection_count
        if total == 0:
            return 0.0
        return self._rejection_count / total

    def _get_ledger(self) -> ExecutiveLedger:
        if self._ledger is None:
            try:
                from core.config import config

                path = config.paths.data_dir / "executive_ledger.jsonl"
            except (ImportError, AttributeError, RuntimeError):
                path = "executive_ledger.jsonl"
            self._ledger = ExecutiveLedger(path)
        return self._ledger

    def _append_decision_event(
        self, intent: Intent, record: DecisionRecord, *,
        ontogeny: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        try:
            gathered = context or self._decision_context()
            temporal = gathered["temporal"]
            epistemic = gathered["epistemic"]
            failure = gathered["failure"]
            self._get_ledger().append(
                {
                    "event": "decision",
                    "intent_id": intent.intent_id,
                    "source": intent.source.value,
                    "goal": intent.goal,
                    "action_type": intent.action_type.value,
                    "priority": intent.priority,
                    "confidence": intent.confidence,
                    "blocking": intent.blocking,
                    "requires_tool": intent.requires_tool,
                    "requires_memory_commit": intent.requires_memory_commit,
                    "payload_keys": sorted(list((intent.payload or {}).keys())),
                    "outcome": record.outcome.value,
                    "reason": record.reason,
                    "coherence": record.coherence_at_decision,
                    "identity_check": record.identity_check,
                    "self_model_available": self._identity_integrity_available(),
                    "temporal_anchor": temporal.get("anchor", ""),
                    "pending_initiatives": temporal.get("pending_count", 0),
                    "active_goals": temporal.get("active_goal_count", 0),
                    "beliefs_contested": epistemic.get("contested", 0),
                    "failure_pressure": failure.get("pressure", 0.0),
                    "constraints": dict(record.constraints or {}),
                    **(
                        {
                            "ontogeny_stage": ontogeny.get("stage"),
                            "ontogeny_decider": ontogeny.get("decider"),
                            "ontogeny_novelty": ontogeny.get("novelty"),
                            "ontogeny_episode": ontogeny.get("episode_id"),
                        }
                        if ontogeny else {}
                    ),
                }
            )
        except (OSError, ConnectionError, TimeoutError) as exc:
            record_degradation('executive_core', exc)
            logger.debug("Executive ledger append failed: %s", exc)


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: ExecutiveCore | None = None
_lock = None


def get_executive_core() -> ExecutiveCore:
    """Get or create the global ExecutiveCore."""
    global _instance, _lock
    if _instance is None:
        _instance = ExecutiveCore()
        try:
            ServiceContainer.register_instance("executive_core", _instance, required=False)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('executive_core', exc)
            logger.error("ExecutiveCore registration failed: %s", exc, exc_info=True)
    return _instance
