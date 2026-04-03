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
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.container import ServiceContainer
from core.executive.executive_ledger import ExecutiveLedger
from core.state.aura_state import _is_speculative_autonomy_label, _normalize_goal_text

logger = logging.getLogger("Aura.Executive")


def _coerce_intent_source(source: str) -> IntentSource:
    normalized = str(source or "").strip().lower()
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
    payload: Dict[str, Any] = field(default_factory=dict)
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
    constraints: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
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

# Tool names that require identity assertion before execution
IDENTITY_SENSITIVE_TOOLS = {
    "self_evolution", "self_repair", "self_improvement",
    "propagation", "auto_refactor", "train_self",
}


# ── Executive Core ───────────────────────────────────────────────────────────

class ExecutiveCore:
    """The single sovereign control plane.

    All significant operations request approval. The executive decides
    based on coherence, identity, resource state, and current intent queue.
    """

    def __init__(self) -> None:
        self._active_intents: Dict[str, Intent] = {}
        self._decision_history: deque[DecisionRecord] = deque(maxlen=500)
        self._approval_count: int = 0
        self._rejection_count: int = 0
        self._lock = asyncio.Lock()
        self._initialized = False
        self._ledger: Optional[ExecutiveLedger] = None
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

    async def approve_tool(self, tool_name: str, args: Dict[str, Any],
                           source: str = "unknown") -> Tuple[bool, str, Dict]:
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
        args: Dict[str, Any],
        source: str = "unknown",
    ) -> Tuple[Intent, DecisionRecord]:
        """Build and evaluate a tool-call intent while preserving the intent id."""
        intent = Intent(
            source=_coerce_intent_source(source),
            goal=f"execute_tool:{tool_name}",
            action_type=ActionType.TOOL_CALL,
            payload={"tool_name": tool_name, "args": args},
            requires_tool=True,
        )

        # User-initiated tools always approved
        if source in ("user", "voice", "admin", "api"):
            intent.source = IntentSource.USER
            intent.priority = 0.9

        record = await self.request_approval(intent)
        return intent, record

    async def approve_emission(self, content: str, source: str = "unknown",
                               urgency: float = 0.5) -> Tuple[bool, str]:
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
                                    source: str = "unknown") -> Tuple[bool, str]:
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

    async def approve_state_mutation(self, origin: str, cause: str) -> Tuple[bool, str]:
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
                                       source: str = "unknown") -> Tuple[bool, str]:
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
            and intent.action_type in {
                ActionType.SPAWN_TASK,
                ActionType.EMIT_MESSAGE,
                ActionType.TOOL_CALL,
                ActionType.REFLECT,
            }
            and intent.priority < 0.85
        ):
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

        # Rule 7: Closed-loop epistemology. Belief churn is deferred while contested
        # beliefs are unresolved instead of silently accumulating.
        epistemic = self._get_epistemic_state()
        if (
            strict_runtime
            and epistemic["contested"] > 0
            and intent.source != IntentSource.USER
            and intent.action_type in {ActionType.UPDATE_BELIEF, ActionType.WRITE_MEMORY}
            and intent.priority < 0.9
        ):
            return self._defer(intent, f"epistemic_reconciliation_required:{epistemic['contested']}")

        # Rule 8: Check coherence
        coherence = await self._get_coherence()

        # Lockdown mode: only essential operations
        if coherence < COHERENCE_LOCKDOWN_THRESHOLD:
            if intent.action_type in LOCKDOWN_BLOCKED:
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
            if intent.priority < 0.7:
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
            and intent.action_type in {
                ActionType.SPAWN_TASK,
                ActionType.EMIT_MESSAGE,
                ActionType.TOOL_CALL,
                ActionType.REFLECT,
            }
            and intent.priority < 0.85
        ):
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
        if (
            strict_runtime
            and epistemic["contested"] > 0
            and intent.source != IntentSource.USER
            and intent.action_type in {ActionType.UPDATE_BELIEF, ActionType.WRITE_MEMORY}
            and intent.priority < 0.9
        ):
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
        record = DecisionRecord(
            intent_id=intent.intent_id,
            outcome=DecisionOutcome.APPROVED,
            reason=reason,
            coherence_at_decision=coherence,
        )
        self._active_intents[intent.intent_id] = intent
        self._decision_history.append(record)
        self._approval_count += 1
        self._append_decision_event(intent, record)
        return record

    def _reject(self, intent: Intent, reason: str,
                coherence: float = 0.0) -> DecisionRecord:
        record = DecisionRecord(
            intent_id=intent.intent_id,
            outcome=DecisionOutcome.REJECTED,
            reason=reason,
            coherence_at_decision=coherence,
            identity_check=False,
        )
        self._decision_history.append(record)
        self._rejection_count += 1
        self._append_decision_event(intent, record)
        self._record_failure_obligation(reason, intent)
        logger.warning("🚫 Executive REJECTED: %s (reason: %s, coherence: %.2f)",
                       intent.goal[:50], reason, coherence)
        return record

    def _defer(self, intent: Intent, reason: str) -> DecisionRecord:
        record = DecisionRecord(
            intent_id=intent.intent_id,
            outcome=DecisionOutcome.DEFERRED,
            reason=reason,
        )
        self._decision_history.append(record)
        self._append_decision_event(intent, record)
        self._record_failure_obligation(reason, intent)
        return record

    def _degrade(self, intent: Intent, reason: str,
                 coherence: float, constraints: Dict = None) -> DecisionRecord:
        record = DecisionRecord(
            intent_id=intent.intent_id,
            outcome=DecisionOutcome.DEGRADED,
            reason=reason,
            coherence_at_decision=coherence,
            constraints=constraints or {},
        )
        self._active_intents[intent.intent_id] = intent
        self._decision_history.append(record)
        self._approval_count += 1
        self._append_decision_event(intent, record)
        logger.info("⚠️ Executive DEGRADED: %s (constraints: %s)",
                    intent.goal[:50], constraints)
        return record

    # ── Intent Lifecycle ─────────────────────────────────────────────────

    def complete_intent(self, intent_id: str, success: bool = True) -> None:
        """Mark an intent as completed. Frees capacity."""
        intent = self._active_intents.pop(intent_id, None)
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
            except Exception as exc:
                logger.debug("Executive ledger completion append failed: %s", exc)

    def get_active_intents(self) -> List[Intent]:
        return list(self._active_intents.values())

    # ── Integration with BindingEngine + CanonicalSelf ───────────────────

    async def _get_coherence(self) -> float:
        """Get current coherence from BindingEngine."""
        try:
            binding = ServiceContainer.get("binding_engine", default=None)
            if binding and hasattr(binding, "get_coherence"):
                return binding.get_coherence()
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
        return 0.75  # conservative default — allows normal ops but not risky ones

    def _get_coherence_sync(self) -> float:
        """Synchronous coherence check."""
        try:
            binding = ServiceContainer.get("binding_engine", default=None)
            if binding and hasattr(binding, "get_coherence"):
                return binding.get_coherence()
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
        return 0.75

    async def _check_identity(self, intent: Intent) -> bool:
        """Check if intent is consistent with identity."""
        try:
            self_engine = ServiceContainer.get("canonical_self_engine", default=None)
            if self_engine and hasattr(self_engine, "assert_identity"):
                return self_engine.assert_identity(intent.goal)
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
        return self._identity_integrity_available()

    def _check_identity_sync(self, intent: Intent) -> bool:
        try:
            self_engine = ServiceContainer.get("canonical_self_engine", default=None)
            if self_engine and hasattr(self_engine, "assert_identity"):
                return bool(self_engine.assert_identity(intent.goal))
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
        return self._identity_integrity_available()

    def _strict_runtime_active(self) -> bool:
        try:
            return (
                ServiceContainer.has("aura_kernel")
                or ServiceContainer.has("kernel_interface")
                or bool(getattr(ServiceContainer, "_registration_locked", False))
            )
        except Exception:
            return False

    def _identity_integrity_available(self) -> bool:
        try:
            if ServiceContainer.get("canonical_self_engine", default=None) is not None:
                return True
            if ServiceContainer.get("canonical_self", default=None) is not None:
                return True
            if ServiceContainer.get("self_model", default=None) is not None:
                return True
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
        return not self._strict_runtime_active()

    def _get_temporal_identity_context(self) -> Dict[str, Any]:
        current_objective = ""
        pending_count = 0
        active_goal_count = 0
        contradiction_count = 0
        commitments: List[str] = []
        anchor = "none"
        try:
            repo = ServiceContainer.get("state_repository", default=None)
            state = getattr(repo, "_current", None) if repo is not None else None
            cognition = getattr(state, "cognition", None) if state is not None else None
            current_objective = str(getattr(cognition, "current_objective", "") or "")
            pending_count = len(list(getattr(cognition, "pending_initiatives", []) or []))
            active_goal_count = len(list(getattr(cognition, "active_goals", []) or []))
            contradiction_count = int(getattr(cognition, "contradiction_count", 0) or 0)
        except Exception as _exc:
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
                pending_count = len(list(obligations.get("pending_initiatives", []) or []))
            if active_goal_count == 0:
                active_goal_count = len(list(obligations.get("active_goals", []) or []))
            contradiction_count = max(
                contradiction_count,
                int(obligations.get("contradiction_count", 0) or 0),
            )
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        current_objective = _normalize_goal_text(current_objective)
        if _is_speculative_autonomy_label(current_objective):
            current_objective = ""
        commitments = [
            text
            for text in (_normalize_goal_text(entry) for entry in commitments)
            if text and not _is_speculative_autonomy_label(text)
        ]

        anchor = current_objective or (commitments[0] if commitments else "none")
        obligation_pressure = min(
            1.0,
            (float(pending_count) * 0.25) + (float(active_goal_count) * 0.2) + (float(len(commitments)) * 0.2),
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

    def _get_epistemic_state(self) -> Dict[str, Any]:
        try:
            from core.constitution import get_constitutional_core

            summary = get_constitutional_core().belief_authority.summary()
            return {
                "contested": int(summary.get("contested", 0) or 0),
                "trusted": int(summary.get("trusted", 0) or 0),
                "coherence_score": float(summary.get("coherence_score", 1.0) or 1.0),
            }
        except Exception:
            return {"contested": 0, "trusted": 0, "coherence_score": 1.0}

    def _get_failure_state(self) -> Dict[str, Any]:
        try:
            from core.health.degraded_events import get_unified_failure_state

            return get_unified_failure_state(limit=25)
        except Exception:
            return {"pressure": 0.0, "count": 0, "critical": 0, "errors": 0, "warnings": 0, "top_subsystems": []}

    def _get_internal_state_constraints(self) -> Dict[str, float | bool]:
        energy = 1.0
        thermal_pressure = 0.0
        load_pressure = 0.0
        distress = 0.0
        identity_mismatch = False
        try:
            repo = ServiceContainer.get("state_repository", default=None)
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
        except Exception as _exc:
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
            repo = ServiceContainer.get("state_repository", default=None)
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
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        try:
            from core.continuity import get_continuity

            continuity = get_continuity()
            if getattr(continuity, "_record", None) is None:
                continuity.load()
            continuity.note_failure_obligation(reason, getattr(intent, "goal", ""))
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

    # ── Observability ────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "approved": self._approval_count,
            "rejected": self._rejection_count,
            "active_intents": len(self._active_intents),
            "recent_decisions": [d.to_dict() for d in list(self._decision_history)[-10:]],
        }

    def get_decision_history(self, n: int = 20) -> List[DecisionRecord]:
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
            except Exception:
                path = "executive_ledger.jsonl"
            self._ledger = ExecutiveLedger(path)
        return self._ledger

    def _append_decision_event(self, intent: Intent, record: DecisionRecord) -> None:
        try:
            temporal = self._get_temporal_identity_context()
            epistemic = self._get_epistemic_state()
            failure = self._get_failure_state()
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
                }
            )
        except Exception as exc:
            logger.debug("Executive ledger append failed: %s", exc)


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: Optional[ExecutiveCore] = None
_lock = asyncio.Lock()


def get_executive_core() -> ExecutiveCore:
    """Get or create the global ExecutiveCore."""
    global _instance
    if _instance is None:
        _instance = ExecutiveCore()
        try:
            ServiceContainer.register_instance("executive_core", _instance, required=False)
        except Exception as exc:
            logger.debug("ExecutiveCore registration skipped: %s", exc)
    return _instance
