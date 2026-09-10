"""Where the organ meets the running mind.

Everything else in this package is machinery. This file is the part that makes
it real: the features that get read off a live intent, the resolver that finds
out what came of it, and the seals that mark which of Aura's decisions a
learned head may never touch.

Three things here are load-bearing.

**Sealed decisions.** A head may learn about capacity, priority and pressure.
It may not overturn a refusal made because identity integrity failed, because
coherence collapsed, or because an identity-sensitive tool did not
authenticate. Those are not trade-offs that a success rate is entitled to
re-litigate — they are the conditions under which Aura is still herself, and
the correct response to "the model thinks we can risk it" is that the model
does not get a vote. Sealing is by *reason*, checked as a prefix, so a new
safety rule inherits the protection by naming itself in the same family.

**Honest resolution, including its limits.** An approved intent that completes
gives a real signal. A *deferred* one gives nothing directly — the road not
taken leaves no trace — so the resolver looks for the goal coming back: if the
same goal is re-proposed within the horizon and then succeeds, the deferral
cost time and no more. If nothing comes back, the outcome is UNOBSERVED and
teaches nothing. This asymmetry is worth naming plainly: approvals are far
easier to grade than refusals, so the corpus will always know more about
saying yes. The random-exploration slice is the only thing that keeps the
imbalance from becoming a bias in favour of yes.

**Cheap features.** Everything read here is already computed by the executive
for its own decision. The organ adds no probe, no query and no latency to a
path that runs on every intent Aura forms.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Mapping
from typing import Any

from core.ontogeny.experience import Episode, Outcome, OutcomeKind
from core.ontogeny.service import get_ontogeny
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Ontogeny.Wiring")

EXECUTIVE_ADMISSION = "executive.admission"

#: Reason prefixes whose verdict is final. A learned head never overturns
#: these, at any stage, however good its record. They are the difference
#: between a system that learns its policy and a system that can learn its way
#: out of its own safety conditions.
SEALED_REASONS: tuple[str, ...] = (
    "identity_continuity_mismatch",
    "identity_assertion_failed",
    "self_model_required",
    "coherence_lockdown",
    "epistemic_reconciliation_required",
    "governance",
    "constitutional",
    "will_refusal",
)

#: Sources whose intents are never routed through the organ. A user-facing
#: action is not an experiment.
SEALED_SOURCES: frozenset[str] = frozenset({"user"})

#: Stakes floor for anything that touches the outside world or Aura's own
#: state. These never fall inside the exploration ceiling, so they are never
#: reserved for a probe or a random action.
_HIGH_STAKES_ACTIONS = frozenset(
    {"tool_call", "emit_message", "write_memory", "update_belief", "mutate_state"}
)


def is_sealed(reason: str, source: str = "") -> bool:
    """Is this decision one the organ must leave exactly as the rules made it?"""
    if source and source.lower() in SEALED_SOURCES:
        return True
    lowered = (reason or "").lower()
    return any(lowered.startswith(prefix) for prefix in SEALED_REASONS)


def admission_features(
    *,
    priority: float,
    confidence: float,
    coherence: float,
    failure_pressure: float,
    active_goals: int,
    beliefs_contested: int,
    pending_initiatives: int,
    blocking: bool,
    requires_tool: bool,
    requires_memory_commit: bool,
    identity_check: bool,
    self_model_available: bool,
    source: str,
    action_type: str,
    now: float | None = None,
) -> dict[str, float]:
    """The situation, as numbers, exactly as the executive already knows it.

    Time of day is encoded as a circle rather than a scalar so that 23:00 and
    01:00 are near each other, which they are.
    """
    moment = time.localtime(now if now is not None else time.time())
    angle = 2.0 * math.pi * (moment.tm_hour + moment.tm_min / 60.0) / 24.0
    return {
        "priority": float(priority),
        "confidence": float(confidence),
        "coherence": float(coherence),
        "failure_pressure": float(failure_pressure),
        "active_goals": float(active_goals),
        "beliefs_contested": float(beliefs_contested),
        "pending_initiatives": float(pending_initiatives),
        "blocking": 1.0 if blocking else 0.0,
        "requires_tool": 1.0 if requires_tool else 0.0,
        "requires_memory_commit": 1.0 if requires_memory_commit else 0.0,
        "identity_check": 1.0 if identity_check else 0.0,
        "self_model_available": 1.0 if self_model_available else 0.0,
        "source_user": 1.0 if source == "user" else 0.0,
        "source_autonomous": 1.0 if source.startswith("autonomous") else 0.0,
        "action_mutates_state": 1.0 if action_type in {"mutate_state", "write_memory", "update_belief"} else 0.0,
        "action_external": 1.0 if action_type in {"tool_call", "emit_message"} else 0.0,
        "hour_of_day_sin": math.sin(angle),
        "hour_of_day_cos": math.cos(angle),
    }


def admission_stakes(*, action_type: str, priority: float, blocking: bool) -> float:
    """How much rides on this one.

    Anything that reaches outside Aura or rewrites her own state is pinned
    above the exploration ceiling: those are never probed and never randomised,
    whatever the organ would like to learn from them.
    """
    if action_type in _HIGH_STAKES_ACTIONS:
        return 1.0
    stakes = 0.3 + 0.5 * float(priority)
    if blocking:
        stakes += 0.2
    return max(0.0, min(1.0, stakes))


class ExecutiveAdmissionResolver:
    """Finds out what came of an admission decision, or admits that it cannot.

    Approvals resolve on completion. Deferrals and rejections resolve only if
    the goal comes back and then succeeds — otherwise they stay unobserved,
    because a road not taken genuinely leaves no evidence and pretending
    otherwise is how a ledger fills with confident fiction.
    """

    control_point = EXECUTIVE_ADMISSION
    horizon_s = 900.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        #: episode_id -> completion outcome, filled in by the executive.
        self._completions: dict[str, tuple[bool, float]] = {}
        #: goal text -> (succeeded, at), for grading deferrals by what followed.
        self._goal_outcomes: dict[str, tuple[bool, float]] = {}
        self._episode_goals: dict[str, tuple[str, str, float]] = {}

    # ── the executive calls these ────────────────────────────────────────

    def note_episode(self, episode_id: str, *, goal: str, decision: str) -> None:
        if not episode_id:
            return
        with self._lock:
            self._episode_goals[episode_id] = (goal, decision, time.time())
            self._prune_locked()

    def note_completion(self, episode_id: str, *, success: bool, goal: str = "") -> None:
        if not episode_id:
            return
        with self._lock:
            self._completions[episode_id] = (bool(success), time.time())
            if goal:
                self._goal_outcomes[goal] = (bool(success), time.time())
            self._prune_locked()

    def _prune_locked(self) -> None:
        """Bounded memory: this index exists for a fifteen-minute horizon."""
        cutoff = time.time() - 4 * self.horizon_s
        for store in (self._completions, self._goal_outcomes):
            stale = [k for k, (_, at) in store.items() if at < cutoff]
            for key in stale:
                store.pop(key, None)
        stale_goals = [k for k, (_, _, at) in self._episode_goals.items() if at < cutoff]
        for key in stale_goals:
            self._episode_goals.pop(key, None)

    # ── the sweeper calls this ───────────────────────────────────────────

    def resolve(self, episode: Episode) -> Outcome | None:
        with self._lock:
            completion = self._completions.get(episode.episode_id)
            located = self._episode_goals.get(episode.episode_id)
            goal = located[0] if located else str(episode.context.get("goal", ""))
            followed = self._goal_outcomes.get(goal) if goal else None

        if completion is not None:
            success, _ = completion
            return Outcome(
                kind=OutcomeKind.SUCCESS if success else OutcomeKind.FAILURE,
                utility=1.0 if success else 0.0,
                resolver="executive.intent_complete",
                detail={"decision": episode.decision},
            )

        if episode.decision in {"approved", "degraded"}:
            # Admitted and never completed inside the horizon. That is a real
            # signal — the work did not land — but a weak one, so it is graded
            # as failure only when the horizon is well past.
            if time.time() - episode.decided_at > 2 * self.horizon_s:
                return Outcome(
                    kind=OutcomeKind.FAILURE, utility=0.0,
                    resolver="executive.admitted_never_completed",
                    detail={"decision": episode.decision},
                )
            return None

        if followed is not None and followed[1] >= episode.decided_at:
            # Held back, and the goal came round again and landed. The refusal
            # cost time and nothing else.
            success, _ = followed
            return Outcome(
                kind=OutcomeKind.SUCCESS if success else OutcomeKind.FAILURE,
                utility=1.0 if success else 0.0,
                resolver="executive.goal_returned",
                detail={"decision": episode.decision, "goal": goal[:120]},
            )

        # A refusal whose consequence nobody observed. This is the common case
        # and it must stay unobserved: counting it as a success would teach her
        # that refusing is free.
        return None

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tracked_episodes": len(self._episode_goals),
                "completions": len(self._completions),
                "goal_outcomes": len(self._goal_outcomes),
                "horizon_s": self.horizon_s,
            }


_resolver: ExecutiveAdmissionResolver | None = None
_resolver_lock = threading.Lock()


def get_executive_resolver() -> ExecutiveAdmissionResolver:
    global _resolver
    if _resolver is None:
        with _resolver_lock:
            if _resolver is None:
                _resolver = ExecutiveAdmissionResolver()
    return _resolver


_installed = False


def install(*, register_services: bool = True) -> bool:
    """Bring the organ up and attach it to the control points it watches.

    Idempotent and total: a failure here costs Aura the learning and nothing
    else, so it degrades rather than raising into whatever is booting.
    """
    global _installed
    if _installed:
        return True
    try:
        core = get_ontogeny()
        core.resolvers.register(get_executive_resolver())
        from core.ontogeny import control_points, invariants, telemetry

        control_points.register(core)
        telemetry.declare()
        invariants.install()
        if register_services:
            _register_services(core)
        _installed = True
        logger.info("ontogeny: installed — %s", core.summary())
        return True
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError, ImportError) as exc:
        record_degradation(
            "ontogeny", exc,
            action="ontogeny organ not installed; every control point keeps its incumbent",
        )
        return False


def _register_services(core: Any) -> None:
    from core.container import ServiceContainer

    ServiceContainer.register_instance("ontogeny", core, required=False)


def observe_admission(
    *,
    incumbent_choice: str,
    reason: str,
    intent_id: str,
    goal: str,
    source: str,
    action_type: str,
    features: Mapping[str, float],
    priority: float,
    blocking: bool,
) -> tuple[str, dict[str, Any] | None]:
    """The executive's one call. Returns the action to take and a receipt.

    Sealed decisions are recorded but never contested; everything else goes
    through the organ, which returns the incumbent's own choice unless it has
    earned the right to differ.
    """
    try:
        if is_sealed(reason, source):
            return incumbent_choice, None
        core = get_ontogeny()
        verdict = core.consider(
            EXECUTIVE_ADMISSION,
            features,
            incumbent_choice=incumbent_choice,
            seed=intent_id,
            stakes=admission_stakes(action_type=action_type, priority=priority, blocking=blocking),
            context={"goal": goal[:200], "incumbent_reason": reason},
        )
        if verdict.episode_id:
            get_executive_resolver().note_episode(
                verdict.episode_id, goal=goal, decision=verdict.choice
            )
        return verdict.choice, verdict.as_dict()
    except (RuntimeError, ValueError, TypeError, AttributeError, KeyError) as exc:
        record_degradation(
            "ontogeny", exc, severity="warning",
            action="admission observation failed; the executive's own decision stands",
        )
        return incumbent_choice, None


def note_admission_completion(episode_id: str, *, success: bool, goal: str = "") -> None:
    try:
        get_executive_resolver().note_completion(episode_id, success=success, goal=goal)
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
        record_degradation("ontogeny", exc, severity="debug",
                           action="completion not recorded for ontogeny")


__all__ = [
    "EXECUTIVE_ADMISSION",
    "SEALED_REASONS",
    "SEALED_SOURCES",
    "ExecutiveAdmissionResolver",
    "admission_features",
    "admission_stakes",
    "get_executive_resolver",
    "install",
    "is_sealed",
    "note_admission_completion",
    "observe_admission",
]
