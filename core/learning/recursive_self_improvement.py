"""Governed recursive self-improvement loop.

This module closes the loop between observed failures, weight-level learning,
safe self-modification, benchmark evaluation, rollback, and the next cycle.
It does not claim metaphysical consciousness or guaranteed AGI. It provides a
real recursive improvement mechanism with explicit safety gates and bounded
depth so improvements can compound without silently eating the runtime.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from core.runtime.errors import FallbackClassification, record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.RecursiveSelfImprovement")
_RSI_RECOVERABLE_ERRORS = (
    AttributeError,
    FileNotFoundError,
    ImportError,
    json.JSONDecodeError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_rsi_degradation(
    subsystem: str,
    error: BaseException,
    *,
    action: str,
    extra: dict[str, Any] | None = None,
):
    return record_degradation(
        subsystem,
        error,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


@dataclass(frozen=True)
class ImprovementSignal:
    """A concrete reason to improve."""

    source: str
    kind: str
    severity: float = 0.5
    metric: str = "quality"
    delta: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ImprovementScorecard:
    """Comparable evaluation snapshot before or after a cycle."""

    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    regressions: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImprovementPlan:
    """One bounded recursive-improvement step."""

    objective: str
    actions: list[str]
    rationale: list[str]
    depth: int
    fine_tune_type: str = "lora"
    full_weights_unlocked: bool = False
    system2_search_id: str = ""
    system2_selected_action: str = ""
    system2_confidence: float = 0.0
    system2_reason: str = ""
    system2_receipt: dict[str, Any] = field(default_factory=dict)
    system2_outcome_receipt_id: str = ""


@dataclass
class ImprovementCycleResult:
    """Durable result of one recursive-improvement cycle."""

    cycle_id: str
    objective: str
    depth: int
    plan: ImprovementPlan
    baseline: ImprovementScorecard
    after: ImprovementScorecard
    attempted_actions: list[str] = field(default_factory=list)
    action_results: dict[str, Any] = field(default_factory=dict)
    promoted: bool = False
    rollback_performed: bool = False
    authorized: bool = True
    authorization_reason: str = ""
    score_delta: float = 0.0
    child_results: list[ImprovementCycleResult] = field(default_factory=list)


Evaluator = Callable[[], Any]


def _the_weight_update_worked(receipt: Any) -> bool:
    """Whether a compounding cycle produced something, asked of its own contract.

    This asked whether the status was "promoted", which is not a value the
    compounding contract has ever produced: a cycle ends as a candidate or a
    qualified adapter, and moving the active model pointer is a separate
    staged act on purpose. So a cycle that trained and qualified an adapter
    was recorded as a failed weight update — and the rollback that only runs
    when a weight action SUCCEEDED could not run either. One wrong string, two
    mechanisms that could not fire, and the tests could not see it because
    their learner returns a boolean and never reaches this branch at all.

    Asked of the contract rather than compared against a copy of it, so the
    two cannot drift apart again.
    """
    from core.learning.weight_compounding import WORKED  # noqa: PLC0415

    if hasattr(receipt, "worked"):
        return bool(receipt.worked())
    if isinstance(receipt, dict):
        return str(receipt.get("status") or "") in WORKED
    return bool(receipt)


class RecursiveSelfImprovementLoop:
    """Coordinates recursive improvement across weights and source code.

    The loop is deliberately bounded:
      * every cycle has a before/after scorecard;
      * learned weights must pass the learner's benchmark and this loop's
        after-evaluation before remaining promoted;
      * failed weight cycles roll back the active adapter when possible;
      * code changes are delegated to the existing safe self-modification
        pipeline, not written directly here;
      * recursion stops at ``max_depth`` or when the marginal gain drops below
        ``min_score_delta``.
    """

    def __init__(
        self,
        *,
        live_learner: Any = None,
        self_modifier: Any = None,
        structural_improver: Any = None,
        evaluator: Evaluator | None = None,
        ledger_path: Path | None = None,
        min_score_delta: float = 0.01,
        max_depth: int = 5,
        auto_recurse: bool = True,
        require_will_authorization: bool = True,
    ):
        self.live_learner = live_learner
        self.self_modifier = self_modifier
        self.structural_improver = structural_improver
        self.evaluator = evaluator
        self.min_score_delta = max(0.0, float(min_score_delta))
        self.max_depth = max(1, int(max_depth))
        self.auto_recurse = bool(auto_recurse)
        self.require_will_authorization = bool(require_will_authorization)
        self._signals: list[ImprovementSignal] = []
        self._cycle_lock = asyncio.Lock()

        if ledger_path is None:
            try:
                from core.config import config

                ledger_path = Path(config.paths.data_dir) / "learning" / "recursive_self_improvement.jsonl"
            except (ImportError, AttributeError, RuntimeError):
                ledger_path = state_root() / "data" / "learning" / "recursive_self_improvement.jsonl"
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def record_signal(
        self,
        source: str,
        kind: str,
        *,
        severity: float = 0.5,
        metric: str = "quality",
        delta: float = 0.0,
        evidence: dict[str, Any] | None = None,
    ) -> ImprovementSignal:
        signal = ImprovementSignal(
            source=source,
            kind=kind,
            severity=max(0.0, min(1.0, float(severity))),
            metric=metric,
            delta=float(delta),
            evidence=evidence or {},
        )
        self._signals.append(signal)
        self._signals = self._signals[-500:]
        return signal

    def get_status(self) -> dict[str, Any]:
        return {
            "signals": len(self._signals),
            "max_depth": self.max_depth,
            "auto_recurse": self.auto_recurse,
            "min_score_delta": self.min_score_delta,
            "ledger_path": str(self.ledger_path),
            "live_learner": bool(self.live_learner),
            "self_modifier": bool(self.self_modifier),
            "structural_improver": bool(self.structural_improver),
        }

    async def run_cycle(
        self,
        objective: str,
        *,
        allow_weight_update: bool = True,
        allow_code_modification: bool = False,
        allow_tool_creation: bool = False,
        force: bool = False,
        depth: int = 0,
    ) -> ImprovementCycleResult:
        """Run one recursive improvement cycle, then recurse if it improves."""
        async with self._cycle_lock:
            return await self._run_cycle_locked(
                objective,
                allow_weight_update=allow_weight_update,
                allow_code_modification=allow_code_modification,
                allow_tool_creation=allow_tool_creation,
                force=force,
                depth=depth,
            )

    async def _run_cycle_locked(
        self,
        objective: str,
        *,
        allow_weight_update: bool,
        allow_code_modification: bool,
        allow_tool_creation: bool = False,
        force: bool,
        depth: int,
    ) -> ImprovementCycleResult:
        baseline = await self._evaluate()
        plan = self._make_plan(
            objective,
            allow_weight_update=allow_weight_update,
            allow_code_modification=allow_code_modification,
            allow_tool_creation=allow_tool_creation,
            force=force,
            depth=depth,
        )
        plan = await self._refine_plan_with_native_system2(plan, baseline)

        authorized, reason = self._authorize(plan)
        if not authorized:
            result = ImprovementCycleResult(
                cycle_id=self._cycle_id(depth),
                objective=objective,
                depth=depth,
                plan=plan,
                baseline=baseline,
                after=baseline,
                authorized=False,
                authorization_reason=reason,
            )
            self._append_ledger(result)
            return result

        # Ranking produced provenance only.  The authorized cycle is the point
        # where the selected plan becomes a real action, so commit its expected
        # outcome here and retain the only id that can resolve it.
        if plan.system2_search_id:
            try:
                from core.container import ServiceContainer

                system2 = ServiceContainer.get("native_system2", default=None)
                if system2 is not None:
                    outcome_id = system2.open_outcome_receipt(
                        plan.system2_search_id,
                        category="recursive_self_improvement",
                        horizon_s=7200.0,
                    )
                    if outcome_id:
                        plan = replace(plan, system2_outcome_receipt_id=outcome_id)
            except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as exc:
                _record_rsi_degradation(
                    "recursive_self_improvement.outcome_open",
                    exc,
                    action="continued authorized cycle without an outcome receipt; "
                    "the cycle remains auditable in the RSI ledger",
                )

        action_results: dict[str, Any] = {}
        attempted: list[str] = []
        weight_action_ran = False
        weight_action_succeeded = False

        for action in plan.actions:
            attempted.append(action)
            if action == "weight_update":
                weight_action_ran = True
                ok = await self._run_weight_update()
                weight_action_succeeded = bool(ok)
                action_results[action] = {"ok": bool(ok)}
            elif action == "code_refinement":
                action_results[action] = await self._run_code_refinement()
            elif action == "tool_creation":
                action_results[action] = await self._run_tool_creation(plan)
            elif action == "collect_more_signal":
                action_results[action] = {"ok": True, "reason": "insufficient signal for mutation"}

        after = await self._evaluate()
        delta = after.score - baseline.score
        action_ok = all(bool(v.get("ok", False)) for v in action_results.values()) if action_results else False
        no_regressions = not after.regressions
        promoted = bool(action_ok and no_regressions and (delta >= self.min_score_delta or force))
        rollback = False

        if weight_action_ran and weight_action_succeeded and not promoted:
            rollback = self._rollback_weights()

        result = ImprovementCycleResult(
            cycle_id=self._cycle_id(depth),
            objective=objective,
            depth=depth,
            plan=plan,
            baseline=baseline,
            after=after,
            attempted_actions=attempted,
            action_results=action_results,
            promoted=promoted,
            rollback_performed=rollback,
            authorized=True,
            authorization_reason=reason,
            score_delta=delta,
        )
        self._append_ledger(result)

        if plan.system2_outcome_receipt_id:
            try:
                from core.container import ServiceContainer

                system2 = ServiceContainer.get("native_system2", default=None)
                if system2 is not None:
                    system2.resolve_outcome_receipt(
                        plan.system2_outcome_receipt_id,
                        1.0 if promoted else 0.0,
                        note=(
                            f"RSI cycle promoted={promoted} delta={delta:.6f} "
                            f"regressions={len(after.regressions)}"
                        ),
                    )
            except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as exc:
                _record_rsi_degradation(
                    "recursive_self_improvement.outcome_resolve",
                    exc,
                    action="kept measured RSI result in its ledger after the "
                    "System 2 outcome receipt could not be resolved",
                )

        if (
            promoted
            and self.auto_recurse
            and depth + 1 < self.max_depth
            and self._should_recurse(result)
        ):
            child = await self._run_cycle_locked(
                f"{objective} :: recursive pass {depth + 2}",
                allow_weight_update=allow_weight_update,
                allow_code_modification=allow_code_modification,
                allow_tool_creation=allow_tool_creation,
                force=False,
                depth=depth + 1,
            )
            result.child_results.append(child)
            self._append_ledger(result)

        return result


    @staticmethod
    def _remember_the_gaps(signals: list) -> None:
        """Tell the gap ledger about capability gaps this loop just observed.

        The two halves of this were never joined. Gaps are recognised here, as
        signals of kind capability_gap, tool_gap, missing_tool and
        unmet_affordance. Counting them, and forging a skill for one seen
        often enough, lives in :mod:`core.agi.skill_synthesizer`, whose
        ``log_gap`` had no caller anywhere but a test — so the forge's tests
        passed over a path nothing in the running system could reach, and
        every gap Aura correctly noticed was forgotten as soon as she noticed
        it.

        Recording only. This counts what happened and decides nothing: the
        forge keeps the gating it already has and nothing here proposes or
        executes anything. What changes is that a recurring gap can now BE
        recurring, which is the premise the ledger was built on.
        """

        if not signals:
            return
        try:
            from core.agi.skill_synthesizer import get_skill_synthesizer

            ledger = get_skill_synthesizer()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "recursive_self_improvement",
                exc,
                action="counted this cycle's capability gaps in memory only",
            )
            return
        for signal in signals:
            # What was missing, in words, because the ledger counts REPEATS of
            # a description. Reading the signal's kind or metric would give
            # every gap the same key — "capability" for all of them — and a
            # ledger built to notice the same gap twice would see one gap
            # forever.
            evidence = getattr(signal, "evidence", None)
            described = ""
            if isinstance(evidence, dict):
                for key in ("task", "detail", "description", "goal", "what"):
                    described = str(evidence.get(key) or "").strip()
                    if described:
                        break
            if not described:
                described = str(getattr(signal, "source", "") or "").strip()
            if not described:
                continue
            try:
                ledger.log_gap(described, str(getattr(signal, "kind", "") or ""))
            except (AttributeError, TypeError, ValueError) as exc:
                record_degradation(
                    "recursive_self_improvement",
                    exc,
                    action="skipped one capability gap the ledger would not accept",
                )

    def _make_plan(
        self,
        objective: str,
        *,
        allow_weight_update: bool,
        allow_code_modification: bool,
        allow_tool_creation: bool = False,
        force: bool,
        depth: int,
    ) -> ImprovementPlan:
        signals = list(self._signals[-50:])
        actions: list[str] = []
        rationale: list[str] = []
        stats = self._learning_stats()
        policy = stats.get("training_policy", {}) if isinstance(stats, dict) else {}

        weight_signal = any(
            s.kind in {"low_quality", "user_confusion", "benchmark_regression", "training_data_ready"}
            or s.metric in {"quality", "accuracy", "preference"}
            for s in signals
        )
        runtime_signal = any(
            s.kind in {"runtime_error", "test_failure", "boot_degradation", "regression"}
            or s.metric in {"stability", "latency", "reliability"}
            for s in signals
        )
        gaps_observed = [
            s
            for s in signals
            if s.kind in {"capability_gap", "tool_gap", "missing_tool", "unmet_affordance"}
            or s.metric in {"capability", "coverage"}
        ]
        capability_gap_signal = bool(gaps_observed)
        self._remember_the_gaps(gaps_observed)

        if allow_weight_update and self.live_learner and (force or weight_signal or self._buffer_size() > 0):
            actions.append("weight_update")
            rationale.append("experience buffer and evaluation signals can update model weights")

        if (
            allow_code_modification
            and (self.self_modifier or self.structural_improver)
            and (force or runtime_signal)
        ):
            actions.append("code_refinement")
            rationale.append("runtime/test signals can be routed through safe self-modification")

        # Tool-creation is the loosest RSI lever, so it is the most tightly gated: opt-in
        # per call, behind an env flag, and only when a real capability gap is observed.
        # Critically it produces a *reversible proposal*, never an executed tool, so the
        # 'keep reversibility' invariant holds even as the loop is allowed to reach for
        # new capabilities.
        if (
            allow_tool_creation
            and os.getenv("AURA_RSI_TOOL_CREATION", "0") == "1"
            and (force or capability_gap_signal)
        ):
            actions.append("tool_creation")
            rationale.append("an observed capability gap can be met by a reversible new-tool proposal")

        if not actions:
            actions.append("collect_more_signal")
            rationale.append("no authorized improvement action has enough evidence yet")

        requested_fine_tune = str(policy.get("fine_tune_type", "lora")).lower()
        full_weights_unlocked = bool(policy.get("full_weights_unlocked", False)) and (
            os.getenv("AURA_RSI_FULL_WEIGHTS_UNLOCKED", "0") == "1"
        )
        fine_tune_type = "full" if requested_fine_tune == "full" and full_weights_unlocked else "lora"

        return ImprovementPlan(
            objective=objective,
            actions=actions,
            rationale=rationale,
            depth=depth,
            fine_tune_type=fine_tune_type,
            full_weights_unlocked=full_weights_unlocked,
        )

    async def _refine_plan_with_native_system2(
        self,
        plan: ImprovementPlan,
        baseline: ImprovementScorecard,
    ) -> ImprovementPlan:
        """Use Aura's governed Native System 2 substrate to rank RSI actions.

        System 2 does not mutate weights or source here. It evaluates the
        candidate improvement actions, records an auditable search receipt, and
        may reorder or defer the planned actions before the existing Will gate
        authorizes anything effectful.
        """
        try:
            from core.container import ServiceContainer
            from core.reasoning.native_system2 import SearchAlgorithm, System2SearchConfig

            system2 = ServiceContainer.get("native_system2", default=None)
            if system2 is None:
                return plan

            candidate_names = list(dict.fromkeys([*plan.actions, "collect_more_signal"]))
            if len(candidate_names) < 2:
                return plan

            action_profiles = {
                "weight_update": {
                    "prior": 0.65,
                    "risk": 0.45,
                    "external_side_effect": True,
                    "summary": "fine-tune/update LoRA weights through governed learner",
                },
                "code_refinement": {
                    "prior": 0.6,
                    "risk": 0.65,
                    "external_side_effect": True,
                    "summary": "route code changes through safe self-modification",
                },
                "collect_more_signal": {
                    "prior": 0.35,
                    "risk": 0.05,
                    "external_side_effect": False,
                    "summary": "defer mutation and collect more evidence",
                },
            }
            ranked = await system2.rank_actions(
                context=json.dumps(
                    {
                        "objective": plan.objective,
                        "current_actions": plan.actions,
                        "rationale": plan.rationale,
                        "depth": plan.depth,
                        "fine_tune_type": plan.fine_tune_type,
                        "baseline": asdict(baseline),
                    },
                    sort_keys=True,
                    default=str,
                )[:1800],
                actions=[
                    {
                        "name": name,
                        "prior": action_profiles.get(name, {}).get("prior", 0.4),
                        "risk": action_profiles.get(name, {}).get("risk", 0.25),
                        "external_side_effect": action_profiles.get(name, {}).get("external_side_effect", False),
                        "metadata": {
                            "summary": action_profiles.get(name, {}).get("summary", name),
                            "rsi_action": True,
                            "score_hint": action_profiles.get(name, {}).get("prior", 0.4),
                        },
                    }
                    for name in candidate_names
                ],
                config=System2SearchConfig(
                    algorithm=SearchAlgorithm.HYBRID,
                    budget=32,
                    max_depth=2,
                    branching_factor=max(2, len(candidate_names)),
                    beam_width=min(4, len(candidate_names)),
                    confidence_threshold=0.55,
                ),
                source="recursive_self_improvement",
            )
            selected = ranked.committed_action
            if selected is None:
                return plan
            chosen = str(selected.metadata.get("verifies") or selected.name)
            if chosen.startswith("verify:"):
                chosen = chosen[len("verify:") :]

            if chosen == "collect_more_signal" and not any(action == "collect_more_signal" for action in plan.actions):
                actions = ["collect_more_signal"]
            elif chosen in plan.actions:
                actions = [chosen, *[action for action in plan.actions if action != chosen]]
            else:
                actions = list(plan.actions)

            rationale = [
                *plan.rationale,
                f"Native System 2 selected {chosen}: {ranked.receipt.commitment_reason}",
            ]
            return replace(
                plan,
                actions=actions,
                rationale=rationale,
                system2_search_id=ranked.search_id,
                system2_selected_action=chosen,
                system2_confidence=ranked.confidence,
                system2_reason=ranked.receipt.commitment_reason,
                system2_receipt=ranked.receipt.to_dict(),
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_rsi_degradation(
                "recursive_self_improvement.native_system2",
                exc,
                action="kept original RSI plan after native System 2 ranking was unavailable",
                extra={"objective": plan.objective, "actions": plan.actions},
            )
            return plan

    def _authorize(self, plan: ImprovementPlan) -> tuple[bool, str]:
        if not self.require_will_authorization:
            return True, "authorization disabled for controlled caller"
        try:
            from core.will import ActionDomain, get_will

            if "tool_creation" in plan.actions:
                domain = ActionDomain.SELF_MODIFICATION
            elif "weight_update" in plan.actions:
                domain = ActionDomain.SEMANTIC_WEIGHT_UPDATE
            else:
                domain = ActionDomain.STATE_MUTATION
            decision = get_will().decide(
                content=f"recursive_self_improvement:{plan.objective}:{','.join(plan.actions)}",
                source="recursive_self_improvement",
                domain=domain,
                priority=0.7,
                context={"plan": asdict(plan)},
            )
            return bool(decision.is_approved()), str(decision.reason)
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_rsi_degradation(
                "recursive_self_improvement",
                exc,
                action="failed RSI authorization closed unless explicit degraded-open override is set",
                extra={"objective": plan.objective, "actions": plan.actions},
            )
            if os.getenv("AURA_RSI_ALLOW_DEGRADED_OPEN", "0") == "1":
                return True, f"authorization_degraded_open:{type(exc).__name__}"
            return False, f"authorization_unavailable:{type(exc).__name__}"

    async def _run_weight_update(self) -> bool:
        if not self.live_learner or not hasattr(self.live_learner, "force_train"):
            return False
        try:
            # Consolidation: when wired with the REAL LiveLearner (live
            # runtime), the weight update routes through the canonical
            # compounding scheduler — Will-approved, admission-controlled,
            # sealed-gate, ledger-recorded. The legacy force_train path stays
            # only for injected test doubles (gauntlet), which validate the
            # plumbing without real training.
            if self._is_real_live_learner():
                from core.runtime.service_access import resolve_weight_compounding

                scheduler = resolve_weight_compounding(default=None)
                if scheduler is not None and hasattr(scheduler, "run_cycle_now"):
                    receipt = await scheduler.run_cycle_now(reason="rsi_weight_update")
                    return _the_weight_update_worked(receipt)

            result = self.live_learner.force_train()
            if inspect.isawaitable(result):
                result = await result
            return bool(result)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_rsi_degradation(
                "recursive_self_improvement",
                exc,
                action="marked weight update failed and blocked RSI promotion",
            )
            logger.error("Recursive weight update failed: %s", exc)
            return False

    def _is_real_live_learner(self) -> bool:
        """True when self.live_learner is the runtime singleton, not a test double."""
        try:
            import core.learning.live_learner as live_learner_module

            return live_learner_module._learner is self.live_learner
        except (ImportError, AttributeError):
            return False

    async def _run_code_refinement(self) -> dict[str, Any]:
        deterministic: dict[str, Any] = {}
        if self.structural_improver and hasattr(self.structural_improver, "find_and_fix"):
            try:
                deterministic = await asyncio.to_thread(
                    self.structural_improver.find_and_fix,
                    max_repairs=3,
                )
                if deterministic.get("repairs_successful", 0) > 0:
                    return {"ok": bool(deterministic.get("ok", False)), "result": deterministic}
            except (OSError, ConnectionError, TimeoutError) as exc:
                _record_rsi_degradation(
                    "recursive_self_improvement",
                    exc,
                    action="continued to governed self-modifier after deterministic structural repair failed",
                    extra={"max_repairs": 3},
                )
                deterministic = {"ok": False, "reason": f"structural_improver:{type(exc).__name__}:{exc}"}

        if not self.self_modifier:
            return {"ok": False, "reason": "self_modifier_unavailable", "deterministic": deterministic}
        try:
            if hasattr(self.self_modifier, "run_refinement_cycle"):
                result = self.self_modifier.run_refinement_cycle()
            elif hasattr(self.self_modifier, "run_auto_fix_cycle"):
                result = self.self_modifier.run_auto_fix_cycle()
            else:
                return {"ok": False, "reason": "no_refinement_api"}
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, dict):
                return {"ok": bool(result.get("success", False)), "result": result, "deterministic": deterministic}
            return {"ok": bool(result), "result": result, "deterministic": deterministic}
        except (OSError, ConnectionError, TimeoutError) as exc:
            _record_rsi_degradation(
                "recursive_self_improvement",
                exc,
                action="failed code refinement closed and returned non-promotable action result",
                extra={"deterministic": deterministic},
            )
            return {"ok": False, "reason": f"{type(exc).__name__}:{exc}", "deterministic": deterministic}

    async def _run_tool_creation(self, plan: ImprovementPlan) -> dict[str, Any]:
        """Propose a new tool to close a capability gap — *reversibly*.

        This never executes or installs a tool. It writes a draft tool proposal to a
        proposals ledger that a downstream governed promotion step (or a human) can review,
        and registers the inverse op (delete the draft) so the action is fully reversible.
        If a self_modifier exposes a dedicated `propose_tool` API we route to it but still
        require the result to declare itself reversible; otherwise we fall back to the
        local draft. Anything that cannot be made reversible is refused.
        """
        proposal = {
            "objective": plan.objective,
            "depth": plan.depth,
            "kind": "tool_creation_proposal",
            "rationale": plan.rationale,
            "created_at": time.time(),
            "status": "proposed",
        }

        # Prefer a governed self-modifier tool-proposal API if present.
        if self.self_modifier and hasattr(self.self_modifier, "propose_tool"):
            try:
                result = self.self_modifier.propose_tool(proposal)
                if inspect.isawaitable(result):
                    result = await result
                reversible = bool(isinstance(result, dict) and result.get("reversible"))
                ok = bool(isinstance(result, dict) and result.get("ok")) and reversible
                if not reversible:
                    return {"ok": False, "reason": "tool_proposal_not_reversible", "result": result}
                return {"ok": ok, "reversible": True, "result": result}
            except (OSError, ConnectionError, TimeoutError, RuntimeError, ValueError) as exc:
                _record_rsi_degradation(
                    "recursive_self_improvement",
                    exc,
                    action="tool-creation proposal via self_modifier failed; refused (no irreversible fallback)",
                )
                return {"ok": False, "reason": f"propose_tool:{type(exc).__name__}:{exc}"}

        # Local fallback: persist a reversible draft proposal (no execution).
        try:
            proposals_path = self.ledger_path.parent / "tool_proposals.jsonl"
            proposals_path.parent.mkdir(parents=True, exist_ok=True)
            proposal_id = f"toolprop-{int(proposal['created_at'])}-{plan.depth}"
            proposal["proposal_id"] = proposal_id
            with proposals_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(proposal) + "\n")
            return {
                "ok": True,
                "reversible": True,
                "proposal_id": proposal_id,
                "path": str(proposals_path),
                "rollback": "remove the appended proposal line",
                "executed": False,
            }
        except OSError as exc:
            _record_rsi_degradation(
                "recursive_self_improvement",
                exc,
                action="failed to persist reversible tool proposal; refused",
            )
            return {"ok": False, "reason": f"proposal_persist:{type(exc).__name__}:{exc}"}

    def _rollback_weights(self) -> bool:
        if self.live_learner and hasattr(self.live_learner, "rollback_adapter"):
            try:
                return bool(self.live_learner.rollback_adapter())
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                _record_rsi_degradation(
                    "recursive_self_improvement",
                    exc,
                    action="reported weight rollback failure and left cycle unpromoted",
                )
        return False

    async def _evaluate(self) -> ImprovementScorecard:
        try:
            if self.evaluator is not None:
                raw = self.evaluator()
                if inspect.isawaitable(raw):
                    raw = await raw
                return self._coerce_scorecard(raw)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_rsi_degradation(
                "recursive_self_improvement",
                exc,
                action="used zero scorecard and regression marker after evaluator failed",
            )
            return ImprovementScorecard(score=0.0, regressions=[f"evaluator_error:{type(exc).__name__}"])

        stats = self._learning_stats()
        quality = float(stats.get("session_avg_quality", 0.0) or 0.0) if isinstance(stats, dict) else 0.0
        recent = self._signals[-20:]
        pressure = sum(s.severity for s in recent) / max(1, len(recent))
        score = max(0.0, min(1.0, quality if quality > 0 else 1.0 - pressure * 0.5))
        return ImprovementScorecard(
            score=score,
            metrics={"session_quality": quality, "signal_pressure": pressure},
            evidence={"signals": len(recent)},
        )

    @staticmethod
    def _coerce_scorecard(raw: Any) -> ImprovementScorecard:
        if isinstance(raw, ImprovementScorecard):
            return raw
        if isinstance(raw, (int, float)):
            return ImprovementScorecard(score=max(0.0, min(1.0, float(raw))))
        if isinstance(raw, dict):
            metrics = {
                str(k): float(v)
                for k, v in (raw.get("metrics") or {}).items()
                if isinstance(v, (int, float))
            }
            if "score" in raw:
                score = float(raw["score"])
            elif metrics:
                score = sum(metrics.values()) / len(metrics)
            else:
                score = 0.0
            return ImprovementScorecard(
                score=max(0.0, min(1.0, score)),
                metrics=metrics,
                regressions=[str(x) for x in raw.get("regressions", [])],
                evidence=dict(raw.get("evidence") or {}),
            )
        return ImprovementScorecard(score=0.0, regressions=["invalid_scorecard"])

    def _learning_stats(self) -> dict[str, Any]:
        if self.live_learner and hasattr(self.live_learner, "get_learning_stats"):
            try:
                return dict(self.live_learner.get_learning_stats() or {})
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                _record_rsi_degradation(
                    "recursive_self_improvement",
                    exc,
                    action="used empty learning stats after learner stats read failed",
                )
        return {}

    def _buffer_size(self) -> int:
        stats = self._learning_stats()
        try:
            return int(stats.get("buffer_size", 0) or 0)
        except (OSError, ConnectionError, TimeoutError) as exc:
            _record_rsi_degradation(
                "recursive_self_improvement",
                exc,
                action="treated buffer size as zero after learning stats parse failed",
            )
            logger.debug("Learning stats buffer-size read failed: %s", exc)
            return 0

    def _should_recurse(self, result: ImprovementCycleResult) -> bool:
        if result.score_delta < self.min_score_delta:
            return False
        if "collect_more_signal" in result.attempted_actions:
            return False
        return True

    def _cycle_id(self, depth: int) -> str:
        return f"rsi-{int(time.time() * 1000)}-{depth}"

    def _append_ledger(self, result: ImprovementCycleResult) -> None:
        try:
            payload = self._serialize_result(result)
            from core.runtime.file_write_gateway import get_file_write_gateway

            get_file_write_gateway().append_text(
                self.ledger_path,
                json.dumps(payload, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
                source="recursive_self_improvement.append_ledger",
            )
        except _RSI_RECOVERABLE_ERRORS as exc:
            _record_rsi_degradation(
                "recursive_self_improvement",
                exc,
                action="kept RSI cycle result in memory after durable ledger append failed",
                extra={"ledger_path": str(self.ledger_path), "cycle_id": result.cycle_id},
            )
            logger.debug("Failed to write RSI ledger: %s", exc)

    def _serialize_result(self, result: ImprovementCycleResult) -> dict[str, Any]:
        payload = asdict(result)
        payload["child_results"] = [self._serialize_result(child) for child in result.child_results]
        return payload


_instance: RecursiveSelfImprovementLoop | None = None


def get_recursive_self_improvement_loop() -> RecursiveSelfImprovementLoop:
    global _instance
    if _instance is None:
        from core.container import ServiceContainer

        _instance = RecursiveSelfImprovementLoop(
            live_learner=ServiceContainer.get("live_learner", default=None),
            self_modifier=ServiceContainer.get("self_modification_engine", default=None),
            structural_improver=ServiceContainer.get("structural_improver", default=None),
        )
    return _instance


def register_recursive_self_improvement_loop(
    *,
    live_learner: Any = None,
    self_modifier: Any = None,
    structural_improver: Any = None,
    evaluator: Evaluator | None = None,
    ledger_path: Path | None = None,
) -> RecursiveSelfImprovementLoop:
    global _instance
    _instance = RecursiveSelfImprovementLoop(
        live_learner=live_learner,
        self_modifier=self_modifier,
        structural_improver=structural_improver,
        evaluator=evaluator,
        ledger_path=ledger_path,
    )
    return _instance
