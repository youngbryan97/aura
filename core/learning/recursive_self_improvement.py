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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.RecursiveSelfImprovement")


@dataclass(frozen=True)
class ImprovementSignal:
    """A concrete reason to improve."""

    source: str
    kind: str
    severity: float = 0.5
    metric: str = "quality"
    delta: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ImprovementScorecard:
    """Comparable evaluation snapshot before or after a cycle."""

    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    regressions: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImprovementPlan:
    """One bounded recursive-improvement step."""

    objective: str
    actions: List[str]
    rationale: List[str]
    depth: int
    fine_tune_type: str = "lora"
    full_weights_unlocked: bool = False


@dataclass
class ImprovementCycleResult:
    """Durable result of one recursive-improvement cycle."""

    cycle_id: str
    objective: str
    depth: int
    plan: ImprovementPlan
    baseline: ImprovementScorecard
    after: ImprovementScorecard
    attempted_actions: List[str] = field(default_factory=list)
    action_results: Dict[str, Any] = field(default_factory=dict)
    promoted: bool = False
    rollback_performed: bool = False
    authorized: bool = True
    authorization_reason: str = ""
    score_delta: float = 0.0
    child_results: List["ImprovementCycleResult"] = field(default_factory=list)


Evaluator = Callable[[], Any]


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
        evaluator: Optional[Evaluator] = None,
        ledger_path: Optional[Path] = None,
        min_score_delta: float = 0.01,
        max_depth: int = 3,
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
        self._signals: List[ImprovementSignal] = []
        self._cycle_lock = asyncio.Lock()

        if ledger_path is None:
            try:
                from core.config import config

                ledger_path = Path(config.paths.data_dir) / "learning" / "recursive_self_improvement.jsonl"
            except Exception:
                ledger_path = Path.home() / ".aura" / "data" / "learning" / "recursive_self_improvement.jsonl"
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
        evidence: Optional[Dict[str, Any]] = None,
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

    def get_status(self) -> Dict[str, Any]:
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
        force: bool = False,
        depth: int = 0,
    ) -> ImprovementCycleResult:
        """Run one recursive improvement cycle, then recurse if it improves."""
        async with self._cycle_lock:
            return await self._run_cycle_locked(
                objective,
                allow_weight_update=allow_weight_update,
                allow_code_modification=allow_code_modification,
                force=force,
                depth=depth,
            )

    async def _run_cycle_locked(
        self,
        objective: str,
        *,
        allow_weight_update: bool,
        allow_code_modification: bool,
        force: bool,
        depth: int,
    ) -> ImprovementCycleResult:
        baseline = await self._evaluate()
        plan = self._make_plan(
            objective,
            allow_weight_update=allow_weight_update,
            allow_code_modification=allow_code_modification,
            force=force,
            depth=depth,
        )

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

        action_results: Dict[str, Any] = {}
        attempted: List[str] = []
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
                force=False,
                depth=depth + 1,
            )
            result.child_results.append(child)
            self._append_ledger(result)

        return result

    def _make_plan(
        self,
        objective: str,
        *,
        allow_weight_update: bool,
        allow_code_modification: bool,
        force: bool,
        depth: int,
    ) -> ImprovementPlan:
        signals = list(self._signals[-50:])
        actions: List[str] = []
        rationale: List[str] = []
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

    def _authorize(self, plan: ImprovementPlan) -> tuple[bool, str]:
        if not self.require_will_authorization:
            return True, "authorization disabled for controlled caller"
        try:
            from core.will import ActionDomain, get_will

            domain = (
                ActionDomain.SEMANTIC_WEIGHT_UPDATE
                if "weight_update" in plan.actions
                else ActionDomain.STATE_MUTATION
            )
            decision = get_will().decide(
                content=f"recursive_self_improvement:{plan.objective}:{','.join(plan.actions)}",
                source="recursive_self_improvement",
                domain=domain,
                priority=0.7,
                context={"plan": asdict(plan)},
            )
            return bool(decision.is_approved()), str(decision.reason)
        except Exception as exc:
            record_degradation("recursive_self_improvement", exc)
            if os.getenv("AURA_RSI_ALLOW_DEGRADED_OPEN", "0") == "1":
                return True, f"authorization_degraded_open:{type(exc).__name__}"
            return False, f"authorization_unavailable:{type(exc).__name__}"

    async def _run_weight_update(self) -> bool:
        if not self.live_learner or not hasattr(self.live_learner, "force_train"):
            return False
        try:
            result = self.live_learner.force_train()
            if inspect.isawaitable(result):
                result = await result
            return bool(result)
        except Exception as exc:
            record_degradation("recursive_self_improvement", exc)
            logger.error("Recursive weight update failed: %s", exc)
            return False

    async def _run_code_refinement(self) -> Dict[str, Any]:
        deterministic: Dict[str, Any] = {}
        if self.structural_improver and hasattr(self.structural_improver, "find_and_fix"):
            try:
                deterministic = await asyncio.to_thread(
                    self.structural_improver.find_and_fix,
                    max_repairs=3,
                )
                if deterministic.get("repairs_successful", 0) > 0:
                    return {"ok": bool(deterministic.get("ok", False)), "result": deterministic}
            except Exception as exc:
                record_degradation("recursive_self_improvement", exc)
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
        except Exception as exc:
            record_degradation("recursive_self_improvement", exc)
            return {"ok": False, "reason": f"{type(exc).__name__}:{exc}", "deterministic": deterministic}

    def _rollback_weights(self) -> bool:
        if self.live_learner and hasattr(self.live_learner, "rollback_adapter"):
            try:
                return bool(self.live_learner.rollback_adapter())
            except Exception as exc:
                record_degradation("recursive_self_improvement", exc)
        return False

    async def _evaluate(self) -> ImprovementScorecard:
        try:
            if self.evaluator is not None:
                raw = self.evaluator()
                if inspect.isawaitable(raw):
                    raw = await raw
                return self._coerce_scorecard(raw)
        except Exception as exc:
            record_degradation("recursive_self_improvement", exc)
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

    def _learning_stats(self) -> Dict[str, Any]:
        if self.live_learner and hasattr(self.live_learner, "get_learning_stats"):
            try:
                return dict(self.live_learner.get_learning_stats() or {})
            except Exception as exc:
                record_degradation("recursive_self_improvement", exc)
        return {}

    def _buffer_size(self) -> int:
        stats = self._learning_stats()
        try:
            return int(stats.get("buffer_size", 0) or 0)
        except Exception:
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
            with open(self.ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as exc:
            record_degradation("recursive_self_improvement", exc)
            logger.debug("Failed to write RSI ledger: %s", exc)

    def _serialize_result(self, result: ImprovementCycleResult) -> Dict[str, Any]:
        payload = asdict(result)
        payload["child_results"] = [self._serialize_result(child) for child in result.child_results]
        return payload


_instance: Optional[RecursiveSelfImprovementLoop] = None


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
    evaluator: Optional[Evaluator] = None,
    ledger_path: Optional[Path] = None,
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
