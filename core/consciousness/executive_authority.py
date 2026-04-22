from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from core.container import ServiceContainer
from core.runtime import service_access

logger = logging.getLogger("Aura.ExecutiveAuthority")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


@dataclass
class ExecutiveAuthoritySnapshot:
    last_action: str = "idle"
    last_reason: str = ""
    last_source: str = ""
    last_goal: str = ""
    last_content: str = ""
    last_target: str = ""
    last_decision_at: float = 0.0
    primary_releases: int = 0
    secondary_releases: int = 0
    suppressed: int = 0
    queued_initiatives: int = 0


class ExecutiveAuthority:
    """Single control surface for autonomous initiatives and spontaneous output.

    This broker does not replace MindTick or ExecutiveClosure. It makes sure
    legacy/background subsystems must ask before they inject new goals or
    user-facing speech into the organism.
    """

    _PRIMARY_SILENCE_WINDOW_S = 45.0
    _VISIBLE_PRESENCE_IDLE_WINDOW_S = 6.0
    _DEDUP_WINDOW_S = 180.0

    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self._snapshot = ExecutiveAuthoritySnapshot()
        self._recent_fingerprints: Dict[str, float] = {}

    def bind(self, orchestrator: Any) -> None:
        if orchestrator is not None:
            self.orchestrator = orchestrator

    @staticmethod
    def _is_visible_presence(metadata: Optional[Dict[str, Any]]) -> bool:
        meta = dict(metadata or {})
        return bool(
            meta.get("visible_presence")
            or meta.get("overt_presence")
            or meta.get("initiative_activity")
        )

    def _autonomy_pause_reason(self) -> str:
        try:
            router = ServiceContainer.get("llm_router", default=None)
            if router and getattr(router, "high_pressure_mode", False):
                return "memory_pressure"

            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "_background_local_deferral_reason"):
                reason = str(gate._background_local_deferral_reason(origin="executive_authority") or "").strip()
                if reason:
                    return reason
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
        return ""

    def get_status(self) -> Dict[str, Any]:
        return asdict(self._snapshot)

    def _build_initiative(
        self,
        *,
        goal: str,
        source: str,
        kind: str,
        urgency: float,
        triggered_by: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
        status: str = "suggested",
    ) -> Dict[str, Any]:
        return {
            "type": kind,
            "goal": goal,
            "source": source,
            "triggered_by": triggered_by or source,
            "urgency": round(_clamp01(urgency), 4),
            "timestamp": float(timestamp or time.time()),
            "status": status,
            "metadata": dict(metadata or {}),
        }

    def _sort_initiatives(self, initiatives: list[dict]) -> list[dict]:
        initiatives.sort(
            key=lambda item: (
                float(item.get("urgency", 0.0) or 0.0),
                float(item.get("timestamp", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return initiatives[:10]

    def _goal_runtime_policy(self, initiative: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(initiative.get("metadata", {}) or {})
        triggered_by = str(initiative.get("triggered_by", "") or metadata.get("triggered_by", "") or "").strip().lower()
        kind = str(initiative.get("type", "") or metadata.get("kind", "") or "").strip().lower()
        source = str(initiative.get("source", "") or metadata.get("source", "") or "").strip().lower()

        horizon = str(metadata.get("horizon", "") or "").strip().lower().replace("-", "_")
        if horizon not in {"short_term", "long_term"}:
            if (
                bool(metadata.get("continuity_obligation", False))
                or bool(initiative.get("continuity_obligation", False))
                or bool(metadata.get("continuity_restored", False))
                or bool(initiative.get("continuity_restored", False))
                or triggered_by in {"continuity", "integrity", "growth", "stability"}
                or source in {"executive_closure", "motivation_update"}
                or kind == "motivational_drive"
            ):
                horizon = "long_term"
            else:
                horizon = "short_term"

        explicit_quick_win = metadata.get("quick_win", None)
        if explicit_quick_win is None:
            quick_win = horizon == "short_term" and triggered_by in {"curiosity", "social", "social_hunger", "boredom"}
        else:
            quick_win = bool(explicit_quick_win)

        attention_policy = str(
            metadata.get("attention_policy")
            or ("interruptible" if quick_win else "sustained")
        ).strip().lower()
        if attention_policy not in {"interruptible", "sustained"}:
            attention_policy = "interruptible" if quick_win else "sustained"

        priority = round(
            _clamp01(
                metadata.get("priority", initiative.get("urgency", 0.5))
            ),
            4,
        )

        return {
            "horizon": horizon,
            "quick_win": quick_win,
            "attention_policy": attention_policy,
            "priority": priority,
        }

    async def _bind_objective_to_goal_engine(self, initiative: Dict[str, Any]) -> Dict[str, Any]:
        goal = _normalize_text(initiative.get("goal", ""))
        if not goal:
            return {}
        try:
            goal_engine = ServiceContainer.get("goal_engine", default=None)
        except Exception:
            goal_engine = None
        if goal_engine is None or not hasattr(goal_engine, "add_goal"):
            return {}

        policy = self._goal_runtime_policy(initiative)
        metadata = dict(initiative.get("metadata", {}) or {})
        binding_metadata = {
            "initiative_source": initiative.get("source"),
            "initiative_kind": initiative.get("type"),
            "triggered_by": initiative.get("triggered_by"),
            "governed": True,
            **metadata,
        }
        try:
            record = await goal_engine.add_goal(
                goal,
                objective=goal,
                status="in_progress",
                horizon=policy["horizon"],
                source="executive_authority",
                priority=policy["priority"],
                quick_win=policy["quick_win"],
                attention_policy=policy["attention_policy"],
                metadata=binding_metadata,
            )
            if isinstance(record, dict):
                return {
                    **policy,
                    "goal_id": str(record.get("id", "") or ""),
                    "status": str(record.get("status", "") or ""),
                }
        except Exception as exc:
            logger.debug("ExecutiveAuthority goal binding skipped: %s", exc)
        return policy

    async def propose_initiative_to_state(
        self,
        state: Any,
        goal: str,
        *,
        source: str,
        kind: str = "autonomous_thought",
        urgency: float = 0.5,
        triggered_by: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        goal = _normalize_text(goal)
        if state is None:
            return state, self._record("rejected", "state_missing", source=source, goal=goal)
        if len(goal) < 4:
            return state, self._record("rejected", "empty_goal", source=source, goal=goal)

        current_objective = _normalize_text(getattr(state.cognition, "current_objective", ""))
        if current_objective and current_objective == goal:
            return state, self._record("duplicate", "already_current_objective", source=source, goal=goal)

        now = time.time()
        initiatives = list(getattr(state.cognition, "pending_initiatives", []) or [])
        matched = None
        for item in initiatives:
            if not isinstance(item, dict):
                continue
            if _normalize_text(item.get("goal", "")) == goal:
                matched = item
                break

        if matched is not None:
            new_state = state.derive("executive_authority_refresh", origin=source)
            refreshed = list(getattr(new_state.cognition, "pending_initiatives", []) or [])
            for item in refreshed:
                if not isinstance(item, dict):
                    continue
                if _normalize_text(item.get("goal", "")) == goal:
                    item["urgency"] = max(float(item.get("urgency", 0.0) or 0.0), _clamp01(urgency))
                    item["timestamp"] = now
                    item["source"] = item.get("source") or source
                    item.setdefault("triggered_by", triggered_by or source)
                    item["metadata"] = {
                        **dict(item.get("metadata", {}) or {}),
                        **dict(metadata or {}),
                    }
                    break
            refreshed = self._sort_initiatives(refreshed)
            new_state.cognition.pending_initiatives = refreshed
            self._record_constitutional_decision(
                kind="initiative",
                source=source,
                summary=goal,
                outcome="queued",
                reason="initiative_refreshed",
                target="pending_initiatives",
                payload={"urgency": urgency, "metadata": metadata or {}},
                state=new_state,
            )
            return new_state, self._record(
                "queued",
                "initiative_refreshed",
                source=source,
                goal=goal,
                queued_initiatives=len(refreshed),
                target="pending_initiatives",
            )

        new_state = state.derive("executive_authority_initiative", origin=source)
        new_initiatives = list(getattr(new_state.cognition, "pending_initiatives", []) or [])
        new_initiatives.append(
            self._build_initiative(
                goal=goal,
                source=source,
                kind=kind,
                urgency=urgency,
                triggered_by=triggered_by,
                metadata=metadata,
                timestamp=now,
            )
        )
        new_initiatives = self._sort_initiatives(new_initiatives)
        new_state.cognition.pending_initiatives = new_initiatives
        self._record_constitutional_decision(
            kind="initiative",
            source=source,
            summary=goal,
            outcome="queued",
            reason="initiative_queued",
            target="pending_initiatives",
            payload={"urgency": urgency, "metadata": metadata or {}},
            state=new_state,
        )

        return new_state, self._record(
            "queued",
            "initiative_queued",
            source=source,
            goal=goal,
            queued_initiatives=len(new_initiatives),
            target="pending_initiatives",
        )

    async def queue_initiative(
        self,
        goal: str,
        *,
        source: str,
        kind: str = "autonomous_thought",
        urgency: float = 0.5,
        triggered_by: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        repo = self._get_state_repository()
        if repo is None:
            return self._record("rejected", "state_repository_missing", source=source, goal=_normalize_text(goal))

        state = await repo.get_current()
        new_state, decision = await self.propose_initiative_to_state(
            state,
            goal,
            source=source,
            kind=kind,
            urgency=urgency,
            triggered_by=triggered_by,
            metadata=metadata,
        )
        if new_state is not None and new_state is not state and decision.get("action") == "queued":
            await repo.commit(new_state, f"executive_authority:{source}")
        return decision

    async def promote_next_initiative(
        self,
        state: Any,
        *,
        source: str = "mind_tick",
    ) -> Tuple[Any, Optional[Dict[str, Any]], Dict[str, Any]]:
        if state is None:
            return state, None, self._record("rejected", "state_missing", source=source)
        pause_reason = self._autonomy_pause_reason()
        if pause_reason:
            return state, None, self._record(
                "suppressed",
                f"autonomy_paused:{pause_reason}",
                source=source,
            )
        if getattr(state.cognition, "current_objective", None):
            return state, None, self._record(
                "duplicate",
                "objective_already_active",
                source=source,
                goal=str(state.cognition.current_objective or ""),
            )

        pending = list(getattr(state.cognition, "pending_initiatives", []) or [])
        if not pending:
            return state, None, self._record("rejected", "no_pending_initiatives", source=source)

        selected = None
        try:
            from core.agency.initiative_arbiter import get_initiative_arbiter

            selected = await get_initiative_arbiter().arbitrate(state)
        except Exception as exc:
            logger.debug("ExecutiveAuthority promotion fallback engaged: %s", exc)

        initiative = dict(selected.initiative) if selected else dict(pending[0])

        # ── Counterfactual deliberation ──────────────────────────────────
        # Before committing to this initiative, let the counterfactual engine
        # evaluate alternatives.  If it picks a different candidate from the
        # pending queue, swap to that one.  Entirely optional — if the engine
        # is unavailable the original selection goes through unchanged.
        cf_candidates = None
        try:
            cf_engine = ServiceContainer.get("counterfactual_engine", default=None)
            if cf_engine is not None:
                affect_ctx = self._gather_affect_context()
                action_dict = {
                    "type": str(initiative.get("type") or initiative.get("source") or "autonomous_thought"),
                    "description": str(initiative.get("goal", "")),
                    "params": dict(initiative.get("metadata", {}) or {}),
                }
                cf_best = await cf_engine.evaluate_autonomous_action(action_dict, affect_ctx)
                if cf_best is not None:
                    cf_candidates = True  # flag for outcome recording
                    # Did the engine recommend a different action type?
                    proposed_type = action_dict["type"].lower()
                    recommended_type = (cf_best.action_type or "").lower()
                    if recommended_type and recommended_type != proposed_type:
                        # Search pending initiatives for one matching the
                        # recommended type — if found, swap to it.
                        better_match = None
                        for item in pending:
                            if not isinstance(item, dict):
                                continue
                            item_type = str(
                                item.get("type") or item.get("source") or ""
                            ).lower()
                            if item_type == recommended_type:
                                better_match = item
                                break
                        if better_match is not None:
                            logger.debug(
                                "Counterfactual deliberation: swapping %s -> %s "
                                "(score=%.3f)",
                                proposed_type,
                                recommended_type,
                                cf_best.score,
                            )
                            initiative = dict(better_match)
                        else:
                            logger.debug(
                                "Counterfactual recommended %s but no matching "
                                "pending initiative — proceeding with %s",
                                recommended_type,
                                proposed_type,
                            )
                    else:
                        logger.debug(
                            "Counterfactual confirmed proposed action %s "
                            "(score=%.3f)",
                            proposed_type,
                            cf_best.score,
                        )
                    # Stash the candidate on the initiative metadata so the
                    # completion path can record the outcome.
                    initiative.setdefault("metadata", {})
                    initiative["metadata"]["_cf_candidate_type"] = cf_best.action_type
                    initiative["metadata"]["_cf_candidate_score"] = round(cf_best.score, 4)
                    initiative["metadata"]["_cf_simulated_gain"] = round(
                        cf_best.simulated_hedonic_gain, 4
                    )
        except Exception as exc:
            logger.debug("Counterfactual deliberation skipped: %s", exc)
        # ── End counterfactual deliberation ───────────────────────────────

        goal = _normalize_text(initiative.get("goal", ""))

        # ── Agency Comparator: emit efference copy (prediction) ──────────
        try:
            from core.consciousness.agency_comparator import get_agency_comparator
            affect_ctx = self._gather_affect_context()
            predicted_state = {
                "goal_completed": 1.0,
                "valence_delta": 0.05,  # Expect mild positive mood shift
                "hedonic_gain": float(initiative.get("urgency", 0.5) or 0.5) * 0.1,
                "closure_change": 0.1,
                "user_engagement": 0.5,
                "baseline_valence": float(affect_ctx.get("valence", 0.0)),
                "baseline_hedonic": float(affect_ctx.get("hedonic_score", 0.5)),
            }
            get_agency_comparator().emit_efference(
                layer="executive_authority",
                predicted_state=predicted_state,
                action_goal=goal,
                action_source=str(initiative.get("source") or source),
            )
        except Exception as exc:
            logger.debug("AgencyComparator efference emission skipped: %s", exc)
        # ── End agency comparator emission ────────────────────────────────

        new_state = state.derive("executive_authority_promote", origin=source)
        remaining = []
        removed = False
        for item in list(getattr(new_state.cognition, "pending_initiatives", []) or []):
            if not removed and isinstance(item, dict) and _normalize_text(item.get("goal", "")) == goal:
                removed = True
                continue
            remaining.append(item)
        new_state.cognition.pending_initiatives = remaining
        new_state.cognition.current_objective = goal or initiative.get("goal")
        new_state.cognition.current_origin = str(initiative.get("source") or initiative.get("type") or source)
        binding = await self._bind_objective_to_goal_engine(initiative)
        modifiers = dict(getattr(new_state.cognition, "modifiers", {}) or {})
        objective_binding = {
            "goal": goal or str(initiative.get("goal", "") or ""),
            "source": str(initiative.get("source") or source or ""),
            "kind": str(initiative.get("type") or ""),
            "triggered_by": str(initiative.get("triggered_by") or source or ""),
            "goal_id": str(binding.get("goal_id", "") or ""),
            "horizon": str(binding.get("horizon", "") or ""),
            "quick_win": bool(binding.get("quick_win", False)),
            "attention_policy": str(binding.get("attention_policy", "") or ""),
            "priority": float(binding.get("priority", initiative.get("urgency", 0.5)) or 0.5),
            "metadata": dict(initiative.get("metadata", {}) or {}),
            "promoted_at": time.time(),
        }
        modifiers["current_objective_binding"] = objective_binding
        modifiers["current_goal_id"] = objective_binding["goal_id"]
        new_state.cognition.modifiers = modifiers

        reason = "initiative_promoted"
        if selected and getattr(selected, "rationale", None):
            reason = f"initiative_promoted:{selected.rationale[:180]}"

        self._record_constitutional_decision(
            kind="initiative",
            source=source,
            summary=goal or "initiative",
            outcome="approved",
            reason=reason,
            target="current_objective",
            payload=initiative,
            state=new_state,
        )
        return new_state, initiative, self._record(
            "promoted",
            reason,
            source=source,
            goal=goal or str(initiative.get("goal", "")),
            queued_initiatives=len(remaining),
            target="current_objective",
        )

    async def complete_current_objective(
        self,
        state: Any,
        reason: str,
        *,
        source: str = "mind_tick",
    ) -> Tuple[Any, Dict[str, Any]]:
        if state is None:
            return state, self._record("rejected", "state_missing", source=source)
        objective = _normalize_text(getattr(state.cognition, "current_objective", ""))
        if not objective:
            return state, self._record("rejected", "no_current_objective", source=source)
        modifiers = dict(getattr(state.cognition, "modifiers", {}) or {})
        binding = dict(modifiers.get("current_objective_binding", {}) or {})
        binding_goal_id = str(binding.get("goal_id", "") or "")
        binding_meta = dict(binding.get("metadata", {}) or {})
        binding_horizon = str(binding.get("horizon", "") or "").strip().lower()
        binding_attention = str(binding.get("attention_policy", "") or "").strip().lower()
        should_hold = False
        goal_engine = None
        bound_goal = None
        try:
            goal_engine = ServiceContainer.get("goal_engine", default=None)
            if goal_engine and binding_goal_id and hasattr(goal_engine, "get_goal"):
                bound_goal = await asyncio.to_thread(goal_engine.get_goal, binding_goal_id)
        except Exception as exc:
            logger.debug("ExecutiveAuthority objective binding probe failed: %s", exc)
            bound_goal = None

        if (
            reason == "tick_cycle_complete"
            and binding_goal_id
            and binding_horizon in {"long_term", "short_term"}
            and binding_attention == "sustained"
        ):
            status = str((bound_goal or {}).get("status", "") or "").strip().lower()
            should_hold = status in {"queued", "in_progress", "blocked", "paused"}

        if should_hold:
            return state, self._record(
                "held",
                "objective_persistence_active",
                source=source,
                goal=objective,
                queued_initiatives=len(list(getattr(state.cognition, "pending_initiatives", []) or [])),
                target="current_objective",
            )

        new_state = state.derive("executive_authority_complete", origin=source)
        pending = [
            item
            for item in list(getattr(new_state.cognition, "pending_initiatives", []) or [])
            if not (isinstance(item, dict) and _normalize_text(item.get("goal", "")) == objective)
        ]
        new_state.cognition.pending_initiatives = pending
        new_state.cognition.current_objective = None
        new_state.cognition.current_origin = None
        next_modifiers = dict(getattr(new_state.cognition, "modifiers", {}) or {})
        next_modifiers.pop("current_objective_binding", None)
        next_modifiers.pop("current_goal_id", None)
        new_state.cognition.modifiers = next_modifiers

        # ── Counterfactual outcome recording ─────────────────────────────
        # Close the learning loop: measure the actual hedonic change from
        # when the initiative was promoted to now, and feed it back to the
        # counterfactual engine so it can compute regret/relief.
        try:
            cf_engine = ServiceContainer.get("counterfactual_engine", default=None)
            if cf_engine is not None:
                # Reconstruct a lightweight ActionCandidate for the completed action.
                from core.consciousness.counterfactual_engine import ActionCandidate

                # Pull stashed metadata from the initiative that was active.
                # We stored _cf_candidate_type / _cf_simulated_gain during promotion.
                meta = dict(binding_meta)
                if not meta:
                    for item in list(getattr(state.cognition, "pending_initiatives", []) or []):
                        if isinstance(item, dict) and _normalize_text(item.get("goal", "")) == objective:
                            meta = dict(item.get("metadata", {}) or {})
                            break

                cf_type = str(meta.get("_cf_candidate_type", "autonomous_thought"))
                cf_gain = float(meta.get("_cf_simulated_gain", 0.05))

                # Measure current hedonic score to compute actual delta.
                affect_ctx = self._gather_affect_context()
                current_hedonic = affect_ctx.get("hedonic_score", 0.5)
                # The baseline is 0.5 (neutral) since we don't persist the
                # pre-promotion hedonic score.  This is a reasonable default
                # and the regret/relief signal is still directionally correct.
                actual_hedonic_change = current_hedonic - 0.5

                selected_candidate = ActionCandidate(
                    action_type=cf_type,
                    action_params=meta,
                    description=objective[:200],
                    simulated_hedonic_gain=cf_gain,
                    heartstone_alignment=float(meta.get("_cf_candidate_score", 0.5)),
                    expected_outcome=f"Completed: {reason}",
                    score=float(meta.get("_cf_candidate_score", 0.5)),
                    selected=True,
                )
                cf_engine.record_outcome(selected_candidate, actual_hedonic_change)
                logger.debug(
                    "Counterfactual outcome recorded for '%s': actual_delta=%.3f",
                    objective[:60],
                    actual_hedonic_change,
                )
        except Exception as exc:
            logger.debug("Counterfactual outcome recording skipped: %s", exc)
        # ── End counterfactual outcome recording ─────────────────────────

        # ── Agency Comparator: compare prediction to actual outcome ──────
        try:
            from core.consciousness.agency_comparator import get_agency_comparator
            affect_ctx = self._gather_affect_context()
            actual_hedonic = float(affect_ctx.get("hedonic_score", 0.5))
            actual_valence = float(affect_ctx.get("valence", 0.0))
            normalized_reason = str(reason or "").lower()
            goal_completed = 0.0 if any(
                t in normalized_reason for t in ("fail", "error", "reject", "abort")
            ) else 1.0

            actual_state = {
                "goal_completed": goal_completed,
                "valence_delta": actual_valence - 0.0,  # Delta from neutral
                "hedonic_gain": actual_hedonic - 0.5,    # Delta from neutral
                "closure_change": float(affect_ctx.get("closure_score", 0.0)),
                "user_engagement": 0.5,  # Default; ideally from interaction signals
                "baseline_valence": actual_valence,
                "baseline_hedonic": actual_hedonic,
            }
            trace = get_agency_comparator().compare_and_attribute(
                efference=None,  # Auto-lookup by goal
                actual_state=actual_state,
                action_goal=objective,
            )
            logger.debug(
                "AgencyComparator: objective '%s' -> %s (agency=%.2f)",
                objective[:50], trace.attribution_label, trace.self_caused_fraction,
            )
        except Exception as exc:
            logger.debug("AgencyComparator comparison skipped: %s", exc)
        # ── End agency comparator comparison ─────────────────────────────

        try:
            if goal_engine and binding_goal_id and hasattr(goal_engine, "update_goal_status"):
                goal_status = "completed"
                normalized_reason = str(reason or "").lower()
                if any(token in normalized_reason for token in ("fail", "error", "reject", "abort")):
                    goal_status = "failed"
                await goal_engine.update_goal_status(
                    binding_goal_id,
                    status=goal_status,
                    summary=str(reason or ""),
                )
        except Exception as exc:
            logger.debug("ExecutiveAuthority goal settlement skipped: %s", exc)

        self._record_constitutional_decision(
            kind="initiative",
            source=source,
            summary=objective,
            outcome="recorded",
            reason=f"objective_completed:{reason}",
            target="current_objective",
            state=new_state,
        )
        return new_state, self._record(
            "completed",
            reason,
            source=source,
            goal=objective,
            queued_initiatives=len(pending),
            target="current_objective",
        )

    async def suppress_initiatives(
        self,
        state: Any,
        predicate: Callable[[Dict[str, Any]], bool],
        reason: str,
        *,
        source: str = "executive_authority",
    ) -> Tuple[Any, Dict[str, Any]]:
        if state is None:
            return state, self._record("rejected", "state_missing", source=source)

        pending = list(getattr(state.cognition, "pending_initiatives", []) or [])
        kept: list[dict] = []
        removed: list[dict] = []
        for item in pending:
            if isinstance(item, dict) and predicate(item):
                removed.append(item)
            else:
                kept.append(item)

        if not removed:
            return state, self._record(
                "duplicate",
                "no_matching_initiatives",
                source=source,
                queued_initiatives=len(kept),
            )

        new_state = state.derive("executive_authority_suppress", origin=source)
        new_state.cognition.pending_initiatives = kept
        self._record_constitutional_decision(
            kind="initiative",
            source=source,
            summary=f"suppressed:{len(removed)}",
            outcome="recorded",
            reason=reason,
            target="pending_initiatives",
            payload={"removed": removed[:10]},
            state=new_state,
        )
        return new_state, self._record(
            "suppressed",
            reason,
            source=source,
            queued_initiatives=len(kept),
            target="pending_initiatives",
        )

    def record_user_objective(
        self,
        state: Any,
        objective: str,
        *,
        source: str,
        mode: str = "",
    ) -> None:
        self.record_objective_binding(
            state,
            objective,
            source=source,
            mode=mode,
            reason=f"user_objective_bound:{mode or 'reactive'}",
        )

    def record_objective_binding(
        self,
        state: Any,
        objective: str,
        *,
        source: str,
        mode: str = "",
        reason: str = "objective_bound",
    ) -> None:
        goal = _normalize_text(objective)
        if not goal:
            return
        self._record_constitutional_decision(
            kind="initiative",
            source=source,
            summary=goal,
            outcome="approved",
            reason=reason,
            target="current_objective",
            payload={"mode": mode},
            state=state,
        )

    async def release_expression(
        self,
        content: str,
        *,
        source: str,
        urgency: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        target: str = "primary",
    ) -> Dict[str, Any]:
        content = _normalize_text(content)
        if len(content) < 4:
            return self._record("suppressed", "empty_content", source=source, content=content, target="discarded")

        urgency = _clamp01(urgency)
        fingerprint = hashlib.sha1(content.lower().encode("utf-8")).hexdigest()
        now = time.time()
        self._prune_fingerprints(now)
        if fingerprint in self._recent_fingerprints:
            return self._record("suppressed", "duplicate_expression", source=source, content=content, target="discarded")

        orch = self._get_orchestrator()
        gate = getattr(orch, "output_gate", None) if orch else None
        if gate is None:
            gate = ServiceContainer.get("output_gate", default=None)
        if gate is None:
            return self._record("suppressed", "output_gate_missing", source=source, content=content, target="discarded")

        closure = self._get_closure_status()
        user_idle = self._get_user_idle_seconds(orch, now)
        is_processing = bool(getattr(getattr(orch, "status", None), "is_processing", False)) if orch else False
        is_busy = bool(getattr(orch, "is_busy", False)) if orch else False

        dominant_need = str(closure.get("dominant_need") or "")
        need_pressure = float(closure.get("need_pressure", 0.0) or 0.0)
        closure_score = float(closure.get("closure_score", 0.0) or 0.0)
        vitality = float(closure.get("vitality", 1.0) or 1.0)
        merged_meta = dict(metadata or {})
        visible_presence = self._is_visible_presence(merged_meta)

        release_target = target
        reason = "approved"
        if release_target == "primary":
            if (
                dominant_need in {"stability", "integrity"}
                and need_pressure >= (0.88 if visible_presence else 0.72)
                and urgency < (0.78 if visible_presence else 0.9)
            ):
                release_target = "secondary"
                reason = "runtime_guard"
            elif closure_score and closure_score < (0.18 if visible_presence else 0.28) and urgency < (0.72 if visible_presence else 0.85):
                release_target = "secondary"
                reason = "closure_low"
            elif vitality < (0.32 if visible_presence else 0.45) and urgency < (0.78 if visible_presence else 0.9):
                release_target = "secondary"
                reason = "vitality_low"
            elif user_idle < (self._VISIBLE_PRESENCE_IDLE_WINDOW_S if visible_presence else self._PRIMARY_SILENCE_WINDOW_S) and urgency < (0.58 if visible_presence else 0.85):
                release_target = "secondary"
                reason = "user_recently_active"
            elif (is_processing or is_busy) and urgency < (0.68 if visible_presence else 0.8):
                release_target = "secondary"
                reason = "processing_guard"

        merged_meta.setdefault("autonomous", True)
        merged_meta.setdefault("executive_authority", True)
        merged_meta.setdefault("visible_presence", visible_presence)
        merged_meta.setdefault("authority_reason", reason)
        merged_meta.setdefault("authority_urgency", round(urgency, 4))

        if release_target == "primary":
            merged_meta.setdefault("spontaneous", True)
            merged_meta.setdefault("force_user", True)
            await gate.emit(content, origin=source, target="primary", metadata=merged_meta)
            self._recent_fingerprints[fingerprint] = now
            self._record_constitutional_decision(
                kind="expression",
                source=source,
                summary=content[:220],
                outcome="released",
                reason=reason,
                target="primary",
                payload={"urgency": urgency, "metadata": merged_meta},
            )
            return self._record("released", reason, source=source, content=content, target="primary")

        merged_meta.setdefault("spontaneous", False)
        await gate.emit(content, origin=source, target="secondary", metadata=merged_meta)
        self._emit_thought_card(source, content, reason)
        self._recent_fingerprints[fingerprint] = now
        self._record_constitutional_decision(
            kind="expression",
            source=source,
            summary=content[:220],
            outcome="released",
            reason=reason,
            target="secondary",
            payload={"urgency": urgency, "metadata": merged_meta},
        )
        return self._record("released", reason, source=source, content=content, target="secondary")

    def _record(
        self,
        action: str,
        reason: str,
        *,
        source: str,
        goal: str = "",
        content: str = "",
        target: str = "",
        queued_initiatives: Optional[int] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        if action == "released" and target == "primary":
            self._snapshot.primary_releases += 1
        elif action == "released" and target == "secondary":
            self._snapshot.secondary_releases += 1
        elif action == "suppressed":
            self._snapshot.suppressed += 1

        if queued_initiatives is not None:
            self._snapshot.queued_initiatives = int(queued_initiatives)

        self._snapshot.last_action = action
        self._snapshot.last_reason = reason
        self._snapshot.last_source = source
        self._snapshot.last_goal = goal[:220]
        self._snapshot.last_content = content[:220]
        self._snapshot.last_target = target
        self._snapshot.last_decision_at = now

        try:
            from core.unified_action_log import get_action_log

            summary = goal[:120] if goal else content[:120]
            outcome = reason if not target else f"{reason}:{target}"
            get_action_log().record(
                summary or action,
                f"ExecutiveAuthority.{source}",
                "gen3_constitutional",
                action,
                outcome,
            )
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        return {
            "ok": action in {"queued", "released", "promoted", "completed"},
            "action": action,
            "reason": reason,
            "target": target,
            "source": source,
        }

    def _emit_thought_card(self, source: str, content: str, reason: str) -> None:
        try:
            from core.thought_stream import get_emitter

            title = "Executive Hold"
            if source:
                title = f"Executive Hold · {source}"
            get_emitter().emit(title, content, level="info", category=f"Authority ({reason})")
        except Exception as exc:
            logger.debug("ExecutiveAuthority thought-card emit skipped: %s", exc)

    def _get_orchestrator(self) -> Any:
        return self.orchestrator or service_access.resolve_orchestrator(default=None)

    def _get_state_repository(self) -> Any:
        return service_access.resolve_state_repository(self._get_orchestrator(), default=None)

    def _get_closure_status(self) -> Dict[str, Any]:
        try:
            closure = ServiceContainer.get("executive_closure", default=None)
            if closure and hasattr(closure, "get_status"):
                return dict(closure.get_status() or {})
        except Exception as exc:
            logger.debug("ExecutiveAuthority: closure lookup failed: %s", exc)
        return {}

    def _gather_affect_context(self) -> Dict[str, Any]:
        """Collect current hedonic/affect state for counterfactual deliberation."""
        ctx: Dict[str, Any] = {}
        try:
            affect = ServiceContainer.get("affect_engine", default=None)
            if affect and hasattr(affect, "get_state"):
                astate = affect.get_state()
                if isinstance(astate, dict):
                    ctx["valence"] = float(astate.get("valence", 0.0))
                    ctx["curiosity"] = float(astate.get("curiosity", 0.5))
                elif astate is not None:
                    ctx["valence"] = float(getattr(astate, "valence", 0.0))
                    ctx["curiosity"] = float(getattr(astate, "curiosity", 0.5))
        except Exception as exc:
            logger.debug("ExecutiveAuthority: affect lookup failed: %s", exc)
        try:
            hedonic = ServiceContainer.get("hedonic_gradient", default=None)
            if hedonic and hasattr(hedonic, "get_status"):
                hstatus = hedonic.get_status()
                if isinstance(hstatus, dict):
                    ctx["hedonic_score"] = float(hstatus.get("hedonic_score", 0.5))
                elif hstatus is not None:
                    ctx["hedonic_score"] = float(getattr(hstatus, "hedonic_score", 0.5))
        except Exception as exc:
            logger.debug("ExecutiveAuthority: hedonic lookup failed: %s", exc)
        closure = self._get_closure_status()
        if closure:
            ctx["closure_score"] = float(closure.get("closure_score", 0.0))
            ctx["vitality"] = float(closure.get("vitality", 1.0))
        ctx.setdefault("hedonic_score", 0.5)
        ctx.setdefault("valence", 0.0)
        ctx.setdefault("curiosity", 0.5)
        return ctx

    def _get_user_idle_seconds(self, orch: Any, now: float) -> float:
        if orch is None:
            return 10_000.0
        return max(0.0, now - float(getattr(orch, "_last_user_interaction_time", 0.0) or 0.0))

    def _prune_fingerprints(self, now: float) -> None:
        stale = [
            fingerprint
            for fingerprint, seen_at in self._recent_fingerprints.items()
            if (now - seen_at) > self._DEDUP_WINDOW_S
        ]
        for fingerprint in stale:
            self._recent_fingerprints.pop(fingerprint, None)

    def _record_constitutional_decision(
        self,
        *,
        kind: str,
        source: str,
        summary: str,
        outcome: str,
        reason: str,
        target: str = "",
        payload: Optional[Dict[str, Any]] = None,
        state: Any = None,
    ) -> None:
        try:
            from core.constitution import ProposalKind, get_constitutional_core

            get_constitutional_core(self.orchestrator).record_external_decision(
                kind=ProposalKind(kind),
                source=source,
                summary=summary,
                outcome=outcome,
                reason=reason,
                target=target,
                payload=payload,
                state=state,
            )
        except Exception as exc:
            logger.debug("ExecutiveAuthority constitutional audit skipped: %s", exc)


def get_executive_authority(orchestrator: Any = None) -> ExecutiveAuthority:
    authority = ServiceContainer.get("executive_authority", default=None)
    if authority and isinstance(authority, ExecutiveAuthority):
        authority.bind(orchestrator)
        return authority

    authority = ExecutiveAuthority(orchestrator=orchestrator)
    try:
        ServiceContainer.register_instance("executive_authority", authority, required=False)
    except Exception as exc:
        logger.debug("ExecutiveAuthority registration skipped: %s", exc)
    return authority
