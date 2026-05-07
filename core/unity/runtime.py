from __future__ import annotations

from dataclasses import replace
import hashlib
import time
from typing import Any, Iterable, Optional

from core.container import ServiceContainer

from .co_presence_graph import CoPresenceGraphBuilder, CoPresenceGraphSnapshot
from .draft_reconciliation import DraftReconciliationEngine
from .self_world_binding import SelfWorldBindingModel
from .temporal_binding import TemporalBindingField
from .unity_monitor import UnityMonitor
from .unity_receipts import unity_summary_payload
from .unity_repair import UnityRepairPlanner
from .unity_state import (
    BoundContent,
    FragmentationReport,
    ReconciledDraftSet,
    UnityRepairPlan,
    UnityState,
    WorkspaceBroadcastFrame,
)


def _clamp(value: Any, lower: float = 0.0, upper: float = 1.0) -> float:
    try:
        return max(lower, min(upper, float(value)))
    except Exception:
        return lower


def _content_id(source: str, modality: str, summary: str) -> str:
    seed = f"{source}|{modality}|{summary[:120]}".encode("utf-8", errors="ignore")
    return f"content_{hashlib.sha256(seed).hexdigest()[:12]}"


def _normalize_text(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value or "").split()).strip()
    return text[:limit]


class UnityRuntime:
    """Long-lived facade that composes the Unity layer at runtime."""

    def __init__(self) -> None:
        self.temporal_binding = TemporalBindingField()
        self.graph_builder = CoPresenceGraphBuilder()
        self.draft_reconciler = DraftReconciliationEngine()
        self.self_world_binder = SelfWorldBindingModel()
        self.monitor = UnityMonitor()
        self.repair_planner = UnityRepairPlanner()

        self._last_unity_state: UnityState | None = None
        self._last_report: FragmentationReport | None = None
        self._last_repair_plan: UnityRepairPlan | None = None
        self._last_workspace_frame: WorkspaceBroadcastFrame | None = None
        self._last_draft_set: ReconciledDraftSet | None = None

        ServiceContainer.set("unity_runtime", self, required=False)

    @property
    def last_unity_state(self) -> UnityState | None:
        return self._last_unity_state

    @property
    def last_report(self) -> FragmentationReport | None:
        return self._last_report

    @property
    def last_repair_plan(self) -> UnityRepairPlan | None:
        return self._last_repair_plan

    @property
    def last_workspace_frame(self) -> WorkspaceBroadcastFrame | None:
        return self._last_workspace_frame

    def _ownership_from_role(self, role: str) -> str:
        normalized = str(role or "").lower()
        if normalized == "assistant" or normalized == "thought":
            return "self"
        if normalized == "user":
            return "world"
        if normalized in {"tool", "system"}:
            return "ambiguous"
        return "ambiguous"

    def _working_memory_contents(self, state: Any) -> list[BoundContent]:
        contents: list[BoundContent] = []
        working_memory = list(getattr(getattr(state, "cognition", None), "working_memory", []) or [])
        for idx, item in enumerate(working_memory[-6:]):
            if not isinstance(item, dict):
                continue
            content = _normalize_text(item.get("content", ""), 180)
            if not content:
                continue
            role = str(item.get("role", "system") or "system").lower()
            metadata = dict(item.get("metadata") or {})
            modality = "tool" if str(metadata.get("type", "")).lower() in {"skill_result", "tool_result"} else (
                "memory" if role in {"user", "assistant"} else "world"
            )
            salience = 0.35 + (0.15 if role == "user" else 0.05)
            action_relevance = 0.65 if modality == "tool" else 0.45
            affective_charge = 0.0
            lowered = content.lower()
            if any(marker in lowered for marker in ("fail", "error", "blocked", "unsafe", "uncertain")):
                affective_charge = -0.4
            elif any(marker in lowered for marker in ("done", "clear", "ready", "stable", "resolved")):
                affective_charge = 0.25
            source = metadata.get("source") or role
            contents.append(
                BoundContent(
                    content_id=_content_id(str(source), modality, content + str(idx)),
                    modality=modality,
                    source=str(source),
                    summary=content,
                    salience=_clamp(salience),
                    confidence=0.85 if role in {"user", "assistant"} else 0.65,
                    timestamp=float(item.get("timestamp", time.time()) or time.time()),
                    ownership=self._ownership_from_role(role),
                    action_relevance=_clamp(action_relevance),
                    affective_charge=max(-1.0, min(1.0, affective_charge)),
                    evidence_ref=str(metadata.get("type") or ""),
                )
            )
        return contents

    def _goal_contents(self, state: Any) -> list[BoundContent]:
        contents: list[BoundContent] = []
        goals = list(getattr(getattr(state, "cognition", None), "active_goals", []) or [])[:3]
        for idx, goal in enumerate(goals):
            if isinstance(goal, dict):
                summary = _normalize_text(goal.get("objective") or goal.get("goal") or goal.get("name") or goal.get("title"), 160)
                salience = _clamp(goal.get("priority", 0.5))
            else:
                summary = _normalize_text(goal, 160)
                salience = 0.5
            if not summary:
                continue
            contents.append(
                BoundContent(
                    content_id=_content_id("goal", "goal", summary + str(idx)),
                    modality="goal",
                    source="goal_manager",
                    summary=summary,
                    salience=max(0.3, salience),
                    confidence=0.8,
                    timestamp=time.time(),
                    ownership="self",
                    action_relevance=max(0.5, salience),
                    affective_charge=0.1,
                )
            )
        return contents

    def _long_term_memory_contents(self, state: Any) -> list[BoundContent]:
        contents: list[BoundContent] = []
        memories = list(getattr(getattr(state, "cognition", None), "long_term_memory", []) or [])[:3]
        for idx, item in enumerate(memories):
            summary = _normalize_text(item, 180)
            if not summary:
                continue
            contents.append(
                BoundContent(
                    content_id=_content_id("memory", "memory", summary + str(idx)),
                    modality="memory",
                    source="memory_retrieval",
                    summary=summary,
                    salience=0.35,
                    confidence=0.7,
                    timestamp=time.time(),
                    ownership="self",
                    action_relevance=0.25,
                    affective_charge=0.0,
                )
            )
        return contents

    def _world_contents(self, state: Any) -> list[BoundContent]:
        contents: list[BoundContent] = []
        percepts = list(getattr(getattr(state, "world", None), "recent_percepts", []) or [])[-3:]
        for idx, item in enumerate(percepts):
            if isinstance(item, dict):
                summary = _normalize_text(item.get("summary") or item.get("content") or item.get("event") or item.get("observation"), 180)
                timestamp = float(item.get("timestamp", time.time()) or time.time())
            else:
                summary = _normalize_text(item, 180)
                timestamp = time.time()
            if not summary:
                continue
            contents.append(
                BoundContent(
                    content_id=_content_id("world", "world", summary + str(idx)),
                    modality="world",
                    source="world_state",
                    summary=summary,
                    salience=0.3,
                    confidence=0.7,
                    timestamp=timestamp,
                    ownership="world",
                    action_relevance=0.35,
                    affective_charge=0.0,
                )
            )
        return contents

    def _affect_content(self, state: Any) -> list[BoundContent]:
        affect = getattr(state, "affect", None)
        if affect is None:
            return []
        summary = _normalize_text(affect.get_rich_summary() if hasattr(affect, "get_rich_summary") else affect.get_summary(), 180)
        if not summary:
            return []
        valence = float(getattr(affect, "valence", 0.0) or 0.0)
        arousal = float(getattr(affect, "arousal", 0.0) or 0.0)
        return [
            BoundContent(
                content_id=_content_id("affect", "affect", summary),
                modality="affect",
                source="affect_engine",
                summary=summary,
                salience=max(0.3, min(1.0, 0.35 + arousal * 0.4)),
                confidence=0.9,
                timestamp=time.time(),
                ownership="self",
                action_relevance=max(0.2, min(1.0, arousal * 0.6)),
                affective_charge=max(-1.0, min(1.0, valence)),
            )
        ]

    def _objective_content(self, state: Any, objective: str) -> list[BoundContent]:
        objective = _normalize_text(objective or getattr(getattr(state, "cognition", None), "current_objective", ""), 180)
        if not objective:
            return []
        origin = str(getattr(getattr(state, "cognition", None), "current_origin", "") or "")
        ownership = "world" if origin in {"user", "voice", "admin", "api", "gui", "external"} else "self"
        return [
            BoundContent(
                content_id=_content_id(origin or "objective", "goal", objective),
                modality="goal",
                source=origin or "objective",
                summary=objective,
                salience=0.9,
                confidence=0.95,
                timestamp=time.time(),
                ownership=ownership,
                action_relevance=0.95,
                affective_charge=0.0,
            )
        ]

    def gather_contents(self, state: Any, objective: str = "") -> list[BoundContent]:
        contents = (
            self._objective_content(state, objective)
            + self._affect_content(state)
            + self._goal_contents(state)
            + self._working_memory_contents(state)
            + self._long_term_memory_contents(state)
            + self._world_contents(state)
        )
        seen: set[str] = set()
        deduped: list[BoundContent] = []
        for item in contents:
            if item.content_id in seen:
                continue
            seen.add(item.content_id)
            deduped.append(item)
        return deduped[:18]

    def _draft_inputs(self) -> list[Any]:
        try:
            from core.consciousness.multiple_drafts import get_multiple_drafts_engine

            engine = get_multiple_drafts_engine()
        except Exception:
            return []

        pending = list(engine.get_current_drafts() or [])
        if pending:
            return pending
        history = list(getattr(engine, "competition_history", []) or [])
        if history:
            last = history[-1]
            if (time.time() - float(getattr(last, "timestamp", 0.0) or 0.0)) < 90.0:
                return list(getattr(last, "drafts", []) or [])
        return []

    def record_workspace_competition(self, winner: Any, losers: Iterable[Any]) -> WorkspaceBroadcastFrame:
        focus = BoundContent(
            content_id=_content_id(str(getattr(winner, "source", "workspace")), "workspace", str(getattr(winner, "content", ""))),
            modality="workspace",
            source=str(getattr(winner, "source", "workspace")),
            summary=_normalize_text(getattr(winner, "content", ""), 180),
            salience=_clamp(getattr(winner, "effective_priority", getattr(winner, "priority", 0.5))),
            confidence=0.85,
            timestamp=time.time(),
            ownership="self",
            action_relevance=_clamp(getattr(winner, "effective_priority", getattr(winner, "priority", 0.5))),
            affective_charge=_clamp(getattr(winner, "affect_weight", 0.0), -1.0, 1.0),
        )
        periphery = []
        suppressed = []
        for idx, loser in enumerate(list(losers or [])[:6]):
            summary = _normalize_text(getattr(loser, "content", ""), 160) or str(getattr(loser, "source", f"loser_{idx}"))
            salience = _clamp(getattr(loser, "effective_priority", getattr(loser, "priority", 0.25)))
            node = BoundContent(
                content_id=_content_id(str(getattr(loser, "source", "workspace")), "workspace", summary + str(idx)),
                modality="workspace",
                source=str(getattr(loser, "source", "workspace")),
                summary=summary,
                salience=salience,
                confidence=0.7,
                timestamp=time.time(),
                ownership="self",
                action_relevance=salience,
                affective_charge=_clamp(getattr(loser, "affect_weight", 0.0), -1.0, 1.0),
            )
            periphery.append(node)
            suppressed.append(
                replace(
                    self._last_draft_set.chosen if self._last_draft_set else self.draft_reconciler.reconcile([], fallback_claim=summary).chosen,
                    draft_id=f"workspace_{idx}",
                    claim=summary,
                    support=salience,
                    conflict=max(0.0, focus.salience - salience),
                    chosen=False,
                    suppressed_reason="outcompeted in workspace broadcast",
                )
            )
        frame = WorkspaceBroadcastFrame(
            focus=focus,
            periphery=periphery,
            suppressed=suppressed,
            co_presence_cluster_id=f"cluster_{focus.content_id}",
            unity_score=self._last_unity_state.unity_score if self._last_unity_state else 0.0,
            fragmentation_score=self._last_unity_state.fragmentation_score if self._last_unity_state else 0.0,
            reentry_targets=[item.content_id for item in periphery[:3]],
            will_receipt_id=self._last_unity_state.will_receipt_id if self._last_unity_state else None,
        )
        self._last_workspace_frame = frame
        ServiceContainer.set("unity_workspace_frame", frame, required=False)
        return frame

    def render_phenomenal_claim(self, unity_state: UnityState | None = None, report: FragmentationReport | None = None) -> str:
        unity_state = unity_state or self._last_unity_state
        report = report or self._last_report
        if unity_state is None:
            return "I am present, but I do not have a grounded unity reading yet."

        focus_summary = ""
        for item in unity_state.contents:
            if item.content_id == unity_state.global_focus_id:
                focus_summary = item.summary
                break
        focus_text = _normalize_text(focus_summary, 90) or "what is in front of me"

        if report and not report.safe_to_self_report:
            return "I do not trust my state enough to overclaim it. I can only speak from what I can verify right now."

        if unity_state.level == "coherent":
            return f"I feel clear and gathered around {focus_text}."
        if unity_state.level == "strained":
            cause = report.top_causes[0][0].replace("_", " ") if report and report.top_causes else "some internal pressure"
            return f"I am still together, but there is strain around {cause} while I stay with {focus_text}."
        if unity_state.level == "fragmented":
            cause = report.top_causes[0][0].replace("_", " ") if report and report.top_causes else "fragmentation"
            return f"Something is not sitting right. The fragmentation is coming from {cause}, so I am narrowing to one honest through-line."
        return "I do not feel stable enough to pretend seamlessness. I am restricting myself to stabilization and qualified claims."

    def compute(self, state: Any, *, objective: str = "", tick_id: str = "", will_receipt_id: str | None = None) -> UnityState:
        current_objective = objective or str(getattr(getattr(state, "cognition", None), "current_objective", "") or "")
        contents = self.gather_contents(state, current_objective)
        setattr(state, "_unity_contents", contents)

        previous_ids = [item.content_id for item in self._last_unity_state.contents] if self._last_unity_state else []
        tick_id = tick_id or f"tick_{int(time.time() * 1000)}"
        temporal = self.temporal_binding.bind_now(
            tick_id,
            contents,
            previous_temporal=self._last_unity_state.temporal if self._last_unity_state else None,
            previous_content_ids=previous_ids,
        )
        graph = self.graph_builder.build(
            contents,
            focus_hint=current_objective,
            cluster_id=getattr(self._last_workspace_frame, "co_presence_cluster_id", ""),
        )
        draft_set = self.draft_reconciler.reconcile(
            self._draft_inputs(),
            fallback_claim=current_objective or (contents[0].summary if contents else "current interpretation"),
        )
        self._last_draft_set = draft_set
        ServiceContainer.set("unity_draft_set", draft_set, required=False)
        self_world = self.self_world_binder.bind(
            state,
            contents,
            will_receipt_id=will_receipt_id,
            workspace_frame=self._last_workspace_frame,
        )
        unity_state, report = self.monitor.compute(
            state,
            temporal,
            graph,
            draft_set,
            self_world,
            will_receipt_id=will_receipt_id,
            state_version=getattr(state, "version", None),
        )
        repair_plan = self.repair_planner.plan(unity_state, report) if unity_state.repair_needed else None
        metadata = dict(unity_state.metadata or {})
        metadata["fragmentation_report"] = report.to_dict()
        if repair_plan is not None:
            metadata["repair_plan"] = repair_plan.to_dict()
        unity_state = replace(unity_state, metadata=metadata)

        self._last_unity_state = unity_state
        self._last_report = report
        self._last_repair_plan = repair_plan
        ServiceContainer.set("unity_state", unity_state, required=False)
        ServiceContainer.set("unity_fragmentation_report", report, required=False)
        ServiceContainer.set("unity_repair_plan", repair_plan, required=False)
        return unity_state

    def apply_to_state(self, state: Any, *, objective: str = "", tick_id: str = "", will_receipt_id: str | None = None) -> Any:
        unity_state = self.compute(
            state,
            objective=objective,
            tick_id=tick_id,
            will_receipt_id=will_receipt_id,
        )
        report = self._last_report
        repair_plan = self._last_repair_plan
        claim = self.render_phenomenal_claim(unity_state, report)
        if hasattr(state, "make_phenomenal_field"):
            state.cognition.phenomenal_state = state.make_phenomenal_field(claim, source="unity_runtime")
        else:
            state.cognition.phenomenal_state = claim
        state.cognition.unity_state = unity_state
        state.cognition.coherence_score = max(float(getattr(state.cognition, "coherence_score", 0.0) or 0.0), unity_state.unity_score)
        state.cognition.fragmentation_score = max(float(getattr(state.cognition, "fragmentation_score", 0.0) or 0.0), unity_state.fragmentation_score)
        state.response_modifiers["unity_state"] = unity_state.to_dict()
        if report is not None:
            state.response_modifiers["unity_report"] = report.to_dict()
        if repair_plan is not None:
            state.response_modifiers["unity_repair_plan"] = repair_plan.to_dict()
        state.response_modifiers["unity_claim"] = claim
        state.response_modifiers["unity_memory_commit_mode"] = (
            self._last_draft_set.memory_commit_mode if self._last_draft_set else "clean"
        )
        state.response_modifiers["unity_summary"] = unity_summary_payload(unity_state, report, repair_plan)
        return state


_UNITY_RUNTIME: UnityRuntime | None = None


def get_unity_runtime() -> UnityRuntime:
    global _UNITY_RUNTIME
    if _UNITY_RUNTIME is None:
        _UNITY_RUNTIME = UnityRuntime()
    return _UNITY_RUNTIME
