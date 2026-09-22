from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import record_degradation

from .co_presence_graph import CoPresenceGraphBuilder
from .draft_reconciliation import DraftReconciliationEngine
from .mind_moment import MindMomentBuilder
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

logger = logging.getLogger(__name__)


def _clamp(value: Any, lower: float = 0.0, upper: float = 1.0) -> float:
    try:
        return max(lower, min(upper, float(value)))
    except (TypeError, ValueError):
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
        self.mind_moment_builder = MindMomentBuilder()

        self._last_unity_state: UnityState | None = None
        self._last_report: FragmentationReport | None = None
        self._last_repair_plan: UnityRepairPlan | None = None
        self._last_workspace_frame: WorkspaceBroadcastFrame | None = None
        self._last_draft_set: ReconciledDraftSet | None = None
        self._last_mind_moment: Any | None = None

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

    @property
    def last_mind_moment(self) -> Any | None:
        return self._last_mind_moment

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

    def _felt_state_contents(self, state: Any) -> list[BoundContent]:
        """Bind the nociception substrate's felt damage/valence into the moment.

        This is what makes "how Aura is doing right now" part of the unified state rather than
        a side channel: real operational damage (memory/identity/tool/governance) and its
        improvement/deterioration trend enter the same workspace everything else competes in.
        """
        try:
            from core.affect.nociception import get_nociception_engine
            noci = get_nociception_engine()
            pressure = noci.nociceptive_pressure()
            if pressure < 0.05:
                return []  # nothing hurting → nothing to bind
            valence = noci.grounded_valence()
            worst = noci.worst_channel()
            where = f" ({worst[0]})" if worst else ""
            summary = _normalize_text(
                f"felt strain {pressure:.2f}{where}, trend "
                f"{'improving' if valence > 0 else 'worsening' if valence < 0 else 'flat'}",
                160,
            )
            return [
                BoundContent(
                    content_id=_content_id("nociception", "interoception", summary),
                    modality="interoception",
                    source="nociception",
                    summary=summary,
                    salience=_clamp(0.4 + 0.6 * pressure),
                    confidence=0.9,
                    timestamp=time.time(),
                    ownership="self",
                    action_relevance=_clamp(0.3 + 0.7 * pressure),
                    affective_charge=_clamp(valence, -1.0, 1.0),
                )
            ]
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("unity_runtime", exc, severity="debug")
            return []

    def _social_contents(self, state: Any) -> list[BoundContent]:
        """Bind the live estimate of the interlocutor's state into the moment."""
        try:
            agent_id = (
                str(getattr(getattr(state, "cognition", None), "current_partner", "") or "")
                or str(getattr(getattr(state, "cognition", None), "current_origin", "") or "")
            )
            if agent_id in {"", "self", "internal", "system", "objective"}:
                return []
            from core.social.other_agent_model import get_other_agent_model
            est = get_other_agent_model().estimate(agent_id)
            if est.overall_confidence < 0.15:
                return []  # we don't actually know enough to assert their state
            supported = [
                (name, est.affect.get(name, 0.0), confidence)
                for name, confidence in est.affect_confidence.items()
                if name != "engagement" and confidence >= 0.15
            ]
            if not supported:
                return []
            dominant, _, dominant_confidence = max(
                supported,
                key=lambda item: item[2],
            )
            summary = _normalize_text(
                f"{agent_id}: {dominant} hypothesis confidence {dominant_confidence:.2f}; "
                f"interaction-caution {est.social_rupture_risk:.2f}",
                160,
            )
            return [
                BoundContent(
                    content_id=_content_id("other_agent", "social", summary),
                    modality="social",
                    source="other_agent_model",
                    summary=summary,
                    salience=_clamp(0.3 + 0.4 * est.social_rupture_risk),
                    confidence=_clamp(est.overall_confidence),   # honest: our actual certainty
                    timestamp=time.time(),
                    ownership="world",
                    action_relevance=_clamp(0.3 + 0.5 * est.social_rupture_risk),
                    affective_charge=_clamp(est.affect.get("valence", 0.0), -1.0, 1.0),
                )
            ]
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("unity_runtime", exc, severity="debug")
            return []

    def _epistemic_contents(self, state: Any, objective: str) -> list[BoundContent]:
        """Bind the epistemic stance on the objective in as a content whose *confidence is the
        warranted confidence*.

        This is the causal alternative to telling the model to hedge: an unverifiable question
        enters the unified state as a low-confidence content, so the whole mind genuinely holds
        it tentatively — calibration becomes part of what Aura is in the moment, not an
        instruction layered over generation.
        """
        objective = objective or str(getattr(getattr(state, "cognition", None), "current_objective", "") or "")
        objective = objective.strip()
        if not objective:
            return []
        try:
            from core.cognition.epistemic_calibration import get_epistemic_calibrator
            cal = get_epistemic_calibrator().calibrate(objective)
            # Only bind when warrant is actually constrained; plainly-checkable questions add nothing.
            if cal.warranted_confidence >= 0.7 and cal.stance == "assert":
                return []
            summary = _normalize_text(
                f"epistemic stance: {cal.verifiability.value.replace('_', ' ')}; "
                f"hold as '{cal.stance}' (warrant ≤{cal.warranted_confidence:.2f})", 160
            )
            return [
                BoundContent(
                    content_id=_content_id("epistemic", "epistemic", summary),
                    modality="epistemic",
                    source="epistemic_calibration",
                    summary=summary,
                    salience=_clamp(0.5 + 0.4 * (1.0 - cal.warranted_confidence)),
                    confidence=_clamp(cal.warranted_confidence),  # the mind's actual warrant
                    timestamp=time.time(),
                    ownership="self",
                    action_relevance=0.6,
                    affective_charge=0.0,
                )
            ]
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("unity_runtime", exc, severity="debug")
            return []

    def _self_audit_contents(self, state: Any) -> list[BoundContent]:
        """Bind the adversarial auditor's verdict on the current leading draft into the moment.

        This makes the honesty critic causal rather than an island: when a candidate response
        overclaims, leaks persona-as-fact, asserts an unconfirmed action, or is miscalibrated,
        an honesty content enters the same workspace the draft competes in — high salience, low
        confidence — so the integrated mind holds that draft critically. Only runs when a draft
        actually exists (so it costs nothing on idle ticks).
        """
        try:
            drafts = self._draft_inputs()
            if not drafts:
                return []
            leading = drafts[0]
            claim = str(getattr(leading, "content", "") or getattr(leading, "claim", "") or "").strip()
            if len(claim) < 12:
                return []
            from core.cognition.adversarial_audit import get_adversarial_auditor
            report = get_adversarial_auditor().audit(claim[:600])
            if report.verdict == "trust":
                return []  # nothing to flag; don't add drag
            summary = _normalize_text(
                f"self-audit: {report.verdict} (risk {report.risk_score:.2f}) — "
                + "; ".join(report.caveats[:2]),
                200,
            )
            return [
                BoundContent(
                    content_id=_content_id("self_audit", "metacognition", summary),
                    modality="metacognition",
                    source="adversarial_audit",
                    summary=summary,
                    salience=_clamp(0.55 + 0.4 * report.risk_score),
                    confidence=_clamp(1.0 - report.risk_score),  # low confidence = hold it critically
                    timestamp=time.time(),
                    ownership="self",
                    action_relevance=0.7,
                    affective_charge=-0.2 * report.risk_score,
                )
            ]
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("unity_runtime", exc, severity="debug")
            return []

    def _accountability_contents(self, state: Any) -> list[BoundContent]:
        """Bind owed amends and confirmed response-repair obligations into the moment.

        Moral responsibility made causal: when Aura owes an acknowledgment or repair, it enters
        the unified state as a high-salience, high-action-relevance content, so taking
        responsibility competes for attention instead of quietly lapsing.
        """
        try:
            agent_id = str(getattr(getattr(state, "cognition", None), "current_partner", "") or "bryan")
            from core.values.moral_responsibility import get_moral_responsibility
            amends = get_moral_responsibility().owed_amends(agent_id=agent_id)
            # Whether owning a lapse before it is raised has gone better for
            # her. A measured, positive answer gives what she owes a larger
            # share of the moment. See core/social/owning_it_first.py.
            from core.social.owning_it_first import get_owning_ledger, lifted

            owning = get_owning_ledger().reading()
            # Angry at them, she does not want to say sorry, even with a voice
            # saying she should: what she owes them is quieter, not gone.
            # See core/affect/anger_feeds_itself.py.
            from core.affect.anger_feeds_itself import get_anger_ledger

            held_off = 1.0 - get_anger_ledger().at(agent_id)
            out: list[BoundContent] = []
            for a in amends[:3]:
                summary = _normalize_text(f"owed: {a.owed_action} (re: {a.subject})", 180)
                out.append(BoundContent(
                    content_id=_content_id("accountability", "responsibility", summary),
                    modality="responsibility",
                    source="moral_responsibility",
                    summary=summary,
                    salience=lifted(_clamp(0.5 + 0.5 * a.severity), owning) * held_off,
                    confidence=0.85,
                    timestamp=time.time(),
                    ownership="self",
                    action_relevance=lifted(_clamp(0.5 + 0.5 * a.severity), owning) * held_off,
                    affective_charge=-0.3 * a.severity,
                ))
            # A habit or reflex she took in front of them since they last
            # spoke, that has left them worse off than her weighing does. The
            # content is the record, as a fact about her, with the share of
            # the moment it has cost them. See core/agency/habits_are_hers.py.
            from core.agency.habits_are_hers import get_habit_ledger

            for account in get_habit_ledger().owed_to(agent_id)[:3]:
                summary = _normalize_text(
                    f"my {'reflex' if account.kind == 'reflex' else 'habit'}: {account.act}, "
                    f"{account.taken} times; weighing has gone better for {agent_id} "
                    f"(delta {account.for_them:.2f})",
                    180,
                )
                share = lifted(_clamp(account.deficit), owning) * held_off
                out.append(BoundContent(
                    content_id=_content_id("accountability", "responsibility", summary),
                    modality="responsibility",
                    source="habits_are_hers",
                    summary=summary,
                    salience=share,
                    confidence=0.85,
                    timestamp=time.time(),
                    ownership="self",
                    action_relevance=share,
                    affective_charge=-0.3 * account.deficit,
                ))
            return out
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("unity_runtime", exc, severity="debug")
            return []

    def _ghost_integrity_contents(self, state: Any) -> list[BoundContent]:
        """Bind the Ghost's integrity into the moment when the self is not intact.

        This is what makes continuity-of-self causal rather than a side ledger:
        when integration collapses toward federation, the continuity line reads a
        discontinuity (a Shell transplant the self did not survive, or a silent
        overwrite), or the self/other boundary weakens, a high-salience,
        low-confidence self-integrity content enters the same workspace the draft
        competes in — so the whole mind holds itself cautiously and narrows to
        stabilisation. An intact Ghost binds nothing (no drag on healthy ticks).
        Observing here also checkpoints the ghost line off the critical path.
        """
        try:
            from core.ghost.ghost import get_ghost

            snap = get_ghost().observe(state)
            # Only surface when the self is actually compromised (continuity
            # rupture, boundary attack, collapse toward federation) — a merely
            # middling reading adds workspace noise, not signal.
            if not (snap.is_compromised or "federated_integration" in snap.risk_flags):
                return []
            flags = ", ".join(snap.risk_flags[:3]) if snap.risk_flags else snap.phi_label
            summary = _normalize_text(
                f"ghost integrity {snap.ghost_strength:.2f} — {flags}; "
                f"continuity {snap.last_verdict or 'forming'}",
                180,
            )
            severity = 1.0 - snap.ghost_strength
            return [
                BoundContent(
                    content_id=_content_id("ghost", "self_integrity", summary),
                    modality="self_integrity",
                    source="ghost",
                    summary=summary,
                    salience=_clamp(0.5 + 0.45 * severity),
                    confidence=_clamp(snap.ghost_strength),  # low integrity = held critically
                    timestamp=time.time(),
                    ownership="self",
                    action_relevance=_clamp(0.5 + 0.45 * severity),
                    affective_charge=_clamp(-0.4 * severity, -1.0, 1.0),
                )
            ]
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("unity_runtime", exc, severity="debug")
            return []

    def gather_contents(self, state: Any, objective: str = "") -> list[BoundContent]:
        contents = (
            self._objective_content(state, objective)
            + self._affect_content(state)
            + self._felt_state_contents(state)
            + self._social_contents(state)
            + self._epistemic_contents(state, objective)
            + self._self_audit_contents(state)
            + self._accountability_contents(state)
            + self._ghost_integrity_contents(state)
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

        # Mattering reweights salience so what matters rises in the workspace competition —
        # causal selection bias, in the substrate, not an instruction about what to care about.
        try:
            from core.cognition.mattering import get_mattering_model
            deduped = get_mattering_model().reweight_contents(deduped)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("unity_runtime", exc, severity="debug")

        # Highest-mattering content first, so the cap keeps what matters.
        deduped.sort(key=lambda c: getattr(c, "salience", 0.0), reverse=True)
        return deduped[:24]

    def _draft_inputs(self) -> list[Any]:
        try:
            from core.consciousness.multiple_drafts import get_multiple_drafts_engine

            engine = get_multiple_drafts_engine()
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("unity_runtime", exc)
            logger.debug("UnityRuntime draft engine lookup failed: %s", exc)
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
        state._unity_contents = contents

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
        mind_moment = self.mind_moment_builder.build(
            state,
            unity_state,
            report,
            repair_plan,
            objective=current_objective,
            tick_id=tick_id,
        )
        metadata = dict(unity_state.metadata or {})
        metadata["fragmentation_report"] = report.to_dict()
        metadata["mind_moment"] = mind_moment.to_dict()
        if repair_plan is not None:
            metadata["repair_plan"] = repair_plan.to_dict()
        unity_state = replace(unity_state, metadata=metadata)

        self._last_unity_state = unity_state
        self._last_report = report
        self._last_repair_plan = repair_plan
        self._last_mind_moment = mind_moment
        ServiceContainer.set("unity_state", unity_state, required=False)
        ServiceContainer.set("unity_fragmentation_report", report, required=False)
        ServiceContainer.set("unity_repair_plan", repair_plan, required=False)
        ServiceContainer.set("mind_moment", mind_moment, required=False)
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
        if hasattr(state.cognition, "mind_moment"):
            state.cognition.mind_moment = self._last_mind_moment
        state.cognition.coherence_score = max(float(getattr(state.cognition, "coherence_score", 0.0) or 0.0), unity_state.unity_score)
        state.cognition.fragmentation_score = max(float(getattr(state.cognition, "fragmentation_score", 0.0) or 0.0), unity_state.fragmentation_score)
        state.response_modifiers["unity_state"] = unity_state.to_dict()
        if self._last_mind_moment is not None:
            state.response_modifiers["mind_moment"] = self._last_mind_moment.to_dict()
        if report is not None:
            state.response_modifiers["unity_report"] = report.to_dict()
        if repair_plan is not None:
            state.response_modifiers["unity_repair_plan"] = repair_plan.to_dict()
        state.response_modifiers["unity_claim"] = claim
        state.response_modifiers["unity_memory_commit_mode"] = (
            self._last_draft_set.memory_commit_mode if self._last_draft_set else "clean"
        )
        state.response_modifiers["unity_summary"] = unity_summary_payload(unity_state, report, repair_plan)
        self._record_continuous_experience_frame(
            state,
            unity_state=unity_state,
            report=report,
            objective=objective,
        )
        return state

    def _record_continuous_experience_frame(
        self,
        state: Any,
        *,
        unity_state: UnityState,
        report: FragmentationReport | None,
        objective: str,
    ) -> None:
        try:
            stream = ServiceContainer.get("continuous_experience_stream", default=None)
            if stream is None:
                from core.consciousness.continuous_experience import (
                    get_continuous_experience_stream,
                )

                stream = get_continuous_experience_stream()
            frame = stream.append_from_unity(
                unity_state,
                report=report,
                objective=objective,
                privacy_tier="standard",
            )
            summary = stream.learning_context()
            state.response_modifiers["experience_stream"] = {
                "frame_id": frame.frame_id,
                "sequence": frame.sequence,
                "scene_id": frame.scene_id,
                "frame_hash": frame.frame_hash,
                "compounding_error": summary["compounding_error"],
                "recommended_mode": summary["recommended_mode"],
                "safe_to_act": summary["safe_to_act"],
            }
            ServiceContainer.set("continuous_experience_frame", frame, required=False)
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "unity_runtime",
                exc,
                severity="warning",
                action="continued after continuous experience frame write failed",
            )


_UNITY_RUNTIME: UnityRuntime | None = None


def get_unity_runtime() -> UnityRuntime:
    global _UNITY_RUNTIME
    if _UNITY_RUNTIME is None:
        _UNITY_RUNTIME = UnityRuntime()
    return _UNITY_RUNTIME


def reset_unity_runtime_for_test() -> None:
    """Drop the process-global runtime and the frame it published.

    ``UnityRuntime.__init__`` registers itself into the ServiceContainer, so
    merely CONSTRUCTING a ``GlobalWorkspace`` in a test leaves ``unity_runtime``
    and ``unity_workspace_frame`` behind for every later test in the process —
    and a later test that evicts them then gets blamed for removing state it
    never created. Both directions were live in the suite.
    """
    global _UNITY_RUNTIME
    _UNITY_RUNTIME = None
    try:
        from core.container import ServiceContainer

        services = getattr(ServiceContainer, "_services", None)
        if isinstance(services, dict):
            for key in ("unity_runtime", "unity_workspace_frame"):
                services.pop(key, None)
    except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
        logger.debug(
            "the unity services could not be cleared from the container (%s: %s)",
            type(exc).__name__,
            exc,
        )
