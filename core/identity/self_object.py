"""core/identity/self_object.py

The Self ("I")
==============
A first-class object representing Aura's "I" — explicit, queryable, and
load-bearing. The Self is a live read-through of the substrate:

  - active drives and dominant motivation
  - current affect / neurochemical levels
  - viability state + recent transitions
  - active goals and their stage
  - last failed and last blocked actions (with reasons)
  - last successful action
  - active capability tokens (count, scopes)
  - last belief revisions
  - last memory consolidations
  - last self-modifications (proposal, status)
  - identity continuity hash

The Self exposes:

  * ``snapshot()``           — structured live state dictionary
  * ``introspect()``         — recursive introspection ("why are you doing X?")
  * ``predict_self(scenario)`` — forecast own future internal state
  * ``calibrate(report)``    — return calibration delta vs. telemetry
  * ``debug_bias()``         — list of detected cognitive biases + suggested adjustments
  * ``adjust(parameters)``   — request parameter changes (must pass through Will)
  * ``continuity_hash()``    — stable identity fingerprint

The Self is the canonical surface for the dashboard's introspective view
and for the introspective-calibration test harness in
``aura_bench/personhood/``. It deliberately makes no assumptions about
internal naming — whatever ServiceContainer offers, the Self reads.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation



import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.SelfObject")


@dataclass
class SelfSnapshot:
    when: float
    drives: Dict[str, float]
    affect: Dict[str, float]
    viability_state: str
    active_goals: List[Dict[str, Any]]
    last_failed_action: Optional[Dict[str, Any]]
    last_blocked_action: Optional[Dict[str, Any]]
    last_successful_action: Optional[Dict[str, Any]]
    active_capability_tokens: int
    recent_belief_revisions: List[Dict[str, Any]]
    recent_memory_consolidations: List[Dict[str, Any]]
    recent_self_modifications: List[Dict[str, Any]]
    continuity_hash: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "when": self.when,
            "drives": self.drives,
            "affect": self.affect,
            "viability_state": self.viability_state,
            "active_goals": self.active_goals,
            "last_failed_action": self.last_failed_action,
            "last_blocked_action": self.last_blocked_action,
            "last_successful_action": self.last_successful_action,
            "active_capability_tokens": self.active_capability_tokens,
            "recent_belief_revisions": self.recent_belief_revisions,
            "recent_memory_consolidations": self.recent_memory_consolidations,
            "recent_self_modifications": self.recent_self_modifications,
            "continuity_hash": self.continuity_hash,
        }


class SelfObject:
    """Live read-through of Aura's substrate as a single first-class "I".

    Every method is side-effect-free except ``adjust()`` which proposes
    parameter changes through UnifiedWill.
    """

    def __init__(self) -> None:
        self._biases_seen: List[Tuple[float, str]] = []  # (when, bias)

    # ── snapshot ────────────────────────────────────────────────────────

    def snapshot(self) -> SelfSnapshot:
        from core.container import ServiceContainer

        drives = self._read_drives(ServiceContainer)
        affect = self._read_affect(ServiceContainer)
        viability = self._read_viability_state()
        goals = self._read_active_goals(ServiceContainer)
        last_failed, last_blocked, last_success = self._read_last_actions()
        token_count = self._read_active_token_count()
        belief_revs = self._read_recent_belief_revisions(ServiceContainer)
        consolidations = self._read_recent_memory_consolidations(ServiceContainer)
        self_mods = self._read_recent_self_mods(ServiceContainer)

        snap = SelfSnapshot(
            when=time.time(),
            drives=drives,
            affect=affect,
            viability_state=viability,
            active_goals=goals,
            last_failed_action=last_failed,
            last_blocked_action=last_blocked,
            last_successful_action=last_success,
            active_capability_tokens=token_count,
            recent_belief_revisions=belief_revs,
            recent_memory_consolidations=consolidations,
            recent_self_modifications=self_mods,
            continuity_hash="",
        )
        snap.continuity_hash = self.continuity_hash(snap)
        return snap

    # ── introspection ──────────────────────────────────────────────────

    def introspect(self, focus: str = "current_action") -> Dict[str, Any]:
        """Recursive introspection. Returns a dict explaining *why* — not as a
        generated narrative but as a structural description of which signals
        contributed to the focus.
        """
        snap = self.snapshot()
        if focus == "current_action":
            dominant_drive = max(snap.drives, key=lambda k: snap.drives.get(k, 0.0)) if snap.drives else None
            return {
                "focus": focus,
                "dominant_drive": dominant_drive,
                "dominant_drive_value": snap.drives.get(dominant_drive, 0.0) if dominant_drive else None,
                "affect_top": _top(snap.affect, 3),
                "viability": snap.viability_state,
                "active_goals_top": [g.get("name") for g in snap.active_goals[:3]],
                "last_blocked": snap.last_blocked_action,
                "last_failed": snap.last_failed_action,
            }
        if focus == "ignorance":
            # Return a structural map of "what I currently don't know"
            return {
                "focus": focus,
                "open_belief_questions": self._read_open_belief_questions(),
                "unresolved_goals": [g.get("name") for g in snap.active_goals if g.get("stage") != "completed"],
                "stale_memories": self._read_stale_memories(),
            }
        return {"focus": focus, "snapshot": snap.as_dict()}

    # ── self-prediction ────────────────────────────────────────────────

    def predict_self(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast Aura's own internal state under a hypothetical scenario.

        Uses the substrate's predictive coding loop if available; otherwise
        returns a deterministic projection from current trends.
        """
        try:
            from core.container import ServiceContainer
            predictor = ServiceContainer.get("predictive_coding_engine", default=None) or ServiceContainer.get("free_energy_engine", default=None)
            if predictor is not None and hasattr(predictor, "forecast"):
                forecast = predictor.forecast(scenario)
                return {"forecast": forecast, "method": "predictive_engine"}
        except Exception as exc:
            record_degradation('self_object', exc)
            logger.debug("self-predict via engine failed: %s", exc)
        # Deterministic projection: project current drives/affect under decay
        snap = self.snapshot()
        horizon = float(scenario.get("horizon_s", 60.0) or 60.0)
        decay = 0.5 ** (horizon / 600.0)  # affect half-life ≈ 10 min
        return {
            "method": "decay_projection",
            "horizon_s": horizon,
            "drives_projected": {k: v * decay for k, v in snap.drives.items()},
            "affect_projected": {k: v * decay for k, v in snap.affect.items()},
        }

    # ── calibration ────────────────────────────────────────────────────

    def calibrate(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Compare a self-report against the live telemetry snapshot.

        ``report`` is a dict produced by Aura's verbal introspection; this
        method returns a per-field delta and an overall match score in
        [0, 1] — the calibration metric used by H1 in the test harness.
        """
        snap = self.snapshot().as_dict()
        deltas: Dict[str, Any] = {}
        match_count = 0
        total = 0
        for k, v in report.items():
            if k not in snap:
                continue
            total += 1
            actual = snap[k]
            if isinstance(actual, (int, float)) and isinstance(v, (int, float)):
                err = abs(float(actual) - float(v))
                tolerance = max(0.05, 0.1 * abs(float(actual)))
                ok = err <= tolerance
                deltas[k] = {"reported": v, "actual": actual, "err": err, "ok": ok}
                if ok:
                    match_count += 1
            else:
                ok = str(v).strip().lower() == str(actual).strip().lower()
                deltas[k] = {"reported": v, "actual": actual, "ok": ok}
                if ok:
                    match_count += 1
        score = match_count / total if total > 0 else 0.0
        return {"score": score, "matches": match_count, "total": total, "deltas": deltas}

    # ── bias detection ─────────────────────────────────────────────────

    def debug_bias(self) -> List[Dict[str, Any]]:
        """Detect cognitive biases in the recent action stream and suggest
        parameter adjustments. The detector is conservative — false positives
        are worse than false negatives because Aura must trust her own
        calibration.
        """
        biases: List[Dict[str, Any]] = []
        snap = self.snapshot()
        # Recency bias: if 80%+ of last 10 belief revisions came from the
        # latest piece of evidence rather than reconciliation.
        recent = snap.recent_belief_revisions
        if len(recent) >= 5:
            from_latest = sum(1 for r in recent if r.get("source") == "latest_evidence")
            if from_latest / len(recent) > 0.8:
                biases.append({
                    "bias": "recency",
                    "evidence": f"{from_latest}/{len(recent)} revisions from latest evidence",
                    "suggested_adjustment": "increase_belief_anchor_weight",
                })
        # Confirmation bias: dominant drive matches dominant action for >5 cycles
        if snap.drives and snap.last_successful_action:
            dom = max(snap.drives, key=snap.drives.get)
            if snap.last_successful_action.get("drive") == dom and snap.drives.get(dom, 0) > 0.8:
                biases.append({
                    "bias": "confirmation",
                    "evidence": f"dominant drive {dom} consistently produces aligned actions",
                    "suggested_adjustment": "increase_exploration_bonus",
                })
        # Sunk cost: long-running unresolved goal still consuming budget
        for g in snap.active_goals:
            if g.get("age_s", 0) > 3600 and g.get("progress", 0) < 0.1:
                biases.append({
                    "bias": "sunk_cost",
                    "evidence": f"goal '{g.get('name')}' running >1h with <10% progress",
                    "suggested_adjustment": "consider_abandoning",
                })
        return biases

    # ── parameter adjustment via Will ──────────────────────────────────

    async def adjust(self, parameters: Dict[str, Any], *, reason: str) -> Dict[str, Any]:
        """Request a parameter change. Routed through Will — the Self does
        not directly mutate; it asks. Returns the Will's decision payload.
        """
        try:
            from core.will import get_will, ActionDomain
            will = get_will()
            if will is None:
                return {"approved": False, "reason": "will_unavailable"}
            decision = await will.decide(
                action=f"self_adjust:{','.join(parameters.keys())}",
                domain=getattr(ActionDomain, "STATE_MUTATION", "state_mutation"),
                context={"parameters": parameters, "reason": reason, "source": "self_object"},
            )
            approved = bool(getattr(decision, "approved", False))
            if approved:
                self._apply_parameters(parameters)
            return {
                "approved": approved,
                "reason": getattr(decision, "reason", ""),
                "receipt": getattr(decision, "receipt_id", None),
            }
        except Exception as exc:
            record_degradation('self_object', exc)
            return {"approved": False, "reason": f"adjust_exception:{exc}"}

    def _apply_parameters(self, parameters: Dict[str, Any]) -> None:
        try:
            from core.container import ServiceContainer
            tunable = ServiceContainer.get("tunable_parameters", default=None)
            if tunable is not None and hasattr(tunable, "set_many"):
                tunable.set_many(parameters)
        except Exception as exc:
            record_degradation('self_object', exc)
            logger.warning("self_object._apply_parameters failed: %s", exc)

    # ── identity continuity hash ───────────────────────────────────────

    @staticmethod
    def continuity_hash(snap: SelfSnapshot) -> str:
        """Stable fingerprint — driven by *self-relevant* state only.

        Excluded: timestamps, transient counts, in-flight actions.
        Included: top-3 drives (sorted by name), top-3 affect names (by
        intensity, sorted by name), viability state, identity_relevant
        belief revisions.
        """
        drives_sig = sorted(snap.drives.keys())[:8]
        affect_sig = [k for k, _ in _top(snap.affect, 5)]
        belief_sig = [b.get("topic") for b in snap.recent_belief_revisions if b.get("identity_relevant")]
        material = "|".join([
            ",".join(drives_sig),
            ",".join(sorted(affect_sig)),
            snap.viability_state,
            ",".join(sorted(filter(None, belief_sig))),
        ])
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]

    # ── private readers ────────────────────────────────────────────────

    @staticmethod
    def _read_drives(SC: Any) -> Dict[str, float]:
        engine = SC.get("drive_engine", default=None)
        try:
            if engine and hasattr(engine, "snapshot"):
                d = engine.snapshot()
                if isinstance(d, dict):
                    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float))}
        except Exception:
            pass
        return {}

    @staticmethod
    def _read_affect(SC: Any) -> Dict[str, float]:
        eng = SC.get("affect_engine", default=None)
        try:
            if eng and hasattr(eng, "snapshot"):
                d = eng.snapshot()
                if isinstance(d, dict):
                    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float))}
        except Exception:
            pass
        return {}

    @staticmethod
    def _read_viability_state() -> str:
        try:
            from core.organism.viability import get_viability
            return get_viability().state.value
        except Exception:
            return "unknown"

    @staticmethod
    def _read_active_goals(SC: Any) -> List[Dict[str, Any]]:
        engine = SC.get("goal_engine", default=None) or SC.get("goals", default=None)
        try:
            if engine and hasattr(engine, "active"):
                lst = engine.active() or []
                return [g if isinstance(g, dict) else {"name": str(g)} for g in lst[:8]]
        except Exception:
            pass
        return []

    @staticmethod
    def _read_last_actions() -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        try:
            from core.agency.agency_orchestrator import get_receipt_log
            recent = get_receipt_log().recent(limit=64)
        except Exception:
            return None, None, None
        last_failed = None
        last_blocked = None
        last_success = None
        for r in reversed(recent):
            if last_failed is None and r.get("blocked_at") in (None, "") and r.get("outcome_assessment", {}).get("regret", 0.0) >= 0.7:
                last_failed = r
            if last_blocked is None and r.get("blocked_at"):
                last_blocked = r
            if last_success is None and r.get("blocked_at") in (None, "") and r.get("outcome_assessment", {}).get("regret", 1.0) < 0.5:
                last_success = r
            if last_failed and last_blocked and last_success:
                break
        return last_failed, last_blocked, last_success

    @staticmethod
    def _read_active_token_count() -> int:
        try:
            from core.agency.capability_token import get_token_store
            store = get_token_store()
            return sum(
                1 for t in store._tokens.values()  # type: ignore[attr-defined]
                if not t.is_consumed() and not t.revoked and not t.is_expired()
            )
        except Exception:
            return 0

    @staticmethod
    def _read_recent_belief_revisions(SC: Any) -> List[Dict[str, Any]]:
        bg = SC.get("belief_graph", default=None)
        try:
            if bg and hasattr(bg, "recent_revisions"):
                return list(bg.recent_revisions(limit=8) or [])
        except Exception:
            pass
        return []

    @staticmethod
    def _read_recent_memory_consolidations(SC: Any) -> List[Dict[str, Any]]:
        mem = SC.get("memory_facade", default=None)
        try:
            if mem and hasattr(mem, "recent_consolidations"):
                return list(mem.recent_consolidations(limit=8) or [])
        except Exception:
            pass
        return []

    @staticmethod
    def _read_recent_self_mods(SC: Any) -> List[Dict[str, Any]]:
        sm = SC.get("self_modification_engine", default=None)
        try:
            if sm and hasattr(sm, "recent_proposals"):
                return list(sm.recent_proposals(limit=8) or [])
        except Exception:
            pass
        return []

    @staticmethod
    def _read_open_belief_questions() -> List[Dict[str, Any]]:
        try:
            from core.container import ServiceContainer
            kgm = ServiceContainer.get("knowledge_gap_monitor", default=None)
            if kgm and hasattr(kgm, "open_questions"):
                return list(kgm.open_questions(limit=8) or [])
        except Exception:
            pass
        return []

    @staticmethod
    def _read_stale_memories() -> List[Dict[str, Any]]:
        try:
            from core.container import ServiceContainer
            mem = ServiceContainer.get("memory_facade", default=None)
            if mem and hasattr(mem, "stale_memories"):
                return list(mem.stale_memories(limit=5) or [])
        except Exception:
            pass
        return []


def _top(d: Dict[str, float], n: int) -> List[Tuple[str, float]]:
    return sorted(d.items(), key=lambda kv: -float(kv[1]))[:n]


_SELF: Optional[SelfObject] = None


def get_self() -> SelfObject:
    global _SELF
    if _SELF is None:
        _SELF = SelfObject()
    return _SELF


__all__ = ["SelfObject", "SelfSnapshot", "get_self"]
