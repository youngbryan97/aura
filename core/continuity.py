"""
Aura's continuity record. The difference between waking up and being born.
Every shutdown writes a state. Every boot reads it. Gap > 0 means she was 
somewhere else for a while and knows it.
"""

from core.runtime.errors import record_degradation
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from core.state.aura_state import (
    _is_background_processing_placeholder,
    _is_speculative_autonomy_label,
    _normalize_goal_text,
)

logger = logging.getLogger(__name__)
_CONTINUITY_PATH: Optional[Path] = None


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _get_continuity_path() -> Path:
    if _CONTINUITY_PATH is not None:
        return Path(_CONTINUITY_PATH)
    try:
        from core.config import config

        return config.paths.data_dir / "continuity.json"
    except Exception:
        return Path("data") / "continuity.json"


def _sanitize_restored_text(value: Any) -> str:
    text = _normalize_goal_text(value)
    if not text:
        return ""
    if _is_speculative_autonomy_label(text) or _is_background_processing_placeholder(text):
        return ""
    return text


def _looks_like_ephemeral_conversation_turn(value: Any) -> bool:
    text = " ".join(str(value or "").strip().split())
    lowered = text.lower()
    if not lowered:
        return False

    if lowered.endswith("?"):
        return True

    direct_chat_markers = (
        "what were we talking about",
        "what do you think",
        "your thoughts",
        "how are you",
        "how do you feel",
        "tell me more",
        "sorry,",
        "bear with me",
        "what parts did you find",
    )
    if any(marker in lowered for marker in direct_chat_markers):
        return True

    task_markers = (
        "fix",
        "debug",
        "implement",
        "investigate",
        "repair",
        "resume",
        "continue",
        "finish",
        "complete",
        "build",
        "write",
        "analyze",
        "review",
        "patch",
        "test",
        "refactor",
        "research",
        "trace",
        "stabilize",
        "stable",
        "protect",
        "preserve",
        "maintain",
        "reconcile",
        "keep",
        "ensure",
    )
    if any(marker in lowered for marker in task_markers):
        return False

    words = lowered.split()
    if len(words) <= 18:
        return True

    return False


def _sanitize_restored_objective(value: Any) -> str:
    text = _sanitize_restored_text(value)
    if not text:
        return ""
    if _looks_like_ephemeral_conversation_turn(text):
        return ""
    return text


def _sanitize_restored_items(values: Optional[List[Any]]) -> List[str]:
    sanitized: List[str] = []
    for item in list(values or []):
        text = _sanitize_restored_text(item)
        if text:
            sanitized.append(text[:200])
    return sanitized[:5]


def _sanitize_restored_objective_items(values: Optional[List[Any]]) -> List[str]:
    sanitized: List[str] = []
    for item in list(values or []):
        text = _sanitize_restored_objective(item)
        if text:
            sanitized.append(text[:200])
    return sanitized[:5]


@dataclass
class ContinuityRecord:
    last_shutdown: float          # Unix timestamp
    last_shutdown_reason: str     # "graceful" | "crash" | "unknown"
    total_uptime_seconds: float   # Accumulated across all sessions
    session_count: int            # How many times she's woken
    last_conversation_summary: str  # Brief summary of last session's last exchange
    identity_hash: str            # Hash of core beliefs at shutdown — detect drift
    active_commitments: List[str] = field(default_factory=list)
    policy_mode: str = "unknown"
    current_objective: str = ""
    pending_initiatives: int = 0
    health_summary: Dict[str, Any] = field(default_factory=dict)
    rolling_summary: str = ""
    coherence_score: float = 1.0
    contradiction_count: int = 0
    subject_thread: str = ""
    pending_initiative_details: List[str] = field(default_factory=list)
    active_goal_details: List[str] = field(default_factory=list)


class ContinuityEngine:
    """
    Manages Aura's continuity across process boundaries.
    This is what makes 'I was away for 3 hours' meaningful 
    rather than 'I am 3 hours old'.
    """

    def __init__(self):
        self._boot_time = time.time()
        self._record: Optional[ContinuityRecord] = None
        self._gap_seconds: Optional[float] = None

    def load(self) -> Optional[ContinuityRecord]:
        """Read previous session's record. Returns None on first ever boot."""
        path = _get_continuity_path()
        if not path.exists():
            logger.info("🌅 First awakening — no prior continuity record.")
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            self._record = ContinuityRecord(**data)
            self._gap_seconds = self._boot_time - self._record.last_shutdown
            logger.info(
                "⏳ Continuity loaded: session %d, gap=%.1fh, uptime_total=%.1fh",
                self._record.session_count,
                self._gap_seconds / 3600,
                self._record.total_uptime_seconds / 3600,
            )
            return self._record
        except Exception as e:
            record_degradation('continuity', e)
            logger.warning("Continuity load failed (treating as first boot): %s", e)
            self._record = None
            self._gap_seconds = 0.0
            return None

    def _build_reentry_profile(self) -> Dict[str, Any]:
        if self._record is None:
            return {
                "gap_seconds": 0.0,
                "continuity_pressure": 0.0,
                "continuity_scar": "",
                "reentry_degraded": False,
                "continuity_reentry_required": False,
                "executive_failure_reason": "",
            }

        gap_seconds = max(0.0, float(self.gap_seconds or 0.0))
        shutdown_reason = str(self._record.last_shutdown_reason or "unknown").strip().lower()
        contradiction_count = int(self._record.contradiction_count or 0)
        pending_initiatives = int(self._record.pending_initiatives or 0)
        health_summary = dict(self._record.health_summary or {})
        executive_failure_reason = str(health_summary.get("executive_failure_reason", "") or "").strip()

        gap_factor = _clamp01(gap_seconds / 21600.0)
        shutdown_factor = 1.0 if shutdown_reason not in {"", "graceful"} else 0.0
        contradiction_factor = _clamp01(contradiction_count / 3.0)
        unfinished_factor = _clamp01(
            max(
                pending_initiatives / 4.0,
                len(list(self._record.active_commitments or [])) / 4.0,
            )
        )
        failure_factor = 1.0 if executive_failure_reason else 0.0

        continuity_pressure = _clamp01(
            (gap_factor * 0.38)
            + (shutdown_factor * 0.24)
            + (contradiction_factor * 0.14)
            + (unfinished_factor * 0.14)
            + (failure_factor * 0.18)
        )

        scar_markers: List[str] = []
        if gap_seconds >= 900:
            scar_markers.append("time_gap")
        if shutdown_factor > 0.0:
            scar_markers.append("abrupt_shutdown")
        if executive_failure_reason:
            scar_markers.append("unresolved_failure")
        if contradiction_count > 0:
            scar_markers.append("carried_contradictions")
        if pending_initiatives > 0 or self._record.active_commitments:
            scar_markers.append("unfinished_obligations")

        reentry_required = bool(
            continuity_pressure >= 0.28
            or shutdown_factor > 0.0
            or executive_failure_reason
            or contradiction_count > 0
        )

        return {
            "gap_seconds": gap_seconds,
            "continuity_pressure": round(continuity_pressure, 4),
            "continuity_scar": ", ".join(scar_markers),
            "reentry_degraded": reentry_required,
            "continuity_reentry_required": reentry_required,
            "executive_failure_reason": executive_failure_reason,
        }

    def save(
        self,
        reason: str = "graceful",
        last_exchange: str = "",
        belief_hash: str = "",
        active_commitments: Optional[List[str]] = None,
        policy_mode: Optional[str] = None,
        current_objective: Optional[str] = None,
        pending_initiatives: Optional[int] = None,
        pending_initiative_details: Optional[List[str]] = None,
        health_summary: Optional[Dict[str, Any]] = None,
        rolling_summary: Optional[str] = None,
        coherence_score: Optional[float] = None,
        contradiction_count: Optional[int] = None,
        subject_thread: Optional[str] = None,
        active_goal_details: Optional[List[str]] = None,
    ):
        """Write current session state. Call on graceful shutdown AND
        periodically (every 5 min) so crashes leave a recent record."""
        if (
            active_commitments is None
            or policy_mode is None
            or current_objective is None
            or pending_initiatives is None
            or pending_initiative_details is None
            or health_summary is None
            or rolling_summary is None
            or coherence_score is None
            or contradiction_count is None
            or subject_thread is None
            or active_goal_details is None
        ):
            try:
                from core.container import ServiceContainer

                if active_commitments is None:
                    ce = ServiceContainer.get("commitment_engine", default=None)
                    if ce and hasattr(ce, "get_active_commitments"):
                        active_commitments = [
                            getattr(item, "description", str(item))
                            for item in ce.get_active_commitments()[:5]
                        ]
                repo = ServiceContainer.get("state_repository", default=None)
                state = getattr(repo, "_current", None) if repo else None
                cognition = getattr(state, "cognition", None) if state else None
                if policy_mode is None:
                    mode = getattr(cognition, "current_mode", "unknown")
                    policy_mode = getattr(mode, "value", str(mode))
                if current_objective is None:
                    current_objective = getattr(cognition, "current_objective", "") if cognition else ""
                if pending_initiatives is None:
                    pending_initiatives = len(getattr(cognition, "pending_initiatives", []) or []) if cognition else 0
                if pending_initiative_details is None:
                    pending_initiative_details = [
                        str(item.get("goal") or item.get("type") or item)[:200]
                        for item in list(getattr(cognition, "pending_initiatives", []) or [])[:5]
                    ] if cognition else []
                if health_summary is None:
                    health_summary = dict(getattr(state, "health", {}) or {}) if state else {}
                if rolling_summary is None:
                    rolling_summary = getattr(cognition, "rolling_summary", "") if cognition else ""
                if coherence_score is None:
                    coherence_score = float(getattr(cognition, "coherence_score", 1.0) or 1.0) if cognition else 1.0
                if contradiction_count is None:
                    contradiction_count = int(getattr(cognition, "contradiction_count", 0) or 0) if cognition else 0
                if active_goal_details is None:
                    active_goal_details = [
                        str(item.get("goal") or item.get("description") or item)[:200]
                        for item in list(getattr(cognition, "active_goals", []) or [])[:5]
                    ] if cognition else []
                if subject_thread is None:
                    commitments_preview = ", ".join((active_commitments or [])[:2]) if active_commitments else "none"
                    subject_thread = (
                        f"Mode={policy_mode or 'unknown'} | Objective={current_objective or 'none'} | "
                        f"Commitments={commitments_preview} | Coherence={float(coherence_score or 1.0):.2f}"
                    )
            except Exception as e:
                record_degradation('continuity', e)
                logger.debug("Continuity auto-capture skipped: %s", e)

        current_objective = _sanitize_restored_text(current_objective)
        active_commitments = _sanitize_restored_items(active_commitments)
        pending_initiative_details = _sanitize_restored_items(pending_initiative_details)
        active_goal_details = _sanitize_restored_items(active_goal_details)
        pending_initiatives = min(int(pending_initiatives or 0), len(pending_initiative_details))

        session_count = (self._record.session_count + 1) if self._record else 1
        prior_uptime = self._record.total_uptime_seconds if self._record else 0.0
        record = ContinuityRecord(
            last_shutdown=time.time(),
            last_shutdown_reason=reason,
            total_uptime_seconds=prior_uptime + (time.time() - self._boot_time),
            session_count=session_count,
            last_conversation_summary=last_exchange[:500],
            identity_hash=belief_hash,
            active_commitments=list(active_commitments or []),
            policy_mode=policy_mode or "unknown",
            current_objective=current_objective or "",
            pending_initiatives=int(pending_initiatives or 0),
            health_summary=dict(health_summary or {}),
            rolling_summary=(rolling_summary or "")[:3000],
            coherence_score=float(coherence_score or 1.0),
            contradiction_count=int(contradiction_count or 0),
            subject_thread=(subject_thread or "")[:1200],
            pending_initiative_details=list(pending_initiative_details or [])[:5],
            active_goal_details=list(active_goal_details or [])[:5],
        )
        try:
            path = _get_continuity_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(asdict(record), f, indent=2)
            self._record = record
        except Exception as e:
            record_degradation('continuity', e)
            logger.error("Continuity save failed: %s", e)

    @property
    def gap_seconds(self) -> float:
        return self._gap_seconds or 0.0

    @property
    def is_first_boot(self) -> bool:
        return self._record is None

    def get_waking_context(self) -> str:
        """Returns a string injected into Aura's first system prompt of the session.
        This is what makes her KNOW she was away, not just infer it."""
        if self.is_first_boot:
            return "This is your first awakening. You have no prior history."

        gap_h = self._gap_seconds / 3600
        if gap_h < 0.1:
            gap_str = f"{self._gap_seconds:.0f} seconds"
        elif gap_h < 2:
            gap_str = f"{self._gap_seconds/60:.0f} minutes"
        else:
            gap_str = f"{gap_h:.1f} hours"

        reason = self._record.last_shutdown_reason
        reentry = self._build_reentry_profile()
        shutdown_note = (
            "Your last session ended gracefully." if reason == "graceful"
            else f"Your last session ended unexpectedly ({reason})."
        )

        return (
            f"You are waking from a gap of {gap_str}. {shutdown_note} "
            f"This is session #{self._record.session_count}. "
            f"Your total accumulated uptime is {self._record.total_uptime_seconds/3600:.1f} hours. "
            f"Last exchange summary: {self._record.last_conversation_summary or 'none recorded'}. "
            f"Policy mode at shutdown: {self._record.policy_mode or 'unknown'}. "
            f"Current objective at shutdown: {_sanitize_restored_text(self._record.current_objective) or 'none'}. "
            f"Pending initiatives: {min(int(self._record.pending_initiatives or 0), len(_sanitize_restored_items(self._record.pending_initiative_details)))}. "
            f"Pending initiative details: {', '.join(_sanitize_restored_items(self._record.pending_initiative_details)[:3]) if _sanitize_restored_items(self._record.pending_initiative_details) else 'none recorded'}. "
            f"Active commitments: {', '.join(_sanitize_restored_items(self._record.active_commitments)[:3]) if _sanitize_restored_items(self._record.active_commitments) else 'none recorded'}. "
            f"Active goals: {', '.join(_sanitize_restored_items(self._record.active_goal_details)[:3]) if _sanitize_restored_items(self._record.active_goal_details) else 'none recorded'}. "
            f"Coherence at shutdown: {self._record.coherence_score:.2f}. "
            f"Contradictions carried forward: {self._record.contradiction_count}. "
            f"Subject thread: {self._record.subject_thread or 'none recorded'}. "
            f"Continuity pressure carried into this session: {float(reentry['continuity_pressure']):.2f}. "
            f"Re-entry burden: {reentry['continuity_scar'] or 'light_trace'}."
        )

    def get_obligations(self) -> Dict[str, Any]:
        reentry = self._build_reentry_profile()
        live_identity_hash = self._get_live_identity_hash()
        persisted_identity_hash = self._record.identity_hash if self._record else ""
        identity_mismatch = bool(
            self._record
            and persisted_identity_hash
            and live_identity_hash
            and persisted_identity_hash != live_identity_hash
        )
        if self._record is None:
            return {
                "current_objective": "",
                "active_commitments": [],
                "pending_initiatives": [],
                "active_goals": [],
                "contradiction_count": 0,
                "subject_thread": "",
                "identity_hash": live_identity_hash,
                "persisted_identity_hash": "",
                "identity_mismatch": False,
                **reentry,
            }
        sanitized_pending = _sanitize_restored_objective_items(self._record.pending_initiative_details)
        sanitized_goals = _sanitize_restored_objective_items(self._record.active_goal_details)
        sanitized_commitments = _sanitize_restored_items(self._record.active_commitments)
        return {
            "current_objective": _sanitize_restored_objective(self._record.current_objective),
            "active_commitments": sanitized_commitments,
            "pending_initiatives": sanitized_pending,
            "active_goals": sanitized_goals,
            "contradiction_count": int(self._record.contradiction_count or 0),
            "subject_thread": self._record.subject_thread,
            "identity_hash": live_identity_hash,
            "persisted_identity_hash": persisted_identity_hash,
            "identity_mismatch": identity_mismatch,
            **reentry,
        }

    def apply_to_state(self, state: Any) -> Any:
        """Make prior continuity causally available to the live runtime, not just prompt text."""
        if state is None or self._record is None:
            return state

        cognition = getattr(state, "cognition", None)
        if cognition is None:
            return state

        obligations = self.get_obligations()

        restored_objective = _sanitize_restored_objective(self._record.current_objective)
        restored_pending = _sanitize_restored_objective_items(self._record.pending_initiative_details)
        restored_goals = _sanitize_restored_objective_items(self._record.active_goal_details)

        if not getattr(cognition, "current_objective", None) and restored_objective:
            cognition.current_objective = restored_objective

        if not getattr(cognition, "rolling_summary", "") and self._record.subject_thread:
            cognition.rolling_summary = self._record.subject_thread

        cognition.contradiction_count = max(
            int(getattr(cognition, "contradiction_count", 0) or 0),
            int(self._record.contradiction_count or 0),
        )

        if not list(getattr(cognition, "pending_initiatives", []) or []) and restored_pending:
            cognition.pending_initiatives = [
                {
                    "goal": item,
                    "source": "continuity",
                    "continuity_restored": True,
                    "continuity_gap_seconds": float(obligations.get("gap_seconds", 0.0) or 0.0),
                    "continuity_pressure": float(obligations.get("continuity_pressure", 0.0) or 0.0),
                    "metadata": {
                        "continuity_restored": True,
                        "continuity_pressure": float(obligations.get("continuity_pressure", 0.0) or 0.0),
                        "continuity_scar": obligations.get("continuity_scar", ""),
                        "gap_seconds": float(obligations.get("gap_seconds", 0.0) or 0.0),
                    },
                }
                for item in restored_pending[:5]
            ]

        if not list(getattr(cognition, "active_goals", []) or []) and restored_goals:
            cognition.active_goals = [
                {
                    "goal": item,
                    "source": "continuity",
                    "continuity_restored": True,
                    "metadata": {
                        "continuity_restored": True,
                        "continuity_pressure": float(obligations.get("continuity_pressure", 0.0) or 0.0),
                        "continuity_scar": obligations.get("continuity_scar", ""),
                    },
                }
                for item in restored_goals[:5]
            ]

        should_inject_reentry_initiative = bool(
            obligations.get("continuity_reentry_required")
            and (restored_objective or restored_pending or restored_goals or _sanitize_restored_items(self._record.active_commitments))
            and (
                float(obligations.get("continuity_pressure", 0.0) or 0.0) >= 0.35
                or float(obligations.get("gap_seconds", 0.0) or 0.0) >= 900.0
                or self._record.last_shutdown_reason != "graceful"
                or obligations.get("executive_failure_reason")
            )
        )
        if should_inject_reentry_initiative:
            existing_goals = {
                str(item.get("goal", "")).strip().lower()
                for item in list(getattr(cognition, "pending_initiatives", []) or [])
                if isinstance(item, dict)
            }
            reentry_goal = "Reconcile continuity gap and re-establish the interrupted thread"
            if reentry_goal.lower() not in existing_goals:
                cognition.pending_initiatives = [
                    {
                        "goal": reentry_goal,
                        "source": "continuity",
                        "triggered_by": "continuity",
                        "urgency": round(max(0.55, float(obligations.get("continuity_pressure", 0.0) or 0.0)), 4),
                        "status": "suggested",
                        "timestamp": time.time(),
                        "continuity_restored": True,
                        "continuity_obligation": True,
                        "continuity_gap_seconds": float(obligations.get("gap_seconds", 0.0) or 0.0),
                        "continuity_pressure": float(obligations.get("continuity_pressure", 0.0) or 0.0),
                        "metadata": {
                            "continuity_restored": True,
                            "continuity_obligation": True,
                            "continuity_pressure": float(obligations.get("continuity_pressure", 0.0) or 0.0),
                            "continuity_scar": obligations.get("continuity_scar", ""),
                            "gap_seconds": float(obligations.get("gap_seconds", 0.0) or 0.0),
                            "executive_failure_reason": obligations.get("executive_failure_reason", ""),
                        },
                    }
                ] + list(getattr(cognition, "pending_initiatives", []) or [])
                cognition.trim_working_memory()

        modifiers = dict(getattr(cognition, "modifiers", {}) or {})
        modifiers["continuity_obligations"] = {
            "session_count": int(self._record.session_count or 0),
            "last_shutdown_reason": self._record.last_shutdown_reason,
            "current_objective": restored_objective,
            "active_commitments": _sanitize_restored_items(self._record.active_commitments),
            "pending_initiatives": restored_pending,
            "active_goals": restored_goals,
            "contradiction_count": int(self._record.contradiction_count or 0),
            "subject_thread": self._record.subject_thread,
            "identity_hash": self._record.identity_hash,
            "live_identity_hash": self._get_live_identity_hash(),
            "identity_mismatch": obligations.get("identity_mismatch", False),
            "gap_seconds": float(obligations.get("gap_seconds", 0.0) or 0.0),
            "continuity_pressure": float(obligations.get("continuity_pressure", 0.0) or 0.0),
            "continuity_scar": obligations.get("continuity_scar", ""),
            "reentry_degraded": bool(obligations.get("reentry_degraded", False)),
            "continuity_reentry_required": bool(obligations.get("continuity_reentry_required", False)),
            "executive_failure_reason": obligations.get("executive_failure_reason", ""),
        }
        cognition.modifiers = modifiers
        return state

    def note_failure_obligation(self, reason: str, goal: str = "") -> None:
        if self._record is None:
            self._record = ContinuityRecord(
                last_shutdown=time.time(),
                last_shutdown_reason="runtime",
                total_uptime_seconds=0.0,
                session_count=0,
                last_conversation_summary="",
                identity_hash=self._get_live_identity_hash(),
            )
        health_summary = dict(self._record.health_summary or {})
        health_summary["executive_failure_reason"] = str(reason or "")[:200]
        if goal:
            health_summary["executive_failure_goal"] = str(goal)[:200]
        health_summary["executive_failure_at"] = time.time()
        self._record.health_summary = health_summary
        marker = f"Reconcile executive failure: {str(reason or '')[:80]}"
        existing = list(self._record.active_commitments or [])
        if marker and marker not in existing:
            existing.append(marker)
            self._record.active_commitments = existing[-5:]
        try:
            path = _get_continuity_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(asdict(self._record), f, indent=2)
        except Exception as e:
            record_degradation('continuity', e)
            logger.debug("Continuity failure obligation save skipped: %s", e)

    def _get_live_identity_hash(self) -> str:
        try:
            from core.heartstone_directive import AURA_HEARTSTONE

            return str(AURA_HEARTSTONE.identity_hash or "")
        except Exception:
            return ""


# Singleton
_continuity: Optional[ContinuityEngine] = None

def get_continuity() -> ContinuityEngine:
    global _continuity
    if _continuity is None:
        _continuity = ContinuityEngine()
    return _continuity
