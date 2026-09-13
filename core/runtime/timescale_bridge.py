"""Bridge continuous metabolism to dialogue-speed foreground turns.

The perceptual pump and background organs tick far faster than a user speaks.
This service keeps a bounded, model-free summary of that continuous state and
reconciles it when the next foreground turn arrives. The goal is to enrich
context with real observations while preventing idle narrative drift from being
mistaken for events that happened.
"""

from __future__ import annotations

import os
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from typing import Any

from core.container import ServiceContainer


def _bounded_float(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError, OverflowError):
        return default


def _bounded_text(value: Any, limit: int = 160) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].strip() or text[:limit]


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class TimescaleObservation:
    timestamp: float
    source: str
    active_app: str = ""
    window_title: str = ""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    thermal_pressure: float = 0.0
    idle_seconds: float = 0.0
    novelty: float = 0.0
    threat: float = 0.0
    social: float = 0.0
    voice_activity: bool = False
    screen_changed: bool = False
    ambient_event_count: int = 0
    ambient_summary: str = ""
    ambient_repair_candidates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ambient_repair_candidates"] = list(self.ambient_repair_candidates)
        return data


@dataclass(frozen=True)
class TimescaleReconciliation:
    schema: str = "aura.timescale_reconciliation.v1"
    at: float = field(default_factory=lambda: time.time())
    idle_gap_s: float = 0.0
    observations_considered: int = 0
    observed_apps: tuple[str, ...] = ()
    observed_windows: tuple[str, ...] = ()
    max_memory_percent: float = 0.0
    max_cpu_percent: float = 0.0
    max_threat: float = 0.0
    max_novelty: float = 0.0
    user_returned_after_idle: bool = False
    narrative_drift_risk: float = 0.0
    foreground_anchor_required: bool = False
    memory_grounding_bias: float = 0.0
    sensory_grounding_bias: float = 0.0
    ambient_event_count: int = 0
    ambient_summaries: tuple[str, ...] = ()
    ambient_repair_candidates: tuple[str, ...] = ()
    summary: str = ""
    directives: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["observed_apps"] = list(self.observed_apps)
        data["observed_windows"] = list(self.observed_windows)
        data["ambient_summaries"] = list(self.ambient_summaries)
        data["ambient_repair_candidates"] = list(self.ambient_repair_candidates)
        data["directives"] = list(self.directives)
        return data


class TimescaleBridge:
    """Bounded bridge between fast background loops and slow user dialogue."""

    def __init__(
        self,
        *,
        sample_interval_s: float | None = None,
        idle_anchor_threshold_s: float | None = None,
        max_observations: int = 240,
    ) -> None:
        self.sample_interval_s = (
            sample_interval_s
            if sample_interval_s is not None
            else _env_float(
                "AURA_TIMESCALE_BRIDGE_SAMPLE_INTERVAL_S",
                5.0,
                minimum=1.0,
                maximum=60.0,
            )
        )
        self.idle_anchor_threshold_s = (
            idle_anchor_threshold_s
            if idle_anchor_threshold_s is not None
            else _env_float(
                "AURA_TIMESCALE_BRIDGE_IDLE_ANCHOR_S",
                300.0,
                minimum=30.0,
                maximum=7200.0,
            )
        )
        self._observations: deque[TimescaleObservation] = deque(maxlen=max(8, max_observations))
        self._last_sample_at = 0.0
        self._last_sample_at_by_source: dict[str, float] = {}
        self._last_user_turn_at = 0.0
        self._last_reconciliation: TimescaleReconciliation | None = None
        self._frames_ingested = 0
        self._registered_at = time.time()

    def _should_sample_source(self, source: str, now: float) -> bool:
        key = _bounded_text(source, 80) or "unknown"
        previous = self._last_sample_at_by_source.get(key, 0.0)
        if (now - previous) < self.sample_interval_s:
            return False
        self._last_sample_at_by_source[key] = now
        self._last_sample_at = now
        return True

    def ingest_perceptual_frame(self, frame: Any, *, source: str = "perceptual_pump") -> None:
        """Summarize a perceptual frame without retaining raw sensory payloads."""

        now = float(getattr(frame, "timestamp", 0.0) or time.time())
        if not self._should_sample_source(source, now):
            return
        self._frames_ingested += 1

        screen = getattr(frame, "screen", None)
        system = getattr(frame, "system", None)
        user = getattr(frame, "user", None)
        audio = getattr(frame, "audio", None)
        try:
            novelty = float(frame.novelty_score())
        except (AttributeError, TypeError, ValueError, OverflowError):
            novelty = 0.0
        try:
            threat = float(frame.threat_score())
        except (AttributeError, TypeError, ValueError, OverflowError):
            threat = 0.0
        try:
            social = float(frame.social_signal())
        except (AttributeError, TypeError, ValueError, OverflowError):
            social = 0.0

        self._observations.append(
            TimescaleObservation(
                timestamp=now,
                source=_bounded_text(source, 80),
                active_app=_bounded_text(getattr(screen, "active_app", ""), 120),
                window_title=_bounded_text(getattr(screen, "window_title", ""), 180),
                cpu_percent=float(getattr(system, "cpu_percent", 0.0) or 0.0),
                memory_percent=float(getattr(system, "memory_percent", 0.0) or 0.0),
                thermal_pressure=_bounded_float(getattr(system, "thermal_pressure", 0.0)),
                idle_seconds=float(getattr(user, "idle_seconds", 0.0) or 0.0),
                novelty=_bounded_float(novelty),
                threat=_bounded_float(threat),
                social=_bounded_float(social),
                voice_activity=bool(getattr(audio, "voice_activity", False)),
                screen_changed=bool(getattr(screen, "screen_changed", False)),
            )
        )

    def ingest_ambient_developer_frame(
        self,
        frame: Any,
        *,
        source: str = "ambient_developer_stream",
    ) -> None:
        """Summarize ambient developer/runtime evidence into the dialogue bridge."""

        now = float(getattr(frame, "timestamp", 0.0) or time.time())
        if not self._should_sample_source(source, now):
            return
        self._frames_ingested += 1

        data = frame.to_dict() if hasattr(frame, "to_dict") else dict(frame or {})
        repair_candidates = tuple(
            _bounded_text(item, 80)
            for item in data.get("repair_candidates", [])[:6]
            if str(item or "").strip()
        )
        self._observations.append(
            TimescaleObservation(
                timestamp=now,
                source=_bounded_text(source, 80),
                novelty=0.25 if data.get("git_dirty_count") or data.get("recent_files") else 0.05,
                threat=0.35 if data.get("log_events") else 0.0,
                ambient_event_count=int(data.get("git_dirty_count", 0) or 0)
                + len(data.get("recent_files", []) or [])
                + len(data.get("log_events", []) or []),
                ambient_summary=_bounded_text(data.get("summary", ""), 220),
                ambient_repair_candidates=repair_candidates,
            )
        )

    def reconcile_foreground_turn(
        self,
        user_message: str,
        *,
        now: float | None = None,
    ) -> TimescaleReconciliation:
        """Return a compact, causal wake-context frame for a user turn."""

        current = float(now if now is not None else time.time())
        idle_gap = max(0.0, current - self._last_user_turn_at) if self._last_user_turn_at else 0.0
        since = current - max(self.idle_anchor_threshold_s * 2.0, 600.0)
        recent = [obs for obs in self._observations if obs.timestamp >= since]

        app_counts = Counter(obs.active_app for obs in recent if obs.active_app)
        window_counts = Counter(obs.window_title for obs in recent if obs.window_title)
        max_memory = max((obs.memory_percent for obs in recent), default=0.0)
        max_cpu = max((obs.cpu_percent for obs in recent), default=0.0)
        max_threat = max((obs.threat for obs in recent), default=0.0)
        max_novelty = max((obs.novelty for obs in recent), default=0.0)
        ambient_recent = [obs for obs in recent if obs.ambient_event_count or obs.ambient_summary]
        ambient_event_count = sum(obs.ambient_event_count for obs in ambient_recent)
        ambient_summaries = tuple(
            obs.ambient_summary for obs in ambient_recent[-3:] if obs.ambient_summary
        )
        ambient_repair_candidates = tuple(
            item
            for obs in ambient_recent[-6:]
            for item in obs.ambient_repair_candidates
        )[:8]
        user_returned = idle_gap >= self.idle_anchor_threshold_s
        low_observation_density = len(recent) < 2
        drift_risk = 0.0
        if user_returned:
            drift_risk = min(1.0, 0.35 + min(0.45, idle_gap / 7200.0))
        if low_observation_density and idle_gap > 0:
            drift_risk = max(drift_risk, 0.45)
        if max_novelty >= 0.45:
            drift_risk = max(0.0, drift_risk - 0.15)

        directives = [
            "Anchor the reply to the user's current message and verified recent conversation.",
            "Do not invent events, emotions, projects, or narrative developments during idle gaps.",
        ]
        if user_returned:
            directives.append("Treat the user turn as a wake reconciliation after idle time.")
        if max_memory >= 80.0 or max_threat >= 0.45:
            directives.append("Prefer compact foreground reasoning and avoid discretionary background expansion.")
        if low_observation_density:
            directives.append("If sensory evidence is sparse, say less about the external world rather than filling gaps.")
        if ambient_event_count:
            directives.append("Use ambient developer stream facts only as verified background evidence, not as invented user intent.")

        apps = tuple(app for app, _ in app_counts.most_common(4))
        windows = tuple(title for title, _ in window_counts.most_common(3))
        summary_parts = []
        if user_returned:
            summary_parts.append(f"user returned after {int(idle_gap)}s idle")
        if apps:
            summary_parts.append("recent apps: " + ", ".join(apps))
        if max_memory or max_cpu:
            summary_parts.append(f"peak load: cpu {max_cpu:.0f}%, memory {max_memory:.0f}%")
        if ambient_summaries:
            summary_parts.append("ambient: " + ambient_summaries[-1])
        if not summary_parts:
            summary_parts.append("no significant idle bridge observations")

        reconciliation = TimescaleReconciliation(
            at=current,
            idle_gap_s=round(idle_gap, 3),
            observations_considered=len(recent),
            observed_apps=apps,
            observed_windows=windows,
            max_memory_percent=round(max_memory, 2),
            max_cpu_percent=round(max_cpu, 2),
            max_threat=round(max_threat, 3),
            max_novelty=round(max_novelty, 3),
            user_returned_after_idle=user_returned,
            narrative_drift_risk=round(drift_risk, 3),
            foreground_anchor_required=bool(user_returned or drift_risk >= 0.35),
            memory_grounding_bias=round(0.35 + min(0.4, drift_risk), 3),
            sensory_grounding_bias=round(0.25 + min(0.45, max_novelty + max_threat), 3),
            ambient_event_count=int(ambient_event_count),
            ambient_summaries=ambient_summaries,
            ambient_repair_candidates=ambient_repair_candidates,
            summary="; ".join(summary_parts),
            directives=tuple(directives),
        )
        self._last_user_turn_at = current
        self._last_reconciliation = reconciliation
        return reconciliation

    def get_status(self) -> dict[str, Any]:
        latest = self._observations[-1].to_dict() if self._observations else None
        return {
            "registered": True,
            "running": True,
            "schema": "aura.timescale_bridge.status.v1",
            "sample_interval_s": self.sample_interval_s,
            "idle_anchor_threshold_s": self.idle_anchor_threshold_s,
            "frames_ingested": self._frames_ingested,
            "observations": len(self._observations),
            "latest_observation": latest,
            "last_reconciliation": (
                self._last_reconciliation.to_dict()
                if self._last_reconciliation is not None
                else None
            ),
            "uptime_s": round(time.time() - self._registered_at, 1),
        }

    status = get_status

    def reset_for_tests(self) -> None:
        self._observations.clear()
        self._last_sample_at = 0.0
        self._last_sample_at_by_source.clear()
        self._last_user_turn_at = 0.0
        self._last_reconciliation = None
        self._frames_ingested = 0


def render_timescale_prompt_block(reconciliation: dict[str, Any] | TimescaleReconciliation) -> str:
    data = reconciliation.to_dict() if isinstance(reconciliation, TimescaleReconciliation) else dict(reconciliation or {})
    if not data:
        return ""
    lines = ["## TIMESCALE RECONCILIATION"]
    summary = str(data.get("summary") or "").strip()
    if summary:
        lines.append(f"- Summary: {summary}")
    lines.append(
        "- Drift guard: "
        f"idle_gap_s={data.get('idle_gap_s', 0)}, "
        f"risk={data.get('narrative_drift_risk', 0)}, "
        f"anchor_required={bool(data.get('foreground_anchor_required'))}"
    )
    directives = data.get("directives") if isinstance(data.get("directives"), list) else []
    for directive in directives[:4]:
        lines.append(f"- {directive}")
    ambient_summaries = data.get("ambient_summaries") if isinstance(data.get("ambient_summaries"), list) else []
    if ambient_summaries:
        lines.append("- Ambient developer stream: " + "; ".join(str(item) for item in ambient_summaries[:2]))
    repair_candidates = (
        data.get("ambient_repair_candidates")
        if isinstance(data.get("ambient_repair_candidates"), list)
        else []
    )
    if repair_candidates:
        lines.append("- Background repair candidates: " + ", ".join(str(item) for item in repair_candidates[:4]))
    lines.append(
        "Use this to reconcile continuous background state with the current user turn; "
        "do not recite it as telemetry."
    )
    return "\n".join(lines)


_TIMESCALE_BRIDGE: TimescaleBridge | None = None


def get_timescale_bridge() -> TimescaleBridge:
    global _TIMESCALE_BRIDGE
    existing = ServiceContainer.get("timescale_bridge", default=None)
    if isinstance(existing, TimescaleBridge):
        _TIMESCALE_BRIDGE = existing
        return existing
    if _TIMESCALE_BRIDGE is None:
        _TIMESCALE_BRIDGE = TimescaleBridge()
    ServiceContainer.register_instance("timescale_bridge", _TIMESCALE_BRIDGE, required=False)
    return _TIMESCALE_BRIDGE


__all__ = [
    "TimescaleBridge",
    "TimescaleObservation",
    "TimescaleReconciliation",
    "get_timescale_bridge",
    "render_timescale_prompt_block",
]
