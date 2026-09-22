"""core/observability/incident_narrator.py — Aura explains her own incidents.

The forensics already exist — stall dumps, degraded events, worker lifecycle
receipts, the memory sentinel ring, boot phase timings — but they are scattered
files and in-memory rings that only a human with an hour and grep can read.
This organ ingests all of them, correlates them into episodes, and produces a
causal, receipt-backed narrative that answers "what happened and why" —

  * for Bryan, via /api/system/incidents;
  * for Aura herself, injected into the conversation lane when the user asks
    why she was slow, stuck, or restarted (grounded self-knowledge instead of
    confabulation).

Everything here is deterministic synthesis over real evidence. No LLM writes
these facts; the cortex may rephrase them, but every claim carries a receipt
(file path, event id, timestamp) a human can check.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.IncidentNarrator")

# Evidence closer together than this belongs to one episode.
_EPISODE_GAP_S = 90.0
# Stack frames that mean "idle worker thread", not a culprit.
_IDLE_FRAME_MARKERS = (
    "threading.py",
    "concurrent/futures",
    "runtime_hygiene.py",
    "socketserver.py",
    "selectors.py",
)
_SEVERITY_RANK = {"info": 0, "warning": 1, "degraded": 2, "error": 3, "critical": 4}


@dataclass
class EvidenceItem:
    """One normalized, receipt-backed observation."""

    at: float
    source: str          # stall_dump | degraded_event | memory_sentinel | boot_profile | incident_manager | log_transport
    kind: str            # e.g. event_loop_stall, token_progress_stalled, process_restart
    severity: str        # info | warning | degraded | error | critical
    summary: str         # one plain-language sentence
    receipt: str         # file path / event id / timestamp a human can check
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "source": self.source,
            "kind": self.kind,
            "severity": self.severity,
            "summary": self.summary,
            "receipt": self.receipt,
            "detail": self.detail,
        }


@dataclass
class Episode:
    """A cluster of temporally adjacent evidence, narrated."""

    started_at: float
    ended_at: float
    severity: str
    headline: str
    narrative: str
    evidence: list[EvidenceItem]

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "severity": self.severity,
            "headline": self.headline,
            "narrative": self.narrative,
            "evidence": [item.to_dict() for item in self.evidence],
        }


class IncidentNarrator:
    """Deterministic incident synthesis over Aura's own forensics."""

    def __init__(
        self,
        error_log_root: Path | None = None,
        boot_profile_path: Path | None = None,
    ) -> None:
        # An EXPLICIT error_log_root is a forensic sandbox (tests, replay):
        # every filesystem source must live under it, or the narrator leaks
        # the real machine's history into the sandbox (a boot profile from
        # the live repo once made a hermetic empty-window test narrate an
        # actual 07:18 boot).
        self._sandboxed = error_log_root is not None
        if error_log_root is not None:
            self._error_log_root = Path(error_log_root)
            self._boot_profile_path = (
                Path(boot_profile_path)
                if boot_profile_path is not None
                else self._error_log_root / "boot_profile.json"
            )
        else:
            from core.utils.paths import forensics_root

            self._error_log_root = forensics_root()
            self._boot_profile_path = (
                Path(boot_profile_path)
                if boot_profile_path is not None
                else Path("artifacts/current/boot_profile.json")
            )
        self._started = False
        # The conversation lane calls get_context_injection synchronously;
        # evidence collection reads bounded forensic files. Cache the rendered
        # block briefly so repeated incident questions cost one read burst,
        # honoring the no-blocking-work-on-the-turn-path discipline.
        self._injection_cache: str | None = None
        self._injection_cache_at = 0.0
        self._injection_cache_ttl_s = 10.0

    async def start(self) -> None:
        if self._started:
            return
        from core.container import ServiceContainer

        ServiceContainer.register_instance("incident_narrator", self, required=False)
        self._started = True
        logger.info("IncidentNarrator ONLINE — Aura can explain her own incidents.")

    # ── evidence collection ────────────────────────────────────────────

    def collect_window(self, minutes: float = 60.0) -> list[EvidenceItem]:
        cutoff = time.time() - minutes * 60.0
        items: list[EvidenceItem] = []
        for collector in (
            self._collect_stall_dumps,
            self._collect_degraded_events,
            self._collect_memory_sentinel,
            self._collect_boot_profile,
            self._collect_incident_manager,
            self._collect_log_transport,
            self._collect_flight_recorder,
            self._collect_triage_classes,
            self._collect_component_conditions,
        ):
            try:
                items.extend(collector(cutoff))
            except (OSError, RuntimeError, ValueError, TypeError, AttributeError, ImportError) as exc:
                record_degradation(
                    "incident_narrator",
                    exc,
                    severity="warning",
                    action=f"continued narration without {collector.__name__} evidence",
                )
        items.sort(key=lambda item: item.at)
        return items

    def _collect_stall_dumps(self, cutoff: float) -> list[EvidenceItem]:
        stall_dir = self._error_log_root / "stalls"
        if not stall_dir.is_dir():
            return []
        items: list[EvidenceItem] = []
        for path in stall_dir.glob("stall_*.txt"):
            match = re.search(r"stall_(\d+)", path.name)
            if not match:
                continue
            at = float(match.group(1))
            if at < cutoff:
                continue
            elapsed, culprit = self._parse_stall_dump(path)
            items.append(
                EvidenceItem(
                    at=at,
                    source="stall_dump",
                    kind="event_loop_stall",
                    severity="error",
                    summary=(
                        f"The event loop froze for {elapsed:.1f}s"
                        + (f" inside {culprit}" if culprit else "")
                        + " — everything (chat, streams, health) paused with it."
                    ),
                    receipt=str(path),
                    detail={"elapsed_s": elapsed, "culprit_frame": culprit},
                )
            )
        return items

    @staticmethod
    def _parse_stall_dump(path: Path) -> tuple[float, str]:
        """Return (elapsed_seconds, attributed culprit frame).

        Attribution is delegated to the shared loop-thread-aware parser —
        the narrator must tell the same story triage does, and neither may
        blame a parked bystander thread (the old "last project frame"
        scan once narrated a sleeping monitor loop as the freeze culprit).
        """
        from core.observability.stall_dump import parse_stall_dump

        verdict = parse_stall_dump(path)
        return verdict.elapsed_s, verdict.described()

    @staticmethod
    def _collect_degraded_events(cutoff: float) -> list[EvidenceItem]:
        from core.health.degraded_events import get_recent_degraded_events

        items: list[EvidenceItem] = []
        for event in get_recent_degraded_events(limit=200):
            at = float(event.get("last_seen") or event.get("timestamp") or 0.0)
            if at < cutoff:
                continue
            kind = str(event.get("reason", "") or "unknown")
            severity = str(event.get("severity", "warning") or "warning")
            subsystem = str(event.get("subsystem", "") or "")
            detail = str(event.get("detail", "") or "")
            count = int(event.get("count", 1) or 1)
            summary = _describe_degraded_event(kind, subsystem, detail)
            if count > 1:
                summary = f"{summary[:-1]} — happened {count} times."
            items.append(
                EvidenceItem(
                    at=at,
                    source="degraded_event",
                    kind=kind,
                    severity=severity,
                    summary=summary,
                    receipt=f"degraded_event:{subsystem}:{kind}@{at:.0f}",
                    detail={"subsystem": subsystem, "detail": detail, "count": count},
                )
            )
        return items

    def _collect_memory_sentinel(self, cutoff: float) -> list[EvidenceItem]:
        """Process restarts + memory trajectory from the sentinel log."""
        sentinel_log = self._error_log_root / "memory" / "sentinel.log"
        if not sentinel_log.is_file():
            return []
        items: list[EvidenceItem] = []
        try:
            lines = sentinel_log.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
        except OSError:
            return []
        for line in lines:
            ts_match = re.match(r"\[([0-9T:+\-]+)\]", line)
            if not ts_match:
                continue
            at = _parse_sentinel_timestamp(ts_match.group(1))
            if at is None or at < cutoff:
                continue
            if "vanished; capturing death syslog" in line:
                items.append(
                    EvidenceItem(
                        at=at,
                        source="memory_sentinel",
                        kind="process_exit",
                        severity="error",
                        summary="The main process exited (sentinel captured a death syslog).",
                        receipt=str(sentinel_log),
                        detail={"line": line.strip()[:300]},
                    )
                )
            elif "armed: target pid=" in line:
                items.append(
                    EvidenceItem(
                        at=at,
                        source="memory_sentinel",
                        kind="process_start",
                        severity="info",
                        summary="A fresh main process came up (sentinel re-armed).",
                        receipt=str(sentinel_log),
                        detail={"line": line.strip()[:300]},
                    )
                )
        return items

    def _collect_boot_profile(self, cutoff: float) -> list[EvidenceItem]:
        path = self._boot_profile_path
        if not path.is_file():
            return []
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        started = float(report.get("started_at_unix", 0.0) or 0.0)
        if started < cutoff:
            return []
        total_s = float(report.get("total_s", 0.0) or 0.0)
        phases = sorted(
            (report.get("phases") or []),
            key=lambda p: float(p.get("duration_s", 0.0) or 0.0),
            reverse=True,
        )
        slowest = phases[0] if phases else {}
        return [
            EvidenceItem(
                at=started,
                source="boot_profile",
                kind="boot",
                severity="info",
                summary=(
                    f"Boot completed in {total_s:.1f}s; slowest phase was "
                    f"{slowest.get('name', 'unknown')} at {float(slowest.get('duration_s', 0.0) or 0.0):.1f}s."
                ),
                receipt=str(path),
                detail={"total_s": total_s, "slowest_phase": slowest},
            )
        ]

    @staticmethod
    def _collect_incident_manager(cutoff: float) -> list[EvidenceItem]:
        from core.container import ServiceContainer

        manager = ServiceContainer.get("incident_manager", default=None)
        if manager is None or not hasattr(manager, "get_active"):
            return []
        items: list[EvidenceItem] = []
        for incident in manager.get_active():
            at = float(incident.get("created_at", 0.0) or 0.0)
            if at and at < cutoff:
                continue
            items.append(
                EvidenceItem(
                    at=at or time.time(),
                    source="incident_manager",
                    kind=str(incident.get("category", "incident")),
                    severity=str(incident.get("severity", "error")),
                    summary=f"Open incident: {incident.get('description', incident.get('category', 'unknown'))}",
                    receipt=str(incident.get("incident_id", "")),
                    detail=dict(incident),
                )
            )
        return items

    def _collect_flight_recorder(self, cutoff: float) -> list[EvidenceItem]:
        """Death reports extracted from the black-box flight ring — the
        ground truth for hard deaths (SIGKILL, OOM-kill, segfault)."""
        flight_dir = self._error_log_root / "flight"
        if not flight_dir.is_dir():
            return []
        items: list[EvidenceItem] = []
        for path in sorted(flight_dir.glob("death_*.json"))[-10:]:
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(report, dict):
                continue
            at = float(report.get("died_at") or report.get("generated_at") or 0.0)
            if at < cutoff:
                continue
            items.append(
                EvidenceItem(
                    at=at,
                    source="flight_recorder",
                    kind="unclean_shutdown",
                    severity="critical",
                    summary=str(
                        report.get("narrative")
                        or "The previous run ended without a clean shutdown."
                    ),
                    receipt=str(path),
                    detail={
                        key: report.get(key)
                        for key in (
                            "previous_boot_id",
                            "uptime_s",
                            "final_tick",
                            "final_stage",
                            "final_rss_mb",
                            "rss_delta_final_minute_mb",
                            "frames_recovered",
                        )
                    },
                )
            )
        return items

    @staticmethod
    def _collect_log_transport(cutoff: float) -> list[EvidenceItem]:
        """Dropped-log pressure from the non-blocking transport."""
        try:
            from core.observability.logging_config import get_dropped_log_count
        except ImportError:
            return []
        dropped = int(get_dropped_log_count())
        if dropped <= 0:
            return []
        return [
            EvidenceItem(
                at=time.time(),
                source="log_transport",
                kind="log_records_dropped",
                severity="warning",
                summary=(
                    f"{dropped} log records were dropped under overflow — the system was "
                    "logging faster than the sinks could drain."
                ),
                receipt="core.observability.logging_config.get_dropped_log_count()",
                detail={"dropped": dropped},
            )
        ]

    def _collect_triage_classes(self, cutoff: float) -> list[EvidenceItem]:
        """Fingerprinted incident classes from the crash-triage pipeline (C2).

        `make triage` writes artifacts/reliability/triage.json; the narrator
        consumes the CLASSES so an episode can say "this is the 14th
        occurrence of a known stall fingerprint", not just describe one
        dump. Sandboxed runs read triage.json under the explicit root.
        """
        # Ask the flag, not the path. Comparing the root against a literal
        # relative path meant "am I sandboxed?" silently answered yes the
        # moment the default root stopped being that literal.
        if self._sandboxed:
            triage_path = self._error_log_root / "triage.json"
        else:
            triage_path = Path("artifacts/reliability/triage.json")
        if not triage_path.is_file():
            return []
        import json as _json

        try:
            report = _json.loads(triage_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        items: list[EvidenceItem] = []
        severity_by_kind = {
            "process_death": "critical",
            "fatal_error": "error",
            "stall": "warning",
            "memory_spike": "warning",
            "orderly_exit": "info",
        }
        for cls in report.get("classes", []):
            if not isinstance(cls, dict):
                continue
            last_seen = float(cls.get("last_seen", 0.0) or 0.0)
            if last_seen < cutoff:
                continue
            kind = str(cls.get("kind", "unknown"))
            count = int(cls.get("count", 0) or 0)
            items.append(
                EvidenceItem(
                    at=last_seen,
                    source="crash_triage",
                    kind=f"triage_class_{kind}",
                    severity=severity_by_kind.get(kind, "warning"),
                    summary=(
                        f"Known incident class '{cls.get('fingerprint', kind)}' "
                        f"({kind}) has {count} occurrence(s) in the triage window."
                    ),
                    receipt=str(triage_path),
                    detail={"fingerprint": cls.get("fingerprint"), "count": count},
                )
            )
        trend = report.get("trend") or {}
        for fingerprint in trend.get("new_classes", []):
            items.append(
                EvidenceItem(
                    at=float(report.get("generated_at", time.time()) or time.time()),
                    source="crash_triage",
                    kind="triage_new_class",
                    severity="error",
                    summary=(
                        f"NEW incident class appeared since the previous triage run: "
                        f"'{fingerprint}'."
                    ),
                    receipt=str(triage_path),
                    detail={"fingerprint": fingerprint},
                )
            )
        return items

    @staticmethod
    def _collect_component_conditions(cutoff: float) -> list[EvidenceItem]:
        """K6 typed conditions: currently-degraded components and recent
        transitions become first-class evidence."""
        try:
            from core.runtime.conditions import all_conditions_report
        except ImportError:
            return []
        items: list[EvidenceItem] = []
        for component, conditions in all_conditions_report().items():
            for payload in conditions.values():
                status = bool(payload.get("status", False))
                kind = str(payload.get("type", ""))
                transitioned = float(payload.get("last_transition_at", 0.0) or 0.0)
                is_bad = (kind == "Degraded" and status) or (
                    kind == "Ready" and not status
                )
                if not is_bad or transitioned < cutoff:
                    continue
                items.append(
                    EvidenceItem(
                        at=transitioned,
                        source="conditions",
                        kind=f"condition_{kind.lower()}_{str(status).lower()}",
                        severity="degraded" if kind == "Ready" else "warning",
                        summary=(
                            f"Component '{component}' condition {kind}={status} "
                            f"({payload.get('reason', 'no reason')}): "
                            f"{payload.get('message', '') or 'no detail'}"
                        ),
                        receipt=f"core.runtime.conditions:{component}/{kind}",
                        detail=dict(payload),
                    )
                )
        return items

    # ── correlation + narration ────────────────────────────────────────

    def narrate(self, minutes: float = 60.0, max_episodes: int = 8) -> dict[str, Any]:
        evidence = self.collect_window(minutes=minutes)
        episodes = self._correlate(evidence)
        episodes.sort(key=lambda ep: (_SEVERITY_RANK.get(ep.severity, 0), ep.ended_at), reverse=True)
        episodes = episodes[:max_episodes]
        return {
            "schema": "aura.incident_narrative.v1",
            "generated_at": time.time(),
            "window_minutes": minutes,
            "evidence_count": len(evidence),
            "episode_count": len(episodes),
            "healthy": not any(
                _SEVERITY_RANK.get(ep.severity, 0) >= _SEVERITY_RANK["error"] for ep in episodes
            ),
            "episodes": [ep.to_dict() for ep in episodes],
        }

    def _correlate(self, evidence: list[EvidenceItem]) -> list[Episode]:
        episodes: list[Episode] = []
        cluster: list[EvidenceItem] = []
        for item in evidence:
            if cluster and item.at - cluster[-1].at > _EPISODE_GAP_S:
                episodes.append(self._narrate_cluster(cluster))
                cluster = []
            cluster.append(item)
        if cluster:
            episodes.append(self._narrate_cluster(cluster))
        return episodes

    def _narrate_cluster(self, cluster: list[EvidenceItem]) -> Episode:
        severity = max(
            (item.severity for item in cluster),
            key=lambda s: _SEVERITY_RANK.get(s, 0),
        )
        headline = self._headline_for(cluster)
        lines = [headline]
        for item in cluster:
            stamp = time.strftime("%H:%M:%S", time.localtime(item.at))
            lines.append(f"- [{stamp}] {item.summary} (receipt: {item.receipt})")
        causal = self._causal_reading(cluster)
        if causal:
            lines.append(f"Reading: {causal}")
        return Episode(
            started_at=cluster[0].at,
            ended_at=cluster[-1].at,
            severity=severity,
            headline=headline,
            narrative="\n".join(lines),
            evidence=list(cluster),
        )

    @staticmethod
    def _headline_for(cluster: list[EvidenceItem]) -> str:
        kinds = [item.kind for item in cluster]
        when = time.strftime("%H:%M", time.localtime(cluster[0].at))
        if "unclean_shutdown" in kinds:
            return f"{when} — a hard death: the runtime went down without a clean shutdown."
        if "process_exit" in kinds:
            return f"{when} — the main process went down and came back."
        if "event_loop_stall" in kinds:
            worst = max(
                (item for item in cluster if item.kind == "event_loop_stall"),
                key=lambda item: float(item.detail.get("elapsed_s", 0.0) or 0.0),
            )
            return (
                f"{when} — the event loop froze "
                f"({float(worst.detail.get('elapsed_s', 0.0) or 0.0):.1f}s at worst)."
            )
        if any(k in kinds for k in ("token_progress_stalled", "generation_deadline_reached", "first_token_sla_exceeded")):
            return f"{when} — a model generation stalled or overran its budget."
        if "warm_lane_preserved_after_soft_cancel" in kinds:
            return f"{when} — a slow generation was cancelled cooperatively; the model stayed warm."
        if "boot" in kinds:
            return f"{when} — a normal boot."
        return f"{when} — {len(cluster)} correlated events."

    @staticmethod
    def _causal_reading(cluster: list[EvidenceItem]) -> str:
        kinds = {item.kind for item in cluster}
        if "event_loop_stall" in kinds:
            culprits = [
                item.detail.get("culprit_frame", "")
                for item in cluster
                if item.kind == "event_loop_stall" and item.detail.get("culprit_frame")
            ]
            if culprits:
                return (
                    f"blocking work ran on the event loop ({', '.join(sorted(set(culprits)))}); "
                    "while it ran, every stream and turn waited."
                )
        if "token_progress_stalled" in kinds and "warm_lane_preserved_after_soft_cancel" in kinds:
            return (
                "a generation stopped making token progress, the worker acknowledged the "
                "soft-cancel, and the warm model was preserved — no reload was paid."
            )
        if {"token_progress_stalled", "process_start"} <= kinds or {"generation_deadline_reached", "process_start"} <= kinds:
            return (
                "a stalled generation escalated to a worker recycle; requests during the "
                "model reload window were the visible casualties."
            )
        if "unclean_shutdown" in kinds:
            return (
                "the black-box flight ring carried no clean-shutdown marker — the "
                "process was killed or crashed rather than stopping itself; its final "
                "recorded moments are preserved in the death report."
            )
        if "process_exit" in kinds and "process_start" in kinds:
            return "the process exited and the supervisor brought a fresh one up."
        if "log_records_dropped" in kinds:
            return "log volume exceeded sink throughput; oldest records were shed to protect latency."
        return ""

    # ── conversation-lane surface ──────────────────────────────────────

    def get_context_injection(self, objective: str = "") -> str:
        """A bounded, receipt-backed block for the conversation lane.

        Only fires when the user is actually asking about slowness, failures,
        restarts, or system state — grounded self-knowledge on demand, not
        noise on every turn.
        """
        if not _asks_about_incidents(objective):
            return ""
        now = time.monotonic()
        if (
            self._injection_cache is not None
            and now - self._injection_cache_at < self._injection_cache_ttl_s
        ):
            return self._injection_cache
        report = self.narrate(minutes=60.0, max_episodes=3)
        if not report["episodes"]:
            block = (
                "## SYSTEM INCIDENT SELF-KNOWLEDGE\n"
                "No stalls, crashes, or degradations in the last hour — if something felt "
                "slow, it did not register on the forensic record; say so honestly."
            )
        else:
            lines = [
                "## SYSTEM INCIDENT SELF-KNOWLEDGE (real forensics — cite honestly, do not invent)"
            ]
            for episode in report["episodes"]:
                lines.append(episode["narrative"])
            lines.append(
                "Ground any explanation of slowness or failure in the receipts above. If the "
                "user's experience isn't covered by them, say you don't have evidence for it."
            )
            block = "\n".join(lines)
        self._injection_cache = block
        self._injection_cache_at = now
        return block


_INCIDENT_QUESTION_MARKERS = (
    "why were you slow",
    "why was that slow",
    "why so slow",
    "what happened",
    "did you crash",
    "did something break",
    "why did you restart",
    "were you down",
    "what went wrong",
    "why are you lagging",
    "system status",
    "health report",
    "incident",
)


def _asks_about_incidents(objective: str) -> bool:
    lowered = (objective or "").lower()
    return any(marker in lowered for marker in _INCIDENT_QUESTION_MARKERS)


def _describe_degraded_event(kind: str, subsystem: str, detail: str) -> str:
    friendly = {
        "token_progress_stalled": "A generation stopped producing tokens",
        "first_token_sla_exceeded": "A generation missed its first-token deadline",
        "generation_deadline_reached": "A generation ran past its overall budget",
        "heartbeat_stalled_during_generation": "The model worker stopped heartbeating mid-generation",
        "warm_lane_preserved_after_soft_cancel": "A cancelled generation was dropped cooperatively; the warm model was preserved",
        "generation_cancelled": "A generation was cancelled",
        "metabolic_self_preservation_block": "A heavy tool was blocked to protect the substrate",
    }
    base = friendly.get(kind, f"{subsystem or 'a subsystem'} reported {kind}")
    if detail:
        return f"{base} ({detail[:120]})."
    return f"{base}."


def _parse_sentinel_timestamp(raw: str) -> float | None:
    from datetime import datetime

    try:
        return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S%z").timestamp()
    except ValueError as exc:
        logger.debug("the sentinel timestamp did not parse (%s: %s)", type(exc).__name__, exc)
        return None


_narrator: IncidentNarrator | None = None


def get_incident_narrator() -> IncidentNarrator:
    global _narrator
    if _narrator is None:
        _narrator = IncidentNarrator()
    return _narrator


__all__ = [
    "EvidenceItem",
    "Episode",
    "IncidentNarrator",
    "get_incident_narrator",
]
