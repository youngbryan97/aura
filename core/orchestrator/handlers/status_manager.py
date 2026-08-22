import asyncio
import inspect
import logging
import time
from typing import Any

from core.config import config
from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.organism_status import get_organism_status

logger = logging.getLogger(__name__)
_STATUS_MANAGER_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    OSError,
    ConnectionError,
    TimeoutError,
    TypeError,
    ValueError,
    Exception,
)


def _record_status_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation("status_manager", error, severity=severity, action=action)


def capture_and_log(e, meta):
    logger.error("Error in status manager: %s | Meta: %s", e, meta)


def _dispose_awaitable(result: Any) -> None:
    if inspect.iscoroutine(result):
        result.close()
        return
    cancel = getattr(result, "cancel", None)
    if callable(cancel):
        cancel()


def _task_scheduled(result: Any) -> bool:
    return isinstance(result, asyncio.Task) or asyncio.isfuture(result)


def _runtime_health_failure_reason(report: Any, *, limit: int = 4) -> str:
    if not isinstance(report, dict):
        return "runtime health report unavailable"

    details: list[str] = []
    failures = report.get("failures")
    if isinstance(failures, dict):
        for tier in ("critical", "important", "optional"):
            entries = failures.get(tier)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                key = str(entry.get("container_key") or entry.get("name") or "unknown")
                error = str(entry.get("error") or "liveness failed").strip()
                details.append(f"{tier}:{key}:{error[:160]}")
                if len(details) >= max(1, int(limit)):
                    return "; ".join(details)

    blockers = report.get("probe_blockers")
    if not details and isinstance(blockers, list):
        details.extend(str(item)[:160] for item in blockers if str(item).strip())
    if details:
        return "; ".join(details[: max(1, int(limit))])
    return f"runtime health status={str(report.get('status') or 'unknown')[:80]}"


class StatusManagerMixin:
    """Mixin for status reporting and telemetry emission."""

    # Type hints for attributes provided by RobustOrchestrator
    status: Any
    start_time: float
    stats: dict[str, Any]
    message_queue: Any
    reply_queue: Any
    liquid_state: Any
    _integrity_monitor: Any
    acceleration_factor: float
    singularity_threshold: bool

    @staticmethod
    def _summarize_runtime_health_failure(report: Any) -> str:
        return _runtime_health_failure_reason(report)

    def get_status(self) -> dict[str, Any]:
        """Provides a comprehensive status report using cached data where possible."""
        # [STABILITY] Recursion Guard
        if getattr(self, "_in_status_call", False):
            return {"status": "recursive_depth_guard", "healthy": False}
        self._in_status_call = True

        try:
            if not hasattr(self, "_cached_status") or self._cached_status is None:
                self._cached_status = {
                    "status": "operational",
                    "uptime": 0.0,
                    "stats": {},
                    "message_queue_size": 0,
                    "reply_queue_size": 0,
                    "initialized": getattr(self.status, "initialized", False),
                    "running": getattr(self.status, "running", False),
                    "cycle_count": getattr(self.status, "cycle_count", 0),
                    "healthy": True,
                }

            self._cached_status["uptime"] = time.time() - self.start_time
            self._cached_status["stats"] = self.stats.copy()
            self._cached_status["message_queue_size"] = (
                self.message_queue.qsize() if hasattr(self, "message_queue") else 0
            )
            self._cached_status["reply_queue_size"] = (
                self.reply_queue.qsize() if hasattr(self, "reply_queue") else 0
            )

            status_report = self._cached_status.copy()
            status_report["config"] = config.model_dump() if hasattr(config, "model_dump") else {}

            if hasattr(self, "status") and self.status:
                if not isinstance(self.status, type) and hasattr(self.status, "model_dump"):
                    try:
                        # Health check before reporting
                        if hasattr(self, "health_check"):
                            self.health_check()
                        m_dump = self.status.model_dump()
                        status_report.update(m_dump)
                        status_report["status"] = m_dump
                        for key in (
                            "initialized",
                            "running",
                            "cycle_count",
                            "is_processing",
                            "mode",
                            "skills_loaded",
                        ):
                            if key not in status_report["status"]:
                                status_report["status"][key] = getattr(
                                    self.status,
                                    key,
                                    True
                                    if key in ("initialized", "running")
                                    else (
                                        0
                                        if key == "cycle_count"
                                        else (
                                            False
                                            if key == "is_processing"
                                            else (0 if key == "skills_loaded" else "neutral")
                                        )
                                    ),
                                )

                        status_report["initialized"] = status_report["status"]["initialized"]
                        status_report["cycle_count"] = getattr(
                            self.status,
                            "cycle_count",
                            status_report["status"].get("cycle_count", 0),
                        )
                    except _STATUS_MANAGER_ERRORS as e:
                        _record_status_degradation(
                            e,
                            action="returned cached/default status fields after status model dump failed",
                        )
                        capture_and_log(e, {"module": __name__})
                else:
                    status_report["running"] = bool(getattr(self.status, "running", True))
                    status_report["initialized"] = bool(getattr(self.status, "initialized", True))
                    status_report["cycle_count"] = int(getattr(self.status, "cycle_count", 0))
                    raw_sk = getattr(self.status, "skills_loaded", 0)
                    status_report["status"] = {
                        "running": status_report["running"],
                        "initialized": status_report["initialized"],
                        "cycle_count": status_report["cycle_count"],
                        "is_processing": bool(getattr(self.status, "is_processing", False)),
                        "mode": getattr(self.status, "mode", "neutral"),
                        "skills_loaded": raw_sk,
                    }
                    if hasattr(self, "health_check"):
                        status_report["healthy"] = self.health_check()

            try:
                evidence = ServiceContainer.get("consciousness_evidence", default=None)
                if evidence and hasattr(evidence, "snapshot"):
                    status_report["consciousness_evidence"] = evidence.snapshot()
            except _STATUS_MANAGER_ERRORS as exc:
                _record_status_degradation(
                    exc,
                    action="returned status report without consciousness-evidence section",
                )
                logger.debug("Consciousness evidence unavailable for status: %s", exc)

            try:
                executive_closure = ServiceContainer.get("executive_closure", default=None)
                if executive_closure and hasattr(executive_closure, "get_status"):
                    status_report["executive_closure"] = executive_closure.get_status()
            except _STATUS_MANAGER_ERRORS as exc:
                _record_status_degradation(
                    exc,
                    action="returned status report without executive-closure section",
                )
                logger.debug("Executive closure unavailable for status: %s", exc)

            try:
                executive_authority = ServiceContainer.get("executive_authority", default=None)
                if executive_authority and hasattr(executive_authority, "get_status"):
                    status_report["executive_authority"] = executive_authority.get_status()
            except _STATUS_MANAGER_ERRORS as exc:
                _record_status_degradation(
                    exc,
                    action="returned status report without executive-authority section",
                )
                logger.debug("Executive authority unavailable for status: %s", exc)

            try:
                status_report["organism"] = get_organism_status(self)
            except _STATUS_MANAGER_ERRORS as exc:
                _record_status_degradation(
                    exc,
                    action="returned status report without organism section",
                )
                logger.debug("Organism status unavailable for status report: %s", exc)

            voice_task = getattr(self, "_voice_listener_task", None)
            status_report["voice_listener"] = {
                "state": str(getattr(self, "_voice_listener_state", "unknown")),
                "ready": bool(getattr(self, "_voice_listener_ready", False)),
                "error": str(getattr(self, "_voice_listener_error", "")),
                "startup_in_flight": bool(voice_task is not None and not voice_task.done()),
            }
            status_report["health_phase"] = str(
                getattr(self, "_last_runtime_health_phase", "unknown") or "unknown"
            )
            status_report["health_reason"] = str(
                getattr(self, "_last_health_reason", "") or ""
            )

            return status_report

        finally:
            self._in_status_call = False

    async def _warm_language_matchers(self) -> None:
        """Decide deferred phrasings only after the foreground is truly idle.

        A live turn answers from what it already knows and queues anything
        new.  A resident-model hidden read is still real 32B inference even
        when it runs in a thread, so moving it off the event loop is not enough:
        it also needs the organism's shared background admission contract.
        """
        import asyncio

        try:
            # The evidence router's embedding model, loaded here or nowhere.
            #
            # Deciding whether a turn needs a camera reading ends in a
            # semantic routing question, and asking it loaded the embedding
            # model in the foreground on the first turn after a restart. A
            # turn now answers from the lexical floor when the model is cold;
            # this is where it stops being cold.
            # It is a small encoder, not the resident cortex, so it does not
            # wait for the blocker that protects the 32B.
            from core.cognition.evidence_relevance import (
                semantic_routing_ready,
                warm_semantic_routing,
            )

            if not semantic_routing_ready():
                if await asyncio.to_thread(warm_semantic_routing):
                    logger.info("🧭 Semantic evidence routing is warm.")

            from core.runtime.background_policy import (
                IDLE_COGNITION_BACKGROUND_POLICY,
                background_activity_reason,
            )

            blocker = background_activity_reason(
                self,
                profile=IDLE_COGNITION_BACKGROUND_POLICY,
                require_conversation_ready=True,
            )
            if blocker:
                logger.debug("Language matcher warm deferred: %s", blocker)
                return

            from core.conversation.response_reliability import warm_language_matchers

            # One pending phrasing per surface. model_hidden_features yields
            # worker ownership between every sentence, so a foreground owner
            # arriving during this warm cycle takes the next slot.
            settled = await asyncio.to_thread(warm_language_matchers, 1)
            if settled:
                logger.info("🧠 Settled %d new phrasing(s) from use.", settled)
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            from core.runtime.errors import record_degradation

            record_degradation(
                "orchestrator.language_matcher_warm",
                exc,
                severity="debug",
                action="left the phrasings for the next tick",
                enforce_failure_policy=False,
            )

    def _emit_telemetry_pulse(self):
        """Emit real-time liquid state telemetry."""
        try:
            ls = getattr(self, "liquid_state", None)
            if ls:
                ls_status = ls.get_status()
                monitor_stats = (
                    self._integrity_monitor.get_stats()
                    if hasattr(self, "_integrity_monitor")
                    else {}
                )

                if hasattr(self, "_publish_telemetry"):
                    self._publish_telemetry(
                        {
                            "energy": ls_status.get("energy", 80),
                            "curiosity": ls_status.get("curiosity", 50),
                            "frustration": ls_status.get("frustration", 0),
                            "confidence": ls_status.get("focus", 50),
                            "mood": ls_status.get("mood", "NEUTRAL"),
                            "acceleration_factor": getattr(self.status, "acceleration_factor", 1.0),
                            "singularity_active": getattr(
                                self.status, "singularity_threshold", 0.0
                            ),
                            "cpu_percent": monitor_stats.get("cpu_percent", 0),
                            "memory_mb": monitor_stats.get("memory_mb", 0),
                            "link_thickness": 5.0,
                        }
                    )
        except _STATUS_MANAGER_ERRORS as exc:
            _record_status_degradation(
                exc,
                action="scheduled stall recovery after telemetry pulse failed",
                severity="error",
            )
            logger.error("Telemetry pulse failure: %s", exc)
            if hasattr(self, "_recover_from_stall"):
                from core.utils.task_tracker import get_task_tracker

                recovery_coro = self._recover_from_stall()
                try:
                    recovery_task = get_task_tracker().track(
                        recovery_coro, name="recover_from_stall"
                    )
                except _STATUS_MANAGER_ERRORS as recovery_exc:
                    _dispose_awaitable(recovery_coro)
                    _record_status_degradation(
                        recovery_exc,
                        action="skipped telemetry stall recovery after task tracker rejected recovery coroutine",
                        severity="error",
                    )
                    logger.debug("Telemetry stall recovery scheduling failed: %s", recovery_exc)
                else:
                    if not _task_scheduled(recovery_task):
                        _dispose_awaitable(recovery_coro)
                        _dispose_awaitable(recovery_task)

    def _emit_telemetry(self, flow: str, text: str):
        """Helper to send updates to Thought Stream UI."""
        try:
            from ...thought_stream import get_emitter

            cycle = self.status.cycle_count if hasattr(self, "status") else 0
            get_emitter().emit(flow, text, level="info", category="Cognition", cycle=cycle)
        except _STATUS_MANAGER_ERRORS as e:
            _record_status_degradation(
                e,
                action="dropped cognition telemetry thought-stream update",
            )
            logger.debug("Telemetry emit failed: %s", e)
