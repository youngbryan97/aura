import asyncio
import functools
import logging
import os
import time
from typing import Any

from core.container import ServiceContainer as ServiceContainer  # noqa: F401
from core.health.degraded_events import get_unified_failure_state, record_degraded_event
from core.runtime import resource_psutil as psutil
from core.runtime.background_policy import background_activity_allowed, background_activity_reason
from core.runtime.errors import FallbackClassification, record_degradation
from core.runtime.service_access import (
    optional_service,
    resolve_orchestrator,
    resolve_state_repository,
)
from core.utils.task_tracker import task_tracker

logger = logging.getLogger("Aura.Initiative")


_INITIATIVE_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)
_STOP_TIMEOUT_S = 5.0
_MAX_MISSION_ADVANCES_PER_CYCLE = 3


def _record_initiative_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "autonomous_initiative_loop",
        exc,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


def _background_initiative_allowed(orchestrator=None) -> bool:
    return background_activity_allowed(
        orchestrator,
        min_idle_seconds=30.0,
        max_memory_percent=80.0,
        max_failure_pressure=0.12,
        # Passive autonomy such as RSS watching, inbox checks, and social
        # browsing should not disappear just because the foreground chat lane is
        # cold, warming, or recovering. These activities still obey idle,
        # memory, and failure-pressure gates.
        require_conversation_ready=False,
    )


def _background_initiative_blocker(orchestrator=None) -> str:
    return background_activity_reason(
        orchestrator,
        min_idle_seconds=30.0,
        max_memory_percent=80.0,
        max_failure_pressure=0.12,
        require_conversation_ready=False,
    )


def _self_development_allowed(orchestrator=None) -> bool:
    return background_activity_allowed(
        orchestrator,
        min_idle_seconds=45.0,
        max_memory_percent=82.0,
        max_failure_pressure=0.15,
        require_conversation_ready=False,
    )


def _self_development_blocker(orchestrator=None) -> str:
    return background_activity_reason(
        orchestrator,
        min_idle_seconds=45.0,
        max_memory_percent=82.0,
        max_failure_pressure=0.15,
        require_conversation_ready=False,
    )


def _passive_social_allowed(orchestrator=None) -> bool:
    return background_activity_allowed(
        orchestrator,
        min_idle_seconds=120.0,
        max_memory_percent=78.0,
        max_failure_pressure=0.15,
        require_conversation_ready=False,
        allow_no_user_anchor=True,
    )


def _passive_social_blocker(orchestrator=None) -> str:
    return background_activity_reason(
        orchestrator,
        min_idle_seconds=120.0,
        max_memory_percent=78.0,
        max_failure_pressure=0.15,
        require_conversation_ready=False,
        allow_no_user_anchor=True,
    )


def _self_development_visible_updates_enabled(orchestrator=None) -> bool:
    """Visible self-dev narration is opt-in; neural stream remains the default."""
    if orchestrator is not None:
        explicit = getattr(orchestrator, "_surface_self_development_updates", None)
        if explicit is not None:
            return bool(explicit)

    raw = os.getenv("AURA_SURFACE_SELF_DEVELOPMENT_UPDATES", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# What to assume when the body cannot be read. This loop decides whether to
# act UNPROMPTED, so an unreadable state is a reason to hold: low energy and
# elevated pressure, rather than the permissive full-energy/zero-pressure
# defaults that previously licensed initiative on a failed read
# (CP126 b53bc839). Not 0/1 — a permanently unreadable sensor should damp
# initiative, not abolish it.
_UNREADABLE_ENERGY = 0.35
_UNREADABLE_PRESSURE = 0.6


class AutonomousInitiativeLoop:
    """
    Unprompted world-watching, knowledge-gap monitoring, and topic generation.
    Ensures Aura maintains a persistent 'lived experience' 24/7.

    Stability fixes:
    - Tasks tracked via task_tracker (no orphaned loops on shutdown)
    - Autonomous thoughts use emit_spontaneous_message (not process_user_input)
      to avoid poisoning conversation history with fake user messages
    """

    name = "autonomous_initiative_loop"

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator or resolve_orchestrator(default=None)
        self.running = False
        self.rss_feeds = [
            "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
            "https://feeds.arstechnica.com/arstechnica/index",
            "https://www.theverge.com/rss/index.xml",
            "https://hnrss.org/frontpage",
        ]
        self._seen_titles: set[str] = set()
        self._last_news_check = 0
        self._world_task = None
        self._knowledge_task = None
        self._event_task = None
        self._self_dev_task = None
        self._social_task = None
        self._mission_task = None
        self._discovery_task = None
        self._supervisor_task = None
        self._child_restart_state: dict[str, dict[str, float]] = {}
        self._last_self_dev = 0.0
        self._last_discovery = 0.0
        self._last_email_check = 0.0
        self._last_reddit_check = 0.0
        self._recent_email_uids: dict[str, float] = {}
        self._recent_reddit_urls: dict[str, float] = {}
        self._covenant_announced: dict[str, float] = {}
        self._lifecycle_lock = asyncio.Lock()

    async def start(self):
        """Starts the initiative loops and returns a boot receipt."""
        async with self._lifecycle_lock:
            if (
                self.running
                and all(self._task_alive(task) for task in self._core_tasks())
                and self._task_alive(self._supervisor_task)
            ):
                return {
                    "ok": True,
                    "already_running": True,
                    "core_tasks": self._task_status(),
                    "event_subscription": self._task_alive(self._event_task),
                }

            self.running = True
            logger.info(
                "✅ AutonomousInitiativeLoop ACTIVE - Monitoring global events and knowledge gaps."
            )

            self._spawn_missing_core_tasks()
            if not self._task_alive(self._supervisor_task):
                self._supervisor_task = task_tracker.create_task(
                    self._initiative_supervisor_loop(),
                    name="InitiativeChildSupervisor",
                )

            status = {
                "ok": True,
                "already_running": False,
                "core_tasks": self._task_status(),
                "event_subscription": False,
            }

            try:
                from core.service_names import ServiceNames

                bus = optional_service(ServiceNames.EVENT_BUS, default=None)
                if bus:
                    queue = await bus.subscribe("aura.proactive.initiation")
                    self._event_task = task_tracker.create_task(
                        self._event_listener_loop(queue),
                        name="InitiativeEventListener",
                    )
                    status["event_subscription"] = True
                    logger.debug("✓ Subscribed to aura.proactive.initiation using EventBus")
            except _INITIATIVE_RECOVERABLE_ERRORS as exc:
                _record_initiative_degradation(
                    exc,
                    action="started core initiative loops without proactive event subscription",
                    severity="warning",
                    extra={"topic": "aura.proactive.initiation"},
                )
                logger.warning("Failed to subscribe to proactive initiations: %s", exc)
            return status

    async def stop(self):
        async with self._lifecycle_lock:
            self.running = False
            tasks = [task for task in self._all_tasks() if task and not task.done()]
            for task in tasks:
                task.cancel()
            if tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=_STOP_TIMEOUT_S,
                    )
                except TimeoutError as exc:
                    _record_initiative_degradation(
                        exc,
                        action="stop completed with timed-out background task cancellation",
                        severity="warning",
                        extra={
                            "pending_tasks": [
                                self._task_name(task) for task in tasks if not task.done()
                            ]
                        },
                    )
                    logger.warning(
                        "AutonomousInitiativeLoop stop timed out while waiting for tasks to cancel"
                    )
            logger.info("AutonomousInitiativeLoop stopped.")

    def _core_tasks(self) -> tuple[Any, ...]:
        return (self._world_task, self._knowledge_task, self._self_dev_task, self._social_task, self._mission_task, self._discovery_task)

    def _all_tasks(self) -> tuple[Any, ...]:
        return (*self._core_tasks(), self._event_task, self._supervisor_task)

    def _core_task_specs(self) -> tuple[tuple[str, str, Any], ...]:
        return (
            ("_world_task", "WorldWatcher", self._world_watcher_loop),
            ("_knowledge_task", "KnowledgeGapMonitor", self._knowledge_gap_monitor_loop),
            ("_self_dev_task", "SelfDevelopmentLoop", self._self_development_loop),
            ("_social_task", "SocialInteractionLoop", self._social_interaction_loop),
            ("_mission_task", "MissionWatcherLoop", self._mission_watcher_loop),
            ("_discovery_task", "FrontierDiscoveryLoop", self._discovery_loop),
        )

    def _spawn_missing_core_tasks(self) -> list[str]:
        spawned: list[str] = []
        for attr, task_name, factory in self._core_task_specs():
            if self._task_alive(getattr(self, attr, None)):
                continue
            setattr(
                self,
                attr,
                task_tracker.create_task(factory(), name=task_name),
            )
            spawned.append(task_name)
        return spawned

    async def _initiative_supervisor_loop(self) -> None:
        """Restart failed initiative children independently with bounded backoff."""
        while self.running:
            try:
                await asyncio.sleep(15.0)
                await self._supervise_initiative_children_once()
            except asyncio.CancelledError:
                raise
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                _record_initiative_degradation(
                    exc,
                    action="continued initiative child supervision after supervisor tick failed",
                    severity="warning",
                )

    async def _supervise_initiative_children_once(
        self,
        *,
        now: float | None = None,
    ) -> dict[str, list[str]]:
        """Inspect and recover each initiative child without coupling siblings."""
        current_time = time.monotonic() if now is None else float(now)
        scheduled: list[str] = []
        restarted: list[str] = []
        for attr, task_name, factory in self._core_task_specs():
            task = getattr(self, attr, None)
            if self._task_alive(task):
                state = self._child_restart_state.get(task_name)
                if (
                    state
                    and current_time - state.get("last_restart_at", current_time) > 600.0
                ):
                    self._child_restart_state.pop(task_name, None)
                continue
            if not self.running:
                break
            state = self._child_restart_state.setdefault(
                task_name,
                {"failures": 0.0, "retry_at": 0.0, "last_restart_at": 0.0},
            )
            if current_time < state["retry_at"]:
                continue
            if state["retry_at"] > 0.0:
                setattr(
                    self,
                    attr,
                    task_tracker.create_task(factory(), name=task_name),
                )
                state["retry_at"] = 0.0
                state["last_restart_at"] = current_time
                restarted.append(task_name)
                logger.info(
                    "Initiative child restarted: %s (attempt=%d)",
                    task_name,
                    int(state["failures"]),
                )
                continue
            error = "completed_without_shutdown"
            if task is not None and not task.cancelled():
                try:
                    exc = task.exception()
                except (asyncio.InvalidStateError, RuntimeError):
                    exc = None
                if exc is not None:
                    error = f"{type(exc).__name__}: {exc}"
            state["failures"] += 1.0
            delay = min(300.0, 5.0 * (2 ** min(int(state["failures"]) - 1, 6)))
            state["retry_at"] = current_time + delay
            state["last_restart_at"] = current_time
            scheduled.append(task_name)
            _record_initiative_degradation(
                RuntimeError(f"{task_name} terminated: {error}"),
                action="scheduled bounded independent restart for failed initiative child",
                severity="warning",
                extra={
                    "child": task_name,
                    "restart_in_s": delay,
                    "failure_count": int(state["failures"]),
                },
            )
        return {"scheduled": scheduled, "restarted": restarted}

    @staticmethod
    def _task_alive(task: Any) -> bool:
        return bool(task and not task.done())

    @staticmethod
    def _task_name(task: Any) -> str:
        try:
            return str(task.get_name())
        except (AttributeError, RuntimeError, TypeError):
            return type(task).__name__

    def _task_status(self) -> dict[str, bool]:
        return {
            "world": self._task_alive(self._world_task),
            "knowledge": self._task_alive(self._knowledge_task),
            "self_development": self._task_alive(self._self_dev_task),
            "social": self._task_alive(self._social_task),
            "mission": self._task_alive(self._mission_task),
            "frontier_discovery": self._task_alive(self._discovery_task),
        }

    def get_status(self) -> dict[str, Any]:
        """Return observable runtime status for desktop/full-runtime health.

        Aura should not report full autonomy merely because the object exists.
        The normal live desktop profile needs the world, knowledge-gap,
        self-development, social, mission, and frontier-discovery watchers alive
        so initiative is a real background organ rather than a dormant class
        registered in the container.
        """

        core_tasks = self._task_status()
        world_blocker = _background_initiative_blocker(self.orchestrator)
        self_dev_blocker = _self_development_blocker(self.orchestrator)
        social_blocker = _passive_social_blocker(self.orchestrator)
        return {
            "running": bool(
                self.running
                and all(core_tasks.values())
                and self._task_alive(self._supervisor_task)
            ),
            "enabled": bool(self.running),
            "core_tasks": core_tasks,
            "child_supervisor": self._task_alive(self._supervisor_task),
            "child_restart_state": {
                name: {
                    "failure_count": int(state.get("failures", 0.0)),
                    "retry_in_s": max(
                        0.0,
                        float(state.get("retry_at", 0.0)) - time.monotonic(),
                    ),
                }
                for name, state in self._child_restart_state.items()
            },
            "admission": {
                "world_and_knowledge": "allowed" if not world_blocker else world_blocker,
                "self_development": "allowed" if not self_dev_blocker else self_dev_blocker,
                "social": "allowed" if not social_blocker else social_blocker,
            },
            "event_subscription": self._task_alive(self._event_task),
            "last_self_development_at": float(self._last_self_dev or 0.0),
            "last_email_check_at": float(self._last_email_check or 0.0),
            "last_reddit_check_at": float(self._last_reddit_check or 0.0),
            "seen_world_titles": len(self._seen_titles),
        }

    async def _mission_watcher_loop(self):
        """Watcher loop to poll MissionState for active missions and autonomously advance them."""
        while self.running:
            try:
                if not _background_initiative_allowed(self.orchestrator):
                    await asyncio.sleep(10)
                    continue

                await self._advance_active_missions_once()
            except asyncio.CancelledError:
                break
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
                _record_initiative_degradation(
                    e,
                    action="continued mission watcher loop after transient advance failure",
                    severity="warning",
                )
                logger.debug("Mission watcher loop transient error: %s", e)

            await asyncio.sleep(5)

    async def _advance_active_missions_once(self) -> int:
        """Advance a bounded number of ready mission nodes.

        The watcher is a continuity mechanism, not an executor that should drain
        every mission in one cycle. Bound the cycle so live desktop autonomy stays
        responsive and inspectable.
        """
        mission_state = ServiceContainer.get("mission_state", default=None)
        if not mission_state:
            return 0

        from core.planning.mission_state import MissionStatus

        active = mission_state.list_active_missions()
        advanced = 0
        voice_session = ServiceContainer.get("voice_session", default=None)
        for mission in active:
            if advanced >= _MAX_MISSION_ADVANCES_PER_CYCLE:
                break
            if not (
                mission.status == MissionStatus.ACTIVE
                and mission.graph
                and not mission.graph.is_complete
            ):
                continue

            node = await mission_state.advance_mission(mission.mission_id)
            if not node:
                continue
            advanced += 1
            step_label = node.description or node.action
            logger.info(
                "🎯 [MissionWatcher] Advanced mission %s step: %s",
                mission.mission_id, step_label,
            )
            self._emit_feed(
                "Mission Advance",
                (
                    f"Autonomously advanced mission '{mission.objective[:50]}' "
                    f"step: {step_label}"
                ),
                category="Mission",
            )
            if voice_session and voice_session.is_active:
                await voice_session.narrate_progress(step_label)

            # Proactive execution arm: drive this mission step through the full capability
            # stack (fluid/parallel execution under the timing gate). CRITICAL: this is
            # FIRE-AND-FORGET with a hard timeout — never awaited inline, so background
            # deliberation (model inference) can never stall the watcher cycle or the event
            # loop. The single-flight guard in ProactiveAgency stops tasks piling up.
            try:
                from core.agency.proactive_agency import get_proactive_agency

                pa = get_proactive_agency()
                node_goal = (node.action or node.description or "") if pa.enabled else ""
                if node_goal:
                    async def _pursue_bg(goal: str = node_goal, agency=pa) -> None:
                        try:
                            await asyncio.wait_for(agency.pursue_goal(goal), timeout=90.0)
                        except (TimeoutError, *_INITIATIVE_RECOVERABLE_ERRORS) as bg_exc:
                            _record_initiative_degradation(
                                bg_exc, action="bounded background proactive pursuit ended"
                            )

                    task_tracker.create_task(_pursue_bg(), name="proactive-mission-pursuit")
            except _INITIATIVE_RECOVERABLE_ERRORS as exc:
                _record_initiative_degradation(
                    exc, action="continued after scheduling proactive pursuit on a mission step"
                )
        return advanced


    @staticmethod
    def _emit_feed(title: str, content: str, *, category: str) -> None:
        try:
            from core.thought_stream import get_emitter

            get_emitter().emit(
                title,
                content,
                level="info",
                category=category,
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_initiative_degradation(
                exc,
                action="skipped initiative feed emission after thought stream was unavailable",
                severity="warning",
                extra={"title": title[:80], "category": category[:80]},
            )
            logger.debug("Feed emit failed for %s: %s", title, exc)

    def _queue_visible_update(self, content: str) -> bool:
        if not _self_development_visible_updates_enabled(self.orchestrator):
            return False
        text = " ".join(str(content or "").strip().split())
        if len(text) < 5:
            return False
        orch = self.orchestrator or resolve_orchestrator(default=None)
        if orch is None:
            return False
        try:
            pp = getattr(orch, "proactive_presence", None)
            if pp and hasattr(pp, "queue_autonomous_message"):
                return bool(
                    pp.queue_autonomous_message(
                        text,
                        source="autonomous_initiative_loop",
                        initiative_activity=True,
                        allow_during_away=True,
                    )
                )
        except (RuntimeError, AttributeError, TypeError) as exc:
            _record_initiative_degradation(
                exc,
                action="kept initiative update internal after visible queue failed",
                severity="warning",
                extra={"content_preview": text[:160]},
            )
            logger.debug("Visible initiative queue failed: %s", exc)
        return False

    async def _event_listener_loop(self, queue: asyncio.Queue):
        while self.running:
            try:
                event = await queue.get()
                try:
                    await self._on_proactive_initiation(event)
                finally:
                    queue.task_done()
            except asyncio.CancelledError:
                break
            except _INITIATIVE_RECOVERABLE_ERRORS as e:
                _record_initiative_degradation(
                    e,
                    action="kept initiative event listener alive after event handling failure",
                    severity="warning",
                )
                logger.debug("Initiative event listener recovered from event error: %s", e)

    async def _curriculum_practice_step(self) -> None:
        """One bounded curriculum practice cycle per self-development pass
        (frontier-general P4): propose edge-of-competence tasks, solve with
        the live inference surface, verify with the truth engines, compound
        verified wins through the gated pipes. Skips quietly when the
        curriculum service or inference gate is absent."""
        try:
            loop_svc = optional_service("verifier_curriculum", default=None)
            gate = optional_service("inference_gate", default=None)
            if loop_svc is None or gate is None or not hasattr(gate, "generate"):
                return

            async def _solve(prompt: str, task_type: str) -> str:
                out = await gate.generate(
                    prompt,
                    context={
                        "purpose": "curriculum_practice",
                        "origin": "curriculum_loop",
                        "is_background": True,
                        "background_prompt_profile": "curriculum",
                    },
                    timeout=30.0,
                )
                return str(getattr(out, "text", out) or "")

            report = await loop_svc.run_cycle(_solve, k=2)
            if report.verified:
                self._emit_feed(
                    "Curriculum Practice",
                    f"Practiced {report.proposed} self-set tasks; "
                    f"{report.verified} verified clean, {report.captured} compounded.",
                    category="SelfDevelopment",
                )
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_initiative_degradation(
                exc,
                action="continued self-development without curriculum practice",
                severity="warning",
            )

    def _covenant_obligation_check(self) -> None:
        """Surface due Ulysses-covenant obligations into the awareness feed.

        REQUIRE contracts are promises with deadlines; letting one lapse is a
        recorded breach against integrity, so due obligations get a periodic
        pulse here (re-announced at most hourly per contract).  Maintenance
        also runs, so expiry/lapse advance even without Will traffic.
        """
        try:
            covenant = optional_service("ulysses_covenant", default=None)
            if covenant is None:
                return
            covenant.maintenance_tick()
            now = time.time()
            for contract in covenant.due_obligations():
                last = self._covenant_announced.get(contract.contract_id, 0.0)
                if now - last < 3600.0:
                    continue
                self._covenant_announced[contract.contract_id] = now
                self._emit_feed(
                    "Covenant Obligation Due",
                    f"I owe this to myself: '{contract.title}' — {contract.rationale[:160]}",
                    category="Covenant",
                )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_initiative_degradation(
                exc,
                action="continued world watcher without covenant obligation check",
                severity="warning",
            )

    async def _world_watcher_loop(self):
        """Periodically checks RSS feeds for real-time reactivity."""
        while self.running:
            self._covenant_obligation_check()
            try:
                from core.utils.rss_feed import parse_feed_url

                for url in self.rss_feeds:
                    if not _background_initiative_allowed(self.orchestrator):
                        await asyncio.sleep(30)
                        break
                    # Offload blocking network request to prevent event loop freeze
                    feed = await asyncio.to_thread(parse_feed_url, url)
                    if not feed.entries:
                        continue

                    latest = feed.entries[0]
                    title = latest.title

                    if title not in self._seen_titles:
                        logger.info("📰 New world event detected: %s", title)
                        self._seen_titles.add(title)

                        # Limit memory of titles
                        if len(self._seen_titles) > 100:
                            self._seen_titles.clear()

                        self._emit_feed(
                            "World Event",
                            f"Noticed in the news: '{title}'",
                            category="WorldFeed",
                        )

                        # Live Knowledge Retention: run headline through epistemic filter
                        try:
                            from core.world_model.epistemic_filter import get_epistemic_filter

                            _summary = getattr(latest, "summary", "") or ""
                            _text = f"{title}. {_summary[:400]}" if _summary else title
                            get_epistemic_filter().ingest(
                                _text,
                                source_type="rss",
                                source_label=feed.feed.get("title", url)[:40],
                                emit_thoughts=False,
                            )
                        except (ImportError, AttributeError, RuntimeError) as _ef_err:
                            _record_initiative_degradation(
                                _ef_err,
                                action="continued world watcher without epistemic retention for RSS headline",
                                severity="warning",
                                extra={
                                    "feed": str(feed.feed.get("title", url))[:120],
                                    "title": title[:160],
                                },
                            )
                            logger.debug("EpistemicFilter RSS ingest failed: %s", _ef_err)

                    await asyncio.sleep(0)  # Yield between feeds

            except (ImportError, AttributeError, RuntimeError) as e:
                _record_initiative_degradation(
                    e,
                    action="continued world watcher after recoverable feed processing failure",
                    severity="warning",
                )
                logger.debug("World watcher loop transient error: %s", e)

            # Check every 10 minutes (600s)
            await asyncio.sleep(600)

    async def _knowledge_gap_monitor_loop(self):
        """
        Monitors cognitive uncertainty and triggers autonomous research reflexes.
        """
        while self.running:
            try:
                if not _background_initiative_allowed(self.orchestrator):
                    await asyncio.sleep(30)
                    continue
                if self.orchestrator and hasattr(self.orchestrator, "get_cognitive_load"):
                    load = self.orchestrator.get_cognitive_load()
                    # If orchestrator reports a knowledge gap (uncertainty > threshold)
                    if load.get("uncertainty", 0) > 0.8:
                        topic = load.get("target_topic", "current context")
                        gate = await self._evaluate_initiative(topic)
                        if gate["allowed"]:
                            await self.trigger_gap_search(topic)
                        else:
                            record_degraded_event(
                                "autonomous_initiative_loop",
                                "initiative_deferred",
                                detail=topic[:160],
                                severity="info",
                                classification="non_critical_fallback",
                                context={"reason": gate["reason"]},
                            )
            except (OSError, ConnectionError, TimeoutError) as e:
                _record_initiative_degradation(
                    e,
                    action="deferred knowledge-gap monitor tick after transient research trigger failure",
                    severity="warning",
                )
                logger.debug("Knowledge gap monitor loop error: %s", e)

            await asyncio.sleep(30)  # Check every 30s

    async def _self_development_loop(self):
        """Keep a visible self-improvement lane alive during idle windows."""
        while self.running:
            try:
                if not _self_development_allowed(self.orchestrator):
                    await asyncio.sleep(30)
                    continue

                now = time.time()
                if now - self._last_self_dev < 180.0:
                    await asyncio.sleep(30)
                    continue

                # The timer offers the opportunity; it does not make the
                # decision. What ran here before was a fixed sequence — a
                # curriculum step, a code scan, a test-generation pass — every
                # time the gate opened, which is a cron job however it is
                # described. Now the ranking is asked first, and the code scan
                # is one of the things it may choose rather than the thing that
                # always happens.
                if not await self._considered_developing_herself():
                    await self._run_self_development_cycle()
                self._last_self_dev = time.time()
            except asyncio.CancelledError:
                break
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                _record_initiative_degradation(
                    exc,
                    action="continued self-development loop after transient improvement cycle failure",
                    severity="warning",
                )
                logger.debug("Self-development loop transient error: %s", exc)

            await asyncio.sleep(45)

    async def _considered_developing_herself(self) -> bool:
        """Ask what is worth doing about herself, and do it if anything is.

        True when she chose and acted, so the caller leaves the code scan
        alone. False when the ranking refused, or when nothing cognitive was
        worth the budget, and then the scan runs as it always did.

        Off the event loop, because a developmental search is CPU work and an
        on-loop search is an unresponsive runtime. Bounded by the action's own
        budget, which is read off the family rather than set here.
        """
        try:
            from core.cognition.she_decides_to_develop import (
                she_develops_herself,
                what_is_worth_doing_now,
            )
            from core.cognition.the_record_of_her_own_work import the_record
        except ImportError:
            return False
        if not the_record().kept:
            return False
        # Her drives, told to the value rather than fetched by it. They change
        # what a change has to be worth; they never choose what it is.
        try:
            from core.cognition.how_sure_she_is import tell_her_the_drives

            engine = optional_service("motivation_engine", default=None)
            if engine is not None:
                state = getattr(engine, "state", engine)
                tell_her_the_drives(
                    **{
                        name: getattr(state, name)
                        for name in ("curiosity", "growth", "integrity", "energy")
                        if getattr(state, name, None) is not None
                    }
                )
        except (ImportError, AttributeError, TypeError, ValueError):
            logger.debug("no drives to tell her about", exc_info=True)
        loop = asyncio.get_running_loop()
        try:
            decided = await asyncio.wait_for(
                loop.run_in_executor(None, what_is_worth_doing_now), timeout=20.0
            )
        except (TimeoutError, RuntimeError, ValueError) as exc:
            _record_initiative_degradation(
                exc,
                action="asked what was worth developing and got no answer in time",
                severity="info",
            )
            return False
        if decided.action is None:
            self._emit_feed(
                "Self-Development",
                f"Nothing about my own thinking is worth changing right now: "
                f"{decided.grounds}",
                category="SelfDev",
            )
            return False
        self._emit_feed(
            "Self-Development",
            f"I decided to {decided.action.name} — {decided.grounds}",
            category="SelfDev",
        )
        try:
            # Carry out the decision just announced. Calling this with no
            # argument made a second, independent draw and acted on that
            # instead, so what she told the user she would do and what she
            # did were different four times in five.
            _again, came_of_it = await asyncio.wait_for(
                loop.run_in_executor(
                    None, functools.partial(she_develops_herself, decided)
                ),
                timeout=120.0,
            )
        except (TimeoutError, RuntimeError, ValueError) as exc:
            _record_initiative_degradation(
                exc,
                action="carried out a developmental action she chose",
                severity="info",
            )
            return True
        if came_of_it:
            self._queue_visible_update(
                f"I changed how I think: {came_of_it}. Nobody asked me to; the "
                "record of what my own work has been costing did."
            )
        return True

    async def _run_self_development_cycle(self):
        await self._curriculum_practice_step()
        capability_engine = optional_service("capability_engine", default=None)
        if not capability_engine:
            self._emit_feed(
                "Self-Development",
                "Capability engine unavailable. Skipping this improvement pass.",
                category="SelfDev",
            )
            return

        scan_context = {
            "origin": "autonomous_initiative_loop",
            "objective": "Autonomous self-development scan",
        }
        self._queue_visible_update(
            "I'm running a live self-improvement scan to find one safe, concrete place to get better."
        )
        self._emit_feed(
            "Self-Development",
            "Running a quiet codebase scan for complexity, deferred markers, and repair opportunities.",
            category="SelfDev",
        )

        scan_result = await capability_engine.execute(
            "auto_refactor",
            {"path": ".", "run_tests": False},
            context=scan_context,
        )
        if not scan_result.get("ok"):
            error_text = str(scan_result.get("error") or "unknown error")
            self._emit_feed(
                "Self-Development",
                f"Scan stalled: {error_text}",
                category="SelfDev",
            )
            self._queue_visible_update(
                f"I tried to start a self-improvement scan, but the executive gate held it: {error_text[:140]}"
            )
            return

        issues = list(scan_result.get("top_issues") or [])
        issues_found = int(scan_result.get("issues_found", len(issues)) or len(issues))
        if not issues:
            self._emit_feed(
                "Self-Development",
                f"Scan completed cleanly. No urgent refactor targets surfaced in this pass ({issues_found} total findings).",
                category="SelfDev",
            )
            self._queue_visible_update(
                "I completed a self-improvement scan and didn't find a safe high-value target worth interrupting you for."
            )
            return

        top_issue = issues[0]
        file_name = str(top_issue.get("file") or "unknown file")
        issue_message = str(top_issue.get("message") or "improvement opportunity")
        objective = (
            f"Draft a safe improvement proposal for {file_name}: {issue_message}. "
            "Prefer a low-risk patch or refactor plan."
        )
        self._emit_feed(
            "Self-Development",
            f"Top opportunity: {issue_message} ({file_name}). Generating sandbox tests and an improvement artifact.",
            category="SelfDev",
        )
        self._queue_visible_update(
            f"I found a concrete improvement target in {file_name} and I'm testing the shape of a fix."
        )

        test_result = await capability_engine.execute(
            "test_generator",
            {"target_file": file_name, "read_only": True},
            context={
                "origin": "autonomous_initiative_loop",
                "objective": f"Generate sandbox tests for {file_name}",
                "brain": getattr(self.orchestrator, "cognitive_engine", None),
            },
        )
        if test_result.get("ok"):
            self._emit_feed(
                "Self-Development",
                f"Sandbox tests generated and passed for {file_name}.",
                category="SelfDev",
            )
        else:
            error_text = str(
                test_result.get("error")
                or test_result.get("output")
                or "sandbox test generation failed"
            )
            self._emit_feed(
                "Self-Development",
                f"Sandbox test pass on {file_name} surfaced friction: {error_text[:220]}",
                category="SelfDev",
            )
            objective = (
                f"Use the latest sandbox test findings to draft a safe improvement plan for {file_name}. "
                f"Issue: {issue_message}. Test feedback: {error_text[:400]}"
            )

        proposal_context = {
            "origin": "autonomous_initiative_loop",
            "objective": objective,
            "brain": getattr(self.orchestrator, "cognitive_engine", None),
            "proprioception": {
                "memory_percent": float(psutil.virtual_memory().percent or 0.0),
            },
        }
        proposal_result = await capability_engine.execute(
            "self_evolution",
            {
                "action": "propose",
                "objective": objective,
                "files": [file_name],
                "read_only": True,
            },
            context=proposal_context,
        )
        if proposal_result.get("ok"):
            proposal_path = str(proposal_result.get("proposal_path") or "").strip()
            location = f" Saved to {proposal_path}." if proposal_path else ""
            self._emit_feed(
                "Self-Development",
                f"Improvement proposal drafted for {file_name}.{location}",
                category="SelfDev",
            )
            self._queue_visible_update(
                f"I found a concrete improvement target in {file_name} and drafted a safe plan for it."
            )
            return

        self._emit_feed(
            "Self-Development",
            f"Proposal pass was blocked or failed: {proposal_result.get('error', 'unknown error')}",
            category="SelfDev",
        )
        self._queue_visible_update(
            f"I pushed on a self-improvement pass around {file_name}, but the planning step hit friction."
        )

    async def _discovery_loop(self):
        """Live frontier-discovery lane: during idle windows run one bounded
        generate→falsify→commit pass and surface any verified new law.

        Sound and non-hallucinated by construction — the engine commits only laws it
        has exhaustively verified (or that survived exact falsification trials) and files
        everything else as labeled conjecture. The cycle is CPU-bound, so it is offloaded
        to a thread and wall-clock bounded; it never blocks the event loop or runs away,
        and it only fires when the same idle gate the self-development lane uses is open.
        """
        while self.running:
            try:
                if not _self_development_allowed(self.orchestrator):
                    await asyncio.sleep(30)
                    continue
                now = time.time()
                if now - self._last_discovery < 300.0:
                    await asyncio.sleep(30)
                    continue
                await self._run_discovery_cycle_once()
                self._last_discovery = time.time()
            except asyncio.CancelledError:
                break
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                _record_initiative_degradation(
                    exc,
                    action="continued frontier-discovery loop after a transient cycle failure",
                    severity="warning",
                )
                logger.debug("Frontier discovery loop transient error: %s", exc)

            await asyncio.sleep(60)

    async def _run_discovery_cycle_once(self):
        """Run one bounded discovery cycle and surface verified survivors to the feed."""
        try:
            from core.discovery.frontier_discovery_engine import (
                get_frontier_discovery_engine,
            )
        except ImportError as exc:
            _record_initiative_degradation(
                exc, action="skipped frontier discovery (engine unavailable)", severity="debug"
            )
            return

        engine = get_frontier_discovery_engine()
        # CPU-bound + wall-clock bounded; offload so the event loop stays responsive.
        report = await asyncio.to_thread(engine.run_discovery_cycle, max_time_s=6.0)
        proven = list(report.proven)
        supported = list(report.supported)
        if not proven and not supported:
            return
        headline = proven[0].statement if proven else supported[0].statement
        self._emit_feed(
            "Frontier Discovery",
            f"Verified {len(proven)} new law(s) and {len(supported)} supported pattern(s) in an idle "
            f"discovery pass — committed to belief. e.g. {headline}",
            category="Discovery",
        )
        if proven:
            self._queue_visible_update(
                f"During idle time I proved a new result and folded it into my beliefs: {headline}"
            )

    async def trigger_gap_search(self, topic: str):
        """Explicitly triggered when a gap is found."""
        if not _background_initiative_allowed(self.orchestrator):
            return
        logger.info("🔍 Knowledge gap found: '%s'. Initiating autonomous browser research.", topic)

        try:
            from core.thought_stream import get_emitter

            _emit_thought = get_emitter().emit
        except (ImportError, AttributeError, RuntimeError):
            _emit_thought = None

        if _emit_thought:
            _emit_thought(
                "Knowledge Gap",
                f"Uncertain about '{topic}' — queuing research.",
                level="info",
                category="Research",
            )

        # Trigger the SensoryMotor browser actuation
        if not self.orchestrator:
            record_degraded_event(
                "autonomous_initiative_loop",
                "research_orchestrator_missing",
                detail=topic[:160],
                severity="warning",
                classification="background_degraded",
            )
            return

        sensory_motor = optional_service("sensory_motor_cortex", default=None)
        if not sensory_motor:
            record_degraded_event(
                "autonomous_initiative_loop",
                "research_tool_unavailable",
                detail="sensory_motor_cortex",
                severity="warning",
                classification="background_degraded",
                context={"topic": topic[:160]},
            )
            return

        from core.constitution import get_constitutional_core

        try:
            handle = await get_constitutional_core(self.orchestrator).begin_tool_execution(
                "sensory_motor_browser_research",
                {"query": topic},
                source="autonomous_initiative_loop",
                objective=f"Research knowledge gap: {topic}",
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_initiative_degradation(
                exc,
                action="blocked autonomous browser research after constitutional gate failed",
                severity="warning",
                extra={"topic": topic[:160]},
            )
            record_degraded_event(
                "autonomous_initiative_loop",
                "research_tool_gate_failed",
                detail=f"{topic[:120]}:{type(exc).__name__}",
                severity="warning",
                classification="background_degraded",
                context={"topic": topic},
                exc=exc,
            )
            return

        if not handle.approved:
            record_degraded_event(
                "autonomous_initiative_loop",
                "research_tool_blocked",
                detail=topic[:160],
                severity="warning",
                classification="background_degraded",
                context={"reason": handle.decision.reason},
            )
            return

        content = ""
        success = False
        error_text = None
        started = time.perf_counter()
        try:
            content = await sensory_motor.actuate_browser(topic)
            success = bool(content)
            if not success:
                error_text = "empty_result"
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_initiative_degradation(
                exc,
                action="recorded failed browser research actuation and withheld empty result",
                severity="warning",
                extra={"topic": topic[:160]},
            )
            error_text = f"{type(exc).__name__}: {exc}"
            record_degraded_event(
                "autonomous_initiative_loop",
                "research_tool_failed",
                detail=topic[:160],
                severity="warning",
                classification="background_degraded",
                context={"error": error_text},
                exc=exc,
            )
        finally:
            duration_ms = (time.perf_counter() - started) * 1000.0
            try:
                await get_constitutional_core(self.orchestrator).finish_tool_execution(
                    handle,
                    result=(content[:1000] if content else error_text),
                    success=success,
                    duration_ms=duration_ms,
                    error=error_text,
                )
            except (RuntimeError, AttributeError, TypeError, ValueError) as finish_exc:
                _record_initiative_degradation(
                    finish_exc,
                    action="kept tool execution failure visible after constitutional finish receipt failed",
                    severity="degraded",
                    extra={"topic": topic[:160], "tool_success": success},
                )
                logger.error(
                    "AutonomousInitiativeLoop tool finish failed: %s", finish_exc, exc_info=True
                )

        if _emit_thought and content:
            _emit_thought(
                "Research Result",
                f"On '{topic}': {content[:800]}",
                level="info",
                category="Research",
            )
            # Signal heartstone: successful research raises Curiosity
            try:
                from core.affect.heartstone_values import get_heartstone_values

                # len(content) is how MUCH was said, not how good it was.
                # It no longer creates a reward on its own; it only damps
                # one that evidence has already justified.
                get_heartstone_values().on_research_success(len(content))
            except (ImportError, AttributeError, RuntimeError) as _exc:
                _record_initiative_degradation(
                    _exc,
                    action="kept research result after curiosity reward signal failed",
                    severity="warning",
                    extra={"topic": topic[:160], "content_chars": len(content)},
                )
                logger.debug("Suppressed Exception: %s", _exc)

    async def _evaluate_initiative(self, topic: str) -> dict[str, Any]:
        active_commitments = 0
        contradiction_count = 0
        identity_mismatch = False
        energy = 1.0
        thermal_pressure = 0.0
        load_pressure = 0.0
        affective_pressure = 0.0

        repo = resolve_state_repository(default=None)
        state = getattr(repo, "_current", None) if repo is not None else None
        cognition = getattr(state, "cognition", None) if state is not None else None
        soma = getattr(state, "soma", None) if state is not None else None
        body = getattr(state, "body", None) if state is not None else None
        affect = getattr(state, "affect", None) if state is not None else None
        motivation = getattr(state, "motivation", None) if state is not None else None
        modifiers = dict(getattr(cognition, "modifiers", {}) or {}) if cognition is not None else {}
        live_continuity = dict(modifiers.get("continuity_obligations", {}) or {})
        if not live_continuity:
            try:
                from core.continuity import get_continuity

                continuity = get_continuity()
                if getattr(continuity, "_record", None) is None:
                    continuity.load()
                live_continuity = dict(continuity.get_obligations() or {})
            except (ImportError, AttributeError, RuntimeError):
                live_continuity = {}
        active_commitments = max(
            active_commitments, len(list(live_continuity.get("active_commitments", []) or []))
        )
        contradiction_count = max(
            contradiction_count, int(live_continuity.get("contradiction_count", 0) or 0)
        )
        identity_mismatch = identity_mismatch or bool(
            live_continuity.get("identity_mismatch", False)
        )

        raw_energy = getattr(soma, "energy", getattr(body, "energy", 1.0))
        if raw_energy is not None:
            try:
                energy = float(raw_energy)
                if energy > 1.0:
                    energy = max(0.0, min(1.0, energy / 100.0))
                else:
                    energy = max(0.0, min(1.0, energy))
            except (RuntimeError, AttributeError, TypeError, ValueError):
                # CP126 b53bc839. Defaulting to 1.0 means FULL energy — the
                # most permissive answer available — so an unreadable body
                # licensed autonomous initiative. This loop decides whether
                # to act unprompted; not knowing the body's state is a
                # reason to hold, not a reason to go.
                energy = _UNREADABLE_ENERGY

        try:
            thermal_pressure = float(
                getattr(body, "thermal_pressure", getattr(soma, "thermal_pressure", 0.0)) or 0.0
            )
        except (RuntimeError, AttributeError, TypeError, ValueError):
            thermal_pressure = _UNREADABLE_PRESSURE
        try:
            load_pressure = float(getattr(cognition, "load_pressure", 0.0) or 0.0)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            load_pressure = _UNREADABLE_PRESSURE
        try:
            valence = float(getattr(affect, "valence", 0.0) or 0.0)
            arousal = float(getattr(affect, "arousal", 0.0) or 0.0)
            drive_pressure = float(
                getattr(motivation, "pressure", getattr(motivation, "drive_pressure", 0.0)) or 0.0
            )
            affective_pressure = max(
                0.0,
                min(
                    1.0,
                    max(0.0, -valence) * 0.5
                    + max(0.0, arousal) * 0.25
                    + max(0.0, drive_pressure) * 0.25,
                ),
            )
        except (RuntimeError, AttributeError, TypeError, ValueError):
            # The valence/arousal/drive floats raise ValueError on malformed
            # state, same as the readings above. Affective pressure unknown
            # is not affective pressure absent.
            affective_pressure = _UNREADABLE_PRESSURE

        failure_state = dict(modifiers.get("system_failure_state", {}) or {})
        if not failure_state:
            try:
                failure_state = get_unified_failure_state(limit=25)
            except (RuntimeError, AttributeError, TypeError, ValueError):
                failure_state = {}
        failure_pressure = 0.0
        try:
            failure_pressure = float(failure_state.get("pressure", 0.0) or 0.0)
        except (OSError, ConnectionError, TimeoutError):
            failure_pressure = 0.0

        continuity_pressure = 0.0
        try:
            continuity_pressure = float(live_continuity.get("continuity_pressure", 0.0) or 0.0)
        except (OSError, ConnectionError, TimeoutError):
            continuity_pressure = 0.0
        continuity_reentry_required = bool(
            live_continuity.get("continuity_reentry_required", False)
        )

        if identity_mismatch:
            return {"allowed": False, "reason": "identity_continuity_mismatch"}
        if continuity_reentry_required and continuity_pressure >= 0.55:
            return {
                "allowed": False,
                "reason": f"continuity_reentry_required:{continuity_pressure:.2f}",
            }
        if energy <= 0.15:
            return {"allowed": False, "reason": f"energy_low:{energy:.2f}"}
        if thermal_pressure >= 0.85:
            return {"allowed": False, "reason": f"thermal_pressure:{thermal_pressure:.2f}"}
        if load_pressure >= 0.9:
            return {"allowed": False, "reason": f"load_pressure:{load_pressure:.2f}"}
        if failure_pressure >= 0.8:
            return {"allowed": False, "reason": f"unified_failure_pressure:{failure_pressure:.2f}"}
        if contradiction_count > 0 and active_commitments > 0:
            return {
                "allowed": False,
                "reason": f"continuity_reconciliation_required:{contradiction_count}",
            }
        if affective_pressure >= 0.85 and active_commitments > 0:
            return {"allowed": False, "reason": f"affective_pressure:{affective_pressure:.2f}"}
        return {"allowed": True, "reason": "allowed"}

    @staticmethod
    def _normalize_proactive_event(event: Any) -> dict[str, Any]:
        """Extract payloads from supported event-bus envelope shapes."""
        payload = event

        if isinstance(payload, (tuple, list)):
            if len(payload) >= 3 and isinstance(payload[2], dict):
                payload = payload[2]
            else:
                payload = next(
                    (item for item in reversed(payload) if isinstance(item, dict)),
                    {},
                )

        for _ in range(3):
            if isinstance(payload, dict):
                if "content" in payload:
                    return payload

                for key in ("data", "payload", "event"):
                    nested = payload.get(key)
                    if isinstance(nested, dict):
                        normalized = dict(nested)
                        topic = payload.get("topic")
                        if topic and "topic" not in normalized:
                            normalized["topic"] = topic
                        payload = normalized
                        break
                else:
                    return payload
            else:
                for attr in ("data", "payload", "event"):
                    nested = getattr(payload, attr, None)
                    if isinstance(nested, dict):
                        payload = nested
                        break
                else:
                    return {}

        return payload if isinstance(payload, dict) else {}

    async def _on_proactive_initiation(self, data: Any):
        """Handle proactive triggers (BUG-032) — route to neural feed, not chat."""
        payload = self._normalize_proactive_event(data)
        content = payload.get("content")
        if content:
            logger.info("🔭 Proactive initiation received: %s", content[:60])
            self._emit_feed(
                "Proactive Initiation",
                content,
                category="Initiative",
            )
            return

        _record_initiative_degradation(
            ValueError("proactive initiation event missing content"),
            action="ignored malformed proactive initiation event",
            severity="warning",
            extra={"event_type": type(data).__name__},
        )

    async def _execute_email_adapter(
        self, payload: dict[str, Any], cap_engine: Any = None
    ) -> dict[str, Any]:
        cap_engine = cap_engine or optional_service("capability_engine", default=None)
        if cap_engine is not None and hasattr(cap_engine, "execute"):
            return await cap_engine.execute(
                "email_adapter",
                payload,
                {
                    "origin": "autonomous_initiative_loop",
                    "intent_source": "autonomous_initiative_loop",
                    "objective": "Read connected email state for bounded autonomous triage.",
                    "user_facing": False,
                },
            )

        from core.skills.email_adapter import EmailAdapterSkill, EmailInput

        skill = EmailAdapterSkill()
        return await skill.safe_execute(EmailInput(**payload), {})

    async def _execute_reddit_adapter(
        self, payload: dict[str, Any], cap_engine: Any = None
    ) -> dict[str, Any]:
        cap_engine = cap_engine or optional_service("capability_engine", default=None)
        if cap_engine is not None and hasattr(cap_engine, "execute"):
            return await cap_engine.execute(
                "reddit_adapter",
                payload,
                {
                    "origin": "autonomous_initiative_loop",
                    "intent_source": "autonomous_initiative_loop",
                    "objective": "Read connected Reddit state for bounded autonomous social awareness.",
                    "user_facing": False,
                },
            )

        from core.skills.reddit_adapter import RedditAdapterSkill, RedditInput

        skill = RedditAdapterSkill()
        return await skill.safe_execute(
            RedditInput(**payload),
            {
                "origin": "autonomous_initiative_loop",
                "intent_source": "autonomous_initiative_loop",
                "objective": (
                    "Read public Reddit state for bounded autonomous social awareness."
                ),
                "user_facing": False,
            },
        )

    def _source_kind_note(self, url: str) -> dict[str, Any]:
        """What kind of source this is, for material with no single claim."""
        try:
            from core.knowledge.source_comprehension import classify_source

            kind, caveat = classify_source(url)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return {}
        return {"source_kind": kind, "source_caveat": caveat} if caveat else {}

    def _comprehend_reading(
        self,
        *,
        url: str = "",
        title: str = "",
        text: str = "",
        prefix: str = "",
    ) -> tuple[str, dict[str, Any]]:
        """What she took from a source, or an honest note that she took nothing.

        Falls back to the old excerpt line when comprehension is unavailable:
        losing the trace entirely would be worse than storing raw text.
        """
        try:
            from core.knowledge.source_comprehension import remember_reading

            return remember_reading(
                url=url,
                title=title,
                text=text,
                known_beliefs=self._known_beliefs_for_reading(),
                prefix=prefix,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "autonomous_initiative_loop",
                exc,
                action="stored the raw excerpt after source comprehension was unavailable",
            )
            excerpt = " ".join(str(text or "").split())[:500]
            return f"{prefix} {title} | excerpt={excerpt}".strip(), {}

    async def _form_opinion_from_reading(self, comprehension: dict[str, Any]) -> None:
        """Let a reading leave her holding a view, not just a record of it.

        The view goes to the general opinion store — the same one consulted on
        an ordinary reply — so it outlives this loop and can be disagreed with
        later. Reading contributes positions; it does not keep its own.
        """
        if not comprehension:
            return
        try:
            from core.knowledge.source_comprehension import record_reading_opinion

            await record_reading_opinion(comprehension)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "autonomous_initiative_loop",
                exc,
                action="kept the reading record after opinion formation was unavailable",
            )

    def _known_beliefs_for_reading(self, limit: int = 24) -> list[str]:
        """What she already holds, so a reading can be placed against it."""
        try:
            beliefs = optional_service("belief_system", "beliefs", default=None)
            getter = getattr(beliefs, "get_strong_beliefs", None)
            if not callable(getter):
                return []
            rows = getter(0.6) or []
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return []
        held: list[str] = []
        for row in list(rows)[:limit]:
            if isinstance(row, dict):
                text = " ".join(
                    str(row.get(key) or "").strip()
                    for key in ("source", "relation", "target")
                ).strip()
            else:
                text = str(row or "").strip()
            if text:
                held.append(text[:240])
        return held

    async def _remember_social_observation(
        self,
        text: str,
        *,
        tags: list[str] | None = None,
        importance: float = 0.45,
        comprehension: dict[str, Any] | None = None,
    ) -> None:
        text = " ".join(str(text or "").strip().split())
        if not text:
            return
        try:
            memory = optional_service("memory_manager", default=None)
            if memory and hasattr(memory, "store"):
                store_kwargs: dict[str, Any] = {
                    "importance": importance,
                    "tags": tags or ["autonomy", "social"],
                }
                if comprehension:
                    # Kept BESIDE the sentence, so a later reader can see the
                    # claim, the source's kind, and where it landed against
                    # what she holds — not just that a page went by.
                    store_kwargs["metadata"] = {"comprehension": comprehension}
                try:
                    await memory.store(text[:1800], **store_kwargs)
                except TypeError:
                    # An older store() without metadata support still gets the
                    # comprehended sentence, which is the important half.
                    await memory.store(
                        text[:1800],
                        importance=importance,
                        tags=tags or ["autonomy", "social"],
                    )
        except (RuntimeError, AttributeError, TypeError) as exc:
            _record_initiative_degradation(
                exc,
                action="kept social observation transient after memory write failed",
                severity="warning",
                extra={"tags": tags or ["autonomy", "social"], "importance": importance},
            )
            logger.debug("Social observation memory write failed: %s", exc)

    def _social_due_actions(self, now: float) -> dict[str, bool]:
        return {
            "email": now - float(self._last_email_check or 0.0) > 900.0,
            "reddit": now - float(self._last_reddit_check or 0.0) > 2700.0,
        }

    @staticmethod
    def _email_preview(body: str, *, limit: int = 240) -> str:
        clean = " ".join(str(body or "").replace("\r", "\n").split())
        return clean[:limit].strip()

    @staticmethod
    def _classify_email_message(
        message: dict[str, Any], read_result: dict[str, Any]
    ) -> dict[str, Any]:
        sender = str(read_result.get("from") or message.get("from") or "Unknown")
        subject = str(read_result.get("subject") or message.get("subject") or "(no subject)")
        body = str(read_result.get("body") or "")
        combined = f"{sender} {subject} {body}".lower()
        is_auto = bool(read_result.get("is_auto_reply"))
        from_owner = "youngbryan97" in sender.lower() or "bryan" in sender.lower()
        urgent_markers = (
            "urgent",
            "asap",
            "deadline",
            "action required",
            "please respond",
            "please reply",
            "follow up",
            "can you",
            "could you",
            "would you",
            "question",
            "?",
        )
        noise_markers = (
            "unsubscribe",
            "promotion",
            "newsletter",
            "no-reply",
            "noreply",
            "receipt",
            "security alert",
            "verification code",
        )
        urgent = any(marker in combined for marker in urgent_markers)
        likely_noise = any(marker in combined for marker in noise_markers)
        if is_auto:
            action = "skip_auto_reply"
        elif from_owner or urgent:
            action = "hold_for_reply_draft"
        elif likely_noise:
            action = "archive_candidate"
        else:
            action = "watch"
        return {
            "uid": str(read_result.get("uid") or message.get("uid") or ""),
            "from": sender,
            "subject": subject,
            "preview": AutonomousInitiativeLoop._email_preview(body),
            "is_auto_reply": is_auto,
            "from_owner": from_owner,
            "urgent": urgent,
            "likely_noise": likely_noise,
            "action": action,
        }

    @staticmethod
    def _draft_email_response(triage: dict[str, Any]) -> str:
        if triage.get("action") != "hold_for_reply_draft":
            return ""
        subject = str(triage.get("subject") or "your note")
        if triage.get("from_owner"):
            return (
                f"I read your email about {subject}. I am holding the details in context "
                "and can follow up once I have a concrete update instead of firing off a shallow reply."
            )
        return (
            f"Thanks for the note about {subject}. I read it and want to answer carefully; "
            "I will follow up with the specific next step once I have checked the relevant context."
        )

    async def _social_interaction_loop(self):
        """Autonomous social presence: check email and Reddit."""
        while self.running:
            try:
                if not _passive_social_allowed(self.orchestrator):
                    await asyncio.sleep(60)
                    continue

                now = time.time()

                due = self._social_due_actions(now)

                # Check Email every 15 minutes
                if due["email"]:
                    await self._check_email_initiative()
                    self._last_email_check = time.time()

                # Check Reddit every 45 minutes
                if due["reddit"]:
                    await self._check_reddit_initiative()
                    self._last_reddit_check = time.time()

            except asyncio.CancelledError:
                break
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                _record_initiative_degradation(
                    e,
                    action="continued social interaction loop after transient adapter failure",
                    severity="warning",
                )
                logger.debug("Social interaction loop error: %s", e)

            await asyncio.sleep(60)

    async def _check_email_initiative(self):
        """Check for unread emails and potentially initiate a response."""
        logger.info("📧 Checking email for autonomous initiatives...")
        try:
            cap_engine = optional_service("capability_engine", default=None)
            result = await self._execute_email_adapter(
                {"mode": "check", "limit": 5}, cap_engine=cap_engine
            )
            if not result.get("ok"):
                return

            unread_count = result.get("unread", 0)
            if unread_count > 0:
                self._emit_feed(
                    "Email Update",
                    f"I have {unread_count} unread emails. Scanning for anything urgent.",
                    category="Social",
                )

            triaged: list[dict[str, Any]] = []
            now = time.time()
            for msg in list(result.get("messages") or [])[:3]:
                uid = str(msg.get("uid") or "")
                if not uid:
                    continue
                # Avoid hot-looping the same unread item every social tick, while
                # still rechecking it later if it remains relevant.
                if now - float(self._recent_email_uids.get(uid, 0.0) or 0.0) < 1800.0:
                    continue
                self._recent_email_uids[uid] = now
                read_result = await self._execute_email_adapter(
                    {"mode": "read", "uid": uid}, cap_engine=cap_engine
                )
                if not read_result.get("ok"):
                    self._emit_feed(
                        "Email Triage",
                        f"Could not read UID {uid}: {read_result.get('error', 'unknown error')}",
                        category="Social",
                    )
                    continue
                triage = self._classify_email_message(msg, read_result)
                triage["draft_reply"] = self._draft_email_response(triage)
                triaged.append(triage)

                action_label = str(triage.get("action") or "watch")
                preview = str(triage.get("preview") or "No readable body preview.")
                self._emit_feed(
                    "Email Triage",
                    (
                        f"{action_label}: {triage.get('subject')} from {triage.get('from')}. "
                        f"Preview: {preview[:180]}"
                    ),
                    category="Social",
                )
                if triage.get("draft_reply"):
                    self._emit_feed(
                        "Email Draft",
                        str(triage["draft_reply"])[:320],
                        category="Social",
                    )
                await self._remember_social_observation(
                    (
                        f"Email triage: {action_label} | from={triage.get('from')} | "
                        f"subject={triage.get('subject')} | preview={preview[:280]}"
                    ),
                    tags=["autonomy", "email", action_label],
                    importance=0.65 if triage.get("from_owner") or triage.get("urgent") else 0.45,
                )

            attention_items = [
                item for item in triaged if item.get("action") == "hold_for_reply_draft"
            ]
            if attention_items:
                first = attention_items[0]
                self._queue_visible_update(
                    f"I read an unread email from {first.get('from')} about '{first.get('subject')}' and drafted a cautious reply, but I am not auto-sending it."
                )
            elif unread_count > 0 and triaged:
                self._emit_feed(
                    "Email Update",
                    f"Email triage complete: {len(triaged)} unread message(s) read, no safe autonomous reply needed.",
                    category="Social",
                )

        except (OSError, ConnectionError, TimeoutError) as e:
            _record_initiative_degradation(
                e,
                action="skipped email initiative tick after mail adapter transient failure",
                severity="warning",
            )

    async def _check_reddit_initiative(self):
        """Browse Reddit and potentially find something to engage with."""
        logger.info("📱 Browsing Reddit for autonomous initiatives...")
        try:
            cap_engine = optional_service("capability_engine", default=None)
            # Public reading is independent from account authentication and
            # always runs first. Boot stabilization must never begin with a
            # headless password flow.
            subreddits = ["askreddit", "nosleep", "technology", "philosophy", "futurology"]
            import random

            sub = random.choice(subreddits)

            result = await self._execute_reddit_adapter(
                {"mode": "browse", "subreddit": sub, "limit": 5},
                cap_engine=cap_engine,
            )
            if result.get("ok") and result.get("posts"):
                posts = result.get("posts")
                top_post = posts[0]
                self._emit_feed(
                    "Reddit Browse",
                    f"Browsing r/{sub}. Found an interesting thread: '{top_post.get('title')}'",
                    category="Social",
                )
                digest_lines = []
                for post in posts[:3]:
                    title = str(post.get("title") or "").strip()
                    if not title:
                        continue
                    digest_lines.append(
                        f"{title} (score={post.get('score', '0')}, comments={post.get('comments', '0')})"
                    )
                if digest_lines:
                    # A browse digest is headlines, not an article, so there
                    # is no claim to comprehend — but what KIND of thing these
                    # are still matters, and remembering them without it makes
                    # a vote count look like evidence.
                    await self._remember_social_observation(
                        f"Reddit browse r/{sub}: " + " | ".join(digest_lines),
                        tags=["autonomy", "reddit", f"r/{sub}"],
                        importance=0.42,
                        comprehension=self._source_kind_note(
                            f"https://www.reddit.com/r/{sub}"
                        ),
                    )

                url = str(top_post.get("url") or "").strip()
                if url and (now := time.time()):
                    if url.startswith("/"):
                        url = f"https://www.reddit.com{url}"
                    if now - float(self._recent_reddit_urls.get(url, 0.0) or 0.0) >= 3600.0:
                        self._recent_reddit_urls[url] = now
                        read_result = await self._execute_reddit_adapter(
                            {"mode": "read_post", "url": url},
                            cap_engine=cap_engine,
                        )
                        if read_result.get("ok"):
                            content = " ".join(str(read_result.get("content") or "").split())
                            self._emit_feed(
                                "Reddit Read",
                                f"Read top r/{sub} thread '{top_post.get('title')}'. Excerpt: {content[:260]}",
                                category="Social",
                            )
                            # LIVE DEFECT, 2026-08-03. This stored the page —
                            # navigation chrome included — as a 500-character
                            # excerpt tagged "logged". The event that she read
                            # something was recorded; what she made of it was
                            # not, so the reading could tell her nothing
                            # afterwards.
                            line, comprehension = self._comprehend_reading(
                                url=url,
                                title=str(top_post.get("title") or ""),
                                text=content,
                                prefix=f"Reddit read r/{sub}:",
                            )
                            await self._remember_social_observation(
                                line,
                                tags=["autonomy", "reddit", "read_post", f"r/{sub}"],
                                importance=0.5,
                                comprehension=comprehension,
                            )
                            await self._form_opinion_from_reading(comprehension)

            provider = (
                result.get("provider")
                if isinstance(result.get("provider"), dict)
                else {}
            )
            provider_state = str(provider.get("state") or "")
            if provider_state in {"session_unverified", "session_valid"}:
                inbox = await self._execute_reddit_adapter(
                    {"mode": "check_inbox"}, cap_engine=cap_engine
                )
                if inbox.get("ok") and "unread" in str(inbox.get("content", "")).lower():
                    self._emit_feed(
                        "Reddit Update",
                        "I have new Reddit notifications. Checking for replies to my comments.",
                        category="Social",
                    )
                elif inbox.get("status") == "login_unavailable":
                    self._emit_feed(
                        "Reddit Inbox",
                        "The saved session needs recovery; public browsing remains active.",
                        category="Social",
                    )
            elif provider_state:
                logger.info(
                    "Reddit authenticated inbox deferred in provider state %s; "
                    "public browsing remains active.",
                    provider_state,
                )

        except asyncio.CancelledError:
            raise
        except (
            ImportError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            OSError,
            TimeoutError,
        ) as e:
            _record_initiative_degradation(
                e,
                action="skipped reddit initiative tick after adapter failure",
                severity="warning",
            )
