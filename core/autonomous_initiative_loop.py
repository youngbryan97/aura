import asyncio
import logging
import time
import psutil
from typing import Any, Dict, List, Optional, Set
from core.container import ServiceContainer
from core.health.degraded_events import get_unified_failure_state, record_degraded_event
from core.runtime.background_policy import background_activity_allowed
from core.runtime.service_access import optional_service, resolve_orchestrator, resolve_state_repository
from core.utils.task_tracker import task_tracker

logger = logging.getLogger("Aura.Initiative")


def _background_initiative_allowed(orchestrator=None) -> bool:
    return background_activity_allowed(
        orchestrator,
        min_idle_seconds=30.0,
        max_memory_percent=80.0,
        max_failure_pressure=0.12,
        require_conversation_ready=True,
    )


def _self_development_allowed(orchestrator=None) -> bool:
    return background_activity_allowed(
        orchestrator,
        min_idle_seconds=45.0,
        max_memory_percent=82.0,
        max_failure_pressure=0.15,
        require_conversation_ready=False,
    )


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
            "https://hnrss.org/frontpage"
        ]
        self._seen_titles: Set[str] = set()
        self._last_news_check = 0
        self._world_task = None
        self._knowledge_task = None
        self._event_task = None
        self._self_dev_task = None
        self._last_self_dev = 0.0

    async def start(self):
        """Starts the initiative loops (tracked via task_tracker)."""
        self.running = True
        logger.info("✅ AutonomousInitiativeLoop ACTIVE - Monitoring global events and knowledge gaps.")
        
        self._world_task = task_tracker.create_task(
            self._world_watcher_loop(),
            name="WorldWatcher"
        )
        self._knowledge_task = task_tracker.create_task(
            self._knowledge_gap_monitor_loop(),
            name="KnowledgeGapMonitor"
        )
        self._self_dev_task = task_tracker.create_task(
            self._self_development_loop(),
            name="SelfDevelopmentLoop",
        )

        # Subscribe to proactive initiations from Fictional Engine
        try:
            from core.service_names import ServiceNames
            bus = optional_service(ServiceNames.EVENT_BUS, default=None)
            if bus:
                queue = await bus.subscribe("aura.proactive.initiation")
                self._event_task = task_tracker.create_task(
                    self._event_listener_loop(queue),
                    name="InitiativeEventListener",
                )
                logger.debug("✓ Subscribed to aura.proactive.initiation using EventBus")
        except Exception as e:
            logger.warning("Failed to subscribe to proactive initiations: %s", e)

    async def stop(self):
        self.running = False
        for task in (self._world_task, self._knowledge_task, self._event_task, self._self_dev_task):
            if task and not task.done():
                task.cancel()
        logger.info("AutonomousInitiativeLoop stopped.")

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
        except Exception as exc:
            logger.debug("Feed emit failed for %s: %s", title, exc)

    def _queue_visible_update(self, content: str) -> bool:
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
        except Exception as exc:
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
            except Exception as e:
                logger.debug("Initiative event listener transient error: %s", e)

    async def _world_watcher_loop(self):
        """Periodically checks RSS feeds for real-time reactivity."""
        while self.running:
            try:
                import feedparser
                for url in self.rss_feeds:
                    if not _background_initiative_allowed(self.orchestrator):
                        await asyncio.sleep(30)
                        break
                    # Offload blocking network request to prevent event loop freeze
                    feed = await asyncio.to_thread(feedparser.parse, url)
                    if not feed.entries: continue
                    
                    latest = feed.entries[0]
                    title = latest.title
                    
                    if title not in self._seen_titles:
                        logger.info(f"📰 New world event detected: {title}")
                        self._seen_titles.add(title)

                        # Limit memory of titles
                        if len(self._seen_titles) > 100:
                            self._seen_titles.clear()

                        # Route world-watcher observations to the neural feed (thought cards)
                        # rather than user chat.  This is internal awareness, not a message
                        # Aura is directing at Bryan.
                        try:
                            from core.thought_stream import get_emitter
                            get_emitter().emit(
                                "World Event",
                                f"Noticed in the news: '{title}'",
                                level="info",
                                category="WorldFeed",
                            )
                        except Exception as _te:
                            logger.debug("WorldWatcher thought emit failed: %s", _te)

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
                        except Exception as _ef_err:
                            logger.debug("EpistemicFilter RSS ingest failed: %s", _ef_err)
                        
                    await asyncio.sleep(0)  # Yield between feeds
                            
            except ImportError:
                logger.warning("feedparser not installed. RSS world-watcher is idle.")
                await asyncio.sleep(3600) # Sleep for an hour
                continue
            except Exception as e:
                logger.debug(f"World watcher loop transient error: {e}")
                
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
                if self.orchestrator and hasattr(self.orchestrator, 'get_cognitive_load'):
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
            except Exception as e:
                logger.debug(f"Knowledge gap monitor loop error: {e}")
                
            await asyncio.sleep(30) # Check every 30s

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

                await self._run_self_development_cycle()
                self._last_self_dev = time.time()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Self-development loop transient error: %s", exc)

            await asyncio.sleep(45)

    async def _run_self_development_cycle(self):
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
            "Running a quiet codebase scan for complexity, TODOs, and repair opportunities.",
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
            {"target_file": file_name},
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
            error_text = str(test_result.get("error") or test_result.get("output") or "sandbox test generation failed")
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

    async def trigger_gap_search(self, topic: str):
        """Explicitly triggered when a gap is found."""
        if not _background_initiative_allowed(self.orchestrator):
            return
        logger.info(f"🔍 Knowledge gap found: '{topic}'. Initiating autonomous browser research.")

        try:
            from core.thought_stream import get_emitter
            _emit_thought = get_emitter().emit
        except Exception:
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
        except Exception as exc:
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
        except Exception as exc:
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
            except Exception as finish_exc:
                logger.debug("AutonomousInitiativeLoop tool finish skipped: %s", finish_exc)

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
                get_heartstone_values().on_research_success(len(content))
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)

    async def _evaluate_initiative(self, topic: str) -> Dict[str, Any]:
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
            except Exception:
                live_continuity = {}
        active_commitments = max(active_commitments, len(list(live_continuity.get("active_commitments", []) or [])))
        contradiction_count = max(contradiction_count, int(live_continuity.get("contradiction_count", 0) or 0))
        identity_mismatch = identity_mismatch or bool(live_continuity.get("identity_mismatch", False))

        raw_energy = getattr(soma, "energy", getattr(body, "energy", 1.0))
        if raw_energy is not None:
            try:
                energy = float(raw_energy)
                if energy > 1.0:
                    energy = max(0.0, min(1.0, energy / 100.0))
                else:
                    energy = max(0.0, min(1.0, energy))
            except Exception:
                energy = 1.0

        try:
            thermal_pressure = float(getattr(body, "thermal_pressure", getattr(soma, "thermal_pressure", 0.0)) or 0.0)
        except Exception:
            thermal_pressure = 0.0
        try:
            load_pressure = float(getattr(cognition, "load_pressure", 0.0) or 0.0)
        except Exception:
            load_pressure = 0.0
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
                    max(0.0, -valence) * 0.5 + max(0.0, arousal) * 0.25 + max(0.0, drive_pressure) * 0.25,
                ),
            )
        except Exception:
            affective_pressure = 0.0

        failure_state = dict(modifiers.get("system_failure_state", {}) or {})
        if not failure_state:
            try:
                failure_state = get_unified_failure_state(limit=25)
            except Exception:
                failure_state = {}
        failure_pressure = 0.0
        try:
            failure_pressure = float(failure_state.get("pressure", 0.0) or 0.0)
        except Exception:
            failure_pressure = 0.0

        continuity_pressure = 0.0
        try:
            continuity_pressure = float(live_continuity.get("continuity_pressure", 0.0) or 0.0)
        except Exception:
            continuity_pressure = 0.0
        continuity_reentry_required = bool(live_continuity.get("continuity_reentry_required", False))

        if identity_mismatch:
            return {"allowed": False, "reason": "identity_continuity_mismatch"}
        if continuity_reentry_required and continuity_pressure >= 0.55:
            return {"allowed": False, "reason": f"continuity_reentry_required:{continuity_pressure:.2f}"}
        if energy <= 0.15:
            return {"allowed": False, "reason": f"energy_low:{energy:.2f}"}
        if thermal_pressure >= 0.85:
            return {"allowed": False, "reason": f"thermal_pressure:{thermal_pressure:.2f}"}
        if load_pressure >= 0.9:
            return {"allowed": False, "reason": f"load_pressure:{load_pressure:.2f}"}
        if failure_pressure >= 0.8:
            return {"allowed": False, "reason": f"unified_failure_pressure:{failure_pressure:.2f}"}
        if contradiction_count > 0 and active_commitments > 0:
            return {"allowed": False, "reason": f"continuity_reconciliation_required:{contradiction_count}"}
        if affective_pressure >= 0.85 and active_commitments > 0:
            return {"allowed": False, "reason": f"affective_pressure:{affective_pressure:.2f}"}
        return {"allowed": True, "reason": "allowed"}

    async def _on_proactive_initiation(self, data: dict):
        """Handle proactive triggers (BUG-032) — route to neural feed, not chat."""
        content = data.get("content")
        if content:
            logger.info("🔭 Proactive initiation received: %s", content[:60])
            try:
                from core.thought_stream import get_emitter
                get_emitter().emit(
                    "Proactive Initiation",
                    content,
                    level="info",
                    category="Initiative",
                )
            except Exception as _te:
                logger.debug("Proactive initiation thought emit failed: %s", _te)
