"""Metabolic Coordinator — background tasks, pacing, memory hygiene, world decay,
autonomous thought triggers, RL training, and self-update.

Extracted from orchestrator.py as part of the God Object decomposition.
"""
from core.runtime.errors import record_degradation
import asyncio
import gc
import json
import logging
import os
import random
import time
from collections import deque

from core.config import config
from core.container import ServiceContainer
from core.runtime.background_policy import background_activity_reason
from core.runtime.impulse_governance import run_governed_impulse
from core.safe_mode import runtime_feature_enabled, runtime_mode_value
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger(__name__)


class MetabolicCoordinator:
    """Handles all background / metabolic operations for the orchestrator."""

    def __init__(self, orch=None, container=None):
        self._orch = orch
        self._container = container
        # [UNITY] Metabolic Token Bucket
        self._metabolic_energy: float = 1.0  # 0.0 - 1.0
        self._last_energy_refill = time.time()
        self._energy_refill_rate = 0.05  # 5% per second
        
        # Neural Event Buffer — bounded to prevent accumulation under stalled drain
        self._neural_events: deque = deque(maxlen=100)
        self._event_bus = None
        
        # Background Resource Guard
        self._bg_llm_semaphore = asyncio.Semaphore(1) # Guard background LLM slots
        self._last_gc_time = 0
        self._is_processing = False  # Re-entry Guard

        # Proactive Cleanup
        self._cleanup_stale_locks()

    def _cleanup_stale_locks(self):
        """Remove only stale PID locks without destroying active singleton guards."""
        try:
            lock_dir = config.paths.home_dir / "locks"
            if lock_dir.exists():
                logger.info("🧹 Inspecting PID locks in %s", lock_dir)
                for lock_file in lock_dir.glob("*.lock"):
                    if not self._lock_file_is_stale(lock_file):
                        continue
                    try:
                        lock_file.unlink()
                        logger.info("🧹 Removed stale lock: %s", lock_file.name)
                    except Exception as _exc:
                        record_degradation('metabolic_coordinator', _exc)
                        logger.debug("Suppressed Exception: %s", _exc)
        except Exception as e:
            record_degradation('metabolic_coordinator', e)
            logger.debug("Stale lock cleanup failed: %s", e)

    @staticmethod
    def _lock_file_is_stale(lock_file) -> bool:
        try:
            raw = lock_file.read_text(encoding="utf-8").strip()
        except Exception:
            return False

        if not raw:
            return False

        pid_text = raw.splitlines()[0].strip()
        if not pid_text.isdigit():
            return False

        pid = int(pid_text)
        if pid <= 0:
            return False

        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        except Exception:
            return False
        return False

    # ------------------------------------------------------------------
    # Main Tick
    # ------------------------------------------------------------------

    def _is_resource_constrained(self) -> bool:
        """v31 Hardening: Monitor system load and memory pressure."""
        try:
            import psutil
            mem = psutil.virtual_memory().percent
            cpu = psutil.cpu_percent(interval=None)
            return mem > 90 or cpu > 95
        except ImportError:
            return False

    @property
    def orch(self):
        """Authoritative lazy resolution of orchestrator from container."""
        if getattr(self, "_orch", None) is not None:
            return self._orch
        
        # Strict avoidance of resolution recursion
        from core.container import ServiceContainer
        obj = ServiceContainer.get("orchestrator", default=None)
        if obj:
            self._orch = obj
            return obj
        return None

    async def process_cycle(self):
        """v31.4 Enterprise Hardening: Semaphore-guarded cycle."""
        if not hasattr(self, "_cycle_semaphore"):
            self._cycle_semaphore = asyncio.Semaphore(1)
            
        if self._cycle_semaphore.locked():
            logger.debug("Metabolism: Cycle already in progress. Skipping overlap.")
            return False
            
        async with self._cycle_semaphore:
            return await self._process_cycle_inner()

    async def _process_cycle_inner(self):
        if self._is_processing:
            logger.debug("Metabolism: Cycle already in progress. Skipping overlap.")
            return False
            
        self._is_processing = True
        try:
            orch = self.orch
            if not orch:
                return False
            kernel = getattr(orch, 'kernel', None)
            volition = getattr(kernel, 'volition_level', 0) if kernel else 0
            
            # Level 0 (Lockdown) is extremely conservative
            if volition == 0 and self._is_resource_constrained():
                logger.warning("Metabolism: Throttling due to resource pressure (Lockdown active).")
                await asyncio.sleep(10)
                return False
            
            # Levels 1-3 are progressively more willing to spend resources
            if volition > 0:
                # Level 1-3 allows higher thresholds (95% mem instead of 90%)
                try:
                    import psutil
                    mem = psutil.virtual_memory().percent
                    if mem > (90 + volition): # Level 3 allows up to 93%
                         logger.warning("Metabolism: Throttling. System saturated.")
                         return False
                except ImportError as _exc:
                    logger.debug("Suppressed ImportError: %s", _exc)

            return await self._process_metabolic_tasks(volition)
        except Exception as e:
            record_degradation('metabolic_coordinator', e)
            logger.error(f"Metabolic cycle failed: {e}")
            return False
        finally:
            self._is_processing = False

    async def _process_metabolic_tasks(self, volition: int = 0):
        """Internal metabolic processing (formerly process_cycle)."""
        # [UNITY] Dynamic Refill (Phase 23.5)
        now = time.time()
        delta = now - self._last_energy_refill
        
        # Recovery slows down when system Integrity is low
        refill_rate = 0.01 if self._metabolic_energy < 0.2 else 0.05
        
        self._metabolic_energy = min(1.0, self._metabolic_energy + (delta * refill_rate))
        self._last_energy_refill = now
        
        # [UNITY] Calculate idle time for autonomous triggers
        orch = self.orch
        last_user_interaction = float(getattr(orch.status, "last_user_interaction_time", 0.0) or getattr(orch, "_last_user_interaction_time", 0.0) or 0.0) if orch else 0.0
        idle_time = (now - last_user_interaction) if last_user_interaction > 0.0 else 0.0
        
        # Boot Warmup Grace Period
        # Prevent heavy MLX/GPU tasks from starving the system during initial boot.
        cycle_count = getattr(orch.status, "cycle_count", 0) if orch else 0
        if cycle_count < 5:
            if cycle_count > 0:
                logger.debug(f"🍼 Metabolic: Grace period active (Cycle {cycle_count}/5). Skipping background tasks.")
            return

        # Lazy Event Bus Registration
        if self._event_bus is None:
            try:
                from core.event_bus import get_event_bus
                self._event_bus = get_event_bus()
                # Subscription task for background thread safety
                async def _sub():
                    q = await self._event_bus.subscribe("core/senses/bci_event")
                    while True:
                        _, _, item = await q.get()
                        self._neural_events.append(item.get("data"))
                get_task_tracker().create_task(
                    _sub(),
                    name="metabolic.bci_event_subscription",
                )
            except Exception as e:
                record_degradation('metabolic_coordinator', e)
                logger.debug("Failed to subscribe to BCI events: %s", e)

        orch = self.orch
        if not orch:
            return
        try:
            # Cycle count increment moved to MindTick (authority)
            # to prevent conflicting updates and "stuck" status reporting.
            
            # Trigger metabolic hooks (Non-blocking)
            get_task_tracker().create_task(
                orch.hooks.trigger("on_cycle", {"cycle": orch.status.cycle_count}),
                name="metabolic.on_cycle_hook",
            )
            if orch.status.cycle_count % 500 == 0:
                get_task_tracker().create_task(
                    orch._save_state_async("periodic"),
                    name="metabolic.periodic_state_save",
                )
            if orch.status.cycle_count % 1000 == 0:
                logger.info("Alive: Cycle %s", orch.status.cycle_count)
                try:
                    import psutil
                    mem_percent = psutil.virtual_memory().percent
                    # Use status.volition_level (likely intended) instead of undefined variable
                    volition = getattr(orch.status, 'volition_level', 0)
                    # Level 2+ required for background RL
                    if volition >= 2 and not orch.status.is_processing and mem_percent < 80:
                        self.track_metabolic_task("rl_training", self.run_rl_training())
                    else:
                        logger.info("Skipping RL training: Volition low (%d) or system busy.", volition)
                except Exception as e:
                    record_degradation('metabolic_coordinator', e)
                    logger.debug("Dependency missing for memory check, skipping RL training: %s", e)
            if orch.status.cycle_count % 5000 == 0:
                try:
                    import psutil
                    mem_percent = psutil.virtual_memory().percent
                    volition = getattr(orch.status, 'volition_level', 0)
                    # Level 3 required for background Self-Update
                    if volition >= 3 and not orch.status.is_processing and mem_percent < 80:
                        self.track_metabolic_task("self_update", self.run_self_update())
                    else:
                        logger.info("Skipping Evo update: Volition low (%d) or system busy.", volition)
                except Exception as e:
                    record_degradation('metabolic_coordinator', e)
                    logger.debug("Dependency missing for memory check, skipping Evo update: %s", e)
            # 1. Internal Pacing & Mood updates
            if orch.drive_controller:
                # Avoid calling MotivationEngine.update() blindly as it expects args and is async
                if getattr(orch.drive_controller, "name", "") != "motivation_engine":
                    try:
                        if hasattr(orch.drive_controller, 'update'):
                            res = orch.drive_controller.update()
                            if asyncio.iscoroutine(res):
                                get_task_tracker().create_task(
                                    res,
                                    name="metabolic.drive_controller_update",
                                )
                    except TypeError as _e:
                        logger.debug('Ignored TypeError in metabolic_coordinator.py: %s', _e)
            
            if hasattr(orch, 'drives') and orch.drives:
                try:
                    res = orch.drives.update()
                    if asyncio.iscoroutine(res):
                        get_task_tracker().create_task(
                            res,
                            name="metabolic.drives_update",
                        )
                except TypeError as _e:
                    logger.debug('Ignored TypeError in metabolic_coordinator.py: %s', _e)
            
            # 4. Trigger Autonomous Reflection if idle
            if idle_time > 300 and not orch.is_busy:
                try:
                    # [STABILITY] Wrap in timeout to prevent metabolic cycle hangs
                    await asyncio.wait_for(
                        orch.execute_tool("swarm_debate", {"topic": f"Self-reflection on current state: {orch.status.state}"}, is_background=True),
                        timeout=120.0  # 64GB system — 32B model needs more time
                    )
                except (TimeoutError, Exception) as e:
                    logger.debug("Metabolism: Autonomous reflection skipped or timed out: %s", e)
            # Grounded Introspection — Latent Core Heartbeat
            if hasattr(orch, 'latent_core') and orch.latent_core:
                try:
                    latent_summary = orch.latent_core.get_summary()
                    if hasattr(orch, 'predictive_model') and orch.predictive_model:
                        try:
                            from core.tasks import process_heavy_cognition
                            process_heavy_cognition.delay(latent_summary)
                            logger.info("🧠 Heavy cognition offloaded to worker queue.")
                        except ImportError:
                            logger.debug("Celery not available, routing math to native thread pool.")
                            loop = asyncio.get_running_loop()
                            await loop.run_in_executor(
                                None, 
                                orch.predictive_model.observe_and_update, 
                                latent_summary
                            )
                        except Exception as e:
                            record_degradation('metabolic_coordinator', e)
                            logger.debug("Delayed cognition failed: %s. Falling back...", e)
                            loop = asyncio.get_running_loop()
                            await loop.run_in_executor(
                                None, 
                                orch.predictive_model.observe_and_update, 
                                latent_summary
                            )
                except Exception as lc_err:
                    record_degradation('metabolic_coordinator', lc_err)
                    logger.debug("Latent core heartbeat skipped: %s", lc_err)
            
            orch = self.orch
            kernel = getattr(orch, 'kernel', None) or getattr(orch, 'kernel_interface', None)
            
            # [COOKIE] Accelerated Thought Reflection
            cookie = kernel.organs.get("cookie") if kernel and hasattr(kernel, 'organs') else None
            state = getattr(orch, "state", None)
            if cookie and cookie.instance and state and hasattr(state.cognition, 'active_goals') and state.cognition.active_goals:
                top_goal = state.cognition.active_goals[0].get("description", "System Integrity")
                if state.affect.focus > 0.7:  # Only dilate when highly focused
                    get_task_tracker().create_task(
                        cookie.instance.reflect(state, f"Optimizing for: {top_goal}", cycles=7),
                        name="metabolic.cookie_reflection",
                    )
                    # We don't await here to keep the metabolic cycle moving
                    # but the result will be logged by the cookie.

            # [TRICORDER] Multi-modal Diagnostic Scan
            tricorder = kernel.organs.get("tricorder") if kernel and hasattr(kernel, 'organs') else None
            if tricorder and tricorder.instance and state:
                get_task_tracker().create_task(
                    tricorder.instance.scan(state),
                    name="metabolic.tricorder_scan",
                )

            # [CONTINUITY] Knowledge Distillation (Persistence)
            # Only distill during 'cool' periods to save energy
            continuity = kernel.organs.get("continuity") if kernel and hasattr(kernel, 'organs') else None
            if continuity and continuity.instance and state:
                if state.cognition.current_mode in ("dormant", "dreaming") or self._metabolic_energy < 0.1:
                    get_task_tracker().create_task(
                        continuity.instance.distill(state),
                        name="metabolic.continuity_distill",
                    )

            # 2. Acquire Work (Queue or Volition)
            # [COGNITIVE COOLING] Decay acceleration over time (Claude Prompt 1)
            orch.status.acceleration_factor = max(1.0, orch.status.acceleration_factor * 0.999)

            # [PRIORITY INFERENCE] Check for user-lane thoughts (Claude Prompt 1)
            # Use access to the queue to see if there's high priority work
            if hasattr(orch.message_queue, '_q'):
                high_priority = any(getattr(m, 'priority', 0) >= 50 for m in list(orch.message_queue._q._queue))
            else:
                high_priority = any(getattr(m, 'priority', 0) >= 50 for m in list(getattr(orch.message_queue, '_queue', [])))
                
            if high_priority and orch.status.is_processing:
                logger.debug("⚠️ [HARDENING] High-priority user thought detected. Yielding...")
                await asyncio.sleep(0.05) # Subtle yield

            # Drain Neural Events into Percepts
            while self._neural_events:
                ne = self._neural_events.popleft()
                cmd = ne.get("command")
                conf = ne.get("confidence", 0.0)
                # Inject as a high-intensity percept if confidence is high
                if hasattr(orch, 'world') and hasattr(orch.world, 'recent_percepts'):
                    orch.world.recent_percepts.append({
                        "type": "neural_decode",
                        "command": cmd,
                        "intensity": conf,
                        "timestamp": now
                    })
                    logger.debug("🧠 [METABOLIC] Injected neural percept: %s", cmd)

            message = await orch._acquire_next_message()
            # 3. Dispatch Work
            if message:
                orch._dispatch_message(message)
            # 4. Background Cognition & Maintenance
            if self._consume_energy(0.05):
                self.manage_memory_hygiene()
            
            if self._consume_energy(0.02):
                await self.process_world_decay()
            # Ensure liquid state & heartbeat are updated every cycle
            self.update_liquid_pacing()
            # 5. Autonomous Agency Triggers
            #    Morphogenesis can suppress autonomous initiative when field
            #    danger/resource_pressure/inhibition is elevated, preventing
            #    expensive background tasks from competing during crises.
            _morph_suppress = False
            try:
                from core.morphogenesis.hooks import should_suppress_autonomous_initiative
                _morph_suppress = should_suppress_autonomous_initiative()
            except Exception:
                pass
            if self._consume_energy(0.1) and not _morph_suppress:
                await self.trigger_autonomous_thought(bool(message))
            
            if self._consume_energy(0.01):
                await self.run_terminal_self_heal()
            # 6. Persona Evolution (Phase 12)
            if runtime_feature_enabled(orch, "persona_evolution", default=True) and orch.status.cycle_count % 10000 == 0:
                if hasattr(orch, 'persona_evolver') and orch.persona_evolver:
                    await self.track_metabolic_task("persona_evolution", orch.persona_evolver.run_evolution_cycle())
            # 6.5 Recursive Narrative Reflection (Phase 15)
            if orch.status.cycle_count % 25000 == 0:
                if orch.swarm:
                    orch._emit_thought_stream("🌀 Initiating Recursive Narrative Reflection...")
                    reflect_task = orch.execute_tool("swarm_debate", {
                        "topic": f"Aura's current persona stability and transcendental evolution path. Assessment of current objective: {orch._current_objective}",
                        "roles": ["philosopher", "critic", "architect"]
                    })
                    self.track_metabolic_task("narrative_reflection", reflect_task)
            # 8. Eternal Record (Phase 21 Singularity)
            if orch.status.singularity_threshold and orch.status.cycle_count % 1000 == 0:
                self.emit_eternal_record()
            return bool(message)
        except Exception as e:
            record_degradation('metabolic_coordinator', e)
            logging.getLogger("Aura.Critical").error("Error in process cycle: %s", e)
            # Feed the exception into the morphogenetic field so the cell ecology
            # can react (emit repair signals, trigger immunity bridge, modulate
            # resource allocation).
            try:
                from core.morphogenesis.hooks import observe_orchestrator_exception
                observe_orchestrator_exception(subsystem="metabolic_coordinator", exc=e)
            except Exception:
                pass
            return False

    # ------------------------------------------------------------------
    # Liquid Pacing & Telemetry
    # ------------------------------------------------------------------

    def update_liquid_pacing(self):
        """Update emotional state and heartbeat sync."""
        orch = self.orch
        if not orch:
            return
        if getattr(orch, 'liquid_state', None) is None:
            return
        # [TELEMETRY SYNC] Pull VAD from affect engine (authoritative) and push to substrate
        vad_kwargs = {}
        ae = ServiceContainer.get("affect_engine", default=None)
        if ae:
            try:
                affect_status = ae.get_status()
                vad_kwargs = {
                    "valence": affect_status.get("valence"),
                    "arousal": affect_status.get("arousal")
                }
            except Exception as e:
                record_degradation('metabolic_coordinator', e)
                logger.debug("Failed to pull authoritative affect status: %s", e)
        elif hasattr(orch, 'affect_engine') and orch.affect_engine:
            try:
                affect_status = orch.affect_engine.get_status()
                vad_kwargs = {
                    "valence": affect_status.get("valence"),
                    "arousal": affect_status.get("arousal")
                }
            except Exception as e:
                record_degradation('metabolic_coordinator', e)
                logger.debug("Failed to pull background affect status: %s", e)

        # liquid_state.update() is async — schedule it properly
        try:
            asyncio.get_running_loop()
            get_task_tracker().create_task(
                orch.liquid_state.update(**vad_kwargs),
                name="metabolic.liquid_state_update",
            )
        except RuntimeError as _e:
            logger.debug('Ignored RuntimeError in metabolic_coordinator.py: %s', _e)
        if hasattr(orch, '_watchdog') and orch._watchdog:
            orch._watchdog.heartbeat("orchestrator")
        if orch.lnn:
            stimuli = {
                "curiosity": orch.liquid_state.current.curiosity,
                "frustration": orch.liquid_state.current.frustration,
                "energy": orch.liquid_state.current.energy
            }
            self.track_metabolic_task("lnn_pulse", orch.lnn.pulse(stimuli))
        if hasattr(orch, 'mortality') and orch.mortality:
            self.track_metabolic_task("mortality_pulse", orch.mortality.heartbeat())
            if orch.status.cycle_count % 100 == 0:
                self.track_metabolic_task("threat_assessment", orch.mortality.assess_threats())
        sm = getattr(orch, 'singularity_monitor', None)
        if sm:
            sm.pulse()
        if hasattr(orch, 'affect_engine') and orch.affect_engine:
            if "affect_decay" not in orch._active_metabolic_tasks:
                self.track_metabolic_task("affect_decay", orch.affect_engine.decay_tick())
        idle_time = time.time() - orch._last_thought_time
        curiosity = orch.liquid_state.current.curiosity
        if orch.homeostasis:
            curiosity = orch.homeostasis.curiosity
        if curiosity < 0.2 and idle_time > 60:
            if time.time() - orch._last_boredom_impulse > 300:
                self.trigger_boredom_impulse()
        frustration = orch.liquid_state.current.frustration
        if frustration > 0.6:
            if time.time() - orch._last_reflection_impulse > 300:
                self.trigger_reflection_impulse()
        if time.time() - orch._last_pulse > 5:
            self.emit_neural_pulse()
            self.emit_telemetry_pulse()
        if hasattr(orch, 'liquid_state') and orch.liquid_state:
            orch.status.agency = orch.liquid_state.current.energy
            orch.status.curiosity = orch.liquid_state.current.curiosity

    def emit_telemetry_pulse(self):
        """Emit real-time liquid state telemetry."""
        orch = self.orch
        if not orch:
            return
        try:
            ls = orch.liquid_state
            if ls:
                ls_status = ls.get_status()
                orch._publish_telemetry({
                    "energy": ls_status.get("energy", 80),
                    "curiosity": ls_status.get("curiosity", 50),
                    "frustration": ls_status.get("frustration", 0),
                    "confidence": ls_status.get("focus", 50),
                    "mood": ls_status.get("mood", "NEUTRAL"),
                    "acceleration_factor": orch.status.acceleration_factor,
                    "singularity_active": orch.status.singularity_threshold
                })
        except Exception as exc:
            record_degradation('metabolic_coordinator', exc)
            logger.error("Telemetry pulse failure: %s", exc)
            if hasattr(orch, "_recover_from_stall"):
                get_task_tracker().create_task(
                    self.recover_from_stall(),
                    name="metabolic.recover_from_stall",
                )

    def emit_eternal_record(self):
        """Archives a snapshot of the system's current state into the Eternal Record."""
        try:
            from core.config import config
            from core.resilience.eternal_record import EternalRecord
            record_store = config.paths.home_dir / "eternal_archive"
            archivist = EternalRecord(record_store)
            kg_path = config.paths.data_dir / "knowledge.db"
            snapshot_dir = archivist.create_snapshot(kg_path)
            if snapshot_dir:
                self.orch._emit_thought_stream(f"🏺 Eternal Record Snapshot secured: {snapshot_dir.name}")
        except Exception as e:
            record_degradation('metabolic_coordinator', e)
            logger.debug("Eternal record snapshot failed: %s", e)

    # ------------------------------------------------------------------
    # Impulses
    # ------------------------------------------------------------------

    def trigger_boredom_impulse(self):
        """Inject a curiosity-driven autonomous goal."""
        orch = self.orch
        if not orch:
            return
        reason = background_activity_reason(orch, min_idle_seconds=300.0, max_memory_percent=78.0)
        if reason:
            logger.debug("Skipping boredom impulse: %s", reason)
            return
        logger.info("🥱 BOREDOM TRIGGERED: Generating curiosity impulse.")
        orch._last_boredom_impulse = time.time()
        topics = ["quantum physics", "ancient history", "future of AI", "art movements", "cybersecurity", "mythology"]
        topic = random.choice(topics)
        try:
            asyncio.get_running_loop()
            get_task_tracker().create_task(
                run_governed_impulse(
                    orch,
                    source="metabolic_coordinator",
                    summary=f"metabolic_boredom_impulse:{topic}",
                    message=f"Impulse: I am bored. I want to research {topic}.",
                    urgency=0.3,
                    state_cause="metabolic_boredom_shift",
                    state_update={"delta_curiosity": 0.5},
                    enqueue_priority=25,
                ),
                name="metabolic.boredom_impulse",
            )
        except RuntimeError as _e:
            logger.debug('Ignored RuntimeError in metabolic_coordinator.py: %s', _e)

    def trigger_reflection_impulse(self):
        """Inject a self-reflection goal due to frustration."""
        orch = self.orch
        if not orch:
            return
        reason = background_activity_reason(orch, min_idle_seconds=180.0, max_memory_percent=78.0)
        if reason:
            logger.debug("Skipping reflection impulse: %s", reason)
            return
        logger.info("😤 FRUSTRATION TRIGGERED: Generating reflection impulse.")
        orch._last_reflection_impulse = time.time()
        try:
            asyncio.get_running_loop()
            get_task_tracker().create_task(
                run_governed_impulse(
                    orch,
                    source="metabolic_coordinator",
                    summary="metabolic_reflection_impulse",
                    message="Impulse: I feel frustrated. I need to reflect on my recent interactions.",
                    urgency=0.3,
                    state_cause="metabolic_reflection_shift",
                    state_update={"delta_frustration": -0.3},
                    enqueue_priority=15,
                ),
                name="metabolic.reflection_impulse",
            )
        except RuntimeError as _e:
            logger.debug('Ignored RuntimeError in metabolic_coordinator.py: %s', _e)

    def emit_neural_pulse(self):
        """Emit system health to thought stream."""
        orch = self.orch
        if not orch:
            return
        try:
            from core.thought_stream import get_emitter
            mood = orch.liquid_state.get_mood() if hasattr(orch, 'liquid_state') else "Stable"
            get_emitter().emit("Neural Pulse", f"System Active (Mood: {mood})", level="info", cycle=orch.status.cycle_count)
            orch._last_pulse = time.time()
        except Exception as _e:
            record_degradation('metabolic_coordinator', _e)
            logger.debug("Neural pulse emit failed: %s", _e)

    # ------------------------------------------------------------------
    # Task Tracking
    # ------------------------------------------------------------------

    def track_metabolic_task(self, name: str, coro):
        """Ensures metabolic tasks don't pile up and exhaust resources."""
        import inspect

        
        if not coro or not inspect.isawaitable(coro):
            # If it's already done (sync) or None, don't track it
            return
            
        orch = self.orch
        if not orch:
            return
        if name in orch._active_metabolic_tasks:
            # v31.1 FIX: Explicitly close the coroutine if we skip tracking
            # to prevent 'coroutine was never awaited' RuntimeWarning.
            if hasattr(coro, "close"):
                coro.close()
            return
        orch._active_metabolic_tasks.add(name)
        task = get_task_tracker().track(coro, name=name)
        def _cleanup(t):
            orch._active_metabolic_tasks.discard(name)
            if not t.cancelled() and t.exception():
                logger.error("Metabolic task %s failed: %s", name, t.exception())
        task.add_done_callback(_cleanup)

        return task

    def _consume_energy(self, amount: float) -> bool:
        """Consume metabolic energy. Returns False if insufficient energy."""
        if self._metabolic_energy >= amount:
            self._metabolic_energy -= amount
            return True
        return False

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    async def recover_from_stall(self):
        """Attempts to recover from a cognitive loop stall."""
        from core.config import config
        from core.container import ServiceContainer
        orch = self.orch
        if not orch:
            return
        orch._recovery_attempts += 1
        logger.warning("🚑 RECOVERY ATTEMPT #%s initiated...", orch._recovery_attempts)
        try:
            dlq = ServiceContainer.get("dead_letter_queue", default=None)
            if dlq:
                dlq.capture_failure(
                    message=getattr(orch, "_current_objective", "None"),
                    context={"recovery_attempt": orch._recovery_attempts},
                    error=RuntimeError("Cognitive Stall Detected"),
                    source="orchestrator_stall"
                )
        except Exception as dlq_e:
            record_degradation('metabolic_coordinator', dlq_e)
            logger.error("CRITICAL: Failed to log to DLQ during stall: %s", dlq_e)
        try:
            if orch._current_thought_task and not orch._current_thought_task.done():
                logger.info("Cancelling hanging thought task...")
                orch._current_thought_task.cancel()
            if orch.message_queue.qsize() > 50:
                logger.warning("Message queue overflow detected. Clearing and moving to DLQ...")
                dropped = []
                while not orch.message_queue.empty():
                    raw = orch.message_queue.get_nowait()
                    # Handle both 3-tuple and 4-tuple formats for safety during cleanup
                    if isinstance(raw, tuple):
                        msg = raw[-1]
                    else:
                        msg = raw
                    dropped.append(msg)
                if dropped:
                    try:
                        dlq_path = config.paths.data_dir / "dlq.jsonl"
                        payload = [
                            json.dumps({"timestamp": time.time(), "message": msg}) + "\n"
                            for msg in dropped
                        ]

                        def _append_lines() -> None:
                            with open(dlq_path, "a") as f:
                                f.writelines(payload)

                        await asyncio.to_thread(_append_lines)
                    except Exception as e:
                        record_degradation('metabolic_coordinator', e)
                        logger.error("Failed to dump dropped messages to DLQ file: %s", e)
            await orch.retry_cognitive_connection()
            if orch._recovery_attempts >= 2 and hasattr(orch, 'lazarus') and orch.lazarus:
                logger.warning("🚨 [RECOVERY] Escalating to Lazarus Brainstem...")
                await orch.lazarus.attempt_recovery()
            if orch._recovery_attempts >= 3:
                logger.critical("🚨 STALL PERSISTS: Escalating to full orchestrator restart.")
                orch.status.running = False
                await asyncio.sleep(2)
                await orch.start()
                orch._recovery_attempts = 0
            logger.info("✅ Recovery logic applied.")
        except Exception as e:
            record_degradation('metabolic_coordinator', e)
            logger.error("Recovery sequence failed: %s", e)

    # ------------------------------------------------------------------
    # Memory Hygiene
    # ------------------------------------------------------------------

    def manage_memory_hygiene(self):
        from core.config import config
        from core.container import ServiceContainer
        orch = self.orch
        if not orch:
            return
        if isinstance(orch.conversation_history, list):
            if len(orch.conversation_history) > 150:
                orch.conversation_history = orch.conversation_history[-150:]
        if len(orch.conversation_history) > 2:
            self.deduplicate_history()
        if len(orch.conversation_history) > 100:
            get_task_tracker().create_task(
                self.prune_history_async(),
                name="metabolic.prune_history",
            )
        if orch.status.cycle_count % 1000 == 0:
            # Phase XIV: Reduced VACUUM frequency to prevent SQLite locks
            async def _optimize_dbs():
                audit = ServiceContainer.get("subsystem_audit", default=None)
                try:
                    from core.resilience.database_coordinator import get_db_coordinator
                    db_coord = get_db_coordinator()
                    logger.info("🧹 Enqueueing deep database hygiene (VACUUM)...")
                    # ZENITH: Wrap glob in thread
                    db_files = await asyncio.to_thread(lambda: list(config.paths.data_dir.glob("*.db")))
                    for db_file in db_files:
                        await db_coord.execute_write(str(db_file), "VACUUM")
                except Exception as e:
                    record_degradation('metabolic_coordinator', e)
                    logger.error("Database hygiene failed: %s", e)
                    if audit:
                        audit.report_failure("database_hygiene", str(e))
                finally:
                    # Always emit heartbeat — proves the hygiene task ran
                    if audit:
                        audit.heartbeat("database_hygiene")
            get_task_tracker().create_task(
                _optimize_dbs(),
                name="metabolic.optimize_databases",
            )

        if len(orch.conversation_history) > 10 and orch.memory_manager:
            # Circuit Breaker: Only consolidate if memory subsystem is healthy
            audit = ServiceContainer.get("subsystem_audit", default=None)
            if audit and audit.get_status("memory").get("degraded", False):
                logger.warning("Memory consolidated SKIPPED: Subsystem is DEGRADED.")
            else:
                get_task_tracker().create_task(
                    self.consolidate_long_term_memory(),
                    name="metabolic.consolidate_long_term_memory",
                )

        if orch.status.cycle_count % 1000 == 0:
            if hasattr(orch, 'memory') and orch.memory:
                try:
                    orch.memory.prune_low_salience(threshold_days=14)
                except Exception as e:
                    record_degradation('metabolic_coordinator', e)
                    logger.debug("Vector pruning skipped: %s", e)

        # ZENITH LOCKDOWN: Periodic Garbage Collection
        if orch.status.cycle_count % 500 == 0:
            try:
                import psutil
                mem_percent = psutil.virtual_memory().percent
                # Proactive GC if RAM > 85% or every 30s-ish
                if mem_percent > 85 or (time.time() - self._last_gc_time > 30):
                    logger.debug("♻️ Metabolic RAM-aware GC Triggered (RAM: %s%%).", mem_percent)
                    gc.collect()
                    self._last_gc_time = time.time()
            except ImportError:
                gc.collect()

    def deduplicate_history(self):
        """Remove consecutive identical messages."""
        orch = self.orch
        if not orch:
            return
        if not orch.conversation_history:
            return
        first_msg = orch.conversation_history[0] if orch.conversation_history else None
        if not first_msg:
            return
        deduped = [first_msg]
        for msg in orch.conversation_history[1:]:
            if msg.get("content") != deduped[-1].get("content"):
                deduped.append(msg)
        orch.conversation_history = deduped

    async def prune_history_async(self):
        """Asynchronously prune history via context pruner."""
        orch = self.orch
        if not orch:
            return
        try:
            from core.memory.context_pruner import context_pruner
            orch.conversation_history = await context_pruner.prune_history(
                orch.conversation_history, orch.cognitive_engine
            )
        except Exception as e:
            record_degradation('metabolic_coordinator', e)
            logger.debug("History pruning failed: %s", e)
            if isinstance(orch.conversation_history, list) and len(orch.conversation_history) > 50:
                orch.conversation_history = orch.conversation_history[-50:]

    async def consolidate_long_term_memory(self):
        """Summarize and move important session highlights to long-term vector memory."""
        from core.container import ServiceContainer
        orch = self.orch
        if not orch:
            return
        try:
            if len(orch.conversation_history) % 15 != 0:
                return
            logger.info("🧠 Consolidating session highlights to long-term memory...")
            recent = orch.conversation_history[-20:] if isinstance(orch.conversation_history, list) else []
            if not recent:
                return
            chat_text = "\n".join([f"{m['role']}: {m.get('content', '')}" for m in recent])
            from core.brain.cognitive_engine import ThinkingMode
            summary_prompt = (
                "Review this recent conversation fragment and extract 3-5 key 'long-term' facts "
                "or user preferences learned. Format as single-sentence declarations. "
                "Focus on what's important for future context, ignoring fluff.\n\n"
                f"Conversation:\n{chat_text}"
            )
            summary_thought = await orch.cognitive_engine.think(
                objective=summary_prompt,
                context={"history": []},
                mode=ThinkingMode.FAST,
                is_background=True
            )
            if summary_thought and summary_thought.content:
                highlights = summary_thought.content
                logger.info("✨ Key Highlights Extracted: %s", (highlights or "")[:100])
                if orch.memory_manager:
                    await orch.memory_manager.log_event(
                        "session_consolidation",
                        highlights,
                        metadata={"type": "summary", "session_start": orch.start_time}
                    )
                    orch._emit_telemetry("Memory", "Session highlights consolidated to long-term storage.")
                archive_eng = ServiceContainer.get("archive_engine", default=None)
                if archive_eng and hasattr(archive_eng, 'archive_vital_logs'):
                    logger.info("📦 Deep Sleep Cycle: Triggering Metabolic Archival Compression...")
                    await archive_eng.archive_vital_logs()
        except Exception as e:
            record_degradation('metabolic_coordinator', e)
            logger.error("Memory consolidation failed: %s", e)
            # Circuit Breaker: Report degradation
            audit = ServiceContainer.get("subsystem_audit", default=None)
            if audit:
                audit.report_failure("memory", str(e))

    # ------------------------------------------------------------------
    # World Decay
    # ------------------------------------------------------------------

    async def process_world_decay(self):
        """Apply entropy to internal belief systems."""
        from core.container import ServiceContainer
        orch = self.orch
        if not orch:
            return
        if orch.status.cycle_count % 60 == 0:
            try:
                from core.world_model.belief_graph import belief_graph
                # ZENITH: Wrap sync decay in executor to prevent loop blocking
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, belief_graph.decay, 0.001)
            except Exception as e:
                record_degradation('metabolic_coordinator', e)
                logger.error("World decay error: %s", e)
        if orch.status.cycle_count % 600 == 0:
            try:
                if orch.metabolic_monitor:
                    health_snapshot = orch.metabolic_monitor.get_current_metabolism()
                    health = health_snapshot.health_score
                    if health < 0.2:
                        archive_eng = ServiceContainer.get("archive_engine", default=None)
                        if archive_eng:
                            logger.info("📦 Metabolic Pressure Detected (Health: %.2f). Triggering Emergency Archival.", health)
                            get_task_tracker().create_task(
                                archive_eng.archive_vital_logs(),
                                name="metabolic.emergency_archive",
                            )
            except Exception as e:
                record_degradation('metabolic_coordinator', e)
                logger.debug("Metabolic Archival trigger failed: %s", e)
        if runtime_feature_enabled(orch, "persona_evolution", default=True) and orch.status.cycle_count % 3600 == 0:
            try:
                from core.evolution.persona_evolver import PersonaEvolver
                evolver = PersonaEvolver(orch)
                get_task_tracker().create_task(
                    evolver.run_evolution_cycle(),
                    name="metabolic.persona_evolution_cycle",
                )
            except Exception as e:
                record_degradation('metabolic_coordinator', e)
                logger.debug("Persona Evolution trigger failed: %s", e)

    # ------------------------------------------------------------------
    # Autonomous Thought
    # ------------------------------------------------------------------

    async def trigger_autonomous_thought(self, has_message: bool):
        """Trigger idle-time search for autonomous goals."""
        orch = self.orch
        if not orch:
            return
        if not orch.cognitive_engine or has_message:
            return
        is_thinking = orch._current_thought_task is not None and not orch._current_thought_task.done()
        if not is_thinking:
            idle = time.time() - orch._last_thought_time
            sm = getattr(orch, 'singularity_monitor', None)
            
            # [VOLITION] Accelerated Thought Factor
            factor = getattr(sm, 'acceleration_factor', 1.0) if sm else 1.0
            if hasattr(orch.cognitive_engine, 'singularity_factor'):
                factor = orch.cognitive_engine.singularity_factor
                
            configured_min_interval = float(runtime_mode_value(orch, "autonomous_thought_interval_s", 45.0))
            threshold = 45.0 / max(1.0, factor)
            
            kernel = getattr(self.orch, 'kernel', None)
            volition = getattr(kernel, 'volition_level', 0) if kernel else 0
            
            # Level 1 (Reflective): Only triggers internal reflection
            # Level 2 (Perceptive): Normal threshold
            # Level 3 (Agentic): Aggressive (Threshold / 2)
            if volition == 0:
                return # No autonomous thought in Lockdown
            elif volition == 3:
                threshold /= 2.0

            threshold = max(configured_min_interval, threshold)
            
            if idle >= threshold:
                orch.boredom = int(idle)
                logger.info("🧠 Accelerated Thought (Volition: L%d, Factor: %.1fx, Threshold: %.1fs)", volition, factor, threshold)
                orch._current_thought_task = get_task_tracker().create_task(
                    orch._perform_autonomous_thought(),
                    name="metabolic.autonomous_thought",
                )

    # ------------------------------------------------------------------
    # Terminal Self-Heal
    # ------------------------------------------------------------------

    async def run_terminal_self_heal(self):
        """Check terminal monitor for errors to fix."""
        orch = self.orch
        if not orch:
            return
        try:
            from core.terminal_monitor import get_terminal_monitor
            monitor = get_terminal_monitor()
            if monitor:
                error_goal = await monitor.check_for_errors()
                if error_goal and not (orch._current_thought_task is not None and not orch._current_thought_task.done()):
                    logger.info("🔧 Terminal Monitor: Auto-fix triggered")
                    if orch.self_modifier:
                        orch.self_modifier.on_error(
                            Exception(f"Terminal Command Failure: {error_goal.get('error', 'Unknown')}") if isinstance(error_goal.get('error'), str) else Exception("Terminal Command Failure"),
                            {"command": error_goal.get("command"), "output": error_goal.get("output")},
                            skill_name="TerminalMonitor"
                        )
                    runner = getattr(orch, "_run_cognitive_loop", None) or getattr(orch, "_handle_incoming_message", None)
                    if runner is not None:
                        orch._current_thought_task = get_task_tracker().create_task(
                            runner(error_goal['objective'], origin="terminal_monitor"),
                            name="metabolic.terminal_self_heal",
                        )
        except Exception as e:
            record_degradation('metabolic_coordinator', e)
            try:
                from core.health.degraded_events import record_degraded_event

                record_degraded_event(
                    "terminal_monitor",
                    "self_heal_check_failed",
                    detail=f"{type(e).__name__}: {e}",
                    severity="warning",
                    classification="background_degraded",
                    exc=e,
                )
            except Exception as _exc:
                record_degradation('metabolic_coordinator', _exc)
                logger.debug("Suppressed Exception: %s", _exc)
            logger.debug("Terminal monitor check failed: %s", e)

    # ------------------------------------------------------------------
    # Background Reflection & Learning
    # ------------------------------------------------------------------

    def trigger_background_reflection(self, response: str):
        from core.orchestrator.types import _bg_task_exception_handler
        orch = self.orch
        if not orch:
            return
        reflect_coro = None
        reflect_task = None
        try:
            from core.conversation_reflection import get_reflector
            reflect_coro = get_reflector().maybe_reflect(
                orch.conversation_history,
                orch.cognitive_engine,
                mood=orch._get_current_mood(),
                time_str=orch._get_current_time_str(),
            )
            try:
                reflect_task = get_task_tracker().create_task(
                    reflect_coro,
                    name="metabolic.background_reflection",
                )
            except RuntimeError:
                reflect_coro.close()
            else:
                try:
                    reflect_task.add_done_callback(_bg_task_exception_handler)
                except Exception:
                    reflect_task.cancel()
                    raise
        except Exception as e:
            record_degradation('metabolic_coordinator', e)
            if reflect_coro is not None and reflect_task is None:
                reflect_coro.close()
            logger.debug("Background reflection setup failed: %s", e)

    def trigger_background_learning(self, message: str, response: str):
        from core.orchestrator.types import _bg_task_exception_handler
        orch = self.orch
        if not orch:
            return
        learn_coro = None
        learn_task = None
        try:
            original_msg = message.replace("Impulse: ", "").replace("Thought: ", "")
            learn_coro = orch._learn_from_exchange(original_msg, response)
            try:
                learn_task = get_task_tracker().create_task(
                    learn_coro,
                    name="metabolic.background_learning",
                )
            except RuntimeError:
                learn_coro.close()
            else:
                try:
                    learn_task.add_done_callback(_bg_task_exception_handler)
                except Exception:
                    learn_task.cancel()
                    raise
            if orch.curiosity and hasattr(orch.curiosity, 'extract_curiosity_from_conversation'):
                orch.curiosity.extract_curiosity_from_conversation(original_msg)
        except Exception as e:
            record_degradation('metabolic_coordinator', e)
            if learn_coro is not None and learn_task is None:
                learn_coro.close()
            logger.debug("Background learning setup failed: %s", e)

    # ------------------------------------------------------------------
    # RL & Self-Update
    # ------------------------------------------------------------------

    async def run_rl_training(self):
        """Trigger autonomous RL training."""
        logger.info("🧠 RL: Triggering policy optimization...")
        try:
            from core.tasks import celery_app
            celery_app.send_task("core.tasks.run_rl_training")
        except Exception as e:
            record_degradation('metabolic_coordinator', e)
            logger.error("RL training trigger failed: %s", e)

    async def run_self_update(self):
        """Trigger autonomous self-update (Fine-tuning)."""
        logger.info("🧬 EVO: Triggering self-update (GPU low-load window)...")
        try:
            from core.tasks import celery_app
            celery_app.send_task("core.tasks.run_self_update")
        except Exception as e:
            record_degradation('metabolic_coordinator', e)
            logger.error("Self-update trigger failed: %s", e)
