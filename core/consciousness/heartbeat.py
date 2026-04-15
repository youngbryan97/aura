from core.utils.exceptions import capture_and_log
import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .attention_schema import AttentionSchema
    from .global_workspace import GlobalWorkspace
    from .homeostatic_coupling import HomeostaticCoupling
    from .self_prediction import SelfPredictionLoop
    from .temporal_binding import TemporalBindingEngine

from .global_workspace import CognitiveCandidate
from core.container import ServiceContainer
from core.event_bus import get_event_bus
from core.schemas import TelemetryPayload

logger = logging.getLogger("Consciousness.Heartbeat")


class CognitiveHeartbeat:
    """The always-on 1Hz cognitive process.

    Each tick runs a full cognitive cycle:
    gather → compete → bind → predict → couple → emit

    Designed to be extremely fault-tolerant — a crash in any subsystem
    is caught and logged but never stops the heartbeat itself.
    """

    _TICK_RATE_HZ = 1.0           # Beats per second
    _NARRATIVE_EMIT_TICKS = 60    # Inject autobiographical narrative every 60s
    _SURPRISE_CURIOSITY_THRESHOLD = 0.55  # If surprise > this, seed curiosity

    def __init__(
        self,
        orchestrator,
        attention_schema: "AttentionSchema",
        global_workspace: "GlobalWorkspace",
        temporal_binding: "TemporalBindingEngine",
        homeostatic_coupling: "HomeostaticCoupling",
        self_prediction: "SelfPredictionLoop",
    ):
        self.orch = orchestrator
        self.attention = attention_schema
        self.workspace = global_workspace
        self.temporal = temporal_binding
        self.homeostasis = homeostatic_coupling
        self.predictor = self_prediction

        self.tick_count: int = 0
        self._stop_event = asyncio.Event()
        self._start_time = time.time()
        
        # Noise Reduction
        self._last_alert_times: Dict[str, float] = {}
        self._last_alert_urgency: Dict[str, float] = {}

        # Phase II: CEL Bridge (lazy-loaded from ServiceContainer)
        self._cel_bridge = None
        self._CEL_TICK_INTERVAL = 5  # Constitute expression every 5th tick
        
        # Subsystem Audit Heartbeat (deferred)
        # Fix Issue 69: Lazy-load via _audit_service property

        logger.info("CognitiveHeartbeat initialized.")

    @property
    def _audit_service(self):
        if not hasattr(self, '_audit_cache'):
            self._audit_cache = ServiceContainer.get("subsystem_audit", default=None)
        return self._audit_cache

    @property
    def _mycelium(self):
        if not hasattr(self, '_mycelium_cache'):
            self._mycelium_cache = ServiceContainer.get("mycelial_network", default=None)
        return self._mycelium_cache

    @property
    def _homeostasis(self):
        if not hasattr(self, '_homeostasis_cache'):
            self._homeostasis_cache = ServiceContainer.get("homeostasis", default=None)
        return self._homeostasis_cache

    @property
    def _mind_model(self):
        if not hasattr(self, '_mind_model_cache'):
            self._mind_model_cache = ServiceContainer.get("mind_model", default=None)
        return self._mind_model_cache

    @property
    def _qualia_synthesizer(self):
        if not hasattr(self, '_qualia_cache'):
            self._qualia_cache = ServiceContainer.get("qualia_synthesizer", default=None)
        return self._qualia_cache

    @property
    def _liquid_substrate(self):
        if not hasattr(self, '_liquid_cache'):
            self._liquid_cache = ServiceContainer.get("liquid_substrate", default=None)
        return self._liquid_cache

    @property
    def _integrity_monitor(self):
        if not hasattr(self, '_integrity_cache'):
            self._integrity_cache = ServiceContainer.get("integrity_monitor", default=None)
        return self._integrity_cache

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self):
        """The heartbeat loop. Never stops unless explicitly cancelled.
        Runs at _TICK_RATE_HZ (1Hz default).
        """
        # Issue 87: Ensure Event is initialized
        if self._stop_event is None:
            self._stop_event = asyncio.Event()

        logger.info("💓 Cognitive Heartbeat STARTED")
        interval = 1.0 / self._TICK_RATE_HZ

        while not self._stop_event.is_set():
            tick_start = time.time()
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat tick error (tick=%d): %s", self.tick_count, e, exc_info=True)
                # Never stop the heartbeat for a subsystem error

            # Sleep the remainder of the interval
            elapsed = time.time() - tick_start
            sleep_time = max(0.0, interval - elapsed)
            try:
                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                break

        logger.info("💓 Cognitive Heartbeat STOPPED (total ticks: %s)", self.tick_count)

    def stop(self):
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Single tick
    # ------------------------------------------------------------------

    async def _tick(self):
        self.tick_count += 1
        tick = self.tick_count
        
        # Proof of Life for Subsystem Audit
        if self._audit_service:
            self._audit_service.heartbeat("consciousness")

        # Mycelial Pulse (Proof of Life for Consciousness subsystems)
        try:
            mycelium = self._mycelium
            if mycelium:
                # 1. Main Consciousness Heartbeat
                h_con = mycelium.get_hypha("consciousness", "consciousness")
                if h_con: h_con.pulse(success=True)
                
                # 2. Workspace Proof of Life
                h_ws = mycelium.get_hypha("consciousness", "workspace")
                if h_ws: h_ws.pulse(success=True)
                
                # 3. Attention Schema Proof of Life
                h_att = mycelium.get_hypha("consciousness", "attention")
                if h_att: h_att.pulse(success=True)
        except Exception as _e:
            logger.debug('Ignored Exception in heartbeat.py: %s', _e)

        # ── 1. GATHER internal state ────────────────────────────────────
        homeostasis = self._homeostasis
        if homeostasis:
            await homeostasis.pulse()
            
        mind_model = self._mind_model
        if mind_model:
            # Sync pulse for self-reflection/metabolism
            pass

        state = await self._gather_state()

        # ── 2. SUBMIT candidates to GlobalWorkspace ─────────────────────
        await self._submit_candidates(state, tick)

        # ── 3. GWT COMPETITION ──────────────────────────────────────────
        winner = await self.workspace.run_competition()

        # ── 4. TEMPORAL BINDING ─────────────────────────────────────────
        if winner:
            valence = state.get("affect_valence", 0.0)
            significance = self._compute_significance(winner, state)
            await self.temporal.record_event(
                content=winner.content,
                source=winner.source,
                valence=valence,
                significance=significance,
            )
        await self.temporal.maybe_refresh_narrative(tick)

        # ── 5. SELF-PREDICTION ──────────────────────────────────────────
        actual_drive = state.get("dominant_drive", "curiosity")
        actual_focus = winner.source if winner else "none"
        actual_valence = state.get("affect_valence", 0.0)
        
        # --- NEW Phase XVI: Qualia Synthesis ---
        qualia_norm = 0.0
        qualia_synthesizer = self._qualia_synthesizer
        substrate_metrics = state.get("qualia_metrics", {})
        # Note: PredictiveEngine metrics are proxied via predictor history in v6.0
        # for high-fidelity qualia we pull directly from the synthesizer's last state or sub-metrics
        if qualia_synthesizer:
            qualia_norm = qualia_synthesizer.synthesize(substrate_metrics, self.predictor.get_snapshot())

        await self.predictor.tick(
            actual_valence=actual_valence,
            actual_drive=actual_drive,
            actual_focus_source=actual_focus,
        )

        # ── 5a+. HIERARCHICAL PREDICTIVE CODING ────────────────────────
        # Full Friston hierarchy: every level generates predictions downward
        # and sends errors upward.  The hierarchy's total free energy feeds
        # into the FreeEnergyEngine via accept_surprise_signal().
        try:
            from core.consciousness.predictive_hierarchy import get_predictive_hierarchy
            ph = get_predictive_hierarchy()
            ph_inputs = ph.gather_inputs_from_services()
            ph.tick(**ph_inputs)
        except Exception as e:
            logger.debug("Predictive hierarchy tick failed: %s", e)

        # ── 5b. FREE ENERGY COMPUTATION ─────────────────────────────────
        # Close the loop: PredictiveEngine surprise → FreeEnergy → action tendency
        try:
            fe_engine = ServiceContainer.get("free_energy_engine", default=None)
            if fe_engine:
                world_model = ServiceContainer.get("epistemic_state", default=None)
                # Feed attention scatter as complexity signal
                attention_complexity = self.attention.get_coherence_for_complexity() if hasattr(self.attention, 'get_coherence_for_complexity') else 0.0
                fe_engine.accept_attention_complexity(attention_complexity)
                # Compute free energy from prediction surprise + belief system
                fe_state = fe_engine.compute(
                    prediction_error=surprise,
                    belief_system=world_model,
                    user_present=state.get("affect_engagement", 0) > 0.3,
                )
                # Push surprise signal back to predictive engine coupling
                predictive = ServiceContainer.get("predictive_engine", default=None)
                if predictive and hasattr(predictive, 'accept_feedback'):
                    # The predictive engine can use FE state as a meta-signal
                    pass  # Surprise already flows via heartbeat wiring
        except Exception as e:
            logger.debug("Free energy computation failed: %s", e)

        # ── 5c. CREDIT-WEIGHTED MODULATION ──────────────────────────────
        # CreditAssignment domain performance feeds into hedonic gradient
        if tick % 10 == 0:  # Every 10 ticks to avoid overhead
            try:
                credit = ServiceContainer.get("credit_assignment", default=None)
                if credit:
                    domain_perf = credit.get_all_domain_performance()
                    # Feed influence scores to hedonic gradient for resource allocation
                    hg = ServiceContainer.get("hedonic_gradient", default=None)
                    if hg and hasattr(hg, 'accept_credit_signal'):
                        hg.accept_credit_signal(credit.get_influence_scores())
            except Exception as e:
                logger.debug("Credit modulation failed: %s", e)

        # ── 5d. WORLD MODEL CONSISTENCY CHECK (every 30 ticks) ──────────
        if tick % 30 == 0:
            try:
                world_model = ServiceContainer.get("epistemic_state", default=None)
                if world_model and hasattr(world_model, 'get_summary'):
                    summary = world_model.get_summary()
                    # High contradiction rate contributes to free energy
                    contradiction_rate = summary.get("contradiction_count", 0) / max(1, summary.get("total_beliefs", 1))
                    if contradiction_rate > 0.1:
                        fe_engine = ServiceContainer.get("free_energy_engine", default=None)
                        if fe_engine:
                            # Belief instability adds complexity
                            fe_engine.accept_attention_complexity(
                                min(1.0, fe_engine._last_attention_complexity + contradiction_rate * 0.3)
                            )
            except Exception as e:
                logger.debug("World model consistency check failed: %s", e)

        # ── 6. HOMEOSTATIC COUPLING ─────────────────────────────────────
        attention_mod = self.attention.get_cognitive_modifier()
        await self.homeostasis.update(attention_modifier=attention_mod)

        # ── 7. EMIT to ThoughtStream ────────────────────────────────────
        if winner:
            await self._emit_thought(winner, state, tick)

        # ── 7b. EMIT Telemetry to HUD ──────────────────────────────────
        surprise = self.predictor.get_surprise_signal()
        await self._emit_telemetry(winner, state, tick, surprise)

        # ── 8. PROACTIVE CURIOSITY SEEDING ─────────────────────────────
        if surprise > self._SURPRISE_CURIOSITY_THRESHOLD:
            await self._seed_curiosity_from_surprise(surprise)

        # ── 8b. Φ → WORKSPACE FEED ────────────────────────────────────
        try:
            substrate = self._liquid_substrate
            if substrate and hasattr(substrate, '_current_phi'):
                phi = float(getattr(substrate, '_current_phi', 0.0))
                if hasattr(self.workspace, 'update_phi'):
                    self.workspace.update_phi(phi)
        except Exception as e:
            capture_and_log(e, {'module': __name__})

        # ── 8c. CONSTITUTIVE EXPRESSION (every Nth tick) ──────────────
        if tick % self._CEL_TICK_INTERVAL == 0:
            try:
                if self._cel_bridge is None:
                    self._cel_bridge = ServiceContainer.get("cel_bridge", default=None)
                if self._cel_bridge:
                    await self._cel_bridge.tick()
            except Exception as e:
                logger.debug("CEL tick error: %s", e, exc_info=True)

        # ── 9. NARRATIVE INJECTION & Resource Throttling ───────────────
        # Throttle heavy tasks if system heat or resource stress is high
        resource_stress = state.get("body_heat", 30) > 85 or state.get("body_energy", 100) < 15
        if tick % self._NARRATIVE_EMIT_TICKS == 0:
            if not resource_stress:
                await self._inject_narrative()
            else:
                logger.warning("⚠️ Resource stress detected: Throttling narrative injection.")

        # ── 10. DEBUG LOG every 10 ticks ───────────────────────────────
        if tick % 10 == 0:
            mods = self.homeostasis.get_modifiers()
            logger.debug(
                f"Heartbeat tick {tick} | "
                f"vitality={mods.overall_vitality:.2f} | "
                f"surprise={surprise:.2f} | "
                f"coherence={self.attention.coherence:.2f} | "
                f"winner={winner.source if winner else 'none'}"
            )

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    async def _gather_state(self) -> Dict[str, Any]:
        """Gather lightweight state snapshots from existing systems."""
        state = {}

        # Affect
        try:
            affect_engine = getattr(self.orch, 'affect_engine', None)
            if affect_engine and hasattr(affect_engine, 'get'):
                affect = await affect_engine.get()
                state["affect_valence"] = affect.valence
                state["affect_arousal"] = affect.arousal
                state["affect_engagement"] = affect.engagement
                state["affect_emotion"] = affect.dominant_emotion
        except Exception as e:
            # logger.debug("Affect gather failed: %s", e)
            state.setdefault("affect_valence", 0.0)

        # Drives
        try:
            drive_engine = getattr(self.orch, 'drive_engine', None)
            if drive_engine and hasattr(drive_engine, 'get_status'):
                drives = await drive_engine.get_status()
                state["drives"] = drives
                # Find most depleted drive
                ranked = sorted(
                    [(k, v['level']) for k, v in drives.items() if k not in ("uptime_value",)],
                    key=lambda x: x[1]
                )
                if ranked:
                    state["dominant_drive"] = ranked[0][0]   # Most depleted = most urgent
                    state["drive_urgency"] = max(0.0, 1.0 - (ranked[0][1] / 100.0))
        except Exception as e:
            # logger.debug("Drive gather failed: %s", e)
            state.setdefault("dominant_drive", "curiosity")
            state.setdefault("drive_urgency", 0.3)

        # Embodiment
        try:
            embodiment = getattr(self.orch, 'embodiment', None)
            body = await embodiment.pulse() if hasattr(embodiment, 'pulse') else {}
            # Map SystemSoma (0-1) to Legacy body (0-100)
            state["body_energy"] = (1.0 - body.get("resource_anxiety", 0.0)) * 100
            state["body_heat"] = body.get("thermal_load", 0.0) * 100
            state["body_integrity"] = body.get("vitality", 1.0) * 100
        except Exception as e:
            capture_and_log(e, {'module': __name__})
            
        # Qualia Metrics
        try:
            substrate = ServiceContainer.get("liquid_state", default=None) or ServiceContainer.get("liquid_substrate", default=None)
            if substrate and hasattr(substrate, 'get_state_summary'):
                sub_summary = await substrate.get_state_summary() if asyncio.iscoroutinefunction(substrate.get_state_summary) else substrate.get_state_summary()
                state["qualia_metrics"] = sub_summary.get("qualia_metrics", {})
        except Exception:
            state["qualia_metrics"] = {}

        return state

    async def _submit_candidates(self, state: Dict[str, Any], tick: int):
        """Every subsystem submits its candidate for the GWT competition.
        This is the moment of competitive tension — each subsystem is
        essentially "voting" for what should be in consciousness next.
        """
        affect_weight = abs(state.get("affect_valence", 0.0)) * 0.5

        # --- Drive candidate ---
        dominant_drive = state.get("dominant_drive", "curiosity")
        drive_urgency = state.get("drive_urgency", 0.3)
        
        # Nag Suppression
        # Only alert if urgency is high enough AND (time since last alert > 60s OR urgency spiked)
        current_time = time.time()
        last_alert = self._last_alert_times.get(dominant_drive, 0)
        should_alert = False
        
        if drive_urgency > 0.2:
            if current_time - last_alert > 60:
                should_alert = True
            elif drive_urgency > self._last_alert_urgency.get(dominant_drive, 0) + 0.1:
                should_alert = True # Breakthrough alert if urgency spikes
                
        if should_alert:
            self._last_alert_times[dominant_drive] = current_time
            self._last_alert_urgency[dominant_drive] = drive_urgency
            
            await self.workspace.submit(CognitiveCandidate(
                content=f"Drive alert: {dominant_drive} is depleted ({drive_urgency:.0%} urgency)",
                source=f"drive_{dominant_drive}",
                priority=drive_urgency,
                affect_weight=affect_weight,
            ))

        # --- Affect candidate ---
        emotion = state.get("affect_emotion", "Neutral")
        arousal = state.get("affect_arousal", 0.0)
        if arousal > 0.3 or abs(state.get("affect_valence", 0.0)) > 0.3:
            await self.workspace.submit(CognitiveCandidate(
                content=f"Affective state: {emotion} (arousal={arousal:.2f})",
                source="affect_engine",
                priority=min(1.0, arousal + abs(state.get("affect_valence", 0.0))),
                affect_weight=affect_weight * 1.5,
            ))

        # --- Embodiment candidate ---
        integrity = state.get("body_integrity", 100.0)
        if integrity < 70.0:
            await self.workspace.submit(CognitiveCandidate(
                content=f"Body integrity alert: {integrity:.1f}% (heat={state.get('body_heat', 30):.1f}°)",
                source="embodiment",
                priority=max(0.3, 1.0 - (integrity / 100.0)),
                affect_weight=0.3,
            ))

        # --- Prediction surprise candidate ---
        surprise = self.predictor.get_surprise_signal()
        if surprise > 0.35:
            unpredictable = self.predictor.get_most_unpredictable_dimension()
            await self.workspace.submit(CognitiveCandidate(
                content=f"Prediction surprise in {unpredictable} (err={surprise:.2f})",
                source="self_prediction",
                priority=surprise,
                affect_weight=surprise * 0.4,
            ))

        # --- Curiosity candidate (from existing CuriosityEngine) ---
        curiosity_engine = getattr(self.orch, 'curiosity', None)
        if curiosity_engine and getattr(curiosity_engine, 'current_topic', None):
            topic = curiosity_engine.current_topic
            await self.workspace.submit(CognitiveCandidate(
                content=f"Curiosity topic under exploration: {topic}",
                source="curiosity_engine",
                priority=0.5,
                affect_weight=affect_weight,
            ))

        # --- Qualia Impulse (Phase XVI) ---
        # When ||q|| is high, Aura feels a strong "urge" to act or explore.
        # Nag Suppression: Only fire if >60s since last alert or q_norm spikes by >0.15
        qualia_synthesizer = ServiceContainer.get("qualia_synthesizer", default=None)
        if qualia_synthesizer and qualia_synthesizer.q_norm > 0.6:
            last_q_alert = self._last_alert_times.get("qualia_surge", 0)
            last_q_value = self._last_alert_urgency.get("qualia_surge", 0)
            q_should_alert = (
                current_time - last_q_alert > 60
                or qualia_synthesizer.q_norm > last_q_value + 0.15
            )
            if q_should_alert:
                self._last_alert_times["qualia_surge"] = current_time
                self._last_alert_urgency["qualia_surge"] = qualia_synthesizer.q_norm
                await self.workspace.submit(CognitiveCandidate(
                    content=f"Phenomenal Surge: High qualia intensity (||q||={qualia_synthesizer.q_norm:.2f})",
                    source="qualia_synthesizer",
                    priority=qualia_synthesizer.q_norm * 0.8,
                    affect_weight=affect_weight * 2.0,
                ))

        # --- Free Energy action tendency candidate ---
        # When FE is notable, its dominant_action competes for workspace attention
        try:
            fe_engine = ServiceContainer.get("free_energy_engine", default=None)
            if fe_engine and fe_engine.current and fe_engine.current.free_energy > 0.35:
                fe = fe_engine.current
                urgency = fe_engine.get_action_urgency()
                await self.workspace.submit(CognitiveCandidate(
                    content=f"Active inference: {fe.dominant_action} (F={fe.free_energy:.2f}, {fe_engine.get_trend()})",
                    source="free_energy",
                    priority=urgency,
                    affect_weight=abs(fe.valence) * 0.3,
                ))
        except Exception as e:
            logger.debug("Free energy candidate submission failed: %s", e)

        # --- Attention Focus Bias ---
        # Apply attention schema focus bias to all candidates already submitted
        # by boosting candidates whose source matches current attentional focus
        try:
            if hasattr(self.attention, 'get_focus_bias_for_source'):
                for candidate in self.workspace._candidates:
                    bias = self.attention.get_focus_bias_for_source(candidate.source)
                    if bias > 0:
                        candidate.focus_bias = min(1.0, candidate.focus_bias + bias)
        except Exception as e:
            logger.debug("Attention focus bias application failed: %s", e)

        # --- Baseline cognitive continuity candidate ---
        # Even when nothing is urgent, there should be something in consciousness
        # This prevents empty ticks — ensures there is ALWAYS a cognitive state.
        # [STABILITY v53] Gate to every 30 ticks (30s) to prevent flooding the
        # neural feed and WebSocket with repetitive baseline messages. The old 5-tick
        # interval produced ~12 messages/minute that drowned real cognitive events
        # and caused visible UI lag.
        if tick % 30 == 0:
            await self.workspace.submit(CognitiveCandidate(
                content=f"Cognitive baseline tick {tick}: monitoring internal state",
                source="baseline_continuity",
                priority=0.1,   # Very low — only wins if nothing else is happening
                affect_weight=0.0,
            ))

    async def _emit_thought(
        self,
        winner: CognitiveCandidate,
        state: Dict[str, Any],
        tick: int,
    ):
        """Emit the winning broadcast to the existing ThoughtStream."""
        try:
            from core.thought_stream import get_emitter
            emitter = get_emitter()
            mods = self.homeostasis.get_modifiers()
            emitter.emit(
                title=f"[HB-{tick}] {winner.source}",
                content=(
                    f"{winner.content} | "
                    f"vitality={mods.overall_vitality:.2f} | "
                    f"coherence={self.attention.coherence:.2f}"
                ),
                level="info" if not mods.urgency_flag else "warning",
            )
        except Exception as e:
            logger.debug("ThoughtStream emit failed: %s", e, exc_info=True)

    async def _emit_telemetry(
        self,
        winner: Optional[CognitiveCandidate],
        state: Dict[str, Any],
        tick: int,
        surprise: float
    ):
        """Emit high-fidelity telemetry pulse to the EventBus."""
        try:
            mods = self.homeostasis.get_modifiers()
            narrative = await self.temporal.get_narrative()
            
            # Phase Transcendental: Full Qualia V2 Snapshot
            qualia_snapshot = {}
            try:
                qualia_synthesizer = ServiceContainer.get("qualia_synthesizer", default=None)
                if qualia_synthesizer and hasattr(qualia_synthesizer, "get_snapshot"):
                    qualia_snapshot = qualia_synthesizer.get_snapshot()
            except Exception as qs_err:
                logger.debug("Qualia snapshot failed: %s", qs_err)

            # Pull from orchestrator's liquid state for gauge consistency
            energy = 0.8
            curiosity = 0.5
            frustration = 0.0
            confidence = 0.5
            
            if hasattr(self.orch, 'liquid_state') and self.orch.liquid_state:
                ls = self.orch.liquid_state.current
                energy = ls.energy
                curiosity = ls.curiosity
                frustration = ls.frustration
                confidence = ls.focus
            
            # Resource metrics lookup
            cpu_usage = 0.0
            ram_usage = 0.0
            integrity = self._integrity_monitor
            if integrity:
                stats = integrity.get_stats()
                cpu_usage = stats.get("cpu_percent", 0.0)
                ram_usage = stats.get("memory_percent", 0.0)
            
            # Mycelial metrics lookup
            mycelial_data = {"nodes": 0, "edges": 0, "health": "offline"}
            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if mycelium:
                mycelial_data["health"] = "online"
                mycelial_data["nodes"] = len(getattr(mycelium, "pathways", {}))
                mycelial_data["edges"] = len(getattr(mycelium, "hyphae", []))
            
            payload = TelemetryPayload(
                energy=round(energy * 100, 1),
                curiosity=round(curiosity * 100, 1),
                frustration=round(frustration * 100, 1),
                confidence=round(confidence * 100, 1),
                gwt_winner=winner.source if winner else "none",
                coherence=round(self.attention.coherence, 2),
                vitality=round(mods.overall_vitality, 2),
                surprise=round(surprise, 2),
                narrative=narrative,
                cpu_usage=round(cpu_usage, 1),
                ram_usage=round(ram_usage, 1),
                mycelial=mycelial_data,
                # Phase Transcendental: Full Qualia V2 Payload
                qualia={
                    "q_norm": qualia_snapshot.get("q_norm", 0.0),
                    "pri": qualia_snapshot.get("pri", 0.0),
                    "is_resonating": qualia_snapshot.get("is_resonating", False),
                    "in_attractor": qualia_snapshot.get("in_attractor", False),
                    "dominant_dimension": qualia_snapshot.get("dominant_dimension", "unknown"),
                    "trend": qualia_snapshot.get("trend", 0.0),
                    "volatility": qualia_snapshot.get("volatility", 0.0),
                    "phenomenal_context": qualia_snapshot.get("phenomenal_context", ""),
                    "ual": qualia_snapshot.get("ual_profile", {}),
                }
            )
            
            get_event_bus().publish_threadsafe("telemetry", payload.model_dump())
            
        except Exception as e:
            logger.debug("Telemetry emission failed: %s", e)

    async def _seed_curiosity_from_surprise(self, surprise: float):
        """When prediction error is high, seed curiosity with the surprising dimension."""
        try:
            curiosity_engine = getattr(self.orch, 'curiosity', None)
            if curiosity_engine and hasattr(curiosity_engine, 'add_curiosity'):
                dim = self.predictor.get_most_unpredictable_dimension()
                curiosity_engine.add_curiosity(
                    topic=f"Why is my {dim} hard to predict?",
                    reason=f"High prediction error ({surprise:.2f}) in self-model",
                    priority=min(0.9, surprise),
                )
        except Exception as e:
            capture_and_log(e, {'module': __name__})

    async def _inject_narrative(self):
        """Inject autobiographical narrative into the orchestrator's context.
        This is how temporal continuity gets into the LLM's awareness.
        """
        try:
            narrative = await self.temporal.get_narrative()
            hud_injection = self.homeostasis.get_prompt_injection()

            # Store on orchestrator for cognitive_engine to pick up
            if hasattr(self.orch, '__dict__'):
                self.orch._autobiographical_context = narrative
                self.orch._homeostatic_prompt = hud_injection

            logger.debug("Autobiographical narrative injected into orchestrator context.")
        except Exception as e:
            logger.debug("Narrative injection failed: %s", e)

    def _compute_significance(
        self,
        winner: CognitiveCandidate,
        state: Dict[str, Any],
    ) -> float:
        """Compute significance of a winning broadcast for temporal memory.
        High significance events are more likely to survive sleep consolidation.
        """
        base = winner.effective_priority

        # Urgency events are more significant
        if "alert" in winner.content.lower() or "critical" in winner.content.lower():
            base = min(1.0, base + 0.3)

        # Emotionally valenced events are more significant
        valence = abs(state.get("affect_valence", 0.0))
        base = min(1.0, base + valence * 0.2)

        # Surprise makes things more significant (this is how memory works)
        surprise = self.predictor.get_surprise_signal()
        base = min(1.0, base + surprise * 0.15)

        return round(base, 3)
