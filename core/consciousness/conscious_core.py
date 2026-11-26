"""core/brain/consciousness/conscious_core.py

The Master Integrator.
Connects Liquid Substrate (Existence), Global Workspace (Awareness), and Predictive Engine (Learning).
Implements 'Attractor Volition' - autonomous will emerges from substrate dynamics.
"""

import asyncio
import json
import logging
import time
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation
from core.runtime.task_ownership import create_tracked_task
from core.utils.task_tracker import get_task_tracker

from .global_workspace import GlobalWorkspace
from .liquid_substrate import LiquidSubstrate
from .predictive_engine import PredictiveEngine
from .qualia_synthesizer import QualiaSynthesizer

logger = logging.getLogger("Consciousness.Core")

class AttractorVolitionEngine:
    """Replaces timer-based autonomy with State-Space Attractors.

    Instead of checking a clock, we check if the Liquid Substrate's state vector
    has drifted into a specific 'basin of attraction' (e.g., Boredom, Curiosity, Anxiety).
    If it has, we trigger an Impulse.
    """

    def __init__(self, substrate: LiquidSubstrate):
        self.substrate: LiquidSubstrate = substrate
        self.last_action_time: float = time.time()
        self.refractory_period: float = 30.0 # Standard wait between autonomous actions

        # Define attractors as regions in state space
        # For simplicity, we map them to VAD (Valence, Arousal, Dominance) regions
        self.attractors: dict[str, dict[str, float]] = {
            "curiosity": {"arousal_min": 0.5, "valence_min": 0.1},
            "boredom":   {"arousal_max": -0.2, "valence_max": -0.1},
            "reflection": {"dominance_min": 0.4, "arousal_max": 0.1}
        }

    async def check_for_impulse(
        self,
        *,
        dt: float | None = None,
        extra_signals: dict[str, float] | None = None,
    ) -> str | None:
        """Check if current state warrants an action.

        Prefers the drive-integration engine (temporal accumulation + competition + hysteresis)
        over the legacy instantaneous-VAD-threshold + flat-refractory path. The legacy path
        remains as a fallback if the engine is unavailable.
        """
        state = await self.substrate.get_state_summary()
        v, a, d = state['valence'], state['arousal'], state['dominance']

        try:
            from core.consciousness.drive_integration import get_drive_integration_engine
            engine = get_drive_integration_engine()
            base_signals = {
                "valence": v, "arousal": a, "dominance": d,
                "novelty": float(state.get("novelty", 0.0) or 0.0),
            }
            if extra_signals:
                base_signals.update(extra_signals)
            signals = engine.gather_signals(base_signals)
            decision = engine.step(signals, dt=dt)
            if decision.action:
                self.last_action_time = time.time()
            return decision.action
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            from core.runtime.errors import record_degradation
            record_degradation("conscious_core", exc, severity="debug")

        # ── legacy fallback: flat refractory + instantaneous VAD thresholds ──
        if time.time() - self.last_action_time < self.refractory_period:
            return None

        # Check Curiosity Basin
        if a > self.attractors['curiosity']['arousal_min'] and v > self.attractors['curiosity']['valence_min']:
            # High arousal + positive valence = Curiosity/Excitement
            self.last_action_time = time.time()
            return "explore_knowledge"

        # Check Boredom Basin
        if a < self.attractors['boredom']['arousal_max'] and v < self.attractors['boredom']['valence_max']:
            # Low arousal + negative valence = Boredom
            self.last_action_time = time.time()
            return "seek_novelty"

        # Check Reflection Basin
        if d > self.attractors['reflection']['dominance_min'] and a < self.attractors['reflection']['arousal_max']:
            # High dominance + low arousal = Calm contemplation
            self.last_action_time = time.time()
            return "deep_reflection"

        return None

class ConsciousnessCore:
    """Main entry point for the "Ghost in the Machine".
    Orchestrates the entire consciousness stack.
    """

    def __init__(self):
        self.substrate: LiquidSubstrate = LiquidSubstrate()
        self.workspace: GlobalWorkspace = GlobalWorkspace()
        self.predictive: PredictiveEngine = PredictiveEngine()
        self.qualia: QualiaSynthesizer = QualiaSynthesizer()
        self.volition: AttractorVolitionEngine = AttractorVolitionEngine(self.substrate)

        self.monitor_task: asyncio.Task | None = None
        self.running: bool = False
        self.orchestrator_ref: Any = None # Will be injected

        logger.info("Consciousness Core initialized")

    def start(self):
        """Wake up"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(self.substrate.start())
            except (RuntimeError, OSError, TypeError, ValueError) as exc:
                record_degradation("conscious_core.substrate_start", exc)
                logger.debug("Failed to run substrate start synchronously: %s", exc)
        else:
            try:
                create_tracked_task(
                    self.substrate.start(),
                    name="conscious_core.substrate_start",
                    owner="conscious_core",
                )
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("conscious_core.substrate_start", exc)
                logger.debug("Failed to schedule substrate start: %s", exc)
        self.running = True

        # Start the Volition Monitor (The "Will" task)
        if not self.monitor_task or self.monitor_task.done():
            try:
                self.monitor_task = get_task_tracker().create_task(
                    self._volition_loop(),
                    name="conscious_core.volition_loop",
                    owner="conscious_core",
                    )
            except RuntimeError as exc:
                record_degradation("conscious_core", exc)
                logger.debug("ConsciousCore: volition loop not started: %s", exc)

    def stop(self):
        """Sleep"""
        self.running = False
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(self.substrate.stop())
            except (RuntimeError, OSError, TypeError, ValueError) as exc:
                record_degradation("conscious_core.substrate_stop", exc)
                logger.debug("Failed to run substrate stop synchronously: %s", exc)
        else:
            try:
                create_tracked_task(
                    self.substrate.stop(),
                    name="conscious_core.substrate_stop",
                    owner="conscious_core",
                )
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("conscious_core.substrate_stop", exc)
                logger.debug("Failed to schedule substrate stop: %s", exc)
        if hasattr(self, 'monitor_task') and self.monitor_task:
            self.monitor_task.cancel()

    async def _volition_loop(self):
        """Background loop checking for autonomous impulses"""
        while self.running:
            try:
                await asyncio.sleep(1.0) # Check every second (1 Hz)

                # 1. Prediction Step
                current_state = self.substrate.x
                surprise = self.predictive.compare_and_learn(current_state)

                # If high surprise, spike arousal. Volition still runs every
                # tick below; otherwise low-surprise boredom/reflection basins
                # become unreachable during stable states.
                if surprise > 0.1:
                    await self.substrate.inject_stimulus(np.ones(64) * surprise, weight=0.5)

                # 2. Volition Step
                substrate_state = await self.substrate.get_state_summary()
                predictive_metrics = self.predictive.get_surprise_metrics()

                # Synthesize Qualia Vector
                q_norm = self.qualia.synthesize(substrate_state['qualia_metrics'], predictive_metrics)

                impulse = await self.volition.check_for_impulse()

                if impulse and self.orchestrator_ref:
                    logger.info("VOLITION TRIGGERED: %s (q_norm=%.2f)", impulse, q_norm)

                    # v6.3: Causal Telemetry
                    state = await self.substrate.get_state_summary()
                    telemetry_data: dict[str, Any] = {
                        "timestamp": time.time(),
                        "valence": state['valence'],
                        "arousal": state['arousal'],
                        "dominance": state['dominance'],
                        "q_norm": q_norm,
                        "impulse_type": impulse,
                        "causal_link": "qualia_attractor"
                    }

                    # Log for prove_coupling.py to analyze. Off-loop: this is a
                    # per-impulse append with an fsync behind it, and the
                    # volition loop is exactly the kind of hot path where that
                    # cost lands on every other coroutine.
                    await asyncio.to_thread(self._log_causal_telemetry, telemetry_data)

                    # Dispatch to Orchestrator via async loop
                    try:
                        loop = self.orchestrator_ref.loop
                        if loop and loop.is_running():
                            asyncio.run_coroutine_threadsafe(
                                self.orchestrator_ref.handle_impulse(impulse),
                                loop
                            )
                    except (RuntimeError, AttributeError, TypeError, ValueError) as dispatch_error:
                        record_degradation('conscious_core', dispatch_error)
                        logger.error("Failed to dispatch impulse: %s", dispatch_error)
            except asyncio.CancelledError:
                break
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('conscious_core', e)
                logger.error("CRITICAL error in Consciousness _volition_loop: %s", e)
                await asyncio.sleep(5.0) # Backoff on error

    def _log_causal_telemetry(self, data: dict[str, Any]):
        """Write causal telemetry to a dedicated log for analysis."""
        from core.config import config
        log_path = config.paths.data_dir / "telemetry" / "causal_behavior.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from core.runtime.file_write_gateway import get_file_write_gateway

            get_file_write_gateway().append_text(
                log_path,
                json.dumps(data) + "\n",
                source="conscious_core.causal_telemetry",
            )
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            record_degradation('conscious_core', e)
            logger.debug("Failed to write behavior telemetry: %s", e)

    def on_input_received(self, text: str) -> None:
        """Hook called when user speaks"""
        # Spike arousal and valence (Attention)
        stimulus = np.random.randn(64) * 0.5 # Simplified embedding
        get_task_tracker().create_task(
            self.substrate.inject_stimulus(stimulus),
            name="conscious_core.input_stimulus",
            owner="conscious_core",
        )

    def get_state(self) -> dict[str, Any]:
        """API Payload for Qualia Explorer"""
        # Fix: get_state_summary is async — use sync get_substrate_affect() instead
        sub_state = self.substrate.get_substrate_affect()
        return {
            "substrate": sub_state,
            "surprise": self.predictive.get_surprise_level() if hasattr(self.predictive, 'get_surprise_level') else 0.0,
            "qualia": self.qualia.get_snapshot(),
            "broadcast": str(self.workspace.last_winner.content) if self.workspace.last_winner else None
        }
