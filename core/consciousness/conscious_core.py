"""core/brain/consciousness/conscious_core.py

The Master Integrator.
Connects Liquid Substrate (Existence), Global Workspace (Awareness), and Predictive Engine (Learning).
Implements 'Attractor Volition' - autonomous will emerges from substrate dynamics.
"""

from core.utils.task_tracker import get_task_tracker
import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

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
        self.attractors: Dict[str, Dict[str, float]] = {
            "curiosity": {"arousal_min": 0.5, "valence_min": 0.1},
            "boredom":   {"arousal_max": -0.2, "valence_max": -0.1},
            "reflection": {"dominance_min": 0.4, "arousal_max": 0.1}
        }
        
    async def check_for_impulse(self) -> Optional[str]:
        """Check if current state warrants an action"""
        if time.time() - self.last_action_time < self.refractory_period:
            return None
            
        state = await self.substrate.get_state_summary()
        v, a, d = state['valence'], state['arousal'], state['dominance']
        
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
        
        self.monitor_task: Optional[asyncio.Task] = None
        self.running: bool = False
        self.orchestrator_ref: Any = None # Will be injected
        
        logger.info("Consciousness Core initialized")
        
    def start(self):
        """Wake up"""
        self.substrate.start()
        self.running = True
        
        # Start the Volition Monitor (The "Will" task)
        if not self.monitor_task or self.monitor_task.done():
            try:
                loop = asyncio.get_running_loop()
                self.monitor_task = loop.create_task(self._volition_loop())
            except RuntimeError:
                # Fallback if start() is called outside a loop
                self.monitor_task = get_task_tracker().create_task(self._volition_loop())
        
    def stop(self):
        """Sleep"""
        self.running = False
        self.substrate.stop()
        if hasattr(self, 'monitor_task'):
            self.monitor_task.cancel()
            
    async def _volition_loop(self):
        """Background loop checking for autonomous impulses"""
        while self.running:
            try:
                await asyncio.sleep(1.0) # Check every second (1 Hz)
                
                # 1. Prediction Step
                current_state = self.substrate.x
                surprise = self.predictive.compare_and_learn(current_state)
                
                # If high surprise, spike arousal!
                if surprise > 0.1:
                    await self.substrate.inject_stimulus(np.ones(64) * surprise, weight=0.5)
                    
                    # 2. Volition Step
                    substrate_state = await self.substrate.get_state_summary()
                    predictive_metrics = self.predictive.get_surprise_metrics()
                    
                    # Synthesize Qualia Vector
                    q_norm = self.qualia.synthesize(substrate_state['qualia_metrics'], predictive_metrics)
                    
                    impulse = await self.volition.check_for_impulse()
                    
                    if impulse and self.orchestrator_ref:
                        logger.info("⚡ VOLITION TRIGGERED: %s (q_norm=%.2f)", impulse, q_norm)
                        
                        # v6.3: Causal Telemetry
                        state = await self.substrate.get_state_summary()
                        telemetry_data: Dict[str, Any] = {
                            "timestamp": time.time(),
                            "valence": state['valence'],
                            "arousal": state['arousal'],
                            "dominance": state['dominance'],
                            "q_norm": q_norm,
                            "impulse_type": impulse,
                            "causal_link": "qualia_attractor"
                        }
                        
                        # Log for prove_coupling.py to analyze
                        self._log_causal_telemetry(telemetry_data)
                    
                        # Dispatch to Orchestrator via async loop
                        try:
                            loop = self.orchestrator_ref.loop
                            if loop and loop.is_running():
                                asyncio.run_coroutine_threadsafe(
                                    self.orchestrator_ref.handle_impulse(impulse),
                                    loop
                                )
                        except Exception as dispatch_error:
                            logger.error("Failed to dispatch impulse: %s", dispatch_error)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("CRITICAL error in Consciousness _volition_loop: %s", e)
                await asyncio.sleep(5.0) # Backoff on error

    def _log_causal_telemetry(self, data: Dict[str, Any]):
        """Write causal telemetry to a dedicated log for analysis."""
        from core.config import config
        log_path = config.paths.data_dir / "telemetry" / "causal_behavior.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.debug("Failed to write behavior telemetry: %s", e)

    def on_input_received(self, text: str) -> None:
        """Hook called when user speaks"""
        # Spike arousal and valence (Attention)
        stimulus = np.random.randn(64) * 0.5 # Simplified embedding
        get_task_tracker().create_task(self.substrate.inject_stimulus(stimulus))
        
    def get_state(self) -> Dict[str, Any]:
        """API Payload for Qualia Explorer"""
        # Fix: get_state_summary is async — use sync get_substrate_affect() instead
        sub_state = self.substrate.get_substrate_affect()
        return {
            "substrate": sub_state,
            "surprise": self.predictive.get_surprise_level() if hasattr(self.predictive, 'get_surprise_level') else 0.0,
            "qualia": self.qualia.get_snapshot(),
            "broadcast": str(self.workspace.last_winner.content) if self.workspace.last_winner else None
        }