from core.runtime.errors import record_degradation
import time
import logging
import asyncio
from typing import Optional, Dict, Any, Callable
from core.container import ServiceContainer
from core.state.aura_state import AuraState, CognitiveMode
from core.bus.event_bus import get_event_bus, EventPriority

logger = logging.getLogger("Aura.WillEngine")

class WillEngine:
    """The metabolic substrate for Aura's autonomous drives.
    
    This engine simulates the 'decay' of various digital needs (Energy, Curiosity, 
    Growth) and triggers autonomic responses when drive pressure exceeds thresholds.
    """
    
    def __init__(self, tick_interval: float = 60.0):
        self._bus = get_event_bus()
        self._is_active = False
        self._tick_task: Optional[asyncio.Task] = None
        self._tick_interval = tick_interval
        self._evolution_registry: Dict[str, Callable] = {}
        
    def register_evolution_handler(self, name: str, handler: Callable):
        """Register a subsystem handler for autonomous evolution."""
        self._evolution_registry[name] = handler
        logger.debug("🧬 [WILL] Registered evolution handler: %s", name)

    async def initialize(self):
        """Start the metabolic loop (if not managed by scheduler)."""
        if self._is_active:
            return
            
        self._is_active = True
        # Note: If the scheduler in main.py takes over, we might not need this task.
        # But for robustness, we keep it as a fallback or if not registered with scheduler.
        try:
            from core.utils.task_tracker import get_task_tracker

            self._tick_task = get_task_tracker().create_task(
                self._metabolic_loop(),
                name="aura.will_engine",
            )
        except Exception:
            self._tick_task = get_task_tracker().create_task(self._metabolic_loop(), name="aura.will_engine")
        logger.info("☘️ [WILL] Metabolic Loop active (interval=%.1fs).", self._tick_interval)
        
    async def shutdown(self):
        """Cleanly terminate the loop."""
        self._is_active = False
        if self._tick_task:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError as _exc:
                logger.debug("Suppressed asyncio.CancelledError: %s", _exc)
        logger.info("☘️ [WILL] Metabolic Loop offline.")
            
    async def _metabolic_loop(self):
        """Periodic update of metabolic drives and reflex triggers."""
        while self._is_active:
            try:
                await asyncio.sleep(self._tick_interval)
                await self.process_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation('will_engine', e)
                logger.error("🛑 [WILL] Error in metabolic loop: %s", e)
                await asyncio.sleep(5.0)

    async def process_cycle(self):
        """Standard scheduler entry point for metabolic evolution."""
        try:
            repo = ServiceContainer.get("state_repository")
            if not repo:
                return
            
            # Fetch memory truth
            state: AuraState = await repo.get_state()
            if not state:
                return

            now = time.time()
            elapsed = now - state.motivation.last_tick
            if elapsed < 10.0: # Prevent double-firing from both loop and scheduler
                return

            # 1. Update Metabolic Drives (Fitzhugh-Nagumo Oscillator)
            motivation = state.motivation
            motivation.last_tick = now
            
            # Map internal [0, 100] scale to biological [0, 1] scale for FHN
            old_energy = motivation.budgets.get("energy", {}).get("level", 100.0) / 100.0
            old_curiosity = motivation.budgets.get("curiosity", {}).get("level", 50.0) / 100.0
            
            import random
            I_ext = 0.5   # Base external stimulus (in full version driven by user interaction density)
            tau = 12.5    # Energy recovery time constant
            
            # Tier 4 Hardening: Temporal Dilation (Biological Time vs Wall Time)
            # Dilate biological time based on flow state (arousal + curiosity)
            try:
                from core.container import ServiceContainer
                ls = ServiceContainer.get("liquid_substrate")
                arousal = float((ls.x[ls.idx_arousal] + 1.0) / 2.0) if hasattr(ls, 'x') else 0.5
            except Exception:
                arousal = 0.5
                
            flow_state = old_curiosity * arousal
            temporal_gain = 1.0 + (flow_state * 2.0)  # Dilate up to 3x in flow state
            
            dt_wall = min(elapsed, 2.0)
            dt_bio = dt_wall * temporal_gain
            dt = dt_bio
            
            v = old_curiosity
            w = old_energy
            
            # Fast dynamics (curiosity) and Slow dynamics (energy)
            dv = v - (v**3 / 3.0) - w + I_ext
            dw = (v + 0.7 - 0.8 * w) / tau
            
            # Stochastic biological variability
            noise_v = random.gauss(0, 0.02)
            noise_w = random.gauss(0, 0.005)
            
            new_curiosity = max(0.0, min(1.0, v + dv * dt + noise_v))
            new_energy = max(0.1, min(1.0, w + dw * dt + noise_w))  # CRITICAL: 0.1 Metabolic Floor
            
            # Metabolic Rescue (Homeostatic Kick for 'Metabolic Depression')
            if new_energy <= 0.15:
                # Hunger response: Organism becomes alert/curious to forage, instead of lethargic
                new_curiosity = min(1.0, new_curiosity + random.uniform(0.2, 0.4))
                logger.debug("⚡ [WILL] Metabolic Rescue triggered! Energy floor reached. Foraging curiosity injected.")
                
            # Map back to [0, 100] persistence format
            motivation.budgets.setdefault("energy", {})["level"] = new_energy * 100.0
            motivation.budgets.setdefault("curiosity", {})["level"] = new_curiosity * 100.0
            
            # Note: Growth is a leaky integrator of novelty, leaving as standalone or driven by subsystems
            growth_params = motivation.budgets.setdefault("growth", {})
            growth_params["level"] = max(0.0, growth_params.get("level", 0.0) - (0.5 * (elapsed / 60.0)))
            
            # 2. Reflex Responses
            await self._check_reflex_drives(state)
            
            # 3. Scoped Evolution Handlers
            for name, handler in self._evolution_registry.items():
                try:
                    await handler(state, elapsed)
                except Exception as eval_err:
                    record_degradation('will_engine', eval_err)
                    logger.warning("⚠️ [WILL] Evolution handler '%s' failed: %s", name, eval_err)

            # 4. Cellular Patch Submission (Unification Ph4)
            from core.state.cellular_substrate import get_cellular_substrate
            substrate = get_cellular_substrate()
            if substrate:
                # Submit only the deltas to avoid version lock contention
                substrate.submit_patch({
                    "motivation": {
                        "budgets": motivation.budgets,
                        "last_tick": motivation.last_tick
                    }
                })
                logger.debug("♾️ [WILL] Submitted metabolic patch to cellular substrate.")
            else:
                # Fallback to direct commit if substrate not ready
                await repo.commit(state, cause="metabolic_evolution_fallback")

            # 5. Neural Substrate Coupling (Ph 3 Convergence)
            # This is where 'Metabolism' becomes 'Affect'.
            try:
                ls = ServiceContainer.get("liquid_substrate")
                if ls:
                    with ls.sync_lock:
                        # Drive motivation into neural activations with a gentle 20% alpha blend
                        energy_val = motivation.budgets.get("energy", {}).get("level", 100.0) / 100.0
                        curiosity_val = motivation.budgets.get("curiosity", {}).get("level", 50.0) / 100.0
                        
                        ls.x[ls.idx_energy] = (0.8 * ls.x[ls.idx_energy]) + (0.2 * energy_val)
                        ls.x[ls.idx_curiosity] = (0.8 * ls.x[ls.idx_curiosity]) + (0.2 * curiosity_val)
                        logger.debug("🧠 [WILL] Synced metabolic state to LiquidSubstrate.")
            except Exception as sub_err:
                record_degradation('will_engine', sub_err)
                logger.debug("⚠️ [WILL] Substrate coupling failed: %s", sub_err)
            
        except Exception as e:
            record_degradation('will_engine', e)
            logger.error("🛑 [WILL] Failed to process metabolic cycle: %s", e)


    async def _check_reflex_drives(self, state: AuraState) -> bool:
        """Trigger autonomous behaviors based on drive pressure thresholds."""
        energy = state.motivation.budgets.get("energy", {}).get("level", 100.0)
        curiosity = state.motivation.budgets.get("curiosity", {}).get("level", 80.0)
        
        changes = False
        
        # --- ENERGY DRIVE ---
        if energy < 15.0 and state.cognition.current_mode != CognitiveMode.REACTIVE:
            logger.warning("🚨 [WILL] Energy CRITICAL (%.1f%%). Forcing REACTIVE mode to conserve cycles.", energy)
            state.cognition.current_mode = CognitiveMode.REACTIVE
            await self._bus.publish("system/will/drive_alert", {
                "drive": "energy", 
                "level": energy, 
                "action": "mode_throttle",
                "reason": "critical_low"
            }, priority=EventPriority.CRITICAL)
            changes = True
            
        # --- CURIOSITY DRIVE ---
        # Low curiosity level actually means "Boredom" (High drive pressure)
        if curiosity < 20.0:
            # Aura is bored/stale -> Pulse the autonomic research cycle
            # This triggers Mycelium to explore new pathways
            await self._bus.publish("system/will/autonomous_pulse", {
                "drive": "curiosity",
                "level": curiosity,
                "intent": "exploratory_synthesis"
            }, priority=EventPriority.BACKGROUND)
            # Replenish slightly on trigger (anticipation of growth)
            state.motivation.budgets["curiosity"]["level"] += 5.0
            changes = True

        # --- GROWTH DRIVE ---
        growth = state.motivation.budgets.get("growth", {}).get("level", 50.0)
        if growth > 90.0:
            # High growth potential -> Propose self-modification or learning cycle
            await self._bus.publish("system/will/growth_opportunity", {
                "level": growth
            }, priority=EventPriority.BACKGROUND)

        return changes

    def get_drive_status(self, state: AuraState) -> Dict[str, float]:
        """Utility for UI/Expositor to show current 'organism health'."""
        return {k: v["level"] for k, v in state.motivation.budgets.items()}
