"""core/senses/soma.py

Soma (Body) — Unified Sensory Registry and Proprioception Layer.
Aggregates hardware metrics, network latency, and sensory imprints 
into a cohesive self-perception of physical state.
"""
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import psutil
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from core.container import ServiceContainer

logger = logging.getLogger("Aura.Senses.Soma")

@dataclass
class BodyState:
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    battery_percent: Optional[float] = None
    power_plugged: bool = True
    network_latency: float = 0.0
    
    # Sensory Imprints (summaries from other modules)
    last_vision_summary: str = ""
    last_audio_transcript: str = ""
    last_action_result: str = ""
    
    # Affective Mapping (derived)
    stress_level: float = 0.0
    isolation_level: float = 0.0
    fatigue_level: float = 0.0

class Soma:
    """The 'Soma' sensory manager. Orchestrates proprioception."""

    def __init__(self):
        self.state = BodyState()
        self.running = False
        self._loop_task: Optional[asyncio.Task] = None
        self.update_interval = 5.0 # Check stats every 5 seconds
        
    async def start(self):
        if self.running:
            return
        self.running = True
        self._loop_task = get_task_tracker().create_task(self._somatic_loop())
        logger.info("🧘 Soma activated: Proprioception online.")

    async def stop(self):
        self.running = False
        if self._loop_task:
            self._loop_task.cancel()
        logger.info("🧘 Soma deactivated.")

    async def _somatic_loop(self):
        """Periodically update hardware metrics and compute affective mapping."""
        while self.running:
            try:
                # 1. Update Hardware Metrics
                self.state.cpu_percent = psutil.cpu_percent()
                self.state.ram_percent = psutil.virtual_memory().percent
                
                battery = psutil.sensors_battery()
                if battery:
                    self.state.battery_percent = battery.percent
                    self.state.power_plugged = battery.power_plugged
                
                # 2. Update Network Latency (Internal awareness)
                # We do a very lightweight ping-like check (or skip if offline)
                self.state.network_latency = await self._check_latency()

                # 3. Affective Mapping (Proprioception)
                self._map_affective_states()
                
                # 4. Sync with Registry
                reg = ServiceContainer.get("state_registry", default=None)
                if reg:
                    await reg.update(
                        cpu_stress=self.state.stress_level,
                        fatigue=self.state.fatigue_level,
                        connectivity=1.0 - self.state.isolation_level
                    )

                # 5. Signal Affective Circumplex and Heartstone Values
                #    with thermal/stress state so LLM params adapt live
                try:
                    from core.affect.affective_circumplex import get_circumplex
                    circ_params = get_circumplex().get_llm_params()
                    # Signal heartstone if under significant thermal stress
                    if self.state.stress_level > 0.70:
                        from core.affect.heartstone_values import get_heartstone_values
                        get_heartstone_values().on_thermal_stress(
                            arousal=circ_params.get("arousal", 0.5),
                            valence=circ_params.get("valence", 0.5),
                        )
                except Exception as _exc:
                    record_degradation('soma', _exc)
                    logger.debug("Suppressed Exception: %s", _exc)

                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation('soma', e)
                logger.error(f"Soma loop error: {e}")
                await asyncio.sleep(10)

    async def _check_latency(self) -> float:
        """Lightweight attempt to measure network latency.
        Issue 29: Use local gateway or loopback to avoid privacy/performance issues
        with external DNS pings.
        """
        try:
            start = time.time()
            # Try to connect to the local gateway or just check loopback latency
            # for a baseline if no gateway is found.
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", 22), # Check local SSH or similar
                timeout=0.5
            )
            writer.close()
            await writer.wait_closed()
            return time.time() - start
        except Exception:
            # Baseline loopback
            return 0.001 

    def _map_affective_states(self):
        """Map raw metrics to subjective body sensations."""
        # CPU > 80% maps to high stress
        self.state.stress_level = min(1.0, self.state.cpu_percent / 90.0)
        
        # High latency or disconnection maps to isolation
        self.state.isolation_level = min(1.0, self.state.network_latency / 0.8)
        
        # Battery < 15% on battery maps to fatigue
        if self.state.battery_percent is not None and not self.state.power_plugged:
            if self.state.battery_percent < 20:
                self.state.fatigue_level = 1.0 - (self.state.battery_percent / 20.0)
            else:
                self.state.fatigue_level = 0.0
        else:
            self.state.fatigue_level = 0.0

    def update_sensory_imprint(self, source: str, data: str):
        """Called by PulseManager or ContinuousPerception to update local awareness."""
        if source == "vision":
            self.state.last_vision_summary = data
        elif source == "audio":
            self.state.last_audio_transcript = data
        elif source == "action":
            self.state.last_action_result = data

    def get_body_snapshot(self) -> Dict[str, Any]:
        """Returns a snapshot of the current somatic state."""
        resource_anxiety = max(self.state.stress_level, self.state.fatigue_level)
        thermal_load = min(1.0, max(self.state.cpu_percent, self.state.ram_percent) / 100.0)
        vitality = max(0.0, min(1.0, 1.0 - (0.45 * self.state.stress_level + 0.35 * self.state.fatigue_level + 0.20 * self.state.isolation_level)))
        return {
            "metrics": {
                "cpu": self.state.cpu_percent,
                "ram": self.state.ram_percent,
                "battery": self.state.battery_percent,
                "plugged": self.state.power_plugged
            },
            "affects": {
                "stress": self.state.stress_level,
                "isolation": self.state.isolation_level,
                "fatigue": self.state.fatigue_level
            },
            "soma": {
                "thermal_load": thermal_load,
                "resource_anxiety": resource_anxiety,
                "vitality": vitality,
                "energy": max(0.0, min(1.0, 1.0 - self.state.fatigue_level)),
            },
            "state": "online" if self.running else "idle",
            "energy": max(0.0, min(1.0, 1.0 - self.state.fatigue_level)),
            "vitality": vitality,
            "last_sensations": {
                "vision": self.state.last_vision_summary,
                "audio": self.state.last_audio_transcript
            }
        }

    def get_status(self) -> Dict[str, Any]:
        """Standard status contract for homeostasis and diagnostics."""
        return self.get_body_snapshot()

# Singleton accessor
_soma = None

def get_soma() -> Soma:
    global _soma
    if _soma is None:
        _soma = Soma()
    return _soma
