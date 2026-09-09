"""core/senses/soma.py

Soma (Body) — Unified Sensory Registry and Proprioception Layer.
Aggregates hardware metrics, network latency, and sensory imprints 
into a cohesive self-perception of physical state.
"""
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from subprocess import SubprocessError
from typing import Any

from core.container import ServiceContainer
from core.runtime import resource_psutil as psutil
from core.runtime.errors import NetworkEffectDenied, record_degradation
from core.runtime.network_gateway import build_stream_endpoint, get_network_gateway
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.Senses.Soma")

@dataclass
class BodyState:
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    battery_percent: float | None = None
    power_plugged: bool = True
    network_latency: float | None = None
    network_latency_source: str = "unmeasured"
    
    # OS as a Body Mappings
    biological_temp: float = 0.0      # CPU -> body temp
    cognitive_load: float = 0.0       # RAM -> cognitive load
    visceral_pressure: float = 0.0     # Disk I/O -> visceral pressure
    genetic_evolution_generation: int = 1 # Git commits -> evolutionary generation
    
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
        self._loop_task: asyncio.Task | None = None
        self.update_interval = 5.0 # Check stats every 5 seconds
        self._last_disk_io = None
        self._last_disk_time = None
        #: (taken_at, cpu percent, ram percent) as the proprioceptive loop last
        #: sensed the host, or None before the loop has run once.
        self._observed_host: tuple[float, float, float] | None = None
        configured_root = str(os.getenv("AURA_ROOT") or "").strip()
        self._repo_dir = (
            Path(configured_root).expanduser().resolve()
            if configured_root
            else Path(__file__).resolve().parents[2]
        )
        
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
                # 1. Update Hardware Metrics. The proprioceptive loop's own
                # reading wins while it is fresh: it is the organ that senses
                # the host for the rest of the system, and a second timer
                # reading the machine behind it is a second body.
                if not self._host_is_observed():
                    self.state.cpu_percent = psutil.cpu_percent()
                    self.state.ram_percent = psutil.virtual_memory().percent
                
                # Visceral Pressure (Disk I/O deltas)
                try:
                    disk_io = psutil.disk_io_counters()
                    now_time = time.time()
                    if disk_io and self._last_disk_io:
                        time_delta = max(0.01, now_time - self._last_disk_time)
                        read_bytes_delta = disk_io.read_bytes - self._last_disk_io.read_bytes
                        write_bytes_delta = disk_io.write_bytes - self._last_disk_io.write_bytes
                        total_bytes_delta = read_bytes_delta + write_bytes_delta
                        # 10MB/s represents maximum pressure (1.0)
                        bytes_per_sec = total_bytes_delta / time_delta
                        self.state.visceral_pressure = min(1.0, bytes_per_sec / (10 * 1024 * 1024))
                    else:
                        self.state.visceral_pressure = 0.0
                    self._last_disk_io = disk_io
                    self._last_disk_time = now_time
                except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as io_err:
                    logger.debug("Disk IO counters read failed: %s", io_err)
                    self.state.visceral_pressure = 0.0

                # Genetic Evolution Generation (Git commits)
                try:
                    res = get_subprocess_gateway().run(
                        ["git", "rev-list", "--count", "HEAD"],
                        cwd=str(self._repo_dir),
                        timeout=3.0,
                        read_only=True,
                        source="soma_git_generation",
                        accelerator_capability="none",
                    )
                    if res.returncode == 0:
                        self.state.genetic_evolution_generation = int(res.stdout.strip())
                except (SubprocessError, OSError, ValueError) as git_err:
                    logger.debug("Git commit count query failed: %s", git_err)

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
                except (ImportError, AttributeError, RuntimeError) as _exc:
                    record_degradation('soma', _exc)
                    logger.debug("Suppressed Exception: %s", _exc)

                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('soma', e)
                logger.error("Soma loop error: %s", e)
                await asyncio.sleep(10)

    async def _check_latency(self) -> float | None:
        """Lightweight attempt to measure network latency.
        Issue 29: Use local gateway or loopback to avoid privacy/performance issues
        with external DNS pings.
        """
        try:
            start = time.time()
            # Try to connect to the local gateway or just check loopback latency
            # for a baseline if no gateway is found.
            admission = await get_network_gateway().connect_stream(
                build_stream_endpoint("127.0.0.1", 22),
                open_timeout=0.5,
                source="telemetry:soma.loopback_latency",
                read_only=True,
                allow_private_target=True,
            )
            writer = admission.writer
            writer.close()
            await asyncio.wait_for(writer.wait_closed(), timeout=0.5)
            self.state.network_latency_source = "loopback_tcp_22"
            return time.time() - start
        except (
            NetworkEffectDenied,
            RuntimeError,
            asyncio.CancelledError,
            TimeoutError,
            AttributeError,
            OSError,
        ):
            self.state.network_latency_source = "unavailable"
            return None

    def _map_affective_states(self):
        """Map raw metrics to subjective body sensations."""
        # OS as a Body Mappings
        self.state.biological_temp = self.state.cpu_percent
        self.state.cognitive_load = self.state.ram_percent
        
        # CPU > 80% maps to high stress
        self.state.stress_level = min(1.0, self.state.cpu_percent / 90.0)
        
        # High latency or disconnection maps to isolation
        if self.state.network_latency is not None:
            self.state.isolation_level = min(1.0, self.state.network_latency / 0.8)
        
        # Battery < 15% on battery maps to fatigue
        if self.state.battery_percent is not None and not self.state.power_plugged:
            if self.state.battery_percent < 20:
                self.state.fatigue_level = 1.0 - (self.state.battery_percent / 20.0)
            else:
                self.state.fatigue_level = 0.0
        else:
            self.state.fatigue_level = 0.0

    async def pulse(self) -> dict[str, float]:
        """Return current somatic state for the affect engine's DamasioMarkers.
        
        This is called by AffectEngineV2.pulse() to feed hardware telemetry
        into the virtual physiology layer.
        """
        self._map_affective_states()
        snapshot = self.get_body_snapshot()
        return snapshot.get("soma", {
            "thermal_load": 0.0,
            "resource_anxiety": 0.0,
            "biological_temp": 0.0,
            "cognitive_load": 0.0,
            "visceral_pressure": 0.0,
            "genetic_evolution_generation": 1,
        })

    def update_sensory_imprint(self, source: str, data: str):
        """Called by PulseManager or ContinuousPerception to update local awareness."""
        if source == "vision":
            self.state.last_vision_summary = data
        elif source == "audio":
            self.state.last_audio_transcript = data
        elif source == "action":
            self.state.last_action_result = data

    #: How long a reading from the proprioceptive loop stands before this
    #: sense goes back to its own timer. Three beats at the slowest cognitive
    #: cadence — ten seconds in sleep mode — because a loop that has missed
    #: three consecutive beats is not sensing, and a frozen body reading is
    #: worse than a stale one taken the long way.
    HOST_OBSERVATION_TTL_S = 30.0

    def observe_host(
        self,
        *,
        cpu_percent: float,
        ram_percent: float,
        temperature_c: float | None = None,
    ) -> None:
        """The body as the proprioceptive loop just sensed it.

        There were three bodies. This one runs its own timer against psutil;
        the resilience engine ran another; and the proprioceptive loop
        publishes a third into `state.soma.hardware`, which affect, the
        workspace and executive closure all read. Homeostasis asks whichever of
        the first two is registered under `soma` for the numbers it turns into
        her will to live — so the one figure that says whether she is holding
        together came from the host by a route her own sensing never touched.

        The two engines now answer the same call, and the sensing organ reports
        into it. One body, whichever object is registered.
        """
        self._observed_host = (
            time.monotonic(),
            max(0.0, min(100.0, float(cpu_percent))),
            max(0.0, min(100.0, float(ram_percent))),
        )
        del temperature_c  # this sense reads temperature off the cpu, not a probe
        self.state.cpu_percent = self._observed_host[1]
        self.state.ram_percent = self._observed_host[2]
        self._map_affective_states()

    def _host_is_observed(self) -> bool:
        observed = self._observed_host
        return (
            observed is not None
            and time.monotonic() - observed[0] <= self.HOST_OBSERVATION_TTL_S
        )

    def get_body_snapshot(self) -> dict[str, Any]:
        """Returns a snapshot of the current somatic state."""
        resource_anxiety = max(self.state.stress_level, self.state.fatigue_level, self.state.cognitive_load / 100.0)
        thermal_load = min(1.0, max(self.state.cpu_percent, self.state.ram_percent) / 100.0)
        vitality = max(0.0, min(1.0, 1.0 - (0.45 * self.state.stress_level + 0.35 * self.state.fatigue_level + 0.20 * self.state.isolation_level)))
        return {
            "metrics": {
                "cpu": self.state.cpu_percent,
                "ram": self.state.ram_percent,
                "battery": self.state.battery_percent,
                "plugged": self.state.power_plugged,
                "network_latency": self.state.network_latency,
                "network_latency_source": self.state.network_latency_source,
                "visceral_pressure": self.state.visceral_pressure,
                "genetic_evolution_generation": self.state.genetic_evolution_generation
            },
            "affects": {
                "stress": self.state.stress_level,
                "isolation": self.state.isolation_level,
                "fatigue": self.state.fatigue_level,
                "biological_temp": self.state.biological_temp,
                "cognitive_load": self.state.cognitive_load,
                "visceral_pressure": self.state.visceral_pressure
            },
            "soma": {
                "thermal_load": thermal_load,
                "resource_anxiety": resource_anxiety,
                "vitality": vitality,
                "energy": max(0.0, min(1.0, 1.0 - self.state.fatigue_level)),
                "biological_temp": self.state.biological_temp,
                "cognitive_load": self.state.cognitive_load,
                "visceral_pressure": self.state.visceral_pressure,
                "genetic_evolution_generation": self.state.genetic_evolution_generation
            },
            "state": "online" if self.running else "idle",
            "energy": max(0.0, min(1.0, 1.0 - self.state.fatigue_level)),
            "vitality": vitality,
            "last_sensations": {
                "vision": self.state.last_vision_summary,
                "audio": self.state.last_audio_transcript
            }
        }

    def get_status(self) -> dict[str, Any]:
        """Standard status contract for homeostasis and diagnostics."""
        return self.get_body_snapshot()

# Singleton accessor
_soma = None

def get_soma() -> Soma:
    global _soma
    if _soma is None:
        _soma = Soma()
    return _soma
