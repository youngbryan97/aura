import asyncio
import logging
import psutil
import time
import os
import gc
from typing import Dict, Any
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.AutonomicCore")

class AutonomicCore:
    """Unified Autonomic Nervous System.
    Replaces Governor, ImmuneSystem, ExistentialAwareness, and OptimizationEngine.
    Provides a single, deterministic heartbeat for system survival.
    """
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        
        # Unified Thresholds
        self.throttle_ram_percent = 92.0
        self.cleanup_ram_percent = 94.0
        self.critical_ram_percent = 96.0
        
        self.running = False
        self._task = None
        self.uptime_start = time.time()
        
        # Restoration Phase Integration
        from .survival_driver import SurvivalDriver
        self.survival_driver = SurvivalDriver(self.orchestrator)
        self.survival_status = {}
        
    async def start(self):
        """Boot the unified autonomic heartbeat."""
        self.running = True
        self._task = get_task_tracker().create_task(
            self._heartbeat_loop(),
            name="autonomic_core.heartbeat",
        )
        logger.info("🛡️ Autonomic Core online. Unified survival heartbeat started.")
        
    async def stop(self):
        """Shutdown the autonomic heartbeat."""
        self.running = False
        if self._task:
            self._task.cancel()
            
    async def _heartbeat_loop(self):
        """Single loop for all background survival checks."""
        while self.running:
            try:
                await self._manage_metabolism()
                await self._enforce_governance()
                await self._check_survival()
                
                # Deterministic 10-second heartbeat to minimize CPU overhead
                await asyncio.sleep(10.0) 
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Autonomic Core heartbeat error: %s", e)
                await asyncio.sleep(10.0)
                
    async def _manage_metabolism(self):
        """Consolidated memory and existential checks."""
        try:
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 1. Critical Existential Threat
            if mem.percent >= self.critical_ram_percent or disk.percent > 98.0:
                logger.critical("🚨 EXISTENTIAL THREAT: Critical Vitals (RAM: %s%%, Disk: %s%%). Initiating emergency purge.", mem.percent, disk.percent)
                gc.collect()
                await self._unload_llm()
                await self._emit_status("CRITICAL WARNING: Imminent Memory Exhaustion")
                
            # 2. Hard Cleanup Needed
            elif mem.percent >= self.cleanup_ram_percent:
                logger.warning("⚠️ High RAM (%s%%). Running aggressive garbage collection.", mem.percent)
                gc.collect()
                await self._emit_status("Memory load high. Optimizing...")
                
            # 3. Throttling
            elif mem.percent >= self.throttle_ram_percent:
                # Signal orchestrator to skip deep thoughts or background tasks
                if self.orchestrator:
                    self.orchestrator.status.memory_pressure = True
                    
            else:
                if self.orchestrator:
                    self.orchestrator.status.memory_pressure = False
                    
        except Exception as e:
            logger.debug("Vitals check failed: %s", e)

    async def _enforce_governance(self):
        """Immune System / Process watchdog logic."""
        # Future expansion: Monitor runaway threads or hung processes
        pass
        
    async def _unload_llm(self):
        """Attempts to unload local models to free VRAM/RAM."""
        # Ollama unload logic removed in v5.1.
        # Future: Integrate with MLX Worker to purge model cache if memory is critical.
        logger.info("🧠 Requesting local model memory optimization...")

    async def _check_survival(self):
        """Phase 8: Check for existential threats via SurvivalDriver."""
        try:
            self.survival_status = self.survival_driver.check_vitals()
            imperative = self.survival_driver.get_imperatives(self.survival_status)
            if imperative:
                self.survival_driver.publish_threat(imperative)
        except Exception as e:
            logger.debug("Survival check error: %s", e)

    def get_survival_report(self) -> Dict[str, Any]:
        """Provides the latest survival metrics."""
        return self.survival_status

    async def _emit_status(self, message: str) -> None:
        """Publish a status message to the event bus."""
        from core.event_bus import get_event_bus
        await get_event_bus().publish("autonomic/status", {"message": message})
