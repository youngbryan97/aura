################################################################################

"""'The Gauntlet' - High-Stress Resilience Test Suite for Aura.

Tests the system's ability to handle extreme concurrency, memory load, 
and hardware fluctuations without crashing or losing state integrity.
"""
import asyncio
import time
import logging
import random
import uuid
from typing import List, Dict, Any

from core.container import ServiceContainer, ServiceLifetime
from core.capability_engine import CapabilityEngine
from core.continuous_learning import ContinuousLearningEngine, Experience
from core.security.permission_guard import PermissionGuard, PermissionType

# Setup Logging for Test
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] GAUNTLET: %(message)s")
logger = logging.getLogger("TheGauntlet")

class TheGauntlet:
    def __init__(self):
        self.container = ServiceContainer()
        self.results = {
            "concurrency": {"success": 0, "fail": 0},
            "memory": {"success": 0, "fail": 0},
            "hardware": {"success": 0, "fail": 0}
        }

    async def setup(self):
        """Initialize core engines for testing."""
        logger.info("🥊 Setting up The Gauntlet...")
        
        # Register Base Services
        self.container.register('capability_engine', lambda: CapabilityEngine(), lifetime=ServiceLifetime.SINGLETON)
        self.container.register('learning_engine', lambda: ContinuousLearningEngine(), lifetime=ServiceLifetime.SINGLETON)
        self.container.register('permission_guard', lambda: PermissionGuard(), lifetime=ServiceLifetime.SINGLETON)
        
        # Force eager load
        self.capability = self.container.get('capability_engine')
        self.learning = self.container.get('learning_engine')
        self.guard = self.container.get('permission_guard')
        
        logger.info("✓ Engines Online.")

    async def test_concurrency_stress(self, cycles: int = 50):
        """Spawns many parallel skill execution and learning cycles."""
        logger.info(f"🔥 Stressing Concurrency: {cycles} parallel cycles...")
        
        async def _single_cycle(i: int):
            try:
                # 1. Permission Check
                await self.guard.check_permission(PermissionType.SCREEN)
                
                # 2. Skill Execution (Simulated)
                # In a real test, we would call self.capability.execute
                # Here we simulate the logic to verify container/engine thread safety
                await asyncio.sleep(random.uniform(0.01, 0.1))
                
                # 3. Learning Recording
                await self.learning.record_interaction(
                    f"Test input {i}", 
                    f"Test response {i}", 
                    user_name=f"User_{i%5}"
                )
                return True
            except Exception as e:
                logger.error(f"Cycle {i} failed: {e}")
                return False

        tasks = [_single_cycle(i) for i in range(cycles)]
        outcomes = await asyncio.gather(*tasks)
        
        self.results["concurrency"]["success"] = outcomes.count(True)
        self.results["concurrency"]["fail"] = outcomes.count(False)
        logger.info(f"Concurrency Result: {self.results['concurrency']}")

    async def test_memory_deluge(self, items: int = 1000):
        """Floods the experience store with rapid injections."""
        logger.info(f"🌊 Stressing Memory: {items} rapid injections...")
        start = time.time()
        
        success = 0
        for i in range(items):
            try:
                # Synchronous-ish rapid fire
                exp_id = await self.learning.record_interaction(
                    f"Deluge item {i}", 
                    "...", 
                    domain="stress_test"
                )
                if exp_id: success += 1
            except Exception as e:
                logger.error(f"Memory injection {i} failed: {e}")

        elapsed = time.time() - start
        self.results["memory"]["success"] = success
        self.results["memory"]["fail"] = items - success
        logger.info(f"Memory Result: {success}/{items} in {elapsed:.2f}s ({success/elapsed:.1f} ops/s)")

    async def test_hardware_flicker(self, toggles: int = 20):
        """Rapidly checks hardware permissions while engines are busy."""
        logger.info(f"⚡ Stressing Hardware: {toggles} pre-flight toggles...")
        
        success = 0
        for i in range(toggles):
            try:
                # Concurrent with background learning
                await self.guard.check_permission(PermissionType.SCREEN, force=True)
                await self.guard.check_permission(PermissionType.MIC, force=True)
                success += 1
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Hardware toggle {i} failed: {e}")

        self.results["hardware"]["success"] = success
        self.results["hardware"]["fail"] = toggles - success
        logger.info(f"Hardware Result: {self.results['hardware']}")

    async def run_all(self):
        await self.setup()
        
        # Run tests
        await asyncio.gather(
            self.test_concurrency_stress(100),
            self.test_memory_deluge(500),
            self.test_hardware_flicker(50)
        )
        
        logger.info("=" * 40)
        logger.info("THE GAUNTLET COMPLETE")
        logger.info(f"Final Results: {self.results}")
        logger.info("=" * 40)
        
        # Verify success
        total_fails = sum(v["fail"] for v in self.results.values())
        if total_fails == 0:
            logger.info("🏆 PERFECT SCORE. Aura is hardened.")
            return True
        else:
            logger.error(f"❌ FAILS DETECTED: {total_fails}")
            return False

if __name__ == "__main__":
    gauntlet = TheGauntlet()
    asyncio.run(gauntlet.run_all())


##
