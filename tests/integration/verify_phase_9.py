"""verify_phase_9.py — Verification for Phase 9: Self-Architect & Recursive Mastery.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add core to path
sys.path.append(str(Path(__file__).parent))

from core.container import ServiceContainer, ServiceLifetime
from core.agency_core import AgencyCore, AgencyState
from core.code_refiner import CodeRefinerService
from core.skill_evolution import SkillEvolutionEngine
from core.system_monitor import SystemStateMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Aura.Verify.Phase9")

async def verify_recursive_mastery():
    logger.info("🚀 Starting Phase 9 Verification: Self-Architect & Recursive Mastery...")

    # Mock Config for CodeRefiner
    class MockConfig:
        def __init__(self):
            class Paths:
                project_dir = Path(__file__).parent
            self.paths = Paths()

    # 1. Setup Mock Container and Services using patch
    from unittest.mock import patch, MagicMock
    
    # Mock SovereignSwarm for SkillEvolution
    class MockSwarm:
        def __init__(self):
            self.shards = []
        async def spawn_shard(self, **kwargs):
            logger.info(f"Mock swarm spawning shard: {kwargs.get('objective')}")
            self.shards.append(kwargs)
            return True

    with patch("core.config.config", MockConfig()), \
         patch("core.container.ServiceContainer.get") as mock_get:
        
        # Setup mock ServiceContainer.get responses
        mock_swarm = MockSwarm()
        mock_refiner = CodeRefinerService()
        mock_evolver = SkillEvolutionEngine()
        mock_monitor = SystemStateMonitor()
        
        def container_get_mock(name, default=None):
            return {
                "code_refiner": mock_refiner,
                "skill_evolution": mock_evolver,
                "system_monitor": mock_monitor,
                "sovereign_swarm": mock_swarm,
                "capability_engine": None
            }.get(name, default)
            
        mock_get.side_effect = container_get_mock

        # 2. Test CodeRefiner
        logger.info("\n--- Test 1: Code Refiner Analysis ---")
        proposals = await mock_refiner.analyze_file(Path(__file__))
        logger.info(f"Proposals for this verifier: {len(proposals)}")
        for p in proposals:
            logger.info(f"Proposal: {p.description}")

        # 3. Test SkillEvolution
        logger.info("\n--- Test 2: Skill Evolution ---")
        targets = await mock_evolver.identify_evolution_targets()
        logger.info(f"Evolution targets: {targets}")
        if targets:
            await mock_evolver.spawn_evolution_shard(targets[0])

        # 4. Test SystemMonitor
        logger.info("\n--- Test 3: System Stability Audit ---")
        health = await mock_monitor.audit_stability()
        if health:
            logger.info(f"System Health: Stability={health.cognitive_stability:.2f}, Active Shards={health.active_shards}")

    # 5. Test AgencyCore Integration
    logger.info("\n--- Test 4: AgencyCore Integration ---")
    agency = AgencyCore()
    # Force high initiative and frustration for trigger
    agency.state.initiative_energy = 0.9
    agency.state.frustration_level = 0.8
    
    # Trigger pathway
    action = await agency._pathway_self_architect(now=sys.float_info.max/2, idle_seconds=300)
    if action:
        logger.info(f"✅ Self-Architect triggered: {action['type']}")
        logger.info(f"Message: {action['message']}")
    else:
        logger.info("ℹ️ Self-Architect did not trigger (random chance or state check).")

    logger.info("\n✅ Phase 9 Verification COMPLETE.")

if __name__ == "__main__":
    asyncio.run(verify_recursive_mastery())
