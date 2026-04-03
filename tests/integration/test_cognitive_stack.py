################################################################################


"""
tests/integration/test_cognitive_stack.py
=========================================
Verifies that all advanced cognitive modules can be instantiated and wired together.
This is a smoke test for the v8.0 architecture upgrade.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test.CognitiveStack")

async def test_stack():
    logger.info("🧪 Testing Cognitive Architecture Initialization...")
    
    try:
        # 1. Import Modules
        logger.info("   [1/6] Importing Core Modules...")
        from core.evolution.liquid_time_engine import ContinuousState
        from core.brain.autopoiesis import AutopoieticGraph
        from core.brain.entropy import PhysicalEntropyInjector
        from core.continuous_learning import ContinuousLearningEngine
        from core.dual_memory import DualMemorySystem
        from core.uncertainty import EpistemicHumilityEngine
        from core.embodiment import ContinuousSensoryFeed
        from core.cognitive_integration import CognitiveIntegrationLayer
        logger.info("         Success.")

        # 2. Instantiate Independent Modules
        logger.info("   [2/6] Instantiating Engines...")
        ltc = ContinuousState()
        autopoiesis = AutopoieticGraph()
        entropy = PhysicalEntropyInjector()
        learning = ContinuousLearningEngine(db_path=":memory:") # Use in-memory DB for test
        memory = DualMemorySystem(base_dir="/tmp/aura_test_memory")
        epistemic = EpistemicHumilityEngine()
        sensory = ContinuousSensoryFeed()
        logger.info("         Success.")

        # 3. Test LTC Pulse
        logger.info("   [3/6] Testing LTC Pulse...")
        await ltc.pulse()
        logger.info(f"         LTC Curiosity: {ltc.nodes['curiosity'].value:.4f}")
        logger.info("         Success.")

        # 4. Test Entropy Injection
        logger.info("   [4/6] Testing Hardware Entropy...")
        chaos = entropy.get_generation_temperature(base_temp=0.7)
        logger.info(f"         Chaos Modifier: {chaos:.4f}")
        logger.info("         Success.")
        
        # 5. Test Integration Layer Wiring (Mock Orchestrator)
        logger.info("   [5/6] Testing Integration Layer...")
        class MockOrchestrator:
            pass
            
        orch = MockOrchestrator()
        integration = CognitiveIntegrationLayer(orchestrator=orch, base_data_dir="/tmp/aura_test_data")
        
        # Initialize (Async)
        await integration.initialize()
        logger.info("         Integration Layer Initialized.")
        
        # Build Context
        ctx = await integration.build_enhanced_context("Hello world", emotional_context=0.5)
        logger.info(f"         Context Built (Length: {len(ctx)})")
        logger.info("         Success.")

        logger.info("✅ COGNITIVE STACK VERIFIED. All systems nominal.")
        return True

    except Exception as e:
        logger.error(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        if asyncio.run(test_stack()):
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        pass


##
