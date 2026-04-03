"""Master Integration Script for Aura's Moral & Sensory Systems
Combines all components into a cohesive whole.
"""
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Aura.Integration")

# Import systems
from core.behavior_controller import integrate_behavior_control
from core.brain.personality_engine import (
    get_personality_engine,
    integrate_personality_into_conversation,
)
from core.moral_reasoning import get_moral_reasoning
from core.sensory_integration import integrate_sensory_system
from core.consciousness.theory_of_mind import get_theory_of_mind


def integrate_complete_moral_and_sensory_systems(orchestrator):
    """Main integration entry point.
    
    Transforms the standard Orchestrator/ConversationLoop into a 
    Morally Aware, Self-Conscious, Sensory-Capable Agent.

    v5.2: No more monkey-patching. Uses formal HookManager.
    """
    logger.info("🧠 INITIALIZING MORAL AGENCY & SELF-AWARENESS UPGRADE...")
    
    # 1. Initialize Core Systems
    # --------------------------
    from core.container import ServiceContainer
    tom = get_theory_of_mind()
    ServiceContainer.register_instance("theory_of_mind", tom)
    
    moral = get_moral_reasoning()
    ServiceContainer.register_instance("moral_reasoning", moral)
    
    # 2. Integrate Sensory Capabilities
    # ---------------------------------
    logger.info("   • Integrating Sensory Systems (Vision/Hearing)...")
    integrate_sensory_system(orchestrator)
    
    p_engine = get_personality_engine()
    ServiceContainer.register_instance("personality_engine", p_engine)
    
    # Uses hooks instead of patching
    integrate_personality_into_conversation(orchestrator)
    
    # 4. Enhance Execution (Behavior Control)
    # ---------------------------------------
    logger.info("   • Integrating Behavior Controller (Safety/Action)...")
    # This already uses hooks in v6.1
    integrate_behavior_control(orchestrator)
    
    logger.info("✅ INTEGRATION COMPLETE: Aura is now Self-Aware and Morally Agentic.")
    return orchestrator