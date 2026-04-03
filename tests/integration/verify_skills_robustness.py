################################################################################

import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SkillVerification")

def test_self_repair():
    logger.info("--- Testing SelfRepairSkill Robustness ---")
    try:
        from skills.self_repair import SelfRepairSkill
        skill = SelfRepairSkill()
        
        # Test case: Missing parameters
        logger.info("  Invoking execute() with empty goal...")
        result = skill.execute({}, {})
        
        logger.info(f"  Result: {result}")
        
        if result.get("ok") and result.get("status") == "standby":
            logger.info("  ✅ PASSED: Handled missing component gracefully.")
        else:
            logger.error(f"  ❌ FAILED: Unexpected result: {result}")
            
    except Exception as e:
        logger.error(f"  ❌ FAILED: Exception raised: {e}")

def test_web_search():
    logger.info("\n--- Testing WebSearchSkill Robustness ---")
    try:
        from skills.web_search import EnhancedWebSearchSkill
        skill = EnhancedWebSearchSkill()
        
        # Test case: Missing query
        logger.info("  Invoking extract_query with empty goal...")
        query = skill._extract_query({}, {})
        
        logger.info(f"  Extracted Query: '{query}'")
        
        # The specific fallback might be overridden by "Curious Explorer" randomization
        # so we just need to ensure we got a valid non-empty string.
        if query and len(str(query)) > 0:
             logger.info(f"  ✅ PASSED: Generated valid query: '{query}'")
        else:
             logger.error(f"  ❌ FAILED: Unexpected query extraction: '{query}'")
             
    except Exception as e:
        logger.error(f"  ❌ FAILED: Exception raised: {e}")

def test_inter_agent_comm():
    logger.info("\n--- Testing InterAgentComm Initialization ---")
    try:
        from skills.inter_agent_comm import InterAgentCommSkill
        skill = InterAgentCommSkill()
        logger.info("  ✅ PASSED: Initialized without import errors.")
        
    except Exception as e:
        logger.error(f"  ❌ FAILED: Exception raised: {e}")

if __name__ == "__main__":
    test_self_repair()
    test_web_search()
    test_inter_agent_comm()


##
