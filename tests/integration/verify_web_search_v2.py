################################################################################

import asyncio
import logging
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skills.web_search import EnhancedWebSearchSkill

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("VerifyWebSearchV2")

async def test_enhanced_search():
    logger.info("🚀 Starting Enhanced Web Search verification...")
    skill = EnhancedWebSearchSkill()
    
    try:
        # 1. Test Deep Search
        logger.info("--- Testing Deep Search (Deep Dive) ---")
        goal = {"query": "Aura AI sovereign engine", "deep": True}
        result = await skill.execute(goal, {})
        
        if result.get("ok"):
            logger.info("✅ Deep Search SUCCESS")
            logger.info(f"Source: {result.get('source')}")
            logger.info(f"Mode: {result.get('mode')}")
            logger.info(f"Snippet: {result.get('result')[:200]}...")
        else:
            logger.error(f"❌ Deep Search FAILED: {result.get('error')}")

        # 2. Test Standard Search (Parsing duckduckgo snippets)
        logger.info("\n--- Testing Standard Search ---")
        goal = {"query": "current weather in Tokyo", "deep": False}
        result = await skill.execute(goal, {})
        
        if result.get("ok"):
            logger.info("✅ Standard Search SUCCESS")
            logger.info(f"Mode: {result.get('mode')}")
            logger.info(f"Result count: {len(result.get('result'))}")
        else:
            logger.error(f"❌ Standard Search FAILED: {result.get('error')}")

        # 3. Test Robust Clicking in PhantomBrowser
        logger.info("\n--- Testing Robust Clicking (via Browser) ---")
        await skill.browser.browse("https://duckduckgo.com")
        # Try to click 'About' link or similar
        click_success = await skill.browser.click(text_match="About")
        if click_success:
            logger.info("✅ Robust click SUCCESS")
        else:
            logger.warning("⚠️ Robust click 'About' failed (might be dynamic or hidden)")

    except Exception as e:
        logger.error(f"Verification encountered an error: {e}")
    finally:
        await skill.on_stop_async()
        logger.info("\nVerification complete.")

if __name__ == "__main__":
    asyncio.run(test_enhanced_search())


##
