"""core/main.py - Sovereign AGI Interaction Loop
Implements the persistent conversation loop for Aura.

v5.2: Added exponential backoff with circuit breaker for fault tolerance.
"""
import asyncio
import logging
import os
import sys
import time

from core.container import get_container

# Ensure parent dir is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .orchestrator import RobustOrchestrator

# Standard logging setup
logger = logging.getLogger("Aura.Main")

# --- THE PANIC ROOM: SAFETY NET ---
try:
    import core.resilience.safety_net
    core.resilience.safety_net.install()
except ImportError:
    logger.debug("Exception caught during execution", exc_info=True)
container = get_container()

# --- Circuit Breaker Constants ---
MAX_CONSECUTIVE_FAILURES = 5
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0
CIRCUIT_BREAKER_COOLDOWN = 30.0


async def conversation_loop():
    logger.info(">>> AURA ONLINE (Sovereign AGI Mode) <<<")
    logger.info(">>> Type 'exit' to enter REM sleep  <<<")

    # 0. Global Registration
    from core.service_registration import register_all_services
    register_all_services()

    # 1. Initialize Orchestrator via Factory (Standardized)
    from .orchestrator import create_orchestrator
    orchestrator = create_orchestrator()
    if orchestrator is None:
        logger.critical("Failed to create orchestrator. Exiting.")
        return

    # Start and run in background so cycles process (Metabolism, etc.)
    from core.utils.task_tracker import fire_and_track
    await orchestrator.start()
    fire_and_track(orchestrator.run(), name="OrchestratorMainLoop")

    consecutive_failures = 0
    backoff = INITIAL_BACKOFF_SECONDS

    while True:
        try:
            # Circuit breaker: if too many failures, cool down
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    f"Circuit breaker tripped after {consecutive_failures} failures. "
                    f"Cooling down for {CIRCUIT_BREAKER_COOLDOWN}s..."
                )
                await asyncio.sleep(CIRCUIT_BREAKER_COOLDOWN)
                consecutive_failures = 0
                backoff = INITIAL_BACKOFF_SECONDS
                logger.info("Circuit breaker reset. Resuming.")

            # 2. Get User Input (using run_in_executor to avoid blocking event loop)
            loop = asyncio.get_running_loop()
            
            # Global exception handler to catch silent background task deaths
            def _handle_exception(loop, context):
                msg = context.get("exception", context["message"])
                logger.critical("🔥 UNHANDLED ASYNC EXCEPTION: %s", msg)
                if "exception" in context:
                    import traceback
                    traceback.print_exception(type(context["exception"]), context["exception"], context["exception"].__traceback__)
            loop.set_exception_handler(_handle_exception)
            
            # Note: We keep print("YOU: ") or input() if it's the intended CLI interface
            print("\n>>> AGI CONVERSATION ENGINE READY <<<")
            sys.stdout.flush()
            user_input = await loop.run_in_executor(None, input, "YOU: ")

            if user_input.lower() in ["exit", "quit", "sleep"]:
                logger.info("Entering REM sleep (Neural Consolidation)...")
                
                # Defer sleep/dreaming to Orchestrator's unified system
                await orchestrator._process_message("System Command: ENTER_REM_SLEEP")
                break

            # 3. The Orchestrator handles the flow entirely
            response_dict = await orchestrator._process_message(user_input)

            # Reset failure counter on success
            consecutive_failures = 0
            backoff = INITIAL_BACKOFF_SECONDS

        except KeyboardInterrupt:
            logger.info("Sudden interruption. Saving state...")
            break
        except asyncio.CancelledError:
            logger.info("Conversation loop cancelled.")
            break
        except (OSError, ConnectionError) as e:
            consecutive_failures += 1
            logger.error("I/O error in conversation loop: %s", e)
            await asyncio.sleep(min(backoff, MAX_BACKOFF_SECONDS))
            backoff *= 2
        except Exception as e:
            consecutive_failures += 1
            logger.error("Unexpected error in conversation loop: %s", e, exc_info=True)
            await asyncio.sleep(min(backoff, MAX_BACKOFF_SECONDS))
            backoff *= 2


if __name__ == "__main__":
    # Configure basic logging for CLI
    logging.basicConfig(level=logging.WARNING, format='%(name)s: %(message)s')
    try:
        asyncio.run(conversation_loop())
    except KeyboardInterrupt:
        import logging
        logger.debug("Exception caught during execution", exc_info=True)