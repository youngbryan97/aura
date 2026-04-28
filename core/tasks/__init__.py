from core.runtime.errors import record_degradation
import os
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any

from .celery_app import celery_app

logger = logging.getLogger("Aura.Tasks")

def dispatch_user_input(message: str):
    """Unified helper to dispatch user input, bypassing Celery to guarantee delivery."""
    try:
        # Force local execution so the orchestrator actually receives the message
        process_user_input(message)
    except Exception as e:
        record_degradation('__init__', e)
        logger.error("Dispatch failed: %s.", e)

# Configuration for Celery is now managed in core.tasks.celery_app

# C-08 FIX: Removed run_orchestrator task.
# The orchestrator belongs on the main event loop, not in a Celery worker.

@celery_app.task(name="core.tasks.process_user_input")
def process_user_input(message: str):
    """
    Dispatches user input to the running orchestrator via message queue or global state.
    """
    logger.info("📥 Received user input: %s...", message[:50])
    
    # Bridge to the running orchestrator via EventBus
    from core.event_bus import get_event_bus
    try:
        bus = get_event_bus()
        # Use publish_threadsafe to handle cases where we are in a thread (like local fallback)
        # or another loop is running.
        logger.debug("Publishing to EventBus topic 'user_input'...")
        bus.publish_threadsafe("user_input", {"message": message})
        logger.debug("EventBus publication successful.")
        return {"status": "dispatched"}
    except Exception as e:
        record_degradation('__init__', e)
        logger.error("EventBus publication failed: %s", e)
        return {"status": "error", "message": str(e)}

@celery_app.task(name="core.tasks.run_rl_training")
def run_rl_training():
    """Executes RL training as a managed Celery task."""
    logger.info("🧠 RL: Starting policy optimization...")
    import subprocess
    import sys
    try:
        result = subprocess.run(
            [sys.executable, "core/rl_train.py"],
            capture_output=True,
            text=True,
            check=True
        )
        return {"status": "success", "stdout": result.stdout}
    except Exception as e:
        record_degradation('__init__', e)
        logger.error("RL training failed: %s", e)
        return {"status": "error", "message": str(e)}

@celery_app.task(name="core.tasks.run_self_update")
def run_self_update():
    """Executes self-evolution update as a managed Celery task."""
    logger.info("🧬 EVO: Starting self-update cycle...")
    import subprocess
    import sys
    try:
        result = subprocess.run(
            [sys.executable, "scripts/self_update.py"],
            capture_output=True,
            text=True,
            check=True
        )
        return {"status": "success", "stdout": result.stdout}
    except Exception as e:
        record_degradation('__init__', e)
        logger.error("Self-update failed: %s", e)
        return {"status": "error", "message": str(e)}

@celery_app.task(name="core.tasks.execute_skill_task")
def execute_skill_task(skill_name: str, params: dict):
    """
    Zenith Zenith: Background execution for heavy/long-running skills.
    M-11 FIX: Use asyncio.run() for efficient loop management.
    """
    logger.info("⚡ Background execution for skill: %s", skill_name)
    
    async def _run_skill():
        from core.runtime import CoreRuntime
        rt = CoreRuntime.get_sync()
        engine = rt.container.get("capability_engine")
        return await engine.execute(skill_name, params)

    try:
        import asyncio
        return asyncio.run(_run_skill())
    except Exception as e:
        record_degradation('__init__', e)
        logger.error("❌ Background execution failed for '%s': %s", skill_name, e)
        return {"ok": False, "error": str(e)}

@celery_app.task(name="core.tasks.run_mutation_tests")
def run_mutation_tests(target_file: str):
    """Runs pytest in the background for code mutations."""
    import subprocess
    logger.info("Running mutation tests for %s...", target_file)
    try:
        result = subprocess.run(
            ["pytest", "-q", target_file],
            capture_output=True,
            text=True,
            check=False
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        record_degradation('__init__', e)
        return {"success": False, "error": str(e)}
