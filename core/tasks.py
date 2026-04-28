"""Celery task definitions for background work.

The celery_app instance is defined once in core/tasks/celery_app.py.
When Celery/Redis isn't available, a MockCelery shim routes tasks locally.
"""
from core.runtime.errors import record_degradation
import logging
from typing import Dict, Any

from core.tasks.celery_app import celery_app

logger = logging.getLogger("Aura.Tasks")


def dispatch_user_input(message: str):
    """Dispatch user input, falling back to local execution if Redis is down."""
    from core.config import config
    try:
        if getattr(config.redis, "enabled", True):
            celery_app.send_task("core.tasks.process_user_input", args=[message])
        else:
            process_user_input(message)
    except Exception as e:
        record_degradation('tasks', e)
        logger.error("Dispatch failed: %s. Falling back to local execution.", e)
        process_user_input(message)


@celery_app.task(name="core.tasks.run_orchestrator")
def run_orchestrator():
    """Long-running task that executes the main cognitive orchestrator loop."""
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    logger.info("Brain Boot: Starting Decoupled Orchestrator...")

    try:
        from core.orchestrator import create_orchestrator
        orchestrator = create_orchestrator()
        if orchestrator is None:
            logger.critical("create_orchestrator() returned None!")
            return
        logger.info("Orchestrator created, entering run loop...")
        loop.run_until_complete(orchestrator.run())
    except Exception:
        logger.critical("Orchestrator task failed", exc_info=True)
    finally:
        loop.close()


@celery_app.task(name="core.tasks.process_user_input")
def process_user_input(message: str):
    """Dispatch user input to the running orchestrator via EventBus."""
    logger.info("Received user input: %s...", message[:50])
    from core.event_bus import get_event_bus
    try:
        bus = get_event_bus()
        bus.publish_threadsafe("user_input", {"message": message})
        return {"status": "dispatched"}
    except Exception as e:
        record_degradation('tasks', e)
        logger.error("EventBus publication failed: %s", e)
        return {"status": "error", "message": str(e)}


@celery_app.task(name="core.tasks.run_rl_training")
def run_rl_training():
    """Execute RL training as a background task."""
    import subprocess, sys
    logger.info("RL: Starting policy optimization...")
    try:
        result = subprocess.run(
            [sys.executable, "core/rl_train.py"],
            capture_output=True, text=True, check=True,
        )
        return {"status": "success", "stdout": result.stdout}
    except Exception as e:
        record_degradation('tasks', e)
        logger.error("RL training failed: %s", e)
        return {"status": "error", "message": str(e)}


@celery_app.task(name="core.tasks.run_self_update")
def run_self_update():
    """Execute self-evolution update as a background task."""
    import subprocess, sys
    logger.info("EVO: Starting self-update cycle...")
    try:
        result = subprocess.run(
            [sys.executable, "scripts/self_update.py"],
            capture_output=True, text=True, check=True,
        )
        return {"status": "success", "stdout": result.stdout}
    except Exception as e:
        record_degradation('tasks', e)
        logger.error("Self-update failed: %s", e)
        return {"status": "error", "message": str(e)}


@celery_app.task(name="core.tasks.run_mutation_tests")
def run_mutation_tests(target_file: str):
    """Run pytest in the background for code mutations."""
    import subprocess
    logger.info("Running mutation tests for %s...", target_file)
    try:
        result = subprocess.run(
            ["pytest", "-q", target_file],
            capture_output=True, text=True, check=False,
        )
        return {"success": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr}
    except Exception as e:
        record_degradation('tasks', e)
        return {"success": False, "error": str(e)}
