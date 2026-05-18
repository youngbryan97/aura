"""Celery task definitions for background work.

The celery_app instance is defined once in core/tasks/celery_app.py.
When Celery/Redis isn't available, a MockCelery shim routes tasks locally.
"""
import logging

from core.runtime.errors import record_degradation
from core.tasks.celery_app import celery_app
from core.tasks.managed_command import run_project_pytest, run_project_python

logger = logging.getLogger("Aura.Tasks")

_RECOVERABLE_TASK_ERRORS = (
    AttributeError,
    ImportError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def dispatch_user_input(message: str):
    """Dispatch user input, falling back to local execution if Redis is down."""
    from core.config import config

    try:
        if getattr(config.redis, "enabled", True):
            celery_app.send_task("core.tasks.process_user_input", args=[message])
        else:
            process_user_input(message)
    except _RECOVERABLE_TASK_ERRORS as exc:
        record_degradation("tasks", exc)
        logger.error("Dispatch failed: %s. Falling back to local execution.", exc)
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
    except _RECOVERABLE_TASK_ERRORS as exc:
        record_degradation("tasks", exc)
        logger.critical("Orchestrator task failed: %s", exc, exc_info=True)
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
    except _RECOVERABLE_TASK_ERRORS as exc:
        record_degradation("tasks", exc)
        logger.error("EventBus publication failed: %s", exc)
        return {"status": "error", "message": str(exc)}


@celery_app.task(name="core.tasks.run_rl_training")
def run_rl_training():
    """Execute RL training as a background task."""
    logger.info("RL: Starting policy optimization...")
    result = run_project_python("core/rl_train.py")
    payload = result.status_payload()
    if not result.ok:
        logger.error("RL training failed: %s", payload["message"])
    return payload


@celery_app.task(name="core.tasks.run_self_update")
def run_self_update():
    """Execute self-evolution update as a background task."""
    logger.info("EVO: Starting self-update cycle...")
    result = run_project_python("scripts/self_update.py")
    payload = result.status_payload()
    if not result.ok:
        logger.error("Self-update failed: %s", payload["message"])
    return payload


@celery_app.task(name="core.tasks.run_mutation_tests")
def run_mutation_tests(target_file: str):
    """Run pytest in the background for code mutations."""
    logger.info("Running mutation tests for %s...", target_file)
    return run_project_pytest(target_file).mutation_payload()
