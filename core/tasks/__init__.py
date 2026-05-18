import logging

from core.runtime.errors import record_degradation

from .celery_app import celery_app
from .managed_command import run_project_pytest, run_project_python

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
    """Unified helper to dispatch user input, bypassing Celery to guarantee delivery."""
    try:
        # Force local execution so the orchestrator actually receives the message
        process_user_input(message)
    except _RECOVERABLE_TASK_ERRORS as exc:
        record_degradation("tasks", exc)
        logger.error("Dispatch failed: %s.", exc)

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
    except _RECOVERABLE_TASK_ERRORS as exc:
        record_degradation("tasks", exc)
        logger.error("EventBus publication failed: %s", exc)
        return {"status": "error", "message": str(exc)}


@celery_app.task(name="core.tasks.run_rl_training")
def run_rl_training():
    """Executes RL training as a managed Celery task."""
    logger.info("🧠 RL: Starting policy optimization...")
    result = run_project_python("core/rl_train.py")
    payload = result.status_payload()
    if not result.ok:
        logger.error("RL training failed: %s", payload["message"])
    return payload


@celery_app.task(name="core.tasks.run_self_update")
def run_self_update():
    """Executes self-evolution update as a managed Celery task."""
    logger.info("🧬 EVO: Starting self-update cycle...")
    result = run_project_python("scripts/self_update.py")
    payload = result.status_payload()
    if not result.ok:
        logger.error("Self-update failed: %s", payload["message"])
    return payload


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
    except _RECOVERABLE_TASK_ERRORS as exc:
        record_degradation("tasks", exc)
        logger.error("❌ Background execution failed for '%s': %s", skill_name, exc)
        return {"ok": False, "error": str(exc)}


@celery_app.task(name="core.tasks.run_mutation_tests")
def run_mutation_tests(target_file: str):
    """Runs pytest in the background for code mutations."""
    logger.info("Running mutation tests for %s...", target_file)
    return run_project_pytest(target_file).mutation_payload()
