import logging
import threading
import time
from typing import Any

from core.runtime.errors import record_degradation

from .celery_app import CELERY_AVAILABLE, celery_app
from .managed_command import run_project_pytest, run_project_python

logger = logging.getLogger("Aura.Tasks")
_TASK_STATUS_LOCK = threading.RLock()
_BACKGROUND_TASK_STATUS: dict[str, dict[str, Any]] = {}

_RECOVERABLE_TASK_ERRORS = (
    AttributeError,
    ImportError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _broker_active() -> bool:
    """True only when a real Celery broker is configured AND importable.

    Otherwise the active ``celery_app`` is the MockCelery shim, whose
    ``send_task`` only RECORDS calls — work must run locally instead of being
    handed to a no-op that silently drops it. The desktop/consumer default is
    no broker (see RedisConfig in core/config.py).
    """
    from core.config import config

    return CELERY_AVAILABLE and getattr(config.redis, "enabled", False)


def _set_background_task_status(name: str, state: str, **details: Any) -> None:
    with _TASK_STATUS_LOCK:
        previous = _BACKGROUND_TASK_STATUS.get(name, {})
        _BACKGROUND_TASK_STATUS[name] = {
            "name": name,
            "state": state,
            "created_at": float(previous.get("created_at", time.time())),
            "updated_at": time.time(),
            **details,
        }


def get_background_task_status(name: str) -> dict[str, Any]:
    """Return the latest observable outcome for a background task route."""
    with _TASK_STATUS_LOCK:
        return dict(_BACKGROUND_TASK_STATUS.get(name, {}))


def dispatch_user_input(message: str):
    """Unified helper to dispatch user input, bypassing Celery to guarantee delivery."""
    try:
        # Force local execution so the orchestrator actually receives the message.
        # (process_user_input just publishes a fast, thread-safe EventBus event,
        # which is itself Redis-bridged when running distributed.)
        process_user_input(message)
    except _RECOVERABLE_TASK_ERRORS as exc:
        record_degradation("tasks", exc)
        logger.error("Dispatch failed: %s.", exc)


def dispatch_background(name: str, args=None, kwargs=None) -> str:
    """Route a registered background task to Celery or a local daemon thread.

    Returns the route taken: ``"celery"``, ``"local"`` or ``"skipped"``.

    Running locally happens off the event loop (a daemon thread) so heavyweight
    tasks (RL training, self-update) neither block the loop nor get silently
    dropped by the MockCelery shim. The underlying entrypoints are themselves
    bounded (see core/tasks/managed_command.py).
    """
    args = list(args or [])
    kwargs = dict(kwargs or {})
    if _broker_active():
        try:
            celery_app.send_task(name, args=args, kwargs=kwargs)
            _set_background_task_status(name, "submitted", route="celery")
            return "celery"
        except _RECOVERABLE_TASK_ERRORS as exc:
            record_degradation(
                "tasks",
                exc,
                severity="warning",
                action="fell back to the registered local handler after broker dispatch failed",
            )
            logger.error("Celery dispatch of %s failed: %s. Running locally.", name, exc)
    fn = _LOCAL_TASKS.get(name)
    if fn is None:
        _set_background_task_status(name, "missing", route="none")
        logger.error("No local handler registered for background task %s; skipping.", name)
        return "skipped"

    def _runner():
        _set_background_task_status(name, "running", route="local")
        try:
            result = fn(*args, **kwargs)
            _set_background_task_status(
                name,
                "succeeded",
                route="local",
                result_type=type(result).__name__,
            )
        except _RECOVERABLE_TASK_ERRORS as exc:
            _set_background_task_status(
                name,
                "failed",
                route="local",
                error_type=type(exc).__name__,
                error=str(exc)[:500],
            )
            record_degradation(
                "tasks",
                exc,
                severity="degraded",
                action="recorded the terminal local-task failure for health and operator inspection",
            )
            logger.error("Local background task %s failed: %s", name, exc)

    short = name.rsplit(".", 1)[-1]
    _set_background_task_status(name, "queued", route="local")
    threading.Thread(target=_runner, name=f"aura-task-{short}", daemon=True).start()
    return "local"

# Configuration for Celery is now managed in core.tasks.celery_app

# C-08 FIX: Removed run_orchestrator task.
# The orchestrator belongs on the main event loop, not in a Celery worker.

@celery_app.task(name="core.tasks.process_user_input")
def process_user_input(message: str):
    """
    Dispatches user input to the running orchestrator via message queue or global state.
    """
    logger.info("Received user input: %s...", message[:50])

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
    logger.info("RL: Starting policy optimization...")
    result = run_project_python("core/learning/rl_train.py")
    payload = result.status_payload()
    if not result.ok:
        logger.error("RL training failed: %s", payload["message"])
    return payload


@celery_app.task(name="core.tasks.run_self_update")
def run_self_update():
    """Executes self-evolution update as a managed Celery task."""
    logger.info("EVO: Starting self-update cycle...")
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
    logger.info("Background execution for skill: %s", skill_name)

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
        logger.error("Background execution failed for '%s': %s", skill_name, exc)
        return {"ok": False, "error": str(exc)}


@celery_app.task(name="core.tasks.run_mutation_tests")
def run_mutation_tests(target_file: str):
    """Runs pytest in the background for code mutations."""
    logger.info("Running mutation tests for %s...", target_file)
    return run_project_pytest(target_file).mutation_payload()


# Local handlers for dispatch_background when no Celery broker is configured.
# Keep in sync with the @celery_app.task names above.
_LOCAL_TASKS = {
    "core.tasks.process_user_input": process_user_input,
    "core.tasks.run_rl_training": run_rl_training,
    "core.tasks.run_self_update": run_self_update,
    "core.tasks.execute_skill_task": execute_skill_task,
    "core.tasks.run_mutation_tests": run_mutation_tests,
}
