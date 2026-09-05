"""Turn-scoped, durable progress publication for foreground chat work.

Progress is operational evidence, not invented chain-of-thought.  The active
delivery owner binds the journal at the HTTP boundary; deeper planners and
effect executors can then publish factual phase changes without depending on
an interface module or a particular UI transport.
"""

from __future__ import annotations

import inspect
import logging
import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from core.runtime.chat_delivery_journal import (
    ChatDeliveryFenceLost,
    ChatDeliveryJournalError,
    DeliveryAdmission,
)

logger = logging.getLogger("Aura.Chat.Progress")


@dataclass(frozen=True)
class ChatDeliveryProgressBinding:
    journal: Any
    admission: DeliveryAdmission


_CURRENT_BINDING: ContextVar[ChatDeliveryProgressBinding | None] = ContextVar(
    "aura_chat_delivery_progress_binding",
    default=None,
)


def capture_generation_progress() -> Callable[..., None] | None:
    """Capture delivery ownership before worker callbacks leave the turn context."""
    binding = _CURRENT_BINDING.get()
    if binding is None:
        return None
    loop = asyncio.get_running_loop()
    pending = False
    last_phase = ""
    last_at = float("-inf")

    async def publish(phase: str, completed: int, total: int) -> None:
        nonlocal pending
        try:
            await binding.journal.publish_progress(
                binding.admission,
                phase=phase,
                message=("Reading the conversation context." if phase == "prefill"
                         else "Working through the response."),
                details={"completed_tokens": completed, "total_tokens": total},
            )
        except (ChatDeliveryJournalError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Generation progress could not be published: %s", exc)
        finally:
            pending = False

    def observe(phase: str, completed: int, total: int) -> None:
        nonlocal pending, last_phase, last_at
        now = loop.time()
        if pending or (phase == last_phase and now - last_at < 2.0):
            return
        pending = True
        last_phase, last_at = phase, now
        from core.utils.task_tracker import get_task_tracker

        get_task_tracker().create_task(
            publish(phase, completed, total), name="ChatGenerationProgress"
        )

    def report(*, phase: str, completed: int, total: int = 0) -> None:
        if not loop.is_closed():
            loop.call_soon_threadsafe(observe, phase, completed, total)

    return report


@contextmanager
def bind_chat_delivery_progress(
    journal: Any,
    admission: DeliveryAdmission,
) -> Iterator[ChatDeliveryProgressBinding]:
    """Bind progress to the exact fenced owner for this execution context."""

    binding = ChatDeliveryProgressBinding(journal=journal, admission=admission)
    token = _CURRENT_BINDING.set(binding)
    try:
        yield binding
    finally:
        _CURRENT_BINDING.reset(token)


async def report_chat_delivery_progress(
    *,
    phase: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> bool:
    """Publish factual progress for the current turn when one is bound.

    Progress is supplementary to terminal delivery.  Losing a stale fence or
    an unavailable progress write must not erase a result that can still be
    sealed by the terminal journal operation.
    """

    binding = _CURRENT_BINDING.get()
    if binding is None:
        return False
    try:
        await binding.journal.publish_progress(
            binding.admission,
            phase=phase,
            message=message,
            details=dict(details or {}),
        )
        return True
    except ChatDeliveryFenceLost:
        logger.debug("Skipped progress after the chat delivery fence moved")
    except (ChatDeliveryJournalError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("Durable chat progress publication failed: %s", exc)
    return False


def current_task_progress_callback() -> Callable[[dict[str, Any]], Any] | None:
    """Return an async task-engine callback bound to the current delivery."""

    if _CURRENT_BINDING.get() is None:
        return None

    async def _publish(event: dict[str, Any]) -> None:
        payload = dict(event or {})
        event_kind = str(payload.get("event") or "step_update").strip().casefold()
        description = " ".join(str(payload.get("description") or "").split())
        status = str(payload.get("status") or "").strip().casefold()
        completed = payload.get("steps_completed")
        total = payload.get("steps_total")
        details = {
            key: payload[key]
            for key in (
                "event",
                "plan_id",
                "step_id",
                "tool",
                "status",
                "verified",
                "steps_completed",
                "steps_total",
            )
            if key in payload
        }

        if event_kind == "planning":
            phase = "planning"
            message = "Building a grounded execution plan."
        elif event_kind == "plan_ready":
            phase = "planning"
            message = f"The plan is ready with {total} executable step{'s' if total != 1 else ''}."
        elif event_kind == "step_started":
            phase = "executing"
            message = f"Working on: {description}" if description else "Starting the next verified step."
        elif status == "succeeded":
            phase = "executing"
            message = f"Completed: {description}" if description else "A verified step completed."
        elif status in {"failed", "skipped", "rolled_back"}:
            phase = "repairing"
            message = (
                f"That step needs another approach: {description}"
                if description
                else "A step needs another approach."
            )
        else:
            phase = "executing"
            message = f"Continuing: {description}" if description else "Continuing the task."
        if isinstance(completed, int) and isinstance(total, int) and total > 0:
            message = f"{message} ({completed}/{total})"
        await report_chat_delivery_progress(
            phase=phase,
            message=message,
            details=details,
        )

    return _publish


async def invoke_progress_callback(
    callback: Callable[[dict[str, Any]], Any] | None,
    payload: dict[str, Any],
) -> None:
    """Invoke either a legacy sync callback or the durable async callback."""

    if callback is None:
        return
    result = callback(dict(payload))
    if inspect.isawaitable(result):
        await result


__all__ = [
    "bind_chat_delivery_progress",
    "current_task_progress_callback",
    "invoke_progress_callback",
    "report_chat_delivery_progress",
]
