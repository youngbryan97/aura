"""A reminder reaches the person when it comes due, on their desktop.

core/agency/reminders.py keeps a reminder durably, and nothing in it notifies: a reminder came due and was said only
to someone who asked what was queued. LIVE 19 November 2026, "Can you remind me in 20 minutes to stretch my back?":
had it been set, nobody would have been told in twenty minutes.

One loop here waits until the soonest pending reminder is due, or until a reminder is added, and sends each due one
to her notify_user capability through her capability engine, under the grant that lets a reminder go out and nothing
else (core/executive/her_own_passes.py, REMINDER_DUE_GRANT). A reminder is marked done only when its notification
went out. One that did not go out stays pending and is said; it is tried again when the store next changes, not in a
loop against the same refusal.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.RemindersDelivered")

__all__ = ["a_reminder_was_added", "deliver_reminders_as_they_come_due", "ensure_reminders_are_delivered"]

#: The title her notification carries: what the person sees above the words they asked to be reminded of.
TITLE = "Reminder"

_RUNNING: dict[str, Any] = {}


def a_reminder_was_added() -> None:
    """Wake the loop: the store changed. Safe from any thread; nothing happens when no loop is running."""
    loop, woken = _RUNNING.get("loop"), _RUNNING.get("changed")
    if loop is None or woken is None or loop.is_closed():
        return
    try:
        loop.call_soon_threadsafe(woken.set)
    except RuntimeError:  # the loop closed between the check and the call
        return


async def _notified(reminder: Any) -> bool:
    """Whether her notify_user capability sent ``reminder`` to the desktop."""
    from core.container import ServiceContainer
    from core.executive.her_own_passes import REMINDER_DUE_ORIGIN

    engine = ServiceContainer.get("capability_engine", default=None)
    if engine is None or not hasattr(engine, "execute"):
        return False
    result = await engine.execute("notify_user", {"title": TITLE, "message": reminder.text},
                                  {"origin": REMINDER_DUE_ORIGIN, "source": REMINDER_DUE_ORIGIN,
                                   "objective": f"reminder due: {reminder.text}"})
    if isinstance(result, dict) and result.get("ok"):
        return True
    logger.info("a reminder came due and did not go out: %r (%s)", reminder.text[:120],
                str((result or {}).get("error") or (result or {}).get("status") or "no result")[:200]
                if isinstance(result, dict) else "no result")
    return False


async def deliver_reminders_as_they_come_due(notify: Callable[[Any], Awaitable[bool]] | None = None,
                                            *, until: Callable[[], bool] | None = None) -> None:
    """Send each reminder as it comes due, for as long as the runtime runs (or until ``until()`` says so)."""
    from core.agency.reminders import complete_reminder, pending_reminders

    notify = notify or _notified
    changed = asyncio.Event()
    _RUNNING.update(loop=asyncio.get_running_loop(), changed=changed)
    not_sent: set[str] = set()
    sent_not_marked: set[str] = set()
    try:
        while True:
            changed.clear()
            pending = await asyncio.to_thread(pending_reminders)
            for reminder in pending:
                if reminder.is_due and reminder.id not in not_sent | sent_not_marked:
                    try:
                        sent = await notify(reminder)
                    except Exception as exc:  # noqa: BLE001 - one reminder that cannot go out does not stop the rest
                        record_degradation("reminders_delivered", exc, severity="warning",
                                           action="a due reminder was not sent; it stays pending")
                        sent = False
                    if sent:
                        logger.info("a reminder came due and went to the desktop: %r", reminder.text[:120])
                        if not await asyncio.to_thread(_marked_done, complete_reminder, reminder.id):
                            # It went out; it is not sent again because the store could not say so.
                            sent_not_marked.add(reminder.id)
                    else:
                        not_sent.add(reminder.id)
            if until is not None and until():
                return
            upcoming = [reminder.due_at for reminder in pending if not reminder.is_due]
            wait = max(0.0, min(upcoming) - time.time()) if upcoming else None
            try:
                await asyncio.wait_for(changed.wait(), timeout=wait)
                not_sent.clear()  # the store changed: what did not go out is tried again with it
            except TimeoutError:
                pass
    finally:
        if _RUNNING.get("changed") is changed:
            _RUNNING.clear()


def _marked_done(complete: Callable[[str], bool], reminder_id: str) -> bool:
    """Mark a sent reminder done, as her runtime's own upkeep: a write from this loop is no person's request.

    LIVE 21 November 2026, 19:41: the first reminder this loop sent went out, and marking it done was refused as a
    write "called outside governed context", so the store kept it pending and the next change would have sent it again.
    """
    from core.governance_context import local_internal_governed_scope

    try:
        with local_internal_governed_scope("reminders_delivered.mark_done", domain="file_write"):
            done = bool(complete(reminder_id))
    except Exception as exc:  # noqa: BLE001 - a reminder that went out is never sent again for want of a mark
        record_degradation("reminders_delivered", exc, severity="warning",
                           action="a reminder went out and was not marked done; it is not sent again this run")
        return False
    if not done:
        logger.warning("a reminder went out and could not be marked done: %s", reminder_id)
    return done


def ensure_reminders_are_delivered() -> bool:
    """Start the loop on the running event loop, once; False when it was already running or there is no loop."""
    task = _RUNNING.get("task")
    if task is not None and not task.done():
        return False
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    from core.utils.task_tracker import get_task_tracker

    _RUNNING["task"] = get_task_tracker().create_task(deliver_reminders_as_they_come_due(),
                                                      name="reminders_delivered")
    return True
