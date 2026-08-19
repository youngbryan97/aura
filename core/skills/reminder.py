"""Setting a reminder, so that saying she set one is true.

LIVE 2026-08-19: "remind me in 20 minutes to check the oven" was answered "I've
set a reminder for 20 minutes to check the oven." No tool ran and no reminder
existed. The person stops thinking about the oven, which is the whole cost of
the sentence.

The store this uses is durable and its failure branch is the important one:
when nothing could be written, this reports ok=False and says the reminder was
NOT set. A receipt that lies is worse than no capability at all, which is the
defect it was built from.
"""

from __future__ import annotations

from typing import Any

from core.skills.base_skill import BaseSkill


class ReminderSkill(BaseSkill):
    name = "reminder"
    description = (
        "Set a reminder or timer for later — 'remind me in 20 minutes to check "
        "the oven', 'set a timer for 5 minutes' — and list what is outstanding. "
        "Reminders are stored durably and reported as due when asked what is "
        "queued."
    )
    effect_scope = "state_mutation"
    inputs = {"objective": "The reminder request, in the person's own words."}
    output = "A confirmation naming the stored reminder, or a statement that it was not set."

    def match(self, goal: dict[str, Any]) -> bool:
        from core.agency.reminders import requested_reminder

        return requested_reminder(str(goal.get("objective") or "")) is not None

    async def execute(self, goal: Any, context: dict[str, Any]) -> dict[str, Any]:
        from core.agency.reminders import (
            add_reminder,
            pending_reminders,
            requested_reminder,
            spoken_delay,
        )

        objective = str(
            (goal or {}).get("objective") if isinstance(goal, dict) else goal or ""
        )
        asked = requested_reminder(objective)
        if asked is None:
            return {
                "ok": False,
                "error": "no_reminder_requested",
                "summary": (
                    "That did not name a delay, so there is nothing to set. Ask "
                    "when it should come back."
                ),
            }
        stored = add_reminder(asked.text, asked.delay_s)
        if stored is None:
            return {
                "ok": False,
                "error": "reminder_not_stored",
                "effect_verified": False,
                "summary": (
                    "I could not store that reminder, so it is NOT set and "
                    "nothing will come back to you about it."
                ),
            }
        outstanding = len(pending_reminders())
        return {
            "ok": True,
            "effect_verified": True,
            "effect_evidence": f"reminder_id={stored.id};due_at={stored.due_at:.0f}",
            "reminder_id": stored.id,
            "text": stored.text,
            "due_at": stored.due_at,
            "outstanding": outstanding,
            "summary": (
                f"Reminder set: {stored.text} — in {spoken_delay(asked.delay_s)} "
                f"(id {stored.id}). {outstanding} outstanding."
            ),
        }
