"""A task whose steps each verified their effect has an effect that was observed, not only asserted.

A lane that runs steps verifies each one's effect on its receipt and says nothing at the top. The receipt the dev mode
records for the whole task read only the top, so every desktop task went on the turn's ledger as asserted, never
confirmed, beside its own steps confirmed under their names, and the turn ended
"requested_effect_observed_but_unconfirmed" (live 13 and 16 November 2026, and 17 November at 01:59, after a ring was
drawn around Finder's back button in 0.03 s).
"""
from __future__ import annotations

import asyncio

import pytest

from core.conversation.turn_evidence_custody import bind_turn_evidence_custody
from core.runtime.turn_outcome import TurnOutcome, bind_turn
from core.transparency.dev_mode import DevMode, ToolExecutionTrace


def _a_step(verified: bool) -> dict:
    return {"index": 1, "action": "highlight_on_screen", "ok": verified, "effect_verified": verified,
            "effect_evidence": "drew a rectangle around 'back button'" if verified else "not found"}


def _the_task_on_the_ledger(steps: list[dict], ok: bool):
    with bind_turn(TurnOutcome(turn_id="t-steps")) as outcome, \
            bind_turn_evidence_custody(session_id="s", turn_id="t-steps"):
        asyncio.run(DevMode().complete_tool_execution(
            ToolExecutionTrace(tool_name="desktop_task", origin="desktop_ui"),
            {"ok": ok, "status": "pointed_on_screen", "receipts": steps}, 58.0))
        return {effect.name: effect for effect in outcome.effects()}


def test_every_step_verified_is_the_task_observed():
    effect = _the_task_on_the_ledger([_a_step(True)], ok=True)["tool:desktop_task:desktop_task"]
    assert effect.is_confirmed


@pytest.mark.parametrize("steps", [[_a_step(True), _a_step(False)], []])
def test_a_step_unverified_or_no_step_at_all_is_still_only_asserted(steps):
    effect = _the_task_on_the_ledger(steps, ok=True)["tool:desktop_task:desktop_task"]
    assert not effect.is_confirmed
