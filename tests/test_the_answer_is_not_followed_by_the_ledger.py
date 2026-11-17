"""What she asked about, without the bookkeeping read out after it.

On the run where she actually reached the tile, the reply was:

    Reached it: '256' appeared after 25 move(s). 6 of them did what I
    expected. (Completed 1/1 governed desktop steps.)

The first two sentences are the answer. The third is the machinery counting
itself, and nobody asked how many governed steps it took. A word list
(``_AN_OUTCOME``: "done", "failed", "could not" ...) decided which observations
kept the count until 17 November 2026, when her record held "failed" wrong more
than once and the formed constraint refused the pattern. No observation keeps it.
"""
from __future__ import annotations

import pytest

ASKED = "Open play2048.co and get a 256 tile, then tell me how it went."


async def _reply_to(monkeypatch, observation: str) -> str:
    from interface.routes import chat as chat_routes

    class _Engine:
        async def execute(self, skill_name, params, context=None):
            return {
                "ok": True, "status": "completed", "summary": "completed 1/1 governed computer-use steps",
                "steps_requested": 1, "steps_completed": 1, "observation": observation,
                "receipts": [{"index": 0, "action": "browse_pursue", "ok": True, "effect_verified": True,
                              "effect_evidence": observation, "result": {"ok": True, "observation": observation}}],
            }

    monkeypatch.setattr(
        chat_routes.ServiceContainer, "get",
        staticmethod(lambda name, default=None: _Engine() if name == "capability_engine" else default),
    )
    monkeypatch.setattr("interface.routes.chat_preflight._looks_like_desktop_objective", lambda _text: True)
    result = await chat_routes._execute_desktop_objective_from_chat(ASKED, cognitive_reply="")
    return str((result or {}).get("response") or "")


@pytest.mark.asyncio
@pytest.mark.parametrize("observation", [
    "Reached it: '256' appeared after 25 move(s). 6 of them did what I expected.",
    "The board filled before a 256 appeared; the highest tile was 128.",
])
async def test_an_observation_is_not_followed_by_the_ledger(monkeypatch, observation):
    response = await _reply_to(monkeypatch, observation)
    assert observation in response
    assert "governed desktop steps" not in response


def test_no_word_list_decides_which_observations_keep_the_count():
    from interface.routes import chat_desktop_objective

    assert not hasattr(chat_desktop_objective, "_AN_OUTCOME")
    assert not hasattr(chat_desktop_objective, "_reports_an_outcome")
