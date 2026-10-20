"""Self-play asks her model for a problem and its resolution; neither becomes an objective she plans.

From 10 to 18 October the adversary's request reached the cognitive engine as an objective, and the task
engine started 218 goals named "[SYSTEM ROLE: THE ADVERSARY]..." with 224 commitments kept to them.
"""
from __future__ import annotations

import asyncio

import core.agency.self_play as self_play


def test_problem_and_resolution_go_to_her_model_as_requests(monkeypatch):
    asked = []

    class Router:
        async def think(self, **kwargs):
            asked.append(kwargs)
            return "A problem." if kwargs["purpose"] == "self_play_problem" else "Because of this, the final answer is 4."

    class Engine:
        async def think(self, **kwargs):
            raise AssertionError("self-play reached the cognitive engine as an objective")

    services = {"llm_router": Router(), "cognitive_engine": Engine()}
    monkeypatch.setattr(self_play, "get_runtime_service", lambda name, default=None: services.get(name, default))
    play = self_play.ContinuousSelfPlay()
    problem = asyncio.run(play._generate_adversarial_problem())
    solution = asyncio.run(play._attempt_solution(problem))
    assert problem == "A problem." and solution.endswith("4.")
    assert all(call["origin"] == "self_play" and call["is_background"] for call in asked)
    assert not any("SYSTEM ROLE" in call["prompt"] for call in asked)
    assert "A problem." in asked[1]["prompt"]
