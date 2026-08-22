"""Answer "who wins?" by enumerating the game, when the game is finite.

The language model reads the rules and fills in a typed spec; the runtime
enumerates every position and reports the verdict, the winning move and the
rule that covers the losing positions. When the description does not resolve
to a solvable game, nothing is served and the turn goes on as it would have.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from core.reasoning.finite_game import describe_solution, solve_game
from core.reasoning.game_planner import describes_a_game, game_schema, plan_from_json
from core.runtime.errors import record_degradation

logger = logging.getLogger(__name__)

__all__ = ["solve_described_game"]

#: A spec is small. Enumeration is the expensive part and it is microseconds.
_PLAN_TOKENS = 700


async def _ask_the_model(text: str) -> str:
    from core.container import ServiceContainer

    gate = ServiceContainer.get("inference_gate", default=None)
    if gate is None or not hasattr(gate, "think"):
        return ""
    return str(
        await gate.think(
            text,
            system_prompt="Return JSON only. No prose, no explanation.",
            max_tokens=_PLAN_TOKENS,
            temperature=0.0,
            origin="reasoning.game_solver",
            serves_current_turn=True,
        )
        or ""
    )


async def solve_described_game(
    message: object, *, propose: Callable[[str], Awaitable[str]] | None = None
) -> str:
    """The solved answer for a described game, or "" when there is not one."""
    text = str(message or "")
    if not describes_a_game(text):
        return ""
    import json

    request = (
        f"Rules as given:\n{text.strip()}\n\n"
        f"Return one JSON object matching this schema and nothing else.\n"
        f"{json.dumps(game_schema(), indent=1)}\n"
    )
    ask = propose or _ask_the_model
    try:
        raw = await ask(request)
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
        record_degradation(
            "reasoning.game_answer",
            exc,
            severity="debug",
            action="left the game question to the model",
            enforce_failure_policy=False,
        )
        return ""
    spec = plan_from_json(raw)
    if spec is None:
        logger.info("game: no solvable spec came back (%d chars).", len(raw or ""))
        return ""
    solution = await asyncio.to_thread(solve_game, spec)
    described = describe_solution(spec, solution)
    if described:
        logger.info(
            "♟️ Solved %r by enumeration: %d positions.",
            spec.title,
            solution.positions_examined if solution else 0,
        )
    return described
