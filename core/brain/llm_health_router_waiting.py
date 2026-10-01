"""How long a caller waits on a model call, measured by whether the work is still moving.

Lifted whole out of `llm_health_router`, which imports it straight back, so
every caller and every patch that names it there still finds it. Seven modules
imported the router for this wait and nothing else, and each depended on the
whole router to wait on a task.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.task_ownership import create_owned_asyncio_task

#: The router's logger, so the wait's lines read as they always have.
logger = logging.getLogger("Brain.HealthRouter")


#: What a turn of this shape costs, for sizing the ceiling a waiting person gets.
#:
#: The tool loop's own allowance is ``max(4, 2 * len(tools) + 2)``, and a turn
#: reaching for one capability is offered one tool.
_GENERATIONS_A_TOOL_TURN_MAY_TAKE = 4

#: The scaffold a desktop turn carries, taken off the prompts this lane logged
#: on 2026-08-29 as the loop accumulated what it had read: 3273, 3702, 5319,
#: 7334 and 8324 characters.
_A_TURNS_PROMPT_CHARS = 8324

#: The tool-call budget a foreground turn is given.
_A_TURNS_ANSWER_TOKENS = 2048


async def _await_while_it_is_working(
    coro: Any,  # a coroutine, or a task a caller already owns
    *,
    budget_s: float,
    user_facing: bool,
    person_is_waiting: bool = False,
) -> Any:
    """Renew a foreground wait from owned progress; keep probe waits bounded.

    The endpoint owns worker liveness and cancellation. This outer wait must
    drain that cancellation before returning, so a timed-out request cannot
    leave its generation holding the lane.
    """
    from core.runtime.response_policy import USER_FACING_COMPLETION_DEADLINE_MAX_S
    from core.runtime.turn_outcome import current_turn
    from core.runtime.turn_progress import (
        capture_progress,
        normal_gap_between_tokens,
        seconds_since_progress,
        still_producing,
    )

    progress = capture_progress()
    owned_foreground = user_facing and person_is_waiting and current_turn() is not None
    # A caller that already owns a task hands it over rather than a coroutine,
    # and wrapping a task in another task raises "a coroutine was expected".
    # The kernel does exactly that: it creates the phase task so it can name
    # and track it, then asks this to wait on it while the person waits.
    task = coro if isinstance(coro, asyncio.Task) else create_owned_asyncio_task(coro)
    started = time.monotonic()
    try:
        done, _ = await asyncio.wait({task}, timeout=budget_s)
        if done:
            return task.result()
        if not user_facing:
            raise TimeoutError

        if owned_foreground:
            # The endpoint owns first-token, token-livelock, worker-heartbeat,
            # memory-pressure and cancellation decisions.  This outer estimate
            # cannot observe native MLX work while the event loop or response
            # queue is delayed, so silence here is not evidence that the owned
            # request stopped.  Wait for an explicit endpoint terminal state or
            # for the caller to cancel the turn.
            # Two different quantities, and the runtime has one number for
            # them: how long the WORK should take, and how long a PERSON will
            # wait. The endpoint's first-token ceiling is the first — it is
            # sized from the prompt and the token budget — and this wait is
            # the second.
            #
            # LIVE, 2026-09-08: a desktop turn with a 49,136-character prompt
            # got a 900-second first-token ceiling, the 27B took the job on a
            # host at 11.8GB free with a 9B already resident, spent fifteen
            # minutes at half a core paging weights, and produced no first
            # token. `is_inference_ready()` said False for 631 seconds while
            # it happened. Nothing was broken; the person was simply not being
            # served, and nothing said so.
            #
            # Still waiting, deliberately: the endpoint owns first-token,
            # livelock, heartbeat, memory-pressure and cancellation, and this
            # outer estimate cannot see native MLX work while the loop is
            # delayed. What changes is that the wait past a person's patience
            # is now named, with the number it passed.
            logger.info(
                "Endpoint past its %.1fs estimate; waiting for its owned terminal state.",
                budget_s,
            )
            try:
                from core.brain.llm.mlx_client import longest_a_turn_may_take

                a_person_waits = float(longest_a_turn_may_take())
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                logger.debug("Turn-length ceiling unavailable, waiting from zero: %s", exc)
                a_person_waits = 0.0
            if a_person_waits > 0.0 and budget_s < a_person_waits:
                async def _say_when_it_passes_a_persons_patience() -> None:
                    try:
                        await asyncio.sleep(max(0.0, a_person_waits - budget_s))
                    except asyncio.CancelledError:
                        # Not a failure: cancellation is the caller's decision, not a fault in the wait.
                        return
                    if not task.done():
                        logger.warning(
                            "A person has been waiting %.0fs for a first token, "
                            "past the %.0fs a turn is meant to take; still "
                            "waiting because the endpoint owns the terminal "
                            "state and has not given one.",
                            a_person_waits,
                            a_person_waits,
                        )

                watcher = create_owned_asyncio_task(_say_when_it_passes_a_persons_patience())
                try:
                    return await task
                finally:
                    watcher.cancel()
            return await task

        try:
            from core.brain.llm.thinking_reserve import seconds_to_decode

            quiet_for = normal_gap_between_tokens(float(seconds_to_decode(64)))
        except (ImportError, AttributeError, TypeError, ValueError):
            quiet_for = normal_gap_between_tokens()
        limit = float(USER_FACING_COMPLETION_DEADLINE_MAX_S)
        if person_is_waiting:
            from core.brain.llm.mlx_client import longest_a_turn_may_take

            limit = longest_a_turn_may_take(
                generations=_GENERATIONS_A_TOOL_TURN_MAY_TAKE,
                prompt_chars=_A_TURNS_PROMPT_CHARS,
                max_tokens=_A_TURNS_ANSWER_TOKENS,
                floor_s=limit,
            )
        ceiling = max(0.0, limit - float(budget_s))
        overrun = ceiling if person_is_waiting else min(ceiling, max(0.0, float(budget_s) * 3.0))
        ends_at = time.monotonic() + overrun
        said_it_once = False
        while time.monotonic() < ends_at:
            if task.done():
                return task.result()
            if not still_producing(within_s=quiet_for, progress=progress):
                break
            if not said_it_once:
                said_it_once = True
                logger.info(
                    "Endpoint past its %.1fs estimate; owned work is still advancing.",
                    budget_s,
                )
            interval = min(2.0, max(0.0, ends_at - time.monotonic()))
            done, _ = await asyncio.wait({task}, timeout=interval)
            if done:
                return task.result()

        since = seconds_since_progress(progress=progress)
        logger.warning(
            "Endpoint stopped %.1fs past its estimate: %s.",
            max(0.0, time.monotonic() - started - float(budget_s)),
            "no work reported for this request"
            if since < 0.0
            else f"last owned progress {since:.1f}s ago, quiet window {quiet_for:.0f}s",
        )
        raise TimeoutError
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # Not a failure: cancellation is the caller's decision, not a fault in the await.
                pass
            except Exception as exc:  # noqa: BLE001 - recorded below, not swallowed
                record_degradation(
                    "llm_health_router",
                    exc,
                    action="abandoned the cancelled endpoint task",
                )
