"""What she judges a move by, asked of her own model.

Lifted whole out of `screen_pursuit`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any


def _judge_what_she_judges_by_in_her_model(
    knows: Any,
    world: Any,
    pending: dict[str, Any],
    matters: Any,
    move_keys: Sequence[str],
    toward: str,
    narrate: bool,
    *,
    within_s: float = 0.0,
    this_world: str = "",
) -> Any:
    """Rehearse every property she invented, and let go of any that does worse.

    And, the first time she has a model of a world to play in, work out what
    each thing about a situation is worth there, in a process of its own —
    see `core/agency/working_out_what_matters`. Every term starts level; what
    she finds is kept with the world, so the next game starts from it.

    Returns the task it runs as, or None where there is nothing to rehearse.

    A property kept on a live trial compared two stretches of a life. Two came
    back that way on 2048, and played out from the same four starts with her
    own rule and search they took her from 2048 in all four to 2048 in one
    (2026-09-19). Where she has a model of the world, she can ask it instead.
    """
    from core.agency.how_good_is_this import INVENTED

    from .screen_pursuit import (
        _keep_what_matters_here,
        _tell,
        logger,
        record_degradation,
    )

    # From where this run began, not where it ended: a run often ends on a
    # finished position, and from there every way of judging goes nowhere.
    start = pending.get("first_arranged") or pending.get("arranged")
    rules = getattr(knows, "rules", None)
    # Only the real record of what matters here can take a weighting: the
    # first game in a world she has a model of is where it is worked out.
    to_work_out = callable(getattr(matters, "settled_by_playing_it_out", None)) and not bool(
        getattr(matters, "played_out", False)
    )
    # Each way of having nothing to rehearse is said, at info: a faculty that
    # returns without a word looks, from outside, like one that never ran.
    if not INVENTED and not to_work_out:
        logger.info("nothing to rehearse: she judges by no property she invented here")
        return None
    if start is None or rules is None or getattr(rules, "rule", lambda: None)() is None:
        logger.info(
            "nothing to rehearse in yet: %s",
            "no board to start from" if start is None else "no rule of how this moves",
        )
        return None

    def rehearse_them() -> None:
        from core.agency import what_she_invented
        from core.agency.how_good_is_this import forget
        from core.agency.looking_ahead import _default_weights
        from core.agency.rehearsing_in_her_model import rehearse

        judged_by = dict(matters.weights() or _default_weights())
        let_go: list[str] = []
        for name in list(INVENTED):
            worth = float(judged_by.get(name, 0.0) or 0.0)
            if worth <= 0.0:
                continue
            without = {key: value for key, value in judged_by.items() if key != name}
            began_rehearsing = time.monotonic()
            rehearsed = rehearse(
                rules, world, start, list(move_keys),
                weights=without, trying={name: worth}, toward=toward, within_s=within_s,
            )
            if rehearsed is None:
                if within_s and time.monotonic() - began_rehearsing >= within_s:
                    logger.info(
                        "could not finish rehearsing %r in the %.0fs there was", name, within_s
                    )
                else:
                    logger.info(
                        "could not rehearse %r: her model of this world does not compile "
                        "from where this game began", name,
                    )
                return
            if not rehearsed.hurts():
                logger.info("kept judging by %r: %s", name, rehearsed.says())
            if rehearsed.hurts():
                forget(name)
                let_go.append(name)
                if narrate:
                    _tell(f"I stopped judging by {name!r}: {rehearsed.says()}.")
        if let_go:
            what_she_invented.keep()
            logger.info("let go of %s after playing them out in her own model", ", ".join(let_go))

    async def work_out_what_matters() -> None:
        from core.agency.looking_ahead import _default_weights
        from core.agency.working_out_what_matters import in_a_process_of_its_own

        began_working_out = time.monotonic()
        said = await in_a_process_of_its_own(
            rules, world, start, list(move_keys),
            weights=dict(matters.weights() or _default_weights()),
            toward=toward, within_s=within_s or 600.0,
        )
        if not said or not said.get("ok"):
            logger.info(
                "did not work out what matters here: %s",
                (said or {}).get("why") or "no answer from the process it ran in",
            )
            return
        matters.settled_by_playing_it_out(said.get("weights") or {}, str(said.get("said") or ""))
        logger.info(
            "worked out what matters here in %.0fs: %s",
            time.monotonic() - began_working_out, said.get("said"),
        )
        if narrate:
            _tell(f"I {said.get('said')}.")
        if this_world:
            await asyncio.to_thread(_keep_what_matters_here, this_world, matters)

    async def beside_the_run() -> None:
        try:
            if INVENTED:
                await asyncio.to_thread(rehearse_them)
            if to_work_out:
                await work_out_what_matters()
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError) as exc:
            record_degradation(
                "screen_pursuit", exc, severity="info",
                action="kept what she judges by without rehearsing it",
            )

    try:
        from core.utils.task_tracker import get_task_tracker

        return get_task_tracker().create_task(beside_the_run(), name="rehearse_what_she_judges_by")
    except (ImportError, RuntimeError, TypeError) as exc:
        record_degradation(
            "screen_pursuit", exc, severity="info", action="did not rehearse what she judges by"
        )
        return None



