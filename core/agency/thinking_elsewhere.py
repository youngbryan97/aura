"""Looking ahead in a process of its own, so the rest of her does not take its time.

Her search is Python holding the interpreter, and the live process is full of
other Python — her mind, her voice, her health checks, her background loops —
all wanting the same interpreter. On her own clock the search reached 3.40
moves ahead alone, 3.00 beside two busy threads and 2.56 beside six (measured
2026-09-26), and a level is the difference between a game that reaches 256
and one that reaches 2048: offline, at a fixed two moves ahead, she reached
256 to 1024; on her own clock, 2048 in six games of six.

So the search runs in a child that holds her rule and her record of what
arrives, in the form they are remembered in, and answers with the scores and
how far it saw. Her eyes read in a process of their own for the same reason.
Nothing here decides anything: if the child is missing, slow or broken, the
caller thinks in its own process instead and is only slower.

The rule and the record change rarely and the board every move, so the child
keeps the last ones it was sent and is sent them again only when they change.
The boards she has stood on grow by one a move, and only the new one is sent.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
from collections.abc import Collection, Mapping, Sequence
from typing import Any

logger = logging.getLogger("Aura.ThinkingElsewhere")

__all__ = ["think_it_through"]

_CHILD: dict[str, Any] = {"process": None, "sent": {}, "stood_on": set(), "gave_up": False}


def _key(held: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(held, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _start() -> Any:
    from core.governance_context import GovernanceViolation
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")
    for threads in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[threads] = "1"
    try:
        return get_subprocess_gateway().spawn(
            [sys.executable, "-m", "core.agency.thinking_elsewhere"],
            cwd=root, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, start_new_session=False, read_only=True,
            source="agency.thinking_elsewhere", accelerator_capability="none",
        )
    except (OSError, ValueError, GovernanceViolation) as why:
        logger.info("thinking stays in this process: %s", why)
        return None


def _the_child() -> Any:
    child = _CHILD["process"]
    if child is not None and child.poll() is None:
        return child
    if _CHILD["gave_up"]:
        return None
    child = _start()
    _CHILD.update(process=child, sent={}, stood_on=set())
    if child is None:
        _CHILD["gave_up"] = True
    return child


def _places(state: Any) -> dict[str, Any]:
    return {
        "rows": int(state.rows),
        "columns": int(state.columns),
        "cells": [[int(cell.row), int(cell.column), str(cell.says)] for cell in state.cells],
    }


def _let_go() -> None:
    child = _CHILD["process"]
    _CHILD.update(process=None, sent={}, stood_on=set())
    if child is not None:
        try:
            child.kill()
        except OSError:
            pass


async def think_it_through(
    knows: Any,
    state: Any,
    actions: Sequence[str],
    *,
    toward: str,
    approach: str,
    budget_s: float,
    world: Any,
    weights: Mapping[str, float] | None,
    no_deeper_than: int = 0,
    been_before: Collection[str] = (),
    depth: int = 0,
) -> tuple[dict[str, tuple[float, str]], int] | None:
    """Her look-ahead, done in the child. The scores and how far it saw, or None to think here instead.

    ``state`` is the board as her rule reads it: the part that behaves like
    one thing, with any furniture already cropped, because the child has her
    rule but not the reading of the screen it was cropped from.
    """
    child = _the_child()
    if child is None:
        return None
    if not weights:
        # What she judges by here, invented measures included: the child's
        # own standing weights do not know what she invented in this process.
        from core.agency.looking_ahead import _default_weights

        weights = dict(_default_weights())
    question: dict[str, Any] = {
        "state": _places(state),
        "actions": [str(action) for action in actions],
        "toward": str(toward or ""),
        "approach": str(approach or ""),
        "budget_s": float(budget_s),
        "no_deeper_than": int(no_deeper_than),
        "depth": int(depth),
        "weights": {str(k): float(v) for k, v in (weights or {}).items()} if weights else None,
    }
    for part, held in (("moves", knows.as_memory()), ("world", world.as_memory() if world is not None else {})):
        signature = _key(held)
        if _CHILD["sent"].get(part) != signature:
            question[part] = held
            _CHILD["sent"][part] = signature
    stood_on = {str(board) for board in been_before}
    if not _CHILD["stood_on"] <= stood_on:
        question["stood_on_from_scratch"] = True
        _CHILD["stood_on"] = set()
    question["stood_on"] = sorted(stood_on - _CHILD["stood_on"])
    _CHILD["stood_on"] |= stood_on

    def ask() -> str:
        child.stdin.write(json.dumps(question) + "\n")
        child.stdin.flush()
        return child.stdout.readline()

    try:
        said = await asyncio.wait_for(asyncio.to_thread(ask), timeout=float(budget_s) * 2.0 + 2.0)
        answer = json.loads(said)
    except (TimeoutError, OSError, ValueError) as why:
        logger.info("the child did not answer in time (%s); thinking here this move", why or "late")
        _let_go()
        return None
    if not answer.get("ok"):
        return None
    scores = {str(act): (float(value), str(why)) for act, (value, why) in (answer.get("scores") or {}).items()}
    return scores, int(answer.get("saw") or 0)


# ── the child ────────────────────────────────────────────────────────────


def _answer_forever() -> int:
    from core.agency.how_good_is_this import AS_GOOD_A_GUESS_AS_ANY, INVENTED, promote
    from core.agency.looking_ahead import how_far_she_can_see, look_ahead
    from core.perception.how_it_moves import HowItMoves
    from core.perception.what_is_there import Arrangement, Cell
    from core.perception.what_the_world_does import WhatTheWorldDoes

    knows: Any = None
    world: Any = None
    stood_on: set[str] = set()
    for line in sys.stdin:
        try:
            question = json.loads(line)
            if "moves" in question:
                knows = HowItMoves.from_memory(question["moves"], 1.0)
            if "world" in question:
                world = WhatTheWorldDoes.from_memory(question["world"], 1.0)
            if question.get("stood_on_from_scratch"):
                stood_on = set()
            stood_on.update(question.get("stood_on") or ())
            weights = question.get("weights")
            if weights:
                from core.agency.inventing_a_measure import measure_named

                for name, worth in weights.items():
                    if name not in INVENTED and name not in AS_GOOD_A_GUESS_AS_ANY:
                        made = measure_named(name)
                        if made is not None:
                            promote(made, worth)
            held = question["state"]
            state = Arrangement(
                rows=int(held["rows"]), columns=int(held["columns"]),
                cells=tuple(Cell(int(r), int(c), str(s), (0.0, 0.0)) for r, c, s in held["cells"]),
                places_seen=True,
            )
            scores = look_ahead(
                knows, state, list(question["actions"]), toward=question["toward"],
                approach=question["approach"], budget_s=float(question["budget_s"]),
                world=world,
                weights=weights, no_deeper_than=int(question["no_deeper_than"]),
                been_before=stood_on, depth=int(question.get("depth") or 0),
            )
            answer = {"ok": True, "scores": {k: list(v) for k, v in scores.items()}, "saw": how_far_she_can_see()}
        except Exception as why:  # noqa: BLE001 - the parent thinks for itself when this fails
            answer = {"ok": False, "why": repr(why)}
        print(json.dumps(answer), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_answer_forever())
