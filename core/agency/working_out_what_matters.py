"""What each thing about a situation is worth in this world, found by playing it out in her own model.

Her search judges a situation by several things at once: how near it is to
what she was asked for, how much room is left, whether things run in order,
how close neighbours are in value, whether she can still act. What each of
those is worth used to be a number somebody wrote down, and the numbers were
measured on 2048: smoothness at 0.4 because six games at each of four weights
said so, room and order at 0.15 by choice. They were right there and nowhere
else was ever asked.

The right weighting is a fact about the world, and she can find it out the way
anyone learns what matters in a game they have just been shown: by trying.
Where she has compiled a world — her own rule for what her acts do, her own
record of what turns up between them — she can play whole games in her head
with one weighting and then with another, from the same start and the same
dice, and keep whichever got further.

`what_makes_it_good_here` tried to learn the same thing from single moves, and
could not, because the terms describe a position and a single move is graded
by what it merged. A whole game is graded by where it ended, which is what the
terms are for. That is the grade this uses.

It starts from every term alike, because before she has played a world she
has no reason to think one thing about it matters more than another. Then it
tries one term at a time: without it, at half, at double. Weights only matter
against each other, so every change is a ratio, and halving and doubling are
the smallest steps that treat up and down alike. Without it asks whether the
term matters here at all.

A change is kept only when the games say so. The same starts are played both
ways, pair by pair, and a change is taken when the difference between the pairs
clears its own standard error by the ordinary line between a difference and
noise, raised for how many changes are being tried so that trying many does
not make one of them look good by luck. It is dropped when the difference
clears the same line the other way, and left alone when the pairs run out
first. Where the games cannot
tell, the weighting she already has stands, and that is the honest outcome: a
world where nothing she changes makes a difference is one where the weights
do not matter.
"""

from __future__ import annotations

import logging
import math
import time as _clock
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.WorkingOutWhatMatters")

__all__ = ["WorkedOut", "every_term_alike", "work_out_what_matters"]

#: What she tries each term at, against what it is now. See the module
#: docstring for why these three.
TRIED_AT = (0.0, 0.5, 2.0)

#: How often she may take a change that does nothing, across a whole round of
#: changes. The ordinary line. Split between every change a round tries and
#: every time the pairs are looked at, so that trying fifteen changes, or
#: looking twice, does not make one of them look good by luck.
FALSE_ALARM = 0.05

#: When the pairs are looked at. Once early, so a change that alters nothing
#: or alters everything is settled in a few games, and once at the end.
#: Four is the fewest a spread can be measured from at all: two pairs that
#: agree have a spread of nought and would clear any line. Past eight, a
#: difference too small to show has cost more games than it could be worth,
#: and the weighting she already has stands.
LOOKS_AT = (4, 8)
FEWEST_PAIRS, MOST_PAIRS = LOOKS_AT[0], LOOKS_AT[-1]


def every_term_alike(names: Sequence[str]) -> dict[str, float]:
    """Every term worth the same: what she knows about a world before playing it."""
    return {str(name): 1.0 for name in names}


@dataclass
class WorkedOut:
    """What she settled on, and what she tried on the way."""

    weights: dict[str, float]
    #: Each change she tried and what the pairs said: (term, from, to, verdict, pairs).
    tried: list[tuple[str, float, float, str, int]] = field(default_factory=list)
    games: int = 0
    took_s: float = 0.0
    #: Why it stopped, for saying so.
    stopped: str = ""

    def changed(self) -> list[tuple[str, float, float]]:
        return [(term, was, now) for term, was, now, verdict, _n in self.tried if verdict == "kept"]

    def says(self) -> str:
        weights = ", ".join(f"{name} {value:g}" for name, value in sorted(self.weights.items()))
        kept = self.changed()
        if not kept:
            return (
                f"played out {self.games} game(s) in my own model and nothing I changed "
                f"made a difference I could tell from luck; judging by {weights}"
            )
        moved = "; ".join(f"{term} {was:g} to {now:g}" for term, was, now in kept)
        return f"played out {self.games} game(s) in my own model: {moved}. Judging by {weights}"


def _how_far_it_got(played: Any, toward: str) -> float:
    """One game's outcome on one scale: how near it got, and how soon if it arrived.

    Nearness is her own measure of the goal. A game that arrived is worth
    one, plus a share that shrinks with its length, so between two that both
    arrive the quicker counts for more. Where the goal names nothing she can
    measure, the largest thing the game made is all there is to go on.
    """
    from core.agency.what_she_is_after import goal_in  # noqa: PLC0415

    if getattr(played, "reached", False):
        moves = max(1, int(getattr(played, "moves", 1) or 1))
        return 1.0 + 1.0 / moves
    if goal_in(toward).names_something():
        return float(getattr(played, "nearest", 0.0) or 0.0)
    furthest = float(getattr(played, "furthest", 0.0) or 0.0)
    return math.log2(furthest) if furthest > 1.0 else 0.0


def _terms_that_can_matter(
    knows: Any, start: Any, actions: Sequence[str], toward: str, names: Sequence[str]
) -> list[str]:
    """The terms whose reading differs somewhere near the start.

    A term that reads the same in every situation adds the same to every
    score and cannot change a choice, so trying it at other weights would
    spend games on nothing. Her line reads nought where she holds none, and
    newness reads nought in anything that cannot say it has been visited.
    """
    from core.agency.how_good_is_this import terms  # noqa: PLC0415

    expect = getattr(knows, "expect", None)
    seen: list[dict[str, float]] = []
    around = [start]
    if callable(expect):
        for action in actions:
            try:
                after = expect(start, action)
            except (AttributeError, TypeError, ValueError):
                continue
            if after is not None:
                around.append(after)
    for state in around:
        try:
            seen.append(terms(state, toward=toward, knows=knows, acts=actions))
        except (AttributeError, TypeError, ValueError):
            continue
    return [
        name for name in names
        if len({round(float(reading.get(name, 0.0)), 9) for reading in seen}) > 1
        or any(float(reading.get(name, 0.0)) for reading in seen)
    ]


class _Baseline:
    """Games played with the weighting she has now, by seed, so each is played once.

    Every change is compared on the same starts and the same dice, so the
    games judged the current way are the same games whichever change they are
    set against. Kept until the weighting changes.
    """

    def __init__(self) -> None:
        self.played: dict[int, Any] = {}

    def forget(self) -> None:
        self.played.clear()


def _line_for(changes: int, pairs: int) -> float:
    """How far clear of its standard error a difference over ``pairs`` has to be.

    Student's t, because a spread measured from a handful of pairs is itself
    uncertain and the normal curve pretends it is not: four pairs of pure
    noise cleared a normal line and a change was kept (2026-09-25, in the
    test that now guards it).
    """
    share = FALSE_ALARM / (2.0 * max(1, int(changes)) * len(LOOKS_AT))
    try:
        from scipy.stats import t as student  # noqa: PLC0415

        return float(student.ppf(1.0 - share, max(1, int(pairs) - 1)))
    except ImportError:
        from statistics import NormalDist  # noqa: PLC0415

        return NormalDist().inv_cdf(1.0 - share) * 2.0


def _paired(
    knows: Any, world: Any, start: Any, actions: Sequence[str], *,
    now: Mapping[str, float], trying: Mapping[str, float], toward: str,
    baseline: _Baseline, changes: int, until: float, looks_ahead: int,
) -> tuple[str, int, int]:
    """Play pairs until they tell the two apart or run out. (verdict, pairs, games)."""
    from core.agency.rehearsing_in_her_model import played_out  # noqa: PLC0415

    differences: list[float] = []
    games = 0
    for seed in range(MOST_PAIRS):
        how = dict(toward=toward, seed=seed, until=until, looks_ahead=looks_ahead)
        before = baseline.played.get(seed)
        if before is None:
            before = played_out(knows, world, start, actions, weights=dict(now), **how)
            games += 1
            if before is not None:
                baseline.played[seed] = before
        after = played_out(knows, world, start, actions, weights=dict(trying), **how)
        games += 1
        if before is None or after is None:
            return "out of time", len(differences), games
        differences.append(_how_far_it_got(after, toward) - _how_far_it_got(before, toward))
        if len(differences) not in LOOKS_AT:
            continue
        pairs = len(differences)
        last_look = pairs == MOST_PAIRS
        mean = sum(differences) / pairs
        spread = math.sqrt(sum((d - mean) ** 2 for d in differences) / (pairs - 1))
        if spread == 0.0:
            # Every pair came out the same. All nought means the change did
            # nothing that shows. Any other constant is as sure as a
            # difference gets — but only over every pair, because a few
            # alike can be luck where outcomes come in steps.
            if mean == 0.0:
                return "no difference", pairs, games
            if last_look:
                return ("kept" if mean > 0 else "worse"), pairs, games
            continue
        line = _line_for(changes, pairs) * spread / math.sqrt(pairs)
        if mean > line:
            return "kept", pairs, games
        if mean < -line:
            return "worse", pairs, games
    return "could not tell", len(differences), games


def work_out_what_matters(
    knows: Any,
    world: Any,
    start: Any,
    actions: Sequence[str],
    *,
    weights: Mapping[str, float] | None = None,
    toward: str = "",
    within_s: float = 120.0,
    looks_ahead: int = 0,
) -> WorkedOut | None:
    """What each term is worth in this world, by playing it out in her own model.

    ``weights`` is where she starts, every term alike when nothing is given.
    ``within_s`` bounds the whole of it, and what was settled before the clock
    ran out is kept. None where there is no model to play in.

    ``looks_ahead`` is how deep her search looks while rehearsing; nought
    means as deep as a rehearsal usually does.
    """
    from core.agency.a_world_compiled import compiled  # noqa: PLC0415
    from core.agency.how_good_is_this import AS_GOOD_A_GUESS_AS_ANY  # noqa: PLC0415
    from core.agency.rehearsing_in_her_model import LOOKS_AHEAD  # noqa: PLC0415

    if compiled(knows, world, start, actions) is None:
        return None
    began = _clock.monotonic()
    until = began + max(0.0, float(within_s)) if within_s else 0.0
    depth = int(looks_ahead) or LOOKS_AHEAD
    now = dict(weights) if weights else every_term_alike(list(AS_GOOD_A_GUESS_AS_ANY))
    found = WorkedOut(weights=now)
    worth_trying = _terms_that_can_matter(knows, start, actions, toward, list(now))
    changes = len(worth_trying) * len(TRIED_AT)
    baseline = _Baseline()
    # Round after round until a whole round changes nothing: a term that
    # did not matter at one weighting of the others can matter at another.
    changed_in_a_round = True
    while changed_in_a_round:
        changed_in_a_round = False
        for name in worth_trying:
            was = float(now.get(name, 0.0))
            for ratio in TRIED_AT:
                if until and _clock.monotonic() > until:
                    found.stopped = "out of time"
                    found.took_s = _clock.monotonic() - began
                    logger.info("%s (stopped: out of time)", found.says())
                    return found
                to = was * ratio if was else (1.0 if ratio else 0.0)
                if to == was:
                    continue
                trying = {**now, name: to}
                verdict, pairs, games = _paired(
                    knows, world, start, actions, now=now, trying=trying, toward=toward,
                    baseline=baseline, changes=changes, until=until, looks_ahead=depth,
                )
                found.games += games
                found.tried.append((name, was, to, verdict, pairs))
                logger.info("%s at %g instead of %g: %s over %d pair(s)", name, to, was, verdict, pairs)
                if verdict == "out of time":
                    found.stopped = "out of time"
                    found.took_s = _clock.monotonic() - began
                    logger.info("%s (stopped: out of time)", found.says())
                    return found
                if verdict == "kept":
                    now[name] = to
                    baseline.forget()
                    changed_in_a_round = True
                    break
    found.stopped = "a whole round changed nothing"
    found.took_s = _clock.monotonic() - began
    logger.info("%s", found.says())
    return found


# ── in a process of its own ──────────────────────────────────────────────
#
# Playing a world out is whole games of her own search, and her search is
# Python holding the interpreter. Beside a live run, in a thread, it would take
# the interpreter from the search she is playing with — the same reason her
# eyes read in a process of their own. So the question goes to a child, as
# what she knows rather than as objects: her rule and her record of what
# arrives, in the form they are remembered in, and the board as places.


def _as_places(state: Any) -> dict[str, Any]:
    return {
        "rows": int(state.rows),
        "columns": int(state.columns),
        "cells": [[int(cell.row), int(cell.column), str(cell.says)] for cell in state.cells],
    }


def _from_places(held: Mapping[str, Any]) -> Any:
    from core.perception.what_is_there import Arrangement, Cell  # noqa: PLC0415

    return Arrangement(
        rows=int(held["rows"]),
        columns=int(held["columns"]),
        cells=tuple(
            Cell(row=int(row), column=int(column), says=str(said), at=(0.0, 0.0))
            for row, column, said in held.get("cells") or ()
        ),
        places_seen=True,
    )


def the_question(
    knows: Any, world: Any, start: Any, actions: Sequence[str], *,
    weights: Mapping[str, float] | None, toward: str, within_s: float,
) -> dict[str, Any]:
    """What a child needs to play this world out: everything as she remembers it."""
    cropped = knows.the_thing(start) if callable(getattr(knows, "the_thing", None)) else start
    return {
        "moves": knows.as_memory(),
        "world": world.as_memory(),
        "start": _as_places(cropped),
        "actions": [str(action) for action in actions],
        "weights": {str(name): float(value) for name, value in (weights or {}).items()},
        "toward": str(toward or ""),
        "within_s": float(within_s),
    }


def answer(question: Mapping[str, Any]) -> dict[str, Any]:
    """Play the world in ``question`` out. What the child says back."""
    from core.agency.how_good_is_this import (  # noqa: PLC0415
        AS_GOOD_A_GUESS_AS_ANY,
        INVENTED,
        promote,
    )
    from core.perception.how_it_moves import HowItMoves  # noqa: PLC0415
    from core.perception.what_the_world_does import WhatTheWorldDoes  # noqa: PLC0415

    knows = HowItMoves.from_memory(dict(question.get("moves") or {}), 1.0)
    world = WhatTheWorldDoes.from_memory(dict(question.get("world") or {}), 1.0)
    weights = {str(k): float(v) for k, v in (question.get("weights") or {}).items()}
    # The properties she invented, made again from their names, so the child
    # judges by everything she does.
    try:
        from core.agency.inventing_a_measure import measure_named  # noqa: PLC0415

        for name, worth in weights.items():
            if name not in INVENTED and name not in AS_GOOD_A_GUESS_AS_ANY:
                made = measure_named(name)
                if made is not None:
                    promote(made, worth)
    except ImportError:
        pass
    found = work_out_what_matters(
        knows, world, _from_places(question["start"]), list(question.get("actions") or ()),
        weights=weights or None, toward=str(question.get("toward") or ""),
        within_s=float(question.get("within_s") or 0.0),
    )
    if found is None:
        return {"ok": False, "why": "her model of this world does not compile from there"}
    return {
        "ok": True,
        "weights": found.weights,
        "said": found.says(),
        "games": found.games,
        "took_s": found.took_s,
        "stopped": found.stopped,
        "tried": [list(one) for one in found.tried],
    }


def _a_child() -> Any:
    """A process to play the world out in, or None when one cannot be had."""
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    from core.governance_context import GovernanceViolation  # noqa: PLC0415
    from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: PLC0415

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")
    for threads in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[threads] = "1"
    try:
        return get_subprocess_gateway().spawn(
            [sys.executable, "-m", "core.agency.working_out_what_matters"],
            cwd=root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=False,
            read_only=True,
            source="agency.working_out_what_matters",
            accelerator_capability="none",
            preexec_fn=lambda: os.nice(10),
        )
    except (OSError, ValueError, GovernanceViolation) as why:
        logger.info("could not play this world out in a process of its own: %s", why)
        return None


async def in_a_process_of_its_own(
    knows: Any, world: Any, start: Any, actions: Sequence[str], *,
    weights: Mapping[str, float] | None = None, toward: str = "", within_s: float = 120.0,
) -> dict[str, Any] | None:
    """Ask a child to play this world out, and wait for it. None when no answer came.

    The child goes when this does. Cancelled — because the run that asked
    ended, or the process is shutting down — it is killed rather than left to
    play on to its own clock with nobody waiting for the answer.
    """
    import asyncio  # noqa: PLC0415
    import json  # noqa: PLC0415

    question = json.dumps(
        the_question(knows, world, start, actions, weights=weights, toward=toward, within_s=within_s)
    )
    child = _a_child()
    if child is None:
        return None

    def ask() -> str:
        child.stdin.write(question)
        child.stdin.close()
        return child.stdout.read()

    try:
        # The child's own clock stops the games; this one only stops waiting
        # on a child that has stopped answering.
        said = await asyncio.wait_for(
            asyncio.to_thread(ask), timeout=float(within_s) * 1.5 + 60.0
        )
    except (TimeoutError, OSError, ValueError) as why:
        logger.info("no answer from playing this world out: %s", why or "past its own clock")
        return None
    finally:
        if child.poll() is None:
            child.kill()
        try:
            child.wait(timeout=5.0)
        # not a failure: a child that will not be reaped is the OS's to finish.
        except Exception:  # noqa: BLE001
            pass
    try:
        return json.loads(said.strip().splitlines()[-1]) if said and said.strip() else None
    except ValueError:
        return None


def _main() -> int:
    import json  # noqa: PLC0415
    import sys  # noqa: PLC0415

    question = json.loads(sys.stdin.read() or "{}")
    print(json.dumps(answer(question)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
