"""How she looks, set by trying it in the world she is in.

Her compiled search's habits (how far a late move's look is shortened, how wide a margin lets a line go unlooked, how
far a passed turn's look is cut) were set by hand and measured in one world. A world of other pieces on another board
may want them elsewhere. Here she moves them herself. Every habit a look of a fixed length can feel is moved at once,
each by a share of itself in a direction drawn at random, and the habits moved one way play the same habits moved the
other way, each opening from both sides. Each pair of games says which way did better, and every habit moves toward its
better side by as much as the pair says. This is simultaneous perturbation stochastic approximation (Spall 1992): one
pair of games informs every habit at once, where moving one at a time would need a match for each.

A habit is moved in shares of itself, so it stays above nought and a small one moves as far, for its size, as a large
one. Which habits a fixed look can feel is measured, not listed: each is set a share higher and a share lower on
positions of the world, and a habit that changes no look's answer, nodes or score there is left where it is. A habit
that only a clock or a second thread can feel is tried only where those are.

A run ends after the pairs it was given. What it found then plays the habits she has, pair by pair, and is kept only
where the sequential test (core/cognition/whether_one_way_is_better.py) finds it better by what a test of that length
can tell. Nothing here knows a game.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, replace
from typing import Any

__all__ = ["Trying", "felt_habits", "tried_against_hers"]

#: The share of itself a habit is moved by to try each side of it: small enough that both sides are habits near hers,
#: large enough that a pair of games can feel the difference.
SHARE = math.log(1.1)
#: Spall's exponents for how the step and the share shrink as pairs come in (Spall 1998, "Implementation of the
#: simultaneous perturbation algorithm for stochastic optimization"): the slowest that still settle.
STEP_FALLS = 0.602
SHARE_FALLS = 0.101


def _engine() -> Any | None:
    try:
        import aura_board_search

        return aura_board_search
    # not a failure: without the compiled search there is no look to set.
    except ImportError:
        return None


def habits_as_set() -> dict[str, float]:
    engine = _engine()
    if engine is None or not hasattr(engine, "habits_as_set"):
        return {}
    return {str(k): float(v) for k, v in json.loads(engine.habits_as_set()).items()}


def felt_habits(family: Any, positions: list[Any], weights: Any, nodes: int) -> list[str]:
    """The habits above nought that change some look's answer, nodes or score on ``positions``, set a share higher or a
    share lower than they stand, at ``nodes`` a look."""
    from core.agency.searching_by_kind import described, placed

    engine = _engine()
    if engine is None or not positions:
        return []
    description = described(family)
    standing = {**habits_as_set(), **dict(weights.habits or {})}

    def looks(habits: dict[str, float]) -> list[tuple]:
        w = replace(weights, habits=habits).as_json()
        found = []
        for position in positions:
            r = engine.search(description, placed(position), w, 60.0, nodes=nodes)
            found.append((tuple(r["move"]), int(r["score"]), int(r["nodes"])))
        return found

    plain = looks(standing)
    felt = []
    for name, value in sorted(standing.items()):
        if value <= 0:
            continue
        for share in (math.exp(SHARE), math.exp(-SHARE)):
            if looks({**standing, name: value * share}) != plain:
                felt.append(name)
                break
    return felt


@dataclass
class Trying:
    """A run of moving her habits: which, where each stands (as the log of its value), and the pairs played."""

    names: list[str]
    at: dict[str, float]
    pairs_given: int
    played: int = 0
    seed: int = 0
    scores: list[float] = field(default_factory=list)
    #: Once the pairs given are played: the habits found, and the pairs they have played against hers since.
    found: dict[str, float] | None = None
    tested: list[float] = field(default_factory=list)

    @classmethod
    def begun(cls, names: list[str], standing: dict[str, float], pairs_given: int, seed: int = 0) -> Trying:
        return cls(names=list(names), at={n: math.log(standing[n]) for n in names}, pairs_given=pairs_given, seed=seed)

    def habits(self) -> dict[str, float]:
        return {name: math.exp(u) for name, u in self.at.items()}

    def _gains(self) -> tuple[float, float]:
        k = self.played
        share = SHARE / (k + 1) ** SHARE_FALLS
        # The step's first pair can move a habit by at most the share it was tried by.
        held_back = self.pairs_given / 10
        step = 2 * SHARE ** 2 * (held_back + 1) ** STEP_FALLS / (k + 1 + held_back) ** STEP_FALLS
        return step, share

    def one_pair(self, family: Any, weights: Any, nodes: int) -> float:
        """One opening played from both sides by her habits moved one way against the same moved the other; every
        habit moved toward the better side. The pair's score for the side moved up."""
        rng = random.Random(self.seed * 1_000_003 + self.played)
        step, share = self._gains()
        sign = {name: rng.choice((-1.0, 1.0)) for name in self.names}
        up = {name: math.exp(u + share * sign[name]) for name, u in self.at.items()}
        down = {name: math.exp(u - share * sign[name]) for name, u in self.at.items()}
        score = paired(family, weights, up, down, nodes, seed=rng.randrange(1 << 30))
        for name in self.names:
            self.at[name] += step * (2 * score - 1) / (2 * share * sign[name])
        self.played += 1
        self.scores.append(score)
        return score

    def done(self) -> bool:
        return self.played >= self.pairs_given

    def a_round(self, family: Any, weights: Any, nodes: int, seconds: float) -> str | None:
        """Pairs played for about ``seconds``: the habits moved while the pairs given last, then what they found
        played against hers. "better" or "no better" once the sequential test decides (undecided after as many pairs
        as the run was given is no better); None while it goes on."""
        import statistics
        import time

        from core.cognition.whether_one_way_is_better import Comparison

        began = time.monotonic()
        while time.monotonic() - began < seconds:
            if not self.done():
                self.one_pair(family, weights, nodes)
                continue
            if self.found is None:
                self.found = self.habits()
            # The margin is what a test of as many pairs as the run had can tell: two standard errors of pairs that
            # spread as the run's did.
            spread = statistics.pstdev(self.scores) if len(self.scores) > 1 else 0.25
            test = Comparison(better=0.5 + 2 * (spread or 0.25) / math.sqrt(max(1, self.pairs_given)),
                              scores=list(self.tested))
            hers = {**habits_as_set(), **dict(weights.habits or {})}
            rng = random.Random(self.seed * 7_919 + len(self.tested))
            said = test.add(paired(family, weights, self.found, {n: hers[n] for n in self.found}, nodes,
                                   seed=rng.randrange(1 << 30)))
            self.tested = test.scores
            if said:
                return said
            if len(self.tested) >= self.pairs_given:
                return "no better"
        return None

    def as_dict(self) -> dict[str, Any]:
        return {"names": self.names, "at": self.at, "pairs_given": self.pairs_given, "played": self.played,
                "seed": self.seed, "scores": self.scores, "found": self.found, "tested": self.tested}

    @classmethod
    def from_dict(cls, held: dict[str, Any]) -> Trying:
        return cls(names=[str(n) for n in held["names"]], at={str(k): float(v) for k, v in held["at"].items()},
                   pairs_given=int(held["pairs_given"]), played=int(held.get("played", 0)),
                   seed=int(held.get("seed", 0)), scores=[float(x) for x in held.get("scores", [])],
                   found=({str(k): float(v) for k, v in held["found"].items()} if held.get("found") else None),
                   tested=[float(x) for x in held.get("tested", [])])


def paired(family: Any, weights: Any, these: dict[str, float], those: dict[str, float], nodes: int, *,
           seed: int) -> float:
    """One opening played twice, ``these`` habits on each side of it once against ``those``: their points over the
    two games, as a half of each."""
    from core.agency.searching_by_kind import described, placed
    from core.agency.what_a_kind_is_worth import GAME_PLIES_AT_MOST, RANDOM_PLIES, _match_scored

    engine = _engine()
    held = dict(weights.habits or {})
    a = replace(weights, habits={**held, **these})
    b = replace(weights, habits={**held, **those})
    description, begin = described(family), placed(family.the_start())
    first = engine.self_play(description, begin, a.as_json(), b.as_json(), 1, nodes, random_plies=RANDOM_PLIES,
                             seed=seed, max_plies=GAME_PLIES_AT_MOST, nodes_per_move_b=nodes)
    second = engine.self_play(description, begin, b.as_json(), a.as_json(), 1, nodes, random_plies=RANDOM_PLIES,
                              seed=seed, max_plies=GAME_PLIES_AT_MOST, nodes_per_move_b=nodes)
    mine, _ = _match_scored(first, a, family)
    theirs, _ = _match_scored(second, b, family)
    return float((mine[0] + 1.0 - theirs[0]) / 2.0)


def tried_against_hers(family: Any, weights: Any, found: dict[str, float], nodes: int, *, pairs_at_most: int,
                       spread: float, seed: int = 0) -> tuple[str, list[float]]:
    """``found`` habits played against the ones she has until the sequential test decides, or ``pairs_at_most`` pairs
    are played. The margin it tests for is what a test of that length can tell: two standard errors of pairs that
    spread as ``spread`` says. Undecided at the end is no better."""
    from core.cognition.whether_one_way_is_better import Comparison

    margin = 2 * spread / math.sqrt(max(1, pairs_at_most))
    test = Comparison(better=0.5 + margin)
    hers = {**habits_as_set(), **dict(weights.habits or {})}
    rng = random.Random(seed)
    for _ in range(pairs_at_most):
        said = test.add(paired(family, weights, found, {n: hers[n] for n in found}, nodes, seed=rng.randrange(1 << 30)))
        if said:
            return said, test.scores
    return "no better", test.scores


#: Pairs a run is given: about an hour of her practice at her longer look in her model of chess, and a test of as many
#: pairs tells a gain of about one game in forty.
PAIRS_GIVEN = 400


def _positions_of(family: Any, seed: int, count: int = 12) -> list[Any]:
    """Positions of the world, each reached by moves drawn at random from its start: where a habit is felt or not."""
    from core.agency.pieces_by_their_kind import after, moves

    rng = random.Random(seed)
    found: list[Any] = []
    for _ in range(count * 4):
        if len(found) >= count:
            break
        position = family.the_start()
        for _ply in range(rng.randint(6, 40)):
            legal = moves(family, position)
            if not legal:
                break
            position = after(family, position, rng.choice(legal))
        if moves(family, position):
            found.append(position)
    return found


def wants_a_round(rehearsal: Any) -> bool:
    """Whether her practice's next round is one of trying how she looks: rounds of it and of rehearsal take turns."""
    return _engine() is not None and hasattr(_engine(), "habits_as_set") and \
        int(getattr(rehearsal, "looks_tried_after", -1)) < int(getattr(rehearsal, "rounds", 0))


def a_round_of_trying(rehearsal: Any, family: Any, seconds: float) -> dict[str, Any]:
    """One round of setting how she looks: a run begun where none is under way, its pairs played for about
    ``seconds``, and what it found kept in the weights she plays by where the test finds it better."""
    from core.agency.what_a_kind_is_worth import LONGER_LOOK_NODES

    weights = rehearsal.weights
    rehearsal.looks_tried_after = int(rehearsal.rounds)
    run = Trying.from_dict(rehearsal.trying_looks) if rehearsal.trying_looks else None
    if run is None:
        felt = felt_habits(family, _positions_of(family, int(rehearsal.rounds)), weights, LONGER_LOOK_NODES)
        if not felt:
            return {"tried": "how she looks", "felt": 0, "kept": False}
        standing = {**habits_as_set(), **dict(weights.habits or {})}
        run = Trying.begun(felt, standing, PAIRS_GIVEN, seed=int(rehearsal.rounds))
    said = run.a_round(family, weights, LONGER_LOOK_NODES, seconds)
    last: dict[str, Any] = {"tried": "how she looks", "felt": len(run.names), "moving": run.played,
                            "tested": len(run.tested), "kept": said == "better"}
    if run.tested:
        last["scored"] = round(sum(run.tested) / len(run.tested), 3)
    if said == "better" and run.found:
        from core.agency.where_her_results_vary import open_a_trial

        open_a_trial(rehearsal, weights)
        rehearsal.weights = replace(weights, habits={**dict(weights.habits or {}), **run.found})
        last["habits"] = {name: round(value, 3) for name, value in run.found.items()}
    rehearsal.trying_looks = None if said else run.as_dict()
    return last
