"""core/learning/manipulable_learning.py — improving how she improves.

Aura adapts. What she cannot do is change the *kind* of adaptation available
to her, because the learning mechanism is code and the things it learns are
data, and those are different sorts of object. Improving a policy is a search;
improving the thing that improves policies is somebody writing a new file.

The requirement is that "the kind of thing I currently know how to change is
not sufficient" becomes a thought she can act on. The way out is not a tower
of meta-learners — MetaImprover, then MetaMetaImprover — but a representation
general enough that the task, the policy and the learning mechanism are all
manipulable objects of the same sort. Then improving how she improves is
another search in the space she already searches.

The claim is falsifiable and this module exists to make it so. One search
function, applied at two levels, with no level-specific code:

    search(candidates, score)          improves a policy
    search(candidates, score)          improves the mechanism that improves it

If the second needs a different function, the representation was not general
enough and the tower starts. :func:`levels_share_a_search` is the test, and it
compares the function objects rather than trusting the prose.

A :class:`Program` is the shared shape. A policy is a program over actions; a
mechanism is a program over programs. Both are a name, a vocabulary of moves,
and parameters, so the same mutation operators apply to both, which is what
makes the second search possible at all rather than merely nameable.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

logger = logging.getLogger("Aura.Learning.Manipulable")

#: How many candidates a search considers per round. Small: the point is that
#: the same search runs at both levels, not that it is a strong optimiser.
DEFAULT_BREADTH = 8

#: Rounds before a search gives up. Bounded so a meta-level search cannot run
#: the level below it forever.
DEFAULT_ROUNDS = 6

#: How far a parameter moves, as a fraction of itself. Relative because the
#: parameters here span orders of magnitude and nothing knows their scale.
_PARAM_STEP_FRACTION = 0.5

#: The smallest step, for a parameter sitting at or near zero, where a
#: fraction of it would be no step at all.
_MIN_PARAM_STEP = 0.5


@dataclass(frozen=True)
class Program:
    """A policy or a learning mechanism. The same sort of object either way.

    ``moves`` is what it does, drawn from a vocabulary; ``params`` are the
    numbers it does it with. A mechanism's moves operate on programs and a
    policy's operate on the task, and nothing in this class knows which —
    which is the property that lets one mutation operator serve both.
    """

    name: str
    moves: tuple[str, ...]
    params: Mapping[str, float] = field(default_factory=dict)
    #: How far up the tower this sits. 0 acts on the task, 1 acts on level 0.
    level: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "moves": list(self.moves),
            "params": dict(self.params),
            "level": self.level,
        }


@dataclass(frozen=True)
class Result:
    """What a search found, and what it cost."""

    best: Program
    score: float
    rounds: int
    considered: int
    improved: bool
    because: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "best": self.best.to_dict(),
            "score": round(self.score, 5),
            "rounds": self.rounds,
            "considered": self.considered,
            "improved": self.improved,
            "because": self.because,
        }


def mutate(
    program: Program, vocabulary: Sequence[str], rng: random.Random
) -> Program:
    """One neighbour of a program. Level-agnostic on purpose.

    The operators are the same whether the program acts on the task or on
    another program, because both are moves and numbers. A mutation that had
    to know which it was holding would be the first plank of the tower.
    """
    moves = list(program.moves)
    params = dict(program.params)
    choice = rng.random()
    if choice < 0.35 and vocabulary:
        moves.append(rng.choice(list(vocabulary)))
    elif choice < 0.6 and len(moves) > 1:
        del moves[rng.randrange(len(moves))]
    elif choice < 0.8 and moves and vocabulary:
        moves[rng.randrange(len(moves))] = rng.choice(list(vocabulary))
    elif params:
        key = rng.choice(sorted(params))
        current = params[key]
        # The step scales with the parameter rather than being a fixed 0.2.
        # A constant step is wrong in an unknown scale: on a breadth of 2 that
        # rounds to an integer, every mutation landed back on 2 and the
        # meta-level search found nothing while reporting that it had looked.
        # Same shape as an absolute neighbour radius in an unknown feature
        # space, which is the third time this has come up.
        spread = max(_MIN_PARAM_STEP, abs(current) * _PARAM_STEP_FRACTION)
        params[key] = current + rng.gauss(0.0, spread)
    elif vocabulary:
        moves.append(rng.choice(list(vocabulary)))
    return replace(program, moves=tuple(moves), params=params)


def search(
    start: Program,
    score: Callable[[Program], float],
    vocabulary: Sequence[str],
    *,
    breadth: int = DEFAULT_BREADTH,
    rounds: int = DEFAULT_ROUNDS,
    seed: int = 0x5EA4,
) -> Result:
    """Improve a program. The same function at every level.

    Nothing here reads ``program.level``. That is the whole design: if this
    function had to branch on it, improving how she improves would need its
    own implementation and the tower would have started.
    """
    rng = random.Random(seed)
    best, best_score = start, score(start)
    initial = best_score
    considered = 1
    for _ in range(max(1, rounds)):
        improved_this_round = False
        for _ in range(max(1, breadth)):
            candidate = mutate(best, vocabulary, rng)
            considered += 1
            value = score(candidate)
            if value > best_score:
                best, best_score = candidate, value
                improved_this_round = True
        if not improved_this_round:
            break
    return Result(
        best=best,
        score=best_score,
        rounds=rounds,
        considered=considered,
        improved=best_score > initial,
        because=(
            f"{best_score - initial:+.4f} over {considered} candidates"
            if best_score > initial
            else f"no candidate beat the starting {initial:.4f}"
        ),
    )


def levels_share_a_search(
    policy_search: Callable[..., Any], mechanism_search: Callable[..., Any]
) -> bool:
    """Whether improving the improver is the same operation as improving.

    The falsifiable form of the claim. Compares the function objects rather
    than trusting a docstring: a second implementation with the same shape
    would pass a prose check and is exactly the tower this avoids.
    """
    return policy_search is mechanism_search


def as_mechanism(
    program: Program,
    vocabulary: Sequence[str],
    *,
    seed: int = 0x5EA4,
) -> Callable[[Program, Callable[[Program], float]], Result]:
    """Turn a program into the learning mechanism it describes.

    This is the step that makes a mechanism manipulable: its parameters —
    how wide to search, how long, from what seed — are numbers in a Program,
    so mutating them is the same operation as mutating a policy's numbers.
    """
    breadth = max(1, int(round(program.params.get("breadth", DEFAULT_BREADTH))))
    rounds = max(1, int(round(program.params.get("rounds", DEFAULT_ROUNDS))))

    def learn(start: Program, score: Callable[[Program], float]) -> Result:
        return search(
            start, score, vocabulary, breadth=breadth, rounds=rounds, seed=seed
        )

    return learn


def improve_the_improver(
    mechanism: Program,
    starting_policy: Program,
    task_score: Callable[[Program], float],
    vocabulary: Sequence[str],
    *,
    seed: int = 0x5EA4,
    budget: int = 60,
) -> Result:
    """Search over mechanisms, scoring each by how well the policy it finds does.

    A mechanism is scored by what it produces, which is the only thing a
    mechanism is for. The budget is on total task evaluations rather than on
    rounds, because a meta-search that spends unboundedly at the level below
    is not a search, it is an outage.
    """
    spent = {"evaluations": 0}

    def bounded_task_score(policy: Program) -> float:
        if spent["evaluations"] >= budget:
            return float("-inf")
        spent["evaluations"] += 1
        return task_score(policy)

    def mechanism_score(candidate: Program) -> float:
        learn = as_mechanism(candidate, vocabulary, seed=seed)
        found = learn(starting_policy, bounded_task_score)
        return found.score

    # The same function that improved the policy, applied to the mechanism.
    return search(
        mechanism,
        mechanism_score,
        vocabulary=("breadth", "rounds"),
        breadth=4,
        rounds=3,
        seed=seed,
    )


__all__ = [
    "DEFAULT_BREADTH",
    "DEFAULT_ROUNDS",
    "Program",
    "Result",
    "as_mechanism",
    "improve_the_improver",
    "levels_share_a_search",
    "mutate",
    "search",
]
