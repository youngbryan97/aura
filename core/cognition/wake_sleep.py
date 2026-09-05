"""core/cognition/wake_sleep.py — solve, abstract, dream, and a proposer that learns.

Aura has all three phases of this loop and no loop. She solves tasks. She grows
representations. She has dream and consolidation machinery. Nothing runs them
as one cycle where each feeds the next, so the abstractions found in one pass
never change what the next pass tries first.

The cycle, adapted clean-room from the published wake-sleep description:

* **Wake** — solve what you can with the library you have, guided by the
  recogniser.
* **Abstract** — compress the corpus of solutions (that is
  :mod:`core.cognition.library_compression`), which changes the library.
* **Dream** — generate problems FROM the new library, whose solutions are known
  by construction, and train the recogniser on them alongside real replay.

The recogniser is the part that pays
------------------------------------
:class:`Recogniser` predicts which library entries a task will need. Aura's
current proposal prior is close to a no-op: it ranks by marginal frequency,
which is the same ranking for every task. This conditions on task features, so
"a task mentioning a grid" and "a task mentioning a sequence" get different
orderings.

It is trained on dreams and replay together, and the mix is deliberate. Replay
alone overfits to the tasks that happened to come up; dreams alone train on a
distribution the library generated, which is circular. Neither is enough and
the failure modes are opposite.

What is measured
----------------
:meth:`WakeSleep.expansion_report` compares expansions per solved task against
the marginal-frequency prior on the same held-out tasks. Fewer expansions at
equal solve rate is the only claim a learned recogniser is entitled to make,
and it is measured against the thing it replaced rather than against nothing.
"""

from __future__ import annotations

from core.runtime.lockdep import checked_lock
import math
import random
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.cognition.library_compression import Abstraction, LibraryCompressor, size

__all__ = ["Task", "Recogniser", "Dream", "WakeSleep"]


@dataclass(frozen=True, slots=True)
class Task:
    """One problem, its features, and what solving it turned out to need."""

    task_id: str
    features: frozenset[str]
    family: str = ""
    solution: Any = None
    used: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class Dream:
    """A problem generated from the library, whose solution is known."""

    task_id: str
    features: frozenset[str]
    solution: Any
    used: frozenset[str]


class Recogniser:
    """Which library entries a task will need, conditioned on its features.

    A count of (feature, abstraction) co-occurrences, scored by pointwise
    mutual information rather than raw frequency. Frequency alone recovers the
    marginal prior this exists to beat: an abstraction used in every solution
    ranks first for every task, which is exactly the no-op.
    """

    def __init__(self) -> None:
        self._lock = checked_lock("core.cognition.wake_sleep.Recogniser", reentrant=True)
        self._joint: dict[tuple[str, str], int] = {}
        self._feature_counts: dict[str, int] = {}
        self._abstraction_counts: dict[str, int] = {}
        self._observations = 0

    def train(self, features: frozenset[str], used: frozenset[str]) -> None:
        with self._lock:
            self._observations += 1
            for feature in features:
                self._feature_counts[feature] = self._feature_counts.get(feature, 0) + 1
                for abstraction in used:
                    key = (feature, abstraction)
                    self._joint[key] = self._joint.get(key, 0) + 1
            for abstraction in used:
                self._abstraction_counts[abstraction] = (
                    self._abstraction_counts.get(abstraction, 0) + 1
                )

    def rank(self, features: frozenset[str], candidates: Sequence[str]) -> list[str]:
        """Order the library for this task. Untrained falls back to marginal."""
        with self._lock:
            if self._observations == 0:
                return list(candidates)
            total = self._observations
            scored = []
            for abstraction in candidates:
                marginal = self._abstraction_counts.get(abstraction, 0) / total
                score = 0.0
                for feature in features:
                    joint = self._joint.get((feature, abstraction), 0) / total
                    feature_p = self._feature_counts.get(feature, 0) / total
                    if joint > 0 and feature_p > 0 and marginal > 0:
                        score += math.log(joint / (feature_p * marginal))
                scored.append((score, marginal, abstraction))
        return [name for _, _, name in sorted(scored, key=lambda row: (-row[0], -row[1]))]

    def marginal_rank(self, candidates: Sequence[str]) -> list[str]:
        """The prior this replaces: the same order for every task."""
        with self._lock:
            return sorted(candidates, key=lambda a: -self._abstraction_counts.get(a, 0))

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "observations": self._observations,
                "features": len(self._feature_counts),
                "abstractions": len(self._abstraction_counts),
                "pairs": len(self._joint),
            }


class WakeSleep:
    """The cycle, and the measurement that says whether the recogniser earns it."""

    def __init__(self, compressor: LibraryCompressor, *, seed: int = 0) -> None:
        self._lock = checked_lock("core.cognition.wake_sleep.WakeSleep", reentrant=True)
        self._compressor = compressor
        self._recogniser = Recogniser()
        self._rng = random.Random(seed)
        self._solved: list[Task] = []
        self._dreams: list[Dream] = []
        self._cycles = 0

    @property
    def recogniser(self) -> Recogniser:
        return self._recogniser

    def wake(self, tasks: Sequence[Task]) -> list[Task]:
        """Record the tasks that were solved and what each needed."""
        with self._lock:
            solved = [t for t in tasks if t.solution is not None]
            self._solved.extend(solved)
            for task in solved:
                self._compressor.add_solution(task.task_id, task.solution, family=task.family)
            return solved

    def abstract(self) -> list[Abstraction]:
        """Compress the corpus, then attribute each solution to what it now uses.

        The attribution is what closes the loop. A task recorded at wake does
        not know which library entries it will turn out to need, because they
        did not exist yet; reading them off the rewritten solution afterwards
        is how replay becomes training data rather than a list of tasks.
        """
        self._compressor.compress()
        library = {a.name for a in self._compressor.library()}
        rewritten = self._compressor.solutions()
        with self._lock:
            self._solved = [
                Task(
                    task_id=task.task_id, features=task.features, family=task.family,
                    solution=task.solution,
                    used=frozenset(_tokens(rewritten.get(task.task_id, ()))) & library,
                )
                for task in self._solved
            ]
        return self._compressor.library()

    def dream(self, *, count: int = 20) -> list[Dream]:
        """Generate problems from the library whose solutions are known.

        Features are derived from the abstraction's own body, so a dream is a
        problem the library can already express; it teaches the recogniser
        which features go with which entry, and nothing about what to build.
        """
        library = self._compressor.library()
        if not library:
            return []
        dreams: list[Dream] = []
        for i in range(count):
            picked = self._rng.sample(library, k=min(len(library), self._rng.randint(1, 2)))
            features = frozenset(
                token for entry in picked
                for token in _tokens(entry.body)
            )
            dreams.append(
                Dream(
                    task_id=f"dream_{self._cycles}_{i}",
                    features=features,
                    solution=tuple(("call", entry.name) for entry in picked),
                    used=frozenset(entry.name for entry in picked),
                )
            )
        with self._lock:
            self._dreams.extend(dreams)
        return dreams

    def train_recogniser(self, dreams: Sequence[Dream]) -> dict[str, Any]:
        """Train on replay and dreams together.

        Replay alone overfits to what happened to come up; dreams alone train
        on a distribution the library generated, which is circular. The mix is
        the point and both counts are reported so a caller can see it.
        """
        with self._lock:
            replay = list(self._solved)
        for task in replay:
            if task.used:
                self._recogniser.train(task.features, task.used)
        for dream in dreams:
            self._recogniser.train(dream.features, dream.used)
        return {
            "replay": sum(1 for t in replay if t.used),
            "dreams": len(dreams),
            "circular": len(dreams) > 0 and not any(t.used for t in replay),
        }

    def cycle(self, tasks: Sequence[Task], *, dreams: int = 20) -> dict[str, Any]:
        """One full turn: wake, abstract, dream, train."""
        with self._lock:
            self._cycles += 1
            index = self._cycles
        solved = self.wake(tasks)
        library = self.abstract()
        dreamt = self.dream(count=dreams)
        training = self.train_recogniser(dreamt)
        return {
            "cycle": index,
            "solved": len(solved),
            "library": len(library),
            "dreams": len(dreamt),
            "training": training,
            "corpus_size": self._compressor.corpus_size(),
        }

    def expansion_report(
        self, held_out: Sequence[Task], *, budget: int = 5
    ) -> dict[str, Any]:
        """Expansions per solved task, learned ordering against the marginal prior.

        An expansion is one library entry tried. Both arms get the same budget
        and the same library; only the order differs, which is the only thing
        the recogniser changes.
        """
        library = [a.name for a in self._compressor.library()]
        if not library:
            return {"measurable": False, "reason": "the library is empty"}

        def run(order: Callable[[Task], Sequence[str]]) -> tuple[int, int]:
            expansions = solved = 0
            for task in held_out:
                if not task.used:
                    continue
                found = False
                for step, name in enumerate(order(task)[:budget], start=1):
                    expansions += 1
                    if name in task.used:
                        found = True
                        break
                solved += 1 if found else 0
            return expansions, solved

        learned = run(lambda t: self._recogniser.rank(t.features, library))
        marginal = run(lambda t: self._recogniser.marginal_rank(library))
        return {
            "measurable": True,
            "learned_expansions": learned[0],
            "learned_solved": learned[1],
            "marginal_expansions": marginal[0],
            "marginal_solved": marginal[1],
            "fewer_expansions": learned[0] < marginal[0],
            "no_solve_regression": learned[1] >= marginal[1],
            "earns_its_place": learned[0] < marginal[0] and learned[1] >= marginal[1],
        }


def _tokens(expression: Any) -> list[str]:
    if not isinstance(expression, tuple):
        return [str(expression)]
    out: list[str] = []
    for part in expression:
        out.extend(_tokens(part))
    return out
