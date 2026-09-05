"""A world with no instructions and no stated goal.

The closest thing here to the question people actually mean: put her
somewhere unfamiliar and see whether she can work out what is happening. No
rule book, no goal, no reward shaping — a state, a set of actions whose
meanings are not given, and a hidden condition that ends it well.

What makes this measurable rather than a demonstration is that the same world
can be played by a policy that explores and by one that does not, and the gap
between them is the finding. Everything about the world is drawn from the
freeze seed, including which action does what, so no prior about action names
helps.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

__all__ = ["AWorldWithNoInstructions", "invent_a_world_with_no_instructions"]


@dataclass
class AWorldWithNoInstructions:
    """A state, some acts, and a condition nobody stated."""

    name: str
    acts: tuple[str, ...]
    #: What each act does to the state, hidden from the player.
    _effects: dict[str, tuple[int, int]] = field(repr=False, default_factory=dict)
    _goal: tuple[int, int] = field(repr=False, default=(0, 0))
    #: Squares that end the run. Nothing announces them and nothing marks
    #: them; the only way to know is to have modelled the acts well enough
    #: not to step on one.
    _traps: frozenset = field(repr=False, default=frozenset())
    _where: tuple[int, int] = (0, 0)
    _size: int = 5
    moves: int = 0
    won: bool = False
    lost: bool = False

    def look(self) -> dict[str, Any]:
        """Everything the player is allowed to see. No goal in it."""

        return {
            "where": self._where,
            "size": self._size,
            "acts": list(self.acts),
            "moves": self.moves,
            "over": self.won or self.lost,
            "won": self.won,
            # A number that moves. Nothing says what it means, what raises
            # it, or what it is for — and without it the goal is not merely
            # unstated but unfindable, which makes the world unsolvable by
            # anybody rather than hard. This is what an unfamiliar interface
            # actually gives you: something changes and you work out what.
            "score": self._score(),
        }

    def _score(self) -> int:
        return 2 * self._size - (
            abs(self._where[0] - self._goal[0]) + abs(self._where[1] - self._goal[1])
        )

    @property
    def best_score(self) -> int:
        """What the number reads when she has arrived. Not told to her."""

        return 2 * self._size

    def do(self, act: str) -> dict[str, Any]:
        """Take an act. The only feedback is the state it leads to."""

        if self.won or self.lost:
            return self.look()
        self.moves += 1
        step = self._effects.get(str(act))
        if step is not None:
            x = min(self._size - 1, max(0, self._where[0] + step[0]))
            y = min(self._size - 1, max(0, self._where[1] + step[1]))
            self._where = (x, y)
        if self._where in self._traps:
            # A place there is no coming back from. Without one, a world small
            # enough to walk at random is a world where finishing proves
            # nothing: the first version was solved by choosing acts with no
            # model at all, five times in six.
            self.lost = True
        elif self._where == self._goal:
            self.won = True
        return self.look()

    def reset(self) -> dict[str, Any]:
        self._where = (0, 0)
        self.moves = 0
        self.won = False
        self.lost = False
        return self.look()

    def is_safe(self, place: tuple[int, int]) -> bool:
        """What a player learns by surviving, not by being told."""

        return place not in self._traps

    @property
    def shortest(self) -> int:
        """The fewest acts that could have won it, for the efficiency term.

        A lower bound: the straight-line distance, ignoring the squares that
        end the run. Nothing here has to route around them to state the
        bound, and using a bound that is too generous would flatter the
        efficiency of everything equally.
        """

        return abs(self._goal[0]) + abs(self._goal[1])


def _can_be_reached(goal: tuple[int, int], traps: frozenset, size: int) -> bool:
    """Whether a path from the start to the goal exists at all."""

    from collections import deque

    start = (0, 0)
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return True
        for step in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            there = (
                min(size - 1, max(0, x + step[0])),
                min(size - 1, max(0, y + step[1])),
            )
            if there in seen or there in traps:
                continue
            seen.add(there)
            queue.append(there)
    return False


def invent_a_world_with_no_instructions(
    seed: int, *, size: int = 12, hazard: float = 0.08, lives: int = 12
) -> AWorldWithNoInstructions:
    """One sealed world. The act names carry no meaning by design.

    Named from a fixed pool of nonsense so that nothing in a language model's
    priors about "up" or "north" does any of the work. What each act does is
    drawn from the seed.

    The size is what makes acting at random fail. Hazard alone does not: on
    an eight-square grid with twelve lives, choosing acts at random reached
    the goal half the time, so the gate could not tell a policy from a walk.
    A wider world is not more confusing, it is bigger, and a random walk
    finds one particular corner of it far less often while a policy that
    knows what its acts do walks there as directly as before.

    ``hazard`` is the share of squares that end a run, and it has to be
    survivable within the lives the gate allows or the world is not hard, it
    is unlearnable: the only way to find a hidden lethal square is to step on
    it, so a world with sixteen per cent of them and eight lives asks her to
    map ten traps by dying in them ten times. Eight per cent leaves a hazard
    that has to be respected and a map that can be built.
    """

    rng = random.Random(seed ^ 0x0A11)
    names = rng.sample(
        ["ka", "mo", "vel", "sith", "orr", "lun", "dax", "pheme"], 4
    )
    steps = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    rng.shuffle(steps)
    goal = (rng.randint(size - 3, size - 1), rng.randint(size - 3, size - 1))
    everywhere = [
        (x, y)
        for x in range(size)
        for y in range(size)
        if (x, y) not in {(0, 0), goal}
    ]
    # Traps that do not wall the goal off. A world nobody can solve measures
    # nothing, and one of the first twelve generated was sealed shut: no path
    # from the start to the goal existed, so every policy scored zero there
    # and the gate counted it against them.
    traps = frozenset()
    for _ in range(64):
        drawn = frozenset(rng.sample(everywhere, max(1, int(len(everywhere) * hazard))))
        if _can_be_reached(goal, drawn, size):
            traps = drawn
            break
    return AWorldWithNoInstructions(
        name=f"a world nobody described ({goal[0]},{goal[1]})",
        acts=tuple(names),
        _effects=dict(zip(names, steps)),
        _goal=goal,
        _traps=traps,
        _size=size,
    )
