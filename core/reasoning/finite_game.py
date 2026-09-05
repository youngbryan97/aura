"""Solve a finite two-player game exactly, instead of reasoning about it.

LIVE, 2026-08-22. Given the rules of an invented game — two pieces on a row of
nine squares, move 1, 2 or 3 toward the other, whoever cannot move loses — she
answered "move your piece one square on every turn", at high confidence, and
called it a variant of Nim that the first player always wins. The conclusion
was right by luck and the strategy loses: after moving one, the gap is six,
the opponent moves two, and the position is lost.

The game has 9 squares. Every position can be enumerated and scored in under a
millisecond, and the answer is then not an opinion. This is the same division
of labour as the app builder: the language model turns a description into a
typed spec, and the runtime solves it.

Nothing here knows about squares or pieces. A game is variables with bounds,
moves that change them, and a rule about who loses when nobody can move.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "Variable",
    "Move",
    "GameSpec",
    "Solution",
    "solve_game",
    "describe_solution",
]

#: A game bigger than this is not one a chat turn should enumerate.
_MAX_STATES = 2_000_000


@dataclass(frozen=True, slots=True)
class Variable:
    """One number that describes a position."""

    name: str
    initial: int
    low: int
    high: int


@dataclass(frozen=True, slots=True)
class Move:
    """A change to the position, optionally over a range of sizes.

    `deltas` maps a variable to how much one step of this move changes it.
    `steps` is how many sizes the move comes in: (1, 3) means a player may
    take one, two or three. A move is legal when every variable stays inside
    its bounds afterwards.
    """

    name: str
    deltas: dict[str, int] = field(default_factory=dict)
    steps: tuple[int, int] = (1, 1)


@dataclass(frozen=True, slots=True)
class GameSpec:
    """A finite, deterministic, perfect-information game for two players."""

    title: str
    variables: tuple[Variable, ...]
    moves: tuple[Move, ...]
    #: True when a player with no legal move loses. False when they win.
    stuck_loses: bool = True

    def start(self) -> tuple[int, ...]:
        return tuple(item.initial for item in self.variables)

    def bounds(self) -> tuple[tuple[int, int], ...]:
        return tuple((item.low, item.high) for item in self.variables)

    def size(self) -> int:
        total = 1
        for low, high in self.bounds():
            total *= max(0, high - low + 1)
        return total

    def problems(self) -> tuple[str, ...]:
        found: list[str] = []
        names = {item.name for item in self.variables}
        if not self.variables:
            found.append("the game has no position to describe")
        for item in self.variables:
            if not item.low <= item.initial <= item.high:
                found.append(f"{item.name} starts outside its own bounds")
            if item.low > item.high:
                found.append(f"{item.name} has an empty range")
        for move in self.moves:
            if not move.deltas:
                found.append(f"move {move.name} changes nothing")
            for target in move.deltas:
                if target not in names:
                    found.append(f"move {move.name} changes unknown {target}")
            low, high = move.steps
            if low < 1 or high < low:
                found.append(f"move {move.name} has an impossible step range")
        if not self.moves:
            found.append("the game has no moves")
        if self.size() > _MAX_STATES:
            found.append(f"the game has {self.size()} positions, too many to enumerate")
        return tuple(dict.fromkeys(found))


@dataclass(frozen=True, slots=True)
class Solution:
    """Who wins from the opening position, and how."""

    first_player_wins: bool
    winning_moves: tuple[tuple[str, int], ...] = ()
    losing_positions: tuple[tuple[int, ...], ...] = ()
    positions_examined: int = 0
    invariant: str = ""


def _legal_moves(spec: GameSpec, state: tuple[int, ...]) -> list[tuple[str, int, tuple[int, ...]]]:
    """Every (move name, step size, resulting position) available here."""
    index = {item.name: position for position, item in enumerate(spec.variables)}
    bounds = spec.bounds()
    out: list[tuple[str, int, tuple[int, ...]]] = []
    for move in spec.moves:
        low, high = move.steps
        for step in range(low, high + 1):
            after = list(state)
            for name, delta in move.deltas.items():
                after[index[name]] += delta * step
            if all(low_bound <= value <= high_bound
                   for value, (low_bound, high_bound) in zip(after, bounds)):
                out.append((move.name, step, tuple(after)))
    return out


def solve_game(spec: GameSpec) -> Solution | None:
    """Score every reachable position. None when the spec cannot be solved."""
    if spec.problems():
        return None
    seen: dict[tuple[int, ...], bool] = {}

    def wins(state: tuple[int, ...]) -> bool:
        """True when the player to move here wins with perfect play."""
        cached = seen.get(state)
        if cached is not None:
            return cached
        options = _legal_moves(spec, state)
        if not options:
            result = not spec.stuck_loses
        else:
            # Mark before recursing so a position that can reach itself does
            # not spin; a game with a cycle is scored on its acyclic part.
            seen[state] = False
            result = any(not wins(after) for _name, _step, after in options)
        seen[state] = result
        return result

    start = spec.start()
    first = wins(start)
    winning = tuple(
        (name, step)
        for name, step, after in _legal_moves(spec, start)
        if not wins(after)
    )
    losing = tuple(sorted(state for state, value in seen.items() if not value))
    return Solution(
        first_player_wins=first,
        winning_moves=winning,
        losing_positions=losing,
        positions_examined=len(seen),
        invariant=_invariant(spec, seen),
    )


def _invariant(spec: GameSpec, scored: dict[tuple[int, ...], bool]) -> str:
    """A rule covering every losing position, proposed and then checked.

    Subtraction games have a modular answer — lose exactly when the quantity
    that shrinks is a multiple of something. The period is proposed from the
    losing positions and then verified against every position enumerated: a
    winning position sharing the residue refutes it. Two samples are enough to
    propose with, because the proof is the check over the whole space rather
    than the size of the sample.
    """
    losing = sorted(state for state, value in scored.items() if not value)
    if len(losing) < 2 or not spec.variables:
        return ""
    for index, variable in enumerate(spec.variables):
        values = sorted({state[index] for state in losing})
        if len(values) < 2:
            continue
        gaps = {second - first for first, second in zip(values, values[1:])}
        if len(gaps) != 1:
            continue
        period = gaps.pop()
        if period < 2:
            continue
        residue = values[0] % period
        # The check: every position in the whole enumerated space must agree.
        if any(
            (state[index] % period == residue) != (not wins_here)
            for state, wins_here in scored.items()
        ):
            continue
        return (
            f"lose exactly when {variable.name} is a multiple of {period}"
            if residue == 0
            else f"lose exactly when {variable.name} leaves {residue} on division by {period}"
        )
    return ""


def describe_solution(spec: GameSpec, solution: Solution | None) -> str:
    """The answer as a sentence, or "" when there is nothing solved."""
    if solution is None:
        return ""
    who = "The player who moves first wins" if solution.first_player_wins else (
        "The player who moves second wins"
    )
    lines = [f"{who}, with perfect play ({solution.positions_examined} positions checked)."]
    if solution.winning_moves:
        best = solution.winning_moves[0]
        named = (
            f"{best[0]} by {best[1]}" if best[1] != 1 or best[0] != "move" else best[0]
        )
        others = len(solution.winning_moves) - 1
        lines.append(
            f"Winning first move: {named}"
            + (f" (and {others} other{'s' if others != 1 else ''} that also win)." if others else ".")
        )
    elif solution.first_player_wins:
        lines.append("No opening move wins outright.")
    if solution.invariant:
        # Name the positions as well as the pattern. The pattern is stated in
        # whatever the spec called the quantity, and "leaves 1 on division by
        # 4" is unreadable without knowing what the quantity counts; the
        # positions themselves are checkable against the rules as written.
        losing = [
            state[0] for state in solution.losing_positions[:6] if len(state) == 1
        ]
        where = (
            f" — that is {', '.join(str(value) for value in sorted(losing))}"
            if losing
            else ""
        )
        lines.append(
            f"The rule: {solution.invariant}{where}. Leave your opponent there every turn."
        )
    return "\n".join(lines)
