"""A finite game is enumerated, not reasoned about.

LIVE, 2026-08-22. Given the rules of an invented game — two pieces on a row of
nine squares, move 1, 2 or 3 toward the other, whoever cannot move loses — she
answered "move your piece one square on every turn", at high confidence, and
called it a variant of Nim that the first player always wins. The conclusion
was right by luck and the strategy loses: after moving one the gap is six, the
opponent moves two, and the position is lost.

There are eight positions. Enumerating them is not an opinion.
"""

from __future__ import annotations

from functools import lru_cache

from core.reasoning.finite_game import (
    GameSpec,
    Move,
    Variable,
    describe_solution,
    solve_game,
)


def row_game(squares: int, reach: int) -> GameSpec:
    return GameSpec(
        title=f"row of {squares}",
        variables=(Variable("the gap", initial=squares - 2, low=0, high=squares - 2),),
        moves=(Move("move toward", {"the gap": -1}, steps=(1, reach)),),
    )


@lru_cache(None)
def truth(gap: int, reach: int) -> bool:
    """Independent brute force: the player to move wins from this gap."""
    if gap == 0:
        return False
    return any(not truth(gap - step, reach) for step in range(1, reach + 1) if step <= gap)


def test_the_invented_game_is_solved_exactly():
    spec = row_game(9, 3)
    solution = solve_game(spec)
    assert solution is not None
    assert solution.first_player_wins is truth(7, 3)
    assert ("move toward", 3) in solution.winning_moves
    # The answer she gave: moving one square loses.
    assert ("move toward", 1) not in solution.winning_moves


def test_it_agrees_with_brute_force_across_many_shapes():
    for squares in range(3, 24):
        for reach in range(1, 6):
            spec = row_game(squares, reach)
            solution = solve_game(spec)
            assert solution is not None, (squares, reach)
            assert solution.first_player_wins is truth(squares - 2, reach), (squares, reach)


def test_the_general_rule_is_derived_and_holds():
    """Not assumed to be modular: proposed from the losing positions, then
    checked against every position enumerated."""
    for reach in (2, 3, 4):
        spec = row_game(20, reach)
        solution = solve_game(spec)
        assert solution is not None
        assert f"multiple of {reach + 1}" in solution.invariant, solution.invariant


def test_a_rule_is_not_reported_when_the_positions_do_not_line_up():
    """Silence beats a pattern that only fits the sample."""
    spec = GameSpec(
        title="two counters",
        variables=(Variable("a", 3, 0, 3), Variable("b", 3, 0, 3)),
        moves=(Move("take a", {"a": -1}, steps=(1, 2)), Move("take b", {"b": -1}, steps=(1, 3))),
    )
    solution = solve_game(spec)
    assert solution is not None
    if solution.invariant:
        period = int(solution.invariant.rsplit(" ", 1)[-1])
        for state, _ in zip(solution.losing_positions, range(50)):
            assert any(value % period == 0 for value in state)


def test_stuck_can_mean_winning_instead():
    """The misere rule is a flag, not a different solver.

    It shifts which positions are lost — 0 and 4 become 1 and 5 — rather than
    flipping every verdict, so the losing set is what this checks.
    """
    normal = solve_game(row_game(9, 3))
    misere = solve_game(
        GameSpec(
            title="misere",
            variables=(Variable("the gap", 7, 0, 7),),
            moves=(Move("move toward", {"the gap": -1}, steps=(1, 3)),),
            stuck_loses=False,
        )
    )
    assert normal is not None and misere is not None
    assert [state[0] for state in normal.losing_positions] == [0, 4]
    assert [state[0] for state in misere.losing_positions] == [1, 5]
    # The empty position is the one whose meaning the flag reverses.
    assert (0,) in normal.losing_positions
    assert (0,) not in misere.losing_positions


def test_a_game_too_big_to_enumerate_is_refused():
    huge = GameSpec(
        title="too big",
        variables=tuple(Variable(f"v{i}", 50, 0, 50) for i in range(6)),
        moves=(Move("take", {"v0": -1}),),
    )
    assert huge.problems()
    assert solve_game(huge) is None


def test_the_description_states_the_move_and_the_rule():
    spec = row_game(9, 3)
    described = describe_solution(spec, solve_game(spec))
    assert "moves first wins" in described
    assert "by 3" in described
    assert "multiple of 4" in described
