"""What changes slowly and decides how things come out is found from what was seen and how each episode came out.

core/cognition/what_changes_slowly_and_decides.py. Bryan asked on 23 November 2026 whether a structural judgement can
bootstrap from implicit structure, patterns in outcomes with no explicit rules, and hold for goals that are not a win
or a loss. These worlds are numbers only: no game, no rules, and how each came out given in whatever units it had.
"""
from __future__ import annotations

import numpy as np

from core.cognition.what_changes_slowly_and_decides import learned, slow_directions


def _world(episodes: int, steps: int, *, slow_decides: float, quick_decides: float = 0.0, as_score: bool = False,
           seed: int = 0) -> tuple[list[np.ndarray], list[float]]:
    """Six numbers seen each step: the first drifts slowly through an episode, the rest are new each step. How an
    episode came out depends on where the slow one stood, by ``slow_decides``, and on the last step's second number,
    by ``quick_decides``."""
    rng = np.random.default_rng(seed)
    seen, ends = [], []
    for _ in range(episodes):
        slow = rng.normal() + np.cumsum(rng.normal(0, 0.1, steps))
        rows = np.column_stack([slow, rng.normal(0, 1, (steps, 5))])
        lean = slow_decides * slow.mean() + quick_decides * rows[-1, 1]
        if as_score:
            ends.append(float(1000 + 250 * lean + rng.normal(0, 50)))
        else:
            ends.append(float(rng.random() < 1 / (1 + np.exp(-lean))))
        seen.append(rows)
    return seen, ends


def test_the_slow_number_is_the_slowest_direction() -> None:
    seen, _ends = _world(60, 30, slow_decides=0.0)
    directions, slowness = slow_directions(seen)
    assert len(directions) >= 1 and slowness[0] < 0.2
    first = directions[0] / np.linalg.norm(directions[0])
    assert abs(first[0]) > 0.95, first


def test_what_decides_is_learned_from_how_episodes_came_out() -> None:
    seen, ends = _world(400, 30, slow_decides=2.0, seed=1)
    found = learned(seen, ends)
    assert found is not None
    weights = found.weights()
    assert weights[0] > 0 and abs(weights[0]) > 5 * np.abs(weights[1:]).max(), weights
    before, after = found.held_out
    assert after < before


def test_a_goal_given_as_a_score_is_learned_as_well_as_a_win() -> None:
    seen, ends = _world(400, 30, slow_decides=2.0, as_score=True, seed=2)
    found = learned(seen, ends)
    assert found is not None and found.weights()[0] > 0


def test_what_only_the_last_step_shows_is_not_taken_for_something_slow() -> None:
    seen, ends = _world(400, 30, slow_decides=0.0, quick_decides=3.0, seed=3)
    found = learned(seen, ends)
    assert found is None or abs(found.weights()[1]) < 0.1


def test_nothing_is_learned_where_nothing_slow_decides() -> None:
    seen, ends = _world(400, 30, slow_decides=0.0, seed=4)
    assert learned(seen, ends) is None


def test_what_the_judgement_held_already_saw_is_not_learned_again() -> None:
    seen, ends = _world(400, 30, slow_decides=2.0, seed=5)
    # A judgement that already reads the slow number as well as it can be read leaves nothing to add.
    offsets = [np.full(len(rows), 2.0 * rows[:, 0].mean()) for rows in seen]
    found = learned(seen, ends, offsets)
    assert found is None or abs(found.weights()[0]) < 0.5, found
