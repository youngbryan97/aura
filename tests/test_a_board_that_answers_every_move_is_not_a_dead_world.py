"""What answers to her is measured against the acts that did nothing.

LIVE 2026-09-04: the test asked whether a place had changed on every
comparison so far, effect or none. On anything she is really driving that
disqualifies the thing itself — a board answers in the same sixteen places
every move — so by the fourth move nothing on it could count as answering and
the run declared the world dead while it was moving under her.

The control is the acts that had no effect. A place that changes just as
readily when she does nothing is changing on its own; one that has never been
seen to do that is answering her.
"""

from __future__ import annotations

from core.perception.where_it_responds import Responsive, noticed


def _screen(**cells: str) -> dict:
    return {
        "layout": [
            {"text": text, "center_x": float(where.split("_")[1]) / 100.0,
             "center_y": float(where.split("_")[2]) / 100.0}
            for where, text in cells.items()
        ]
    }


def _board(*values: str) -> dict:
    return {
        "layout": [
            {"text": value, "center_x": 0.3 + 0.1 * (n % 4), "center_y": 0.4 + 0.1 * (n // 4)}
            for n, value in enumerate(values)
            if value
        ]
    }


def test_a_board_that_moves_every_time_never_reads_as_dead():
    state = Responsive()
    boards = [
        _board("2", "", "", "", "", "2", "", ""),
        _board("4", "", "", "", "", "", "2", ""),
        _board("8", "2", "", "", "", "", "", ""),
        _board("8", "4", "", "", "", "2", "", ""),
        _board("16", "", "2", "", "", "", "", ""),
        _board("16", "2", "", "", "", "", "4", ""),
    ]
    for before, after in zip(boards, boards[1:], strict=False):
        noticed(state, before, after, worked=True)
        assert not state.nothing_answers(), f"called dead after {state.acts} act(s)"


def test_the_first_act_can_answer():
    state = Responsive()
    noticed(state, _board("2", "", "", ""), _board("", "", "", "2"), worked=True)
    assert state.unanswered == 0


def test_a_clock_that_ticks_regardless_is_not_an_answer():
    """The defect the old rule existed for, and it stays fixed."""
    state = Responsive()
    # One act longer than the verdict needs, because the first act is where
    # the control comes from: with nothing yet shown to change on its own, a
    # change is evidence. The second one onwards has the clock's own history
    # to hold it against.
    for tick in range(Responsive.DEAD_AFTER + 1):
        before = _screen(a_50_10=f"12:0{tick}", b_50_50="Game Over")
        after = _screen(a_50_10=f"12:0{tick + 1}", b_50_50="Game Over")
        noticed(state, before, after, worked=False)
    assert state.nothing_answers()


def test_a_world_that_stops_answering_still_reads_as_dead():
    state = Responsive()
    live = [_board("2", "4"), _board("4", "8"), _board("8", "16")]
    for before, after in zip(live, live[1:], strict=False):
        noticed(state, before, after, worked=True)
    frozen = _board("8", "16")
    for _ in range(Responsive.DEAD_AFTER):
        noticed(state, frozen, frozen, worked=False)
    assert state.nothing_answers()


def test_a_place_that_moves_on_every_idle_act_is_discounted_once_it_can_be():
    """A banner cycling behind a finished board, with her acts doing nothing."""
    state = Responsive()
    for step in range(Responsive.DEAD_AFTER + 2):
        before = _screen(banner_20_10=f"advert {step}", board_50_50="8")
        after = _screen(banner_20_10=f"advert {step + 1}", board_50_50="8")
        noticed(state, before, after, worked=False)
    assert state.nothing_answers()


# ── and one act doing nothing is not everything doing nothing ─────────────


def test_one_act_repeated_into_a_wall_is_not_a_dead_world():
    """LIVE 2026-09-04: four presses of up, correctly refused because there
    was nothing above anything, read as the game being over — and a New Game
    clicked over a game that was very much alive."""
    state = Responsive()
    board = _board("2", "4", "8", "16")
    noticed(state, board, _board("4", "8", "16", "2"), worked=True, acting="left")
    for _ in range(Responsive.DEAD_AFTER + 2):
        noticed(state, board, board, worked=False, acting="up")
    assert state.unanswered > Responsive.DEAD_AFTER
    assert not state.nothing_answers()


def test_every_act_doing_nothing_is_a_dead_world():
    state = Responsive()
    board = _board("2", "4", "8", "16")
    for act in ("up", "down", "left", "right"):
        noticed(state, board, board, worked=False, acting=act)
    assert state.nothing_answers()


def test_a_caller_that_names_no_act_is_judged_on_the_count_alone():
    state = Responsive()
    board = _board("2", "4")
    for _ in range(Responsive.DEAD_AFTER):
        noticed(state, board, board, worked=False)
    assert state.nothing_answers()


def test_an_act_that_works_again_clears_the_run():
    state = Responsive()
    board = _board("2", "4", "8", "16")
    for act in ("up", "down", "left", "right"):
        noticed(state, board, board, worked=False, acting=act)
    assert state.nothing_answers()
    noticed(state, board, _board("4", "8", "16", "2"), worked=True, acting="left")
    assert not state.nothing_answers()


def test_starting_again_forgets_which_acts_failed():
    state = Responsive()
    board = _board("2", "4")
    for act in ("up", "down"):
        noticed(state, board, board, worked=False, acting=act)
    state.began_again()
    assert not state.unanswered_by


def test_the_pursuit_says_which_act_it_was():
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    at = source.index("worked=attempt.verdict.observed_change,")
    assert "acting=previous.chosen.name" in source[at : at + 400]
