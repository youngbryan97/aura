"""She knows how a contest stands: who is ahead, what winning takes, what is left to lose, and whether it has stalled."""
from __future__ import annotations

from core.agency.how_the_contest_stands import ContestStands, what_wins


def test_what_wins_is_read_from_the_games_own_words():
    pong = what_wins("Up and down arrows move your paddle. Keep the ball from getting past you. First to 5 points wins.")
    assert pong.to_score == 5 and not pong.timed
    assert what_wins("Score 10 points to win!").to_score == 10
    assert what_wins("Reach level 3 to save the city. Don't lose all your lives.").to_level == 3
    assert what_wins("Reach level 3 to save the city. Don't lose all your lives.").lives_lose
    assert what_wins("Collect all 8 stars before time runs out").to_collect == 8
    assert what_wins("Collect all 8 stars before time runs out").timed
    assert what_wins("Use the arrow keys to move.").says() == ""


def _two_sides(contest: ContestStands, mine: int, theirs: int, at: float) -> None:
    contest.counted({"number at 0.45,0.1": mine, "number at 0.55,0.1": theirs},
                    {"number at 0.45,0.1": (0.45, 0.08), "number at 0.55,0.1": (0.55, 0.08)}, her_x=0.05, at=at)


def test_two_scores_on_a_row_are_hers_and_theirs_and_settle_the_game():
    contest = ContestStands()
    contest.heard("First to 5 points wins.")
    _two_sides(contest, 3, 1, at=10.0)
    assert contest.standing() == "ahead by 2" and contest.to_go() == (2, 4)
    assert contest.says() == "3–1, I'm ahead by 2; 2 more to win, they need 4."
    assert contest.settled() == ""
    _two_sides(contest, 5, 1, at=20.0)
    assert contest.settled() == "won"
    lost = ContestStands()
    lost.heard("First to 5 points wins.")
    _two_sides(lost, 0, 5, at=5.0)
    assert lost.settled() == "lost" and lost.standing() == "behind by 5"


def test_named_counters_are_read_by_what_they_name():
    contest = ContestStands()
    contest.counted({"you": 2, "computer": 4, "lives": 1}, {}, her_x=None, at=1.0)
    assert (contest.mine, contest.theirs, contest.lives) == (2, 4, 1)
    contest.counted({"you": 2, "computer": 4, "lives": 0}, {}, her_x=None, at=2.0)
    assert contest.settled() == "lost"


def test_a_contest_where_nothing_changes_for_long_past_its_pace_has_stalled():
    contest = ContestStands()
    for n, at in enumerate((0.0, 5.0, 10.0, 15.0)):
        _two_sides(contest, n, 0, at=at)
    assert not contest.stalled(30.0)
    assert contest.stalled(15.0 + 31.0)  # four of its five-second gaps is under the floor of thirty seconds
