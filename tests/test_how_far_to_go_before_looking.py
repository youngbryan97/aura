"""How many acts to take on the model before checking it against the world.

Looking costs about half a second and the thing answers a second after she
acts, so a read-decide-confirm move costs two. A game of 2048 is five hundred
moves — nineteen minutes of watching a screen, and none of it is the thinking.
"""

from __future__ import annotations

from pathlib import Path

from core.cognition.how_far_to_go_before_looking import HowFarToGo


def test_she_will_not_run_ahead_of_a_model_she_does_not_trust() -> None:
    far = HowFarToGo()
    for _ in range(20):
        far.it_was_where_she_said(1)
    assert far.how_many(trusted=0.0) == 1
    assert far.how_many(trusted=1.0) > 1


def test_it_goes_further_as_the_model_keeps_being_right() -> None:
    far = HowFarToGo()
    seen = [far.how_many(trusted=1.0)]
    for _ in range(6):
        far.it_was_where_she_said(seen[-1])
        seen.append(far.how_many(trusted=1.0))
    assert seen[0] == 1
    assert seen == sorted(seen), seen
    assert seen[-1] > seen[0]
    assert seen[-1] <= HowFarToGo.NEVER_MORE_THAN


def test_never_further_than_the_model_has_earned() -> None:
    """A rule right four times in five has no business carrying her eight."""
    far = HowFarToGo()
    for _ in range(20):
        far.it_was_where_she_said(1)
    assert far.far == HowFarToGo.NEVER_MORE_THAN
    assert far.how_many(trusted=1.0) == HowFarToGo.NEVER_MORE_THAN
    assert far.how_many(trusted=0.5) == HowFarToGo.NEVER_MORE_THAN // 2
    assert far.how_many(trusted=0.1) == 1


def test_one_disagreement_puts_her_back_to_looking_every_time() -> None:
    far = HowFarToGo()
    for _ in range(8):
        far.it_was_where_she_said(far.how_many(trusted=1.0))
    assert far.how_many(trusted=1.0) > 1
    far.it_was_not(4)
    assert far.how_many(trusted=1.0) == 1


def test_a_run_that_broke_is_re_walked_one_at_a_time() -> None:
    """A run of four that disagrees at the end says one of four things went
    wrong and not which, and a rule cannot be corrected by that."""
    far = HowFarToGo()
    far.it_was_not(4)
    assert far.walking_back == 3
    for _ in range(3):
        assert far.how_many(trusted=1.0) == 1
        far.it_was_where_she_said(1)
    assert far.walking_back == 0
    # And only then does it start earning distance again.
    far.it_was_where_she_said(1)
    assert far.how_many(trusted=1.0) == 2


def test_how_often_it_held_is_laplace_and_not_a_bare_fraction() -> None:
    far = HowFarToGo()
    assert far.how_often_it_held == 0.5, "nothing seen is not certainty either way"
    far.it_was_where_she_said(1)
    assert far.how_often_it_held < 1.0, "one run right is not proof"


def test_a_run_of_one_counts_or_it_can_never_grow() -> None:
    """It starts at one, it only grows when a prediction holds, and if only
    multi-act runs counted then growing would need the very thing it makes
    possible. Measured live over three games and six hundred moves: the
    distance stayed at one for every move and no run was ever tried.
    """
    far = HowFarToGo()
    assert far.how_many(trusted=1.0) == 1
    far.it_was_where_she_said(1)
    assert far.how_many(trusted=1.0) == 2, "one act landing where she said is evidence"


def test_the_loop_checks_a_run_of_any_length_including_one() -> None:
    """A prediction is made and checked however many acts the run is.

    This used to grep for the branch that handled a run of one on its own,
    and that branch is gone: the count and the prediction are worked out once
    for every run, folding the model over each act in it, so a run of one is
    checked by the same line as a run of four. The guarantee is stronger and
    the literal it was written against no longer exists.
    """

    from core.skills import screen_pursuit

    text = Path(screen_pursuit.__file__).read_text(encoding="utf-8")
    assert 'expected["took"] >= 1' in text, "a run of one has to be checked"
    at = text.index('expected["took"] = len(follow_on) + 1')
    near = text[at : at + 500]
    assert "for step in (key, *follow_on):" in near, (
        "the prediction is no longer folded over the whole run"
    )
    assert "where = foresee(where, step)" in near
