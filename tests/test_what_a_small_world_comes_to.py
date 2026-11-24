"""A world small enough to list is worked out to the end, whoever acts in it.

The same working-out answers a world she acts in alone, one where another side answers her in turn, and one that
deals something of its own between her acts; nothing in it knows which (core/cognition/what_a_small_world_comes_to.py).
"""
from __future__ import annotations

from core.cognition.what_a_small_world_comes_to import AGAINST, CHANCE, HERS, worked_out


def test_a_world_she_acts_in_alone_is_worked_out_to_the_fewest_acts() -> None:
    # Steps along a line toward its far end, one or two at a time.
    worked = worked_out([0], lambda at: [(f"+{n}", at + n) for n in (1, 2) if at + n <= 6],
                        lambda at: 1.0 if at == 6 else None, lambda at: HERS)
    assert worked.complete and worked.what_it_comes_to(0) == (1, 3)
    assert worked.best(0) == "+2" and worked.best(5) == "+1"


def test_a_world_where_another_side_answers_her_is_worked_out_under_best_play_on_both_sides() -> None:
    # Stones in a pile, each side in turn taking one or two; whoever takes the last wins.
    def next_of(state):
        stones, mover = state
        return [(f"take {n}", (stones - n, 1 - mover)) for n in (1, 2) if n <= stones]

    def end_of(state):
        stones, mover = state
        return None if stones else (-1.0 if mover == 0 else 1.0)

    worked = worked_out([(n, m) for n in range(11) for m in (0, 1)], next_of, end_of,
                        lambda state: HERS if state[1] == 0 else AGAINST)
    for stones in range(1, 11):
        forced, _turns = worked.what_it_comes_to((stones, 0))
        assert forced == (-1 if stones % 3 == 0 else 1), stones
    assert worked.best((4, 0)) == "take 1" and worked.best((5, 0)) == "take 2"
    # A loss is put off as long as it can be: from three, either take leaves her a win to lose, two turns on.
    assert worked.what_it_comes_to((3, 0)) == (-1, 2)


def test_a_world_that_deals_between_her_acts_is_worked_out_as_her_chance() -> None:
    shares = {"sure": [("good", 0.9), ("bad", 0.1)], "even": [("good", 0.5), ("bad", 0.5)]}

    def next_of(state):
        if state == "start":
            return [("the sure way", "sure"), ("the even way", "even")]
        return [(f"dealt {end}", end, share) for end, share in shares[state]]

    worked = worked_out(["start"], next_of, lambda s: {"good": 1.0, "bad": -1.0}.get(s),
                        lambda s: HERS if s == "start" else CHANCE)
    assert worked.what_it_comes_to("start") == (0, -1), "the world can deal her the bad end either way"
    assert abs(worked.chance[worked.at("start")] - 0.9) < 1e-12 and worked.best("start") == "the sure way"


def test_a_world_larger_than_it_may_list_is_worked_out_as_far_as_it_was_listed() -> None:
    worked = worked_out([0], lambda at: [("+1", at + 1)], lambda at: 1.0 if at == 50 else None, lambda at: HERS,
                        most=10)
    assert not worked.complete and len(worked) == 10
    assert worked.what_it_comes_to(0) == (0, -1) and worked.best(9) == "+1"


def test_her_look_ahead_works_a_world_small_enough_to_list_out_to_the_end() -> None:
    from dataclasses import dataclass

    from core.agency import looking_ahead

    @dataclass(frozen=True)
    class _APlace:
        at: int

        def as_text(self) -> str:
            return str(self.at)

        def numbers(self) -> tuple[float, ...]:
            return (2.0 ** self.at,)

        def newness(self) -> float:
            return 1.0

        def empty(self) -> int:
            return 2

        def places(self) -> int:
            return 4

    class _ALine:
        """Nine places along a line, a step either way; the far end reads 256."""

        def expect(self, state, act):
            there = state.at + {"on": 1, "back": -1}.get(act, 0)
            return _APlace(there) if 0 <= there <= 8 and act in ("on", "back") else None

        def confidence(self) -> float:
            return 1.0

    looking_ahead._TOO_BIG.clear()
    ranked = looking_ahead.look_ahead(_ALine(), _APlace(4), ["on", "back"], toward="256", budget_s=0.5)
    assert ranked["on"][0] > ranked["back"][0]
    assert ranked["on"][1] == "it reaches 256 for certain in 4 acts", ranked["on"][1]


def test_a_cap_on_situations_holds_where_none_are_kept() -> None:
    # Asked to stop at ten, with nothing kept: suggestions of 23 November 2026 found it listed 51.
    worked = worked_out([0], lambda at: [("+1", at + 1), ("+2", at + 2)], lambda at: None, lambda at: HERS,
                        most=10, keep=False)
    assert len(worked) == 10 and not worked.complete


def test_a_world_that_deals_the_same_situation_again_and_again_is_worked_out_to_its_true_chance() -> None:
    # Half the time the good end, half the time back where she was: in the end, certain. A cap of as many passes as
    # there are situations stopped it at three in four.
    def next_of(state):
        return [("try", "dealing")] if state == "start" else [("good", "good", 0.5), ("again", "start", 0.5)]

    worked = worked_out(["start"], next_of, lambda s: 1.0 if s == "good" else None,
                        lambda s: CHANCE if s == "dealing" else HERS)
    assert worked.chance_settled and abs(worked.chance[worked.at("start")] - 1.0) < 1e-9
    # Where the world deals the same situation back nearly every time, the chance settles only after millions of
    # passes: out of time, it says so, and what it says is a bound from below.
    def slow(state):
        return [("try", "dealing")] if state == "start" else [("good", "good", 1e-6), ("again", "start", 1 - 1e-6)]

    rushed = worked_out(["start"], slow, lambda s: 1.0 if s == "good" else None,
                        lambda s: CHANCE if s == "dealing" else HERS, seconds=0.05)
    assert rushed.complete and not rushed.chance_settled and 0.0 < rushed.chance[rushed.at("start")] < 1.0
