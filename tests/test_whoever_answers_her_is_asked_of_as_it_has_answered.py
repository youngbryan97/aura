"""Whoever answers her turn is looked ahead at as it has been seen to answer: by luck, or choosing against her.

Bryan, 23 November 2026, of her everyday thinking outside board games: "I go here. If they come here, I can go
there", chained, and the consequences weighed. Her look ahead over a world that answers her used to take every answer
as luck and average it, so a world that kept answering against her was planned for as if it were dice. She now keeps
where each answer fell among the answers that could have come, by her own judgement, and weighs the other side's turn
by it.
"""
from __future__ import annotations

import random

import pytest

from core.agency.looking_ahead import answers_ready, how_it_was_answered, look_ahead
from core.cognition.how_they_answer import HowTheyAnswer, where_it_fell
from core.perception.how_it_moves import RULES, HowItMoves
from core.perception.what_is_there import Arrangement, Cell
from core.perception.where_it_responds import what_is_there

ACTS = ("left", "right", "up", "down")


def test_answers_by_luck_lean_no_way_and_answers_against_her_lean_against() -> None:
    rng = random.Random(7)
    luck = HowTheyAnswer()
    for _ in range(150):
        could = [rng.random() for _ in range(6)]
        luck.answered(could, rng.choice(could))
    assert luck.leaning() == 0.0 or abs(luck.leaning()) < 0.15
    against = HowTheyAnswer()
    for _ in range(40):
        could = [rng.random() for _ in range(6)]
        against.answered(could, min(could))
    assert against.leaning() < -0.7
    # Half the time the worst for her, otherwise luck: about half a lean.
    half = HowTheyAnswer()
    for i in range(200):
        could = [rng.random() for _ in range(6)]
        half.answered(could, min(could) if i % 2 else rng.choice(could))
    assert -0.75 < half.leaning() < -0.3


def test_a_few_answers_are_not_yet_a_lean() -> None:
    once = HowTheyAnswer()
    once.answered([0.1, 0.5, 0.9], 0.1)
    assert once.leaning() == 0.0
    assert where_it_fell([0.2, 0.2], 0.2) is None, "where nothing could have gone differently, nothing was chosen"


def test_their_turn_is_weighed_toward_the_worst_for_her_as_far_as_they_choose() -> None:
    against = HowTheyAnswer(fell=[0.0] * 60, ways=[6] * 60)
    # Sixty answers, every one the worst for her: a lean of about 0.8, so their turn comes to near the worst.
    assert 1.0 <= against.weigh([1.0, 5.0, 9.0], [1, 1, 1]) < 2.0
    assert HowTheyAnswer().weigh([1.0, 5.0, 9.0], [1, 1, 1]) == pytest.approx(5.0)
    kept = HowTheyAnswer.from_memory(against.as_memory())
    assert kept.leaning() == against.leaning()


def _board(text: str):
    return what_is_there({"ok": True, "text": text, "layout": [], "bounds": []}, None)


def _one_that_knows() -> HowItMoves:
    knows = HowItMoves()
    for rule in RULES:
        knows.tried[rule.name] = knows.tried_when_it_moved[rule.name] = 20
        got = 20 if rule.name == "slides and combines" else 0
        knows.right[rule.name] = knows.right_when_it_moved[rule.name] = got
    knows.seen = knows.moved = 20
    return knows


class _AWorldThatPutsThingsDown:
    """A world that drops a 2 into one of the free places after each of her acts."""

    def __init__(self, answers: HowTheyAnswer | None = None) -> None:
        self.answers = answers if answers is not None else HowTheyAnswer()

    def might_do(self, state):
        free = [(r, c) for r in range(state.rows) for c in range(state.columns) if state.at(r, c) is None][:4]
        return tuple(
            (Arrangement(rows=state.rows, columns=state.columns,
                         cells=(*state.cells, Cell(row=r, column=c, says="2", at=(float(c), float(r)))),
                         down_at=state.down_at, across_at=state.across_at), 1.0 / len(free))
            for r, c in free)


BOARD = "2 4 . .\n. 8 . .\n. . 2 .\n4 . . 8"


def test_against_a_world_that_answers_against_her_every_move_is_judged_by_its_worse_answers(monkeypatch) -> None:
    import core.agency.looking_ahead as looking

    # The same search both times: the compiled one takes answers as luck, and is not what is tested here.
    monkeypatch.setattr(looking, "_through_a_compiled_world", lambda *a, **k: None)
    knows, state = _one_that_knows(), _board(BOARD)
    by_luck = look_ahead(knows, state, ACTS, world=_AWorldThatPutsThingsDown(), depth=2, budget_s=2.0)
    against = look_ahead(knows, state, ACTS,
                         world=_AWorldThatPutsThingsDown(HowTheyAnswer(fell=[0.0] * 60, ways=[4] * 60)),
                         depth=2, budget_s=2.0)
    assert by_luck and set(by_luck) == set(against)
    assert all(against[act][0] <= by_luck[act][0] + 1e-9 for act in by_luck)
    assert any(against[act][0] < by_luck[act][0] - 1e-9 for act in by_luck)


def test_where_the_answer_that_came_fell_is_kept_with_the_world() -> None:
    knows, state = _one_that_knows(), _board(BOARD)
    world = _AWorldThatPutsThingsDown()
    made = knows.expect(state, "left")
    ways = [way for way, _share in world.might_do(made)]
    from core.agency.how_good_is_this import how_good

    worst = min(ways, key=lambda way: how_good(way, knows=knows))
    how_it_was_answered(knows, made, worst, world)
    assert world.answers.fell == [0.0] or world.answers.fell[0] < 0.5


def test_she_has_her_answer_ready_to_each_likely_answer_and_to_the_worst() -> None:
    knows, state = _one_that_knows(), _board(BOARD)
    ready = answers_ready(knows, state, "left", ACTS, world=_AWorldThatPutsThingsDown())
    assert ready, "if they come there, she has somewhere to go"
    assert all(act in ACTS for _way, act, _worth in ready)


def test_where_it_answers_against_her_the_move_is_said_with_her_answers_ready() -> None:
    from types import SimpleNamespace

    from core.skills.screen_pursuit_decision import _with_her_answers_ready

    knows = SimpleNamespace(rules=_one_that_knows())
    laid_out = _board(BOARD)
    ahead = {"left": (3.0, "keeps the largest in the corner"), "right": (1.0, "")}
    choosing = _AWorldThatPutsThingsDown(HowTheyAnswer(fell=[0.0] * 60, ways=[4] * 60))
    said = _with_her_answers_ready(ahead, knows=knows, laid_out=laid_out, choices=list(ACTS), world=choosing,
                                   toward="", approach="", weights=None)
    assert said["left"][1].startswith("keeps the largest in the corner — if it answers with 2 at row")
    assert "I'll " in said["left"][1] and said["right"] == ahead["right"]
    # By luck, nothing is added: a die is not answered.
    assert _with_her_answers_ready(ahead, knows=knows, laid_out=laid_out, choices=list(ACTS),
                                   world=_AWorldThatPutsThingsDown(), toward="", approach="", weights=None) == ahead
