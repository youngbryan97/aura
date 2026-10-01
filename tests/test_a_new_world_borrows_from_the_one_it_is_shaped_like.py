"""A world she has never met starts from the solved world shaped like it.

External review, 2026-10-01: the structure mapper was the best generality idea
in the repository and nothing she played used it; it was capped at seven
objects; and a scrambled distractor scored 0.67 against a null mean of 0.50,
which a mean-based control called structural. Each of those is pinned here.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

from core.agency.the_world_it_is_most_like import (
    graph_from_memory,
    graph_to_memory,
    lend_to,
    lent,
    most_like,
    shape_of,
)
from core.cognition.structure_mapping import Graph, Relation, map_structures, shuffled_null
from core.perception.how_it_moves import HowItMoves, composed
from core.perception.what_is_there import Arrangement, Cell

pytestmark = pytest.mark.unit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import check_the_world_it_is_most_like as experiment  # noqa: E402


def _board(rows: list[str]) -> Arrangement:
    cells = tuple(
        Cell(row, column, said, (0.0, 0.0))
        for row, line in enumerate(rows)
        for column, said in enumerate(line.split())
        if said != "."
    )
    return Arrangement(len(rows), len(rows[0].split()), cells)


NUMBERS = _board(["2 4 8 .", "2 2 16 .", ". 4 8 2", ". . 4 4"])
LETTERS = _board(["A B C .", "A A D .", ". B C A", ". . B B"])
OTHER_NUMBERS = _board(["3 6 12 . .", "3 3 24 . .", ". 6 12 3 .", ". . 6 6 .", ". . . . ."])


def test_the_words_on_a_world_are_not_its_shape():
    """3s, 6s and 12s are laid out like 2s, 4s and 8s; letters are not."""
    numbers, others, letters = shape_of(NUMBERS), shape_of(OTHER_NUMBERS), shape_of(LETTERS)
    alike = map_structures(others, numbers, same_vocabulary=True)
    unlike = map_structures(letters, numbers, same_vocabulary=True)
    assert alike is not None and unlike is not None
    assert alike.score > unlike.score
    assert alike.mapping["thing 6"] == "thing 4"


def test_an_inventory_is_not_a_shape():
    """Every world has things and acts; counting them matched everything."""
    look = shape_of(_board(["3 . .", ". . .", ". . 3"]), ("a", "d", "w", "s"))
    predicates = {relation.predicate for relation in look.relations}
    assert "an act" not in predicates and "there" not in predicates


def test_a_shape_survives_being_kept():
    graph = shape_of(NUMBERS)
    back = graph_from_memory(graph_to_memory(graph), "kept")
    assert back is not None and back.relations == graph.relations


def test_the_closest_world_is_lent_when_it_clears_its_control():
    likeness = most_like(
        shape_of(OTHER_NUMBERS), {"numbers": shape_of(NUMBERS), "letters": shape_of(LETTERS)}
    )
    assert likeness is not None and likeness.world == "numbers"
    assert likeness.score > likeness.scrambled_at
    assert 0.0 < likeness.margin <= 1.0


def test_the_control_checks_the_choice_and_does_not_make_it(monkeypatch):
    """A worse match that clears its control is not lent in place of a better one."""
    from core.agency import the_world_it_is_most_like as module

    here = Graph("here", (Relation("r", ("x", "y")),))
    perfect = Graph("perfect", (Relation("r", ("p", "q")),))
    worse = Graph(
        "worse", tuple(Relation("r", (f"o{i}", f"o{i + 1}")) for i in range(6))
    )
    assert module._shared(here, perfect)[0] > module._shared(here, worse)[0]
    assert most_like(here, {"perfect": perfect, "worse": worse}) is None, (
        "the closest world could not be told from its scrambled copies, so "
        "nothing is lent — not the next one down"
    )


def test_a_tie_between_worlds_that_taught_different_things_lends_nothing():
    look = shape_of(OTHER_NUMBERS)
    same = shape_of(NUMBERS)
    assert most_like(look, {"a": same, "b": same}, teaches={"a": "slides", "b": "steps"}) is None
    lent_by = most_like(look, {"a": same, "b": same}, teaches={"a": "slides", "b": "slides"})
    assert lent_by is not None and lent_by.world == "a"


def test_what_is_lent_is_added_to_what_she_has_seen_here():
    mine = HowItMoves(right={"slides": 2}, tried={"slides": 2}, seen=2, moved=2,
                      right_when_it_moved={"slides": 2}, tried_when_it_moved={"slides": 2})
    likeness = most_like(shape_of(OTHER_NUMBERS), {"numbers": shape_of(NUMBERS)})
    assert likeness is not None
    knew = {
        "moves": {"right": {"slides": 200}, "tried": {"slides": 200}, "seen": 200,
                  "moved": 200, "right_when_it_moved": {"slides": 200},
                  "tried_when_it_moved": {"slides": 200}, "pushes": {"left": {"left": 50}},
                  "read_through": [4, 4]},
        "world": {"arrives": {"2": 90, "4": 10}, "acts": 100, "acts_with_arrivals": 100},
    }
    given = lent(knew, likeness)
    assert "pushes" not in given["moves"], "which act pushes which way is not lent"
    assert set(given["world"]["arrives"]) <= {"3", "6", "12", "24"}, "arrivals renamed into this world"
    assert lend_to(mine, given)
    assert mine.tried["slides"] > 2, "lent counts are added"
    assert mine.tried["slides"] <= 2 + 4, "no more than a fresh start is worth"


def test_past_seven_objects_the_mapper_still_answers_and_says_how():
    chain = Graph("a", tuple(Relation("next", (f"a{i}", f"a{i + 1}")) for i in range(11)))
    other = Graph("b", tuple(Relation("next", (f"b{i}", f"b{i + 1}")) for i in range(11)))
    alignment = map_structures(chain, other)
    assert alignment is not None and alignment.score == 1.0 and alignment.exhaustive is False


def test_a_distractor_that_shares_a_little_does_not_clear_the_upper_tail():
    """The review's case: a mean-based control called a thin match structural."""
    source = Graph("heat", (
        Relation("hotter", ("cup", "cube")), Relation("conducts", ("cup", "cube")),
        Relation("touches", ("bar", "cup")), Relation("touches", ("bar", "cube")),
    ))
    real = Graph("water", (
        Relation("higher", ("tank", "end")), Relation("flows", ("tank", "end")),
        Relation("joins", ("pipe", "tank")), Relation("joins", ("pipe", "end")),
    ))
    assert shuffled_null(source, real)["beats_the_upper_tail"]
    fooled_the_mean = 0
    for seed in range(10):
        rng = random.Random(seed)
        thin = Graph("thin", tuple(
            Relation(r.predicate, tuple(rng.choice(["tank", "end", "pipe"]) for _ in r.args))
            for r in real.relations
        ))
        verdict = shuffled_null(source, thin)
        assert not verdict["beats_the_upper_tail"], f"seed {seed}: {verdict['score']}"
        fooled_the_mean += bool(verdict["structural"])
    # Measured: eight of these ten clear the mean. That is the review's point.
    assert fooled_the_mean >= 5


def test_the_pursuit_asks_by_shape_and_lends_once(monkeypatch):
    from core.runtime import what_she_learned
    from core.skills.screen_pursuit_decision_steps import (
        _borrow_from_the_world_it_is_most_like as borrow,
    )

    solved = HowItMoves()
    truth = composed("all the way", True, "everything")
    rng = random.Random(3)
    state = NUMBERS
    for _ in range(60):
        way = rng.choice(("left", "right", "up", "down"))
        after = truth.apply(state, way) or state
        solved.watched(state, way, after)
        state = after if after.cells else NUMBERS
    assert solved.rule() is not None
    record = {"moves": solved.as_memory(), "shape": graph_to_memory(shape_of(NUMBERS))}
    monkeypatch.setattr(what_she_learned, "kept_worlds", lambda: ["numbers"])
    monkeypatch.setattr(what_she_learned, "recall", lambda name: dict(record))
    said: list[str] = []
    monkeypatch.setattr("core.skills.screen_pursuit._tell", said.append)

    class Knows:
        rules = HowItMoves()

    like_it: dict = {"kind": "", "looked": True, "this_world": "new"}
    borrow(knows=Knows, laid_out=OTHER_NUMBERS, like_it=like_it, world=None)
    assert like_it.get("lent") == "numbers"
    assert Knows.rules.tried, "the solved world's counts were lent"
    assert said and "shaped like numbers" in said[0]
    before = dict(Knows.rules.tried)
    borrow(knows=Knows, laid_out=OTHER_NUMBERS, like_it=like_it, world=None)
    assert Knows.rules.tried == before, "lent at most once"


def test_on_a_held_out_variant_what_is_lent_helps_and_a_wrong_loan_is_overturned(monkeypatch, capsys):
    """Twelve seeds of the offline experiment, crowded boards met mid-way."""
    monkeypatch.setattr(
        sys, "argv",
        ["check", "--seeds", "12", "--moves", "40", "--look-after", "0", "--crowded"],
    )
    experiment.main()
    import json

    out = capsys.readouterr().out
    report = json.loads(out[out.index("{"):])
    held_out = report["held out"]
    assert held_out["chosen"].get("solved", 0) >= 10
    assert held_out["lent_faster"] > held_out["lent_slower"]
    for name in ("held out", "distractor", "misleading"):
        assert report[name]["never_settled_lent"] == 0, f"{name}: a loan stopped her settling"
