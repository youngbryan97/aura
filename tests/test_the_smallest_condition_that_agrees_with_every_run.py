"""Condition synthesis: the conditions that decide as every recorded run needed, the smallest change first.

Pure: the examples are written here as semantic_probe records them, and nothing is run. What the probe records and
how the outcomes are forced is tested on planted faults in test_semantic_repair_mends_from_her_programs_values.py.
"""
from __future__ import annotations

from core.self_modification.condition_synthesis import Example, conditions, ranked_conditions
from core.self_modification.fix_templates import changes_at


def _runs(rows, original_parts=False):
    """Examples from (terms, want, decided)."""
    return [Example(terms, want=want, decided=decided, parts=dict(terms) if original_parts else None)
            for terms, want, decided in rows]


def test_a_comparison_is_turned_at_the_boundary():
    runs = _runs([({"item": 1, "limit": 2}, True, True), ({"item": 2, "limit": 2}, True, False),
                  ({"item": 3, "limit": 2}, False, False), ({"item": 5, "limit": 10}, True, True)], True)
    assert conditions(runs, original="item < limit", names=["item", "limit"])[0] == "item <= limit"


def test_and_and_or_are_swapped_when_the_runs_need_it():
    runs = _runs([({"age": 30, "member": True}, True, True), ({"age": 10, "member": False}, False, False),
                  ({"age": 10, "member": True}, True, False), ({"age": 30, "member": False}, True, False)], True)
    assert conditions(runs, original="age >= 18 and member", names=["age", "member"])[0] == "age >= 18 or member"


def test_a_missing_conjunct_is_joined_to_what_was_there():
    runs = _runs([({"price": 200, "member": True}, True, True), ({"price": 50, "member": False}, False, False),
                  ({"price": 50, "member": True}, False, True)], True)
    found = conditions(runs, original="member", names=["member"])
    assert found[0].startswith("member and price ")


def test_a_condition_that_cannot_come_out_true_is_never_offered():
    """LIVE 16 November 2026: ``len(rows) < 0`` before a raise passed the fault's test and left the refusal dead."""
    runs = [Example({"rows is None": False, "len(rows)": 2}, want=False, decided=True, parts={"len(rows)": 2})]
    found = conditions(runs, original="len(rows) > 0", names=["rows"])
    assert "len(rows) < 0" not in found and "len(rows) >= 0" not in found
    assert found[0] == "len(rows) <= 0"


def test_a_bound_moved_to_a_value_one_run_showed_comes_after_a_turned_comparison():
    runs = [Example({"len(rows)": 2}, want=False, decided=True, parts={"len(rows)": 2})]
    ranked = ranked_conditions(runs, original="len(rows) > 0", names=["rows"])
    costs = dict((text, cost) for cost, text in ranked)
    assert costs["len(rows) <= 0"] < costs["len(rows) > 2"]


def test_a_string_one_run_had_is_not_what_a_condition_compares_with():
    """LIVE 16 November 2026 the first guard offered for blank text was ``text == '   '``: true for the one run
    that failed and for no other blank text."""
    runs = _runs([({"text": "   ", "len(text)": 3}, True, -1),
                  ({"text": "hello there", "len(text)": 11}, False, -1),
                  ({"text": "a b", "len(text)": 3}, False, -1)])
    # Nothing recorded here tells blank text from "a b" but the text itself: the comparison is offered, as the
    # costliest kind there is, for the tests to judge.
    cost, found = ranked_conditions(runs, names=["text"])[0]
    assert found == "text == '   '" and cost >= 5
    sized = _runs([({"text": "   ", "len(text)": 3, "len(words)": 0}, True, -1),
                   ({"text": "hello there", "len(text)": 11, "len(words)": 2}, False, -1)])
    assert conditions(sized, names=["words"])[0] == "not words"


def test_runs_that_contradict_each_other_give_nothing():
    runs = _runs([({"x": 1}, True, True), ({"x": 1}, False, True)])
    assert conditions(runs, original="x > 0") == []


def test_the_templates_cannot_write_a_missing_conjunct_or_a_guard():
    """What semantic repair adds: the fix templates change what is on a line; they cannot join a test to it or put
    a guard before it."""
    price = "def price_after(price, member):\n    if member:\n        return price * 0.9\n    return price\n"
    assert not any("and" in changed.split("\n")[1] for _, changed in changes_at(price, 2))
    words = ("def first_word(text):\n    if text is None:\n        return ''\n    words = text.split()\n"
             "    return words[0]\n")
    assert not any(changed.count("if ") > 1 for line in range(1, 6) for _, changed in changes_at(words, line))


def test_of_two_conditions_that_cost_the_same_the_one_reading_every_name_of_the_faulty_statement_comes_first():
    """LIVE 17 November 2026: ``prices[item]`` was mended with a size test that never read the item (052670e4a)."""
    examples = [
        Example({"basket_item": "tea", "len(prices)": 2, "basket_item in prices": True}, want=True),
        Example({"basket_item": "oats", "len(prices)": 1, "basket_item in prices": False}, want=False),
    ]
    found = conditions(examples, names=("prices", "basket_item"))
    assert found[0] == "basket_item in prices", found
