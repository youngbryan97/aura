"""The outside check returns every input behind its verdict as data (Master Todo F16, to build).

Its words name an input in a line of prose cut at 160 characters. A repair search reading that has to parse prose to
learn which input to keep. Each input the check counts now comes back as a record: the function, the input, what it
came to before and after, the kind of difference, and whether it counted as lost or as a doubt.
"""
from __future__ import annotations

from outside_bar.verify_change import difference, verify
from tests.test_outside_bar_verifies_a_change import (  # noqa: F401 - shop is a fixture, found by its own name
    _GUARD,
    _PRICES,
    _committed,
    _git,
    shop,
)


def test_each_kind_of_difference_is_named():
    assert difference("3", "None") == "another value"
    assert difference("raised ValueError: out of range", "-5") == "a refusal gone"
    assert difference("3", "raised KeyError: 'tea'") == "now raises"
    assert difference("raised ValueError: x", "raised TypeError: y") == "another refusal"
    assert difference("3", "took too long") == "no longer finishes"
    assert difference("3", None) == "came to nothing"


def test_a_mend_that_holds_has_no_counterexample(shop):  # noqa: F811 - the fixture
    result = verify(shop, _committed(shop, _PRICES.replace("return table[item]", "return table.get(item)")))
    assert result["held"] and result["counterexamples"] == [], result


def test_an_input_near_the_fault_that_lost_its_value_comes_back_as_data(shop):  # noqa: F811 - the fixture
    (shop / "tests" / "test_prices.py").write_text(_GUARD.replace('price({"tea": 3}, "tea") == 3',
                                                                  'price({"tea": 3, "oats": 2}, "tea") == 3'))
    _git(shop, "add", "-A")
    _git(shop, "commit", "-q", "-m", "a guard with two items")
    overfit = _PRICES.replace("    return table[item]", "    if len(table) <= 1:\n        return None\n    return table[item]")
    result = verify(shop, _committed(shop, overfit))
    lost = [one for one in result["counterexamples"] if one["counted"] == "lost"]
    assert any(one["function"].endswith("::price") and one["before"] == "3" and one["after"] == "None"
               and one["kind"] == "another value" and isinstance(one["input"], dict) for one in lost), result
    assert len(lost) >= len(result["values_kept"]["changed_near"]), "every line it counted has its input"


def test_a_value_a_passing_test_returned_comes_back_with_its_test_and_readable(shop):  # noqa: F811 - the fixture
    (shop / "tests" / "test_prices.py").write_text(_GUARD.replace('assert total({"tea": 3, "oats": 2}, ["tea", "oats"]) == 5',
                                                                  'assert total({"tea": 3, "oats": 2}, ["tea", "oats"]) > 0'))
    _git(shop, "add", "-A")
    _git(shop, "commit", "-q", "-m", "a looser test")
    drifted = _PRICES.replace("return table[item]", "return table.get(item)").replace(
        "return sum(price(table, item) for item in items)", "return sum(price(table, item) for item in items) + 1")
    result = verify(shop, _committed(shop, drifted))
    shown = result["counterexamples"]
    assert any(one.get("test", "").endswith("test_a_basket_adds_up") and one["function"] == "total"
               and one["as"] == "digests" and one["before"] != one["after"] and one["counted"] == "lost"
               for one in shown), "the watch keeps a test's values as digests, and the record says so"
    assert any(one["function"].endswith("::total") and one["input"] == {"args": {"items": "['tea', 'oats']",
                                                                                "table": "{'tea': 3, 'oats': 2}"}}
               and one["before"] == "5" and one["after"] == "6" for one in shown), "the same call, readable"


# ── every line it wrote runs in some test ───────────────────────────────


def test_the_lines_a_change_wrote_are_known_by_whether_a_test_ran_them(shop):  # noqa: F811 - the fixture
    """A mend with a branch no test reaches: its line is named, the mend's own lines are not."""
    mend = _PRICES.replace("    return table[item]",
                           "    if item == 'never asked':\n        return 'stuffed in'\n    return table.get(item)")
    result = verify(shop, _committed(shop, mend))
    unrun = result["values_kept"]["lines_no_test_ran"]
    assert unrun == {"shop/prices.py": [3]}, result
    assert any("no test ran these lines it wrote in shop/prices.py: 3" in line for line in result["doubts"]), result


def test_a_mend_whose_every_line_runs_names_none(shop):  # noqa: F811 - the fixture
    result = verify(shop, _committed(shop, _PRICES.replace("return table[item]", "return table.get(item)")))
    assert result["values_kept"]["lines_no_test_ran"] == {} and result["held"], result


def test_a_comment_or_a_blank_line_asks_nothing_of_a_test(shop):  # noqa: F811 - the fixture
    from outside_bar.verify_change import lines_it_wrote_that_run

    commit = _committed(shop, _PRICES.replace("    return table[item]",
                                              "    # what it costs\n\n    return table.get(item)"))
    assert lines_it_wrote_that_run(shop, commit, "shop/prices.py") == {4}


def test_the_verdict_is_written_as_data_for_a_gate_to_decide_by(shop, tmp_path):  # noqa: F811 - the fixture
    """Her yes read a doubt off the words "on probation:" in the last lines printed; it reads this file now."""
    import json

    from outside_bar.verify_change import main

    overfit = _PRICES.replace("    return table[item]", "    if len(table) <= 1:\n        return None\n    return table[item]")
    commit = _committed(shop, overfit)
    written = tmp_path / "verdict.json"
    assert main(["--repo", str(shop), "--commit", commit, "--result", str(written)]) == 1
    said = json.loads(written.read_text())
    assert said["verdict"] == "REFUSED" and said["counterexamples"], said


def test_a_commit_that_cannot_be_checked_still_writes_its_verdict(shop, tmp_path):  # noqa: F811 - the fixture
    import json

    from outside_bar.verify_change import main

    written = tmp_path / "verdict.json"
    assert main(["--repo", str(shop), "--commit", "0" * 40, "--result", str(written)]) != 0
    assert json.loads(written.read_text())["verdict"] in ("REFUSED", "")
