"""Ways a program in Python could be put right, written by her own modules and searched by running its cases with each."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.making import searching_python
from core.making.searching import could_be_believed, is_believed
from core.making.tools import Case, languages, lines_each_ran, try_tool

pytest.importorskip("tree_sitter_python")


@pytest.fixture(autouse=True)
def written(monkeypatch):
    from core.making import tools
    from core.rebuilding import her_build_writes

    def governed(what, path, *args, **_how):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True) if what == "ensure_directory" else path.write_text(args[0], encoding="utf-8")

    for module in (her_build_writes, tools, searching_python):
        monkeypatch.setattr(module, "governed", governed)


TILLS = '''import sys


def price(line):
    name, count, each = line.split(",")
    return int(count) * int(each)


def off(total, count):
    if count >= 3 and total > 1000:
        return total * 10 // 100
    return 0


def main():
    lines = [line for line in sys.stdin.read().splitlines() if line.strip()]
    if not lines:
        print("nothing to add up", file=sys.stderr)
        return 2
    total = 0
    for line in lines:
        total += price(line)
    total -= off(total, len(lines))
    print(f"{len(lines)} lines")
    print(f"total {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''
CASES = [
    Case(rule="It adds up what each line comes to", typed="tea,2,150\n", prints=["total 300", "1 lines"], leaves_with=0),
    Case(rule="It adds up what each line comes to", typed="tea,2,150\noats,1,400\n", prints=["total 700", "2 lines"], leaves_with=0),
    Case(rule="With 3 lines or more and over 1000, 10 in 100 is taken off", typed="a,1,500\nb,1,500\nc,1,500\n", prints=["total 1350"]),
    Case(rule="With 3 lines or more and over 1000, 10 in 100 is taken off", typed="a,2,400\nb,1,500\nc,1,100\nd,1,600\n", prints=["total 1800"]),
    Case(rule="Nothing is taken off a small one", typed="a,1,300\nb,1,300\nc,1,300\n", prints=["total 900"]),
    Case(rule="Nothing is taken off a short one", typed="a,1,900\nb,1,900\n", prints=["total 1800"]),
    Case(rule="Given nothing it says so and leaves with 2", typed="\n\n", prints=["nothing to add up"], leaves_with=2),
    Case(rule="A line of nothing but spaces is passed over", typed="tea,1,5\n   \n", prints=["total 5", "1 lines"], leaves_with=0),
]

MISTAKES = {
    "a bound one off": ("count >= 3", "count > 3"),
    "a wrong number": ("total * 10 // 100", "total * 5 // 100"),
    "added where taken": ("total -= off", "total += off"),
    "one name for another": ("return int(count) * int(each)", "return int(count) * int(count)"),
    "half a condition": ("count >= 3 and total > 1000", "count >= 3"),
    "a guard left out": ("if line.strip()]", "]"),
    "a condition turned": ("if not lines:", "if lines:"),
    "a wrong exit": ("return 2", "return 1"),
    "an or for an and": ("count >= 3 and total > 1000", "count >= 3 or total > 1000"),
    "words where a number is": ("int(count) * int(each)", "int(count) * each"),
}


def _searched(code: str):
    language = languages()["python"]

    async def run():
        trial = await try_tool(code, language, CASES)
        lines = await lines_each_ran(code, language, CASES)
        found = await searching_python.search_python(code, CASES, trial.rules, lines, seconds=40.0)
        best = None
        for mend in found.mends[:8]:
            again = await try_tool(searching_python.made_python(code, mend.way), language, CASES)
            if all(r.ok for r in again.rules):
                best = mend
                break
        return trial, found, best

    return asyncio.run(run())


@pytest.mark.parametrize("what", list(MISTAKES))
def test_a_mistake_in_a_tool_is_found_and_put_right_with_no_model(what):
    was, instead = MISTAKES[what]
    assert TILLS.count(was) == 1, what
    trial, found, best = _searched(TILLS.replace(was, instead))
    assert not all(r.ok for r in trial.rules), "the mistake shows"
    assert best is not None, f"{found.says()}; {[(m.way.sort, m.way.says, m.mends, m.breaks) for m in found.mends[:6]]}"
    assert could_be_believed(best)


def test_a_program_with_nothing_wrong_is_found_nothing_to_change():
    trial, found, best = _searched(TILLS)
    assert all(r.ok for r in trial.rules) and found.mends == [] and best is None


def test_a_rule_tested_twice_is_believed_and_once_is_not():
    _trial, found, best = _searched(TILLS.replace("count >= 3 and total > 1000", "count >= 3"))
    assert "With 3 lines or more and over 1000, 10 in 100 is taken off" in found.twice and "Given nothing it says so and leaves with 2" not in found.twice
    assert best is not None and is_believed(best, found) == (len(best.mends) > 1 or set(best.mends) <= found.twice)


def test_a_way_that_never_ends_is_stopped_and_the_rest_are_tried(monkeypatch):
    monkeypatch.setattr(searching_python, "A_RUN_S", 0.2)
    code = TILLS.replace("    for line in lines:\n        total += price(line)\n", "    at = 0\n    while at < len(lines):\n        total += price(lines[at])\n        at += 1\n    total += 1\n")
    _trial, found, best = _searched(code)
    assert found.tried > 50 and best is not None


def test_what_differs_between_two_texts_is_one_change():
    for a, b in (("abc", "axc"), ("abc", "abc"), ("ab", "aab"), ("aab", "ab"), ("", "x")):
        start, end, text = searching_python._where(a, b)
        assert a[:start] + text + a[end:] == b


def test_what_a_branch_could_be_decided_by_is_what_is_in_reach_there():
    import ast

    code = "def off(total, count):\n    if count:\n        return 1\n    return 0\n"
    (test,) = searching_python._branches(ast.parse(code), {2})
    found = searching_python.conditions_python(code, [test], searching_python.Evidence(numbers=("3",)))
    texts = {w.text for w in found}
    assert "(count) and total > 0" in texts and "count >= 3" in texts and all(code[w.start:w.end] == "count" for w in found)
