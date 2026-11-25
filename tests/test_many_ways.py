"""The ways a program in JavaScript could be changed, searched by running its tests with each, with no model asked."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.making import many_ways, searching
from core.making.code_parts import assembled, each_test_in, patched
from core.making.design import Design, Rule
from core.making.trying import page_of, try_page
from tests.test_making_programs import _in_a_browser

pytest.importorskip("tree_sitter_javascript")

PANTRY = '''({
  initial() { return { items: [], name: "", count: 1, only: false, taken: 0 }; },
  low(s) { return s.items.filter((x) => x.count <= x.least); },
  total(s) { return s.items.reduce((n, x) => n + x.count, 0); },
  update(s, e) {
    if (e.type === "input") { s[e.name] = e.value; return s; }
    if (e.type === "do") {
      if (e.name === "add") {
        if (s.name.trim()) { s.items.push({ id: s.items.length + 1, name: s.name.trim(), count: Number(s.count), least: 2 }); s.name = ""; s.count = 1; }
      } else if (e.name === "take") {
        const it = s.items.find((x) => x.id === e.value);
        if (it && it.count > 0) { it.count -= 1; s.taken += 1; }
      } else if (e.name === "only") { s.only = !s.only; }
    }
    s.items.sort((a, b) => a.name.localeCompare(b.name));
    return s;
  },
  view(s, ui) {
    const list = s.only ? this.low(s) : s.items;
    return ui.column(ui.title("Pantry"), ui.input("Name", "name", s.name), ui.number("Count", "count", s.count), ui.button("Add it", "add"),
      ui.button("Only what is low", "only"), ui.text("In all " + this.total(s)), ui.text("Taken " + s.taken),
      ui.list(list, (x) => ui.row(ui.text(x.name + " x" + x.count), ui.button("Take " + x.name, "take", x.id))));
  },
})'''
PANTRY_TESTS = '''test("What is added is in the list", t => { t.type("Name", "rice"); t.type("Count", 3); t.click("Add it"); t.expect(t.s.items.length === 1 && t.s.items[0].count === 3, "not there"); t.expect(t.sees("rice x3"), "not shown"); });
test("A thing with no name is not added", t => { t.type("Name", "  "); t.click("Add it"); t.expect(t.s.items.length === 0, "it was"); });
test("Taking one leaves one fewer and is counted", t => { t.type("Name", "rice"); t.type("Count", 3); t.click("Add it"); t.click("Take rice"); t.expect(t.s.items[0].count === 2, "count " + t.s.items[0].count); t.expect(t.s.taken === 1, "taken " + t.s.taken); });
test("None are taken where there are none", t => { t.s.items = [{ id: 1, name: "rice", count: 0, least: 2 }]; t.click("Take rice"); t.expect(t.s.items[0].count === 0 && t.s.taken === 0, "it went to " + t.s.items[0].count); });
test("What has 2 or fewer is low", t => { t.s.items = [{ id: 1, name: "rice", count: 2, least: 2 }, { id: 2, name: "tea", count: 3, least: 2 }]; t.click("Only what is low"); t.expect(t.sees("rice") && !t.sees("tea"), "it shows " + t.shown); });
test("They are in order of their names", t => { t.type("Name", "tea"); t.click("Add it"); t.type("Name", "oats"); t.click("Add it"); t.expect(t.comesBefore("oats", "tea"), "they are not"); });
test("How many there are in all is shown", t => { t.s.items = [{ id: 1, name: "rice", count: 2, least: 2 }, { id: 2, name: "tea", count: 3, least: 2 }]; t.expect(t.sees("In all 5"), "it shows " + t.shown); });
test("The name is empty once it is added", t => { t.type("Name", "tea"); t.click("Add it"); t.expect(t.s.name === "", "it is " + t.s.name); });'''
PANTRY_STATE = {"items": [{"id": 1, "name": "rice", "count": 3, "least": 2}], "name": "", "count": 1, "only": False, "taken": 0}

#: A mistake, as a program might be written with it: what is there, and what is there instead.
MISTAKES = {
    "a bound one off": ("x.count <= x.least", "x.count < x.least"),
    "a count nothing adds to": (" s.taken += 1;", ""),
    "a name of another event": ('e.name === "take"', 'e.name === "use"'),
    "one field for another": ("n + x.count", "n + x.least"),
    "a wrong number": ("x.count <= x.least", "x.count <= 1"),
    "the order turned": ("a.name.localeCompare(b.name)", "b.name.localeCompare(a.name)"),
    "a condition turned": ("if (s.name.trim())", "if (!s.name.trim())"),
    "a guard left out": ("it && it.count > 0", "it"),
    "too many taken": ("it.count -= 1", "it.count -= 2"),
    "added where taken": ("it.count -= 1", "it.count += 1"),
    "words where a number is": ("x.id === e.value", "x.id === String(e.value)"),
    "a function asked for without this": ("this.low(s)", "low(s)"),
    "nothing given back": ("(n, x) => n + x.count", "(n, x) => { n + x.count; }"),
    "put to nothing and not the other": ("s.only = !s.only", "s.only = s.only"),
    "a name never emptied": (' s.name = "";', ""),
}


def _design(**over) -> Design:
    return Design(**{"name": "Pantry", "kind": "app", "about": "What is in it.", "rules": [Rule(says="x")], "state": PANTRY_STATE, **over})


def _searched(code: str, tests: str, tmp_path: Path, design: Design, **over):
    async def run(browser):
        trial = await try_page(page_of(design, code, tests=tests), tmp_path / "t.html", browser)
        found = await searching.search(design, code, tests, trial.failing(), browser, tmp_path / "s.html", functions=list(patched({}, code)[0]), **over)
        best = None
        for mend in found.mends[:12]:
            again = await try_page(page_of(design, many_ways.made(code, mend.way), tests=tests), tmp_path / "t.html", browser)
            if not again.failing():
                best = mend.way
                break
        return trial, found, best

    return _in_a_browser(run)


@pytest.mark.parametrize("what", list(MISTAKES))
def test_a_mistake_in_a_program_is_found_and_put_right_with_no_model(what, tmp_path):
    was, instead = MISTAKES[what]
    assert PANTRY.count(was) == 1, what
    code = assembled(patched({}, PANTRY.replace(was, instead))[0])
    trial, found, best = _searched(code, PANTRY_TESTS, tmp_path, _design(), seconds=60.0)
    assert trial.failing(), "the mistake shows"
    assert best is not None, f"{found.says()}; {[(m.way.sort, m.way.says, m.mends, m.breaks) for m in found.mends[:6]]}"


def test_what_is_changed_is_what_the_failing_tests_run_and_the_passing_ones_do_not(tmp_path):
    code = assembled(patched({}, PANTRY.replace("s.taken += 1;", "s.taken += 0;"))[0])
    _trial, found, best = _searched(code, PANTRY_TESTS, tmp_path, _design(), seconds=60.0)
    assert best is not None and "taken" in many_ways.made(code, best)
    read = many_ways.Read.of(code, PANTRY_STATE, ["low", "total"])
    assert any(way.block == found.mends[0].way.block for way in many_ways.ways(read, many_ways.Evidence()))


def test_a_way_that_never_ends_is_stopped_and_the_rest_are_tried(tmp_path):
    code = assembled(patched({}, PANTRY.replace("return s.items.reduce((n, x) => n + x.count, 0);",
                                                "let n = 0; for (let i = 0; i < s.items.length; i++) { n += s.items[i].least; } return n;"))[0])
    _trial, found, best = _searched(code, PANTRY_TESTS, tmp_path, _design(), seconds=60.0)
    assert found.tried > 100 and best is not None and "s.items[i].count" in many_ways.made(code, best)
    began_at_one = [m for m in found.mends if "let n = 1" in many_ways.made(code, m.way)]
    assert began_at_one and began_at_one[0].other == began_at_one[0].held, "it fits its test, and changes what came of every other"


def test_a_program_that_cannot_be_read_or_has_nothing_failing_is_left(tmp_path):
    design = _design()

    async def run(browser):
        whole = await searching.search(design, PANTRY, PANTRY_TESTS, [], browser, tmp_path / "s.html")
        broken = await searching.search(design, "({ initial( { })", PANTRY_TESTS, [], browser, tmp_path / "s.html")
        return whole, broken

    whole, broken = _in_a_browser(run)
    assert whole.mends == [] and broken.mends == [] and broken.read is None


def test_the_program_with_every_way_in_it_does_what_it_did(tmp_path):
    """None let through, it is the program: every test that held holds."""
    code = assembled(patched({}, PANTRY)[0])
    read = many_ways.Read.of(code, PANTRY_STATE, list(patched({}, PANTRY)[0]))
    found = many_ways.ways(read, many_ways.evidence_of([(name, text, "") for name, text in each_test_in(PANTRY_TESTS)[1]]))
    assert len(found) > 800 and len({way.id for way in found}) == len(found)
    whole = many_ways.with_all(read, found)

    async def run(browser):
        return await searching._in_page(_design(), whole, PANTRY_TESTS, len(read.blocks), browser, tmp_path / "s.html",
                                        [(searching._COVER, [name for name, _text in each_test_in(PANTRY_TESTS)[1]])])

    (rows,) = _in_a_browser(run)
    assert [row["ok"] for row in rows] == [True] * 8 and all(row["hit"] for row in rows)


def test_what_a_failing_test_says_is_what_a_change_is_made_from():
    evidence = many_ways.evidence_of([("It is late after three days", 'test("It is late after three days", t => { t.s.day = 14; t.expect(t.sees("late")); })',
                                       "Cannot read properties of undefined (reading 'due') (in update)")])
    assert evidence.numbers == ("3",) and evidence.words == (), "what is expected of it is not for the program to say back"
    assert "day" in evidence.names and evidence.broke
    read = many_ways.Read.of("({ update(s, e) { if (s.day > 5) { s.late = s.plant.due; } return s; } })", {"day": 1, "late": False, "plant": {"due": 2}}, ["update"])
    found = many_ways.ways(read, evidence)
    made = {many_ways.made(read.code.decode(), way) for way in found}
    assert any("s.day > 3" in code for code in made) and any("(s.plant || {}).due" in code for code in made)
    assert {way.sort for way in found if re.search(r"\|\| \{\}", way.text)} == {"guard"}


def test_nothing_that_is_given_to_called_or_named_is_put_something_else_in_place_of():
    """A change there would not be read, or would lose what a function is of."""
    code = "({ update(s, { name, value = 2 }) { s.list[name] = value; s.n++; delete s.old; for (const k of s.keys) { this.go(k); } try { s.go(); } catch (err) { s.n = 0; } return s; } })"
    read = many_ways.Read.of(code, {"list": {}, "n": 1, "keys": ["a"], "old": 1}, ["update", "go"])
    found = many_ways.ways(read, many_ways.Evidence(numbers=("4",), names=frozenset({"n"})))
    whole = many_ways.with_all(read, found)
    assert whole and not many_ways._parser().parse(whole.encode()).root_node.has_error
    for way in found:
        assert not many_ways._parser().parse(many_ways.made(code, way).encode()).root_node.has_error, way


def test_every_way_is_read_as_it_is_written_into_the_program_by_itself():
    """LIVE 24 November 2026: the brackets of an if were taken for part of what was in them, and the mend could not be read."""
    code = assembled(patched({}, PANTRY)[0])
    read = many_ways.Read.of(code, PANTRY_STATE, list(patched({}, PANTRY)[0]))
    evidence = many_ways.evidence_of([(name, text, "") for name, text in each_test_in(PANTRY_TESTS)[1]])
    found = many_ways.ways(read, evidence)
    spots = [(w.start, w.end) for w in many_ways.forced(read, dict.fromkeys(read.blocks.values(), 1.0), 0)]
    parser = many_ways._parser()
    for way in [*found[::7], *many_ways.conditions(read, spots, evidence)[::9]]:
        assert not parser.parse(many_ways.made(code, way).encode()).root_node.has_error, way
