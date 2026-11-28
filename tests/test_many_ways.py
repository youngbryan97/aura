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


# ── a list kept in a form ────────────────────────────────────────────

SHARED = '''({
  initial() { return { people: ["Ana", "Ben", "Cy"], paid: [], form: { by: "", amount: 0, shares: [] } }; },
  update(s, e) {
    if (e.type === "input") { return { ...s, form: { ...s.form, [e.name]: e.value } }; }
    if (e.type === "do") {
      if (e.name === "row") return { ...s, form: { ...s.form, shares: [...s.form.shares, { who: "", part: 0 }] } };
      if (e.name === "save") {
        const f = s.form, shares = f.shares.filter((x) => x.who);
        if (!f.by || !(f.amount > 0) || !shares.length) return s;
        return { ...s, paid: [...s.paid, { id: s.paid.length + 1, by: f.by, amount: f.amount, shares }], form: { by: "", amount: 0, shares: [] } };
      }
    }
    return s;
  },
  view(s, ui) {
    const f = s.form, people = [["", "Choose"], ...s.people.map((p) => [p, p])];
    const rows = f.shares.map((x, i) => ui.row(ui.select("Sharer", `share_who_${i}`, x.who, people), ui.number("Part", `share_part_${i}`, x.part)));
    return ui.column(ui.title("Shared"), ui.select("Paid by", "by", f.by, people), ui.number("Amount", "amount", f.amount), rows, ui.button("Add a sharer", "row"), ui.button("Save", "save"),
      ui.list(s.paid, (p) => ui.text(p.by + " paid " + p.amount + " for " + p.shares.map((x) => x.who + " " + x.part).join(", "))));
  },
})'''
SHARED_TESTS = '''test("What is paid is kept with who it is for", t => { t.choose("Paid by", "Ana"); t.type("Amount", 30); t.choose("Shares", "Ben"); t.click("Save"); t.expect(t.s.paid.length === 1, "it was not kept"); t.expect(t.s.paid[0].shares[0].who === "Ben", "for nobody"); });
test("It can be for several", t => { t.choose("Paid by", "Ben"); t.type("Amount", 30); t.choose("Shares", ["Ana", "Cy"]); t.click("Save"); t.expect(t.sees("Ben paid 30 for Ana 0, Cy 0"), "it shows " + t.shown); });
test("What each part is is kept", t => { t.choose("Paid by", "Ben"); t.type("Amount", 30); t.choose("Shares", "Ana"); t.type("Part", 12); t.click("Save"); t.expect(t.sees("Ana 12"), "it shows " + t.shown); });
test("What is for nobody is not kept", t => { t.choose("Paid by", "Ana"); t.type("Amount", 30); t.click("Save"); t.expect(t.s.paid.length === 0, "it was"); });'''
SHARED_STATE = {"people": ["Ana", "Ben", "Cy"], "paid": [{"id": 1, "by": "Ana", "amount": 30, "shares": [{"who": "Ben", "part": 30}]}],
                "form": {"by": "", "amount": 0, "shares": [{"who": "Ben", "part": 30}]}}


def test_what_is_put_in_the_row_of_a_list_goes_in_that_row(tmp_path):
    """LIVE 25 November 2026: the beneficiaries of an expense were kept by the names of their rows, which nothing read, and an hour
    of her model's mends took twenty two of the twenty three that failed for sixteen different faults."""
    code = assembled(patched({}, SHARED)[0])
    trial, found, best = _searched(code, SHARED_TESTS, tmp_path, _design(name="Shared", state=SHARED_STATE), seconds=60.0)
    shown = next(r for r in trial.failing() if r.name == "what is put in a control is what it shows")
    assert "“Sharer”" in shown.why and "share_who_0" in shown.why, shown.why
    assert best is not None and best.sort == "row", f"{[(m.way.sort, m.mends) for m in found.mends[:4]]}"
    mended = many_ways.made(code, best)
    assert "s.form.shares.map(" in mended and "who: e.value" in mended and "part: e.value" in mended


def test_a_label_that_no_control_has_is_what_one_of_them_sets(tmp_path):
    """The design has a control called "Shares" and the program has a row for each of them: what a test puts in "Shares" is put there."""
    code = assembled(patched({}, SHARED.replace('if (e.type === "input") {', 'if (e.type === "input") { const at = /^share_(who|part)_(\\d+)$/.exec(e.name); if (at) return { ...s, form: { ...s.form, shares: s.form.shares.map((x, i) => (i === Number(at[2]) ? { ...x, [at[1]]: e.value } : x)) } };'))[0])

    async def run(browser):
        return await try_page(page_of(_design(name="Shared", state=SHARED_STATE), code, tests=SHARED_TESTS + '\ntest("No such thing", t => { t.choose("Colour", "Ben"); });\ntest("Nobody by that name", t => { t.click("Add a sharer"); t.choose("Shares", "Zed"); });'), tmp_path / "t.html", browser)

    trial = _in_a_browser(run)
    by = {r.name: r for r in trial.rules if not r.ok}
    assert set(by) == {"No such thing", "Nobody by that name"}, [(r.name, r.why[:120]) for r in trial.failing()]
    assert "there is no “Colour” to use" in by["No such thing"].why
    assert by["Nobody by that name"].the_test and "takes one of Choose, Ana, Ben, Cy" in by["Nobody by that name"].why


def test_a_mend_is_believed_for_as_many_things_as_it_leaves_otherwise():
    way = many_ways.Way(1, 1, "row", 0, 0, "x", 1)
    mend = lambda rules, other: searching.Mend(way, tuple(rules), (), other, 8)  # noqa: E731
    assert searching.could_be_believed(mend(list("abcdefgh"), 6)) and not searching.could_be_believed(mend(["a", "b"], 6))


def test_a_test_that_gets_further_is_nearer(tmp_path):
    from core.making.maker import _how_much_is_wrong
    from core.making.trying import Result

    near, far = Result("a", False, "not shown", got=6), Result("a", False, "there is no Save", got=1)
    assert _how_much_is_wrong([near]) < _how_much_is_wrong([far]) < _how_much_is_wrong([Result("a", False, "x is not a function", broke=True)])
    code = assembled(patched({}, SHARED)[0])

    async def run(browser):
        return await try_page(page_of(_design(name="Shared", state=SHARED_STATE), code, tests=SHARED_TESTS), tmp_path / "t.html", browser)

    trial = _in_a_browser(run)
    assert {r.name: r.got for r in trial.rules}["What is paid is kept with who it is for"] >= 3


def test_what_a_test_made_for_itself_breaking_is_the_tests_own(tmp_path):
    """LIVE 25 November 2026: "r.int is not a function", of a test's own r, was mended in the program."""
    from core.making.maker import Another, another_way

    code = assembled(patched({}, PANTRY)[0])
    check = Another(claims="The total is the sum of the counts", function="total", inputs="r => [{ items: Array.from({ length: r.int(0, 4) }, () => ({ count: r.int(0, 9), least: r.pick([1, 2]) })) }]",
                    surely="s => s.items.reduce((n, x) => n + x.count, 0)", holds="(got, sure) => got === sure")
    broken = 'test("Its own", t => { const q = {}; q.none(); });'

    async def run(browser):
        return await try_page(page_of(_design(), code, tests=f"{PANTRY_TESTS}\n{another_way(check)}\n{broken}"), tmp_path / "t.html", browser)

    trial = _in_a_browser(run)
    by = {r.name: r for r in trial.rules}
    assert next(r for name, r in by.items() if name.startswith("The total is the sum")).ok
    assert not by["Its own"].ok and by["Its own"].the_test


# ── what a test put in, and what the program shows ───────────────────

DIARY = '''({
  initial() { return { kept: [], form: { what: "", least: 0 }, at: null, said: "" }; },
  update(s, e) {
    if (e.type === "input") return { ...s, form: { ...s.form, [e.name]: e.value } };
    if (e.type === "do") {
      if (e.name === "save") {
        if (!(s.form.least > 0)) return { ...s, said: "Not a number of minutes" };
        const one = { what: s.form.what, least: s.form.least };
        const kept = s.at === null ? [...s.kept, one] : s.kept.map((x, i) => (i === s.at ? one : x));
        return { ...s, kept, at: null, said: "", form: { what: "", least: 0 } };
      }
      if (e.name === "edit") return { ...s, at: e.value, form: { ...s.kept[e.value] } };
    }
    return s;
  },
  view(s, ui) {
    return ui.column(ui.title("Diary"), ui.card(ui.title("Add a line"), ui.input("What", "what", s.form.what), ui.number("Minutes", "least", s.form.least), ui.button("Save", "save")), ui.text(s.said),
      ui.list(s.kept, (x, i) => ui.row(ui.text(x.what + " " + x.least), ui.button("Edit", "edit", i))));
  },
})'''
DIARY_STATE = {"kept": [{"what": "Walk", "least": 20}], "form": {"what": "", "least": 0}, "at": None, "said": ""}
DIARY_TESTS = '''test("What is not a number of minutes is not kept", t => { t.type("What", "Swim far"); t.type("Minutes", -3); t.click("Save"); t.expect(t.sees("Not a number"), "nothing is said"); t.expect(!t.sees("Swim far"), "it was kept"); });
test("What is changed is shown as it is now", t => { t.type("What", "Walk"); t.type("Minutes", 20); t.click("Save"); t.click("Edit"); t.type("What", "A long walk"); t.click("Save"); t.expect(t.sees("A long walk 20"), "it is not changed"); t.expect(!t.sees("Walk"), "the old one is still there"); });
test("What is put in a field by the program is seen there", t => { t.type("What", "Read"); t.type("Minutes", 5); t.click("Save"); t.click("Edit"); t.expect(!t.sees("Read"), "the field is filled"); });'''


def test_what_a_test_put_in_itself_is_not_the_program_showing_it(tmp_path):
    """LIVE 25 November 2026: "Invalid expense should not be saved" failed for what the test had typed, still in its field; and "Old
    description should be replaced" for "Groceries" in "Supermarket Groceries"."""

    async def run(browser):
        return await try_page(page_of(_design(name="Diary", state=DIARY_STATE), DIARY, tests=DIARY_TESTS), tmp_path / "t.html", browser)

    by = {r.name: r for r in _in_a_browser(run).rules}
    assert by["What is not a number of minutes is not kept"].ok, by["What is not a number of minutes is not kept"].why
    assert by["What is changed is shown as it is now"].ok, by["What is changed is shown as it is now"].why
    assert not by["What is put in a field by the program is seen there"].ok, "what the program put there is shown by it"


def test_a_program_that_does_keep_what_it_should_not_is_still_caught(tmp_path):
    for wrong in (DIARY.replace('if (!(s.form.least > 0)) return { ...s, said: "Not a number of minutes" };', 'const said = s.form.least > 0 ? "" : "Not a number of minutes";').replace('at: null, said: "", form', "at: null, said, form"),
                  DIARY.replace("const kept = s.at === null ? [...s.kept, one] : s.kept.map((x, i) => (i === s.at ? one : x));", "const kept = [...s.kept, one];")):
        assert wrong != DIARY

        async def run(browser, wrong=wrong):
            return await try_page(page_of(_design(name="Diary", state=DIARY_STATE), wrong, tests=DIARY_TESTS), tmp_path / "t.html", browser)

        assert any(not r.ok for r in _in_a_browser(run).rules[:2])


def test_several_controls_of_one_name_are_taken_as_somebody_would_take_them(tmp_path):
    """LIVE 25 November 2026 three people put in the rows of a list, a row at a time, were each put over the one before."""
    code = assembled(patched({}, SHARED.replace('if (e.type === "input") {', 'if (e.type === "input") { const at = /^share_(who|part)_(\\d+)$/.exec(e.name); if (at) return { ...s, form: { ...s.form, shares: s.form.shares.map((x, i) => (i === Number(at[2]) ? { ...x, [at[1]]: e.value } : x)) } };'))[0])
    tests = '''test("Each in a row of its own", t => { t.choose("Paid by", "Ana"); t.type("Amount", 30); for (const [who, part] of [["Ana", 5], ["Ben", 10], ["Cy", 15]]) { t.click("Add a sharer"); t.choose("Sharer", who); t.type("Part", part); } t.click("Save"); t.expect(t.sees("Ana paid 30 for Ana 5, Ben 10, Cy 15"), "it shows " + t.shown); });
test("And again after they were let go", t => { for (const who of ["Ana", "Ben"]) { t.choose("Paid by", "Cy"); t.type("Amount", 12); t.click("Add a sharer"); t.choose("Sharer", who); t.type("Part", 12); t.click("Save"); } t.expect(t.sees("for Ana 12") && t.sees("for Ben 12"), "it shows " + t.shown); });'''

    async def run(browser):
        return await try_page(page_of(_design(name="Shared", state=SHARED_STATE), code, tests=tests), tmp_path / "t.html", browser)

    assert [(r.name, r.why[:160]) for r in _in_a_browser(run).rules if not r.ok] == []


def test_a_part_of_the_design_is_there_by_its_title(tmp_path):
    async def run(browser, controls):
        return await try_page(page_of(_design(name="Diary", state=DIARY_STATE, controls=controls), DIARY, tests=""), tmp_path / "t.html", browser)

    def said(*names):
        trial = _in_a_browser(lambda browser: run(browser, [{"input": name, "does": "x"} for name in names]))
        return next(r for r in trial.generic if r.name == "the controls it was designed with are there")

    assert said("What", "Save", "Add a line").ok, "a title over what it is worked by"
    missing = said("Save", "Send to somebody", "Diary of it")
    assert not missing.ok and "“Send to somebody”" in missing.why and "“Diary of it”" in missing.why


def test_what_is_named_more_fully_than_it_is_called_is_the_one_that_has_just_come(tmp_path):
    """LIVE 25 November 2026 a test confirmed a deletion by "confirm delete", and what asked was called Confirm."""
    asking = DIARY.replace('if (e.name === "edit")', 'if (e.name === "drop") return { ...s, at: e.value, said: "Sure?" }; if (e.name === "yes") return { ...s, kept: s.kept.filter((x, i) => i !== s.at), at: null, said: "" }; if (e.name === "edit")').replace(
        'ui.button("Edit", "edit", i)', 'ui.button("Edit", "edit", i), ui.button("Delete", "drop", i)').replace("ui.text(s.said),", 'ui.text(s.said), s.said === "Sure?" ? ui.row(ui.button("Confirm", "yes"), ui.button("Cancel", "no")) : "",')
    tests = '''test("What is deleted is gone", t => { t.type("What", "Walk"); t.type("Minutes", 20); t.click("Save"); t.click("Delete"); t.click("confirm delete"); t.expect(t.s.kept.length === 0, "it is still there"); });
test("What is nowhere is nowhere", t => { t.click("send it away"); });'''

    async def run(browser):
        return await try_page(page_of(_design(name="Diary", state=DIARY_STATE), asking, tests=tests), tmp_path / "t.html", browser)

    by = {r.name: r for r in _in_a_browser(run).rules}
    assert by["What is deleted is gone"].ok, by["What is deleted is gone"].why
    assert not by["What is nowhere is nowhere"].ok


def test_what_is_there_only_once_something_is_kept_is_come_to_by_the_tests(tmp_path):
    """LIVE 25 November 2026 a mend that put in the control that was missing was not kept, for a random walk had kept nothing that
    time and two others were not come upon."""
    code = DIARY.replace("if (!(s.form.least > 0))", 'if (s.form.what !== "Walk to the sea and back" || !(s.form.least > 0))')
    assert code != DIARY
    controls = [{"input": name, "does": "x"} for name in ("What", "Save", "Edit")]
    keeps = 'test("It is kept", t => { t.type("What", "Walk to the sea and back"); t.type("Minutes", 4); t.click("Save"); t.expect(t.s.kept.length === 1, "it is not"); });'

    def said(tests, state=None):
        async def run(browser):
            return await try_page(page_of(_design(name="Diary", state=state or {"kept": []}, controls=controls), code, tests=tests), tmp_path / "t.html", browser)

        return next(r for r in _in_a_browser(run).generic if r.name == "the controls it was designed with are there")

    assert said(keeps).ok, said(keeps).why
    assert not said("").ok, "with nothing that keeps anything there is nothing to come to it by"


def test_what_is_not_offered_is_put_in_by_nobody(tmp_path):
    """LIVE 25 November 2026 a test put a name nobody had in a list of names, and the program was blamed for taking it."""
    code = assembled(patched({}, SHARED)[0])
    tests = '''test("Nobody by that name", t => { t.choose("Paid by", "Zed"); t.type("Amount", 3); t.click("Save"); t.expect(t.s.paid.length === 0, "it was kept"); });
test("Somebody it knows of", t => { t.s = { ...t.s, former: "Dee" }; t.choose("Paid by", "Dee"); });
test("Somebody it has", t => { t.choose("Paid by", "ana"); t.expect(t.s.form.by === "Ana", "it is " + t.s.form.by); });'''

    async def run(browser):
        return await try_page(page_of(_design(name="Shared", state={**SHARED_STATE, "former": ""}), code, tests=tests), tmp_path / "t.html", browser)

    by = {r.name: r for r in _in_a_browser(run).rules}
    assert by["Nobody by that name"].the_test and "nobody could put “Zed” in it" in by["Nobody by that name"].why
    assert not by["Somebody it knows of"].ok and not by["Somebody it knows of"].the_test, "what it knows of and does not offer is the program's"
    assert by["Somebody it has"].ok, by["Somebody it has"].why


def test_what_puts_something_in_is_not_drawn_as_taking_something_away_and_is_drawn_right_with_no_model(tmp_path):
    """LIVE 25 November 2026 the buttons that added a flatmate, a plant and an expense were drawn in red."""
    marked = DIARY.replace('ui.button("Save", "save")', 'ui.button("Save", "save", null, { tone: "danger" })').replace(
        'ui.button("Edit", "edit", i)', 'ui.button("Edit", "edit", i), ui.button("Forget", "drop", i, { tone: "danger" })').replace(
        'if (e.name === "edit")', 'if (e.name === "drop") return { ...s, kept: s.kept.filter((x, i) => i !== e.value) }; if (e.name === "edit")')
    assert marked.count("danger") == 2
    code = assembled(patched({}, marked)[0])
    state = {**DIARY_STATE, "kept": [{"what": "Walk", "least": 20}]}
    trial, found, best = _searched(code, DIARY_TESTS.splitlines()[0], tmp_path, _design(name="Diary", state=state), seconds=40.0)
    shown = next(r for r in trial.failing() if r.name == "what is marked as taking something away does")
    assert "“Save”" in shown.why and "Forget" not in shown.why
    assert best is not None and best.sort == "tone", f"{[(m.way.sort, m.mends) for m in found.mends[:4]]}"
    mended = many_ways.made(code, best)
    assert mended.count("danger") == 1 and 'ui.button("Forget", "drop", i, { tone: "danger" })' in mended


def test_what_is_worked_out_from_the_state_is_worked_out_where_it_is_read(tmp_path):
    """LIVE 25 November 2026 a shopping list went on holding a dish taken off it: it was kept, and set by the test it was not
    worked out again."""
    worked_out = DIARY.replace("  view(s, ui) {", "  derived(s) { return { minutes: s.kept.reduce((n, x) => n + x.least, 0), first: [...s.kept].sort((a, b) => a.what.localeCompare(b.what)).map((x) => x.what) }; },\n  view(s, ui) {").replace(
        "ui.text(s.said),", "ui.text(s.said), ui.note(`${s.minutes} minutes`), ui.text(s.first.join(\" then \")),")
    tests = '''test("What is set is what is worked out from", t => { t.s = { kept: [{ what: "Swim", least: 30 }, { what: "Bike", least: 15 }] }; t.expect(t.sees("45 minutes"), "it shows " + t.shown); t.expect(t.sees("Bike then Swim"), "not in order"); t.expect(t.s.minutes === 45, "t.s says " + t.s.minutes); });
test("And what is kept", t => { t.type("What", "Row"); t.type("Minutes", 10); t.click("Save"); t.expect(t.sees("10 minutes"), "it shows " + t.shown); });'''

    async def run(browser):
        return await try_page(page_of(_design(name="Diary", state={**DIARY_STATE, "minutes": 0}), worked_out, tests=tests), tmp_path / "t.html", browser)

    trial = _in_a_browser(run)
    assert [(r.name, r.why[:200]) for r in [*trial.rules, *trial.generic] if not r.ok and r.name in {"What is set is what is worked out from", "And what is kept", "its state is as it was designed", "it begins"}] == []


def test_a_button_that_does_nothing_says_what_it_sends_and_that_nothing_looks_at_it(tmp_path):
    """LIVE 25 November 2026 a form's Cancel sent "cancelForm", which update had no branch for, and the app was called done."""
    dead = DIARY.replace('ui.button("Save", "save")', 'ui.button("Save", "save"), ui.button("Cancel", "cancelForm")')

    async def run(browser):
        return await try_page(page_of(_design(name="Diary", state=DIARY_STATE), dead, tests=DIARY_TESTS), tmp_path / "t.html", browser)

    shown = next(r for r in _in_a_browser(run).generic if r.name == "every control does something")
    assert not shown.ok and "Cancel (sends “cancelForm”, which update never looks at)" in shown.why


def test_a_button_that_does_nothing_is_mended_before_rules_that_fail():
    from core.making.maker import _how_much_is_wrong
    from core.making.trying import Result

    assert _how_much_is_wrong([Result("every control does something", False, "these change nothing: Cancel")]) > _how_much_is_wrong([Result("A rule", False, "not shown", broke=True)])


DISHES = """({
  initial() { return { dishes: [], asking: false, said: "" }; },
  update(s, e) {
    if (e.name === "add") return { ...s, dishes: [...s.dishes, "Soup " + (s.dishes.length + 1)], said: "Soup " + (s.dishes.length + 1) + " added" };
    if (e.name === "clear") return { ...s, asking: true };
    if (e.name === "keep") return { ...s, asking: false };
    if (e.name === "sure") return { ...s, dishes: [], asking: false, said: "Cleared" };
    return s;
  },
  view(s, ui) {
    return ui.column(ui.title("Dishes"), ui.row(ui.button("Add", "add", null, { icon: "plus" }), ui.button("Clear", "clear", null, { tone: "danger", icon: "trash" })),
      s.dishes.length ? ui.list(s.dishes) : ui.empty("No dishes yet", "Add one to begin", "food"),
      s.asking ? ui.dialog("Clear every dish?", ui.text("They will be gone."), ui.row(ui.button("Keep them", "keep"), ui.button("Clear them", "sure", null, { tone: "danger" }))) : null,
      ui.toast(s.said, "good"));
  },
})"""


def test_what_an_app_is_shown_with_has_icons_an_empty_state_a_dialog_over_it_and_a_word_that_comes_again(tmp_path):
    """27 November 2026: Bryan asked that what she makes have the finish of something made with care, and the frame had no icons,
    nothing to say where there was nothing yet, no way to ask before something was done, and no word that something was."""

    async def run(browser):
        page = await browser.new_page()
        try:
            await page.set_content(page_of(_design(name="Dishes", state={"dishes": [], "asking": False, "said": ""}), DISHES))
            seen = {"icons": await page.locator("main button svg.ico path").count(),
                    "empty": await page.locator(".empty strong").inner_text(), "dialog": await page.locator("[role=dialog]").count()}
            await page.click("main button:has-text('Add')")
            await page.evaluate("document.querySelector('.toast').__first = true")
            await page.click("main button:has-text('Add')")
            seen["toast"] = await page.evaluate("[document.querySelector('.toast').textContent, !!document.querySelector('.toast').__first]")
            seen["listed"] = await page.locator("main .list li").count()
            await page.click("main button:has-text('Clear')")
            seen["asked"] = await page.locator(".scrim [role=dialog][aria-modal=true] h2").inner_text()
            seen["focus"] = await page.evaluate("document.activeElement.textContent")
            await page.keyboard.press("Escape")
            seen["escaped"] = [await page.locator("[role=dialog]").count(), await page.locator("main .list li").count()]
            await page.click("main button:has-text('Clear')")
            await page.click(".dialog button:has-text('Clear them')")
            seen["after"] = [await page.locator("[role=dialog]").count(), await page.locator(".empty").count(), await page.locator(".toast").inner_text()]
            return seen
        finally:
            await page.close()

    seen = _in_a_browser(run)
    assert seen["icons"] == 2 and seen["empty"] == "No dishes yet" and seen["dialog"] == 0, seen
    assert seen["toast"] == ["Soup 2 added", False], "a new word is shown as new, and comes in again"
    assert seen["listed"] == 2 and seen["asked"] == "Clear every dish?", seen
    assert seen["focus"] == "Keep them" and seen["escaped"] == [0, 2], "Escape leaves things as they were"
    assert seen["after"] == [0, 1, "Cleared"], seen


def test_a_badge_is_as_wide_as_what_it_says_first_in_a_row_of_a_list(tmp_path):
    """LIVE 28 November 2026 "Reading", first in a row of a book's card in a list, was as wide as half the card."""
    shelf = """({
  initial() { return { books: [{ title: "The Hobbit", read: 155, of: 310 }] }; },
  update(s, e) { return s; },
  view(s, ui) { return ui.column(ui.list(s.books, (b) => ui.row(ui.badge("Reading", "warm"), ui.text(`${b.read}/${b.of} pages`)))); },
})"""

    async def run(browser):
        page = await browser.new_page(viewport={"width": 1000, "height": 600})
        try:
            await page.set_content(page_of(_design(name="Shelf", state={"books": []}), shelf))
            return await page.evaluate("document.querySelector('.list .badge').getBoundingClientRect().width")
        finally:
            await page.close()

    assert _in_a_browser(run) < 120


def test_what_is_painted_in_an_app_is_painted_when_it_is_tried_and_the_pen_has_the_canvas_s_own_too(tmp_path):
    """LIVE 28 November 2026 a book's spine painted with g.fillRect broke the page, and every test of the app held: what an app
    paints was never painted while it was tried, and the pen had no fillRect."""
    spine = """({
  initial() { return { read: 155, of: 310 }; },
  update(s, e) { return s; },
  view(s, ui) { return ui.column(ui.text(`${s.read} of ${s.of} pages`), ui.canvas("spine", 60, 200, (g, w, h) => { g.fillStyle = "#b5523b"; g.fillRect(20, 0, 20, h); g.at(30, h * s.read / s.of, 0, (p) => p.fillRect(-14, -2, 28, 4)); })); },
})"""
    broken = spine.replace("g.fillRect(20, 0, 20, h)", "g.paintTheSpine(h)")

    def shown(code):
        async def run(browser):
            return await try_page(page_of(_design(name="Shelf", state={"read": 0, "of": 1}), code, tests=""), tmp_path / "t.html", browser)

        return next(r for r in _in_a_browser(run).generic if r.name == "it is shown")

    assert shown(spine).ok, shown(spine).why
    seen = shown(broken)
    assert not seen.ok and "drawing it raised" in seen.why and "paintTheSpine" in seen.why
