"""Bryan, 27 November 2026: "she shouldnt be blind to her own autonomous activity. she should have the ability to actually reason
about what's going on (which is where taste can come in ...) and just give input, thoughts, opinions, etc (if she wants)"."""
from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import pytest

from core.agency.her_own_time import HerOwnTime, Thought
from core.agency.what_she_does_next import CameOf, Option, WhatSheDoesNext
from core.making import her_view
from core.making.design import Design, Rule
from core.making.her_view import AChange, Compared, HerView, Shot, most_unlike
from tests.test_making_programs import _in_a_browser
from tests.test_many_ways import DISHES

_STATIC = Path(__file__).resolve().parents[1] / "interface" / "static"


def _dishes() -> Design:
    return Design(name="Dishes", kind="app", about="Dishes to cook.", rules=[Rule(says="a dish can be added")], state={"dishes": [], "asking": False, "said": ""})


# ── what she is shown ────────────────────────────────────────────────

def test_she_is_shown_the_states_least_like_how_it_opens_and_each_other():
    first = {"dishes": [], "said": ""}
    ends = [("adds one", {"dishes": ["Soup 1"], "said": "Soup 1 added"}), ("adds one again", {"dishes": ["Soup 1"], "said": "Soup 1 added"}),
            ("asks", {"dishes": [], "said": "", "asking": True}), ("as it began", first)]
    picked = [name for name, _state in most_unlike(first, ends)]
    assert picked[0] == "adds one" and "adds one again" not in picked and "as it began" not in picked and "asks" in picked


def test_what_she_made_is_looked_at_as_it_opens_at_a_phone_s_width_and_where_its_tests_leave_it():
    tests = 'test("Two are added", t => { t.click("Add"); t.click("Add"); t.expect(t.s.dishes.length === 2); });\ntest("Clearing asks", t => { t.click("Clear"); t.expect(t.s.asking); });'

    async def run(browser):
        return await her_view.looked_at(_dishes(), DISHES, tests, browser)

    shots = _in_a_browser(run)
    assert [s.caption for s in shots] == ["as it opens", "in a narrow window, as it opens", "where the test “Two are added” leaves it", "where the test “Clearing asks” leaves it"]
    pictures = [base64.b64decode(s.png) for s in shots]
    assert all(p.startswith(b"\x89PNG") for p in pictures) and len(set(pictures)) == 4, "each is a picture of its own"
    assert "buttons: “Undo (cannot be used now)”; “Start over”; “Help”; “Add”; “Clear”" in shots[0].holds and "“No dishes yet”" in shots[0].holds, shots[0].holds
    assert "lists holding 2 items" in shots[2].holds, "what the page holds is read from it, beside the picture"


# ── her view, in the making ──────────────────────────────────────────

def _a_maker(tmp_path, view: HerView, verdict: Compared | None):
    from core.making import maker
    from core.making.trying import Result, Trial
    from core.understanding.workbench import Work

    made = maker._Maker("Dishes to cook, please", ask=None, write=None, work=Work("w", "dishes"), folder=tmp_path, browser=None)
    made.design, made.trial = _dishes(), Trial(generic=[Result("nothing in it overlaps, is cut off or is hard to read", True)])
    made.parts.functions = {"view": "view(s, ui) { return ui.text('x'); }"}
    made.has = ["a dish can be added"]
    built, looks = [], iter([[Shot("as it opens", "", "a plain list")], [Shot("as it opens", "", "a list with an empty state")]])

    thought = []

    async def written(prompt, n, *, given="", **how):
        thought.append((prompt, given, how))
        return "I would leave the list, but the empty page needs saying." if view is not None else ""

    async def asked(prompt, schema, n, **_how):
        assert prompt.startswith("What you wrote of it:") and thought, "what she thinks is written before it is put in shape"
        return view if schema is HerView else verdict

    async def screens():
        return next(looks)

    async def tested(rules):
        made.parts.tests += "\n" + "\n".join(f'test("{r}", t => {{}});' for r in rules)

    async def added(rules, words=""):
        built.append(rules)
        made.has += rules
        made.parts.functions = {"view": "view(s, ui) { return ui.empty('Nothing yet'); }"}

    async def kept():
        built.append("kept")
        return tmp_path / "index.html"

    made.ask, made.write, made.eyes, made.thinking, made.thought = asked, written, object(), {"cognitive_mode": "reflective"}, thought
    made._screens, made.tested, made.added_where_it_is_not, made.kept = screens, tested, added, kept
    return made, built


_VIEW = HerView(thinks="It works, but an empty list is just a blank page.", works=["adding is quick"],
                would_change=[AChange(change="where there are no dishes it says so, with what to do", why="a blank page reads as broken")])


def test_what_she_would_change_of_what_she_made_is_built_in_and_kept_where_she_thinks_it_better(tmp_path):
    made, built = _a_maker(tmp_path, _VIEW, Compared(better=True, thinks="Much clearer when it is empty."))
    asyncio.run(made.looked_at_by_her())
    rule = "where there are no dishes it says so, with what to do"
    assert built == [[rule], "kept"] and rule in [r.says for r in made.design.rules] and rule in made.has
    views = [row for row in made.work.seen()["happened"] if row["what"] == "her_view"]
    assert [how for _prompt, _given, how in made.thought] == [{"cognitive_mode": "reflective"}] * 2, "she is given room to think, of it and of the change"
    assert "a plain list" in made.thought[0][1] and "a list with an empty state" in made.thought[1][1] and "Is it better made now?" == made.thought[1][0]
    assert "Dishes to cook, please" in made.thought[0][1] and "a dish can be added" not in made.thought[0][1], "she sees it as whoever asked, not by its rules"
    assert "Undo, Start over and Help of its own" in made.thought[0][1], "what the window does is not the program's to change"
    assert "Measured in its page, in a wide and a narrow window: nothing overlaps" in made.thought[0][1], "what was measured stands beside what her eyes said"
    assert views[0]["saw"] == [{"caption": "as it opens", "seen": "a plain list"}], "what she saw is kept with what she thought"
    assert views[0]["thinks"].startswith("It works") and views[0]["would_change"][0]["why"] == "a blank page reads as broken"
    assert views[1] == {**views[1], "verdict": "kept", "thinks": "Much clearer when it is empty."}
    asyncio.run(made.looked_at_by_her())
    assert built == [[rule], "kept"], "she looks once a making"


def test_what_she_does_not_think_better_made_is_taken_back(tmp_path):
    made, built = _a_maker(tmp_path, _VIEW, Compared(better=False, thinks="The note crowds the list."))
    asyncio.run(made.looked_at_by_her())
    assert [r.says for r in made.design.rules] == ["a dish can be added"] and made.has == ["a dish can be added"]
    assert made.parts.functions["view"] == "view(s, ui) { return ui.text('x'); }" and "test(" not in made.parts.tests
    assert any(row.get("verdict") == "as it was" for row in made.work.seen()["happened"]) and built[-1] == "kept"


def test_what_she_would_change_that_cannot_be_built_is_taken_back_before_she_is_asked(tmp_path):
    from core.making.trying import Result, Trial

    made, built = _a_maker(tmp_path, _VIEW, Compared(better=True, thinks="It looks the same, and I like it."))

    async def added_worse(rules, words=""):
        built.append(rules)
        made.has += rules
        made.trial = Trial(rules=[Result(rules[0], False, "nothing says so where there are no dishes")])

    made.added_where_it_is_not = added_worse
    asyncio.run(made.looked_at_by_her())
    assert [r.says for r in made.design.rules] == ["a dish can be added"] and made.has == ["a dish can be added"] and not made.trial.failing()
    assert "test(" not in made.parts.tests and len(made.thought) == 1, "she is not asked to compare what was not built"
    assert any(row.get("verdict") == "as it was" for row in made.work.seen()["happened"]) and built[-1] == "kept"
    assert [row["title"] for row in made.work.seen()["happened"] if row["what"] == "tried"][-1] == "As it was before", "the work shows it as it is kept"


def test_without_her_eyes_she_does_not_look_and_nothing_is_changed(tmp_path):
    made, built = _a_maker(tmp_path, _VIEW, None)
    made.eyes = None
    asyncio.run(made.looked_at_by_her())
    assert built == [] and not made.viewed


def test_where_she_writes_nothing_of_it_nothing_is_put_in_her_words(tmp_path):
    made, built = _a_maker(tmp_path, None, None)
    asyncio.run(made.looked_at_by_her())
    assert built == [] and [row for row in made.work.seen()["happened"] if row["what"] == "her_view"] == []


def test_what_she_would_leave_as_it_is_is_left(tmp_path):
    made, built = _a_maker(tmp_path, HerView(thinks="I like it as it is.", works=["it is calm"]), None)
    asyncio.run(made.looked_at_by_her())
    assert built == [] and [row["thinks"] for row in made.work.seen()["happened"] if row["what"] == "her_view"] == ["I like it as it is."]


# ── what she makes of what she did on her own time ───────────────────

def _hers(tmp_path) -> HerOwnTime:
    return HerOwnTime(WhatSheDoesNext(tmp_path / "c.json", seed=3), tmp_path / "t.json")


def test_what_she_makes_of_what_she_did_is_hers_kept_counted_and_hers_to_tell(tmp_path):
    hers, asked = _hers(tmp_path), []

    async def does():
        return CameOf(got=0.5, what="asked “Why two tides a day?”, and it is answered in part; 3 pages read")

    async def reflects(facts):
        asked.append(facts)
        return Thought(thinks="The moon half of it made sense; the sun's part I still don't follow.", worth=0.9, tell=True)

    hers.reflects = reflects
    hers.offers("wants", lambda: [Option("find_out:tides", "find_out", "find out more about tides", "she wondered", does=does)])
    done = asyncio.run(hers.once())
    assert asked[0]["what_you_did"] == "find out more about tides" and asked[0]["how_far_it_got"] == 0.5
    assert done["thinks"].startswith("The moon half") and hers.thoughts[-1]["worth"] == 0.9
    assert done["reward"] == pytest.approx(CameOf(0.5, 0.9).reward(60.0), abs=0.02), "what it was worth to her counts beside how far it got"
    assert hers.found[-1]["thought"] and "what you made of it: The moon half" in hers.where_she_is()
    from core.agency import what_she_could_do

    async def tell(_find):
        return True

    (option,) = what_she_could_do.things_to_tell(hers, tell, lambda: 1.0)
    assert option.what == "tell them what she thinks of find out more about tides"
    again = _hers(tmp_path)
    again._load()
    assert again.thoughts[-1]["thinks"].startswith("The moon half"), "what she thought is there after she is closed"


def test_where_she_has_nothing_to_say_of_it_nothing_is_kept_and_where_she_cannot_be_asked_it_is_learned_as_ever(tmp_path):
    hers = _hers(tmp_path)

    async def does():
        return CameOf(got=1.0, what="done")

    async def quiet(_facts):
        return Thought()

    hers.reflects = quiet
    hers.offers("goals", lambda: [Option("goal:a", "goal", "a goal", does=does)])
    done = asyncio.run(hers.once())
    assert hers.thoughts == [] and done["thinks"] == "" and done["reward"] > 0

    async def away(_facts):
        raise RuntimeError("her model is away")

    hers.reflects = away
    assert asyncio.run(hers.once())["reward"] > 0


# ── the panel ────────────────────────────────────────────────────────

_PAGE = """<!doctype html><html><body><div id="workbench" class="wb wb-wide" style="width:1300px;height:900px"></div><script>
const happened = [{ n: 1, at: 1, what: 'step', title: 'Looking at it as somebody using it would' },
  { n: 2, at: 2, what: 'her_view', thinks: 'It works, but an empty list is just a blank page.', works: ['adding is quick'], would_change: [{ change: 'where there are no dishes it says so', why: 'a blank page reads as broken' }], looked_at: ['as it opens', 'at a phone width'] },
  { n: 3, at: 3, what: 'her_view', thinks: 'Much clearer when it is empty.', verdict: 'kept' }];
const one = { id: 'one', title: 'Dishes to cook', name: 'Dishes', kind: 'program', state: 'done', began: 1, ended: 5, now: {}, stalled: '' };
window.fetch = async (url) => { const u = new URL(url, 'http://x');
  if (u.pathname === '/api/work') return { ok: true, json: async () => ({ works: [one] }) };
  if (u.pathname === '/api/her_own_time') return { ok: true, json: async () => ({ doing: {}, before: {}, choices: [], finding_out: [], learned: {}, thoughts: [{ at: 1, of: 'find out more about tides', thinks: 'The moon half of it made sense.', worth: 0.9 }] }) };
  return { ok: true, json: async () => ({ ...one, happened: Number(u.searchParams.get('after') || 0) ? [] : happened }) }; };
window.EventSource = class { constructor() { setTimeout(() => this.onerror && this.onerror(), 5); } addEventListener() {} close() {} };
</script><script>%s</script><script>mountWorkbench(document.getElementById('workbench'), { wide: true });</script></body></html>"""


def test_what_she_thinks_is_shown_in_her_voice_beside_what_she_did(tmp_path):
    api = pytest.importorskip("playwright.async_api")
    path = tmp_path / "bench.html"
    path.write_text(_PAGE % (_STATIC / "workbench.js").read_text(encoding="utf-8"), encoding="utf-8")

    async def go():
        async with api.async_playwright() as started:
            try:
                browser = await started.chromium.launch(headless=True)
            except api.Error as exc:
                pytest.skip(f"no browser: {exc}")
            page = await browser.new_page()
            await page.goto(path.as_uri())
            await page.wait_for_selector(".wb-her-view.kept", timeout=8000)
            views = await page.locator(".wb-log .wb-her-view").all_inner_texts()
            await page.click("[data-work=her-own-time]")
            await page.wait_for_selector(".wb-own-time .wb-her-view")
            own = await page.locator(".wb-own-time").inner_text()
            await browser.close()
            return views, own

    views, own = asyncio.run(go())
    assert "What she thinks of it" in views[0] and "2 screens looked at" in views[0] and "What she would change" in views[0] and "a blank page reads as broken" in views[0]
    assert "Looked at again, and kept" in views[1] and "Much clearer" in views[1]
    assert "What she thinks of what she did" in own and "Find out more about tides" in own and "The moon half of it made sense." in own and "+0.90" in own


def test_she_does_not_look_again_at_what_has_not_changed_and_is_told_what_she_thought_where_it_has(tmp_path):
    """LIVE 28 November 2026 she looked again, minutes after she had looked, at a dinner planner whose request was tried again."""
    import hashlib

    made, built = _a_maker(tmp_path, _VIEW, Compared(better=False, thinks="The note crowds the list."))
    asyncio.run(made.looked_at_by_her())
    view = dict(made.last_view)
    assert view["thinks"].startswith("It works") and "taken back" in view["came"]
    assert view["code"] == hashlib.sha256(made.parts.code().encode("utf-8")).hexdigest()[:16], "what she looked at is what is kept"

    again, built_again = _a_maker(tmp_path, _VIEW, Compared(better=True, thinks="Clearer."))
    again.last_view = view
    asyncio.run(again.looked_at_by_her())
    assert built_again == [] and again.thought == [], "the same code is not looked at again"

    changed, _built = _a_maker(tmp_path, _VIEW, Compared(better=True, thinks="Clearer."))
    changed.last_view = view
    changed.parts.functions = {"view": "view(s, ui) { return ui.text('something else'); }"}
    asyncio.run(changed.looked_at_by_her())
    assert "When you last looked at it you thought: It works" in changed.thought[0][1] and "taken back" in changed.thought[0][1]


def test_what_she_thought_is_kept_with_the_program_and_read_back(tmp_path, monkeypatch):
    from core.making import maker
    from core.rebuilding import her_build_writes

    def governed(how, path, *payload, **_how):
        Path(path).write_text(payload[0], encoding="utf-8") if how == "write_text" else None

    monkeypatch.setattr(her_build_writes, "governed", governed)
    made, _built = _a_maker(tmp_path, _VIEW, None)
    made.kept = maker._Maker.kept.__get__(made)
    made.folder, made.design.kind = tmp_path, "tool"
    made.design.language = "python"
    made.last_view = {"code": "abc", "thinks": "Calm.", "would_change": [], "came": "nothing to change"}
    asyncio.run(made.kept())
    assert maker._her_view_of(tmp_path) == made.last_view


def test_what_she_would_have_changed_and_does_not_hold_is_let_go_and_what_was_asked_is_not(tmp_path):
    """LIVE 28 November 2026 a vocabulary trainer was held to "Fix the rendering bug where 'Add a New Word' and 'Start Quiz'
    buttons appear twice", which her eyes had seen and was not so."""
    from core.making.trying import Result, Trial

    made, built = _a_maker(tmp_path, _VIEW, None)
    phantom = "Fix the rendering bug where the Add button appears twice"
    made.design.rules += [Rule(says=phantom)]
    made.has = ["a dish can be added", phantom]
    made.last_view = {"code": "x", "thinks": "", "would_change": [phantom], "came": ""}
    made.parts.tests = 'test("a dish can be added", t => {});\n\ntest("' + phantom + '", t => {});\n\ntest("' + phantom + ', tested again", t => {});'
    made.trial = Trial(rules=[Result("a dish can be added", False, "no"), Result(phantom, False, "seen once"), Result(phantom + ", tested again", False, "seen once")])

    async def tried(*_a, **_k):
        return Trial(rules=[Result("a dish can be added", False, "no")])

    made.tried = tried
    asyncio.run(made.her_own_that_do_not_hold_let_go())
    assert [r.says for r in made.design.rules] == ["a dish can be added"] and made.has == ["a dish can be added"]
    assert phantom not in made.parts.tests and 'test("a dish can be added"' in made.parts.tests and built == ["kept"]
    assert any("I let go" in (row.get("words") or "") for row in made.work.seen()["happened"])
    assert [r.name for r in made.trial.failing()] == ["a dish can be added"], "what was asked still has to hold"
