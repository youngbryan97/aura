"""Programs she makes: the pieces hold, code comes apart and goes together, a program is tried, and what fails is mended."""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from core.making import design as how
from core.making import maker, pieces
from core.making.code_parts import (
    BEFORE,
    assembled,
    each_test_in,
    parts_in,
    patched,
    python_patched,
)
from core.making.design import Asked, Design, Rule, in_proportion, left_out, with_every_ask
from core.making.tools import Case, Cases, languages, try_tool
from core.making.trying import as_object, page_of, site_of, try_page
from core.understanding.workbench import PutDown, Work, get_workbench, this_work, working_on

_CHANCE_JS = ("let seed=1;const rnd=()=>{seed=(seed*16807)%2147483647;return seed/2147483647},between=(a,b)=>a+rnd()*(b-a),"
              "int=(a,b)=>Math.floor(a+rnd()*(b-a+1)),pick=l=>l[Math.floor(rnd()*l.length)],shuffle=l=>{const o=l.slice();"
              "for(let i=o.length-1;i>0;i--){const j=Math.floor(rnd()*(i+1));[o[i],o[j]]=[o[j],o[i]]}return o},W=960,H=600,play=()=>{};")


#: What something draws, drawn with the frame's own pen: how much of the view is left bare, how many colours, and where.
_PAINTED_JS = ("const pen_ = (() => { " + (Path(maker.__file__).parent / "frame.js").read_text() + "\nreturn Frame.pen; })();"
               "const painted = (draw, t = 1) => { const c = document.createElement('canvas'); c.width = W; c.height = H;"
               "const x = c.getContext('2d', { willReadFrequently: true }); x.save(); draw(pen_(x, t)); x.restore(); const d = x.getImageData(0, 0, W, H).data;"
               "let bare = 0, sum = 0; const seen = new Set(); for (let i = 0; i < d.length; i += 4) { if (d[i + 3] === 0) bare++; "
               "if (i % 28 === 0) seen.add((d[i] >> 2) * 4096 + (d[i + 1] >> 2) * 64 + (d[i + 2] >> 2)); sum = (sum * 31 + d[i] + d[i + 1] + d[i + 2]) % 1e9; }"
               "return { bare: bare / (W * H), colours: seen.size, sum, at: (a, b) => [...d.slice((Math.round(b) * W + Math.round(a)) * 4, (Math.round(b) * W + Math.round(a)) * 4 + 4)] }; };")


def _in_a_browser(act):
    """What ``act(browser)`` gives, in a Chromium of its own."""
    api = pytest.importorskip("playwright.async_api")

    async def run():
        async with api.async_playwright() as started:
            try:
                browser = await started.chromium.launch(headless=True)
            except api.Error as exc:
                pytest.skip(f"no Chromium here: {exc}")
            try:
                return await act(browser)
            finally:
                await browser.close()

    return asyncio.run(run())


def _values(expressions: list[str]) -> list:
    async def act(browser):
        page = await browser.new_page()
        return [await page.evaluate(expression) for expression in expressions]

    return _in_a_browser(act)


# ── the pieces ───────────────────────────────────────────────────────

def test_every_piece_says_what_it_is_for_how_it_is_called_and_is_checked():
    held = pieces.pieces()
    assert len(held) >= 60
    for name, piece in held.items():
        assert piece.for_what and piece.calls and piece.code and piece.checks, name
        assert all(need in held for need in piece.needs), name
        assert f"const {name} =" in piece.code, name


def test_every_check_of_every_piece_holds():
    checks = [(name, check) for name, piece in pieces.pieces().items() for check in piece.checks]
    held = _values([f"() => {{ {_CHANCE_JS}{_PAINTED_JS}{pieces.bundle([name])}\n return ({check}); }}" for name, check in checks])
    assert [c for c, ok in zip(checks, held, strict=True) if ok is not True] == []


def test_the_pieces_for_tools_hold():
    held = pieces.pieces("python")
    assert len(held) >= 14
    for name, piece in held.items():
        assert piece.for_what and piece.calls and piece.code and piece.checks, name
        space: dict = {}
        exec(compile(pieces.bundle([name], "python"), name, "exec"), space)  # noqa: S102 - the repository's own pieces
        for check in piece.checks:
            assert eval(check, space) is True, f"{name}: {check}"  # noqa: S307 - the repository's own checks


def test_a_tool_has_the_pieces_it_calls_on_above_it(tmp_path):
    made = maker._Maker("x", ask=None, write=None, work=Work("w", "x"), folder=tmp_path, browser=None)
    made.design = _design(kind="tool", language="python", controls=[], state={})
    made._written("Here.\n```python\nimport sys\nprint(table([['a', 1]]))\n```", "it")
    whole = made.whole(made.parts)
    assert whole.index("def table(") < whole.index("import sys") and "def bars(" not in whole
    assert pieces.used_in("def table(x):\n    return x\ntable(1)", "python") == [], "its own function of that name is its own"


def test_a_piece_brings_what_it_stands_on_and_is_found_where_it_is_called():
    assert pieces.with_needs(["path", "mazes"]) == ["grid", "path", "mazes"]
    assert pieces.used_in("const d = vec.dist(a, b); this.grid.x; s.timers.y") == ["vec"]


# ── code in its parts ────────────────────────────────────────────────

CODE = '''const SPEED = 4;
function near(a, b) { return Math.hypot(a.x - b.x, a.y - b.y) < 10; } // a, b
({
  // how it begins, with a comma
  initial() { return { x: 1, list: [1, 2], say: `a ${[1, 2].map((n) => { return n; }).join(",")} b` }; },
  update(s, e) { if (/[,}]/.test("x")) return s; const q = "a, {b}"; s.x += SPEED; return s; },
  score: (s) => s.x / 2,
});'''


def test_code_comes_apart_by_its_functions_and_goes_together_again():
    parts = parts_in(CODE)
    assert list(parts) == [BEFORE + "SPEED", BEFORE + "near", "initial", "update", "score"]
    assert parts_in(assembled(parts)) == parts


def test_a_change_takes_the_place_of_what_it_names_and_leaves_the_rest():
    parts = parts_in(CODE)
    new, changed = patched(parts, "({ update(s, e) { s.x = 9; }, more(s) { return 1; } })")
    assert changed == ["update", "more"] and "s.x = 9" in new["update"]
    assert new["initial"] == parts["initial"] and new[BEFORE + "near"] == parts[BEFORE + "near"]


@pytest.mark.parametrize("code", [
    "const game = { initial() { return {}; }, update(s, e) {} };\nexport default game;",
    "function initial() { return { a: 1 }; }\nfunction update(s, e) { s.a++; }",
    "export default { initial() { return {}; }, update(s, e) {} }",
])
def test_code_is_taken_however_it_was_written(code):
    assert {"initial", "update"} <= set(parts_in(code))


def test_a_line_quoted_from_a_function_is_no_part_of_the_program():
    said = "const lid = s.kite.letters.pop();\nconst FAR = 40;\nconst near = (s) => s.x < FAR;\n({ update(s, e) { const L = s.letters[0]; } })"
    assert list(parts_in(said)) == [BEFORE + "FAR", BEFORE + "near", "update"]
    assert parts_in("const L = s.letters.find(l => l.id === lid);") == {}


def test_an_object_is_had_from_whatever_gives_one():
    ways = ["({ a() { return 1; } })", "{ a() { return 1; } }", "const p = { a() { return 1; } };",
            "const K = 1; ({ a() { return K; } })", "function update() { return 1; }", "const p = (() => { return { a() { return 1; } }; })();"]
    assert _values([f"() => {{ const o = {as_object(code)}; return (o.a || o.update)(); }}" for code in ways]) == [1] * len(ways)


def test_a_test_that_was_cut_off_is_left_out_and_takes_no_other_with_it(tmp_path):
    cut = TESTS + '\ntest("cut off", t => { t.tick(1); if (t.s.lamp'
    before, found = each_test_in("const K = 1;\n" + cut)
    assert before.strip() == "const K = 1;" and [name for name, _text in found] == ["a moth that is caught is saved", "it ends when the lamp is out", "hollow"]
    trial = _tried(page_of(_design(pieces=["vec"]), GOOD, tests=cut + "); ) }"), tmp_path)
    assert trial.failing() == []


TOOL = """import sys

LIMIT = 3


def shout(word):
    return word.upper()


# says it
def main():
    print(shout(sys.argv[1])[:LIMIT])


if __name__ == "__main__":
    main()
"""


def test_a_tool_is_changed_by_the_functions_that_change():
    new, changed = python_patched(TOOL, "import json\n\ndef shout(word):\n    return soft(word) + '!'\n\n\ndef soft(word):\n    return word.lower()\n")
    assert set(changed) == {"import import json", "shout", "soft"}
    space: dict = {"__name__": "x"}
    exec(compile(new, "t", "exec"), space)  # noqa: S102 - the test's own text
    assert space["shout"]("AB") == "ab!" and space["LIMIT"] == 3 and new.rstrip().endswith("main()") and "# says it" in new
    assert python_patched(TOOL, TOOL.replace("upper", "lower"))[1] == ["it"], "a whole file is taken whole"
    assert python_patched(TOOL, "from __future__ import annotations\n\ndef shout(word):\n    return word\n")[0].startswith("from __future__")
    assert python_patched(TOOL, "def broken(:")[0] == "def broken(:"


# ── the design ───────────────────────────────────────────────────────

def _design(**over) -> Design:
    said = {"name": "Moth Lamp", "kind": "game", "about": "Keep the moths from the lamp.",
            "rules": [Rule(says="a moth that is caught is saved", of="catch the moths"), Rule(says="it ends when the lamp is out")],
            "state": {"x": 0, "moths": [{"x": 1, "y": 1}], "saved": 0, "lamp": 3},
            "controls": [how.Control(input="pointer", does="move the hand")]}
    return Design(**{**said, **over})


def test_everything_asked_for_is_a_rule_in_their_own_words_where_no_rule_is_of_it():
    design = with_every_ask(_design(), ["catch the moths", "in the rain"])
    assert [rule.says for rule in design.rules] == ["a moth that is caught is saved", "it ends when the lamp is out", "in the rain"]


def test_a_design_that_is_not_whole_says_so():
    assert left_out(_design()) == []
    assert left_out(_design(controls=[], state={})) == ["it has no controls", "it has no state"]


def test_a_small_request_is_not_given_forty_rules():
    rules = [*(Rule(says=f"theirs {n}", of="catch the moths") for n in range(18)), *(Rule(says=f"expected {n}") for n in range(18))]
    kept = in_proportion(_design(rules=rules), ["I want to catch the moths"])
    assert len(kept) == 2 * how.A_FEW and kept[0].says == "theirs 0"


# ── trying one ───────────────────────────────────────────────────────

GOOD = '''({
  initial() { return { x: W / 2, y: H / 2, moths: [], next: 0.5, saved: 0, lamp: 3, t: 0 }; },
  update(s, e) {
    if (e.type === "pointer") { s.x = e.x; s.y = e.y; }
    if (e.type === "tick") {
      s.t += e.dt; s.next -= e.dt;
      if (s.next <= 0) { s.next = 1; const a = rnd() * 6.28; s.moths.push({ x: W / 2 + Math.cos(a) * 400, y: H / 2 + Math.sin(a) * 280 }); }
      for (const m of s.moths) {
        const d = Math.hypot(W / 2 - m.x, H / 2 - m.y) || 1; m.x += ((W / 2 - m.x) / d) * 70 * e.dt; m.y += ((H / 2 - m.y) / d) * 70 * e.dt;
        if (d < 24) { m.gone = true; s.lamp--; } else if (vec.dist(s, m) < 34) { m.gone = true; s.saved++; }
      }
      s.moths = s.moths.filter((m) => !m.gone);
    }
  },
  score(s) { return s.saved; },
  over(s) { return s.lamp <= 0 && { won: false, say: "The lamp went out." }; },
  hud(s) { return [{ label: "Lamp", value: s.lamp }]; },
  draw(s, g) { g.sky("#11142a", "#1d1233"); g.circle(W / 2, H / 2, 30, "#ffd977"); for (const m of s.moths) g.circle(m.x, m.y, 9, "#cfc7ff"); g.circle(s.x, s.y, 30, null, { stroke: "#9fe8ff" }); },
})'''
TESTS = '''test("a moth that is caught is saved", t => { t.s.moths = [{ x: 500, y: 100 }]; t.s.next = 9; t.point(500, 100, 0.1); t.expect(t.score === 1, "score " + t.score); });
test("it ends when the lamp is out", t => { t.s.lamp = 1; t.s.next = 9; t.s.x = 0; t.s.moths = [{ x: W / 2 + 26, y: H / 2 }]; t.tick(1); t.expect(t.over, "still going"); });
test("hollow", t => t.expect(t.s.lamp === 3));'''


def _tried(html: str, tmp_path: Path):
    return _in_a_browser(lambda browser: try_page(html, tmp_path / "trial.html", browser))


def test_a_program_that_works_is_seen_to(tmp_path):
    trial = _tried(page_of(_design(pieces=["vec"]), GOOD.replace("vec.dist", "this.vec.dist").replace("rnd() * 6.28", "this.between(0, 6.28)"), tests=TESTS), tmp_path)
    assert [r.name for r in trial.failing()] == []
    assert any("scored" in r.why for r in trial.generic) and trial.hollow == ["hollow"]


@pytest.mark.parametrize(("change", "named"), [
    (("s.saved++", "s.saved += 0"), "it can be scored in"),
    (("s.lamp--", "s.lamp += 0"), "it can be lost or won"),
    (("if (e.type === \"pointer\") { s.x = e.x; s.y = e.y; }", ""), "Mouse does something (move the hand)"),
    (("rnd() * 6.28", "Math.random() * 6.28"), "it is the same twice from the same beginning"),
    (("s.t += e.dt;", "s.t += undefined;"), "it runs without breaking, whatever is pressed"),
    (("g.sky(\"#11142a\", \"#1d1233\");", "nothing.there();"), "it is drawn"),
    (("lamp: 3, t: 0", "t: 0"), "its state is as it was designed"),
])
def test_a_program_with_something_wrong_is_told_what(tmp_path, change, named):
    assert change[0] in GOOD
    trial = _tried(page_of(_design(pieces=["vec"]), GOOD.replace(*change), tests=TESTS), tmp_path)
    assert named in [r.name for r in trial.failing()], [(r.name, r.why) for r in trial.failing()]


def test_a_test_that_sets_a_state_sets_it_over_one_that_has_begun(tmp_path):
    tests = 'test("set", t => { t.s = { moths: [{ x: 500, y: 100 }], next: 9 }; t.point(500, 100, 0.1); t.expect(t.s.lamp === 3 && t.score === 1); });'
    assert _tried(page_of(_design(pieces=["vec"]), GOOD, tests=tests), tmp_path).rules[0].ok


def test_tests_are_read_in_either_manner(tmp_path):
    tests = ('test("a", t => { t.tick(0.1); expect(t.s.lamp).toBe(3); expect(0).toBe(0); expect([1]).toEqual([1]); t.expect(true); });'
             'test("b", t => { expect(t.s.lamp).toBeGreaterThan(5); });test("c", t => { expect(t.s.lamp === 9, "nine"); });'
             'test("d", t => { expect(2).not.toBe(2); });')
    got = {r.name: (r.ok, r.why) for r in _tried(page_of(_design(pieces=["vec"]), GOOD, tests=tests), tmp_path).rules}
    assert got["a"][0] and not got["b"][0] and "more than 5" in got["b"][1] and "nine" in got["c"][1] and not got["d"][0]


def test_a_test_that_asks_for_what_there_is_not_is_told_from_a_program_that_fails_it(tmp_path):
    tests = 'test("a", t => { t.nothing(); });test("b", t => { t.s.moths = null; t.tick(1); });test("c", t => { t.initial(); t.update({ type: "pointer", x: 3, y: 4 }); t.expect(t.s.x === 3); });'
    got = {r.name: r for r in _tried(page_of(_design(pieces=["vec"]), GOOD, tests=tests), tmp_path).rules}
    assert got["a"].the_test and not got["b"].the_test and got["b"].broke and got["c"].ok


def test_what_broke_is_told_by_the_functions_it_broke_in(tmp_path):
    code = GOOD.replace("({\n  initial", "({\n  glow(m) { return m.lamps.bright; },\n  initial", 1).replace("s.t += e.dt;", "s.t += e.dt; this.glow(s);")
    failing = {r.name: r for r in _tried(page_of(_design(pieces=["vec"]), code, tests=TESTS), tmp_path).failing()}
    assert "in glow, called by update" in failing["it runs without breaking, whatever is pressed"].why


def test_a_piece_asked_for_what_it_has_not_is_told_with_what_it_has(tmp_path):
    failing = {r.name: r for r in _tried(page_of(_design(pieces=["vec"]), GOOD.replace("vec.dist(s, m)", "vec.apart(s, m)"), tests=TESTS), tmp_path).failing()}
    assert "vec.apart is not a function; vec has " in failing["it runs without breaking, whatever is pressed"].why and "dist" in failing["it runs without breaking, whatever is pressed"].why


def test_code_written_in_more_than_one_block_is_all_taken(tmp_path):
    made = maker._Maker("x", ask=None, write=None, work=Work("w", "x"), folder=tmp_path, browser=None)
    made.design = _design()
    assert made._written("One.\n```js\n({ initial() { return {}; } })\n```\nTwo.\n```js\n({ update(s, e) {} })\n```", "x") == ["initial", "update"]


def test_what_is_drawn_can_be_tested(tmp_path):
    tests = ('test("a", t => { t.s.moths = [{ x: 9, y: 9 }]; expect(t.draw().filter(c => c.what === "circle").length).toBe(3); });'
             'test("b", t => { expect(t.draw().some(c => c.what === "wedge")).toBe(true); });')
    assert [r.ok for r in _tried(page_of(_design(pieces=["vec"]), GOOD, tests=tests), tmp_path).rules if r.name == "b"] == [False]


def test_a_key_for_what_the_window_does_is_no_control_of_the_program(tmp_path):
    design = _design(controls=[how.Control(input="pointer", does="move the hand"), how.Control(input="Enter", does="start again")])
    trial = _tried(page_of(design.model_copy(update={"pieces": ["vec"]}), GOOD, tests=""), tmp_path)
    made = maker._Maker("x", ask=None, write=None, work=Work("w", "x"), folder=tmp_path, browser=None)
    made.design = design
    assert made._dead_controls_go(trial.failing()) == ["start again"]
    assert [c.input for c in design.controls] == ["pointer"]


#: A game of two steps: what is carried from one side counts only at the other, and only there.
TWO_STEPS = '''({
  initial() { return { me: { x: 480, y: 300, has: 0 }, stones: [{ x: 60, y: 60 }], home: { x: 900, y: 540 }, brought: 0, left: 90 }; },
  update(s, e) {
    if (e.type === "key" && e.key === "Space" && e.down) this.put(s);
    if (e.type !== "tick") return;
    s.left -= e.dt; const v = 240 * e.dt;
    s.me.x = Math.max(0, Math.min(W, s.me.x + ((e.held.ArrowRight ? 1 : 0) - (e.held.ArrowLeft ? 1 : 0)) * v));
    s.me.y = Math.max(0, Math.min(H, s.me.y + ((e.held.ArrowDown ? 1 : 0) - (e.held.ArrowUp ? 1 : 0)) * v));
    for (const p of s.stones) if (Math.hypot(p.x - s.me.x, p.y - s.me.y) < 40 && !s.me.has) { s.me.has = 1; p.x = between(0, W); p.y = between(0, 120); play("pick"); }
  },
  put(s) { if (s.me.has && Math.hypot(s.home.x - s.me.x, s.home.y - s.me.y) < 60) { s.me.has = 0; s.brought++; play("bell"); } },
  score(s) { return s.brought; },
  over(s) { return s.left <= 0 && { won: s.brought > 3, say: "Dark." }; },
  draw(s, g) { g.clear("#20242c"); g.circle(s.me.x, s.me.y, 40, "#ffd977"); g.rect(s.home.x - 60, s.home.y - 60, 120, 120, "#335566"); for (const p of s.stones) g.circle(p.x, p.y, 8, "#cfc7ff"); g.text("home", s.home.x, s.home.y, { color: "#ffffff" }); },
})'''


def _two_steps(code: str, tmp_path: Path, tests: str = ""):
    keys = [how.Control(input=k, does=f"go {k}") for k in ("ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown")]
    design = _design(state={}, rules=[Rule(says="a stone brought home counts", of="bring stones home")], controls=[*keys, how.Control(input="Space", does="put the stone down")])
    return _tried(page_of(design, code, tests=tests), tmp_path)


def test_what_takes_two_steps_is_found_by_playing_it(tmp_path):
    trial = _two_steps(TWO_STEPS, tmp_path)
    assert [(r.name, r.why) for r in trial.failing()] == []
    assert any(r.name == "it can be scored in" and "scored" in r.why for r in trial.generic)


def test_a_picture_that_slides_out_of_the_view_is_told_and_one_in_its_own_place_is_not(tmp_path):
    moved = TWO_STEPS.replace('g.clear("#20242c");', 'g.clear("#20242c"); g.ctx.translate(W / 2 - s.me.x * 6, 0); BACK')
    sky = "g.ctx.save(); for (let i = 0; i < 12; i++) g.rect(2400 + i * 80, 0, 80, 600, ['#112233','#445566','#778899','#aa3355'][i % 4]); g.ctx.restore();"
    world = _two_steps(moved.replace("BACK", sky), tmp_path)
    own = _two_steps(moved.replace("BACK", sky.replace("g.ctx.save();", "g.screen(() => {").replace("2400 + i * 80", "i * 80").replace(" g.ctx.restore();", "});")), tmp_path)
    named = "the picture is filled wherever play goes"
    assert next(r for r in world.generic if r.name == named).why.startswith("at the beginning") and next(r for r in own.generic if r.name == named).ok


def test_whoever_writes_a_game_is_told_the_pieces_it_is_drawn_with(tmp_path):
    made = maker._Maker("x", ask=None, write=None, work=Work("w", "x"), folder=tmp_path, browser=None)
    made.design = _design()
    assert "scene.sky(g," in made._given() and "figures.draw(g," in made._given()
    made.design = _design(kind="app")
    assert "scene.sky(g," not in made._given()
    design = _design(pieces=["vec"])
    page = page_of(design, GOOD.replace('g.sky("#11142a", "#1d1233");', 'scene.sky(g, { kind: "night" });'), tests=TESTS)
    made.design = design
    made._written(f"```js\n{GOOD.replace('g.sky(', 'scene.sky(')}\n```", "x")
    assert "scene" in design.pieces and "const scene" not in page


def test_what_can_never_be_reached_is_told_with_what_never_ran(tmp_path):
    trial = _two_steps(TWO_STEPS.replace("if (e.type === \"key\" && e.key === \"Space\" && e.down) this.put(s);\n    if (e.type !== \"tick\") return;",
                                         "if (e.type !== \"tick\") return;\n    if (e.key === \"Space\") this.put(s);"), tmp_path)
    failing = {r.name: r.why for r in trial.failing()}
    assert "Space does something (put the stone down)" in failing
    assert "its score never rose from 0" in failing["it can be scored in"] and "me.has 0 to 1" in failing["it can be scored in"]
    assert any(text.startswith("put(...) was never called") for text in trial.never_ran), trial.never_ran
    blocked = _two_steps(TWO_STEPS.replace("< 60) { s.me.has = 0;", "< -1) { s.me.has = 0;"), tmp_path)
    assert any(text.startswith("in put:") and "< -1) { s.me.has = 0; s.brought++" in text for text in blocked.never_ran), blocked.never_ran


def test_what_never_ran_goes_to_the_mend(tmp_path):
    from core.making.trying import Result, Trial

    asked: list[str] = []

    async def write(prompt, most, **_how):
        asked.append(prompt)
        return "```js\n({ score(s) { return 1; } })\n```"

    made = maker._Maker("x", ask=None, write=write, work=Work("w", "x"), folder=tmp_path, browser=None)
    made.design = _design()
    made.trial = Trial(generic=[Result("it can be scored in", False, "never")], never_ran=["put(...) was never called"])
    asyncio.run(made._a_mend(made.trial.failing(), warm=0.2))
    assert "none of this of it ever ran:\n- put(...) was never called" in asked[0]


def test_a_test_of_a_state_there_is_not_is_the_tests_fault_and_what_changed_is_told(tmp_path):
    tests = ('test("a", t => { t.tick(0.1); t.expect(t.s.camera.x === 4, "no camera"); });'
             'test("b", t => { const s = t.s; t.tick(0.1); t.expect(s.wind === 4, "no wind"); });'
             'test("c", t => { t.hold("ArrowRight", 0.5); t.expect(t.s.me.has === 1, "has nothing"); });'
             'test("d", t => { t.expect(t.sees("home") && t.sees("score"), "not shown"); });')
    tests += 'test("e", t => { t.expect(t.hud()[0].value === 0 && t.hud(t.s).length === 1 && t.put() === undefined, "no hud"); });'
    got = {r.name: r for r in _two_steps(TWO_STEPS.replace("  score(s)", "  hud(s) { return [{ label: 'Stones', value: s.brought }]; },\n  score(s)"), tmp_path, tests).rules}
    assert got["e"].ok, got["e"].why
    assert got["a"].the_test and "s.camera" in got["a"].why and "it has me, stones, home, brought, left" in got["a"].why
    assert got["b"].the_test and not got["c"].the_test and got["d"].ok
    assert "me.x 480 to 600" in got["c"].why and "left 90 to 89.5" in got["c"].why


def test_a_rule_is_held_to_with_nothing_else_about_where_something_else_came_between(tmp_path):
    code = TWO_STEPS.replace("stones: [{ x: 60, y: 60 }]", "stones: [{ x: 60, y: 60 }], thieves: [{ x: 480, y: 300 }]").replace(
        "s.left -= e.dt;", "s.left -= e.dt; for (const q of s.thieves) if (Math.hypot(q.x - s.me.x, q.y - s.me.y) < 30) s.me.has = 0;")
    tests = ('test("a", t => { t.s.stones = [{ x: 480, y: 300 }]; t.tick(0.1); t.expect(t.s.me.has === 1, "has it not"); });'
             'test("b", t => { t.s.stones = [{ x: 480, y: 300 }]; t.tick(0.05); t.s.thieves[0].x = 480; t.tick(0.1); t.expect(t.s.me.has === 0, "has it still"); });'
             'test("c", t => { t.tick(0.1); t.expect(t.s.me.has === 1, "has nothing"); });')
    more = code.replace("< 30) s.me.has = 0;", "< 30) s.me.has = 0; if (s.thieves.length < 9) s.thieves.push({ x: s.me.x, y: s.me.y });")
    tests += ('test("d", t => { t.s.thieves = []; t.s.stones = [{ x: 480, y: 300 }]; t.tick(2); t.expect(t.s.me.has === 1, "has it not"); });')
    got = {r.name: r for r in _two_steps(more, tmp_path, tests).rules}
    assert got["a"].ok and got["a"].why == "with nothing else about"
    assert got["d"].ok and got["d"].why == "with nothing else about", "kept empty, whatever comes into it"
    assert got["b"].ok and not got["c"].ok


def test_a_rule_nobody_asked_for_that_the_window_keeps_is_not_built_again(tmp_path, monkeypatch):
    import core.cognition.evidence_relevance as relevance

    def vectors(texts):
        return [[float(any(w in text for w in ("start screen", "pause"))), float("wind" in text), float("crow" in text)] for text in texts]

    monkeypatch.setattr(relevance, "vectors_if_ready", vectors)
    made = maker._Maker("x", ask=None, write=None, work=Work("w", "x"), folder=tmp_path, browser=None)
    rules = [Rule(says="the wind rises", of="windier"), Rule(says="a pause that slows the wind", of="pause slowly"), Rule(says="the game has a start screen"),
             Rule(says="crows fly in pairs")]
    kept = made._not_the_windows(_design(rules=rules))
    assert [rule.says for rule in kept] == ["the wind rises", "a pause that slows the wind", "crows fly in pairs"]
    assert [rule.says for rule in made._not_the_windows(_design(kind="site", rules=rules))] == [rule.says for rule in rules]
    design = _design(rules=rules, controls=[how.Control(input="Enter", does="go past the start screen"), how.Control(input="ArrowUp", does="climb the wind")])
    made._not_the_windows(design)
    assert [control.input for control in design.controls] == ["ArrowUp"]


def test_putting_right_what_breaks_counts_for_more_than_what_it_uncovers():
    from core.making.trying import Result

    broken = [Result("it runs without breaking, whatever is pressed", False, "state.t became NaN"), Result("a rule", False)]
    uncovered = [Result("a rule", False), Result("another", False), Result("a third", False), Result("Space does something (clap)", False, "pressing it changes nothing")]
    assert maker._how_much_is_wrong(uncovered) < maker._how_much_is_wrong(broken)


def test_a_name_used_before_it_has_a_value_is_told_and_breaks_nothing(tmp_path):
    trial = _tried(page_of(_design(), "(() => { const a = b + 1; const b = 2; return { initial() { return { a }; } }; })()", tests=TESTS), tmp_path)
    assert [r.name for r in trial.failing()] == ["it can be read"] and "before initialization" in trial.failing()[0].why


def test_code_that_cannot_be_read_or_does_not_end_is_told_as_that(tmp_path, monkeypatch):
    trial = _tried(page_of(_design(), "({ initial( { return {}; } })", tests=""), tmp_path)
    assert not trial.generic[0].ok
    from core.making import trying

    monkeypatch.setattr(trying, "TRIAL_S", 2.0)
    trial = _tried(page_of(_design(), GOOD.replace("s.t += e.dt;", "while (true) {}"), tests=""), tmp_path)
    assert "without end" in trial.failing()[0].why


APP = '''({
  initial() { return { name: "", items: [] }; },
  update(s, e) {
    if (e.type === "input") s[e.name] = e.value;
    if (e.type === "do" && e.name === "add" && s.name.trim()) { s.items.push({ name: s.name.trim() }); s.name = ""; }
    if (e.type === "do" && e.name === "remove") s.items.splice(e.value, 1);
  },
  view(s, ui) { return ui.column(ui.title("Loaves"), ui.row(ui.input("Loaf", "name", s.name, { enter: "add" }), ui.button("Add", "add")),
    ui.list(s.items, (i, n) => ui.row(ui.text(i.name), ui.button("Remove " + i.name, "remove", n)))); },
})'''


def test_an_app_is_tried_by_its_controls(tmp_path):
    design = _design(name="Loaves", kind="app", controls=[], state={"name": "", "items": [{"name": "rye"}]})
    tests = ('test("adding shows it", t => { t.type("Loaf", "rye").click("Add"); t.expect(t.sees("Remove rye"), "no rye"); });'
             'test("removing", t => { t.type("Loaf", "rye").click("Add").click("Remove rye"); t.expect(!t.sees("rye"), "still there"); });')
    assert _tried(page_of(design, APP, tests=tests), tmp_path).failing() == []
    dead = _tried(page_of(design, APP.replace('e.name === "remove"', 'e.name === "never"'), tests=tests), tmp_path).failing()
    assert {"removing"} <= {r.name for r in dead}


HIDDEN = '''({
  initial() { return { plants: [], adding: false, name: "", gone: 0 }; },
  update(s, e) {
    if (e.type === "input") s[e.name] = e.value;
    if (e.type === "do" && e.name === "open") s.adding = true;
    if (e.type === "do" && e.name === "clear") { s.plants = []; s.gone++; }
    if (e.type === "do" && e.name === "add") { s.plants.push({ id: s.plants.length + 1, name: s.name, wet: 0 }); s.adding = false; s.name = ""; }
    if (e.type === "do" && e.name === "water") s.plants.find((p) => p.id === e.value).wet++;
  },
  view(s, ui) {
    const list = s.plants.slice().sort((a, b) => a.wet - b.wet);
    return ui.column(ui.button("Add plant", "open"), ui.button("Clear all", "clear"), s.adding ? ui.row(ui.input("Name", "name", s.name), ui.button("Save", "add", null, { disabled: !s.name })) : null,
      list.map((p) => ui.card(ui.heading(p.name), ui.text("Watered " + p.wet), ui.button("Water", "water", p.id))));
  },
})'''


def test_what_is_used_is_found_as_a_person_finds_it(tmp_path):
    design = _design(name="Plants", kind="app", controls=[], state={"plants": [{"id": 1, "name": "fig", "wet": 0}], "adding": False, "name": "", "gone": 0})
    two = 't.s.plants = [{ id: 1, name: "A", wet: 0 }, { id: 2, name: "Fig", wet: 3 }]; '
    tests = ('test("behind a button", t => { t.type("Name", "Ivy").click("Save"); t.expect(t.s.plants[0].name === "Ivy", "no ivy"); });'
             f'test("by what it stands with", t => {{ {two}t.click("Fig"); t.expect(t.s.plants[1].wet === 4 && t.s.plants[0].wet === 0, "not watered"); }});'
             f'test("whole words", t => {{ {two}t.click("A"); t.expect(t.s.plants[0].wet === 1 && !t.s.adding, "not A"); }});'
             f'test("in order", t => {{ {two}t.expect(t.comesBefore("A", "Fig") && !t.comesBefore("Fig", "A")); t.click("A").click("A").click("A").click("A"); t.expect(t.comesBefore("Fig", "A"), "not after"); }});'
             'test("not yet", t => { t.click("Add plant"); t.click("Save"); });'
             'test("nothing taken away to find it", t => { t.s.plants = [{ id: 1, name: "A", wet: 0 }]; t.type("Colour", "red"); });')
    got = {r.name: r for r in _tried(page_of(design, HIDDEN, tests=tests), tmp_path).rules}
    assert all(got[name].ok for name in ("behind a button", "by what it stands with", "whole words", "in order")), [(r.name, r.why) for r in got.values() if not r.ok]
    assert "cannot be used now" in got["not yet"].why and "it shows “" in got["not yet"].why
    assert "no “Colour” to use" in got["nothing taken away to find it"].why


def test_a_mend_that_gains_nothing_whole_is_tried_a_part_at_a_time(tmp_path):
    work = Work("w", "moths")
    wrong = _RULES.replace("s.saved++", "s.saved += 0")
    update = "update" + _RULES.split("  update", 1)[1].split("  score(s)")[0].rstrip().rstrip(",")
    both = f"```js\n({{ {update}, hud(s) {{ return s.nothing.there; }}, over(s) {{ return s.nothing.there; }} }})\n```"
    made = _make(tmp_path, _Model([f"```js\n{TESTS}\n```", f"```js\n{wrong}\n```", f"```js\n{_LOOKS}\n```", both]), work)
    assert made is not None and made.trial.failing() == [] and "s.saved++" in made.parts.functions["update"]
    assert "nothing.there" not in made.parts.code()
    assert any(row["what"] == "tried" and row["title"].startswith("Trying the part of it that gains: update") for row in work.seen()["happened"])


def test_an_app_that_shows_the_machines_words_or_begins_filled_is_told_so(tmp_path):
    design = _design(name="Loaves", kind="app", controls=[], state={"name": "", "items": [{"id": 1, "name": "rye"}]})
    for change, named in (('ui.title("Loaves")', "it is shown"), ("items: [] }", "it begins with nothing in it that nobody put there")):
        bad = APP.replace(change, {'ui.title("Loaves")': 'ui.text("" + s.never)', "items: [] }": 'items: [{ id: 1, name: "rye" }] }'}[change], 1)
        assert bad != APP
        assert named in [r.name for r in _tried(page_of(design, bad, tests=""), tmp_path).failing()]


def test_what_looks_unfinished_is_measured_and_mended_where_that_leaves_less_of_it(tmp_path):
    design = _design(name="Loaves", kind="app", controls=[], state={"name": "", "items": [{"name": "rye"}]}, rules=[Rule(says="adding shows it", of="catch the moths")])
    wide = APP.replace('ui.title("Loaves")', 'ui.canvas("c", 10, 10, () => {}), ui.title("Loaves")')
    tests = '```js\ntest("adding shows it", t => { t.type("Loaf", "rye").click("Add"); t.expect(t.sees("Remove rye")); });\n```'
    rules, looks = wide[: wide.index("  view(s, ui)")] + "})", "({" + wide[wide.index("  view(s, ui)"):].rstrip()[:-2] + "})"
    measured = iter([["two things overlap"], []])

    async def unfinished(self, parts):
        return next(measured)

    import unittest.mock

    with unittest.mock.patch.object(maker._Maker, "_unfinished", unfinished):
        made = _make(tmp_path, _Model([tests, f"```js\n{rules}\n```", f"```js\n{looks}\n```", "Apart.\n```js\n({ view(s, ui) { return ui.column(ui.title(\"Loaves\"), "
                                       "ui.row(ui.input(\"Loaf\", \"name\", s.name), ui.button(\"Add\", \"add\")), ui.list(s.items, (i, n) => ui.row(ui.text(i.name), "
                                       "ui.button(\"Remove \" + i.name, \"remove\", n)))); } })\n```"], design))
    assert made is not None and made.trial.failing() == [] and "canvas" not in made.parts.functions["view"]


def test_a_site_is_tried_by_reading_it(tmp_path):
    design = _design(name="Tide Pools", kind="site", controls=[], state={})
    body = "<h1>Tide pools</h1><a href='#hours'>Hours</a><p>" + "Come and see what the sea leaves. " * 20 + "</p><h2 id='hours'>Hours</h2>"
    assert _tried(site_of(design, body, tests='test("hours", () => see("hours"));'), tmp_path).failing() == []
    bad = _tried(site_of(design, body.replace("id='hours'", ""), tests='test("prices", () => see("prices"));'), tmp_path)
    assert {r.name for r in bad.failing()} == {"every link goes somewhere", "prices"}


def test_a_tool_is_run_with_no_network_and_no_writing_outside(tmp_path):
    if not shutil.which("sandbox-exec"):
        pytest.skip("no sandbox here")
    outside = tmp_path / "outside.txt"
    code = ("import sys\nif '--help' in sys.argv: print('usage: prints its word backwards'); sys.exit(0)\n"
            "print(sys.argv[1][::-1])\n")
    cases = [Case(rule="backwards", args=["abc"], prints=["cba"], leaves_with=0), Case(rule="wrong", args=["abc"], prints=["abc!"])]
    trial = asyncio.run(try_tool(code, languages()["python"], cases))
    assert [r.ok for r in trial.rules] == [True, False] and trial.generic[-1].ok
    bad = "import sys\nopen(%r, 'w').write('x')\nprint('done')\n" % str(outside)
    trial = asyncio.run(try_tool(bad, languages()["python"], [Case(rule="writes", prints=["done"])]))
    assert not trial.rules[0].ok and not outside.exists()
    net = "import urllib.request\nprint(urllib.request.urlopen('http://127.0.0.1:9').status)\n"
    assert not asyncio.run(try_tool(net, languages()["python"], [Case(rule="net", prints=["200"])])).rules[0].ok


# ── making one ───────────────────────────────────────────────────────

class _Model:
    """Answers as a model would, from what it was given to answer with."""

    def __init__(self, written: list[str], design: Design | None = None) -> None:
        self.written, self.design, self.asked = list(written), design or _design(pieces=["vec"]), []

    async def ask(self, prompt, schema, n, *, given=""):
        self.asked.append(given + prompt)
        if schema is Asked:
            return Asked(asks=["catch the moths"])
        if schema is Cases:
            return Cases(cases=[])
        return self.design.model_copy(deep=True)

    async def write(self, prompt, n, *, warmth=0.2, given=""):
        self.asked.append(given + prompt)
        return self.written.pop(0)


def _make(tmp_path, model: _Model, work: Work | None = None):
    return _in_a_browser(lambda browser: maker.make("I want to catch the moths", ask=model.ask, write=model.write,
                                                    work=work or Work("w", "moths"), where=tmp_path, browser=browser))


_RULES = GOOD[: GOOD.index("  draw(s, g)")] + "})"
_LOOKS = "({" + GOOD[GOOD.index("  draw(s, g)"):].rstrip()[:-2] + "})"


def test_it_is_tested_before_it_is_written_and_kept_whole(tmp_path):
    work = Work("w", "moths")
    made = _make(tmp_path, _Model([f"Tests.\n```js\n{TESTS}\n```", f"I will move the hand.\n```js\n{_RULES}\n```", f"A dark room.\n```js\n{_LOOKS}\n```"]), work)
    assert made is not None and made.working() and made.trial.failing() == []
    assert made.path.is_file() and "Frame.open(PROGRAM)" in made.path.read_text()
    design, parts, asked = maker.parts_of(made.folder)
    assert design.name == "Moth Lamp" and "update" in parts.functions and asked == "I want to catch the moths"
    told = [row["what"] for row in work.seen()["happened"]]
    assert told.index("design") < told.index("code") < told.index("tried")
    assert any(row["what"] == "says" and "move the hand" in row["words"] for row in work.seen()["happened"])


def test_what_she_told_them_as_she_began_is_held_to_by_the_design(tmp_path):
    model = _Model([])
    made = maker._Maker("I want to catch the moths", ask=model.ask, write=model.write, work=Work("w", "x"), folder=tmp_path, browser=None)
    made.told = "I'll give you a jar, and E lets one go."
    asyncio.run(made.designed())
    assert any("E lets one go" in prompt and "kept to where it goes with what they asked" in prompt for prompt in model.asked)
    made.told = ""
    model.asked.clear()
    asyncio.run(made.designed())
    assert not any("You have told them" in prompt for prompt in model.asked)


def test_what_fails_is_mended_by_the_part_that_is_wrong_and_kept_only_where_less_fails(tmp_path):
    wrong = _RULES.replace("s.saved++", "s.saved += 0")
    worse = "Trying.\n```js\n({ update(s, e) { s.moths = null; } })\n```"
    right = "The catch did not count.\n```js\n({ update" + _RULES.split("  update", 1)[1].split("  score(s)")[0].rstrip().rstrip(",") + " })\n```"
    work = Work("w", "moths")
    made = _make(tmp_path, _Model([f"```js\n{TESTS}\n```", f"```js\n{wrong}\n```", f"```js\n{_LOOKS}\n```", worse, right]), work)
    assert made is not None and made.trial.failing() == [] and "s.saved++" in made.parts.functions["update"]
    said = " ".join(row.get("words", "") for row in work.seen()["happened"])
    assert "Not kept" in said and "Kept" in said


def test_a_test_that_is_wrong_is_put_right_by_its_rule_and_one_that_is_right_stands(tmp_path):
    wrong = TESTS.replace('t.expect(t.score === 1, "score " + t.score)', 't.expect(t.score === 7, "score " + t.score)')
    right = 'The test is wrong: one moth is one.\n```js\n' + TESTS.splitlines()[0] + "\n```"
    made = _make(tmp_path, _Model([f"```js\n{wrong}\n```", f"```js\n{_RULES}\n```", f"```js\n{_LOOKS}\n```",
                                   "It is right as it is.", right]))
    assert made is not None and made.trial.failing() == [] and "=== 1" in made.parts.tests
    made = _make(tmp_path / "b", _Model([f"```js\n{wrong}\n```", f"```js\n{_RULES}\n```", f"```js\n{_LOOKS}\n```",
                                         "It is right as it is.", "It is right as it is.", "It is the program: a moth is worth seven."]))
    assert made is not None and [r.name for r in made.trial.failing()] == ["a moth that is caught is saved"]


def test_a_word_in_the_middle_of_it_goes_back_to_the_design(tmp_path):
    work = Work("w", "moths")
    model = _Model([f"```js\n{TESTS}\n```", f"```js\n{_RULES}\n```", f"```js\n{_LOOKS}\n```"])
    later = model.design.model_copy(deep=True)
    later.rules.append(Rule(says="the lamp is green", of="make the lamp green"))
    model.written += ['```js\ntest("the lamp is green", t => { t.s.lamp = 2; t.tick(0.1); t.expect(t.s.lamp === 2); });\n```',
                      "Green now.\n```js\n({ draw(s, g) { g.clear(\"#020\"); g.circle(W / 2, H / 2, 30, \"#0f0\"); for (const m of s.moths) g.circle(m.x, m.y, 9, \"#cfc7ff\"); } })\n```"]
    real = model.write

    async def write(prompt, n, **how):
        answer = await real(prompt, n, **how)
        if len(model.written) == 2 and not work.somebody_spoke() and model.design is not later:
            work.say("make the lamp green")
            model.design = later
        return answer

    model.write = write
    made = _make(tmp_path, model, work)
    assert made is not None and "#0f0" in made.parts.functions["draw"] and "make the lamp green" in maker.parts_of(made.folder)[2]


def test_what_breaks_off_leaves_the_program_as_it_stood(tmp_path):
    work = Work("w", "moths")
    model = _Model([f"```js\n{TESTS}\n```", f"```js\n{_RULES.replace('s.saved++', 's.saved += 0')}\n```", f"```js\n{_LOOKS}\n```"])
    real, kept = model.write, []

    async def write(prompt, n, **how):
        if not model.written:
            raise RuntimeError("her model went away")
        return await real(prompt, n, **how)

    model.write = write
    made = _in_a_browser(lambda browser: maker.make("I want to catch the moths", ask=model.ask, write=model.write, work=work,
                                                    where=tmp_path, browser=browser, when_kept=kept.append))
    assert made is not None and made.working() and made.trial.failing() and set(kept) == {made.path}
    assert any(row["what"] == "trouble" and "her model went away" in row["why"] for row in work.seen()["happened"])
    # And it is gone on with from there.
    right = "```js\n({ update" + _RULES.split("  update", 1)[1].split("  score(s)")[0].rstrip().rstrip(",") + " })\n```"
    again = _Model([right])
    made = _in_a_browser(lambda browser: maker.mend(made.folder, ask=again.ask, write=again.write, work=Work("w", "x"), browser=browser))
    assert made is not None and made.trial.failing() == []


def test_what_was_cut_short_before_all_of_it_was_built_is_built(tmp_path):
    model = _Model([f"```js\n{TESTS}\n```", f"```js\n{_RULES}\n```", f"```js\n{_LOOKS}\n```"])
    made = _make(tmp_path, model)
    record = json.loads((made.folder / "program.json").read_text())
    assert record["built"] == [rule.says for rule in made.design.rules]
    record["design"]["rules"].append({"says": "the lamp hums", "of": ""})
    (made.folder / "program.json").write_text(json.dumps(record))
    more = _Model(['```js\ntest("the lamp hums", t => { t.tick(0.1); t.expect(t.s.hum === 1); });\n```',
                   "```js\n({ initial() { return { x: W / 2, y: H / 2, moths: [], next: 0.5, saved: 0, lamp: 3, t: 0, hum: 1 }; } })\n```"])
    made = _in_a_browser(lambda browser: maker.mend(made.folder, ask=more.ask, write=more.write, work=Work("w", "x"), browser=browser))
    assert made is not None and made.trial.failing() == [] and "the lamp hums" in json.loads((made.folder / "program.json").read_text())["built"]
    assert any("To be built in now:\n- the lamp hums" in prompt for prompt in more.asked)


def test_a_work_told_to_stop_is_put_down(tmp_path):
    work = Work("w", "moths")
    model = _Model([f"```js\n{TESTS}\n```"])

    async def write(prompt, n, **_how):
        work.stop()
        raise PutDown("it was stopped")

    model.write = write
    with pytest.raises(PutDown):
        _make(tmp_path, model, work)


# ── the workbench ────────────────────────────────────────────────────

def test_a_work_shows_what_happens_and_what_it_waits_on():
    work = Work("w", "a clock")
    work.step("Writing it")
    work.writing("const a = 1;", 4)
    seen = work.seen()
    assert seen["now"]["what"] == "Writing it" and seen["now"]["writing"] == "const a = 1;" and seen["stalled"] == ""
    work.now["last_wrote"] -= 100
    assert "nothing written for" in work.stalled()
    work.wrote_it()
    work.waits("the conversation, which comes first")
    assert "waiting for the conversation" in work.stalled()
    n = work.seen()["happened"][-1]["n"]
    work.trouble("it broke")
    assert [row["what"] for row in work.seen(after=n)["happened"]] == ["trouble"]


def test_a_work_is_held_let_go_spoken_to_and_stopped():
    async def run():
        work = Work("w", "a clock")
        work.hold()
        waiting = asyncio.create_task(work.may_go_on())
        await asyncio.sleep(0.02)
        assert not waiting.done()
        work.go_on()
        await asyncio.wait_for(waiting, 1)
        work.say("bigger numbers")
        assert work.somebody_spoke() and work.heard() == ["bigger numbers"] and not work.somebody_spoke()
        work.stop()
        with pytest.raises(PutDown):
            await work.may_go_on()

    asyncio.run(run())


def test_an_ask_belongs_to_the_work_it_is_made_in():
    work = get_workbench().begin("one", "a clock")
    with working_on(work):
        assert this_work() is work
    assert this_work() is None
    work.finished("done")
    assert get_workbench().begin("one", "a clock") is work and work.state == "going"
    work.finished("done")


def test_what_is_being_written_reaches_whoever_watches_that_request_and_nobody_else():
    from core.brain.llm import what_is_being_written as written

    seen: list[tuple[str, int]] = []
    with written.watched(lambda text, tokens: seen.append((text, tokens))):
        assert written.watch_for("a")
    assert not written.watch_for("b")
    written.pass_on({"id": "a", "written": "const x", "tokens_generated": 3})
    written.pass_on({"id": "b", "written": "private", "tokens_generated": 3})
    written.forget("a")
    written.pass_on({"id": "a", "written": "more", "tokens_generated": 9})
    assert seen == [("const x", 3)]


def test_the_workbench_is_in_her_window_and_its_script_is_whole():
    root = Path(__file__).resolve().parents[1]
    index = (root / "interface/static/index.html").read_text()
    assert 'data-tab="work"' in index and 'id="pane-work"' in index and "workbench.js" in index
    if shutil.which("node"):
        for script in (root / "interface/static/workbench.js", root / "core/making/frame.js"):
            assert subprocess.run(["node", "--check", str(script)], check=False, capture_output=True).returncode == 0, script


def test_the_works_routes_show_and_steer(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from interface.auth import _require_internal
    from interface.routes import work as routes

    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    app.dependency_overrides[_require_internal] = lambda: None
    client = TestClient(app)
    work = get_workbench().begin("seen1", "a clock")
    work.step("Writing it")
    assert "seen1" in [w["id"] for w in client.get("/api/work").json()["works"]]
    got = client.get("/api/work/seen1").json()
    assert got["happened"][-1]["title"] == "Writing it"
    assert client.get(f"/api/work/seen1?after={got['happened'][-1]['n']}").json()["happened"] == []
    assert client.post("/api/work/seen1/hold").json()["state"] == "held"
    assert client.post("/api/work/seen1/go_on").json()["state"] == "going"
    assert client.get("/api/work/none").status_code == 404
    assert "workbench" in client.get("/api/workbench").text
    work.finished("done")


# ── a request carried out by it ──────────────────────────────────────

def _request(tmp_path, said="Could you make me a moth game? Put it in my Documents."):
    from core.understanding.requests_of_her import RequestsOfHer

    ledger = RequestsOfHer(tmp_path / "r.json")
    return ledger, ledger.asked(said, effect="program", conversation="c")


def test_what_is_so_of_a_program_is_told_rule_by_rule_and_where_it_is(tmp_path, monkeypatch):
    from core.making.trying import Result, Trial
    from core.understanding.carriers import building

    _ledger, request = _request(tmp_path)
    home = tmp_path / "home"
    (home / "Documents").mkdir(parents=True)
    (home / "Music").mkdir()
    monkeypatch.setattr(building.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(building, "runs_here", lambda app, **kw: True)
    from core.rebuilding import as_a_mac_app as apps

    def installed(page, name, *, accent=""):
        app = tmp_path / f"{name}.app"
        app.mkdir(exist_ok=True)
        return app

    monkeypatch.setattr(apps, "as_a_mac_app", installed)
    page = tmp_path / "index.html"
    page.write_text("x")
    made = maker.Made(_design(), maker.Parts(), tmp_path, page,
                      Trial([Result("it begins", True), Result("it is drawn", True)], [Result("a moth is saved", True), Result("it ends", False, "still going")]), 60.0)
    came = asyncio.run(building._as_it_came_out(request, made))
    stands = {part.what: part.stands for part in came.parts}
    assert stands["a moth is saved"] == "met" and stands["it ends"] == "unmet" and stands["it is in Documents"] == "met"
    assert (home / "Documents" / "Moth Lamp.app").is_symlink() and not (home / "Music" / "Moth Lamp.app").exists()
    assert came.made["app"].endswith("Moth Lamp.app") and came.again


def test_something_of_theirs_of_the_same_name_is_left(tmp_path):
    from core.understanding.carriers.building import _linked_in

    (tmp_path / "Clock.app").mkdir()
    assert _linked_in(tmp_path, tmp_path / "made" / "Clock.app", "Clock") is None


def test_a_request_told_to_stop_is_set_aside_and_not_reported_as_failed(tmp_path):
    from core.understanding import carrying_out
    from core.understanding.requests_of_her import SET_ASIDE

    ledger, request = _request(tmp_path)

    async def carrier(request, tell):
        raise PutDown("it was stopped")

    asyncio.run(carrying_out.carry_out(request, carrier, tell=lambda line: None, ledger=ledger))
    assert request.stands == SET_ASIDE and request.tries[-1].how == "it was stopped by them"


def test_words_said_to_a_work_reach_it_or_take_the_request_up_again(tmp_path, monkeypatch):
    from core.understanding import carrying_out
    from interface.routes import chat_carrying_out as chat

    ledger, request = _request(tmp_path)
    monkeypatch.setattr(chat, "get_requests_of_her", lambda: ledger)
    work = get_workbench().begin(request.id, "a moth game")
    monkeypatch.setattr(carrying_out, "is_under_way", lambda r: True)
    assert asyncio.run(chat.said_to_work(request.id, "make it green")) == "heard" and work.heard() == ["make it green"]
    begun: list[str] = []
    monkeypatch.setattr(carrying_out, "is_under_way", lambda r: False)
    monkeypatch.setattr(carrying_out, "begin", lambda r, **kw: begun.append(r.effect) or True)
    request.made["page"] = str(tmp_path / "index.html")
    assert asyncio.run(chat.said_to_work(request.id, "and slower")) == "taken up again"
    assert begun == ["program_changed"] and request.added == ["make it green", "and slower"]
    work.finished("done")


def test_she_can_say_how_a_work_goes_from_the_work(tmp_path, monkeypatch):
    from interface.routes import chat_carrying_out as chat

    _ledger, request = _request(tmp_path)
    work = get_workbench().begin(request.id, "a moth game")
    work.step("Mending it: the lamp")
    work.tried("Trying it", [{"name": "it ends", "ok": False}, {"name": "it begins", "ok": True}])
    told = chat._as_it_goes(request)
    assert told["doing_now"] == "Mending it: the lamp" and "failing now: it ends" in told["lately"]
    work.finished("done")
    assert chat._as_it_goes(request) == {}


# ── her own mends, before her model's ─────────────────────────────────

ADDER = '''import sys


def total(rows):
    out = 0
    for row in rows:
        out = out - float(row)
    return out


def main():
    if "--help" in sys.argv:
        print("adds up the numbers it is given, one a line")
        return
    rows = [line for line in sys.stdin.read().split() if line]
    print("total", round(total(rows), 2))
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
'''
ADDS = [{"rule": "adds", "typed": "1\n2\n", "prints": ["total 3.0"]}, {"rule": "adds", "typed": "5\n5\n5\n", "prints": ["total 15.0"]},
        {"rule": "none", "typed": "", "prints": ["total 0"]}]


def _tool_maker(tmp_path, text: str, cases: list, model) -> maker._Maker:
    made = maker._Maker("add these up", ask=model.ask, write=model.write, work=Work("w", "x"), folder=tmp_path, browser=None)
    made.design = _design(name="Adder", kind="tool", language="python", state={}, controls=[], rules=[Rule(says="adds", of="add")])
    made.parts = maker.Parts(text=text, cases=[Case.model_validate(c).model_dump() for c in cases])
    return made


class _Nobody:
    """A model that is not to be asked."""

    def __init__(self):
        self.asked = []

    async def ask(self, prompt, *_a, **_k):
        self.asked.append(prompt)

    async def write(self, prompt, *_a, **_k):
        self.asked.append(prompt)
        return ""


def test_her_own_python_mends_a_tool_before_her_model_is_asked(tmp_path):
    model = _Nobody()
    made = _tool_maker(tmp_path, ADDER, ADDS, model)
    asyncio.run(made.mended())
    assert made.trial.failing() == [] and model.asked == []
    assert "import json" in made.parts.text and "out + float(row)" in made.parts.text
    said = [row["words"] for row in made.work.seen()["happened"] if row["what"] == "says"]
    assert any(words.startswith("Kept a mend of her own, written without her model: added `import json`") for words in said)
    assert json.loads((tmp_path / "program.json").read_text())["parts"]["text"] == made.parts.text


def test_a_change_made_to_fit_one_case_is_not_kept(tmp_path):
    """One case of a rule, and a template that makes it so: nothing says the rule holds anywhere else."""
    model = _Nobody()
    code = ADDER.replace("import sys", "import json\nimport sys").replace("out = out - float(row)", "out = out + float(row)").replace("round(total(rows), 2)", "round(total(rows), 2) - 0")
    made = _tool_maker(tmp_path, code, [{"rule": "adds", "typed": "1\n2\n", "prints": ["total 4.0"]}, ADDS[1]], model)
    asyncio.run(made.mended())
    assert made.parts.text == code and [r.ok for r in made.trial.rules] == [False, True]


def test_what_can_be_read_to_be_wrong_is_told_without_running_it(tmp_path):
    trial = asyncio.run(try_tool(ADDER, languages()["python"], []))
    assert any(r.name == "nothing in it is wrong that can be read" and "json" in r.why and not r.ok for r in trial.generic)


def test_a_piece_above_a_tool_is_never_what_is_mended(tmp_path):
    from core.making import her_own_mends as own

    assert own.ran({"a": {3, 40, 41}}, {"b": {3, 41}}, after=10) == [(40, 1.0), (41, pytest.approx(0.707, abs=0.01))]
    assert list(own.missing_imports("x = 1\n", "name 'nowhere' is not defined")) == []
    (_sort, says, code), = own.missing_imports('"""What it is."""\nfrom __future__ import annotations\nx = 1\n', "NameError: name \\'re\\' is not defined")
    assert says == "added `import re`" and code.splitlines()[2] == "import re"


def test_small_edits_are_guesses_and_the_one_that_gains_is_kept(tmp_path):
    from core.self_modification.edits_her_model_proposes import ProposedEdit, ProposedEdits

    wrong = _RULES.replace("s.saved++", "s.saved += 0")
    work = Work("w", "moths")
    model = _Model([f"```js\n{TESTS}\n```", f"```js\n{wrong}\n```", f"```js\n{_LOOKS}\n```"])
    real = model.ask

    async def ask(prompt, schema, n, *, given=""):
        if schema is not ProposedEdits:
            return await real(prompt, schema, n, given=given)
        line = next(int(row[:4]) for row in prompt.splitlines() if row[4:5] == ":" and "s.saved += 0" in row)
        return ProposedEdits(edits=[ProposedEdit(line=line, old="s.lamp--", new="s.lamp = 0", why="the lamp"),
                                    ProposedEdit(line=line, old="s.nowhere", new="1", why="not there"),
                                    ProposedEdit(line=line, old="s.saved += 0", new="s.saved++", why="a catch did not count.")])

    model.ask = ask
    made = _make(tmp_path, model, work)
    assert made is not None and made.trial.failing() == [] and "s.saved++" in made.parts.code() and "s.lamp = 0" not in made.parts.code()
    assert any(row["what"] == "says" and row["words"].startswith("Kept a small edit: a catch did not count.") for row in work.seen()["happened"])
