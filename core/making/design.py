"""What a program is to be, before any of it is written: its kind, what has to be so of it, and what it is made from.

Everything asked for is kept in the person's own words beside the rules it became, and a design that leaves any of it out
is asked for again. The rules are what the program is tested by, so they are things that can be seen to be so.
"""
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = ["APP", "Asked", "Control", "Design", "GAME", "Rule", "SITE", "TOOL", "how_it_is_written", "in_proportion", "left_out", "with_every_ask"]


class Asked(BaseModel):
    asks: list[str] = Field(default_factory=list, max_length=16, description="each thing the program itself is to be or do, in words copied exactly from what they said")
    put: str = Field("", description="where they want it put, in their words; empty where they say nothing of it")


class Rule(BaseModel):
    says: str = Field(description="one thing that is so when it works, that can be seen to be so")
    of: str = Field("", description="the words of theirs it comes from, copied exactly; empty where it is only what anybody would expect")


class Control(BaseModel):
    input: str = Field(description='for a game, a key ("ArrowLeft", "Space", "Enter", "a") or "pointer"; for an app, the label of a button or a field, exactly as it is shown')
    does: str


class Look(BaseModel):
    accent: str = Field("#4c6ef5", description="#rrggbb")
    palette: list[str] = Field(default_factory=list, max_length=6, description="colours as #rrggbb")
    scene: str = Field("", description="what is seen, and how it is drawn or laid out")


class Design(BaseModel):
    name: str = Field(max_length=60, description="what they call it, or a name of its own where they gave none")
    kind: Literal["game", "app", "site", "tool"] = Field(description=(
        '"game": played in a window, drawn, with time passing. "app": used through controls and lists, or drawn on and worked '
        'with. "site": pages to be read and looked around. "tool": run from a command line or called by other code'))
    language: str = Field("", description="for a tool, what it is written in")
    about: str = Field(max_length=300, description="one sentence, for whoever opens it")
    rules: list[Rule] = Field(default_factory=list, max_length=36, description="what has to be so for it to work, the ones it cannot be without first")
    state: dict[str, Any] = Field(default_factory=dict, description="for a game or an app, its state some way in, as an example: every name it has, with one of everything it can hold")
    means: dict[str, str] = Field(default_factory=dict, description="for each name in the state, and each in what it holds, what it is and in what unit: angles in radians from the right and clockwise, distances in pixels, times in seconds")
    controls: list[Control] = Field(default_factory=list, max_length=12)
    functions: list[str] = Field(default_factory=list, max_length=24, description="for a game or an app, the functions it is written in, each doing one thing, as name(s, ...): what it does")
    ends: str = Field("", description="for a game, how it is won or lost; empty where it goes on")
    scored: str = Field("", description="for a game, what is counted; empty where nothing is")
    moves_by_itself: bool = Field(False, description="for an app, whether time passing changes it (a timer, a simulation)")
    pieces: list[str] = Field(default_factory=list, max_length=14, description="the pieces it is made from, by name")
    look: Look = Field(default_factory=Look)

    def said(self) -> str:
        """The design as whoever writes it reads it."""
        lines = [f"{self.name}: {self.about}", "Rules:", *(f"  - {rule.says}" for rule in self.rules)]
        if self.state:
            lines.append(f"State, some way in, with every name it has and no other: {json.dumps(self.state, ensure_ascii=False)}")
        if self.means:
            lines += ["where", *(f"  {name}: {what}" for name, what in self.means.items())]
        if self.functions:
            lines += ["Written in these functions, each short, called as this.name(...):", *(f"  {f}" for f in self.functions)]
        if self.controls:
            lines += ["Controls, by the labels they have:" if self.kind == "app" else "Controls:",
                      *(f"  {control.input}: {control.does}" for control in self.controls)]
        for label, text in (("Ends", self.ends), ("Counts", self.scored), ("Looks", self.look.scene)):
            if text:
                lines.append(f"{label}: {text}")
        if self.look.palette:
            lines.append("Colours: " + " ".join(self.look.palette))
        return "\n".join(lines)


def left_out(design: Design) -> list[str]:
    """What a design lacks to be one."""
    wrong = []
    if not design.rules:
        wrong.append("it has no rules")
    if design.kind == "game" and not design.controls:
        wrong.append("it has no controls")
    if design.kind in {"game", "app"} and not design.state:
        wrong.append("it has no state")
    return wrong


def with_every_ask(design: Design, asks: list[str]) -> Design:
    """The design with everything they asked for among its rules: what no rule is of is a rule in their own words."""
    from core.understanding.carriers.finding import quoted_in

    for ask in asks:
        if not any(rule.of and (quoted_in(ask, rule.of) or quoted_in(rule.of, ask)) for rule in design.rules):
            design.rules.append(Rule(says=ask, of=ask))
    return design


def in_proportion(design: Design, asks: list[str]) -> list[Rule]:
    """The rules, as many as what was asked bears: a few for each thing asked, and a few of what anybody would expect. A
    small request given forty rules is an hour of somebody's evening."""
    from core.understanding.carriers.finding import quoted_in

    kept: list[Rule] = []
    of_each: dict[str, int] = {}
    for rule in design.rules:
        key = next((ask for ask in asks if rule.of and (quoted_in(ask, rule.of) or quoted_in(rule.of, ask))), "")
        if of_each.get(key, 0) < A_FEW:
            of_each[key] = of_each.get(key, 0) + 1
            kept.append(rule)
    return kept


#: How many rules a design has for each thing they asked, and for all that nobody asked and anybody would expect.
A_FEW = 5

_CHANCE = ("Use rnd() between(a,b) int(a,b) pick(list) shuffle(list) for chance, never Math.random. play(name) makes a sound: blip "
           "pick hit win lose step shoot jump bell tick.")

GAME = f"""A game is one JavaScript object. Its state is plain data: numbers, strings, lists and objects, no functions.
  initial()       the state as play begins
  update(s, e)    changes s for one event e. e.type is "tick" (60 a second: e.dt seconds, e.held such as {{ArrowLeft: true}}, e.pointer {{x, y, down}}), "key" (e.key, e.down true or false) or "pointer" (e.x, e.y, e.down, e.moved)
  score(s)        a number, where something is counted
  over(s)         false while it goes on, then {{won: true or false, say: "..."}}
  hud(s)          [{{label, value}}] shown above it
  draw(s, g)      draws s and changes nothing. Anything longer is split into functions beside these, called as this.name(...). The world is W=960 wide and H=600 high with 0,0 at the top left
Keys: "ArrowLeft" "ArrowRight" "ArrowUp" "ArrowDown" "Space" "Enter" or a lower-case letter. {_CHANCE}
The window, the start, pause and end screens, the score and the best score are there already."""

PEN = """g.clear(color) g.sky(top, bottom) g.rect(x,y,w,h,color,{r,stroke,width,alpha,glow}) g.circle(x,y,r,color,{..}) g.ellipse(x,y,rx,ry,color,{turn}) g.line(x1,y1,x2,y2,color,width) g.poly([[x,y],..],color,{..}) g.wedge(x,y,r,fromAngle,toAngle,color,{solid}) fades outward unless solid, g.text(words,x,y,{size,color,align,weight}) g.at(x,y,angle,g => ...) draws turned about a place, g.glow(x,y,r,color,strength) is light added. g.screen(g => ...) draws in the view's own place while the world is moved (a sky, a frame, words). g.t seconds, g.ctx the canvas context. A color of null with {stroke} draws an outline."""

APP = f"""An app is one JavaScript object. Its state is plain data: numbers, strings, lists and objects, no functions. It is kept between openings, and Undo is there already.
  initial()       the state as it begins, before anybody has used it: empty of records
  update(s, e)    changes s for one event e: {{type:"do", name, value}} from a button, {{type:"input", name, value}} from a field, {{type:"key", key, down}}, and {{type:"tick", dt, now}} ten times a second where it moves by itself (now is milliseconds)
  view(s, ui)     what is shown for s, changing nothing: a tree of ui parts. Anything longer is split into functions beside these, called as this.name(...).
ui.column(..) ui.row(..) ui.card(..) ui.grid(..) ui.title(t) ui.heading(t) ui.text(t) ui.note(t) ui.big(t) ui.badge(t,"good|bad|warm") ui.divider() ui.spacer()
ui.button(label, name, value, {{tone:"primary|danger", disabled, hint}}) ui.toggle(label, name, value, on) sends "do"
ui.input(label, name, value, {{placeholder, enter:"name sent on Enter", type}}) ui.number(label,name,value,{{min,max,step}}) ui.date(label,name,"2026-11-23") ui.area(label,name,value,{{rows}}) ui.check(label,name,on) ui.select(label,name,value,[value or [value,label]]) ui.slider(label,name,value,{{min,max,step}}) send "input"
ui.list(items, (item, i) => part) ui.table(heads, rows of cells) ui.tabs(names, chosen, name) ui.progress(value, of) ui.bars([{{label,value,show}}]) ui.lines([numbers]) ui.canvas(name, w, h, (g, w, h) => ...) sends "do" with {{x, y}} when pressed
Labels of buttons that stand together differ, so each can be told from the rest. {_CHANCE}"""

SITE = """A site is one HTML document body, with a <style> of its own and, where needed, one <script>. No address outside the page is used: pictures are drawn as inline SVG, fonts are the system's. Every link goes to a part of the page that is there. It reads well at 400 and at 1200 pixels wide, in light and in dark."""

TOOL = """A tool is one file of {language} that is run as: {run}. It reads its arguments and its standard input, writes its answer to standard output and what went wrong to standard error, and leaves with 0 when it did what was asked and not 0 when it did not. With --help it says how it is used. It uses no network and nothing that has to be installed: {has}."""


#: The pieces a game is drawn with, which are told to whoever writes one: what is drawn with them is drawn well.
DRAWN_WITH = ("scene", "figures", "light", "weather", "meters", "trails", "cloth")


def how_it_is_written(design: Design) -> str:
    return {"game": GAME, "app": APP, "site": SITE}.get(design.kind, "")
