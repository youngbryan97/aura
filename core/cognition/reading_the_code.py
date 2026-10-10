"""Reading the code of the program a thing runs, for how it is played: a player's manual made from its code.

Its screens tell a player some of how a thing works; its code holds all of it. What the code tests to count the thing
won or lost, what each control does, what comes after what, the tools a player is given and what each does, and the
rules no screen states (a part only connects at a lit square; a timer runs out; the arrow sets where the next part
goes). A person who reads code reads it once and plays knowing how the thing works. So does she: the program's
telling part (its screens in order, its parts by name, the statements that decide things) is read by her own model
beside her work, in turn with her other questions (core/cognition/asking_in_turn.py), into a manual. The manual goes
into the guide to the place (core/cognition/a_guide_to_a_place.py) at the program's standing, under everything the
place itself shows her, and is kept, so a program understood once is understood again at once.

It is understanding, not help from outside the game: cheats, debugging keys, passwords and answer keys are left out of
what is asked for and of what is kept.

Nothing here knows a program.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

__all__ = ["read_the_code_beside", "take_in_the_code", "the_telling_part"]

logger = logging.getLogger("Aura.ReadingTheCode")

#: The most of a program put before her model, in characters, and the most her model writes back.
MOST_EXCERPT = 9_000
MOST_TOKENS = 900
#: The most names of its parts, and of its screens, shown.
MOST_NAMES = 160
MOST_LABELS = 80

#: A statement that decides something: a test, a jump to another screen, a collision, a drag, a turn, a count.
_DECIDES = re.compile(r"\bif\b|goto|hittest|drag|_rotation|rotation\s*=|\bkey\.|keycode|\+\+|--|\+=|-=|\bfunction\b|"
                      r"\breturn\b|\bwhile\b|\bfor\b", re.I)
#: Statements that only stop, start or step a timeline say nothing of how a thing is played.
_SAYS_NOTHING = re.compile(r"^(?:stop|play|nextframe|prevframe)\(\)$|^//", re.I)
#: What would play it for her rather than tell her how it works.
_SPOILS = re.compile(r"\b(?:cheats?|debug\w*|passwords?|answer keys?|god ?mode|skip (?:to|level)|unlock (?:all|every)|"
                     r"hack)\b", re.I)
#: The same in code, where it is part of a name: "cheatMode", "debugKey", "skipLevel".
_SPOILS_IN_CODE = re.compile(r"cheat|debug|god_?mode|passw|hack|skip_?level|unlock_?all", re.I)


def the_telling_part(read: Any) -> str:
    """The part of a program read that says how its thing is played: its screens in order (frame labels), its parts by
    name, its words, and the statements that decide things, each under the header of where it is; "" where there is no
    code to show."""
    code = [str(line) for line in getattr(read, "code", []) or []]
    if not code:
        return ""
    labels = list(dict.fromkeys(getattr(read, "labels", []) or []))[:MOST_LABELS]
    names = [n for n in dict.fromkeys(getattr(read, "names", []) or []) if not re.fullmatch(r"\w+_\d+|i_\d+", n)]
    parts = ["Its screens, in order: " + ", ".join(labels) if labels else "",
             "Its parts by name: " + ", ".join(names[:MOST_NAMES]) if names else "",
             "Its words: " + " | ".join(getattr(read, "words", []) or [])[:2000]]
    head = "\n".join(p for p in parts if p)
    room = MOST_EXCERPT - len(head)
    chosen: list[str] = []
    seen: set[str] = set()
    where = ""
    for line in code:
        # Braces are left out: written-out bodies ("{...}") say nothing, and an answer that echoed one was read as empty.
        text = " ".join(line.replace("{...}", "").replace("{", "").replace("}", "").split())
        if text.startswith("//"):
            where = text
            continue
        if not text or text in seen or _SAYS_NOTHING.match(text) or not _DECIDES.search(text) or _SPOILS_IN_CODE.search(text):
            continue
        seen.add(text)
        piece = (f"{where}\n" if where else "") + text
        where = ""
        if room - len(piece) - 1 < 0:
            break
        chosen.append(piece)
        room -= len(piece) + 1
    return head + "\nIts code, the statements that decide things:\n" + "\n".join(chosen)


async def what_the_code_says(place: str, excerpt: str, ask: Callable[..., Awaitable[Any]]) -> dict[str, Any] | None:
    """Her model's player's manual of a program, from its telling part; None where it gave none."""
    from pydantic import BaseModel, Field

    from core.cognition.asking_in_turn import THE_PLACE
    from core.cognition.what_things_are import asked_patiently

    class _Control(BaseModel):
        control: str = Field(max_length=60, description="a key, a button, or what the pointer does")
        does: str = Field(default="", max_length=140, description="what it does")

    class _Tool(BaseModel):
        name: str = Field(max_length=60, description="its name in the program or on its screens")
        does: str = Field(default="", max_length=160, description="what it does when used")

    class _Manual(BaseModel):
        goal: str = Field(default="", max_length=240, description="what a player is there to do")
        win: str = Field(default="", max_length=240, description="what the code checks to count it won or done")
        lose: str = Field(default="", max_length=240, description="what the code counts as lost or failed")
        controls: list[_Control] = Field(default_factory=list, max_length=8)
        steps: list[str] = Field(default_factory=list, max_length=8, description="how to play it, in order")
        progression: list[str] = Field(default_factory=list, max_length=5,
                                       description="what comes after what: rooms, levels, what opens what")
        tools: list[_Tool] = Field(default_factory=list, max_length=8, description="what a player has to work with")
        hidden: list[str] = Field(default_factory=list, max_length=6,
                                  description="how it works that its screens do not say")
        watch_for: list[str] = Field(default_factory=list, max_length=4, description="what to watch for while playing")

    prompt = (f"This is from the program of “{place or 'a thing on a screen'}”, read from its file:\n{excerpt}\n\n"
              "As someone who reads code, write a player's manual from what this code shows: what a player is there to "
              "do, what it checks to count it won and lost, each control and what it does, how to play it in order, "
              "what comes after what, what a player has to work with and what each does, how it works that its screens "
              "do not say, and what to watch for. Leave out cheats, debugging keys, passwords and answer keys. Leave a "
              "part empty where the code does not show it.")
    try:
        got = await asked_patiently(ask, prompt, _Manual, MOST_TOKENS, matters=THE_PLACE)
    except (RuntimeError, OSError, ValueError, TypeError, TimeoutError) as why:
        logger.info("the code of %r could not be read: %s", place, str(why)[:160])
        return None
    manual = _without_spoilers(got.model_dump()) if got is not None and hasattr(got, "model_dump") else None
    return manual if manual and any(manual.values()) else None


def _without_spoilers(manual: dict[str, Any]) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, str):
            return "" if _SPOILS.search(value) else " ".join(value.split())
        if isinstance(value, list):
            return [v for v in (clean(x) for x in value) if v]
        if isinstance(value, dict):
            spoils = any(_SPOILS.search(str(v)) for v in value.values())
            return {} if spoils else {k: " ".join(str(v).split()) for k, v in value.items()}
        return value

    return {k: clean(v) for k, v in manual.items()}


def take_in_the_code(guide: Any, manual: dict[str, Any]) -> list[str]:
    """A manual made from a program's code, into the guide at the program's standing: its goal where the place has
    said none, how it is won and lost, its controls and mechanics as the program's, and the rest as lines for her to
    reason with. What to say of it."""
    from core.cognition.a_guide_to_a_place import PROGRAM

    if not manual:
        return []
    goal, win, lose = (str(manual.get(k) or "") for k in ("goal", "win", "lose"))
    if goal and not guide.goals:
        guide.goals.append(goal)
    for said, kept in ((win, guide.win), (lose, guide.lose)):
        if said and said not in kept:
            kept.append(said)
    controls = [f"{c['control']} to {c['does']}" for c in manual.get("controls") or [] if c and c.get("does")]
    steps, hidden = list(manual.get("steps") or []), list(manual.get("hidden") or [])
    guide.take_in(PROGRAM, [*controls, *steps, *hidden], passages=[])
    tools = [f"{t['name']} ({t['does']})" if t.get("does") else t["name"] for t in manual.get("tools") or [] if t]
    lines = [("How to play it: " + " → ".join(steps)) if steps else "",
             ("What comes after what: " + " | ".join(manual.get("progression") or [])) if manual.get("progression") else "",
             ("What I work with: " + "; ".join(tools)) if tools else "",
             ("What its screens don't say: " + " | ".join(hidden)) if hidden else "",
             ("Watch for: " + " | ".join(manual.get("watch_for") or [])) if manual.get("watch_for") else ""]
    guide.from_its_code = [line for line in lines if line]
    parts = [f"it's won when {_lower(win)}" if win else "", f"it's lost when {_lower(lose)}" if lose else "",
             f"what its screens don't say: {_lower(hidden[0])}" if hidden else ""]
    said = "; ".join(p for p in parts if p)
    return [f"From its code: {said}."] if said else []


def read_the_code_beside(guide: Any, read: Any, ask: Callable[..., Awaitable[Any]] | None,
                         tell: Callable[[str], Any] | None = None) -> bool:
    """The program's manual taken into the guide: at once where it was made before, else asked of her model beside her
    work and kept. Whether there was anything to read."""
    from core.runtime.what_she_learned import named, recall

    place = str(getattr(guide, "place", "") or "")
    if guide is None or not place:
        return False
    world = named("a program understood", place)
    try:
        held = recall(world) or {}
    except (RuntimeError, OSError, ValueError, TypeError):
        held = {}
    if isinstance(held.get("manual"), dict):
        for line in take_in_the_code(guide, held["manual"]):
            if tell is not None:
                tell(line)
        return True
    excerpt = the_telling_part(read)
    if ask is None or not excerpt:
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False

    async def asked() -> None:
        manual = await what_the_code_says(place, excerpt, ask)
        if not manual:
            logger.info("her model gave no manual of the code of %r", place)
            return
        logger.info("the code of %r, as she read it: %s", place, {k: v for k, v in manual.items() if v})
        await asyncio.to_thread(_keep, world, manual)
        for line in take_in_the_code(guide, manual):
            if tell is not None:
                tell(line)

    guide.reading_the_code = loop.create_task(asked())
    return True


def _keep(world: str, manual: dict[str, Any]) -> None:
    from core.runtime.what_she_learned import remember

    remember(world, {"manual": manual})


def _lower(text: str) -> str:
    from core.language.words_of_the_language import as_said_inside

    return as_said_inside(str(text).rstrip("."))
