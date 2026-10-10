"""Reading a place's rules and lessons for what they ask: each sentence an act of hers, done to what, where to, with
what, when, or never; and a lesson's sentences, in their order, a procedure followed a step at a time.

A person reading "Click here to open the device library. Choose the type of device you wish to use. Click and drag a
device from the library. Place the device at the end of another device's arrow to make a connection. Use the controls
to rotate or delete the device. Continue adding devices until you are ready to test the trap" does not match words
against lists. They understand each sentence (what to do, to what, where), see that together they are the way to work
the place, and do them in order: the library opened, a type chosen, a device dragged to the end of the last one's
arrow, turned, again, and the trap tested. LIVE 2026-10-10 she read those very sentences, matched "drag" against a list,
and carried the library's tab names onto a spot in the room thirty-five times.

So each sentence a place shows (its own words, its page, what was found out about it) is understood once by her own
model, beside her work (core/rebuilding/her_model.py, asked patiently: her thinking at each move holds it), into a
frame whose act is one of her own ways of acting (core/agency/ways_of_playing.py), held so by the decoder, and whose
parts are the sentence's own words. A frame once read is kept for every place: a sentence understood is understood.
Every frame is also an example for the learned surfaces over her model's own representation of sentences
(core/language/learned_matcher.py): "asks to carry something somewhere", "says what the place is for". Those decide
the next sentence like it at once, without asking her model, and their word-list floors only stand until they do.

The frames are used as a person uses rules: what the place says it is for leads her reading of it
(core/cognition/a_guide_to_a_place.py); a sentence that asks to carry says where to; the lesson's next step not yet
done, whose control is on the screen, is what she does first (core/skills/screen_pursuit_decision.py); what a rule
forbids is not done; and the procedure is in what she reasons with.

Nothing here knows a place.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["ASKS_TO_CARRY", "SAYS_WHAT_IT_IS_FOR", "Frame", "Rules", "read_the_rules_beside"]

logger = logging.getLogger("Aura.ReadingTheRules")

#: Acts a sentence may ask for beyond her ways of playing: something only to read, or nothing of hers.
READ, NOTHING = "read", "nothing"
#: How many sentences are read at one ask, the most her model writes for them, the most frames kept for every place,
#: and the fewest words a sentence has to be read as a rule.
AT_ONCE = 8
MOST_TOKENS = 1100
MOST_KEPT = 600
FEWEST_WORDS = 3

_WORD = re.compile(r"[a-z]{3,}")
#: Tries of a step that went unanswered before it is passed over.
PASSED_OVER_AFTER = 2
#: Words two names share without being one thing.
_COMMON = frozenset("the and you your for with from that this into onto any each other another one all".split())


@dataclass(frozen=True)
class Frame:
    """What one sentence asks: an act of hers, done to what, where to, with what control, when, never, and what for;
    whether it says what the place is for; and where it stands in its lesson."""

    sentence: str
    act: str = NOTHING
    thing: str = ""
    where: str = ""
    using: str = ""
    when: str = ""
    never: bool = False
    for_what: str = ""
    is_what_it_is_for: bool = False
    order: int = 0

    def as_memory(self) -> dict[str, Any]:
        held = asdict(self)
        held.pop("order", None)
        return held

    def said(self) -> str:
        """The frame as a step, in a few words."""
        from core.cognition.what_this_place_is import _SAID_AS

        how = _SAID_AS.get(self.act, self.act)
        parts = [("never " if self.never else "") + how]
        parts += [f"({self.thing})" if self.thing else "", f"to {self.where}" if self.where else "",
                  f"with {self.using}" if self.using else "", f"when {self.when}" if self.when else ""]
        return " ".join(p for p in parts if p)


def _words(said: str) -> set[str]:
    return {w.rstrip("s") for w in _WORD.findall(str(said or "").lower()) if w not in _COMMON}


def names_it(name: str, label: str) -> bool:
    """Whether a label on the screen is the thing a rule names: a word of what it is shared ("the device library" and
    "DEVICE LIBRARY"; "test the trap" and "TEST TRAP")."""
    return bool(_words(name) & _words(label))


# -- the learned surfaces ---------------------------------------------------------------------------------------------

def _surface(name: str, positives: tuple[str, ...], negatives: tuple[str, ...]) -> Any:
    from core.language.learned_matcher import LearnedMatcher
    from core.language.model_features import model_hidden_features

    return LearnedMatcher(name=name, positives=positives, negatives=negatives, features=model_hidden_features)


#: Whether a sentence asks for a thing to be carried somewhere: learned over her model's own representation of
#: sentences, from these and from every rule she reads.
ASKS_TO_CARRY = _surface(
    "asks_to_carry",
    ("Drag the file into the folder you want to keep it in.", "Drop a photo onto the canvas to add it.",
     "Move each crate onto a marked square.", "Slide the tiles into the empty slots.",
     "Place your ships anywhere on your half of the grid.", "Drag a widget from the panel onto the page."),
    ("Click the folder to open it.", "Use the arrow keys to move.", "Press space to jump.",
     "Choose a colour from the palette.", "Avoid the walls and collect the stars.", "Press the green button to start."),
)
#: Whether a sentence says what the place is for: its goal, what one is there to do.
SAYS_WHAT_IT_IS_FOR = _surface(
    "says_what_it_is_for",
    ("Your goal is to get every passenger to the right floor.", "Help the farmer gather the sheep into the pen.",
     "The aim is to clear every row before the blocks reach the top.", "Fill in the form to open your account.",
     "Collect all the gems before time runs out."),
    ("Click here to open the menu.", "Use the controls to rotate or delete the piece.", "Press P to pause.",
     "Loading 40%", "Score: 120"),
)


def _decided(surface: Any, sentence: str) -> bool | None:
    try:
        return surface.decide_without_waiting(sentence)
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as why:
        logger.debug("a learned surface could not decide: %s", why)
        return None


# -- what a place's rules are -------------------------------------------------------------------------------------------

@dataclass
class Rules:
    """A place's sentences in the order heard, what each was read as, and which steps of its lesson are done."""

    heard: list[str] = field(default_factory=list)
    frames: dict[str, Frame] = field(default_factory=dict)
    done: set[str] = field(default_factory=set)
    #: How often a move doing each step went unanswered: a step tried this often to no effect is passed over, so a
    #: sentence read wrongly, or a control that does nothing yet, never holds her (PASSED_OVER_AFTER).
    unanswered_steps: dict[str, int] = field(default_factory=dict)
    asking: Any = None
    unanswered: int = 0

    def hear(self, sentences: Iterable[str]) -> list[str]:
        """Sentences the place said, in order: kept, and read at once where a sentence like it was read before.
        The sentences not yet read."""
        store = _kept()
        for sentence in sentences:
            text = " ".join(str(sentence or "").split())
            if len(text.split()) < FEWEST_WORDS or text in self.heard:
                continue
            self.heard.append(text)
            known = store.get(_key(text))
            if known is not None:
                self.frames[text] = Frame(**{**known, "sentence": text, "order": self.heard.index(text)})
        return self.unread()

    def unread(self) -> list[str]:
        return [s for s in self.heard if s not in self.frames]

    def took(self, frames: Sequence[Frame]) -> list[Frame]:
        """Frames her model read: kept for this place and every place, and taught to the learned surfaces."""
        taken = []
        for frame in frames:
            if frame.sentence not in self.heard:
                continue
            frame = Frame(**{**asdict(frame), "order": self.heard.index(frame.sentence)})
            self.frames[frame.sentence] = frame
            taken.append(frame)
            _keep(frame)
            for surface, holds in ((ASKS_TO_CARRY, frame.act == "carry"), (SAYS_WHAT_IT_IS_FOR, frame.is_what_it_is_for)):
                try:
                    surface.observe(frame.sentence, holds=holds)
                except (RuntimeError, OSError, ValueError, TypeError) as why:
                    logger.debug("a learned surface could not take an example: %s", why)
        return taken

    def asks_to_carry(self) -> bool:
        """Whether any rule asks for a thing to be carried: as read, else as the learned surface decides."""
        return any(f.act == "carry" for f in self.frames.values()) or any(
            _decided(ASKS_TO_CARRY, s) for s in self.unread())

    def carried_to(self) -> str:
        """Where the rules say things are carried to, the first that says."""
        return next((f.where for f in self.in_order() if f.act == "carry" and f.where and not f.never), "")

    def what_it_is_for(self) -> list[str]:
        """The sentences that say what the place is for: as read, else as the learned surface decides."""
        read = [f.sentence for f in self.in_order() if f.is_what_it_is_for]
        return read or [s for s in self.unread() if _decided(SAYS_WHAT_IT_IS_FOR, s)]

    def in_order(self) -> list[Frame]:
        return sorted(self.frames.values(), key=lambda f: f.order)

    def steps(self) -> list[Frame]:
        """The lesson as a procedure: what the rules ask to be done, in their order, less the steps passed over."""
        return [f for f in self.in_order() if f.act not in (READ, NOTHING) and not f.never
                and self.unanswered_steps.get(f.sentence, 0) < PASSED_OVER_AFTER]

    def next_step(self) -> Frame | None:
        """The earliest step not yet done."""
        return next((f for f in self.steps() if f.sentence not in self.done), None)

    def step_of(self, move: str) -> Frame | None:
        """The earliest step not yet done that a move does: a click on what a step names, a carry of what it names to
        where it says."""
        from core.agency.acts_on_two_places import CARRY, two_places_of
        from core.agency.what_i_can_do_here import what_is_clicked

        clicked, two = what_is_clicked(move), two_places_of(move)
        for frame in self.steps():
            if frame.sentence in self.done:
                continue
            named = " ".join((frame.thing, frame.using))
            if clicked and frame.act != "carry" and names_it(named, clicked):
                return frame
            if two is not None and two.act == CARRY and frame.act == "carry" and (
                    names_it(frame.where, two.other) or names_it(frame.thing, two.one)):
                return frame
        return None

    def forbids(self, move: str) -> bool:
        """Whether a rule says never to do what a move does."""
        from core.agency.what_i_can_do_here import what_is_clicked

        clicked = what_is_clicked(move) or str(move)
        return any(f.never and names_it(" ".join((f.thing, f.using)), clicked) for f in self.frames.values())

    def tried(self, move: str, changed: bool) -> None:
        """A move made, and whether the screen answered it: a step it does is done once it is answered."""
        frame = self.step_of(move)
        if frame is None:
            return
        if changed:
            self.done.add(frame.sentence)
        else:
            self.unanswered_steps[frame.sentence] = self.unanswered_steps.get(frame.sentence, 0) + 1

    def for_thinking(self) -> str:
        """The lesson as she has read it, for reasoning with: its steps in order, the next marked."""
        steps = self.steps()
        if not steps:
            return ""
        nxt = self.next_step()
        said = [("→ " if f is nxt else "✓ " if f.sentence in self.done else "") + f.said() for f in steps[:8]]
        never = [f.said() for f in self.in_order() if f.never][:3]
        return "The rules, as I read them, in order: " + "; ".join(said) + (". Never: " + "; ".join(never) if never else "")

    def as_memory(self) -> dict[str, Any]:
        return {"heard": self.heard[-40:], "done": sorted(self.done)[:40]}


# -- reading them, beside her work ------------------------------------------------------------------------------------

def read_the_rules_beside(guide: Any, ask: Callable[..., Awaitable[Any]] | None,
                          then: Callable[[list[Frame]], Any] | None = None) -> bool:
    """Ask her model to read the place's sentences not yet read, a few at a time, beside her work; ``then`` told of the
    frames it read. Whether asked."""
    rules: Rules | None = getattr(guide, "rules", None)
    if rules is None or ask is None or (rules.asking is not None and not rules.asking.done()) or rules.unanswered >= 3:
        return False
    sentences = rules.unread()[:AT_ONCE]
    if not sentences:
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False

    async def asked() -> None:
        frames = await _read(sentences, getattr(guide, "place", ""), ask)
        if not frames:
            rules.unanswered += 1
            return
        taken = rules.took(frames)
        logger.info("the rules, as she read them: %s", [(f.sentence[:60], f.act, f.thing, f.where) for f in taken])
        if then is not None:
            then(taken)

    rules.asking = loop.create_task(asked())
    return True


async def _read(sentences: list[str], place: str, ask: Callable[..., Awaitable[Any]]) -> list[Frame]:
    """Her model's reading of sentences into frames, its act held to her ways and its parts to the sentence's words."""
    from typing import Literal

    from pydantic import BaseModel, Field, create_model

    from core.agency.ways_of_playing import WAYS
    from core.cognition.what_things_are import asked_patiently
    from core.cognition.what_this_place_is import _ACTS

    acts = (*_ACTS, READ, NOTHING)
    one: type[BaseModel] = create_model(
        "_Frame",
        number=(int, Field(description="the sentence's number")),
        act=(Literal[acts], Field(description="what it asks the reader to do, as one of the ways listed")),  # type: ignore[valid-type]
        thing=(str, Field(default="", max_length=60, description="what it is done to, in the sentence's own words")),
        where=(str, Field(default="", max_length=60, description="where it is done to or carried to, in its own words")),
        using=(str, Field(default="", max_length=60, description="the control or button it is done with, in its own words")),
        when=(str, Field(default="", max_length=60, description="when, if it says")),
        never=(bool, Field(default=False, description="true where it says not to do it")),
        for_what=(str, Field(default="", max_length=60, description="what it is for, if it says")),
        is_what_it_is_for=(bool, Field(default=False, description="true where it says what the whole place is for")),
    )
    schema = create_model("_Frames", frames=(list[one], Field(default_factory=list, max_length=AT_ONCE)))  # type: ignore[valid-type]
    ways = "; ".join(f"{w.name}: {w.asks}" for w in WAYS if w.name in _ACTS)
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    prompt = (f"These sentences are shown, in this order, by “{place or 'a place on a screen'}” to whoever uses it:\n"
              f"{numbered}\nFor each, what it asks the reader to do, as one of these ways ({ways}; {READ}: only to be "
              f"read; {NOTHING}: asks nothing), what it is done to, where to, with which control, when, whether it says "
              "not to, what for, and whether it says what the whole place is for. Use the sentence's own words for the "
              "parts, and leave a part empty where it does not say.")
    try:
        got = await asked_patiently(ask, prompt, schema, MOST_TOKENS)
    except (RuntimeError, OSError, ValueError, TypeError, TimeoutError) as why:
        logger.info("the rules could not be read: %s", str(why)[:160])
        return []
    out: list[Frame] = []
    for item in getattr(got, "frames", None) or []:
        said = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        at = int(said.get("number") or 0) - 1
        if not 0 <= at < len(sentences):
            continue
        sentence = sentences[at]

        def own(part: str, sentence: str = sentence, said: dict[str, Any] = said) -> str:
            # A part is the sentence's own words or nothing: what is not in it was not read from it.
            text = " ".join(str(said.get(part) or "").split())
            return text if text and text.lower() in sentence.lower() else ""

        out.append(Frame(sentence=sentence, act=str(said.get("act") or NOTHING), thing=own("thing"), where=own("where"),
                         using=own("using"), when=own("when"), never=bool(said.get("never")), for_what=own("for_what"),
                         is_what_it_is_for=bool(said.get("is_what_it_is_for"))))
    return out


# -- kept for every place -------------------------------------------------------------------------------------------

_KEPT: dict[str, dict[str, dict[str, Any]]] = {}
_KEEPING = threading.Lock()


def _key(sentence: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", str(sentence or "").lower()).split())


def _kept() -> dict[str, dict[str, Any]]:
    """Every sentence read before, by its words: one store for her, kept between sittings."""
    from core.runtime.what_she_learned import _kept_in, named, recall

    where = str(_kept_in())
    with _KEEPING:
        if where not in _KEPT:
            try:
                held = recall(named("what holds everywhere", "rules read")) or {}
            except (RuntimeError, OSError, ValueError, TypeError) as why:
                logger.info("the rules read before could not be recalled: %s", why)
                held = {}
            _KEPT[where] = {k: v for k, v in (held.get("frames") or {}).items() if isinstance(v, dict)}
        return _KEPT[where]


def _keep(frame: Frame) -> None:
    from core.runtime.what_she_learned import named, remember

    store = _kept()
    with _KEEPING:
        store[_key(frame.sentence)] = frame.as_memory()
        while len(store) > MOST_KEPT:
            del store[next(iter(store))]
        held = {"frames": dict(store)}
    try:
        remember(named("what holds everywhere", "rules read"), held)
    except (RuntimeError, OSError, ValueError, TypeError) as why:
        logger.info("the rules read could not be kept: %s", why)
