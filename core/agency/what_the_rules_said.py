"""What a game's own words said about playing it, held ready for her fast loop.

The words are read by the language substrate
(core/language/what_an_instruction_asks.py) into instructions: an act, the
thing it is toward, whether it is forbidden, the control that does it. This
turns those into what the fast loop can use without stopping to read:

- a stance for each kind of thing the words can be tied to. The tie is a
  colour named in the words and measured on the kind ("click the orange
  targets"), because a colour is a word and a measurement at once. A thing
  named only by a noun ("the bombs") is tied by play, not guessed.
- a role for each named control: the key that dodges, the key that fires,
  whether a click is a shot.
- whether some thing the words warned about is still unnamed, which is a
  reason for care with kinds nobody has explained yet.

Nothing here knows a game. A rule read wrongly is overruled by what play
measures (core/agency/what_meeting_things_does.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.agency.what_meeting_things_does import AVOID, CLICK, MEET, SHOOT
from core.language.what_an_instruction_asks import Instruction, what_the_words_ask

__all__ = ["WhatTheRulesSaid"]

#: The colour words a kind can be tied by: the names her eyes give colours.
_COLOUR_WORDS = (
    "black", "white", "grey", "gray", "red", "orange", "yellow", "green",
    "blue", "cyan", "purple", "pink", "brown",
)

#: Which keys a named control means.
_KEYS_OF = {"space": ("space",), "return": ("return",), "shift": ("shift",), "up": ("up",),
            "down": ("down",), "left": ("left",), "right": ("right",),
            "arrows": ("up", "down", "left", "right")}


def _stance(instruction: Instruction) -> str | None:
    by_mouse = instruction.control == "mouse"
    asked = {
        "get": MEET,
        "keep clear": AVOID,
        "hit": CLICK if by_mouse else SHOOT,
        "click": CLICK,
    }.get(instruction.act)
    if asked is None:
        return None
    if instruction.forbidden:
        return AVOID
    return asked


@dataclass
class WhatTheRulesSaid:
    """The instructions in a game's words, and what follows from them for play."""

    words: str = ""
    instructions: list[Instruction] = field(default_factory=list)

    @classmethod
    def read(cls, words: str) -> WhatTheRulesSaid:
        return cls(words=words, instructions=what_the_words_ask(words))

    def stance_for_colour(self, colour: str) -> str | None:
        """What the words say to do about things of this colour, if they name it."""
        named = "grey" if colour == "gray" else colour
        for instruction in self.instructions:
            words = {"grey" if w == "gray" else w for w in instruction.thing}
            if named in words:
                stance = _stance(instruction)
                if stance is not None:
                    return stance
        return None

    def keys_for(self, *acts: str) -> tuple[str, ...]:
        """The keys the words give to any of these acts."""
        keys: list[str] = []
        for instruction in self.instructions:
            if instruction.act in acts and not instruction.forbidden:
                keys.extend(key for key in _KEYS_OF.get(instruction.control, ()) if key not in keys)
        return tuple(keys)

    @property
    def dodge_keys(self) -> tuple[str, ...]:
        return self.keys_for("keep clear", "jump")

    @property
    def fire_keys(self) -> tuple[str, ...]:
        return self.keys_for("hit")

    @property
    def covers(self) -> bool:
        """Whether the words ask for a place to be gone over, all of it: painted, mown, filled in, explored."""
        return any(i.act == "cover" and not i.forbidden for i in self.instructions)

    @property
    def a_click_is_a_shot(self) -> bool:
        return any(i.control == "mouse" and i.act in ("hit", "click") and not i.forbidden for i in self.instructions)

    @property
    def warned_of_something_unnamed(self) -> bool:
        """Whether the words warn about a thing no colour ties to anything."""
        return any(
            _stance(i) == AVOID and not any(w in _COLOUR_WORDS for w in i.thing) and i.thing
            for i in self.instructions
        )

    def said(self) -> str:
        """The instructions as she understood them, one short line, for a watcher."""
        parts = []
        for instruction in self.instructions[:4]:
            act = {"get": "get", "keep clear": "keep clear of", "hit": "hit", "click": "click",
                   "move": "move", "jump": "jump", "stop": "stop"}.get(instruction.act, instruction.act)
            thing = " ".join(instruction.thing) or "it"
            line = f"{'not ' if instruction.forbidden else ''}{act} {thing}"
            if instruction.control:
                line += f" ({instruction.control})"
            parts.append(line)
        return "; ".join(parts)
