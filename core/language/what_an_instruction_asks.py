"""What an instruction asks the reader to do, and to what: read off its clauses.

A game's rules are instructions to the player, and they are written the same
way everywhere: "Catch the fruit. Not the bombs." "Dodge food by pressing the
space bar." "Click the orange targets. Do not hit the green ones." "Keep the
ball from getting past you." Each clause asks for one kind of act toward one
kind of thing, or forbids it, and sometimes names the control that does it.

This reads exactly that and nothing more. The act is matched against declared
families, the way `action_semantics` matches a commitment; the polarity is the
imperative's own (a bare verb asks, "do not", "never" and "don't" forbid, and a
clause that opens with "not" forbids the last act again for a new thing); the
thing is the noun phrase after the act. It abstains on a clause it cannot
place, so what it returns is what the words said.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

__all__ = ["ACT_FAMILIES", "Instruction", "what_the_words_ask"]

#: Kinds of act a player is asked for, each a set of phrasings.
ACT_FAMILIES: Final[dict[str, frozenset[tuple[str, ...]]]] = {
    "get": frozenset({
        ("catch",), ("collect",), ("grab",), ("get",), ("pick", "up"), ("eat",), ("gather",),
        ("save",), ("rescue",), ("reach",), ("find",), ("match",), ("feed",),
    }),
    "keep clear": frozenset({
        ("avoid",), ("dodge",), ("keep", "away", "from"), ("stay", "away", "from"), ("escape",),
        ("run", "from"), ("watch", "out", "for"), ("look", "out", "for"), ("duck",), ("evade",),
    }),
    "stop": frozenset({
        ("keep",), ("stop",), ("block",), ("defend",), ("protect",), ("guard",), ("return",),
    }),
    "hit": frozenset({
        ("shoot",), ("hit",), ("splat",), ("zap",), ("blast",), ("throw", "at"), ("fire", "at"),
        ("attack",), ("smash",), ("destroy",), ("defeat",), ("knock", "out"), ("bump",), ("pop",),
        ("whack",), ("squash",), ("beat",), ("throw",), ("fire",),
        # Blows at close range: nothing flies, what is close is struck.
        ("punch",), ("kick",), ("strike",), ("slash",), ("slice",), ("stab",), ("chop",), ("stomp",), ("swat",),
        ("melee",), ("melee", "attack"),
    }),
    "click": frozenset({("click", "on"), ("click",), ("tap",), ("press", "on")}),
    "move": frozenset({("move",), ("steer",), ("walk",), ("drive",), ("fly",), ("guide",), ("control",)}),
    "jump": frozenset({("jump",), ("hop",), ("leap",)}),
    # Holding to build something up, let go to use it: "hold X to charge up", "press and hold Z to power up".
    "charge": frozenset({("charge",), ("charge", "up"), ("power", "up"), ("wind", "up")}),
    # Going over all of a place: painting it, mowing it, filling it in, exploring it. The ground is the point.
    "cover": frozenset({
        ("cover",), ("paint",), ("coat",), ("fill", "in"), ("colour", "in"), ("color", "in"), ("mow",), ("clean", "up"),
        ("explore",), ("visit", "every"), ("visit", "all"), ("spray",), ("sweep",),
    }),
}

#: Words that name a control, and the control they name.
_CONTROLS: Final[tuple[tuple[tuple[str, ...], str], ...]] = (
    (("space", "bar"), "space"), (("spacebar",), "space"), (("space",), "space"),
    (("arrow", "keys"), "arrows"), (("arrows",), "arrows"), (("arrow",), "arrows"),
    (("mouse",), "mouse"), (("cursor",), "mouse"), (("click",), "mouse"),
    (("enter",), "return"), (("return",), "return"), (("shift",), "shift"),
    (("up",), "up"), (("down",), "down"), (("left",), "left"), (("right",), "right"),
)

_FORBIDDING: Final = frozenset({"not", "never", "dont", "no"})

#: What a thing does to the player, said of it as its subject: "rocks cost a
#: life", "the red ones hurt", "coins are worth five points".
_COSTS: Final = frozenset({"cost", "costs", "hurt", "hurts", "kill", "kills", "damage", "damages", "harm", "harms", "lose", "loses", "ends", "end"})
_PAYS: Final = frozenset({"worth", "gives", "give", "earns", "earn", "scores", "score", "adds", "add", "heals", "heal"})
_NUMBERS: Final = frozenset({"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "times", "points", "point", "wins", "win", "seconds"})
_DETERMINERS: Final = frozenset({"the", "a", "an", "all", "any", "every", "each", "your", "their", "its", "those", "these", "some", "of", "as", "many", "much"})
_ENDS_A_THING: Final = frozenset({
    "by", "with", "using", "to", "and", "or", "before", "while", "when", "until", "from", "in", "on",
    "at", "into", "for", "so", "if", "that", "past", "getting", "get", "away",
})
_CLAUSES: Final = re.compile(r"[.!?;:]+|,|\b(?:but|and then|then)\b")


@dataclass(frozen=True, slots=True)
class Instruction:
    """One clause's ask: an act, the thing it is toward, whether it is forbidden, and its control."""

    act: str
    thing: tuple[str, ...]
    forbidden: bool
    control: str
    clause: str


def _plain(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii").casefold()
    value = value.replace("don't", "dont").replace("do not", "dont").replace("’", "'")
    return re.sub(r"\s+", " ", value).strip()


def _words(clause: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", clause)


#: Words that finish a verb when it is split from them by what it is done to: "knock the tops out", "pick the coins up".
_PARTICLES: Final = frozenset({"out", "up", "down", "off", "away", "over", "in", "back"})


def _as_said(word: str, before: str = "") -> set[str]:
    """The plain forms a word may be of an act's verb: "knocking" is knock, "dodged" is dodge. A form that follows a
    word like "the" describes a thing ("the flying saucers"), and is said only as itself."""
    forms = {word}
    if before in _DETERMINERS or word in _ENDS_A_THING:
        return forms
    for ending in ("ing", "ed"):
        if word.endswith(ending) and len(word) > len(ending) + 2:
            base = word[: -len(ending)]
            forms |= {base, base + "e"}
            if len(base) > 2 and base[-1] == base[-2]:
                forms.add(base[:-1])                                            # running is run, stopped is stop
    return forms


def _act_at(words: list[str], index: int) -> tuple[str, int] | None:
    """The act a clause names at ``index``, in any of its forms, and how many words of the clause it takes before what it
    is done to: a verb whose last word is split from it ("knock the other tops out") takes only itself."""
    best: tuple[str, int] | None = None
    forms = _as_said(words[index], words[index - 1] if index else "")
    for family, phrasings in ACT_FAMILIES.items():
        for phrase in phrasings:
            if phrase[0] not in forms:
                continue
            if words[index + 1 : index + len(phrase)] == list(phrase[1:]):
                taken = len(phrase)
            elif len(phrase) == 2 and phrase[1] in _PARTICLES and phrase[1] in words[index + 2 : index + 8]:
                taken = 1
            else:
                continue
            if best is None or len(phrase) > best[1]:
                best = (family, taken)
    return best


def _the_thing(words: list[str], start: int, stop: int | None = None) -> tuple[str, ...]:
    thing: list[str] = []
    for word in words[start : min(stop if stop is not None else len(words), start + 8)]:
        if len(word) == 1 and word.isalpha() and word not in _DETERMINERS:
            break                                                        # a letter here names the next act's key
        if (word in _ENDS_A_THING or word in _PARTICLES or word in _NUMBERS or word.isdigit()) and thing:
            break
        if word in _NUMBERS or word.isdigit():
            continue
        if word in _DETERMINERS or word in _ENDS_A_THING:
            continue
        thing.append(word)
        if len(thing) >= 4:
            break
    return tuple(thing)


#: Words around a single letter that make it the name of a key: "the Z key", "press X", "hold down C".
_KEY_WORDS: Final = frozenset({"key", "keys", "button"})
_PRESSING: Final = frozenset({"press", "pressing", "presses", "hold", "holding", "tap", "tapping", "hit", "use"})


def _a_letter_key(words: list[str]) -> str:
    """A letter named as a key: followed by "key", after a pressing verb, or before "to" and an act ("D to kick")."""
    for index, word in enumerate(words):
        if len(word) != 1 or not word.isalpha():
            continue
        after = words[index + 1] if index + 1 < len(words) else ""
        before = [w for w in words[max(0, index - 3) : index] if w not in ("the", "down", "on")]
        if after in _KEY_WORDS or (before and before[-1] in _PRESSING and word not in ("a", "i")):
            return word
        if after == "to" and word not in ("a", "i") and (index + 2 >= len(words) or _act_at(words, index + 2) is not None):
            return word
    return ""


def _the_control(words: list[str]) -> str:
    letter = _a_letter_key(words)
    if letter:
        return letter
    for phrase, control in _CONTROLS:
        for index in range(len(words)):
            if words[index : index + len(phrase)] == list(phrase):
                # "up" and "down" name keys only beside a word for keys or a
                # pressing verb; otherwise they are directions in a sentence.
                if control in ("up", "down", "left", "right"):
                    around = set(words[max(0, index - 2) : index + 3])
                    if not around & {"key", "keys", "arrow", "arrows", "press", "pressing"}:
                        continue
                return control
    return ""


def _what_it_does_to_her(words: list[str], clause: str) -> Instruction | None:
    """A thing said to cost or pay: its subject is to be kept clear of, or got."""
    for index, word in enumerate(words):
        if word in _COSTS or word in _PAYS:
            subject = tuple(w for w in words[:index] if w not in _DETERMINERS and w not in _FORBIDDING and not w.isdigit())[-3:]
            if subject:
                return Instruction("keep clear" if word in _COSTS else "get", subject, False, "", clause)
    return None


def _acts_in(words: list[str]) -> list[tuple[int, str, int]]:
    acts = []
    index = 0
    while index < len(words):
        act = _act_at(words, index)
        if act is not None:
            acts.append((index, act[0], act[1]))
            index += act[1]
        else:
            index += 1
    return acts


def what_the_words_ask(text: str) -> list[Instruction]:
    """Every clause of ``text`` that asks for, or forbids, an act toward a thing."""
    asked: list[Instruction] = []
    last: Instruction | None = None
    for clause in (part.strip() for part in _CLAUSES.split(_plain(text))):
        words = _words(clause)
        if not words:
            continue
        acts = _acts_in(words)
        if not acts:
            if last is not None and words[0] in _FORBIDDING:
                # "Not the bombs." asks the last act again, forbidden, of a new thing.
                thing = _the_thing(words, 1)
                if thing:
                    asked.append(Instruction(last.act, thing, not last.forbidden, last.control, clause))
                continue
            consequence = _what_it_does_to_her(words, clause)
            if consequence is not None:
                asked.append(consequence)
            continue
        control = _the_control(words)
        for place, (index, family, length) in enumerate(acts):
            ends = acts[place + 1][0] if place + 1 < len(acts) else None
            forbidden = any(word in _FORBIDDING for word in words[max(0, index - 3) : index])
            thing = _the_thing(words, index + length, ends)
            if family == "stop" and "past" in words:
                # "Keep the ball from getting past you": the thing is to be met.
                family = "get"
            # "Space to fire", "click to throw": the control is what does it. Where one clause gives two acts their
            # own keys ("S to punch, D to kick"), each act's is the one named since the act before it.
            since = acts[place - 1][0] + acts[place - 1][2] if place else 0
            does_it = ((_the_control(words[since:index]) or control) if (index >= 1 and words[index - 1] == "to") or not thing
                       else _the_control(words[max(0, index - 6) : (ends or len(words))]))
            # Going over a place names the place before the act as often as after it ("give the whole
            # campground a coat of paint"), or not at all: it stands with no thing named.
            if not thing and not does_it and family != "cover":
                continue
            last = Instruction(family, thing, forbidden, does_it, clause)
            asked.append(last)
    return asked
