"""The controls a game's own words name: which keys, and whether the pointer.

"Use the arrow keys to move and space to jump" names five keys; "Move the
mouse to aim, click to throw" names the pointer; "USE WASD TO MOVE" names four
letters as one word; "grenades (activated with the X key)" names one. A word
for a way ("up", "left") is a key only beside a word for keys or pressing:
"when the hamster lines up with the pillow" names no key.

Nothing here knows a game.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from core.agency.which_one_answers_to_her import WAYS

__all__ = ["a_legend_of_controls", "controls_named_in", "without_a_legend"]

#: Words that say a game is played with the pointer.
_POINTER_WORDS = ("mouse", "cursor", "pointer", "click", "drag", "aim", "trackpad")

#: What the usual names of keys mean.
_NAMED_KEYS = (
    (("arrow", "arrows", "cursor keys", "direction"), ("up", "down", "left", "right")),
    # The clusters of letters keyboards are steered by, named as one word: LIVE 2026-10-09 "USE WASD TO MOVE" was
    # read as no keys at all, and she played a game steered by keys with the pointer for three minutes.
    (("wasd",), ("w", "a", "s", "d")),
    (("ijkl",), ("i", "j", "k", "l")),
    (("esdf",), ("e", "s", "d", "f")),
    (("zqsd",), ("z", "q", "s", "d")),
    (("space", "spacebar", "space bar"), ("space",)),
    (("up",), ("up",)),
    (("down",), ("down",)),
    (("left",), ("left",)),
    (("right",), ("right",)),
    (("enter", "return"), ("return",)),
    (("shift",), ("shift",)),
)


#: Words beside a way that say a key is meant by it.
_KEY_CUES = frozenset({"key", "keys", "arrow", "arrows", "press", "pressing", "hold", "holding", "tap", "hit", "push"})


#: Words just after a way's name that say it is how a button or the pointer is pressed, not a key: "hold down the mouse
#: button", "press down on the ball". LIVE 2026-10-09 "hold down the mouse while you aim" was read as the down arrow,
#: and a game played with the mouse was taken to be played with keys.
_NOT_A_KEY_AFTER = frozenset({"mouse", "button", "on"})


def _a_key_is_meant(lowered: str, way: str) -> bool:
    import re

    words = re.findall(r"[a-z]+", lowered)
    if any(word == way and _KEY_CUES & set(words[max(0, at - 2):at + 3]) and not _NOT_A_KEY_AFTER & set(words[at + 1:at + 4])
           and not _holding_down_another(words, at) for at, word in enumerate(words)):
        return True
    return any(_for_an_act(re.findall(r"[a-z0-9]+", clause), way) for clause in re.split(r"[.,;:!?()]", lowered))


def _holding_down_another(words: list[str], at: int) -> bool:
    """Whether "down" at ``at`` is how another key is pressed, not a key: "hold down the X key", "press down space"."""
    return (words[at] == "down" and at > 0 and words[at - 1] in ("hold", "holding", "press", "pressing", "push")
            and bool({"key", "keys", "space", "spacebar", "shift", "enter", "return"} & set(words[at + 1:at + 4])))


def _for_an_act(words: list[str], way: str) -> bool:
    """Whether a clause opens with ways said as what an act is done with: "left and right to move", "up to jump".

    "Up to 5 players" and "from left to right" are not, for what follows "to" there is no act of play; nor is "turn
    left to go home", where the way is where to go and not what to press.
    """
    from core.language.what_an_instruction_asks import ACT_FAMILIES

    if way not in words[:4] or any(w not in WAYS and w not in ("and", "or") for w in words[: words.index(way)]):
        return False
    after = words.index(way)
    while after < len(words) and (words[after] in WAYS or words[after] in ("and", "or")):
        after += 1
    if after + 1 >= len(words) or words[after] != "to":
        return False
    acts = {phrase[0] for family in ACT_FAMILIES.values() for phrase in family}
    return words[after + 1] in acts | {"go", "run", "turn", "rotate", "crouch", "climb", "slide", "accelerate", "brake"}


#: A single letter named as a key: "the X key", "X key"; or a capital after a word for pressing, or before what it is
#: for: "press Z", "Z to shoot". Read in the case it was written: "press a button" names no key called A.
_A_LETTER_KEY = re.compile(
    r"\b(?:the\s+)?([A-Za-z])\s+key\b"
    r"|\b(?:press|hit|tap|hold|use|push)\s+(?:the\s+)?[\"'“]?([A-Z])[\"'”]?(?![\w'])(?:\s+(?:key\s+)?to\s+([a-z]+))?"
    r"|(?<![\w'])([A-Z])\s+to\s+([a-z]+)\b")

#: What a key pressed only to begin or begin again is for: the menu's, not play's.
_TO_BEGIN = frozenset({"start", "begin", "restart", "continue", "play", "pause", "quit"})


def _letter_keys(text: str, *, during_play: bool = False) -> list[str]:
    """Letter keys a game's own words name (LIVE 2026-10-09 "grenades (activated with the X key)" was not read as one)."""
    keys: list[str] = []
    for match in _A_LETTER_KEY.finditer(text):
        letter = match.group(1) or match.group(2) or match.group(4)
        purpose = (match.group(3) or match.group(5) or "").lower()
        if not letter or (match.group(4) and letter in "IA"):
            continue
        if during_play and purpose in _TO_BEGIN:
            continue
        if letter.lower() not in keys:
            keys.append(letter.lower())
    return keys


#: "space" said where a key is meant: the space bar, a key word or a press beside it, or what it is for after it ("Space:
#: jump", "space to fire"). LIVE 2026-10-09 "through treacherous space hazards" made space a key of a lander game.
_SPACE_A_KEY = re.compile(r"\bspace ?bar\b|\bspace\s*(?:key|button)\b|\b(?:press|hit|tap|hold|push|use)\s+(?:the\s+)?space\b|"
                          r"(?<!outer )(?<!deep )(?<!open )(?<!empty )(?<!free )(?<!of )(?<!into )(?<!through )"
                          r"\bspace\s*(?::|=|-|to\s+\w+|for\s+\w+|and\s+(?:the\s+)?(?:arrow|left|right|up|down|enter|shift))|"
                          r"(?:,|and|or)\s+space\b|^\s*space\b")
#: LIVE-shaped 2026-10-10: "navigating the dangers of outer space to secure a safe landing" is a place, not a key.


def _space_is_a_key(lowered: str) -> bool:
    return bool(_SPACE_A_KEY.search(lowered))


def controls_named_in(text: str, *, keys_without_words: Sequence[str] = ("up", "down", "left", "right", "space"),
                      during_play: bool = False) -> tuple[list[str], bool]:
    """The keys a game's own words name, and whether they name the pointer.

    "Use the arrow keys to move and space to jump" names five keys; "Move the
    mouse to aim, click to throw" names the pointer. With no keys named, the
    keys most games use are tried, after the pointer when it is named.
    """
    import re

    from core.runtime.watched_goal import keys_named_in

    lowered = " ".join(str(text or "").lower().split())
    if during_play:
        # A lifecycle command belongs to the menu, rather than to the active
        # controls. Keep other uses of the same key, such as space to jump.
        lowered = re.sub(r"\b(?:press|tap|hit)\s+[^.!?]{0,40}?\b(?:to\s+)?"
                         r"(?:start|begin|restart|play\s+again)\b", "", lowered)
    words = set(re.findall(r"[a-z]+", lowered))
    # Restrict generic arrow instructions only when directions qualify the
    # arrows themselves. "Pick up coins" says nothing about which arrows work.
    arrow_directions = re.findall(
        r"\b((?:(?:up|down|left|right)\s*(?:[,/&+-]|\band\b|\bor\b)?\s*)+)"
        r"(?:arrows?\b|arrow\s+keys?\b|cursor\s+keys?\b)", lowered)
    keys: list[str] = []
    for names, meant in _NAMED_KEYS:
        if meant == ("up", "down", "left", "right") and arrow_directions:
            continue
        if meant[0] in WAYS and len(meant) == 1:
            # A way said alone is a key only beside a word for keys or pressing: LIVE 2026-10-07 "when the hamster
            # lines up with the pillow" was read as the up key, and "left-click fires" as the left one.
            qualified = meant[0] in re.findall(r"up|down|left|right", " ".join(arrow_directions))
            if qualified or re.search(rf"\b{meant[0]}\b(?!-?\s?click)", lowered) and _a_key_is_meant(lowered, meant[0]):
                keys.extend(key for key in meant if key not in keys)
            continue
        if meant == ("space",) and not _space_is_a_key(lowered):
            continue
        if any((name in words) if " " not in name else (name in lowered) for name in names):
            keys.extend(key for key in meant if key not in keys)
    keys.extend(key for key in keys_named_in(lowered) if key not in keys)
    keys.extend(key for key in _letter_keys(str(text or ""), during_play=during_play) if key not in keys)
    pointer = any(word in words or word + "s" in words for word in _POINTER_WORDS)
    if not keys:
        keys = list(keys_without_words)
    return keys, pointer


#: What a control is called by what it does, as legends name them beside pictures of keys, and the key that by custom
#: does it where the picture cannot be read. Two words first: "turn left" before "left".
_FUNCTIONS: tuple[tuple[str, str], ...] = (
    ("turn left", "left"), ("turn right", "right"), ("steer left", "left"), ("steer right", "right"),
    ("rotate left", "left"), ("rotate right", "right"), ("move left", "left"), ("move right", "right"),
    ("move up", "up"), ("move down", "down"), ("go left", "left"), ("go right", "right"),
    ("forward", "up"), ("forwards", "up"), ("accelerate", "up"), ("gas", "up"), ("thrust", "up"),
    ("reverse", "down"), ("backward", "down"), ("backwards", "down"), ("brake", "down"),
    ("left", "left"), ("right", "right"), ("up", "up"), ("down", "down"),
    ("jump", ""), ("fire", ""), ("shoot", ""), ("action", ""), ("punch", ""), ("kick", ""), ("duck", ""),
    ("pause", ""), ("quit", ""), ("menu", ""), ("help", ""), ("music on/off", ""), ("sound on/off", ""),
    ("music", ""), ("sound", ""), ("mute", ""),
)

#: The fewest names in a row for them to be a legend of controls, not words of a sentence that happen to be such names.
A_LEGEND_AT_LEAST = 3


def _legend_runs(text: str) -> list[tuple[int, int, list[tuple[str, str]]]]:
    """Each run of control names in ``text``: where it starts and ends (characters), and its names with their keys."""
    lowered = str(text or "").lower()
    runs: list[tuple[int, int, list[tuple[str, str]]]] = []
    at, start, names = 0, None, []
    while at < len(lowered):
        while at < len(lowered) and lowered[at] in " ,·|/-":
            at += 1
        hit = next(((name, key) for name, key in _FUNCTIONS
                    if lowered.startswith(name, at) and (at + len(name) == len(lowered) or not lowered[at + len(name)].isalpha())), None)
        if hit is None:
            if start is not None and len(names) >= A_LEGEND_AT_LEAST:
                runs.append((start, at, names))
            start, names = None, []
            while at < len(lowered) and lowered[at] not in " ,·|/-":
                at += 1
            continue
        start = at if start is None else start
        names.append(hit)
        at += len(hit[0])
    if start is not None and len(names) >= A_LEGEND_AT_LEAST:
        runs.append((start, len(lowered), names))
    return runs


def a_legend_of_controls(text: str) -> list[tuple[str, str]]:
    """The controls a legend in ``text`` names by what they do ("Forward Reverse Turn left Turn right"), each with the
    key that by custom does it ("" where custom says none): a screen draws the keys beside the names, and the names are
    what can be read. LIVE 2026-10-10 a car game's legend was read as its goal, and no key was found to drive the car."""
    return [pair for _start, _end, names in _legend_runs(text) for pair in names]


def without_a_legend(text: str) -> str:
    """``text`` with any legend of controls in it taken out: what is left is what the place says of itself."""
    out, last = [], 0
    for start, end, _names in _legend_runs(text):
        out.append(str(text)[last:start])
        last = end
    out.append(str(text)[last:])
    return " ".join(" ".join(out).split())
