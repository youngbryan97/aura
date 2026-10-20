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
from core.verify import invariant

__all__ = ["a_legend_of_controls", "controls_named_in", "the_pointer_only_clicks", "tried_in_turn", "without_a_legend"]

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


#: Words just after a way's name that say it names a key.
_NAMES_A_KEY_AFTER = frozenset({"key", "keys", "arrow", "arrows"})


#: Words just after a way's name that say it is how a button or the pointer is pressed, not a key: "hold down the mouse
#: button", "press down on the ball". LIVE 2026-10-09 "hold down the mouse while you aim" was read as the down arrow,
#: and a game played with the mouse was taken to be played with keys.
_NOT_A_KEY_AFTER = frozenset({"mouse", "button", "on"})


def _a_key_is_meant(lowered: str, way: str) -> bool:
    import re

    words = re.findall(r"[a-z]+", lowered)
    # A press is said before the key it presses ("press up"), a key is named after its name ("up arrow"): offline
    # 2026-10-13 "press or [S] to pick it up Press or [S] again", its full stop lost, made "up" a key of a game.
    if any(word == way and (_KEY_CUES & set(words[max(0, at - 2):at]) or _NAMES_A_KEY_AFTER & set(words[at + 1:at + 3])
                            or _cued_through_a_list(words, at))
           and not _NOT_A_KEY_AFTER & set(words[at + 1:at + 4]) and not _holding_down_another(words, at)
           for at, word in enumerate(words)):
        return True
    return any(_for_an_act(re.findall(r"[a-z0-9]+", clause), way) for clause in re.split(r"[.,;:!?()]", lowered))


def _cued_through_a_list(words: list[str], at: int) -> bool:
    """Whether a way at ``at`` is one of a list of ways ("left and right", "up, down") said after a word for pressing or
    using: "press left and right to steer", "use left and right to move" name both (offline 2026-10-11, "right" was
    taken for no key). A way alone after "use" is not ("use up your lives")."""
    ways, joining = ("up", "down", "left", "right"), ("and", "or", "the")
    first = last = at
    while first > 0 and (words[first - 1] in ways or words[first - 1] in joining):
        first -= 1
    while last + 1 < len(words) and (words[last + 1] in ways or words[last + 1] in joining):
        last += 1
    listed = sum(word in ways for word in words[first:last + 1])
    return first > 0 and listed >= 2 and words[first - 1] in (*_KEY_CUES, "use", "using")


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


#: A number key named after a word for pressing: "hit 1 to switch to Anais", "press 2".
_A_NUMBER_KEY = re.compile(r"\b(?:press|hit|tap|push)\s+(?:the\s+)?[\"'“]?([0-9])[\"'”]?(?![\w'.,]?\d)(?:\s+key)?\b")


def _letter_keys(text: str, *, during_play: bool = False) -> list[str]:
    """Letter keys a game's own words name (LIVE 2026-10-09 "grenades (activated with the X key)" was not read as one),
    and number keys (offline 2026-10-13 "hit 1 to switch to anais. hit 2 to switch to gumball." named none)."""
    keys: list[str] = [match.group(1) for match in _A_NUMBER_KEY.finditer(text)]
    keys = list(dict.fromkeys(keys))
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


#: "Arrows" said where keys are meant: the arrow keys, arrows used or pressed, arrows to move. A drawn arrow is not one:
#: LIVE 2026-10-10 "place the device at the end of another device's arrow to make a connection" made up, down, left and
#: right the keys of a game played with the mouse alone ("up, down, left and right to make").
_ARROWS_AS_KEYS = re.compile(
    r"\b(?:arrow|cursor|direction(?:al)?)\s*(?:keys?|buttons?|pad)\b"
    r"|\b(?:use|press|hold|tap|hit|with|using)\s+(?:the\s+|your\s+)?(?:arrows?|directions?)\b"
    r"|\b(?:arrows?|directions?)\s+(?:to|for)\s+(?:move|moving|steer|walk|run|go|control|turn|aim|jump|drive|fly|navigate)"
    r"|(?:up|down|left|right)\s*(?:[,/&+-]|\band\b|\bor\b)?\s*(?:up|down|left|right)?\s*arrows?\b")


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
    # A key's name in quotes is its name: offline 2026-10-13 "use your 'Up' arrow key to change direction, and press the
    # 'Space' bar to fire" was read as all four arrows and no space, and the game fires with space alone.
    lowered = re.sub(r"[\"'‘’“”`](\w+)[\"'‘’“”`]", r"\1", lowered)
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
        if meant == ("up", "down", "left", "right") and (arrow_directions or not _ARROWS_AS_KEYS.search(lowered)):
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
        if meant == ("return",) and not (_a_key_is_meant(lowered, "enter") or _a_key_is_meant(lowered, "return")):
            # Offline 2026-10-13 "Enter your name." made the Enter key one of a golf game's controls.
            continue
        if any((name in words) if " " not in name else (name in lowered) for name in names):
            keys.extend(key for key in meant if key not in keys)
    keys.extend(key for key in keys_named_in(lowered) if key not in keys)
    keys.extend(key for key in _letter_keys(str(text or ""), during_play=during_play) if key not in keys)
    # Its forms too: "score points by clicking on the ghosts", "aiming" (offline 2026-10-11).
    # Substring, deliberately: none; `words` is the set of the text's own words, so this is a whole word's membership.
    pointer = any(form in words for word in _POINTER_WORDS for form in (word, word + "s", word + "ing", word + word[-1] + "ing"))
    if not keys:
        keys = list(keys_without_words)
    return keys, pointer


#: The keys that move a body by custom.
_MOVING = ("up", "down", "left", "right", "w", "a", "s", "d")


def tried_in_turn(keys: Sequence[str], said_now: str) -> list[str]:
    """``keys`` in the order they are tried for what moves her: keys for moving that the screen in front of her names
    first, then other keys for moving, then the rest (keys named for an act: space to stomp, X to fire). Offline
    2026-10-11 a bowling game's "USE WASD TO MOVE" was tried after space and four arrows it did not use, and its
    two-heart player was out before her keys were known."""
    named_now, _ = controls_named_in(said_now, keys_without_words=(), during_play=True)
    return sorted(keys, key=lambda key: 0 if key in named_now and key in _MOVING else 1 if key in _MOVING else 2)


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


#: The pointer worked in time: moved, steered or aimed with, or clicked to do something that has a moment (a throw, a
#: shot, a jump, a swing). Said either way round: "move the mouse to steer", "the mouse moves your basket", "aim with
#: your mouse", "click to throw".
_THE_POINTER_IN_TIME = re.compile(
    r"\b(?:move|moving|steer|steering|guide|guiding|control|controlling|drive|driving|fly|flying|aim|aiming|follow\w*)\b"
    r"[^.!?]{0,40}\b(?:mouse|cursor|pointer)\b"
    r"|\b(?:mouse|cursor|pointer)\b[^.!?]{0,30}\b(?:moves?|steers?|guides?|controls?|drives?|aims?|follows?|to (?:move|"
    r"steer|guide|control|drive|aim|fly))\b"
    r"|\bclick\w*\s+(?:\w+\s+){0,3}to\s+(?:throw|toss|shoot|fire|jump|hit|swing|attack|whack|splat|zap|flap|kick|punch|"
    r"strike|smash|bounce)\b", re.I)


# A route between views has no direct object whose pose follows the mouse.
# Keep separate aiming or body-control clauses in the same instructions.
_VIEW = r'(?:rooms?|pages?|screens?|panels?|views?|sections?|tabs?|locations?|places?)'
_ARTICLE = r'(?:(?:the next|the previous|the other|a different|one|another|the|a|different)\s+)?'
_VIEW_ROUTE = re.compile(
    rf'\b(?:from\s+{_ARTICLE}{_VIEW}\s+to\s+{_ARTICLE}{_VIEW}'
    rf'|between\s+{_ARTICLE}{_VIEW}|to\s+(?:the\s+)?(?:next|previous|another)\s+{_VIEW})\b', re.I)
_NAVIGATION_MOTION = re.compile(
    rf'\b(?:move|moves|moving|go|going|travel|navigate)\s+{_VIEW_ROUTE.pattern}', re.I)


def pointer_motion_text(text: str) -> str:
    """Keep movement evidence after removing explicit view-navigation routes.

    A named body remains a direct object: moving a robot from room to room
    still describes moving that robot. A route alone names no moving body.
    """
    navigation = _NAVIGATION_MOTION.sub('navigate', ' '.join(str(text or '').split()))
    # After navigation verbs are resolved, a route modifying an explicit
    # direct object can be removed without losing that object's control.
    return ' '.join(_VIEW_ROUTE.sub('', navigation).split())


@invariant('agency.view_navigation_does_not_name_a_following_body', scope='agency',
           owner='core/agency/the_controls_a_game_names.py', observational=False)
def _navigation_motion_invariant() -> tuple:
    assert the_pointer_only_clicks('Use your mouse to move from room to room.')
    assert the_pointer_only_clicks('Use the pointer to move between pages.')
    assert not the_pointer_only_clicks('Use your mouse to move the robot from room to room.')
    assert not the_pointer_only_clicks('Use the mouse to move between rooms, and click to shoot.')
    return ()


def the_pointer_only_clicks(text: str) -> bool:
    """Whether a place's words name the pointer only to click on things, drag them, or start, and name no key for moving:
    a moving scene in such a place is no world to play in time, however its pictures move. "Click on items as you find
    them. Then drag them here" is clicked through; "aim with your mouse and click to throw" and "use the mouse to move your
    basket" are played as they happen. Offline 2026-10-11 a hidden-object game whose rooms animated was handed to her
    fast play thirty-two times, three seconds at a time, and nothing was found."""
    keys, pointer = controls_named_in(text, keys_without_words=(), during_play=True)
    moving = {"up", "down", "left", "right", "w", "a", "s", "d"} & set(keys)
    return bool(pointer and not moving and not _THE_POINTER_IN_TIME.search(pointer_motion_text(text)))

