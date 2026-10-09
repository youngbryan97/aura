"""Reading the program behind a thing: what it says, what keys it listens for, what it does with the pointer.

Whatever runs a thing on a screen was written by someone, and what they wrote
says how it is worked: the keys its handlers test ("Key.isDown(37)", "keyCode
=== 32", "e.key === 'ArrowLeft'"), whether it follows the pointer, waits for a
click, lets a thing be dragged; the names of its buttons ("pauseBtn",
"play_again"); the words it shows its user. A person who can read code reads
it once and knows how the thing is worked before touching it. So does she,
wherever the program can be had: a Flash file (core/perception/reading_a_flash_program.py),
a page's own scripts, a script file it loads.

What is read is data from the network: it is parsed, never run.

What it comes to is plain: ``words`` it says, ``keys`` it listens for and what
each is near in its code, ``pointer`` (how it uses the mouse), ``buttons`` (its
controls by name), and ``text`` (all of it, lowered, for the signs of each
mechanic, core/agency/mechanics_she_knows.py). Nothing here knows a game.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["ProgramRead", "key_named", "merged", "read_program", "says_how_it_is_worked"]

#: The most of a program's text read, in characters.
MOST_TEXT = 4_000_000

#: Key codes as programs test them, and the names the hands press them by.
_CODES = {8: "backspace", 9: "tab", 13: "return", 16: "shift", 17: "control", 18: "alt", 27: "escape", 32: "space",
          33: "pageup", 34: "pagedown", 35: "end", 36: "home", 37: "left", 38: "up", 39: "right", 40: "down", 46: "delete"}
_CODES.update({c: chr(c).lower() for c in range(65, 91)})
_CODES.update({c: chr(c) for c in range(48, 58)})

#: A Flash button's own key codes, which are not the keyboard's below 32.
_BUTTON_CODES = {1: "left", 2: "right", 3: "home", 4: "end", 5: "insert", 6: "delete", 8: "backspace", 13: "return",
                 14: "up", 15: "down", 16: "pageup", 17: "pagedown", 18: "tab", 19: "escape"}

#: Names programs give keys, as the hands name them.
_NAMES = {"left": "left", "right": "right", "up": "up", "down": "down", "space": "space", "spacebar": "space",
          "enter": "return", "return": "return", "shift": "shift", "control": "control", "ctrl": "control",
          "escape": "escape", "esc": "escape", "arrowleft": "left", "arrowright": "right", "arrowup": "up",
          "arrowdown": "down", " ": "space", "tab": "tab"}

#: Where a program tests a key, by the code it tests.
_A_CODE_TESTED = re.compile(
    r"(?:key\.isdown\(\s*|key\.getcode\(\)\s*[=!]==?\s*|keycode\s*[=!]==?\s*|\.which\s*[=!]==?\s*|keycode\s*==\s*|"
    r"case\s+)(\d{1,3})\b")
#: Where a program tests a key by its name: Key.LEFT, Keyboard.SPACE, e.key === 'ArrowUp', e.code === "KeyW".
_A_NAME_TESTED = re.compile(r"(?:\bkey|\bkeyboard)\.(left|right|up|down|space|enter|shift|control|escape|tab)\b"
                            r"|\.key\s*[=!]==?\s*['\"]([^'\"]{1,10})['\"]|\.code\s*[=!]==?\s*['\"]([a-z]+[a-z0-9]*)['\"]")
#: A Flash button's key event, as written out: on(keyPress key=37).
_A_BUTTON_KEY = re.compile(r"\bon\([a-z,]*\s*key=(\d{1,3})\)")

#: What the code near a key says it is for: the first of these words in the statements just after its test.
_PURPOSES = ("jump", "fire", "shoot", "throw", "attack", "punch", "kick", "stomp", "pause", "mute", "sound", "quit",
             "restart", "thrust", "boost", "brake", "accelerat", "rotate", "duck", "crouch", "block", "dash", "switch",
             "swap", "menu", "help")
#: How far after a key's test its purpose is looked for, in characters.
PURPOSE_WITHIN = 120

#: Keys a compiled program names as constants of its keyboard class (Keyboard.LEFT): read where it handles keys at all.
_KEY_CONSTANTS = {"LEFT": "left", "RIGHT": "right", "UP": "up", "DOWN": "down", "SPACE": "space", "ENTER": "return",
                  "SHIFT": "shift", "CONTROL": "control", "ESCAPE": "escape"}
#: A key kept in a variable named for what it does: "_stompKey", "jumpKey", "fire_key".
_A_KEY_FOR = re.compile(r"^_?([a-z]+)_?key$", re.I)

#: What a control may be named for: the controls of a thing, as against its parts.
_CONTROLS = ("play again", "try again", "level select", "more games", "how to play", "instructions", "play", "start",
             "again", "restart", "replay", "retry", "pause", "resume", "unpause", "mute", "unmute", "sound", "music",
             "menu", "next", "back", "prev", "previous", "quit", "exit", "help", "continue", "skip", "ok", "yes", "no",
             "submit", "done", "credits", "options", "settings", "fullscreen", "close", "home", "reset", "send", "go")

#: The ways a program uses the pointer, and the signs of each in its code.
_POINTER = {
    "steers": re.compile(r"(?:_x|_y|\.x|\.y|left|top)\s*=\s*[^\n;]{0,60}(?:_xmouse|_ymouse|mousex|mousey|clientx|clienty|pagex|pagey)"),
    "aims": re.compile(r"(?:atan2|_rotation|rotation)[^\n;]{0,80}(?:_xmouse|_ymouse|mousex|mousey|clientx|clienty)"),
    "clicks": re.compile(r"on\([a-z,]*(?:press|release)|onpress|onrelease|mouseevent\.click|['\"]click['\"]|\.onclick"),
    "holds": re.compile(r"onmousedown|onmouseup|mouse_down|mouse_up|['\"]mouse(?:down|up)['\"]|['\"]pointer(?:down|up)['\"]"),
    "drags": re.compile(r"startdrag|stopdrag|['\"]drag(?:start|end)?['\"]"),
    "hides it": re.compile(r"mouse\.hide|cursor\s*[:=]\s*['\"]none"),
}

#: A control by its name in a program: "pauseBtn", "btn_play", "muteButton", "play_again_mc".
_BUTTON_MARK = re.compile(r"btn|button", re.I)


@dataclass
class ProgramRead:
    """What a program says of how it is worked."""

    words: list[str] = field(default_factory=list)
    #: Each key it tests, and the word for what it is for read near it ("" where none was).
    keys: dict[str, str] = field(default_factory=dict)
    #: What it keeps keys for, by its variables' names ("stomp", "jump").
    keyed: list[str] = field(default_factory=list)
    pointer: list[str] = field(default_factory=list)
    buttons: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    #: Everything read, lowered: its code, its words and its names. Read for the signs of each mechanic only.
    text: str = ""
    kind: str = ""
    #: The mechanics its text showed, where it was read before and only that was kept.
    mechanics_found: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.text or self.keys or self.words or self.pointer or self.mechanics_found)


def key_named(name: str) -> str:
    """A key as a program names it ("ArrowLeft", "KeyW", "Space", " "), as the hands press it; "" where not a key."""
    lowered = str(name or "").lower()
    if lowered in _NAMES:
        return _NAMES[lowered]
    if re.fullmatch(r"key[a-z]", lowered):
        return lowered[-1]
    if re.fullmatch(r"digit\d", lowered):
        return lowered[-1]
    if re.fullmatch(r"[a-z0-9]", lowered):
        return lowered
    return ""


def _purpose(text: str, at: int) -> str:
    """What the statements just after a key's test are for, by the first word for a purpose in them; "cheat" where
    they are a cheat's or a debugger's."""
    after = text[at:at + PURPOSE_WITHIN]
    # Only the statements of this key's own test: the next test is another key's.
    ends = re.search(r"\bif\s*[(!]|\bcase\s|\belse\b", after[3:])
    after = after[:ends.start() + 3] if ends else after
    if any(word in after for word in _SPOILING_KEYS):
        return "cheat"
    found = [(after.find(word), word) for word in _PURPOSES if after.find(word) >= 0]
    return min(found)[1] if found else ""


def _keys_in(text: str, names: list[str] = ()) -> dict[str, str]:
    keys: dict[str, str] = {}
    if "keyboardevent" in text or "keycode" in text:
        for name in names:
            if name in _KEY_CONSTANTS:
                keys.setdefault(_KEY_CONSTANTS[name], "")
    for match in _A_CODE_TESTED.finditer(text):
        code = int(match.group(1))
        name = _CODES.get(code, "")
        # "case 37:" counts only in a program that reads key codes at all.
        if name and (not match.group(0).startswith("case") or "keycode" in text or "getcode" in text):
            keys.setdefault(name, _purpose(text, match.end()))
    for match in _A_NAME_TESTED.finditer(text):
        name = key_named(match.group(1) or match.group(2) or match.group(3) or "")
        if name:
            keys.setdefault(name, _purpose(text, match.end()))
    for match in _A_BUTTON_KEY.finditer(text):
        code = int(match.group(1))
        name = _BUTTON_CODES.get(code) or _CODES.get(code, "")
        if name:
            keys.setdefault(name, _purpose(text, match.end()))
    return keys


def _buttons_in(names: list[str]) -> list[str]:
    """The controls a program names as buttons, by what each is for: "pause", "play again", "mute"."""
    found: list[str] = []
    for name in names:
        if not _BUTTON_MARK.search(name or ""):
            continue
        said = " ".join(re.sub(r"(?<=[a-z])(?=[A-Z])|[_.\d]", " ", name).lower().split())
        control = next((c for c in _CONTROLS if re.search(rf"\b{c}\b", said)), "")
        if control and control not in found:
            found.append(control)
    return found


def _acts_keyed(names: list[str]) -> list[str]:
    """What a program keeps keys for, by the names of the variables it keeps them in: "stomp", "jump", "fire"."""
    acts: list[str] = []
    for name in names:
        match = _A_KEY_FOR.match(name or "")
        act = match.group(1).lower() if match else ""
        if act and act not in acts and act not in ("get", "set", "on", "the", "my", "a", "campaign", "clear", "destroy"):
            acts.append(act)
    return acts


#: Writing that says how a thing is worked or what it wants: its instructions. Only this is taken from a program; the
#: rest of what it holds (its story, its answers, what is ahead) is for play to show, in its time.
_HOW_IT_IS_WORKED = re.compile(
    r"\b(?:press|click|hold|use|tap|drag|move|steer|jump|shoot|fire|aim|avoid|collect|catch|dodge|grab|keys?|mouse|"
    r"arrows?|space ?bar|instructions|how to play|controls?|goal|object of|try to|don'?t let|watch out|lives|health|"
    r"score|points|timer?|level|power[- ]?ups?|bonus|to (?:move|jump|shoot|start|play|pause|switch|win))\b", re.I)
#: What would spoil it or play it for her: cheats, debugging, answers, solutions, secrets.
_SPOILS = re.compile(r"\b(?:cheats?|debug\w*|password|secret|answers? (?:is|are)|solution|walkthrough|skip (?:to|level)|"
                     r"god ?mode|unlock (?:all|every)|correct answer|hack)\b", re.I)
#: What a key may be kept for that is not hers to use.
_SPOILING_KEYS = ("cheat", "debug", "skip", "hack")


def says_how_it_is_worked(sentence: str) -> bool:
    """Whether writing says how a thing is worked or what it wants, and spoils nothing."""
    return bool(_HOW_IT_IS_WORKED.search(sentence or "")) and not _SPOILS.search(sentence or "")


def _prose(texts: list[str]) -> list[str]:
    """The writing among a program's strings meant for its user that says how it is worked: words, not names or markup,
    and not what would spoil it."""
    out: list[str] = []
    for text in texts:
        said = " ".join(re.sub(r"<[^>]+>", " ", str(text or "")).split())
        if len(re.findall(r"[A-Za-z]{2,}", said)) >= 3 and not re.search(r"[{}();=]|\w\.\w+\(|https?:", said):
            if said not in out and _HOW_IT_IS_WORKED.search(said) and not _SPOILS.search(said):
                out.append(said)
    return out


def read_program(data: bytes, name: str = "") -> ProgramRead:
    """Read a program: a Flash file, or a script or page as text. An empty read where it cannot be read."""
    from core.perception.reading_a_flash_program import is_flash, read_flash

    if is_flash(data):
        try:
            got = read_flash(data)
        except (ValueError, EOFError, OSError, MemoryError):
            return ProgramRead()
        code = "\n".join(got["code"])
        names = list(dict.fromkeys([*got["symbols"], *got["instances"], *got["fields"], *got["as3_names"]]))
        texts = [*got["texts"], *(s for s in got["as3_strings"] if " " in s)]
        kind = "flash (AS3)" if got["as3_names"] else "flash"
        labels = list(dict.fromkeys(got["labels"]))
    else:
        code = data[:MOST_TEXT].decode("utf-8", "replace")
        names = list(dict.fromkeys(re.findall(r"\b([A-Za-z_$][\w$]*(?:Btn|Button|_btn|_button))\b", code)))
        texts = re.findall(r"(?:\"([^\"\n]{12,240})\"|'([^'\n]{12,240})'|>([^<>\n]{12,240})<)", code)
        texts = [next(t for t in group if t) for group in texts]
        kind, labels = "script" if not re.search(r"<html|<body", code[:4000], re.I) else "page", []
    text = "\n".join([code, *texts, " ".join(names), " ".join(labels)])[:MOST_TEXT].lower()
    pointer = [way for way, sign in _POINTER.items() if sign.search(text)]
    keys = {key: purpose for key, purpose in _keys_in(text, names).items() if purpose != "cheat"}
    return ProgramRead(words=_prose(texts)[:80], keys=keys, keyed=_acts_keyed(names), pointer=pointer,
                       buttons=_buttons_in(names),
                       names=[n for n in names if 3 <= len(n) <= 32][:600], labels=labels[:200], text=text, kind=kind)


def merged(reads: list[ProgramRead]) -> ProgramRead:
    """Several programs one thing runs (a page, its scripts, the file it plays) read as one."""
    out = ProgramRead(kind=", ".join(dict.fromkeys(r.kind for r in reads if r.kind)))
    for read in reads:
        out.words += [w for w in read.words if w not in out.words]
        for key, purpose in read.keys.items():
            out.keys[key] = out.keys.get(key) or purpose
        out.keyed += [k for k in read.keyed if k not in out.keyed]
        out.pointer += [p for p in read.pointer if p not in out.pointer]
        out.buttons += [b for b in read.buttons if b not in out.buttons]
        out.names += read.names
        out.labels += read.labels
        out.text += "\n" + read.text
        out.mechanics_found += [m for m in read.mechanics_found if m not in out.mechanics_found]
    return out

