"""How a program plays, read from its code: what each key does, what the keys move, how fast, and what passes a level,
fails it, wins the game or ends it.

A person handed a game's source knows, before playing a frame of it, that the arrows run the hero, that z jumps, that
the level is passed once coverage reaches its quota and failed once the paint runs out. The statements a Flash
program's ActionScript is read into (core/perception/reading_a_flash_program.py) say all of that, as code says
things: a key tested and what is done when it is down; a call that ends a level under the condition that guards it.
This reads them for it, in general:

* the program is cut into its functions, each with the class it is a method of, where it is one;
* every key it tests (``Key.isDown(37)``, ``Key.getCode() == 90``, a constant standing for a code) is read with what
  is done when it is down: the method called or the value set, and what that is (moving, jumping, acting, pausing,
  sounds, or ending a level outright, which is no part of play: a key that passes a level is a cheat left in);
* what the moving keys move is the class whose methods test them, and the symbol it is made from, with the numbers it
  is given for its speed;
* every place a level is passed or failed, the game won or over, is read with the condition it happens under, put the
  right way round, and the quantities in it with what they are made of where a function returns them.

Nothing here knows a game; nothing is run. Read from AS1/AS2 statements; AS3 is not read for this yet.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["AKey", "AFunction", "AnEnd", "HowItPlays", "functions_of", "how_it_plays", "key_named", "kept", "unkept"]

#: Key codes as a player names the keys.
_KEYS = {8: "backspace", 9: "tab", 13: "enter", 16: "shift", 17: "control", 18: "alt", 27: "escape", 32: "space",
         33: "page up", 34: "page down", 35: "end", 36: "home", 37: "left", 38: "up", 39: "right", 40: "down",
         45: "insert", 46: "delete", 186: ";", 187: "=", 188: ",", 189: "-", 190: ".", 191: "/"}
_KEYS.update({c: chr(c).lower() for c in range(65, 91)})
_KEYS.update({c: str(c - 48) for c in range(48, 58)})
_KEYS.update({c: f"numpad {c - 96}" for c in range(96, 106)})
#: The keys Flash names as constants of Key.
_KEY_CONSTANTS = {"LEFT": 37, "UP": 38, "RIGHT": 39, "DOWN": 40, "SPACE": 32, "ENTER": 13, "SHIFT": 16, "CONTROL": 17,
                  "ESCAPE": 27, "TAB": 9, "BACKSPACE": 8, "HOME": 36, "END": 35, "PGUP": 33, "PGDN": 34,
                  "INSERT": 45, "DELETEKEY": 46, "CAPSLOCK": 20}

#: What a key's effect is, by the words of what it calls or sets, in order: the first that matches says.
_WHAT_IT_DOES = (
    ("ends the level", r"\bend ?level|\blevel ?complete|\bnext ?level|\bgoto ?next ?level|\blevel ?up\b|\bskip ?level|"
                       r"\bwin(?: game| level)?\b|\bcheat|\bdebug"),
    ("pauses", r"\bpaus"),
    ("sounds", r"\bsound|\bmusic|\bmute|\bvolume|\bsnd\b"),
    ("quits", r"\bquit|\bexit|\bmenu\b|\btitle\b"),
    ("jumps", r"\bjump|\bhop\b|\bleap|\bflap"),
    ("ducks", r"\bcrouch|\bduck\b|\bslide\b"),
    ("acts", r"\bfire\b|\bshoot|\bshot\b|\bthrow|\battack|\bpunch|\bkick|\bswing|\baction\b|\bbomb|\blaser|"
             r"\bspray|\bgrab|\buse\b|\bboost|\bdash\b"),
    ("moves", r"\bmove|\brun\b|\brunning|\bwalk|\baccel|\bturn\b|\bturning|\bsteer|\bthrust|\bbrake|\bspeed|"
              r"\bvel(?:ocity)?\b|\bv[xy]\b|\bdir\b|\bdirection|\bleft\b|\bright\b|\bup\b|\bdown\b|\b[xy] ?speed|"
              r"\btourne|\bkey (?:left|right|up|down)\b"),
)
#: Where a key's effect may end a level, by the literal it calls with: 'pass', 'win', 'fail', 'lose'.
_ENDING_WORDS = {"won": r"\bwin\b|\bwon\b|victory|gamecomplete|youwin|well_?done|congrat",
                 "passed": r"\bpass(?:ed)?\b|levelcomplete|level_?complete|\bnext_?level|\bcleared?\b|\bsuccess",
                 "failed": r"\bfail(?:ed|ure)?\b|\blose\b|\blost\b|retry|try_?again",
                 "over": r"game_?over|gameover|\bdead\b|\bdeath\b|\bdied\b|is_dead"}
_KEY_TEST = re.compile(r"Key\.isDown\(([^()]*)\)|Key\.getCode\(\)\s*==+\s*([\w.']+)|([\w.']+)\s*==+\s*Key\.getCode\(\)")
_REGISTER = re.compile(r"\br\d+\.")
_CALL = re.compile(r"([A-Za-z_$][\w$]*)\s*\(")
_SET = re.compile(r"^\s*(?:var\s+)?([\w$./:\[\]]+?)\s*=(?!=)\s*(.+)$")
_IF = re.compile(r"^\s*if\s+(.+?)\s+goto\s+[+-]\d+\s*$")
_NAMED = re.compile(r"^\s*([\w$.\[\]{}()', ]*?)\s*=\s*function\s*\(([^)]*)\)\s*\{\.\.\.\}\s*$")
_ANONYMOUS = re.compile(r"^(\s*)// anonymous function \(([^)]*)\)")
_DECLARED = re.compile(r"^(\s*)function ([\w$]+)\(([^)]*)\) \{\s*$")
_CLASS = re.compile(r"_global\.([\w$]+)\s*=\s*function")
_CONSTANT = re.compile(r"(?:^|[.\s])([A-Za-z_$][\w$]*)\s*=\s*(-?\d+(?:\.\d+)?)\s*$")
_SYMBOL = re.compile(r"attachMovie\('([^']+)'")


def key_named(code: int) -> str:
    """A key code as a player names the key."""
    return _KEYS.get(int(code), f"key {int(code)}")


@dataclass
class AFunction:
    """A function of the program: its name, the class it is a method of (where it is one), and its statements."""

    name: str
    owner: str
    params: str
    body: list[str]
    at: int
    ends_at: int = 0


@dataclass
class AKey:
    """A key the program tests, and what is done when it is down."""

    key: str
    does: str
    by: str
    where: str


@dataclass
class AnEnd:
    """A place the program passes or fails a level, wins the game or ends it: what happens, and when."""

    kind: str
    happens: str
    when: str
    where: str
    made_of: dict[str, str] = field(default_factory=dict)


@dataclass
class HowItPlays:
    """What a program's code says of how it is played."""

    keys: dict[str, AKey] = field(default_factory=dict)
    hers: dict[str, Any] = field(default_factory=dict)
    ends: list[AnEnd] = field(default_factory=list)
    #: The stage it draws on, in its own pixels, where the file said.
    stage: tuple[float, float] = (0.0, 0.0)
    #: Where a level is passed by covering ground: the cells it marks as covered (width, height, in stage pixels),
    #: the rectangle play is kept inside (left, top, right, bottom) where it says, and what counts the cells.
    covers: dict[str, Any] = field(default_factory=dict)
    #: Arithmetic proposals retain their unresolved conditions separately from body physics.
    source_motion: Any = None

    def keys_for(self, does: str) -> list[str]:
        return [key for key, what in self.keys.items() if what.does == does]

    def played_with(self) -> dict[str, str]:
        """The keys that are part of play, with what each does: not those that end a level outright."""
        return {key: what.does for key, what in self.keys.items() if what.does != "ends the level"}

    def when(self, kind: str) -> list[str]:
        return [end.when for end in self.ends if end.kind == kind and end.when]

    def __bool__(self) -> bool:
        return bool(self.keys or self.hers or self.ends)


def _plain(line: str) -> str:
    """A statement without its registers: ``r1._speed`` is ``_speed``, as the author wrote ``this._speed``."""
    return _REGISTER.sub("", line.strip()).replace("this.", "")


def functions_of(code: Iterable[str]) -> list[AFunction]:
    """The program cut into its functions, each with the class it is a method of. An anonymous function is written as
    a marker (``// anonymous function (x, y)``), its body at the marker's indent or deeper, and then, less indented,
    the line that names it (``X.prototype.update = function (x, y) {...}``); a class as ``_global.X = function``; a
    declared function as ``function name(...) {`` to its closing brace. Functions inside functions are functions of
    their own too. What is in no function (frame scripts, clip events) is one function each, named by its heading."""
    lines = list(code)
    out: list[AFunction] = []
    state = {"owner": ""}

    def indent_of(line: str) -> int:
        return len(line) - len(line.lstrip())

    def read(i: int, until: int, loose_name: str) -> None:
        loose: list[str] = []
        loose_at = i
        while i < until:
            line = lines[i]
            found = _ANONYMOUS.match(line)
            declared = _DECLARED.match(line)
            if found or declared:
                indent = indent_of(line)
                j = i + 1
                if found:
                    while j < until and (not lines[j].strip() or indent_of(lines[j]) >= indent):
                        j += 1
                else:
                    while j < until and not (indent_of(lines[j]) == indent and lines[j].strip() == "}"):
                        j += 1
                name, after = "", j
                if declared:
                    name, after = declared.group(2), min(j + 1, until)
                elif j < until:
                    classed = _CLASS.search(lines[j])
                    named = _NAMED.match(lines[j])
                    if classed:
                        state["owner"], name = classed.group(1), "constructor"
                    elif named:
                        target = named.group(1).replace("{...}", "")
                        name = re.split(r"[.\[\]\s]", target.strip())[-1] or target.strip()
                        after = j + 1
                params = found.group(2) if found else declared.group(3)
                body = lines[i + 1:j]
                out.append(AFunction(name or "a function", state["owner"], params, body, i, j))
                read(i + 1, j, name)                 # the functions inside it are its own, and not its body's lines
                i = after
                continue
            if line.startswith("// "):
                if loose:
                    out.append(AFunction(loose_name or "the main timeline", "", "", loose, loose_at))
                loose, loose_name, loose_at = [], line[3:].strip(), i
            classed = _CLASS.search(line)
            if classed:
                state["owner"] = classed.group(1)
            loose.append(line)
            i += 1
        if loose and until == len(lines):
            out.append(AFunction(loose_name or "the main timeline", "", "", loose, loose_at))

    read(0, len(lines), "the main timeline")
    # A function's own statements are not those of the functions inside it.
    for function in out:
        if not function.ends_at:
            continue
        inner = [(f.at, f.ends_at) for f in out if function.at < f.at < function.ends_at and f.ends_at]
        if inner:
            function.body = [line for k, line in enumerate(lines[function.at + 1:function.ends_at], function.at + 1)
                             if not any(a <= k < b for a, b in inner)]
    return out


def _constants(code: list[str]) -> dict[str, float]:
    """Names given a number anywhere in the program (``KUP = 38``, ``prototype._speedMax = 7``), and Flash's own."""
    found: dict[str, float] = {}
    for line in code:
        hit = _CONSTANT.search(_plain(line))
        if hit:
            found.setdefault(hit.group(1), float(hit.group(2)))
    return found


def _a_code(arg: str, constants: dict[str, float]) -> int | None:
    arg = arg.strip().strip("'")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", arg):
        return int(float(arg))
    last = arg.split(".")[-1]
    if arg.startswith("Key.") and last in _KEY_CONSTANTS:
        return _KEY_CONSTANTS[last]
    if last in constants:
        return int(constants[last])
    if len(last) == 1 and last.isalpha():
        return ord(last.upper())
    return None


def _codes_in(line: str, constants: dict[str, float]) -> set[int]:
    return {c for hit in _KEY_TEST.finditer(line) for c in [_a_code(next(g for g in hit.groups() if g), constants)]
            if c is not None}


def _effect(function: AFunction, at: int, code: int, constants: dict[str, float]) -> tuple[str, str]:
    """What is done when the key tested at ``at`` in a function is down: the first call made or value set after it,
    past the lines that test it again (one key read into a sum with others is read on to what the sum is used for)
    and the conditions that only lead on; not past a line that tests other keys alone, which is their branch, not
    this key's. Else what the function is named for. And what that is."""
    line = _plain(function.body[at])
    own = _SET.match(line)
    candidates: list[str] = []
    if own and "Key." in own.group(2) and not own.group(1).startswith("if "):
        candidates.append(own.group(1))                      # key_jump = Key.isDown(32)
    for after in function.body[at + 1:at + 60]:
        if candidates:
            break
        plain = _plain(after)
        codes = _codes_in(plain, constants)
        if codes and code not in codes:
            break                                            # another key's branch
        if codes or not plain or plain.startswith(("goto", "return", "//", "if ")):
            continue
        calls = [c for c in _CALL.findall(plain) if c not in ("Math", "random", "int", "Number", "String")]
        setting = _SET.match(plain)
        if setting:
            candidates.append(f"{setting.group(1)} = {setting.group(2)[:60]}")
        elif calls:
            candidates.append(plain[:90])
    # A handler's own name says when it runs, not what it does: "onKeyUp" is no moving up.
    candidates.append(re.sub(r"on_?(?:key|mouse)_?(?:up|down|press)|key_?(?:up|down|press)|onEnterFrame", "",
                             function.name, flags=re.I))
    for said in candidates:
        does = _what_it_is(said)
        if does:
            return does, said
    return "does something", candidates[0]


def _what_it_is(said: str) -> str:
    """What a call or a value set is, by its words, in the order of ``_WHAT_IT_DOES``: '' where they do not say. Its
    words are its names cut where they change case or have a mark between them: ``moveDir`` is "move dir",
    ``key_jump`` "key jump"."""
    words = re.sub(r"[_$.]+", " ", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", said)).lower()
    for does, pattern in _WHAT_IT_DOES:
        if re.search(pattern, words):
            return does
    return ""


def _keys(functions: list[AFunction], constants: dict[str, float]) -> dict[str, AKey]:
    keys: dict[str, AKey] = {}
    for function in functions:
        for at, line in enumerate(function.body):
            for hit in _KEY_TEST.finditer(line):
                code = _a_code(next(g for g in hit.groups() if g), constants)
                if code is None:
                    continue
                key = key_named(code)
                does, by = _effect(function, at, code, constants)
                where = f"{function.owner}.{function.name}" if function.owner else function.name
                kept = keys.get(key)
                # What a key does in play outweighs what it does in a menu or a cheat: the first moving or acting
                # use is kept over one that only pauses, quits or ends the level.
                if kept is None or (kept.does in ("does something", "ends the level", "quits")
                                    and does not in ("does something",)):
                    keys[key] = AKey(key, does, by, where)
    return keys


def _hers(functions: list[AFunction], keys: dict[str, AKey], code: list[str], constants: dict[str, float]) -> dict[str, Any]:
    """What the moving keys move: the thing their calls are made on where it is named (``Player.mAccelerate(...)``),
    else the class whose methods test them; the symbol it is made of; and the numbers it is given for its speeds."""
    moving = [k for k in keys.values() if k.does in ("moves", "jumps")]
    if not moving:
        return {}
    named = [m.group(1) for k in moving for m in [re.match(r"([A-Za-z_$][\w$]*)\.[\w$]+\(", k.by)] if m]
    named = [n for n in named if n not in ("this", "Key", "Math", "_root", "_parent", "_fsm", "super")]
    owners = [k.where.split(".")[0] for k in moving if "." in k.where]
    candidates = named or owners
    if not candidates:
        return {}
    owner = max(set(candidates), key=candidates.count)
    symbol = ""
    for line in code:
        if f"new {owner}(" in line:
            found = _SYMBOL.search(line)
            if found:
                symbol = found.group(1)
                break
    numbers: dict[str, float] = {}
    wanted = re.compile(r"speed|vel|accel|grav|jump|friction|max|spawn|thrust|turn", re.I)
    for function in functions:
        if function.owner != owner and not function.name.startswith(owner):
            continue
        for line in function.body:
            hit = _CONSTANT.search(_plain(line))
            if hit and wanted.search(hit.group(1)) and not hit.group(1).isupper():
                numbers.setdefault(hit.group(1), float(hit.group(2)))
    for line in code:
        plain = _plain(line)
        if ("prototype." in plain or plain.startswith(f"{owner}.")) and "function" not in plain.split("=")[-1]:
            hit = _CONSTANT.search(plain)
            if hit and wanted.search(hit.group(1)) and not hit.group(1).isupper() and \
                    (plain.startswith(f"{owner}.") or any(f.owner == owner for f in functions)):
                numbers.setdefault(hit.group(1), float(hit.group(2)))
    return {"class": owner, "symbol": symbol, "numbers": numbers}


def _negated(condition: str) -> str:
    """A condition put the other way round, simply: ``!X`` is ``X``; ``a < b`` is ``a >= b``."""
    c = condition.strip()
    while c.startswith("!!"):
        c = c[2:].strip()
    if c.startswith("!"):
        return _bare(c[1:])
    inner = _bare(c)
    for this, that in (("<=", ">"), (">=", "<"), ("==", "!="), ("!=", "=="), ("<", ">="), (">", "<=")):
        parts = _top_level_split(inner, this)
        if parts:
            return f"{parts[0]} {that} {parts[1]}"
    return f"not {inner}"


def _bare(c: str) -> str:
    c = c.strip()
    while c.startswith("!!"):
        c = c[2:].strip()
    while c.startswith("(") and c.endswith(")") and _balanced(c[1:-1]):
        c = c[1:-1].strip()
    return c


def _balanced(text: str) -> bool:
    depth = 0
    for ch in text:
        depth += ch == "("
        depth -= ch == ")"
        if depth < 0:
            return False
    return depth == 0


def _top_level_split(text: str, op: str) -> tuple[str, str] | None:
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and text.startswith(op, i) and not (op in "<>" and text[i + 1:i + 2] == "=") \
                and not (op in ("<", ">") and text[i - 1:i] in "<>=!"):
            return text[:i].strip(), text[i + len(op):].strip()
        i += 1
    return None


def _an_ending(line: str) -> tuple[str, str] | None:
    """Whether a statement ends a level or the game, and how: ``('passed', "endLevel('pass')")``."""
    plain = _plain(line)
    if plain.startswith(("if ", "goto", "//", "trace(")) or re.search(r"(?:sound|music|snd|audio)\w*\.|play_?(?:sound|music|clip)",
                                                                       plain, re.I):
        return None                                          # a sound named 'fail' is no failing
    literals = " ".join(re.findall(r"'([^']*)'", plain)).lower()
    calls = " ".join(_CALL.findall(plain)).lower()
    setting = _SET.match(plain)
    said = f"{literals} {calls}"
    if setting and setting.group(2).strip() == "true":
        said += " " + re.sub(r"([a-z])([A-Z])", r"\1 \2", setting.group(1)).lower().replace("_", " ")
    elif not (literals or "gotoandplay" in calls or "gotoandstop" in calls or "gotolabel" in calls):
        return None
    for kind, pattern in _ENDING_WORDS.items():
        if re.search(pattern, said.replace(" ", "_")) or re.search(pattern, said):
            return kind, plain[:120]
    return None


def _ends(functions: list[AFunction], returns: dict[str, str]) -> list[AnEnd]:
    out: list[AnEnd] = []
    seen: set[tuple[str, str]] = set()
    for function in functions:
        for at, line in enumerate(function.body):
            ending = _an_ending(line)
            if ending is None:
                continue
            kind, happens = ending
            when = ""
            for before in reversed(function.body[max(0, at - 3):at]):
                guard = _IF.match(_plain(before))
                if guard:
                    when = _negated(guard.group(1))
                    break
            where = f"{function.owner}.{function.name}" if function.owner else function.name
            if _KEY_TEST.search(when):
                kind = "by a key"                            # a key that ends the level outright: left in, not play
            if (happens, when) in seen:
                continue
            seen.add((happens, when))
            made_of = {name: said for name, said in returns.items() if re.search(rf"\b{re.escape(name)}\(", when)}
            out.append(AnEnd(kind, happens, when, where, made_of))
    return out


def _returns(functions: list[AFunction]) -> dict[str, str]:
    """What the program's small functions return: ``getCoverage`` returns ``(_paintCount / 2.4)``."""
    out: dict[str, str] = {}
    for function in functions:
        said = [_plain(line) for line in function.body if _plain(line)]
        returned = [line[7:] for line in said if line.startswith("return ") and line != "return undefined"]
        if len(said) <= 4 and len(returned) == 1 and function.name not in ("a function", "constructor"):
            out.setdefault(function.name, _bare(returned[0]))
    return out


#: A cell of a grid marked: ``grid[Math.floor((x / 30))][Math.floor((y / 25))] = 1``.
_A_CELL_MARKED = re.compile(r"\[Math\.floor\((.+?)\)\]\[Math\.floor\((.+?)\)\]\s*=\s*(?:1|true)\s*$")
_DIVIDED_BY = re.compile(r"/\s*(\d+(?:\.\d+)?)\)")
_COUNTED = re.compile(r"^([\w$.]+)\s*=\s*\(\1\s*\+\s*1\)$")
_A_RECT = re.compile(r"^([\w$.]+)\s*=\s*new\s+[\w$.]*Rect(?:angle)?\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*"
                     r"(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)$")


def _covers(functions: list[AFunction], ends: list[AnEnd], hers: dict[str, Any]) -> dict[str, Any]:
    """Whether a level is passed by covering ground: a function marks a cell of a grid, by a place divided by the cell's
    width and height, and counts it; and what a level is passed by is made of that count. The cell, the rectangle
    play is kept inside where her class keeps her in one, and the count."""
    passing = " ".join(end.when + " " + " ".join(end.made_of.values()) for end in ends if end.kind == "passed")
    for function in functions:
        plain = [_plain(line) for line in function.body]
        for line in plain:
            marked = _A_CELL_MARKED.search(line)
            if not marked:
                continue
            across, down = (_DIVIDED_BY.findall(part) for part in marked.groups())
            if not across or not down:
                continue
            counts = [m.group(1) for other in plain for m in [_COUNTED.match(other)] if m]
            counted = [c for c in counts if c.split(".")[-1] in passing]
            if not counted:
                continue
            found = {"cell": (float(across[-1]), float(down[-1])), "counted_by": counted[0].split(".")[-1],
                     "marked_in": f"{function.owner}.{function.name}" if function.owner else function.name}
            field_ = _kept_inside(functions, hers)
            if field_:
                found["field"] = field_
            return found
    return {}


def _kept_inside(functions: list[AFunction], hers: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """The rectangle her class keeps her inside: one made with four numbers and named, whose name her class's
    methods test her place against."""
    rects: dict[str, tuple[float, float, float, float]] = {}
    for function in functions:
        for line in function.body:
            found = _A_RECT.match(_plain(line))
            if found:
                rects[found.group(1).split(".")[-1]] = tuple(float(v) for v in found.groups()[1:])
    owner = hers.get("class")
    for function in functions:
        if not owner or function.owner != owner:
            continue
        text = " ".join(function.body)
        for name, rect in rects.items():
            if re.search(rf"\b{re.escape(name)}\b", text):
                return rect
    return None


def how_it_plays(code: Iterable[str], stage: tuple[float, float] = (0.0, 0.0)) -> HowItPlays:
    """What a program's statements say of how it is played (see the module's words); ``stage`` is the size of what it
    draws on, where its file says."""
    lines = [line for line in code if isinstance(line, str)]
    if not lines:
        return HowItPlays()
    functions = functions_of(lines)
    constants = _constants(lines)
    keys = _keys(functions, constants)
    hers = _hers(functions, keys, lines, constants)
    ends = _ends(functions, _returns(functions))
    return HowItPlays(keys=keys, hers=hers, ends=ends, stage=tuple(stage), covers=_covers(functions, ends, hers))


def kept(plays: HowItPlays | None) -> dict[str, Any]:
    """What a program was read to say of how it is played, as plain data to be kept with the read."""
    if not plays:
        return {}
    from core.perception.motion_inside_source_methods import kept as motion_kept

    return {"keys": {k: [a.does, a.by, a.where] for k, a in plays.keys.items()}, "hers": dict(plays.hers),
            "ends": [[e.kind, e.happens, e.when, e.where, dict(e.made_of)] for e in plays.ends],
            "stage": list(plays.stage), "covers": {k: list(v) if isinstance(v, tuple) else v
                                                   for k, v in plays.covers.items()},
            "source_motion": motion_kept(plays.source_motion)}


def unkept(held: Any) -> HowItPlays | None:
    """What was kept of how a program plays, as it was read; None where nothing was."""
    if not isinstance(held, dict) or not held:
        return None
    from core.perception.motion_inside_source_methods import unkept as motion_unkept

    try:
        covers = {k: tuple(v) if isinstance(v, list) else v for k, v in (held.get("covers") or {}).items()}
        return HowItPlays(keys={k: AKey(k, *v) for k, v in (held.get("keys") or {}).items()},
                          hers=dict(held.get("hers") or {}), ends=[AnEnd(*e) for e in held.get("ends") or []],
                          stage=tuple(held.get("stage") or (0.0, 0.0)), covers=covers,
                          source_motion=motion_unkept(held.get('source_motion'))) or None
    except TypeError:
        return None
