"""A program's code as its parts, each by its name, so that a change is the parts that change and not the whole again.

A program is one object of functions, with what stands before it. Its code is taken apart at the commas and statements
that separate them, and nowhere inside one: strings, templates, comments and patterns are passed over as what they are.
Put together again it is the same program. A part written again takes the place of the one of its name, and one that was
not there is added.
"""
from __future__ import annotations

import re
from collections.abc import Iterator

__all__ = ["BEFORE", "assembled", "halves", "parts_in", "patched", "each_test_in"]

#: What marks a part that stands before the object: a constant or a function of its own.
BEFORE = "before:"

_OPEN, _CLOSE = "([{", ")]}"


def _passed_over(code: str, at: int, last: str) -> int:
    """Where a string, template, comment or pattern that begins here ends; ``at`` where none begins."""
    c, two = code[at], code[at:at + 2]
    if two == "//":
        end = code.find("\n", at)
        return len(code) if end < 0 else end
    if two == "/*":
        end = code.find("*/", at + 2)
        return len(code) if end < 0 else end + 2
    if c in "\"'":
        i = at + 1
        while i < len(code) and code[i] != c and code[i] != "\n":
            i += 2 if code[i] == "\\" else 1
        return min(len(code), i + 1)
    if c == "`":
        i = at + 1
        while i < len(code) and code[i] != "`":
            if code[i] == "\\":
                i += 2
            elif code[i:i + 2] == "${":
                i = _closing(code, i + 1) + 1
            else:
                i += 1
        return min(len(code), i + 1)
    if c == "/" and (last == "" or last in "(,=:[!&|?{};+-*%<>~^" or last == "return"):
        i, inside = at + 1, False
        while i < len(code) and code[i] != "\n":
            if code[i] == "\\":
                i += 2
                continue
            if code[i] == "[":
                inside = True
            elif code[i] == "]":
                inside = False
            elif code[i] == "/" and not inside:
                i += 1
                while i < len(code) and code[i].isalpha():
                    i += 1
                return i
            i += 1
    return at


def _walk(code: str, start: int = 0) -> Iterator[tuple[int, str]]:
    """Each character of the code that is code, with where it is."""
    at, last = start, ""
    while at < len(code):
        end = _passed_over(code, at, last)
        if end > at:
            if code[at:at + 2] not in {"//", "/*"}:
                last = "x"
            at = end
            continue
        c = code[at]
        yield at, c
        if not c.isspace():
            word = re.search(r"[A-Za-z_$]+$", code[max(0, at - 8):at + 1])
            last = "return" if word and word.group(0) == "return" else c
        at += 1


def _closing(code: str, opened: int) -> int:
    """Where the bracket opened here is closed."""
    depth = 0
    for at, c in _walk(code, opened):
        if c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
            if depth == 0:
                return at
    return len(code) - 1


def _the_object(code: str) -> tuple[int, int] | None:
    """Where the object of functions begins and ends: the last one at the top of the code that has a function in it."""
    depth, opened, found = 0, -1, None
    for at, c in _walk(code):
        if c == "{" and depth == 0 or (c == "{" and depth == 1 and opened >= 0 and code[opened] == "("):
            end = _closing(code, at)
            before = code[:at].rstrip()
            literal = before.endswith(("(", "=", "return", "default", "assign(")) or not before
            if literal and re.search(r"[\w$]\s*\([^)]*\)\s*\{|=>|function", code[at:end]):
                found = (at, end)
        if c in _OPEN:
            if depth == 0:
                opened = at
            depth += 1
        elif c in _CLOSE:
            depth -= 1
            if depth == 0:
                opened = -1
    return found


def _split(code: str, start: int, end: int, by: str) -> list[str]:
    pieces, depth, last = [], 0, start
    for at, c in _walk(code, start):
        if at >= end:
            break
        if c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
        elif c == by and depth == 0:
            pieces.append(code[last:at])
            last = at + 1
    pieces.append(code[last:end])
    return [p.strip() for p in pieces if p.strip()]


def _name(part: str) -> str:
    bare = re.sub(r"^(?:\s*(?://[^\n]*\n|/\*[\s\S]*?\*/))*\s*", "", part)
    found = re.match(r"(?:async\s+|\*\s*|get\s+|set\s+)*([A-Za-z_$][\w$]*)\s*(?:\(|:|$)", bare) or re.match(r"[\"']([^\"']+)[\"']\s*:", bare)
    return found.group(1) if found else ""


def parts_in(code: str) -> dict[str, str]:
    """The parts of some code, by name, in order."""
    code = re.sub(r"^\s*export\s+default\s+", "", code.strip())
    parts: dict[str, str] = {}
    spot = _the_object(code)
    head = code if spot is None else code[: spot[0]]
    for statement in _statements(head):
        found = re.match(r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*\S|(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)", statement)
        name = (found.group(1) or found.group(2)) if found else ""
        if name and not re.fullmatch(r"(?:const|let|var)\s+[\w$]+\s*=\s*\(?", statement.strip()):
            parts[BEFORE + name] = statement.rstrip(";").strip()
    if spot is not None:
        for part in _split(code, spot[0] + 1, spot[1], ","):
            name = _name(part)
            if name:
                parts[name] = part
    else:  # functions of their own names, and no object
        for key in [k for k in parts if k[len(BEFORE):] in {"initial", "update", "score", "over", "hud", "draw", "view"}]:
            text = parts.pop(key)
            name = key[len(BEFORE):]
            parts[name] = re.sub(r"^(?:async\s+)?function\s*", "", text) if text.startswith(("function", "async")) else re.sub(
                rf"^(?:const|let|var)\s+{re.escape(name)}\s*=\s*", f"{name}: ", text)
    return parts


def _statements(code: str) -> list[str]:
    """The statements at the top of some code."""
    out, depth, last = [], 0, 0
    for at, c in _walk(code):
        if c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
            if depth == 0 and c == "}" and re.match(r"\s*(?:async\s+)?function\b", code[last:at]):
                out.append(code[last:at + 1])
                last = at + 1
        elif c == ";" and depth == 0:
            out.append(code[last:at])
            last = at + 1
    out.append(code[last:])
    cleaned = [re.sub(r"^(?:\s*(?://[^\n]*\n|/\*[\s\S]*?\*/))*\s*", "", s).strip() for s in out]
    return [s for s in cleaned if s]


def patched(parts: dict[str, str], code: str) -> tuple[dict[str, str], list[str]]:
    """The parts with those written again in ``code`` in their places; and the names of what changed."""
    new = parts_in(code)
    out, changed = dict(parts), []
    for name, text in new.items():
        if out.get(name) != text:
            changed.append(name)
        out[name] = text
    return out, changed


def each_test_in(code: str) -> tuple[str, list[tuple[str, str]]]:
    """What stands before some tests, and each test that is whole: (its name, its text). One that was cut off is left out."""
    before, found, at = "", [], 0
    for spot in re.finditer(r"(?<![\w.$])test\s*\(", code):
        if spot.start() < at:
            continue
        opened = spot.end() - 1
        closed = _closing(code, opened)
        if code[closed] != ")" or _open_at_the_end(code[opened: closed + 1]):
            break
        if not found:
            before = code[: spot.start()]
        text = code[spot.start(): closed + 1].strip()
        name = re.match(r"test\s*\(\s*([\"'`])((?:\\.|(?!\1).)*)\1", text, flags=re.S)
        found.append((name.group(2).replace("\\", "") if name else text[:60], text))
        at = closed
    return (before if found else ""), found


def _open_at_the_end(code: str) -> bool:
    depth = 0
    for _at, c in _walk(code):
        depth += (c in _OPEN) - (c in _CLOSE)
    return depth != 0


def halves(parts: dict[str, str]) -> tuple[str, str]:
    """What stands before the object, and the object."""
    before = "\n".join(text + (";" if not text.rstrip().endswith("}") or re.match(r"(?:const|let|var)\b", text) else "")
                       for name, text in parts.items() if name.startswith(BEFORE))
    body = ",\n".join("  " + text for name, text in parts.items() if not name.startswith(BEFORE))
    return before, "({\n" + body + "\n})"


def assembled(parts: dict[str, str]) -> str:
    """The parts as one program."""
    before, body = halves(parts)
    return f"{before}\n{body}" if before else body
