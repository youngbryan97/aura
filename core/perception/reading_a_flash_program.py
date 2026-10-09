"""Reading a Flash program: its text, its frame labels and symbol names, and its ActionScript as statements.

A Flash file holds everything its author wrote: the words drawn on its screens
(as glyphs of its own fonts), the names given to its parts, and the code that
runs it, AS1/AS2 as stack bytecode and AS3 as compiled classes. A person who
can read it knows what keys it listens for, what its buttons do, what it
counts, and what it says to its player, before playing a frame of it.

AS1/AS2 bytecode (frame scripts, init scripts, clip events and button actions)
is turned into statements by running its stack symbolically: nothing is run,
values are only written out. AS3 (DoABC) is read for its constant pool: class,
method and property names and string literals.

The file is data from the network: it is parsed, never executed, and a file
that would unpack larger than ``MOST_UNPACKED`` bytes is not read.

Nothing here knows a game.
"""
from __future__ import annotations

import lzma
import re
import struct
import zlib
from collections.abc import Iterator
from typing import Any

__all__ = ["MOST_UNPACKED", "is_flash", "read_flash"]

#: The most a Flash file may unpack to before it is not read, in bytes.
MOST_UNPACKED = 80 * 1024 * 1024

#: The longest an expression is written out: one built from copies of itself doubles each time.
LONGEST_EXPRESSION = 160


def is_flash(data: bytes) -> bool:
    return data[:3] in (b"FWS", b"CWS", b"ZWS")


class _Bits:
    def __init__(self, data: bytes, pos: int = 0):
        self.d, self.byte, self.bit = data, pos, 0

    def u(self, n: int) -> int:
        v = 0
        for _ in range(n):
            if self.byte >= len(self.d):
                return v
            v = (v << 1) | ((self.d[self.byte] >> (7 - self.bit)) & 1)
            self.bit += 1
            if self.bit == 8:
                self.bit, self.byte = 0, self.byte + 1
        return v

    def s(self, n: int) -> int:
        v = self.u(n)
        return v - (1 << n) if n and v & (1 << (n - 1)) else v

    def align(self) -> int:
        if self.bit:
            self.bit, self.byte = 0, self.byte + 1
        return self.byte


def _rect_end(data: bytes, pos: int) -> int:
    b = _Bits(data, pos)
    n = b.u(5)
    for _ in range(4):
        b.s(n)
    return b.align()


def _matrix_end(data: bytes, pos: int) -> int:
    b = _Bits(data, pos)
    for _ in range(2):
        if b.u(1):
            n = b.u(5)
            b.s(n)
            b.s(n)
    n = b.u(5)
    b.s(n)
    b.s(n)
    return b.align()


def _cxform_end(data: bytes, pos: int) -> int:
    b = _Bits(data, pos)
    add, mult, n = b.u(1), b.u(1), b.u(4)
    for _ in range((4 if mult else 0) + (4 if add else 0)):
        b.s(n)
    return b.align()


def _cstr(data: bytes, pos: int) -> tuple[str, int]:
    end = data.find(b"\0", pos)
    if end < 0:
        end = len(data)
    raw = data[pos:end]
    try:
        return raw.decode("utf-8"), end + 1
    except UnicodeDecodeError:
        return raw.decode("latin-1"), end + 1


def _body(raw: bytes) -> tuple[int, bytes]:
    """The file's version and its tags, unpacked."""
    sig, version = raw[:3], raw[3]
    if sig == b"FWS":
        data = raw[8:]
    elif sig == b"CWS":
        unpack = zlib.decompressobj()
        data = unpack.decompress(raw[8:], MOST_UNPACKED)
    elif sig == b"ZWS":
        size = struct.unpack("<I", raw[4:8])[0] - 8
        if size > MOST_UNPACKED:
            raise ValueError("too large to read")
        header = raw[12:17] + struct.pack("<Q", size)
        data = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(header + raw[17:], MOST_UNPACKED)
    else:
        raise ValueError("not a Flash file")
    return version, data[_rect_end(data, 0) + 4:]


def _tags(data: bytes, pos: int = 0, end: int | None = None) -> Iterator[tuple[int, bytes]]:
    end = len(data) if end is None else end
    while pos + 2 <= end:
        head = struct.unpack("<H", data[pos:pos + 2])[0]
        code, length = head >> 6, head & 0x3F
        pos += 2
        if length == 0x3F:
            length = struct.unpack("<I", data[pos:pos + 4])[0]
            pos += 4
        yield code, data[pos:pos + length]
        pos += length
        if code == 0:
            return


# -- AS1/AS2 ------------------------------------------------------------------

_BINARY = {0x0A: "+", 0x0B: "-", 0x0C: "*", 0x0D: "/", 0x0E: "==", 0x0F: "<", 0x10: "&&", 0x11: "||", 0x13: "eq",
           0x21: "add", 0x29: "lt", 0x47: "+", 0x48: "<", 0x49: "==", 0x3F: "%", 0x60: "&", 0x61: "|", 0x62: "^",
           0x63: "<<", 0x64: ">>", 0x65: ">>>", 0x66: "===", 0x67: ">", 0x68: "gt", 0x54: "instanceof"}

_SIMPLE = {0x04: "nextFrame()", 0x05: "prevFrame()", 0x06: "play()", 0x07: "stop()", 0x08: "toggleQuality()",
           0x09: "stopSounds()", 0x2B: "", 0x00: ""}


def _un(s: str) -> str:
    return s[1:-1] if len(s) >= 2 and s[0] == "'" and s[-1] == "'" else s


def _count(s: str) -> int:
    try:
        return max(0, min(16, int(float(s))))
    except ValueError:
        return 0


class _Stack:
    """The values an action list pushes, written out as expressions."""

    def __init__(self) -> None:
        self.values: list[str] = []

    def push(self, value: str) -> None:
        self.values.append(value)

    def pop(self) -> str:
        v = self.values.pop() if self.values else "?"
        return v if len(v) <= LONGEST_EXPRESSION else v[:LONGEST_EXPRESSION - 10] + "…" + (")" if v.startswith("(") else "")


def _pushed(payload: bytes, stack: _Stack, regs: dict[int, str], pool: list[str]) -> None:
    p = 0
    while p < len(payload):
        t = payload[p]
        p += 1
        if t == 0:
            s, p = _cstr(payload, p)
            stack.push(repr(s))
        elif t == 1:
            stack.push(f"{struct.unpack('<f', payload[p:p + 4])[0]:g}")
            p += 4
        elif t in (2, 3):
            stack.push("null" if t == 2 else "undefined")
        elif t == 4:
            stack.push(regs.get(payload[p], f"r{payload[p]}"))
            p += 1
        elif t == 5:
            stack.push("true" if payload[p] else "false")
            p += 1
        elif t == 6:
            stack.push(f"{struct.unpack('<d', payload[p + 4:p + 8] + payload[p:p + 4])[0]:g}")
            p += 8
        elif t == 7:
            stack.push(str(struct.unpack("<i", payload[p:p + 4])[0]))
            p += 4
        elif t in (8, 9):
            i = payload[p] if t == 8 else struct.unpack("<H", payload[p:p + 2])[0]
            p += 1 if t == 8 else 2
            stack.push(repr(pool[i]) if i < len(pool) else f"c{i}")
        else:
            return


def _actions(data: bytes, pos: int, end: int, out: list[str], depth: int = 0, pool: list[str] | None = None) -> list[str]:
    """An action list as statements, appended to ``out``."""
    pool = pool if pool is not None else []
    stack, regs, pad = _Stack(), {}, "  " * depth

    def emit(line: str) -> None:
        out.append(pad + line)

    while pos < end:
        code = data[pos]
        pos += 1
        if code == 0:
            break
        length = 0
        if code >= 0x80:
            if pos + 2 > end:
                break
            length = struct.unpack("<H", data[pos:pos + 2])[0]
            pos += 2
        payload = data[pos:pos + length]
        pos += length
        try:
            pos = _one_action(code, payload, data, pos, stack, regs, pool, out, emit, depth, pad)
        except (IndexError, struct.error):
            continue
    for v in stack.values:
        if "(" in v and not v.startswith("function"):
            emit(v)
    return out


def _one_action(code: int, payload: bytes, data: bytes, pos: int, stack: _Stack, regs: dict[int, str], pool: list[str],
                out: list[str], emit: Any, depth: int, pad: str) -> int:
    """One action: what it pushes, or the statement it makes; where the action list goes on from."""
    pop, push = stack.pop, stack.push
    if code == 0x88:
        n = struct.unpack("<H", payload[:2])[0]
        p, pool[:] = 2, []
        for _ in range(n):
            s, p = _cstr(payload, p)
            pool.append(s)
    elif code == 0x96:
        _pushed(payload, stack, regs, pool)
    elif code in _BINARY:
        b, a = pop(), pop()
        push(f"({_un(a)} {_BINARY[code]} {_un(b)})")
    elif code == 0x12:
        push(f"!{pop()}")
    elif code == 0x1C:
        push(_un(pop()))
    elif code == 0x1D:
        v, n = pop(), pop()
        emit(f"{_un(n)} = {v}")
    elif code == 0x4E:
        m, o = pop(), pop()
        push(f"{o}.{_un(m)}" if m.startswith("'") else f"{o}[{m}]")
    elif code == 0x4F:
        v, m, o = pop(), pop(), pop()
        emit(f"{o}.{_un(m)} = {v}" if m.startswith("'") else f"{o}[{m}] = {v}")
    elif code in (0x3D, 0x52, 0x40, 0x53):
        if code in (0x52, 0x53):
            m, o = pop(), pop()
            callee = f"{o}.{_un(m)}" if m not in ("undefined", "''") else o
        else:
            callee = _un(pop())
        args = [pop() for _ in range(_count(pop()))]
        push(f"{'new ' if code in (0x40, 0x53) else ''}{callee}({', '.join(args)})")
    elif code == 0x17:
        v = pop()
        if "(" in v:
            emit(v)
    elif code == 0x3E:
        emit(f"return {pop()}")
    elif code == 0x4C:
        v = pop()
        push(v)
        push(v)
    elif code == 0x4D:
        b, a = pop(), pop()
        push(b)
        push(a)
    elif code == 0x87:
        regs[payload[0]] = stack.values[-1] if stack.values else "?"
    elif code == 0x3C:
        v, n = pop(), pop()
        emit(f"var {_un(n)} = {v}")
    elif code == 0x41:
        emit(f"var {_un(pop())}")
    elif code in (0x50, 0x51):
        push(f"({pop()} {'+' if code == 0x50 else '-'} 1)")
    elif code in (0x9D, 0x99):
        emit(f"{'if ' + pop() + ' ' if code == 0x9D else ''}goto {struct.unpack('<h', payload[:2])[0]:+d}")
    elif code == 0x8C:
        emit(f"gotoLabel({_cstr(payload, 0)[0]!r})")
    elif code == 0x81:
        emit(f"gotoFrame({struct.unpack('<H', payload[:2])[0]})")
    elif code == 0x9F:
        emit(f"gotoAndPlay/Stop({pop()})")
    elif code == 0x8B:
        emit(f"tellTarget({_cstr(payload, 0)[0]!r})")
    elif code == 0x20:
        emit(f"tellTarget({pop()})")
    elif code == 0x83:
        a, p2 = _cstr(payload, 0)
        emit(f"getURL({a!r}, {_cstr(payload, p2)[0]!r})")
    elif code == 0x9A:
        b, a = pop(), pop()
        emit(f"getURL({a}, {b})")
    elif code in (0x9B, 0x8E):
        return _a_function(code, payload, data, pos, stack, regs, pool, out, emit, depth, pad)
    elif code == 0x94:
        size = struct.unpack("<H", payload[:2])[0]
        emit(f"with ({pop()}) {{")
        _actions(data, pos, pos + size, out, depth + 1, pool)
        emit("}")
        return pos + size
    else:
        _a_named_action(code, stack, emit)
    return pos


def _a_function(code: int, payload: bytes, data: bytes, pos: int, stack: _Stack, regs: dict[int, str], pool: list[str],
                out: list[str], emit: Any, depth: int, pad: str) -> int:
    name, p = _cstr(payload, 0)
    n = struct.unpack("<H", payload[p:p + 2])[0]
    p += 2
    params = []
    if code == 0x8E:
        p += 3
        for _ in range(n):
            reg = payload[p]
            p += 1
            s, p = _cstr(payload, p)
            params.append(s)
            if reg:
                regs[reg] = s
    else:
        for _ in range(n):
            s, p = _cstr(payload, p)
            params.append(s)
    size = struct.unpack("<H", payload[p:p + 2])[0]
    inner: list[str] = []
    _actions(data, pos, pos + size, inner, depth + 1, list(pool))
    head = f"function {name}({', '.join(params)})"
    if name:
        emit(head + " {")
        out.extend(inner)
        emit("}")
    else:
        stack.push(head + " {...}")
        out.append(pad + "  // anonymous " + head)
        out.extend(inner)
    return pos + size


def _a_named_action(code: int, stack: _Stack, emit: Any) -> None:
    pop, push = stack.pop, stack.push
    if code == 0x26:
        emit(f"trace({pop()})")
    elif code == 0x24:
        d, t, s = pop(), pop(), pop()
        emit(f"duplicateMovieClip({s}, {t}, {d})")
    elif code == 0x25:
        emit(f"removeMovieClip({pop()})")
    elif code == 0x27:
        t = pop()
        emit(f"startDrag({t}, lock={pop()})")
    elif code == 0x28:
        emit("stopDrag()")
    elif code == 0x22:
        i, t = pop(), pop()
        push(f"getProperty({t}, {i})")
    elif code == 0x23:
        v, i, t = pop(), pop(), pop()
        emit(f"setProperty({t}, {i}, {v})")
    elif code in (0x30, 0x45, 0x18, 0x44, 0x46):
        name = {0x30: "random", 0x45: "targetPath", 0x18: "int", 0x44: "typeof", 0x46: "enumerate"}[code]
        push(f"{name}({pop()})")
    elif code == 0x34:
        push("getTimer()")
    elif code == 0x3A:
        m, o = pop(), pop()
        emit(f"delete {o}.{_un(m)}")
    elif code == 0x42:
        push("[" + ", ".join(pop() for _ in range(_count(pop()))) + "]")
    elif code == 0x43:
        pairs = []
        for _ in range(_count(pop())):
            v, k = pop(), pop()
            pairs.append(f"{_un(k)}: {v}")
        push("{" + ", ".join(pairs) + "}")
    elif code in _SIMPLE and _SIMPLE[code]:
        emit(_SIMPLE[code])
    elif code in (0x8A, 0x8D):
        emit("waitForFrame")


#: The clip events, by their bits in the event flags (little-endian): byte, bit, name.
_CLIP_EVENTS = ((0, 7, "keyUp"), (0, 6, "keyDown"), (0, 5, "mouseUp"), (0, 4, "mouseDown"), (0, 3, "mouseMove"),
                (0, 2, "unload"), (0, 1, "enterFrame"), (0, 0, "load"), (1, 7, "dragOver"), (1, 6, "rollOut"),
                (1, 5, "rollOver"), (1, 4, "releaseOutside"), (1, 3, "release"), (1, 2, "press"), (1, 1, "initialize"),
                (1, 0, "data"), (2, 2, "construct"), (2, 1, "keyPress"), (2, 0, "dragOut"))


def _clip_actions(data: bytes, pos: int, version: int, out: list[str], label: str) -> None:
    wide = version >= 6
    pos += 2 + (4 if wide else 2)
    while pos < len(data):
        flags = struct.unpack("<I", data[pos:pos + 4])[0] if wide else struct.unpack("<H", data[pos:pos + 2])[0]
        pos += 4 if wide else 2
        if flags == 0:
            return
        size = struct.unpack("<I", data[pos:pos + 4])[0]
        pos += 4
        start, key = pos, ""
        if wide and flags & 0x00020000:
            key = f" key={data[pos]}"
            pos += 1
        raw = flags.to_bytes(4 if wide else 2, "little")
        names = ",".join(n for b, bit, n in _CLIP_EVENTS if b < len(raw) and raw[b] >> bit & 1)
        out.append(f"onClipEvent({names}{key}) on {label} {{")
        _actions(data, pos, start + size, out, 1)
        out.append("}")
        pos = start + size


def _placed(data: bytes, version: int, out: list[str], *, third: bool = False) -> str:
    """A PlaceObject2 or 3: the name it gives what it places, and its clip events."""
    flags = data[0]
    pos = 3
    if third:
        flags2, pos = data[1], 4
        if flags2 & 0x08 or (flags2 & 0x10 and flags & 0x02):
            _class, pos = _cstr(data, pos)
    if flags & 0x02:
        pos += 2
    if flags & 0x04:
        pos = _matrix_end(data, pos)
    if flags & 0x08:
        pos = _cxform_end(data, pos)
    if flags & 0x10:
        pos += 2
    name = ""
    if flags & 0x20:
        name, pos = _cstr(data, pos)
    if flags & 0x40:
        pos += 2
    if flags & 0x80:
        _clip_actions(data, pos, version, out, name or "a clip")
    return name


_BUTTON_EVENTS = ("idleToOverDown", "outDownToIdle", "outDownToOverDown", "overDownToOutDown", "release", "press",
                  "rollOut", "rollOver", "dragOut")


def _button_actions(data: bytes, out: list[str]) -> None:
    pos = 3
    offset = struct.unpack("<H", data[pos:pos + 2])[0]
    if not offset:
        return
    pos = 3 + offset
    while pos < len(data):
        size = struct.unpack("<H", data[pos:pos + 2])[0]
        cond = struct.unpack("<H", data[pos + 2:pos + 4])[0]
        key = (cond >> 9) & 0x7F
        events = [n for i, n in enumerate(_BUTTON_EVENTS) if cond >> i & 1]
        out.append(f"on({','.join(events)}{' key=' + str(key) if key else ''}) {{")
        _actions(data, pos + 4, pos + size if size else len(data), out, 1)
        out.append("}")
        if not size:
            return
        pos += size


# -- text ----------------------------------------------------------------------

def _font_codes(code: int, data: bytes) -> tuple[int, list[int]] | None:
    fid = struct.unpack("<H", data[:2])[0]
    flags = data[2]
    wide_offsets, wide_codes = flags & 0x08, flags & 0x04
    pos = 5 + data[4]
    n = struct.unpack("<H", data[pos:pos + 2])[0]
    table = pos + 2
    if n == 0:
        return fid, []
    if wide_offsets:
        code_off = struct.unpack("<I", data[table + 4 * n:table + 4 * n + 4])[0]
    else:
        code_off = struct.unpack("<H", data[table + 2 * n:table + 2 * n + 2])[0]
    p = table + code_off
    return fid, [struct.unpack("<H", data[p + 2 * i:p + 2 * i + 2])[0] if wide_codes else data[p + i] for i in range(n)]


def _font_info(code: int, data: bytes) -> tuple[int, list[int]]:
    fid = struct.unpack("<H", data[:2])[0]
    pos = 3 + data[2]
    flags = data[pos]
    pos += 2 if code == 62 else 1
    rest = data[pos:]
    return fid, [struct.unpack("<H", rest[i:i + 2])[0] for i in range(0, len(rest) - 1, 2)] if flags & 0x01 else list(rest)


def _static_text(code: int, data: bytes, fonts: dict[int, list[int]]) -> str:
    pos = _matrix_end(data, _rect_end(data, 2))
    gbits, abits = data[pos], data[pos + 1]
    pos += 2
    words: list[str] = []
    font: list[int] = []
    while pos < len(data):
        flags = data[pos]
        pos += 1
        if flags == 0 or not flags & 0x80:
            break
        if flags & 0x08:
            fid = struct.unpack("<H", data[pos:pos + 2])[0]
            pos += 2
            font = fonts.get(fid, [])
        pos += (4 if code == 33 else 3) if flags & 0x04 else 0
        pos += (2 if flags & 0x01 else 0) + (2 if flags & 0x02 else 0) + (2 if flags & 0x08 else 0)
        count = data[pos]
        pos += 1
        b = _Bits(data, pos)
        chars = []
        for _ in range(count):
            g = b.u(gbits)
            b.s(abits)
            if g < len(font):
                chars.append(chr(font[g]))
        pos = b.align()
        words.append("".join(chars))
    return " ".join(w for w in words if w)


def _edit_text(data: bytes) -> tuple[str, str]:
    pos = _rect_end(data, 2)
    f1, f2 = data[pos], data[pos + 1]
    pos += 2
    pos += 4 if f1 & 0x01 else 0
    if f2 & 0x80:
        _font_class, pos = _cstr(data, pos)
    pos += (4 if f1 & 0x04 else 0) + (2 if f1 & 0x02 else 0) + (9 if f2 & 0x20 else 0)
    var, pos = _cstr(data, pos)
    text = _cstr(data, pos)[0] if f1 & 0x80 else ""
    return var, " ".join(re.sub(r"<[^>]+>", " ", text).split())


# -- AS3 -------------------------------------------------------------------------

def _abc_strings(abc: bytes) -> tuple[list[str], list[str]]:
    """An AS3 constant pool's strings, and the names its multinames use."""
    pos = 4

    def u30() -> int:
        nonlocal pos
        v, shift = 0, 0
        while True:
            b = abc[pos]
            pos += 1
            v |= (b & 0x7F) << shift
            if not b & 0x80 or shift >= 28:
                return v
            shift += 7

    for _ in range(2):  # ints, uints
        for _ in range(max(0, u30() - 1)):
            u30()
    doubles = u30()
    pos += 8 * max(0, doubles - 1)
    strings = [""]
    for _ in range(max(0, u30() - 1)):
        size = u30()
        strings.append(abc[pos:pos + size].decode("utf-8", "replace"))
        pos += size
    for _ in range(max(0, u30() - 1)):  # namespaces
        pos += 1
        u30()
    for _ in range(max(0, u30() - 1)):  # namespace sets
        for _ in range(u30()):
            u30()
    names = []
    for _ in range(max(0, u30() - 1)):
        kind = abc[pos]
        pos += 1
        if kind in (0x07, 0x0D):
            u30()
            i = u30()
            names.append(strings[i] if i < len(strings) else "")
        elif kind in (0x0F, 0x10):
            i = u30()
            names.append(strings[i] if i < len(strings) else "")
        elif kind in (0x09, 0x0E):
            i = u30()
            u30()
            names.append(strings[i] if i < len(strings) else "")
        elif kind in (0x1B, 0x1C):
            u30()
        elif kind == 0x1D:
            u30()
            for _ in range(u30()):
                u30()
        elif kind not in (0x11, 0x12):
            break
    return strings, names


# -- the whole program -------------------------------------------------------------

def read_flash(raw: bytes) -> dict[str, Any]:
    """Everything a Flash file holds that can be read: ``code`` (statements), ``texts`` (its screens' words), ``labels``
    (frame labels), ``symbols`` (exported names), ``fields`` (text field names), ``as3_strings`` and ``as3_names``."""
    version, data = _body(raw)
    got: dict[str, Any] = {"version": version, "code": [], "texts": [], "labels": [], "symbols": [], "fields": [],
                           "instances": [], "as3_strings": [], "as3_names": []}
    fonts: dict[int, list[int]] = {}
    pending: list[tuple[int, bytes]] = []
    _walk(_tags(data), "the main timeline", version, got, fonts, pending)
    for code, tag in pending:
        try:
            text = _static_text(code, tag, fonts)
        except (IndexError, struct.error):
            continue
        if text.strip():
            got["texts"].append(text)
    return got


def _walk(stream: Iterator[tuple[int, bytes]], where: str, version: int, got: dict[str, Any], fonts: dict[int, list[int]],
          pending: list[tuple[int, bytes]]) -> None:
    out = got["code"]
    for code, tag in stream:
        try:
            if code == 12:
                out.append(f"// frame script in {where}")
                _actions(tag, 0, len(tag), out)
            elif code == 59:
                out.append(f"// init script for symbol {struct.unpack('<H', tag[:2])[0]}")
                _actions(tag, 2, len(tag), out)
            elif code == 39:
                _walk(_tags(tag, 4), f"sprite {struct.unpack('<H', tag[:2])[0]}", version, got, fonts, pending)
            elif code in (26, 70):
                name = _placed(tag, version, out, third=code == 70)
                if name:
                    got["instances"].append(name)
            elif code == 34:
                _button_actions(tag, out)
            elif code == 43:
                got["labels"].append(_cstr(tag, 0)[0])
            elif code in (48, 75):
                found = _font_codes(code, tag)
                if found:
                    fonts[found[0]] = found[1]
            elif code in (13, 62):
                fid, codes = _font_info(code, tag)
                fonts[fid] = codes
            elif code in (11, 33):
                pending.append((code, tag))
            elif code == 37:
                var, text = _edit_text(tag)
                if var:
                    got["fields"].append(var)
                if text:
                    got["texts"].append(text)
            elif code in (76, 56):
                p = 2
                for _ in range(struct.unpack("<H", tag[:2])[0]):
                    s, p = _cstr(tag, p + 2)
                    got["symbols"].append(s)
            elif code in (82, 72):
                abc = tag[_cstr(tag, 4)[1]:] if code == 82 else tag
                strings, names = _abc_strings(abc)
                got["as3_strings"].extend(strings)
                got["as3_names"].extend(names)
        except (IndexError, struct.error, ValueError):
            continue
