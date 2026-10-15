"""Locate AVM1 action lists without dropping their SWF byte addresses.

Addresses refer to the uncompressed tag stream returned by the existing Flash
reader. Each source includes the original file digest. Unsupported action
containers remain explicit issues instead of producing a guessed action list.
"""
from __future__ import annotations

import hashlib
import lzma
import struct
import zlib
from dataclasses import dataclass

from core.perception import reading_a_flash_program as flash
from core.perception.flash_actions_with_source import (
    MAX_DEPTH,
    ActionIssue,
    ActionStream,
    parse_action_stream,
)

MAX_STREAMS = 4096
MAX_TAGS = 131_072
MAX_SOURCE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class FlashActionSources:
    source: str
    streams: tuple[ActionStream, ...]
    issues: tuple[ActionIssue, ...]

    @property
    def complete(self) -> bool:
        return not self.issues and all(stream.complete for stream in self.streams)


class _BadSourceError(ValueError):
    pass


def read_flash_action_streams(raw: bytes) -> FlashActionSources:
    """Read frame, init, button and clip action lists as separate sources.

    A stream's completeness qualifies that stream. The enclosing result also
    records unsupported containers, malformed tags and exhausted budgets.
    """
    if not isinstance(raw, bytes):
        raise ValueError("immutable Flash bytes are required")
    source = "swf:" + hashlib.sha256(raw).hexdigest()
    if len(raw) < 8 or len(raw) > flash.MOST_UNPACKED or not flash.is_flash(raw):
        return FlashActionSources(source, (), (ActionIssue(0, "invalid or oversized Flash input"),))
    try:
        version, data = _strict_body(raw)
    except (IndexError, ValueError, struct.error, EOFError, zlib.error, lzma.LZMAError) as exc:
        return FlashActionSources(source, (), (ActionIssue(0, f"invalid Flash body: {exc}"),))
    streams: list[ActionStream] = []
    issues: list[ActionIssue] = []
    _walk(data, 0, len(data), version, source, streams, issues, [MAX_TAGS], 0)
    return FlashActionSources(source, tuple(streams), tuple(issues))


class _StrictBits:
    def __init__(self, data: bytes, pos: int = 0):
        self.data, self.bit = data, pos * 8

    def u(self, count: int) -> int:
        if not 0 <= count <= 32 or self.bit + count > len(self.data) * 8:
            raise _BadSourceError("truncated bit field")
        value = 0
        for _ in range(count):
            value = value * 2 + ((self.data[self.bit // 8] >> (7 - self.bit % 8)) & 1)
            self.bit += 1
        return value

    def s(self, count: int) -> int:
        value = self.u(count)
        return value - (1 << count) if count and value & (1 << (count - 1)) else value

    def end(self) -> int:
        return (self.bit + 7) // 8


def _strict_body(raw: bytes) -> tuple[int, bytes]:
    version, declared = raw[3], int.from_bytes(raw[4:8], "little")
    if not version or not 8 <= declared <= flash.MOST_UNPACKED:
        raise _BadSourceError("invalid version or declared FileLength")
    expected = declared - 8
    if raw[:3] == b"FWS":
        body = raw[8:]
    elif raw[:3] == b"CWS":
        if version < 6:
            raise _BadSourceError("zlib compression requires SWF version 6")
        unpack = zlib.decompressobj()
        body = unpack.decompress(raw[8:], expected + 1)
        if not unpack.eof or unpack.unused_data or unpack.unconsumed_tail:
            raise _BadSourceError("incomplete or trailing zlib stream")
    else:
        if version < 13 or len(raw) < 17:
            raise _BadSourceError("invalid LZMA header or version")
        compressed_length = int.from_bytes(raw[8:12], "little")
        if compressed_length != len(raw) - 17:
            raise _BadSourceError("LZMA compressed length differs from input")
        if int.from_bytes(raw[13:17], "little") > flash.MOST_UNPACKED:
            raise _BadSourceError("LZMA dictionary exceeds memory budget")
        unpack_lzma = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE,
                                           memlimit=flash.MOST_UNPACKED + 8 * 1024 * 1024)
        body = unpack_lzma.decompress(raw[12:17] + b"\xff" * 8 + raw[17:], expected + 1)
        if not unpack_lzma.eof or unpack_lzma.unused_data:
            raise _BadSourceError("incomplete or trailing LZMA stream")
    if len(body) != expected:
        raise _BadSourceError("FileLength differs from uncompressed input")
    bits = _StrictBits(body)
    count = bits.u(5)
    xmin, xmax, ymin, ymax = (bits.s(count) for _ in range(4))
    if xmin >= xmax or ymin >= ymax:
        raise _BadSourceError("stage rectangle has no positive extent")
    pos = bits.end()
    if not _u16(body, pos, len(body)) or not _u16(body, pos + 2, len(body)):
        raise _BadSourceError("frame rate and frame count must be positive")
    return version, body[pos + 4:]


def _matrix_end(data: bytes, pos: int) -> int:
    bits = _StrictBits(data, pos)
    for _ in range(2):
        if bits.u(1):
            count = bits.u(5)
            bits.s(count)
            bits.s(count)
    count = bits.u(5)
    bits.s(count)
    bits.s(count)
    return bits.end()


def _cxform_end(data: bytes, pos: int) -> int:
    bits = _StrictBits(data, pos)
    add, multiply, count = bits.u(1), bits.u(1), bits.u(4)
    for _ in range((4 if add else 0) + (4 if multiply else 0)):
        bits.s(count)
    return bits.end()


def _add(data: bytes, start: int, end: int, source: str, streams: list[ActionStream]) -> None:
    if len(streams) >= MAX_STREAMS:
        raise _BadSourceError("action source budget exceeded")
    if not 0 <= start <= end <= len(data):
        raise _BadSourceError("action list exceeds its container")
    if len(source) > 1024:
        raise _BadSourceError("action source name budget exceeded")
    if end - start + sum(stream.end_offset - stream.start_offset for stream in streams) > MAX_SOURCE_BYTES:
        raise _BadSourceError("total action source byte budget exceeded")
    streams.append(parse_action_stream(data[start:end], source=source, base_offset=start))


def _u16(data: bytes, pos: int, end: int) -> int:
    if pos < 0 or pos + 2 > end:
        raise _BadSourceError("truncated container integer")
    return int.from_bytes(data[pos:pos + 2], "little")


def _u32(data: bytes, pos: int, end: int) -> int:
    if pos < 0 or pos + 4 > end:
        raise _BadSourceError("truncated container integer")
    return int.from_bytes(data[pos:pos + 4], "little")


def _walk(data: bytes, pos: int, end: int, version: int, source: str,
          streams: list[ActionStream], issues: list[ActionIssue], budget: list[int], depth: int) -> None:
    while pos < end:
        tag_offset = pos
        try:
            if depth > MAX_DEPTH or not budget[0]:
                raise _BadSourceError("tag or nesting budget exceeded")
            budget[0] -= 1
            header = _u16(data, pos, end)
            code, size = header >> 6, header & 63
            pos += 2
            if size == 63:
                size = _u32(data, pos, end)
                pos += 4
            tag_end = pos + size
            if tag_end > end:
                raise _BadSourceError("tag exceeds its container")
            label = f"{source}/tag-{code}@{tag_offset}"
            if code == 0:
                if size or any(data[tag_end:end]):
                    raise _BadSourceError("invalid trailing End tag bytes")
                return
            if code == 12:
                _add(data, pos, tag_end, label + "/frame", streams)
            elif code == 59:
                symbol = _u16(data, pos, tag_end)
                _add(data, pos + 2, tag_end, label + f"/init-{symbol}", streams)
            elif code == 39:
                symbol = _u16(data, pos, tag_end)
                _u16(data, pos + 2, tag_end)
                _walk(data, pos + 4, tag_end, version, label + f"/sprite-{symbol}",
                      streams, issues, budget, depth + 1)
            elif code in (26, 70):
                _clip_sources(data, pos, tag_end, code, version, label, streams)
            elif code in (7, 34):
                _button_sources(data, pos, tag_end, code, label, streams)
            elif code in (72, 82):
                issues.append(ActionIssue(tag_offset, "AS3 action container is unsupported by AVM1 source reader"))
            pos = tag_end
        except (_BadSourceError, IndexError, struct.error) as exc:
            issues.append(ActionIssue(tag_offset, str(exc)))
            return
    issues.append(ActionIssue(end, "tag stream lacks its End tag"))


def _clip_sources(data: bytes, start: int, end: int, code: int, version: int,
                  source: str, streams: list[ActionStream]) -> None:
    tag = data[start:end]
    if not tag:
        raise _BadSourceError("truncated placement flags")
    flags, pos = tag[0], 3
    if not flags & 0x80:
        return
    if code == 70:
        if len(tag) < 4:
            raise _BadSourceError("truncated PlaceObject3 flags")
        flags2, pos = tag[1], 4
        if flags2 & 0x08 or (flags2 & 0x10 and flags & 0x02):
            pos = _string_end(tag, pos)
        if flags2 & ~0x18:
            raise _BadSourceError("PlaceObject3 extension before clip actions is unsupported")
    if flags & 0x02:
        pos += 2
    if flags & 0x04:
        pos = _matrix_end(tag, pos)
    if flags & 0x08:
        pos = _cxform_end(tag, pos)
    if flags & 0x10:
        pos += 2
    if flags & 0x20:
        pos = _string_end(tag, pos)
    if flags & 0x40:
        pos += 2
    width = 4 if version >= 6 else 2
    if pos + 2 + width > len(tag):
        raise _BadSourceError("truncated clip action header")
    pos += 2 + width
    while pos < len(tag):
        if pos + width > len(tag):
            raise _BadSourceError("truncated clip event flags")
        flags = int.from_bytes(tag[pos:pos + width], "little")
        pos += width
        if not flags:
            if pos != len(tag):
                raise _BadSourceError("trailing bytes after clip events")
            return
        size = _u32(tag, pos, len(tag))
        pos += 4
        action_end = pos + size
        if action_end > len(tag):
            raise _BadSourceError("clip event exceeds its placement")
        action_start = pos + (1 if version >= 6 and flags & 0x20000 else 0)
        _add(data, start + action_start, start + action_end, source + f"/clip-{flags:x}", streams)
        pos = action_end
    raise _BadSourceError("clip events lack their terminator")


def _string_end(data: bytes, pos: int) -> int:
    end = data.find(b"\0", pos)
    if end < 0:
        raise _BadSourceError("unterminated placement name")
    return end + 1


def _button_sources(data: bytes, start: int, end: int, code: int,
                    source: str, streams: list[ActionStream]) -> None:
    tag = data[start:end]
    _u16(tag, 0, len(tag))
    if code == 7:
        pos = _button_records_end(tag, 2, second=False)
        _add(data, start + pos, end, source + "/button-release", streams)
        return
    if len(tag) < 5 or tag[2] & 0xFE:
        raise _BadSourceError("invalid DefineButton2 header flags")
    offset = _u16(tag, 3, len(tag))
    record_end = _button_records_end(tag, 5, second=True)
    if not offset:
        if record_end != len(tag):
            raise _BadSourceError("bytes after button records without an action offset")
        return
    pos = 3 + offset
    if pos != record_end:
        raise _BadSourceError("button action offset differs from record boundary")
    while pos < len(tag):
        size = _u16(tag, pos, len(tag))
        conditions = _u16(tag, pos + 2, len(tag))
        action_end = pos + size if size else len(tag)
        if action_end < pos + 4 or action_end > len(tag):
            raise _BadSourceError("button action exceeds its container")
        _add(data, start + pos + 4, start + action_end, source + f"/button-{conditions:x}", streams)
        if not size:
            return
        pos = action_end
    raise _BadSourceError("button actions lack their final record")


def _button_records_end(tag: bytes, pos: int, *, second: bool) -> int:
    while pos < len(tag):
        flags = tag[pos]
        if not flags:
            return pos + 1
        if flags & (0xC0 if second else 0xF0):
            raise _BadSourceError("reserved button record flags")
        if second and flags & 0x30:
            raise _BadSourceError("button filter or blend extension is unsupported")
        _u16(tag, pos + 1, len(tag))
        _u16(tag, pos + 3, len(tag))
        pos = _matrix_end(tag, pos + 5)
        if second:
            pos = _cxform_end(tag, pos)
    raise _BadSourceError("button records lack their terminator")
