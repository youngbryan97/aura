"""Source addresses distinguish frames, function bodies, sprites and events."""
from __future__ import annotations

import lzma
import struct
import zlib

import pytest

from core.perception.flash_action_sources import read_flash_action_streams


def _tag(code: int, payload: bytes) -> bytes:
    if len(payload) < 63:
        return struct.pack("<H", code << 6 | len(payload)) + payload
    return struct.pack("<HI", code << 6 | 63, len(payload)) + payload


def _swf(tags: bytes) -> bytes:
    body = _rect() + struct.pack("<HH", 24 * 256, 1) + tags + b"\0\0"
    return b"FWS\x06" + struct.pack("<I", len(body) + 8) + body


def _rect(values: tuple[int, ...] = (0, 20, 0, 20)) -> bytes:
    count = max(abs(value).bit_length() + 1 for value in values)
    bits = f"{count:05b}" + "".join(f"{value & ((1 << count) - 1):0{count}b}" for value in values)
    bits += "0" * (-len(bits) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, "big")


def _compressed(raw: bytes, signature: bytes) -> bytes:
    if signature == b"CWS":
        return signature + raw[3:8] + zlib.compress(raw[8:])
    compressed = lzma.compress(raw[8:], format=lzma.FORMAT_ALONE)
    return (b"ZWS\x0d" + raw[4:8] + struct.pack("<I", len(compressed) - 13)
            + compressed[:5] + compressed[13:])


def test_frame_and_init_actions_keep_distinct_absolute_offsets() -> None:
    first = _tag(12, b"\x06\0")
    second = _tag(59, struct.pack("<H", 7) + b"\x07\0")
    got = read_flash_action_streams(_swf(first + second))
    assert got.complete
    assert len(got.streams) == 2
    frame, init = got.streams
    assert frame.start_offset == 2 and frame.end_offset == 4
    assert init.start_offset == len(first) + 4
    assert frame.instructions[0].name == "Play" and init.instructions[0].name == "Stop"
    assert frame.source.endswith("/tag-12@0/frame")
    assert init.source.endswith("/tag-59@4/init-7")
    assert frame.source.startswith(got.source)


def test_sprite_actions_keep_their_parent_container_address() -> None:
    sprite = struct.pack("<HH", 3, 1) + _tag(12, b"\x06\0") + b"\0\0"
    got = read_flash_action_streams(_swf(_tag(39, sprite)))
    assert got.complete and len(got.streams) == 1
    action = got.streams[0]
    assert action.start_offset == 8
    assert action.source.endswith("/tag-39@0/sprite-3/tag-12@6/frame")


def test_clip_event_actions_keep_the_event_flags_and_physical_byte_range() -> None:
    actions = b"\x06\0"
    clip_header = b"\0\0" + struct.pack("<I", 4)
    event = struct.pack("<II", 4, len(actions)) + actions + b"\0\0\0\0"
    placement = b"\x80\x01\0" + clip_header + event
    got = read_flash_action_streams(_swf(_tag(26, placement)))
    assert got.complete and len(got.streams) == 1
    assert got.streams[0].start_offset == 19
    assert got.streams[0].source.endswith("/tag-26@0/clip-4")
    assert got.streams[0].instructions[0].opcode == 6


def test_keypress_event_does_not_misread_its_keycode_as_an_opcode() -> None:
    actions = b"\x07\0"
    flags = 0x20000
    placement = b"\x80\x01\0" + b"\0\0" + struct.pack("<I", flags)
    placement += struct.pack("<II", flags, len(actions) + 1) + b"\xff" + actions + b"\0\0\0\0"
    got = read_flash_action_streams(_swf(_tag(26, placement)))
    assert got.complete
    assert got.streams[0].instructions[0].name == "Stop"
    assert got.streams[0].start_offset == 20


@pytest.mark.parametrize("code,payload,offset", [
    (7, b"\x01\0\0\x06\0", 5),
    (34, b"\x01\0\0\x03\0\0\0\0\x10\0\x07\0", 12),
])
def test_button_action_sources_skip_button_headers_and_records(code: int, payload: bytes, offset: int) -> None:
    got = read_flash_action_streams(_swf(_tag(code, payload)))
    assert got.complete and len(got.streams) == 1
    assert got.streams[0].start_offset == offset
    assert got.streams[0].instructions[0].name in ("Play", "Stop")


@pytest.mark.parametrize("raw", [b"", b"FWS", b"CWS\x06\0\0\0\0nonsense"])
def test_malformed_flash_is_data_not_an_exception(raw: bytes) -> None:
    got = read_flash_action_streams(raw)
    assert not got.complete
    assert got.issues


def test_a_truncated_container_does_not_turn_its_prefix_into_a_complete_program() -> None:
    got = read_flash_action_streams(_swf(struct.pack("<H", 12 << 6 | 20) + b"\x06\0"))
    assert not got.complete and not got.streams
    assert "container" in got.issues[0].reason


def test_as3_is_explicitly_outside_this_avm1_reader() -> None:
    got = read_flash_action_streams(_swf(_tag(82, b"\0\0\0\0\0")))
    assert not got.complete and not got.streams
    assert "AS3" in got.issues[0].reason


@pytest.mark.parametrize("signature", [b"CWS", b"ZWS"])
def test_complete_compressed_stream_preserves_the_same_action_addresses(signature: bytes) -> None:
    raw = _swf(_tag(12, b"\x06\0"))
    got = read_flash_action_streams(_compressed(raw, signature))
    assert got.complete and got.streams[0].start_offset == 2


@pytest.mark.parametrize("signature", [b"CWS", b"ZWS"])
@pytest.mark.parametrize("change", ["truncated", "trailing", "short_length", "long_length"])
def test_compression_eof_and_declared_output_length_are_required(signature: bytes, change: str) -> None:
    raw = _compressed(_swf(_tag(12, b"\x06\0")), signature)
    if change == "truncated":
        raw = raw[:-1]
        if signature == b"ZWS":
            raw = raw[:8] + struct.pack("<I", len(raw) - 17) + raw[12:]
    elif change == "trailing":
        raw += b"garbage"
        if signature == b"ZWS":
            raw = raw[:8] + struct.pack("<I", len(raw) - 17) + raw[12:]
    else:
        declared = int.from_bytes(raw[4:8], "little") + (-1 if change == "short_length" else 1)
        raw = raw[:4] + struct.pack("<I", declared) + raw[8:]
    got = read_flash_action_streams(raw)
    assert not got.complete and not got.streams


@pytest.mark.parametrize("body", [
    b"", b"\xff", _rect(), _rect() + b"\0", _rect() + b"\0\0\x01\0\0\0",
    _rect((20, 0, 0, 20)) + struct.pack("<HH", 24 * 256, 1) + b"\0\0",
])
def test_rect_and_frame_header_must_be_complete_and_have_valid_bounds(body: bytes) -> None:
    raw = b"FWS\x06" + struct.pack("<I", len(body) + 8) + body
    got = read_flash_action_streams(raw)
    assert not got.complete and not got.streams


def test_uncompressed_file_length_cannot_hide_a_missing_movie_header() -> None:
    got = read_flash_action_streams(b"FWS\x06\x08\0\0\0")
    assert not got.complete and not got.streams


def test_uncompressed_file_length_must_match_the_supplied_bytes() -> None:
    raw = _swf(_tag(12, b"\x06\0"))
    for changed in (raw[:-1], raw + b"\0"):
        assert not read_flash_action_streams(changed).complete


def test_an_entire_missing_end_tag_is_not_a_complete_empty_program() -> None:
    raw = _swf(_tag(12, b"\x06\0"))[:-2]
    raw = raw[:4] + struct.pack("<I", len(raw)) + raw[8:]
    got = read_flash_action_streams(raw)
    assert not got.complete and "End" in got.issues[0].reason


@pytest.mark.parametrize("offset", [1, 2, 4, 5, 100])
def test_button2_actions_cannot_start_inside_header_or_after_record_boundary(offset: int) -> None:
    payload = b"\x01\0\0" + struct.pack("<H", offset) + b"\0\0\0\x10\0\x07\0"
    got = read_flash_action_streams(_swf(_tag(34, payload)))
    assert not got.complete and not got.streams
    assert "boundary" in got.issues[0].reason


def test_button2_records_are_parsed_before_an_action_offset_is_trusted() -> None:
    record = b"\x01" + struct.pack("<HH", 4, 1) + b"\0\0"
    records = record + b"\0"
    payload = b"\x01\0\0" + struct.pack("<H", 2 + len(records)) + records + b"\0\0\x10\0\x07\0"
    got = read_flash_action_streams(_swf(_tag(34, payload)))
    assert got.complete and got.streams[0].instructions[0].name == "Stop"
    changed = payload[:3] + struct.pack("<H", 3) + payload[5:]
    refused = read_flash_action_streams(_swf(_tag(34, changed)))
    assert not refused.complete and not refused.streams
