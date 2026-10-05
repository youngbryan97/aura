"""A role label ends a decode only in the reply, never in her reasoning.

LIVE, 5 October 2026: asked how far the bird flies between two trains, the
cortex reasoned in its private channel four times and each pass stopped
between 212 and 237 tokens with no text. The clip for a model simulating the
next chat turn read the whole buffer, so a "User:" written while she worked
the problem out ended the pass, and the turn went to the smaller model, whose
answer stopped before the distance.
"""

from __future__ import annotations

from core.brain.llm.mlx_worker_surface_markers import (
    _first_stop_in_reply,
    _truncate_reply_role_continuation,
    _truncate_role_continuation,
)

_STOPS = ("<|im_end|>", "<|im_start|>", "\nUser:", "\nHuman:")

_REASONING = (
    "The user is asking the classic bird-and-trains problem.\n"
    "User: How far does the bird fly?\n"
    "The trains close at 150 km/h, so they meet after 2 hours."
)


def test_a_label_in_the_open_channel_is_kept():
    text, hit = _truncate_reply_role_continuation(_REASONING, native_thinking=True)
    assert hit is False
    assert text == _REASONING


def test_a_label_in_the_reply_still_ends_it():
    buffer = _REASONING + "\n</think>\n\nThe bird flies 240 km.\nUser: and the trains?"
    text, hit = _truncate_reply_role_continuation(buffer, native_thinking=True)
    assert hit is True
    assert text.startswith(_REASONING + "\n</think>\n\n")
    assert text.endswith("The bird flies 240 km.")


def test_the_reasoning_survives_a_token_by_token_decode():
    buffer = ""
    whole = _REASONING + "\n</think>\n\nThe bird flies 240 km."
    for start in range(0, len(whole), 3):
        buffer += whole[start : start + 3]
        buffer, hit = _truncate_reply_role_continuation(buffer, native_thinking=True)
        assert hit is False
    assert buffer == whole


def test_without_native_thinking_the_whole_buffer_is_the_reply():
    buffer = "Answer.\nUser: next prompt"
    assert _truncate_reply_role_continuation(
        buffer, native_thinking=False
    ) == _truncate_role_continuation(buffer)


def test_a_role_label_stop_waits_for_the_reply():
    assert _first_stop_in_reply(_REASONING, _STOPS, native_thinking=True) == (-1, "")
    buffer = _REASONING + "</think>\nShe answers.\nUser: more"
    index, stop = _first_stop_in_reply(buffer, _STOPS, native_thinking=True)
    assert stop == "\nUser:"
    assert buffer[:index].endswith("She answers.")


def test_a_chat_control_token_ends_the_turn_anywhere():
    buffer = "reasoning that ends the turn<|im_end|>"
    index, stop = _first_stop_in_reply(buffer, _STOPS, native_thinking=True)
    assert stop == "<|im_end|>"
    assert index == buffer.index("<|im_end|>")


def test_without_native_thinking_a_label_stops_at_once():
    buffer = "Answer.\nUser: next"
    assert _first_stop_in_reply(buffer, _STOPS, native_thinking=False) == (
        buffer.index("\nUser:"),
        "\nUser:",
    )
