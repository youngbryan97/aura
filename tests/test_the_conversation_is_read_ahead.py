"""Between turns her worker reads the conversation the next turn will go on from (core/brain/llm/reading_ahead.py)."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from core.brain import her_turn_as_asked
from core.brain.llm.prompt_cache import PromptCacheLRU
from core.brain.llm.reading_ahead import read_through, the_talk_so_far
from interface.routes import chat_reading_ahead as ahead

TALK = [{"role": "user", "content": "how is the basil"}, {"role": "assistant", "content": "It wants water."}]


def _rendered(messages, *, newest_assistant_kept=True):
    """A template of the kind hers is: readings go to the model as a tool's results, inside the user's turn."""
    written, wrapped = "", False
    for message in messages:
        role, content = message["role"], message["content"]
        if role == "runtime_evidence":
            written += ("" if wrapped else "<|im_start|>user") + f"\n<tool_response>\n{content}\n</tool_response>"
            wrapped = True
            continue
        if wrapped:
            written, wrapped = written + "<|im_end|>\n", False
        written += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return written + "<|im_start|>assistant\n"


def test_the_talk_so_far_is_what_any_next_turn_begins_with():
    talk = the_talk_so_far(TALK, _rendered)
    assert talk == "<|im_start|>user\nhow is the basil<|im_end|>\n<|im_start|>assistant\nIt wants water.<|im_end|>\n"
    for following in ([{"role": "user", "content": "and the mint?"}],
                      [{"role": "runtime_evidence", "content": "## MOOD\ncalm"}, {"role": "user", "content": "and the mint?"}]):
        assert _rendered([*TALK, *following]).startswith(talk)


def test_where_a_next_turn_changes_what_came_before_it_nothing_is_taken_for_shared():
    def a_template_that_marks_the_last_reply(messages):
        last = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
        return _rendered([{**m, "content": f"{len(messages)} {m['content']}"} if i == last else m for i, m in enumerate(messages)])

    assert the_talk_so_far(TALK, a_template_that_marks_the_last_reply) == "<|im_start|>user\nhow is the basil<|im_end|>\n"


class _Cache(list):
    """What was read into it, in order."""


def _reading(lru, tokens, *, stop_after=None, step=4):
    parts = []

    def forward(part, cache):
        parts.append(list(part))
        cache.extend(part)

    read = read_through(
        tokens, lru=lru, model_key=("m", "user_surface"), make_cache=_Cache, forward=forward, step=step,
        stopped=lambda: stop_after is not None and len(parts) >= stop_after, can_trim=lambda cache: False, trim=lambda cache, n: None,
    )
    return read, parts


def test_it_is_read_once_and_the_next_turn_finds_it():
    lru, tokens = PromptCacheLRU(max_size=5), list(range(100, 110))
    assert _reading(lru, tokens) == ((10, 0), [[100, 101, 102, 103], [104, 105, 106, 107], [108, 109]])
    assert _reading(lru, tokens) == ((0, 10), [])
    cache, left = lru.fetch_nearest_cache(("m", "user_surface"), [*tokens, 7, 8], can_trim_prompt_cache=lambda c: False,
                                          trim_prompt_cache=lambda c, n: None)
    assert left == [7, 8] and list(cache) == tokens


def test_only_what_was_not_read_is_read():
    lru, tokens = PromptCacheLRU(max_size=5), list(range(100, 106))
    _reading(lru, tokens)
    read, parts = _reading(lru, [*tokens, 1, 2, 3])
    assert read == (3, 6) and parts == [[1, 2, 3]]


def test_a_turn_that_comes_meanwhile_stops_it_and_what_was_read_is_kept():
    lru, tokens = PromptCacheLRU(max_size=5), list(range(100, 112))
    assert _reading(lru, tokens, stop_after=2)[0] == (8, 0)
    assert _reading(lru, tokens)[0] == (4, 8)


def _served(reply):
    return SimpleNamespace(body=json.dumps({"response": reply, "status": "ok"}).encode())


@pytest.mark.parametrize(("dispatched", "read"), [
    ([*TALK, {"role": "runtime_evidence", "content": "## MOOD\ncalm"}, {"role": "user", "content": "[WHAT THE RECORD SAYS] and the mint?"}],
     [*TALK, {"role": "user", "content": "and the mint?"}, {"role": "assistant", "content": "Fine."}]),
    ([{"role": "system", "content": "Answer as JSON."}, {"role": "user", "content": "and the mint?"}], []),
    ([*TALK], []),
])
def test_the_next_turn_reads_the_conversation_and_not_this_turns_readings(dispatched, read):
    assert ahead.what_the_next_turn_reads(dispatched, "and the mint?", "Fine.") == read


def _after_a_turn(*, good_for_s, then):
    asked = []

    async def read_ahead(messages):
        asked.append(messages)
        return 5, 0

    async def turn():
        with her_turn_as_asked.a_turn_is_asked("c"):
            her_turn_as_asked.keep(prompt="", messages=[*TALK, {"role": "user", "content": "and the mint?"}], system_prompt="", max_tokens=1)
            assert ahead.read_ahead_of_the_next_turn("and the mint?", _served("Fine."), client=SimpleNamespace(read_ahead=read_ahead),
                                                     good_for_s=good_for_s)
        await then()

    asyncio.run(turn())
    return asked


def test_where_no_state_is_held_for_the_next_turn_it_is_read_at_once():
    async def a_moment():
        await asyncio.sleep(0.01)

    assert _after_a_turn(good_for_s=0.0, then=a_moment)[0][-1] == {"role": "assistant", "content": "Fine."}


def test_where_one_is_held_it_is_read_only_once_that_is_no_longer_good():
    async def a_turn_comes():
        await asyncio.sleep(0.01)
        ahead.stop_waiting_to_read()
        await asyncio.sleep(0.1)

    assert _after_a_turn(good_for_s=0.05, then=a_turn_comes) == []

    async def nobody_comes():
        await asyncio.sleep(0.1)

    assert len(_after_a_turn(good_for_s=0.05, then=nobody_comes)) == 1


def test_a_reply_nobody_was_given_and_a_client_that_cannot_read_ahead_begin_nothing():
    async def turn():
        with her_turn_as_asked.a_turn_is_asked("c"):
            her_turn_as_asked.keep(prompt="", messages=[{"role": "user", "content": "hi"}], system_prompt="", max_tokens=1)
            assert not ahead.read_ahead_of_the_next_turn("hi", _served(""), client=SimpleNamespace(read_ahead=None), good_for_s=0)

    asyncio.run(turn())
    assert ahead.what_the_next_turn_reads([{"role": "user", "content": "hi"}], "hi", "") == []
