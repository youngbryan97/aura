"""A typed turn while her runtime is still starting is answered by her cortex, in her words.

Bryan, 11 November 2026: "She should say she's still waking up (in her words) if she still is." LIVE 12 November the
first typed turn a minute after her app opened got a line her chat route wrote in her first person: "the live answer
lane could not finish preparing before a reasoning turn began."
"""
from __future__ import annotations

import asyncio

from core.voice.duplex import live_context
from interface.routes import chat_memory_state, chat_preflight, chat_while_waking


def _patched(monkeypatch, *, waking: str, reply: str | None):
    asked, logged = [], []

    async def generate(prompt, *, scope, words, reply_stream, timeout_s):
        asked.append(prompt)
        return reply

    async def exchanges(**_kwargs):
        return []

    async def log(words, said, session_id=""):
        logged.append((words, said))

    monkeypatch.setattr(live_context, "still_waking", lambda: waking)
    monkeypatch.setattr(live_context, "_generate_on_her_cortex", generate)
    monkeypatch.setattr(live_context, "the_turns_evidence",
                        lambda: "[GROUNDING EVIDENCE FOR THIS TURN]\n- Still starting: the conversation lane reads "
                                "'cold', not yet serving\n[END GROUNDING EVIDENCE FOR THIS TURN]\n\n" if waking else "")
    monkeypatch.setattr(chat_memory_state, "_recent_completed_conversation_exchanges", exchanges)
    monkeypatch.setattr(chat_preflight, "_log_exchange", log)
    return asked, logged


def test_while_she_wakes_her_cortex_answers_on_the_reading_and_the_exchange_is_kept(monkeypatch):
    asked, logged = _patched(monkeypatch, waking="cold", reply="Still waking up, give me a second. I'm here though.")
    said = asyncio.run(chat_while_waking.her_reply_while_waking("How are you doing?", session_id="s", timeout_s=30))
    assert said == "Still waking up, give me a second. I'm here though."
    assert "Still starting" in asked[0] and asked[0].rstrip().endswith("</think>")
    assert "You are" not in asked[0] and "Do not" not in asked[0], "a reading, never an instruction"
    assert logged == [("How are you doing?", said)]


def test_awake_or_with_nothing_formed_the_route_keeps_its_own_path(monkeypatch):
    asked, _logged = _patched(monkeypatch, waking="", reply="unused")
    assert asyncio.run(chat_while_waking.her_reply_while_waking("Hi", session_id="s", timeout_s=30)) is None
    assert asked == [], "awake, her chat lane answers; nothing is generated here"
    _patched(monkeypatch, waking="cold", reply=None)
    assert asyncio.run(chat_while_waking.her_reply_while_waking("Hi", session_id="s", timeout_s=30)) is None
