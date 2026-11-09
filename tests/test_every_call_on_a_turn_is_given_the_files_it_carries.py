"""Every model call on a turn that carries files is given them, whichever path makes it.

LIVE 9 November 2026: the final paper was attached and her cortex was handed it; the reply failed a check and the
stabilizer rewrote it from the person's words alone. The rewrite began "I want to be straight with you: I can't see it
right", the gate refused it as contradicting what the turn held, and she answered "I couldn't get my full attention
onto that one". The fix of 8 November had given the files to the fallback ladder only; four other paths (the
stabilizer, the grounded recovery, the social-grounding repair, the protected foreground lane) ask a model on the same
turn. The gate now hands every foreground request the turn's files.
"""
from __future__ import annotations

import asyncio
import contextlib

from core.brain.inference_gate import InferenceGate
from core.conversation import attachments

BLOCK = "[The person attached files to this message.]\n\n=== ATTACHED FILE: ISC_Paper.pdf ===\nAbstract ...\n=== END ISC_Paper.pdf ==="


def _stub_block(monkeypatch):
    def context_block(ids, *, query="", room):
        return (BLOCK, [("ISC_Paper.pdf", "Abstract ...")]) if ids else ("", [])

    monkeypatch.setattr(attachments, "context_block", context_block)


def _on_a_turn_with_the_paper(call):
    async def turn():
        attachments.carry_this_turn(["c9b44bce5996af62"])
        return await call()

    return asyncio.run(turn())


def test_a_prompt_alone_is_given_the_files(monkeypatch):
    _stub_block(monkeypatch)

    async def call():
        return attachments.carried_into_call('The user said: "What do you make of this?"')

    prompt, messages = _on_a_turn_with_the_paper(call)
    assert prompt.startswith(BLOCK) and messages is None


def test_the_last_message_from_the_person_is_given_the_files(monkeypatch):
    _stub_block(monkeypatch)
    asked = [{"role": "system", "content": "You are Aura."}, {"role": "user", "content": "Rewrite the answer."}]

    async def call():
        return attachments.carried_into_call("Rewrite the answer.", asked)

    prompt, messages = _on_a_turn_with_the_paper(call)
    assert prompt == "Rewrite the answer."
    assert messages[0] == asked[0] and messages[1]["content"].startswith(BLOCK)
    assert asked[1]["content"] == "Rewrite the answer.", "the caller's list is not changed under it"


def test_a_call_that_already_holds_them_is_given_nothing_twice(monkeypatch):
    _stub_block(monkeypatch)

    async def call():
        return attachments.carried_into_call(f"{BLOCK}\nUser message: What do you make of this?")

    prompt, _ = _on_a_turn_with_the_paper(call)
    assert prompt.count(attachments.ATTACHED_MARK) == 1


def test_a_turn_with_no_files_changes_nothing(monkeypatch):
    _stub_block(monkeypatch)

    async def call():
        return attachments.carried_into_call("you ok?", [{"role": "user", "content": "you ok?"}])

    prompt, messages = asyncio.run(call())
    assert prompt == "you ok?" and messages == [{"role": "user", "content": "you ok?"}]


class _Gate:
    """InferenceGate.think and .generate as written, around a stand-in for everything past the files."""

    def __init__(self):
        self.seen = []

    def _generation_metadata_sink_slot(self):
        class Slot:
            def get(self):
                return None

            def set(self, value):
                return None

            def reset(self, token):
                return None

        return Slot()

    async def _think_with_generation_metadata_sink(self, prompt, system_prompt="", **kwargs):
        self.seen.append((prompt, kwargs.get("messages")))
        return "read it"


def test_the_stabilizer_s_rewrite_through_the_gate_is_given_the_files(monkeypatch):
    _stub_block(monkeypatch)
    gate = _Gate()
    rewrite = [{"role": "system", "content": "You are Aura."},
               {"role": "user", "content": 'The user said: "Read it and tell me what you think."'}]

    async def call():
        return await InferenceGate.think(gate, rewrite[1]["content"], system_prompt="You are Aura.",
                                         messages=rewrite, origin="api_stabilizer", foreground_request=True)

    assert _on_a_turn_with_the_paper(call) == "read it"
    _prompt, messages = gate.seen[0]
    assert messages[-1]["content"].startswith(BLOCK)


def test_background_work_on_the_same_turn_is_not_given_them(monkeypatch):
    _stub_block(monkeypatch)
    gate = _Gate()

    async def call():
        return await InferenceGate.think(gate, "reflect", foreground_request=False, is_background=True)

    _on_a_turn_with_the_paper(call)
    assert gate.seen[0][0] == "reflect"


def test_a_turn_s_files_do_not_outlive_it():
    with contextlib.suppress(LookupError):
        assert asyncio.run(asyncio.sleep(0, result=attachments.this_turn())) == ()
