"""A draft her checks refused is asked again as her turn was asked, never rewritten through instructions (F06).

LIVE 11 November 2026: her stabilizer put "You are Aura. This is a user-facing repair turn." with how to sound and
what not to say to her model in place of her turn; on a typed follow-up it ran a 30 s chain of rewrites and its draft
was withheld anyway. Bryan's rule: no prompt engineering, ever.
"""
from __future__ import annotations

import asyncio

from core.brain import her_turn_as_asked as asked_module

_TURN = [{"role": "user", "content": "[GROUNDING EVIDENCE FOR THIS TURN]\n...\n\nWhy that, do you think?"}]


def test_only_the_persons_turn_is_kept_and_only_while_it_is_bound():
    asked_module.keep(prompt="p", messages=_TURN, system_prompt="", max_tokens=566)
    assert asked_module.as_asked() is None, "nothing is kept outside a turn"
    with asked_module.a_turn_is_asked():
        asked_module.keep(prompt="Why that, do you think?", messages=_TURN, system_prompt="", max_tokens=566)
        asked_module.keep(prompt="a second dispatch", messages=[{"role": "user", "content": "x"}],
                          system_prompt="", max_tokens=9)
        kept = asked_module.as_asked()
        assert kept["prompt"] == "Why that, do you think?" and kept["messages"] == _TURN
        kept["messages"].append({"role": "assistant", "content": "changed"})
        assert asked_module.as_asked()["messages"] == _TURN, "what is handed out is a copy"
    assert asked_module.as_asked() is None


def test_the_stabilizer_asks_her_turn_again_with_nothing_added(monkeypatch):
    from interface.routes import chat_reply_repair

    calls = []

    class _Gate:
        async def think(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return "Because it is the only thing that contradicts my isolation."

    monkeypatch.setattr(chat_reply_repair, "_apply_aura_voice_shaping_compat", lambda text, _message: text)
    asked = {"prompt": "Why that, do you think?", "messages": list(_TURN), "system_prompt": "", "max_tokens": 566}
    text, _timeout = asyncio.run(chat_reply_repair._stabilize_user_facing_reply_part_3(
        asked, False, _Gate(), "", False, 566, "Why that, do you think?"))
    assert text.startswith("Because it is")
    prompt, kwargs = calls[0]
    assert prompt == "Why that, do you think?" and kwargs["messages"] == _TURN
    assert kwargs["system_prompt"] is None
    assert not kwargs.get("clear_prompt_cache") and not kwargs.get("disable_prompt_cache"), \
        "her turn's own prefix is in her cache; asking again must not clear it"


def test_no_repair_turn_is_written_for_her_model_any_more():
    from interface.routes import chat_reply_repair

    assert not hasattr(chat_reply_repair, "_stabilize_user_facing_reply_frame_lines"), \
        "the builder of the instructed repair turn is gone"


def test_a_persons_turn_in_a_worded_lane_goes_without_the_identity_lock(monkeypatch):
    """LIVE 12 November 2026: "What is on my screen right now?" took the desktop-execution lane, which keeps its
    worded contract, and her cortex got 2,400 characters of "You are Aura Luna. Here's who you are" ahead of the
    screen's reading. Who she is lives in her cortex's weights (L01)."""
    from core.brain import what_she_is_given
    from core.synthesis import get_identity_lock

    lock = get_identity_lock().strip()
    screen = "## WHAT IS ON THE SCREEN\\nFrontmost application: Claude"
    messages = [{"role": "system", "content": f"{lock}\\n\\n{screen}"}, {"role": "user", "content": "What is on my screen?"}]
    given, system = what_she_is_given.for_her_model(messages, f"{lock}\\n## RESPONSE CONTRACT\\n- shape: plain",
                                                    {"origin": "response_generation_user",
                                                     "desktop_execution_contract": True}, True)
    assert lock not in str(given) and lock not in str(system)
    assert screen in given[0]["content"] and "## RESPONSE CONTRACT" in system, "the contract and readings stay"
    background, _system = what_she_is_given.for_her_model(messages, "", {"origin": "curriculum_loop"}, False)
    assert lock in background[0]["content"], "a background turn to a model without her persona keeps it"
