"""Whether a turn probes her mind is read from the person's words, not from the prompt built around them.

LIVE 2026-10-05: "The Pong game at .../pong.html is broken. Fix it, then play
it against the computer until you win." was answered as conversation. The
prompt the gate was handed carried her own scaffold and a history of talk about
consciousness, it read as a deep probe of her mind, and a probe's turn has its
tools taken away, so the repair was never offered.
"""
from __future__ import annotations

from pathlib import Path

from core.conversation.session_scope import the_persons_own_words, user_question_var
from core.runtime.turn_analysis import looks_like_deep_mind_probe

_ASKED = "The Pong game at /Users/bryan/aura-demos/pong/pong.html is broken. Fix it, then play it against the computer until you win."
_AROUND_IT = "You are Aura. Questions about your consciousness deserve an honest answer, and sentience is open.\n\n" + _ASKED


def test_the_words_around_a_request_are_not_the_request():
    assert looks_like_deep_mind_probe(_AROUND_IT)
    token = user_question_var.set(_ASKED)
    try:
        assert not looks_like_deep_mind_probe(the_persons_own_words(_AROUND_IT))
    finally:
        user_question_var.reset(token)


def test_the_gate_reads_the_probe_from_the_person():
    source = Path("core/brain/inference_gate.py").read_text(encoding="utf-8")
    assert "looks_like_deep_mind_probe(the_persons_own_words(prompt))" in source
