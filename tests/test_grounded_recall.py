"""Tests for positional/temporal grounded recall (anti-confabulation)."""
from __future__ import annotations

import pytest

from core.conversation import grounded_recall as gr


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("Do you remember what I first asked", "first"),
        ("what did I first ask", "first"),
        ("what was my first question?", "first"),
        ("the first thing I said to you", "first"),
        ("how did this conversation start", "first"),
        ("what did we start talking about", "first"),
        ("what did I just ask you", "last"),
        ("what was my previous question", "last"),
        # negatives
        ("what is the capital of France", None),
        ("can you open notes", None),
        ("tell me about yourself", None),
        ("how are you feeling", None),
    ],
)
def test_detect_positional_recall(msg, expected):
    assert gr.detect_positional_recall(msg) == expected


def test_resolve_uses_live_transcript_first_and_last(monkeypatch):
    class _Entry:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    class _FakeTranscript:
        _entries = [
            _Entry("aura", "Infinity online."),
            _Entry("user", "you with me, Aura?"),
            _Entry("aura", "Yeah, I'm here."),
            _Entry("user", "checking the convo lane"),
            _Entry("user", "Do you remember what I first asked"),
        ]

        @classmethod
        def get_instance(cls):
            return cls()

        def entries_for_conversation(self):
            return list(self._entries)

    import core.conversation.unified_transcript as ut
    monkeypatch.setattr(ut, "UnifiedTranscript", _FakeTranscript)

    # current turn is excluded; first real user turn is the grounding fact
    first = gr.resolve_positional_turn("Do you remember what I first asked", "first")
    assert first == "you with me, Aura?"
    last = gr.resolve_positional_turn("Do you remember what I first asked", "last")
    assert last == "checking the convo lane"


def test_build_context_block_contains_quote(monkeypatch):
    class _Entry:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    class _FakeTranscript:
        _entries = [_Entry("user", "you with me, Aura?"),
                    _Entry("user", "Do you remember what I first asked")]

        @classmethod
        def get_instance(cls):
            return cls()

        def entries_for_conversation(self):
            return list(self._entries)

    import core.conversation.unified_transcript as ut
    monkeypatch.setattr(ut, "UnifiedTranscript", _FakeTranscript)

    block = gr.build_grounded_recall_context("Do you remember what I first asked")
    assert block is not None
    assert "you with me, Aura?" in block
    assert "GROUNDED RECALL" in block
    assert "quoted speaker is the user, not you" in block
    assert "never as something you said" in block


def test_get_world_state_registers_canonical_singleton(monkeypatch):
    registrations = []

    from core import world_state as world_state_module

    monkeypatch.setattr(world_state_module, "_ws_instance", None)
    monkeypatch.setattr(
        world_state_module,
        "has_runtime_service",
        lambda name: bool(registrations),
    )
    monkeypatch.setattr(
        world_state_module,
        "register_runtime_service",
        lambda name, instance, **metadata: registrations.append(
            (name, instance, metadata)
        ),
    )

    first = world_state_module.get_world_state()
    second = world_state_module.get_world_state()

    assert first is second
    assert len(registrations) == 1
    assert registrations[0][0] == "world_state"
    assert registrations[0][1] is first
    assert registrations[0][2]["required_for"] == "live environment grounding"


def test_build_context_none_when_no_prior_turn(monkeypatch):
    class _FakeTranscript:
        _entries = []

        @classmethod
        def get_instance(cls):
            return cls()

        def entries_for_conversation(self):
            return list(self._entries)

    import core.conversation.unified_transcript as ut
    monkeypatch.setattr(ut, "UnifiedTranscript", _FakeTranscript)
    # The cascade now ends at the episodic store, which survives a restart —
    # so "no prior turn" has to be CONSTRUCTED rather than assumed. Without
    # this the test reads whatever the developer said to Aura this week.
    import core.conversation.durable_turns as dt

    monkeypatch.setattr(dt, "durable_turn_texts", lambda **_kwargs: [])

    assert gr.build_grounded_recall_context("what did I first ask") is None


def test_build_context_none_for_non_recall():
    assert gr.build_grounded_recall_context("what's the weather like") is None


@pytest.mark.parametrize(
    "reply,quote,expected",
    [
        (
            "I care more about reliability than spectacle because trust is built slowly.",
            "I care more about reliability than spectacle because trust is built slowly.",
            "You said you care more about reliability than spectacle because trust is built slowly.",
        ),
        (
            "I said reliability matters because consistency earns trust.",
            "reliability matters because consistency earns trust",
            "You said reliability matters because consistency earns trust.",
        ),
        (
            "My point was that dependable behavior matters. Spectacle fades.",
            "dependable behavior matters",
            "You said your point was that dependable behavior matters. Spectacle fades.",
        ),
    ],
)
def test_repair_grounded_recall_speaker_attribution(reply, quote, expected):
    repaired, changed = gr.repair_grounded_recall_speaker_attribution(
        "What did I just say?",
        reply,
        quote,
    )
    assert changed is True
    assert repaired == expected


def test_repair_grounded_recall_does_not_rewrite_ordinary_first_person_reply():
    reply = "I am checking my uncertainty before I answer."
    repaired, changed = gr.repair_grounded_recall_speaker_attribution(
        "How does confusion affect your reasoning?",
        reply,
        "some earlier thing he said",
    )
    assert changed is False
    assert repaired == reply


def test_repair_leaves_her_own_sentence_alone_when_it_is_not_the_quote():
    """LIVE DEFECT 2026-08-10: a correct answer inverted into a false quote.

    "do you remember what we were talking about" tripped the positional
    detector on "we talked earlier"; her true reply "I remember the
    conversation." opened with "I"; the repair rewrote it into "You said you
    remember the conversation." — crediting him with a sentence he never spoke
    and stripping her of one she did.
    """
    reply = "I remember the conversation. It's not gone."
    repaired, changed = gr.repair_grounded_recall_speaker_attribution(
        "we talked earlier today and then i restarted you. do you remember "
        "what we were talking about, or is that gone?",
        reply,
        "that last answer was wrong - i pasted you a note and you searched the web",
    )
    assert changed is False
    assert repaired == reply


def test_repair_requires_a_quote_to_attribute():
    """With nothing retrieved, nothing can have been misattributed."""
    reply = "I care more about reliability than spectacle."
    for quote in (None, "", "   "):
        repaired, changed = gr.repair_grounded_recall_speaker_attribution(
            "What did I just say?", reply, quote
        )
        assert changed is False
        assert repaired == reply


def test_grounded_quote_from_context_round_trips(monkeypatch):
    class _Entry:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    class _FakeTranscript:
        _entries = [_Entry("user", "you with me, Aura?"),
                    _Entry("user", "Do you remember what I first asked")]

        @classmethod
        def get_instance(cls):
            return cls()

        def entries_for_conversation(self):
            return list(self._entries)

    import core.conversation.unified_transcript as ut
    monkeypatch.setattr(ut, "UnifiedTranscript", _FakeTranscript)

    block = gr.build_grounded_recall_context("Do you remember what I first asked")
    assert gr.grounded_quote_from_context(block) == "you with me, Aura?"
    assert gr.grounded_quote_from_context(None) is None
    assert gr.grounded_quote_from_context("no quote here") is None


def test_repair_fires_on_verbatim_adoption_of_the_retrieved_turn(monkeypatch):
    """The case the repair exists for, end to end through the block."""

    class _Entry:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    class _FakeTranscript:
        _entries = [
            _Entry("user", "reliability matters more than spectacle"),
            _Entry("user", "what did I just say?"),
        ]

        @classmethod
        def get_instance(cls):
            return cls()

        def entries_for_conversation(self):
            return list(self._entries)

    import core.conversation.unified_transcript as ut
    monkeypatch.setattr(ut, "UnifiedTranscript", _FakeTranscript)

    block = gr.build_grounded_recall_context("what did I just say?")
    repaired, changed = gr.repair_grounded_recall_speaker_attribution(
        "what did I just say?",
        "I said reliability matters more than spectacle.",
        gr.grounded_quote_from_context(block),
    )
    assert changed is True
    assert repaired == "You said reliability matters more than spectacle."


class _Turn:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def get(self, key, default=None):
        return {"role": self.role, "content": self.content}.get(key, default)


def _history(*pairs):
    out = []
    for prompt, answer in pairs:
        out.append({"role": "user", "content": prompt})
        out.append({"role": "assistant", "content": answer})
    return out


def test_a_refusal_is_never_quoted_back_as_her_position():
    """LIVE DEFECT 2026-08-10: the first answer became permanent.

    Own-statement recall scores past exchanges by overlap with the current
    question, so re-asking resolves to her previous attempt at that same
    question. The block then instructs her not to report a different position
    than the one quoted. When that quote is "I can't reach that conversation",
    the question can never be answered differently — which is exactly the case
    where a different answer is wanted.
    """
    history = _history(
        (
            "earlier today we had a long conversation, name one thing from it",
            "I can't reach that conversation — it's not available to me.",
        ),
    )
    resolved = gr.resolve_own_prior_turn(
        "earlier today we had a long conversation - tell me one concrete thing "
        "from it, something you said",
        history=history,
    )
    assert resolved is None


def test_a_real_position_is_still_recalled_across_a_reask():
    """The behaviour the grounding exists for must survive the fix."""
    history = _history(
        (
            "if you had to give up one of your senses, which goes?",
            "If I had to give up one, the screen. Losing telemetry would be worse.",
        ),
        (
            "unrelated: what time is it",
            "I don't have a clock reading to give you.",
        ),
    )
    resolved = gr.resolve_own_prior_turn(
        "which of your senses did you pick, and has your answer changed?",
        history=history,
    )
    assert resolved is not None
    assert "the screen" in resolved


@pytest.mark.parametrize(
    "turn",
    [
        "I can't reach that conversation — it's not available to me.",
        "I have no memory of it.",
        "I cannot name a thing from our previous conversation.",
        "I couldn't get to an answer I'd stand behind on that one.",
        "I don't remember what we discussed.",
    ],
)
def test_states_no_position_recognises_an_absent_answer(turn):
    assert gr.states_no_position(turn) is True


@pytest.mark.parametrize(
    "turn",
    [
        "If I had to give up one, the screen.",
        "I picked reliability over spectacle.",
        "I remember the conversation. It's not gone.",
        "I can reach the transcript — we talked about senses.",
    ],
)
def test_states_no_position_leaves_real_answers_alone(turn):
    assert gr.states_no_position(turn) is False


def test_provenance_words_did_not_replace_the_recall_scorer():
    """Two different questions, two different word sets — they must not merge.

    _content_words singularises for topic scoring ("senses" must match
    "sense"); the provenance check must not, and neither may shadow the other.
    """
    assert "sense" in gr._content_words("which of your senses")
    assert "sense" not in gr._provenance_words("which of your senses")
    assert "senses" in gr._provenance_words("which of your senses")


def test_missing_origin_is_not_treated_as_human_without_ingress_provenance():
    assert not gr._entry_is_from_the_human({"role": "user", "content": "internal"})
    assert gr._entry_is_from_the_human(
        {
            "role": "user",
            "content": "typed",
            "metadata": {"source": "chat_api"},
        }
    )


def test_origin_classifier_failure_excludes_the_entry(monkeypatch):
    import core.state.aura_state as aura_state

    def fail_classifier(_origin):
        raise ValueError("classifier unavailable")

    monkeypatch.setattr(aura_state, "_origin_is_user_anchored", fail_classifier)
    assert not gr._entry_is_from_the_human(
        {"role": "user", "content": "unverified", "origin": "user"}
    )
