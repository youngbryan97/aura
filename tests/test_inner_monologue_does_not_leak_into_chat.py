"""Aura's private thoughts must not enter the conversation as her own turns.

LIVE DEFECT, 2026-07-25. A user said "Just checking in" and got back an
unprompted monologue about ghosts — "leftover bits from other people's
experiences" — followed by the model inventing "<dispatch a somatic probe>"
as if it were speech. The next user message was "Ghosts? what're you talking
about".

Nothing was hallucinated out of nowhere. personhood_engine._emit_thought
writes spontaneous thoughts into ``cognition.working_memory`` as
``role="assistant"`` with ``origin="spontaneous"``. The context assembler
filtered background material by origin, and its prefix list contained
``"spontaneous:"`` — with a colon. ``"spontaneous".startswith("spontaneous:")``
is False, so every spontaneous thought reached the conversational prompt as
one of Aura's own prior TURNS. The model read them as shared context and
continued that voice. ``somatic_noise`` and ``baseline_continuity``, both
global-workspace winners, arrived the same way.

One character of vocabulary drift between the writer and the filter.

The real lesson is that a hand-maintained string list silently diverged from
the code that writes those strings, and the symptom was incoherent speech
rather than a failing test. These tests pin both directions so the next
divergence fails here instead of in front of the user.
"""
from __future__ import annotations

import inspect
import re

import pytest

from core.brain.llm.context_assembler import ContextAssembler
from tests.chat_lane_support import chat_lane_source


def _classifier():
    """The assembler's own background vocabulary, read from its source."""
    source = inspect.getsource(ContextAssembler)
    sources_block = re.search(r"background_sources = \{(.*?)\n        \}", source, re.S)
    prefixes_block = re.search(r"background_prefixes = \((.*?)\)", source, re.S)
    assert sources_block and prefixes_block, "background vocabulary not found"
    sources = {
        line.strip().strip('",')
        for line in sources_block.group(1).split("\n")
        if line.strip().startswith('"')
    }
    prefixes = tuple(
        line.strip().strip('",')
        for line in prefixes_block.group(1).split("\n")
        if line.strip().startswith('"')
    )

    def is_background(origin: str) -> bool:
        value = str(origin or "").strip().lower()
        return value in sources or any(value.startswith(p) for p in prefixes)

    return is_background


# Origins that represent Aura thinking, not Aura speaking to someone.
INTERNAL_ORIGINS = (
    "spontaneous",
    "spontaneous:idle",
    "somatic_noise",
    "baseline_continuity",
    "drive_growth",
    "drive_social",
    "mind_tick",
    "mind_tick_fallback",
    "autonomous_thought",
    "autonomous_volition",
    "subconscious_dream",
    "reflection_impulse",
    "impulse",
    "intention_loop",
    "memory_consolidation",
    "cognitive_background",
    "proactive_presence",
    "agency_core",
)

# Origins that ARE real dialogue and must survive filtering, or the model
# loses the conversation it is supposed to be having.
DIALOGUE_ORIGINS = (
    "user",
    "desktop_quick_user",
    "voice",
    "admin",
    "api",
)


class TestInternalThoughtsAreFilteredFromTheConversation:
    @pytest.mark.parametrize("origin", INTERNAL_ORIGINS)
    def test_internal_origins_do_not_reach_the_prompt(self, origin):
        assert _classifier()(origin) is True, origin

    def test_the_exact_origin_the_writer_emits_is_covered(self):
        """The regression, stated as the writer states it.

        personhood_engine writes the bare string "spontaneous". If this ever
        fails again, Aura will start answering greetings with her own inner
        monologue.
        """
        from core.autonomy import personhood_engine

        source = inspect.getsource(personhood_engine.PersonhoodEngine._emit_thought)
        emitted = re.search(r'"origin":\s*"([^"]+)"', source)
        assert emitted, "could not find the origin this writer emits"
        assert _classifier()(emitted.group(1)) is True, emitted.group(1)

    def test_spontaneous_thoughts_are_written_as_assistant_turns(self):
        """Documents WHY filtering matters: they are indistinguishable from
        speech once they are in working memory."""
        from core.autonomy import personhood_engine

        source = inspect.getsource(personhood_engine.PersonhoodEngine._emit_thought)
        assert '"role": "assistant"' in source
        assert "working_memory.append" in source


class TestRealDialogueSurvives:
    @pytest.mark.parametrize("origin", DIALOGUE_ORIGINS)
    def test_dialogue_origins_are_never_filtered(self, origin):
        """Over-filtering is the opposite failure and just as bad: it erases
        the conversation the reply is supposed to be about."""
        assert _classifier()(origin) is False, origin

    def test_an_unknown_origin_is_treated_as_dialogue(self):
        """Unknown origins default to visible. That is the right default for
        a conversation surface — but it is exactly why the vocabulary must
        be kept in step with the writers, since a NEW internal origin will
        leak until it is listed here."""
        assert _classifier()("some_new_surface") is False


class TestTheVocabularyCannotSilentlyDrift:
    def test_every_working_memory_writer_origin_is_classified(self):
        """The actual defect was drift between writer and filter.

        Any literal origin written into working_memory anywhere in core/
        must be a string this classifier has an opinion about — either
        internal (filtered) or dialogue (kept). A new internal origin that
        nobody adds here fails this test instead of reaching the user.
        """
        import pathlib

        root = pathlib.Path(inspect.getfile(ContextAssembler)).resolve()
        core_dir = root.parent.parent.parent  # core/
        known = set(INTERNAL_ORIGINS) | set(DIALOGUE_ORIGINS)
        unclassified: list[str] = []

        for path in core_dir.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for block in re.finditer(
                r"working_memory\.append\((.{0,400}?)\)", text, re.S,
            ):
                origin = re.search(r'"origin":\s*"([a-z0-9_:.]+)"', block.group(1))
                if origin and origin.group(1) not in known:
                    unclassified.append(f"{path.name}: {origin.group(1)}")

        assert not unclassified, (
            "working-memory origins not classified as internal or dialogue: "
            + "; ".join(sorted(set(unclassified)))
        )


class TestTheLiveStateResolverReturnsAState:
    """LIVE DEFECT, 2026-07-25. ``_resolve_live_aura_state`` returned
    whatever was registered under "aura_state" without checking its type. On
    a live boot that was a StateRepository, so every caller hit ``.cognition``
    on it:

        AttributeError: 'StateRepository' object has no attribute 'cognition'

    That crashed the required-search contract, which is why "search for an
    article on how LeBron James will fit in with the 76ers" produced no
    search at all — the capability did not decline, it raised.
    """

    def _state(self):
        from core.state.aura_state import AuraState

        return AuraState.default()

    def test_a_real_state_passes_through(self):
        from interface.routes.chat import _unwrap_state

        state = self._state()
        assert _unwrap_state(state) is state

    def test_a_repository_is_unwrapped_to_its_state(self):
        from interface.routes.chat import _unwrap_state

        state = self._state()

        class _Repo:
            _current = state

        assert _unwrap_state(_Repo()) is state

    def test_a_repository_is_not_mistaken_for_a_state(self):
        from interface.routes.chat import _looks_like_aura_state

        class _Repo:
            _current = None

        assert _looks_like_aura_state(_Repo()) is False

    def test_an_unusable_object_resolves_to_none(self):
        from interface.routes.chat import _unwrap_state

        assert _unwrap_state(object()) is None
        assert _unwrap_state(None) is None

    def test_the_resolver_validates_every_source(self):
        import inspect

        from interface.routes import chat as mod

        source = inspect.getsource(mod._resolve_live_aura_state)
        # No raw registry value may be returned unchecked.
        assert 'ServiceContainer.get("aura_state", default=None)' in source
        assert "_unwrap_state(ServiceContainer.get" in source
        assert 'getattr(repo, "_current", None) if repo is not None else None' not in source


class TestBodyPressureActuallyReads:
    """LIVE DEFECT, 2026-07-25. ``total_pressure`` is a @property, and the
    caller invoked it as ``total_pressure()`` — so every call raised
    ``TypeError: 'float' object is not callable`` and fell into the handler,
    which answered 0.0. On this scale 0.0 means MAXIMUM headroom, so the
    latent cortex ran undamped and never once saw a real reading.

    It surfaced only because the handler was changed to record itself.
    """

    def test_total_pressure_is_a_property_not_a_method(self):
        from core.being.aura_now import BodyState

        assert isinstance(
            inspect.getattr_static(BodyState, "total_pressure"), property,
        )

    def test_the_caller_reads_it_as_a_property(self):
        from core.brain import latent_cortex_service as mod

        source = inspect.getsource(mod.LatentCortexService._body_pressure)
        assert ".total_pressure)" in source
        assert ".total_pressure()" not in source

    def test_a_real_reading_comes_back(self):
        from core.being.aura_now import BodyState

        value = float(BodyState.from_aura_state(None).total_pressure)
        assert 0.0 <= value <= 1.0


class TestASessionIsAConversationNotAMachine:
    """LIVE DEFECT, 2026-07-25. The fallback chat session id was the client
    IP, with the comment "client host is good enough for single-user local
    Aura". It was not.

    Every desktop conversation ever held collapsed into ONE session — 881
    turns filed under "127.0.0.1", mixing Bryan's real chats with Codex's
    automated live-route probes and latency samples — and each new chat
    replayed that soup as its own prior thread.

    That is what produced the confabulations. Asked "Hey, bud. Are you with
    me?", Aura answered "Sitting with you on the drive. They're still
    outside?" and then "Well, the cops. You called them?" — a scene that
    appears nowhere in her stored memory except those two replies. She was
    not recalling anything; she was continuing unrelated fragments the replay
    had put in front of her and inventing a situation that made them cohere.
    """

    def _sid(self, host, now):
        from interface.routes.chat import _conversation_session_id

        return _conversation_session_id(host, now=now)

    def test_a_continuing_turn_keeps_the_same_conversation(self):
        assert self._sid("h", 1000.0) == self._sid("h", 1060.0)

    def test_an_idle_gap_starts_a_new_conversation(self):
        from interface.routes.chat import _CONVERSATION_IDLE_GAP_S as gap

        first = self._sid("h2", 1000.0)
        later = self._sid("h2", 1000.0 + gap + 1.0)
        assert first != later

    def test_different_clients_never_share_a_conversation(self):
        assert self._sid("a", 1000.0) != self._sid("b", 1000.0)

    def test_the_session_id_is_never_a_bare_host(self):
        """The exact shape that collapsed 881 turns into one thread."""
        assert self._sid("127.0.0.1", 1000.0) != "127.0.0.1"

    def test_a_restart_is_a_new_conversation(self):
        """A boot id is mixed in, so history from a previous runtime cannot
        replay as the current thread."""
        from interface.routes import chat as mod

        assert mod._CONVERSATION_BOOT_ID
        assert mod._CONVERSATION_BOOT_ID in self._sid("h", 1000.0)

    def test_the_host_only_fallback_is_gone(self):
        source = chat_lane_source()
        assert "client host is good enough for single-user local Aura" not in source
        assert "_conversation_session_id(_host)" in source


class TestDegradedTurnsSpeakToAPerson:
    """LIVE DEFECT, 2026-07-25. Two defects in one sentence, both of which
    reached Bryan verbatim:

        "the answer path did not produce a clean enough draft, so I should
        not treat a synthetic fallback as my real answer. The grounded
        anchor is your question about reference and confus"

    His whole reply was "I'm confused. What is this in reference to".

    1. "confus" — ``_normalize_topic_token`` strips suffixes so that tokens
       COMPARE equal ("confused" -> "confus"). Correct for matching, wrong
       for speech, and the stem was rendered straight into the sentence.
    2. The register — a degraded turn narrated its own pipeline. Saying "I
       couldn't answer" is honest; saying "the answer path did not produce a
       clean enough draft" is the same fact in a vocabulary that leaves the
       person with nothing to do.

    Being honest about a bad turn is right. This pins that it is done in
    words a person can act on.
    """

    def _reply(self, message, reason="degraded_turn"):
        from interface.routes.chat import _build_degraded_live_reply

        return _build_degraded_live_reply(
            {"attention_focus": "", "dominant_action": "reflect"},
            message,
            reason=reason,
        )

    def test_stems_are_matching_keys_not_words_to_say(self):
        from interface.routes.chat import _normalize_topic_token

        # The stemmer itself is unchanged; it is supposed to do this.
        assert _normalize_topic_token("confused") == "confus"

    def test_no_truncated_stem_reaches_the_reply(self):
        assert "confus" not in self._reply(
            "I am confused. What is this in reference to",
        ).replace("confused", "")

    def test_the_reply_admits_the_failure_plainly(self):
        """Plainly, in any of the ways a person says it.

        This asserted the literal contraction "couldn't". The reply now says
        "no reading — it has never produced a sample, so I cannot tell", which
        is the same admission in the same register, and the test failed it.
        A test that pins one spelling of an idea makes every rewording a
        regression and every register change invisible — and the register is
        the thing this class exists to protect.
        """

        reply = self._reply("I am confused. What is this in reference to").lower()
        plain_admissions = (
            "no reply formed",
            "couldn't",
            "could not",
            "cannot",
            "can't",
            "no reading",
            "never sampled",
            "don't have",
            "do not have",
            "wasn't able",
        )
        assert any(phrase in reply for phrase in plain_admissions), (
            f"a degraded turn said nothing a person can read as a failure: {reply!r}"
        )

    def test_the_reply_carries_no_implementation_vocabulary(self):
        """The exact phrases that made the live reply unreadable."""
        leaks = (
            "answer path",
            "synthetic fallback",
            "grounded anchor",
            "clean enough draft",
            "recovery shape",
            "repair path",
            "bounded to that anchor",
        )
        for reason in (
            "degraded_turn",
            "repeated_reflex",
            "desktop_cognitive_engine_repair_failed",
        ):
            reply = self._reply("why is the deployment failing", reason=reason).lower()
            for phrase in leaks:
                assert phrase not in reply, f"{reason}: {phrase}"

    def test_the_reply_tells_the_user_what_to_do_next(self):
        reply = self._reply("why is the deployment failing").lower()
        assert "send it again" in reply

    def test_the_notice_is_the_runtimes_not_hers(self):
        """Since 18 November the last resort is the runtime's notice. The
        person sees their own message beside it, so it does not give their
        words back, and it says nothing in her first person."""
        from core.conversation.runtime_notice import is_the_runtimes_notice

        reply = self._reply("why is the deployment failing")
        assert is_the_runtimes_notice(reply)
        assert "I " not in f" {reply} "
