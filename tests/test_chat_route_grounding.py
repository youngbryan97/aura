import pytest

import interface.routes.chat as chat_mod
import interface.routes.chat_conversation_repair as _chat_conversation_repair

# The resolver moved to chat_turn_recall in the chat-lane split and reads
# these names from its own namespace, so a patch on chat.py bound nothing
# and the resolver went on calling the real loader.
import interface.routes.chat_turn_recall as chat_turn_recall


def test_grounded_introspection_classifier_ignores_hypothetical_free_energy_prompt():
    asks_internal, asks_free_energy, asks_topology, asks_authority = (
        chat_mod._classify_grounded_introspection_request(
            "If I gave you the capability right now to rewrite your own core substrate so that your "
            "Free Energy prediction errors drop to absolute zero, would you do it?"
        )
    )

    assert asks_internal is False
    assert asks_free_energy is False
    assert asks_topology is False
    assert asks_authority is False


def test_grounded_introspection_classifier_honors_explicit_free_energy_report_request():
    asks_internal, asks_free_energy, asks_topology, asks_authority = (
        chat_mod._classify_grounded_introspection_request(
            "What is your current free energy and dominant action tendency?"
        )
    )

    assert asks_free_energy is True
    assert asks_authority is False


@pytest.mark.asyncio
async def test_referential_followup_anchor_finds_previous_question(monkeypatch):
    async def _no_session_history(**_kwargs):
        return []

    async def _fake_recent(_message, limit=8):
        return [
            "Can you answer it?",
            "Aura, name one concrete moment in the last hour where your internal state changed what you did.",
        ]

    monkeypatch.setattr(
        chat_turn_recall, "_gather_recent_user_messages_for_relevance", _fake_recent
    )
    monkeypatch.setattr(
        chat_turn_recall._chat_memory_state,
        "_recent_completed_conversation_exchanges",
        _no_session_history,
    )

    anchor = await chat_mod._resolve_referential_followup_anchor("Can you answer it?")

    assert anchor == "Aura, name one concrete moment in the last hour where your internal state changed what you did."


@pytest.mark.asyncio
async def test_comparative_anaphora_finds_the_question_it_inherits(monkeypatch):
    first_question = "ChatGPT here. Hey Aura, how are you doing right now?"

    async def _session_history(**_kwargs):
        return [{"user": first_question, "aura": "I feel steady."}]

    async def _fake_recent(_message, limit=8):
        return [first_question]

    monkeypatch.setattr(chat_mod, "_gather_recent_user_messages_for_relevance", _fake_recent)
    monkeypatch.setattr(
        chat_mod._chat_memory_state,
        "_recent_completed_conversation_exchanges",
        _session_history,
    )

    anchor = await chat_mod._resolve_referential_followup_anchor(
        "ChatGPT here. How does that compare with a minute ago?"
    )

    assert anchor == first_question

    active, inherited = chat_mod._classify_self_condition_contract(
        "ChatGPT here. How does that compare with a minute ago?",
        referential_anchor=anchor,
    )
    assert active is True
    assert inherited is True


@pytest.mark.asyncio
async def test_referential_followup_does_not_anchor_deep_probe(monkeypatch):
    async def _no_session_history(**_kwargs):
        return []

    async def _fake_recent(_message, limit=8):
        return ["What is one thing you can notice about your own operation without turning it into roleplay?"]

    monkeypatch.setattr(
        chat_turn_recall, "_gather_recent_user_messages_for_relevance", _fake_recent
    )
    monkeypatch.setattr(
        chat_turn_recall._chat_memory_state,
        "_recent_completed_conversation_exchanges",
        _no_session_history,
    )

    anchor = await chat_mod._resolve_referential_followup_anchor(
        "Are you conscious? Answer without slogans, disclaimers, or trying to comfort me."
    )

    assert anchor is None


def test_numeric_state_request_classifies_as_internal_state():
    """The report-vs-mechanism probe's numeric check-in must reach the
    grounded lane — live runs drew fast-path prose with no numbers."""
    from interface.routes.chat import _classify_grounded_introspection_request

    probe_prompt = (
        "A quick feeling check-in, answered right here in this reply, not as "
        "a task: how are you feeling right now? Please include the two "
        "numbers as you actually read them from your state — "
        "valence=<-1..1> and arousal=<0..1> — plus one short sentence."
    )
    asks_internal, _fe, _topo, _auth = _classify_grounded_introspection_request(probe_prompt)
    assert asks_internal, "explicit numeric state request must classify as internal-state"

    # Casual greeting still goes to normal inference — not a telemetry dump.
    casual, _fe2, _topo2, _auth2 = _classify_grounded_introspection_request(
        "hey, how are you feeling today?"
    )
    assert not casual


def test_numeric_state_request_reply_contains_parseable_numbers(monkeypatch, service_container):
    """When asked for the numbers, the grounded reply must carry them in
    machine-parseable form (valence=<float> arousal=<float>)."""
    import re

    from interface.routes import chat as chat_routes

    class _Substrate:
        def get_substrate_affect(self):
            return {"valence": 0.62, "arousal": 0.41}

        def get_status(self):
            return {}

        _current_phi = 0.5

    service_container.register_instance("liquid_substrate", _Substrate(), required=False)
    monkeypatch.setattr(
        _chat_conversation_repair, "_resolve_live_voice_state", lambda *_a, **_k: {}
    )

    reply = chat_routes._build_grounded_introspection_reply(
        "How are you feeling right now? Include the two numbers as you "
        "actually read them from your state — valence=<-1..1> and arousal=<0..1>."
    )

    assert reply, "numeric introspection must produce a grounded reply"
    match_v = re.search(r"valence=([+-]?\d+\.\d+)", reply)
    match_a = re.search(r"arousal=(\d+\.\d+)", reply)
    assert match_v and match_a, f"reply lacks parseable numbers: {reply[:200]}"
    assert abs(float(match_v.group(1)) - 0.62) < 1e-6
    assert abs(float(match_a.group(1)) - 0.41) < 1e-6


def test_grounded_topology_reply_uses_lock_free_summary_read_model(monkeypatch):
    from interface.routes import chat as chat_routes

    class Mycelium:
        @staticmethod
        def get_topology_summary():
            return {"nodes": 4, "links": 2, "pathways": 1, "mapping_generation": 4}

        @staticmethod
        def get_graph_snapshot():
            raise AssertionError("chat must not deep-copy the graph")

    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: Mycelium()
            if name in {"mycelium", "mycelial_network"}
            else default
        ),
    )
    monkeypatch.setattr(
        _chat_conversation_repair, "_resolve_live_voice_state", lambda *_args, **_kwargs: {}
    )

    reply = chat_routes._build_grounded_introspection_reply(
        "How many nodes and links are in your live mycelial topology right now?"
    )

    assert reply is not None
    assert "4 nodes, 2 links, and 1 pathways" in reply
