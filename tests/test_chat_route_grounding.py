import pytest

import interface.routes.chat as chat_mod


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
    async def _fake_recent(_message, limit=8):
        return [
            "Can you answer it?",
            "Aura, name one concrete moment in the last hour where your internal state changed what you did.",
        ]

    monkeypatch.setattr(chat_mod, "_gather_recent_user_messages_for_relevance", _fake_recent)

    anchor = await chat_mod._resolve_referential_followup_anchor("Can you answer it?")

    assert anchor == "Aura, name one concrete moment in the last hour where your internal state changed what you did."


@pytest.mark.asyncio
async def test_referential_followup_does_not_anchor_deep_probe(monkeypatch):
    async def _fake_recent(_message, limit=8):
        return ["What is one thing you can notice about your own operation without turning it into roleplay?"]

    monkeypatch.setattr(chat_mod, "_gather_recent_user_messages_for_relevance", _fake_recent)

    anchor = await chat_mod._resolve_referential_followup_anchor(
        "Are you conscious? Answer without slogans, disclaimers, or trying to comfort me."
    )

    assert anchor is None
