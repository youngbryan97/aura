"""What the person asked her to keep is recalled by what a question means, on her kernel's route too (Master Todo U07).

LIVE 19 November 2026, the outside bar's cold restart: she pinned "the locker code for the Riklumquo depot is 3455" when
told, and asked for it in a new conversation she searched the web and said the evidence did not mention it, nine times
in twelve. A pin is sealed and indexed under one placeholder, so no memory search found it, and the only recall was a
list of phrasings on the chat route. The owner's pins are now opened and weighed against the question by meaning, as
memories her retrieval ranks beside every other. Anyone else's stay with the session that pinned them.
"""
from __future__ import annotations

import json
import re

import pytest

from core.memory import what_she_was_asked_to_keep as kept
from core.memory.session_pin_cipher import SessionPinCipher
from core.memory.session_pin_ledger import SESSION_PIN_LEDGER_FILENAME

LOCKER = "the locker code for the Riklumquo depot is 3455"
GATE = "the gate code for the Telgordal depot is 9313"
CABINET = "the cabinet code for the Varpashfen depot is 7963"


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """Her pin ledger in a test folder, sealed with a test key, holding the owner's pins and someone else's."""
    cipher = SessionPinCipher(b"k" * 32)
    monkeypatch.setattr(SessionPinCipher, "from_system", classmethod(lambda cls: cipher))
    monkeypatch.setattr(kept, "where_her_pins_are", lambda: tmp_path / "memory" / SESSION_PIN_LEDGER_FILENAME)
    (tmp_path / "memory").mkdir()
    rows = [cipher.seal(content=content, source=f"Please remember this for later: {content}.", timestamp="t",
                        session_id=session, principal_id=who, principal_surface=surface)
            for content, session, who, surface in ((LOCKER, "s1", "owner:bryan", "owner"),
                                                   (GATE, "s2", "owner:bryan", "owner"),
                                                   (CABINET, "s3", "guest:someone", "guest"))]
    (tmp_path / "memory" / SESSION_PIN_LEDGER_FILENAME).write_text("".join(json.dumps(row) + "\n" for row in rows))
    return tmp_path


def _speaking(who: str, surface: str):
    return kept.THE_PERSON.set(who), kept.THEIR_SURFACE.set(surface)


def _words(texts):
    """A stand-in encoder: each text as counts of the words the test pins use."""
    vocabulary = ["locker", "gate", "cabinet", "riklumquo", "telgordal", "varpashfen", "depot", "code"]
    return [[float(re.findall(r"[a-z]+", text.lower()).count(word)) for word in vocabulary] for text in texts]


def test_the_owner_reads_her_own_pins_across_conversations_and_no_one_elses(ledger):
    tokens = _speaking("owner:bryan", "owner")
    try:
        assert [pin["content"] for pin in kept.pins_of_the_person_speaking()] == [LOCKER, GATE]
    finally:
        kept.THE_PERSON.reset(tokens[0])
        kept.THEIR_SURFACE.reset(tokens[1])


@pytest.mark.parametrize(("who", "surface"), [("guest:someone", "guest"), ("", "owner"), ("owner:bryan", "")])
def test_anyone_but_the_owner_gets_nothing_across_conversations(ledger, who, surface):
    tokens = _speaking(who, surface)
    try:
        assert kept.pins_of_the_person_speaking() == []
    finally:
        kept.THE_PERSON.reset(tokens[0])
        kept.THEIR_SURFACE.reset(tokens[1])


def test_the_question_is_closest_to_the_pin_it_asks_about(ledger, monkeypatch):
    monkeypatch.setattr(kept, "_by_meaning", _words)
    tokens = _speaking("owner:bryan", "owner")
    try:
        offered = kept.pins_asked_about("What is the locker code for the Riklumquo depot?")
    finally:
        kept.THE_PERSON.reset(tokens[0])
        kept.THEIR_SURFACE.reset(tokens[1])
    best = max(offered, key=lambda item: item["score"])
    assert best["content"].endswith(LOCKER) and best["metadata"]["type"] == "pinned_by_them"
    assert len(offered) == 2, "every pin of hers is offered; the retrieval phase ranks them with every other memory"


def test_without_her_encoder_no_pin_is_offered(ledger, monkeypatch):
    monkeypatch.setattr(kept, "_by_meaning", lambda texts: [])
    tokens = _speaking("owner:bryan", "owner")
    try:
        assert kept.pins_asked_about("What is the locker code for the Riklumquo depot?") == []
    finally:
        kept.THE_PERSON.reset(tokens[0])
        kept.THEIR_SURFACE.reset(tokens[1])


def test_the_chat_route_and_the_kernel_read_the_same_identity():
    from interface.routes import chat_common

    assert chat_common._CHAT_REQUEST_PRINCIPAL is kept.THE_PERSON
    assert chat_common._CHAT_REQUEST_SURFACE is kept.THEIR_SURFACE


@pytest.mark.model
def test_on_her_encoder_the_cold_restart_question_is_closest_to_its_own_pin(ledger):
    if not kept._by_meaning(["probe"]):
        pytest.skip("her encoder is not available here")
    tokens = _speaking("owner:bryan", "owner")
    try:
        offered = kept.pins_asked_about("What is the gate code for the Telgordal depot?")
    finally:
        kept.THE_PERSON.reset(tokens[0])
        kept.THEIR_SURFACE.reset(tokens[1])
    assert max(offered, key=lambda item: item["score"])["content"].endswith(GATE)
