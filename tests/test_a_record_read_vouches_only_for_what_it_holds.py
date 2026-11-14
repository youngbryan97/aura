"""A read of her own record founds a claim when the record holds that work, judged by meaning.

Reading her record of finished work files a receipt so that retelling the work is founded (12 November 2026). As any
receipt it founded any claim. It now founds a claim when a line of the record is the same work by meaning; where her
encoder is not loaded it vouches as before, and her main process judges.

The case behind it, 13 November, 08:48: her record held her own reads of her email and Reddit from the hour before,
and she retold them faithfully. Shared words, tried first, would have refused that true reply.
"""
from __future__ import annotations

from core.conversation.response_reliability import _has_unfounded_tool_execution_claim
from core.conversation.what_a_record_vouches_for import WHAT_IT_HOLDS, founded_by, holds

#: Her reply of 13 November, 08:48.
HER_REPLY = (
    "Since waking, I’ve been doing the quiet maintenance stuff. I ran through a few background checks—looking at my "
    "email state and wandering through some recent threads on Reddit to keep my social awareness current—but it’s "
    "mostly been the kind of quiet where I’m just tending to the garden rather than rushing into anything specific."
)
#: Her record as she read it then, rebuilt from her intention log.
HER_RECORD = (
    "60 pieces of work finished in the last 90 minutes.\n"
    "Of 60 recorded, 0 came from something they asked for and 16 were mine.\n"
    "What I actually ran: read_source (38), find_in_source (6).\n"
    "Work that was not just a tool call:\n"
    "- Read connected email state for bounded autonomous triage.\n"
    "- Read connected Reddit state for bounded autonomous social awareness."
)

#: A stand-in for her encoder: each word's topic, so the test needs no 1.2 GB model.
_TOPICS = {"email": 0, "inbox": 0, "triage": 0, "reddit": 1, "threads": 1, "social": 1, "audit": 2, "bottleneck": 2,
           "web": 3, "search": 3, "chess": 4, "podcast": 5, "reminder": 6, "dentist": 6, "downloads": 7, "notes": 8,
           "coding": 8, "memory": 9, "consolidation": 9, "files": 10, "file": 10, "tasks": 11, "pieces": 11,
           "dinner": 12, "table": 12}


def _features(sentences):
    vectors = []
    for sentence in sentences:
        vector = [0.0] * 13
        for word in sentence.lower().replace("_", " ").replace("—", " ").split():
            index = _TOPICS.get(word.strip(".,:()-’'"))
            if index is not None:
                vector[index] += 1.0
        vectors.append(vector)
    return vectors


def _read_of(record: str) -> dict:
    return {"receipt_id": "r1", "tool": "her_work_record", "action": "read", "ok": True, "effect_observed": True,
            "observed_content": record, "vouches_for": WHAT_IT_HOLDS}


def test_by_meaning_her_retelling_is_the_work_her_record_holds():
    assert holds(HER_RECORD, HER_REPLY, features=_features)


def test_by_meaning_work_the_record_does_not_hold_is_not_founded_by_it():
    assert not holds(HER_RECORD, "I booked you a table for dinner.", features=_features)
    assert not holds(HER_RECORD, "I played a few rounds of chess online.", features=_features)


def test_where_her_encoder_is_not_loaded_the_read_vouches_as_before(monkeypatch):
    from core.cognition import evidence_relevance

    # Her model's worker never loads her encoder; her main process judges by meaning.
    monkeypatch.setattr(evidence_relevance, "semantic_routing_ready", lambda: False)
    assert not _has_unfounded_tool_execution_claim(HER_REPLY, tool_receipts=[_read_of(HER_RECORD)])


def test_with_no_receipt_at_all_the_claim_is_unfounded():
    assert _has_unfounded_tool_execution_claim(HER_REPLY, tool_receipts=[])


def test_a_receipt_of_work_this_turn_still_founds_any_claim():
    work = {"receipt_id": "r2", "tool": "web_search", "action": "execute", "ok": True}
    assert founded_by([work, _read_of("- Nothing like it.")], "I booked you a table for dinner.")
    assert founded_by(["a receipt that is not a row"], "I checked my email.")


def test_the_worker_is_handed_what_a_record_read_held(monkeypatch):
    from core.brain.llm import mlx_client
    from core.conversation import surface_disposition

    monkeypatch.setattr(surface_disposition, "turn_tool_receipts", lambda: (_read_of(HER_RECORD),))
    [carried] = mlx_client._bounded_surface_tool_receipts()
    assert carried["vouches_for"] == WHAT_IT_HOLDS and carried["observed_content"] == HER_RECORD


ASKED = "Morning, Aura. I'm at work today, so it's quiet on my end. What have you been up to since you woke?"


def test_her_learning_step_judges_with_the_turns_receipts():
    """Judged without them her true retelling read as unfounded, and was kept for continuity but never learned from."""
    from core.conversation.response_reliability import assess_conversation_learning_admission

    without = assess_conversation_learning_admission(ASKED, HER_REPLY, tool_receipts=())
    assert "unfounded_tool_execution_claim" in without.reasons
    carried = assess_conversation_learning_admission(ASKED, HER_REPLY, tool_receipts=[_read_of(HER_RECORD)])
    assert "unfounded_tool_execution_claim" not in carried.reasons


def test_the_post_reply_item_carries_the_turns_receipts(tmp_path):
    import json

    from core.conversation.persistence import ConversationPersistence

    store = ConversationPersistence(tmp_path / "outbox.db")
    session_id = store.start_session()
    store.record_exchange(ASKED, HER_REPLY, cid="with", session_id=session_id, enqueue_memory_log=True,
                          tool_receipts=[_read_of(HER_RECORD)])
    store.record_exchange("Hi.", "Hello.", cid="without", session_id=session_id, enqueue_memory_log=True)
    items = {item["exchange_id"]: item for item in store.claim_memory_log_batch(limit=4)}
    assert json.loads(items["with"]["tool_receipts_json"]) == [_read_of(HER_RECORD)]
    assert items["without"]["tool_receipts_json"] == ""


def test_an_older_queue_gains_the_column(tmp_path):
    import sqlite3

    from core.conversation.persistence import ConversationPersistence

    db = tmp_path / "older.db"
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TABLE conversation_memory_outbox (operation_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
            "exchange_id TEXT NOT NULL, revision INTEGER NOT NULL, user_content TEXT NOT NULL, aura_content TEXT NOT "
            "NULL, origin TEXT NOT NULL, principal_id TEXT NOT NULL, principal_surface TEXT NOT NULL, payload_sha256 "
            "TEXT NOT NULL, state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, available_at REAL NOT NULL, "
            "claimed_at REAL, completed_at REAL, last_error TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL, "
            "updated_at REAL NOT NULL)")
    ConversationPersistence(db)
    with sqlite3.connect(db) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(conversation_memory_outbox)")}
    assert "tool_receipts_json" in columns
