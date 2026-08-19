"""A fact read off a record does not inherit the draft's confidence.

LIVE, 2026-08-19. The verbatim conversation history — five turns with their
times, read off disk — arrived badged "Partial". The exact product 50,420,273
arrived badged "No answer". Both were composed from records; the badge came
from the generated draft each one replaced.

Whatever the model's attempt scored says nothing about a fact that was looked
up, and to anyone deciding how much to trust a reply, an accurate record
labelled unreliable is worse than no label.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from interface.routes.chat import (
    _SERVED_FROM_RECORD_OPENINGS,
    _reply_was_served_from_a_record,
)


def test_a_generated_reply_is_not_mistaken_for_a_record():
    for draft in (
        "You asked about my cognitive architecture.",
        "I held the position that affect was a side effect of cognition.",
        "Sure — here's how a hash map works.",
        "",
    ):
        assert not _reply_was_served_from_a_record(draft), draft


def test_every_composer_output_is_recognised():
    """The list and the composers must not drift apart.

    Recognising the openings by text is a coupling: change a composer's first
    line and the badge silently reverts to the draft's. So the real outputs
    are generated here and checked against it.
    """
    import sqlite3

    from core.self.belief_history import describe_belief_changes

    now = time.time()
    machine_state = SimpleNamespace(
        snapshots={
            "a": SimpleNamespace(ts=now - 86400 * 4, beliefs={"e": {"n": 1}}, revision_note=None),
            "b": SimpleNamespace(ts=now - 86400 * 2, beliefs={"e": {"n": 2}}, revision_note=None),
        }
    )
    assert _reply_was_served_from_a_record(describe_belief_changes(machine_state))

    # And the conversation composer, against a store built here.
    from core.conversation import durable_turns as module

    path = Path(__file__).resolve().parent / "_served_record_probe.db"
    try:
        connection = sqlite3.connect(path)
        connection.execute(
            "CREATE TABLE episodes (episode_id TEXT, timestamp REAL, "
            "context TEXT, action TEXT, outcome TEXT)"
        )
        connection.execute(
            "INSERT INTO episodes VALUES (?, ?, ?, ?, ?)",
            ("1", now - 7200, "User asked: run a tiny bit of python", "a", "b"),
        )
        connection.commit()
        connection.close()
        original = module._episodic_path
        module._episodic_path = lambda: path
        try:
            composed = module.earlier_conversation_answer()
        finally:
            module._episodic_path = original
    finally:
        path.unlink(missing_ok=True)
    assert composed
    assert _reply_was_served_from_a_record(composed)


def test_the_openings_are_declared_not_guessed():
    assert _SERVED_FROM_RECORD_OPENINGS
    for opening in _SERVED_FROM_RECORD_OPENINGS:
        assert opening.strip() == opening
        assert _reply_was_served_from_a_record(opening + " and then some text")
