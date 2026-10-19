"""Rows of her training buffer that no person said are moved out of it, whole, and the buffer copied first; a person's
row stays, whether its words are in her chat store or it was recorded as a conversation turn before that store began."""
from __future__ import annotations

import json
import sqlite3
import sys

import pytest

pytestmark = pytest.mark.unit


def _row(user: str, reply: str, source: str | None = None) -> str:
    row = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": user},
                        {"role": "assistant", "content": reply}]}
    if source:
        row["_meta"] = {"source": source}
    return json.dumps(row)


def test_only_what_a_person_said_stays(tmp_path, monkeypatch):
    from tools import quarantine_what_no_person_said as tool

    db = tmp_path / "conversations.db"
    with sqlite3.connect(db) as conn:
        conn.execute("create table turns (role text, content text)")
        conn.execute("insert into turns values ('user', 'What did you make of the essay?')")
    buffer = tmp_path / "experience_buffer.jsonl"
    rows = [
        _row("What did you make of the  essay?", "It argued from one case to every case, and I didn't buy the jump."),
        _row("Hey, Aura. It's fantastic to finally meet you.", "It's good to meet you too, Bryan.", source="conversation"),
        _row("[SWARM PROTOCOL: You are 'The Critic'. Analyze the proposal.]", "The retry path is unguarded."),
        _row("Ok. Once more. You with me?", "I'm here."),
    ]
    buffer.write_text("\n".join(rows) + "\n")
    monkeypatch.setattr(sys, "argv", ["q", "--buffer", str(buffer), "--conversations", str(db), "--apply"])

    assert tool.main() == 0

    kept = buffer.read_text().splitlines()
    assert kept == rows[:2]
    moved = next(tmp_path.glob("experience_buffer.no-person-said-*.jsonl")).read_text().splitlines()
    assert moved == rows[2:]
    assert next(tmp_path.glob("experience_buffer.jsonl.before-quarantine-*")).read_text().splitlines() == rows
