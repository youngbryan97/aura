"""Her recall store is built from what people ask her: every question's best passage before any question's second."""
from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path
from types import SimpleNamespace

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "build_her_recall_store.py"
_spec = importlib.util.spec_from_file_location("build_her_recall_store", _TOOL)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def _passage(doc: int, title: str) -> SimpleNamespace:
    return SimpleNamespace(doc_id=doc, position=0, title=title, paragraph=f"{title} paragraph text")


def test_rounds_cover_every_question_before_any_twice():
    index = {"a": [_passage(1, "A1"), _passage(2, "A2")], "b": [_passage(3, "B1"), _passage(1, "A1"), _passage(4, "B3")]}
    taken = [p["title"] for p in tool.passages_in_rounds(["a", "b"], lambda q, limit: index[q][:limit])]
    assert taken == ["A1", "B1", "A2", "B3"]  # B's second is A1, already taken


def test_people_asked_is_newest_first_and_distinct(tmp_path):
    db = tmp_path / "c.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE turns (role TEXT, content TEXT, created_at REAL)")
        conn.executemany("INSERT INTO turns VALUES (?, ?, ?)", [
            ("user", "what is the capital of Peru", 1.0), ("assistant", "Lima.", 2.0),
            ("user", "how far is the moon from here", 3.0), ("user", "what is the capital of Peru", 4.0),
            ("user", "ok", 5.0)])
    assert tool.people_asked(db) == ["what is the capital of Peru", "how far is the moon from here"]
