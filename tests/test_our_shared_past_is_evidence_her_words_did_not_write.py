"""What she and the person share is handed to the shared-past check as evidence, from sources her words did not write.

LIVE 7 November 2026: her answer was refused twice for "fabricated_shared_history" against the 93 characters of the
request alone, though she had watched with him before.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def test_what_they_watched_and_said_is_evidence_and_her_own_words_are_not(monkeypatch, tmp_path):
    import core.config as config_module
    import core.conversation.our_shared_past as past
    from core.conversation import persistence

    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "20261022-201227.jsonl").write_text(
        json.dumps({"kind": "before", "title": "I built a glove that Explodes whatever it Punches"}) + "\n")
    monkeypatch.setattr(config_module, "config", SimpleNamespace(paths=SimpleNamespace(data_dir=tmp_path)))
    now = time.time()
    history = [{"role": "user", "content": "Now let's do Allen Pan.", "created_at": now - 3600},
               {"role": "assistant", "content": "We went hiking in the Alps last summer.", "created_at": now - 3500},
               {"role": "user", "content": "too old to count", "created_at": now - 60 * 24 * 3600}]
    monkeypatch.setattr(persistence, "get_persistence",
                        lambda: SimpleNamespace(get_recent_history=lambda limit: history))
    monkeypatch.setattr(past, "_KEPT", {})
    shared = past.our_shared_past(now=now)
    text = " ".join(shared)
    assert "I built a glove that Explodes whatever it Punches" in text and "Allen Pan" in text
    assert "Alps" not in text and "too old" not in text  # her own words, and the old, are not evidence
    assert len(shared) <= past.ITEMS and all(len(item) <= past.ITEM_CHARS for item in shared)
