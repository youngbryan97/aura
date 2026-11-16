"""A request reaches the capability it means by her embedding of it, measured; and what a capability serves teaches it.

The embedding here is a stand-in that counts shared words; what is tested is
the rest: a capability is taken only when it stands out from all of them and
clearly from the next, and a request served is a meaning kept.
"""
from __future__ import annotations

import re
import zlib
from types import SimpleNamespace

import numpy as np

from core.intent.capability_by_meaning import Meanings


def _embed(texts):
    rows = []
    for text in texts:
        v = np.zeros(1 << 16, dtype=np.float32)
        for word in re.findall(r"[a-z]+", text.lower()):
            v[zlib.crc32(word.encode()) % (1 << 16)] += 1.0
        rows.append(v)
    return np.stack(rows)


_SKILLS = {name: SimpleNamespace(description=said) for name, said in {
    "repair_a_program": "Fix a broken program somebody else wrote, then play it.",
    "build_app": "Build a web application to a person's specification.",
    "web_search": "Search the web for pages about a question.",
    "reminder": "Remind the person of something at a time.",
    "image_gen": "Draw a picture from a description.",
    "file_operation": "Read, write, list or delete a file on disk.",
}.items()}
# A catalogue the size of hers: how far one stands out is measured against all of them.
_SKILLS.update({f"other_{n}": SimpleNamespace(description=f"Handle {topic} for the person.") for n, topic in enumerate(
    "music weather calendar email contacts maps photos notes timers alarms clocks news stocks sports recipes translation "
    "dictionary calculator units podcasts radio camera scanner printer bluetooth wifi battery".split())})


def test_a_capability_that_stands_out_is_taken_and_none_otherwise(tmp_path):
    meanings = Meanings(_embed, tmp_path / "meanings.json")
    assert meanings.which("please fix my broken program and play it", _SKILLS) == "repair_a_program"
    assert meanings.which("how are you today", _SKILLS) is None


def test_a_request_served_is_a_meaning_kept(tmp_path):
    meanings = Meanings(_embed, tmp_path / "meanings.json")
    asked = "my pong thing misbehaves badly"
    assert meanings.which(asked, _SKILLS) is None
    meanings.learned(asked, "repair_a_program")
    again = Meanings(_embed, tmp_path / "meanings.json")  # kept on disk
    assert again.which("the pong thing misbehaves again", _SKILLS) == "repair_a_program"


def test_the_served_request_worker_owns_its_atomic_save(tmp_path, monkeypatch):
    import asyncio
    import json

    from core.brain import requests_by_meaning
    from core.governance_context import get_active_governance
    from core.runtime.file_write_gateway import get_file_write_gateway

    path = tmp_path / "new" / "meanings.json"
    meanings = Meanings(_embed, path)
    gateway = get_file_write_gateway()
    original = gateway.write_text
    writes = []
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")
    monkeypatch.setattr(requests_by_meaning, "_the_meanings", lambda: meanings)

    def save(destination, payload, **kwargs):
        token = get_active_governance()
        assert token is not None and token.domain == "state_mutation"
        writes.append((destination, kwargs["source"], kwargs["durable"], json.loads(payload)))
        original(destination, payload, **kwargs)

    monkeypatch.setattr(gateway, "write_text", save)
    asked = ("the pong thing misbehaves", "my screen tool crashed")
    capability = "repair_a_program"

    async def serve_both():
        await asyncio.gather(*(asyncio.to_thread(requests_by_meaning.served, text, capability) for text in asked))

    asyncio.run(serve_both())

    assert [write[:3] for write in writes] == [(path, "capability_by_meaning.learned", False)] * 2
    first, latest = (write[3][capability] for write in writes)
    assert len(first) == 1 and latest[:1] == first and set(latest) == set(asked)
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text()) == {capability: latest}
    assert Meanings(_embed, path).served == {capability: latest}
