"""A lane that gives no answer forms no opinion and raises no incident.

Her background lane, queued behind admission, answered None, the opinion
engine read it as JSON, and the fail-closed subsystem turned the TypeError into
a critical incident on her desktop (23 October, 15:42). No answer is
backpressure; an answer that will not parse is still a fault.
"""

from __future__ import annotations

import asyncio

import core.epistemics.opinion_engine as opinions


class _Silent:
    async def generate(self, *args, **kwargs):
        return None


def test_no_answer_is_backpressure_not_a_fault(monkeypatch, tmp_path):
    recorded = []
    monkeypatch.setattr(opinions, "record_degradation", lambda *a, **k: recorded.append(a))
    engine = opinions.OpinionEngine(db_path=tmp_path / "opinions.json")
    monkeypatch.setattr(engine, "_get_brain", lambda: _Silent())
    assert asyncio.run(engine.form_opinion("tides")) is None
    assert recorded == []
