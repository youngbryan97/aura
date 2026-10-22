"""A view she forms, lets go of or turns is on her record.

Bryan, 21 October 2026, listed a change of mind among the behaviour coordination
alone does not show, and her record had nothing to read one from: each liking
is kept as a running estimate with its own uncertainty and no history. Each
time a liking comes to sit more than two standard errors from indifference,
falls back within them, or comes to be held the other way, the record says so.
"""
from __future__ import annotations

import json

from core.conation.what_she_wants import WhatSheWants
from core.conation.engine import ConationEngine

KEY = "topic:bicycles"


def _lived(record: WhatSheWants, liking: float, turns: int) -> None:
    for _ in range(turns):
        record.met([KEY], liking, reachable=[KEY], labels={KEY: "bicycles"})


def test_a_view_forms_and_then_turns(tmp_path):
    record = WhatSheWants(tmp_path / "wants.json", engine=ConationEngine())
    _lived(record, 0.8, 12)
    _lived(record, -0.9, 120)
    minds = record._minds
    # Liked, let go, then disliked: the third is a change of mind.
    assert [(entry["from"], entry["to"], entry["turned"]) for entry in minds] == [
        (0, 1, False), (1, 0, False), (0, -1, True),
    ]


def test_the_record_keeps_what_changed_across_a_restart(tmp_path):
    from core.conation.what_she_wants import _held

    first = WhatSheWants(tmp_path / "first.json", engine=ConationEngine())
    _lived(first, 0.8, 12)
    saved = tmp_path / "again.json"
    saved.write_text(json.dumps(_held(first)), encoding="utf-8")
    again = WhatSheWants(saved, engine=ConationEngine())
    assert again._minds == first._minds and again._minds[0]["thing"] == KEY


def test_her_interjections_in_a_with_stay_on_her_record(tmp_path, monkeypatch):
    """A with is held in memory, so her remarks nobody asked for, and whether
    they were taken up, went with the process."""
    import importlib.util
    import sys
    from pathlib import Path

    from core.conation import what_she_wants
    from core.together.a_with import HER, THEM, Together

    record = WhatSheWants(tmp_path / "wants.json", engine=ConationEngine())
    monkeypatch.setattr(what_she_wants, "get_what_she_wants", lambda: record)
    together = Together()
    held = together.begin("ines", "watching a video", at=0.0)
    held.said(HER, 10.0, about=["topic:wrens"], until=12.0, unprompted=True)
    held.said(THEM, 13.0, about=["topic:wrens"], until=15.0)
    held.said(HER, 40.0, about=["topic:boots"], until=42.0, unprompted=True)
    held.said(HER, 60.0, about=["topic:rain"], until=61.0, unprompted=True)
    together.end("ines", by=THEM, at=90.0)
    assert record._withs[-1]["remarks"] == 3 and record._withs[-1]["taken_up"] == 1

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("own_under_test", repo / "tools" / "what_she_does_of_her_own.py")
    tool = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tool
    spec.loader.exec_module(tool)
    from core.conation.what_she_wants import _held

    report = tool.report(_held(record))
    assert report["interjections"] == 3 and report["interjections_taken_up"] == 1
