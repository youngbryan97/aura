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
