"""The neural feed's face says what is happening in plain words; the line her code wrote is under the arrow.

Bryan, 16 November 2026: "neural feed messages should be plain english and easily readable. a lot of them are coming in
as code lines ... when you click the down arrow to expand, it can be the underlying technical explanation or technical
message/code. but the face should be readable by people who havent been consistently working with the system". Through
the card's own rule table, 30% of that afternoon's feed came out reworded; the legibility tool said 94.9%, measured
against a sample from 8 September.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

import pytest

from core.observability import plain_feed

ROOT = Path(__file__).resolve().parents[1]
ROWS = json.loads((ROOT / "config" / "plain_feed.json").read_text(encoding="utf-8"))["lines"]
AURA_JS = (ROOT / "interface" / "static" / "aura.js").read_text(encoding="utf-8")
sys.path.insert(0, str(ROOT / "tools"))


def _values_for(says: str) -> tuple:
    """A value of the right type for each conversion in a format string."""
    values = []
    for match in plain_feed._PERCENT.finditer(says):
        conversion = match.group()
        if conversion == "%%":
            continue
        values.append(7 if conversion[-1] in "dixXc" else 0.5 if conversion[-1] in "feEgG" else "drive_social")
    return tuple(values)


@pytest.mark.parametrize("row", ROWS, ids=lambda row: f"{row['logger']}:{row['says'][:40]}")
def test_each_sentence_reads_with_the_values_its_line_carries(row):
    if row.get("kind") == "f":
        line = re.sub(r"(?<!\{)\{\}(?!\})", "drive_social", row["says"]).replace("{{", "{").replace("}}", "}")
        face = plain_feed.plain_face(row["logger"], line, ())
    else:
        face = plain_feed.plain_face(row["logger"], row["says"], _values_for(row["says"]))
    assert face, f"{row['plain']!r} did not format with the values of {row['says']!r}"
    assert "{" not in face and "}" not in face
    assert "_" not in face, "an identifier on the face reads as words"


def test_every_sentence_is_for_a_line_still_in_her_code():
    import plain_feed as tool

    assert tool.stale_entries() == []


def test_a_call_that_gives_its_own_sentence_is_shown_by_it():
    record = logging.LogRecord("Aura.Core.Orchestrator", logging.INFO, __file__, 1, "%s", ("═══ PULSE ═══",), None)
    record.plain = "Health check: all her systems are running, and she is ready to talk."
    assert plain_feed.plain_face_of(record) == record.plain


def test_the_health_pulse_reads_as_a_sentence():
    from core.ops.subsystem_audit import SubsystemAudit

    well = ("═══ UNIFIED HEALTH PULSE ═══\nSystem: CPU 20.2% | RAM 59.5% | Uptime: 3533s\nRuntime: HEALTHY | "
            "Required probes: PASS | Subsystem audit: PASS | Conversation: PASS | Heartbeats: 11/11 ACTIVE")
    unwell = well.replace("Runtime: HEALTHY", "Runtime: DEGRADED").replace(
        "Uptime: 3533s\n", "Uptime: 3533s\n  warning mind_tick: STALE (40s)\n")
    assert SubsystemAudit.plain_pulse(well) == (
        "Health check: all her systems are running, and she is ready to talk. "
        "The computer is at 20% processor and 60% memory.")
    assert "Needs attention: mind tick." in SubsystemAudit.plain_pulse(unwell)


def test_her_server_puts_the_sentence_on_the_feed_event():
    pytest.importorskip("fastapi")
    import interface.server as server
    from interface.websocket_manager import log_queue

    row = next(row for row in ROWS if row["says"].startswith("GW IGNITION"))
    saved = list(log_queue)
    log_queue.clear()
    try:
        handler = server._QueueHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.emit(logging.LogRecord(row["logger"], logging.INFO, __file__, 1, row["says"],
                                       (58, "drive_social", 1.0, 0.2437), None))
        entry = log_queue[-1]
    finally:
        log_queue.clear()
        log_queue.extend(saved)
    assert entry["message"].startswith("GW IGNITION #58")
    assert entry["plain"] == "Something took the centre of her attention: drive social."


def test_the_card_shows_the_sentence_and_keeps_the_line_under_the_arrow():
    card = AURA_JS[AURA_JS.index("function addThoughtCard"):]
    card = card[:card.index("\n}\n")]
    assert "const plainFace = String(data.plain || '').trim();" in card
    assert "const msg = plainFace || toPlainEnglish(technical);" in card
    detail = card[card.index('<div class="thought-detail"'):card.index('<div class="thought-body thought-preview">')]
    assert 'class="thought-detail-raw"' in detail and "fullMsg" in detail, "the line itself is under the arrow"
    fingerprint = AURA_JS[AURA_JS.index("function buildThoughtFingerprint"):]
    assert "data.plain" in fingerprint[:fingerprint.index("\n}\n")], "repeats merge by what the card shows"


def test_the_share_of_the_feed_with_a_plain_face_does_not_fall():
    import plain_feed as tool

    started = time.monotonic()
    measured = tool.measure()
    baseline = json.loads(tool.BASELINE.read_text(encoding="utf-8"))["share"]
    assert measured["share"] + 1e-9 >= baseline, (measured["share"], baseline, measured["missing"][:5])
    assert time.monotonic() - started < 120
