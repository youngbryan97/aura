"""A revision must pass every check that forced the re-ask.

LIVE DEFECT, 2026-08-10. A reply was re-asked for two reasons at once: it
denied a capability the runtime measured as present, AND it quoted "about 18
seconds" as the retention time of her own memory buffer — Peterson and
Peterson's figure for human short-term memory, from no instrument here.

The acceptance test only re-read the capability claims. The revision stopped
denying, passed, and was served with the invented number still in it — twice
in the same reply. A partial re-check licenses exactly the part it does not
read.
"""
from __future__ import annotations

import re
from tests.chat_lane_support import chat_lane_source

SOURCE = chat_lane_source()


def test_the_acceptance_test_reads_every_check():
    body = re.search(
        r"def _still_contradicts_the_runtime.*?\n\n\n", SOURCE, re.S
    ).group(0)
    assert "contradicted_claims" in body
    assert "unsupported_self_specification" in body
    assert "fabricated_self_metrics" in body


def test_the_re_ask_uses_that_test_rather_than_one_check():
    reanswer = re.search(
        r"async def _reanswer_when_the_runtime_contradicts_her.*?\n\n\ndef ",
        SOURCE,
        re.S,
    ).group(0)
    assert re.search(
        r"_still_contradicts_the_runtime\(\s*revised_text",
        reanswer,
    )
    # The narrow test that let the fabrication through must not be the gate.
    assert "not ledger.contradicted_claims(revised_text)" not in reanswer


def test_a_revision_that_still_contradicts_ends_in_no_note():
    """With no capability claims, the fallback used to render an empty note; then a bracketed one in her voice. A
    revision that did not settle it now ends with what the measurements leave (Bryan, 14 November 2026)."""
    reanswer = re.search(
        r"async def _reanswer_when_the_runtime_contradicts_her.*?\n\n\ndef ",
        SOURCE,
        re.S,
    ).group(0)
    assert "_what_the_measurements_leave(" in reanswer
    assert "Correcting myself" not in reanswer
