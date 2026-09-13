"""The typing label says what was measured, not what the message says.

The delivery journal publishes a phase, a message and details — tokens read
against the prompt's length, tokens written so far, when it was observed —
and the label showed the message alone. "Working through the response." for
ninety seconds claims progress nothing behind it supports. These run the
shipped function through node, so the label cannot drift from what renders.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

AURA_JS = Path(__file__).resolve().parents[1] / "interface" / "static" / "aura.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is needed to run the shipped label"
)


def _label(progress: dict, now: float) -> str:
    script = """
    const fs = require('fs');
    const src = fs.readFileSync(process.argv[1], 'utf8');
    const start = src.indexOf('const PROGRESS_QUIET_AFTER_S');
    const end = src.indexOf('\\n}\\n', src.indexOf('function truthfulProgressLabel')) + 3;
    const label = new Function(src.slice(start, end) + '\\nreturn truthfulProgressLabel;')();
    const [progress, now] = JSON.parse(process.argv[2]);
    console.log(JSON.stringify(label(progress, now)));
    """
    result = subprocess.run(
        ["node", "-e", script, str(AURA_JS), json.dumps([progress, now])],
        capture_output=True, text=True, timeout=60, check=True,
    )
    return json.loads(result.stdout)


def test_reading_the_prompt_shows_how_much_of_it_is_read() -> None:
    label = _label(
        {"phase": "prefill", "message": "Reading the conversation context.",
         "details": {"completed_tokens": 1200, "total_tokens": 2967}, "observed_at": 100},
        101,
    )
    assert label == "Reading the conversation — 1,200 of 2,967 tokens"


def test_writing_shows_a_count_and_no_denominator() -> None:
    """The answer has a ceiling, not a length; "146 of 1,143" would be a claim."""
    label = _label(
        {"phase": "generating", "message": "Working through the response.",
         "details": {"completed_tokens": 146, "total_tokens": 1143}, "observed_at": 100},
        101,
    )
    assert label == "Writing — 146 tokens so far"
    assert "1,143" not in label


def test_a_reading_that_went_quiet_says_for_how_long() -> None:
    label = _label(
        {"phase": "generating", "message": "Working through the response.",
         "details": {"completed_tokens": 146}, "observed_at": 100},
        131,
    )
    assert label.endswith("(nothing heard for 31s)")


def test_a_message_with_no_numbers_is_shown_as_it_is() -> None:
    assert _label({"phase": "plan", "message": "Building a grounded execution plan."}, 0) == (
        "Building a grounded execution plan."
    )
