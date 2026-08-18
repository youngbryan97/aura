"""The feed's promise is plain English on the card, engineering one click away.

The rules that keep that promise had no test, and one of them had never
matched anything. On 2026-08-10 the live feed was showing, verbatim:

    UNIFIED HEALTH PULSE
    System: CPU 0.0% | RAM 71.9% | Uptime: 5648s
    Runtime: HEALTHY | Required probes: PASS | ...

    Signal Routed: voice_engine -> sensory_gate | Payload: {'event':
    'threshold_shift', 'rms_gate': 0.01, 'conf_gate': -0.7}

    PhiCore is reporting a state_summary measurement because better-grounded
    estimators could not run: residual_stream_grassmann
    (insufficient_history:0/50 grassmann transitions), mesh (...)

A Python dict repr in the corner of someone's eye all day is the difference
between an instrument and a log tail.

These run the REAL rules out of ``interface/static/aura.js`` through node, so
the assertions are about shipped behaviour rather than a restatement of it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AURA_JS = ROOT / "interface/static/aura.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is needed to run the shipped rules"
)

#: The exact strings the live feed was showing, and what a person should read.
CASES = [
    (
        "═══ UNIFIED HEALTH PULSE ═══\n"
        "System: CPU 0.0% | RAM 71.9% | Uptime: 5648s\n"
        "Runtime: HEALTHY | Required probes: PASS",
        "Vitals steady — processor 0%, memory 72%, awake 1.6 hours.",
    ),
    (
        "DRIFT [managed_rss_mb]: rising 860.1/h — watching whether mitigation holds it",
        "Her memory footprint is rising at 860MB per hour — watching whether it settles.",
    ),
    (
        "Signal Routed: voice_engine -> sensory_gate | Payload: {'event': 'threshold_shift'}",
        "voice engine passed a signal to sensory gate.",
    ),
    (
        "WS: Client connected. Total: 3",
        "Another window connected to her (3 open).",
    ),
]


def _plain(messages: list[str]) -> list[str]:
    """Run the shipped PLAIN_LANGUAGE_RULES over each message."""
    script = """
    const fs = require('fs');
    const src = fs.readFileSync(process.argv[1], 'utf8');
    const start = src.indexOf('const PLAIN_LANGUAGE_RULES');
    const end = src.indexOf('function plainLanguageThought');
    const tailEnd = src.indexOf('\\n}', src.indexOf(
        'for (const [pattern, render] of PLAIN_LANGUAGE_RULES)')) + 2;
    const plain = new Function(
        src.slice(start, end) + '\\n' + src.slice(end, tailEnd) +
        '\\nreturn plainLanguageThought;')();
    const input = JSON.parse(process.argv[2]);
    console.log(JSON.stringify(input.map(plain)));
    """
    result = subprocess.run(
        ["node", "-e", script, str(AURA_JS), json.dumps(messages)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return json.loads(result.stdout)


def test_the_raw_lines_the_feed_was_showing_are_translated():
    raw = [case[0] for case in CASES]
    expected = [case[1] for case in CASES]
    assert _plain(raw) == expected


def test_the_health_pulse_rule_matches_the_shape_actually_emitted():
    """It required a literal " | " where the emitter writes a newline.

    core/ops/subsystem_audit.py builds the pulse as a heading line followed by
    metric lines. The rule read `UNIFIED HEALTH PULSE\\s*\\|\\s*System:` and `.`
    does not cross a newline, so it never fired once and the raw block was on
    screen the whole time the rule existed to replace it.
    """
    emitted = (
        "═══ UNIFIED HEALTH PULSE ═══\n"
        "System: CPU 12.5% | RAM 40.0% | Uptime: 120s\n"
        "Runtime: HEALTHY"
    )
    (rendered,) = _plain([emitted])
    assert "UNIFIED HEALTH PULSE" not in rendered
    assert rendered.startswith("Vitals steady")
    assert "13%" in rendered and "40%" in rendered


def test_no_translated_card_leaks_an_identifier_or_a_dict():
    """Whatever a rule returns has to be sayable out loud."""
    rendered = _plain([case[0] for case in CASES])
    for text in rendered:
        assert "_" not in text, f"identifier survived translation: {text}"
        assert "{" not in text and "}" not in text, f"dict repr survived: {text}"
        assert "|" not in text, f"log delimiter survived: {text}"


def test_a_rate_keeps_its_unit():
    """"rising at 860 per hour" is a number wearing a measurement."""
    (rendered,) = _plain(["DRIFT [managed_rss_mb]: rising 860.1/h — watching"])
    assert "860MB" in rendered


def test_a_line_nobody_wrote_a_rule_for_still_says_something():
    """The general path must not be the one guaranteed to be useless.

    Every card that reads well does so because a hand-written rule exists for
    its exact shape. The fallback that catches everything else returned
    "<subsystem> — internal measurements (SHOW ALL for the numbers)" — the same
    sentence whatever was happening, costing a line of attention and returning
    nothing.

    The keys are already language with the underscores removed and the values
    are the measurement, so no rule per subsystem is needed.
    """
    (rendered,) = _plain(
        ["Router: admission_state=deferred lane=background reason=headroom_reserved queue_depth=3"]
    )
    assert "admission state deferred" in rendered
    assert "internal measurements" not in rendered
    assert "_" not in rendered


def test_the_generic_path_needs_no_prior_knowledge():
    """A subsystem invented today reads the same as one from a year ago."""
    (rendered,) = _plain(["SomeNewSubsystem: widget_count=12 phase=settling drift=0.004"])
    assert "widget count 12" in rendered
    assert "phase settling" in rendered


def test_ordinary_prose_is_left_alone():
    (rendered,) = _plain(["A perfectly ordinary sentence about nothing in particular."])
    assert rendered == "A perfectly ordinary sentence about nothing in particular."
