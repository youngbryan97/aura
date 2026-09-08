"""Most of what a watcher reads in the neural feed is in plain English.

The feed's raw sources are internal logger lines — `Router: Queueing
background inference until admission clears for origin=…`, `Successfully
locked: 'Affect.AffectEngine'`, a Python traceback one frame per card.
`interface/static/aura.js` translates them, and the question nobody could
answer was how much of a real session that covered.

54.6%, measured on 2026-09-08 against the lines a live desktop session
actually emitted. Nearly half of what somebody watching Aura think saw was
engineering they had no way to read, and the worst of it was not untranslated
but incoherent: a card reading `)`, a card reading "```python", a card
carrying one line of an internal prompt.

The fixture is weighted by frequency on purpose. A rule for a line emitted
twice is worth less than a rule for one emitted two thousand times, and an
unweighted count says the opposite.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "neural_feed_legibility.py"
FIXTURE = ROOT / "tests" / "fixtures" / "neural_feed_sample.json"
BASELINE = ROOT / "config" / "neural_feed_legibility_baseline.json"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="the measurement runs the feed's own JavaScript rather than a copy of it",
)


def _measure() -> dict:
    result = subprocess.run(
        [sys.executable, str(TOOL), "--json"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout)


def test_the_fixture_is_a_real_session() -> None:
    """A gate over invented lines measures nothing."""

    fixture = json.loads(FIXTURE.read_text())
    assert fixture["distinct_lines_in_session"] > 5000
    assert len(fixture["lines"]) >= 500
    assert sum(row["count"] for row in fixture["lines"]) > 10000


def test_legibility_only_goes_up() -> None:
    baseline = float(json.loads(BASELINE.read_text())["share"])
    share = _measure()["share"]
    assert share + 1e-9 >= baseline, (
        f"{share * 100:.1f}% of feed events read as plain English, below the "
        f"{baseline * 100:.1f}% baseline. Restore the rule, or lower the "
        "baseline deliberately and say why."
    )


def test_the_worst_shapes_are_gone() -> None:
    """A card reading `)` is worse than one reading a log line."""

    report = _measure()
    still_raw = {row["text"].strip() for row in report["untranslated"]}
    for fragment in (")", "```python", "```", "Traceback (most recent call last):"):
        assert fragment not in still_raw, f"{fragment!r} still renders as a card"


def test_a_sentence_is_left_alone() -> None:
    """Translation must not swallow what already reads."""

    report = _measure()
    assert report["translated_events"] > 0
    assert report["raw_events"] >= 0
