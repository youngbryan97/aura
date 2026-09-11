"""How far she is standing behind an answer has to reach the person.

``interface/routes/chat.py`` sets ``response_confidence`` on 71 exits across a
vocabulary of ten values — degraded, bounded, guarded, failed, failed_closed,
principled_refusal, not_generated, … — and ``interface/static/aura.js``
referenced it exactly once, to WRITE it into a synthetic local payload.
Nothing ever read it.

So a turn the pipeline itself had already classified reached the person as an
ordinary message with no mark on it. Measured live 2026-08-10: a reply that
failed its own reliability gate (fabricated_shared_history,
missing_requested_objective_facets), exhausted bounded correction, and was
served as a salvaged draft with ``response_confidence="degraded"`` looked
exactly like a good answer.

Serving the draft rather than an apology is right — discarding genuine work
helps nobody. Serving it SILENTLY is the defect.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AURA_JS = ROOT / "interface/static/aura.js"
AURA_CSS = ROOT / "interface/static/aura.css"
CHAT_ROUTE = ROOT / "interface/routes/chat.py"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is needed to run the shipped badge logic"
)

#: Values that mean the answer is fully backed and need no mark.
UNMARKED = {"high", "scoped"}


def _server_vocabulary() -> set[str]:
    """Every response_confidence value the chat route actually emits."""
    source = CHAT_ROUTE.read_text(encoding="utf-8")
    return set(re.findall(r'"response_confidence":\s*"([a-z_]+)"', source))


def _badge_for(values: list[str]) -> list[str]:
    """Run the shipped badge logic and return the visible label, or ''."""
    script = """
    const fs = require('fs');
    const src = fs.readFileSync(process.argv[1], 'utf8');
    const start = src.indexOf('const REPLY_CONFIDENCE_BADGES');
    const end = src.indexOf('function messageBadgeHtml');
    const badge = new Function('escHtml',
        src.slice(start, end) + '\\nreturn replyConfidenceBadgeHtml;')(s => String(s));
    const out = JSON.parse(process.argv[2]).map(v => {
        const html = badge(v);
        const m = html.match(/>([^<]*)</);
        return m ? m[1] : '';
    });
    console.log(JSON.stringify(out));
    """
    result = subprocess.run(
        ["node", "-e", script, str(AURA_JS), json.dumps(values)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return json.loads(result.stdout)


def test_every_value_the_server_emits_is_accounted_for():
    """No value the route can send may fall through unnoticed."""
    vocabulary = sorted(_server_vocabulary())
    assert vocabulary, "the route should still be setting response_confidence"

    labels = _badge_for(vocabulary)
    for value, label in zip(vocabulary, labels):
        if value in UNMARKED:
            assert label == "", f"{value} should read as an ordinary answer"
        else:
            assert label, f"{value} reaches the person with no mark on it"


def test_a_degraded_reply_is_marked_rather_than_passing_as_a_good_one():
    """The exact live case: a salvaged draft served after its gate failed."""
    (label,) = _badge_for(["degraded"])
    assert label == "Unverified"


def test_an_unmapped_value_still_discloses():
    """Silently ignoring the field is what made this channel invisible.

    A value nobody has mapped yet is precisely the one worth seeing, so the
    fallback must disclose rather than drop.
    """
    (label,) = _badge_for(["some_confidence_nobody_mapped_yet"])
    assert label, "an unknown confidence value was dropped"


def test_absence_is_not_a_verdict():
    """A reply carrying no confidence at all must not be accused of anything."""
    assert _badge_for([""]) == [""]


def test_the_route_actually_carries_the_field_to_the_client():
    source = AURA_JS.read_text(encoding="utf-8")
    assert "metadata.responseConfidence = data.response_confidence" in source
    assert "replyConfidenceBadgeHtml(metadata.responseConfidence)" in source


def test_every_delivery_path_can_carry_the_mark():
    """A reply reaches the transcript three ways. All three must mark it.

    Wiring only ``appendMsg`` left every STREAMED reply unmarked — which is
    most of them, including the honest refusal "I couldn't get to an answer
    I'd stand behind on that one", the turn where the mark is most deserved.
    Fixing the confidence channel in one path and not the others reproduces
    exactly the half-wiring the channel was fixed for.
    """
    source = AURA_JS.read_text(encoding="utf-8")

    # 1. Non-streamed HTTP response.
    assert "metadata.responseConfidence = data.response_confidence" in source

    # 2. Socket stream that ends with the confidence attached.
    assert "finishStreamMsg(data.response_confidence)" in source
    assert re.search(r"function finishStreamMsg\(confidence\)", source)
    assert re.search(
        r"function finishStreamMsg\(confidence\)\s*\{\s*(?://[^\n]*\n\s*)*"
        r"markReplyConfidence\(activeStreamDiv, confidence\)",
        source,
    ), "finishStreamMsg does not apply the mark"

    # 3. Text already streamed, confidence arriving late on the HTTP response.
    delivery = source.split("function renderChatDeliveryAnswer", 1)[1].split("\nfunction handleWsEvent", 1)[0]
    assert "discardChatDraft(item)" in delivery
    assert "markReplyConfidence(existing, data.response_confidence)" in delivery
    assert "metadata.responseConfidence = data.response_confidence" in delivery


def test_the_mark_is_applied_once():
    """Two paths can race for the same message; it must not be double-badged."""
    source = AURA_JS.read_text(encoding="utf-8")
    body = source.split("function markReplyConfidence", 1)[1].split("\n}", 1)[0]
    assert "querySelector('.aura-badge')" in body


def test_the_mark_reads_as_a_caveat_not_an_error():
    """It marks a reply she MEANT. A solid warning block would say otherwise."""
    css = AURA_CSS.read_text(encoding="utf-8")
    block = css.split(".aura-badge.unverified", 1)[1].split("}", 1)[0]
    assert "background: transparent" in block
    assert "border:" in block
