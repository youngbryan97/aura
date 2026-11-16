"""A core cannot be declared causally closed by definition.

K is ten domains of declared columns, and the declaration is a claim: that
everything persistent which changes what she does next is either read by one of
those columns, or is outside the core and findable by the closure test. This
checks the claim against the state objects themselves rather than against a
list of suspects, because a list of suspects is a list of the ones already
thought of.

It found one. `affect.mood_baselines` is what an arriving feeling is priced
against — the workspace bids a feeling at `intensity - baseline`, so the same
intensity claims more or less of her attention depending on it. Persistent
state deciding what wins the competition, and no column read it. A carries it
now.

The unmapped list is not a list of defects. Most of what a state carries is
bookkeeping and most of the rest is a readout. It is the list of fields that
could change the core's future with no line of the battery watching, and it
only shrinks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "config" / "state_outside_the_core_baseline.json"


@pytest.fixture(scope="module")
def audit():
    from tools.audit_state_outside_the_core import findings

    return findings()


def test_no_new_state_field_is_unmapped(audit) -> None:
    allowed = set(json.loads(BASELINE.read_text()).get("allowed", []))
    unmapped = {row["path"] for row in audit["unmapped"]}
    new = sorted(unmapped - allowed)
    assert not new, (
        "state fields nothing in K reads and nothing has explained: "
        + ", ".join(new)
    )


def test_the_baseline_only_shrinks(audit) -> None:
    allowed = set(json.loads(BASELINE.read_text()).get("allowed", []))
    unmapped = {row["path"] for row in audit["unmapped"]}
    gone = sorted(allowed - unmapped)
    assert not gone, (
        "these are resolved; rerun with --write-baseline to lower the bar: "
        + ", ".join(gone)
    )


def test_every_field_outside_the_core_says_why(audit) -> None:
    """A field declared outside K without a reason is a field nobody checked."""
    for row in audit["periphery"]:
        assert row["why"], f"{row['path']} is outside K with no reason given"


def test_the_core_reads_something_from_every_state_object(audit) -> None:
    roots = {row["path"].split(".")[0] for row in audit["in K"] if "." in row["path"]}
    from tools.audit_state_outside_the_core import ROOTS

    missing = sorted(set(ROOTS) - roots)
    assert not missing, f"no domain reads anything from: {missing}"


def test_the_standing_mood_is_in_the_core() -> None:
    """The finding this audit was written to catch.

    The workspace prices an arriving feeling against what she has come to
    expect of it, so this decides what wins attention.
    """
    from core.subject.state import schema

    sources = set(schema("A").sources)
    assert "affect.mood_baselines" in sources


def test_the_thing_that_prices_a_feeling_is_what_the_workspace_reads() -> None:
    """And the column is not a guess about which field that is."""
    feed = (REPO / "core" / "consciousness" / "workspace_feed.py").read_text(encoding="utf-8")
    assert 'baselines = getattr(affect, "mood_baselines", {}) or {}' in feed
    assert "charge = _clamp(intensity - _clamp(baselines.get(name, 0.0)))" in feed


def test_her_record_of_how_she_answers_is_read_by_the_self_state() -> None:
    """Her reply notes are read back to her when she is asked how she would answer, so they change what she says.

    Found by the audit above on 15 November 2026, read by nothing in K. S reads them now, and S's clamp holds them,
    so a self held still by a lesion does not go on learning how it answers.
    """
    from core.state.aura_state import AuraState
    from core.subject.clamp import CLAMPED_FIELDS
    from core.subject.state import Organs, schema
    from core.subject.state_readers import _read_S

    assert "cognition.reply_notes" in schema("S").sources
    assert "cognition.reply_notes" in CLAMPED_FIELDS["S"]
    state = AuraState()
    before = _read_S(state, Organs())
    state.cognition.reply_notes = [{"from_the_question": True, "uses_i": False}] * 4
    after = _read_S(state, Organs())
    features = schema("S").features
    moved = {features[i] for i in range(len(features)) if before[i] != after[i]}
    assert moved == {"answers_noted", "answers_open_from_the_question"}


def test_what_recall_weighed_is_read_by_active_memory() -> None:
    """The pool the affect phase settles recall on. M's clamp held it while no column of M read it."""
    from core.state.aura_state import AuraState
    from core.subject.state import schema
    from core.subject.state_readers import _read_M

    sources = schema("M").sources
    assert "cognition.recall_pool.candidates" in sources
    assert "cognition.recall_pool.salience" in sources
    state = AuraState()
    before = _read_M(state)
    state.cognition.recall_pool = {"salience": 0.4, "candidates": [{"text": "a"}, {"text": "b"}, {"text": "c"}]}
    after = _read_M(state)
    features = schema("M").features
    moved = {features[i] for i in range(len(features)) if before[i] != after[i]}
    assert moved == {"recall_weighed", "recall_salience"}
