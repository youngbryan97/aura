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
