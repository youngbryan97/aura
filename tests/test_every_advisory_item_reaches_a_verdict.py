"""Every external review item has a verdict, and "read" is not one.

Four advisory corpora reached this repository — 414 gap-atlas cards, the
developmental-agency council, the interiority council, the D-series review of
the gauntlet. "All advisory items were read" is not checkable and does not say
what happened to any of them.

So the items are extracted structurally and each one carries adopted,
superseded, rejected or open. Adopted names what holds it. Open names the
queue item that owns it. Rejected carries a reason. A corpus cannot be counted
as incorporated by a sentence saying it was.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report():
    import sys

    sys.path.insert(0, str(ROOT))
    from tools.reqproof.advisory import build

    return build(ROOT)


def test_every_item_has_a_verdict(report) -> None:
    undecided = report["summary"]["undecided"]
    assert not undecided, (
        f"{len(undecided)} advisory items with no disposition: {undecided[:12]}"
    )


def test_the_corpora_are_still_being_read(report) -> None:
    """A gate over an empty extraction reports green forever."""

    assert report["summary"]["corpora"] == 4
    assert report["summary"]["items"] >= 600
    for corpus, counts in report["summary"]["per_corpus"].items():
        assert counts["items"] > 0, f"{corpus} extracted nothing"


def test_nothing_is_adopted_on_nobody_word(report) -> None:
    """Adopted means something said so — a reviewed file, or the corpus."""

    assert not report["summary"]["adopted_without_evidence"]


def test_a_rejection_carries_a_reason(report) -> None:
    unreasoned = [
        f"{item['corpus']}:{item['id']}"
        for item in report["items"]
        if item["disposition"] == "rejected"
        and item["source"] == "config/advisory_dispositions.json"
        and not item.get("reason")
    ]
    assert not unreasoned, f"rejections with no reason: {unreasoned}"


def test_an_open_item_names_who_owns_it(report) -> None:
    """An open item with no owner is a dropped item with a label."""

    orphans = [
        f"{item['corpus']}:{item['id']}"
        for item in report["items"]
        if item["disposition"] == "open"
        and item["source"] == "config/advisory_dispositions.json"
        and "Owned by" not in str(item.get("reason") or "")
    ]
    assert not orphans, f"open items naming no owner: {orphans}"
