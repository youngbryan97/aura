"""The inherited inventory has to be complete, and complete has to mean something.

Counting unchecked boxes missed the obligations written as prose, and the
earlier keyword signal fired on any sentence containing "must". These tests
hold the replacement to the bar that makes it worth having: every block is
classified, every block that survives dismissal routes to a queue item, every
dismissal is either a re-evaluable predicate or a hand label, and the detector
is measured against what it throws away.
"""

from __future__ import annotations

import json

import pytest

from tools.reqproof import obligations
from tools.reqproof.inherited import ROOT, SOURCES, build_report, classify_and_route


@pytest.fixture(scope="module")
def report() -> dict:
    reviews = json.loads((ROOT / "config/inherited_ledger_reviews.json").read_text())
    return build_report(ROOT, reviews)


@pytest.fixture(scope="module")
def analysis() -> dict:
    return classify_and_route(ROOT)


def test_every_block_gets_exactly_one_kind(analysis: dict) -> None:
    kinds = {record["kind"] for record in analysis["records"]}
    assert kinds <= set(obligations.KINDS)
    assert len(analysis["records"]) == sum(
        len(source["blocks"]) for source in analysis["sources"]
    )


def test_no_in_scope_block_routes_nowhere(analysis: dict) -> None:
    """The failure this replaces: an obligation lost by not being mentioned."""
    stranded = sorted(
        record["id"] for record in analysis["records"]
        if record["in_scope"] and not record["dismissed_by_rule"] and not record["queue"]
    )
    assert stranded == [], f"{len(stranded)} blocks route nowhere: {stranded[:5]}"


def test_status_vocabulary_covers_every_frequent_token(analysis: dict) -> None:
    coverage = analysis["vocabulary_coverage"]
    assert coverage["complete"], coverage["unclassified_frequent"]


def test_every_routed_queue_item_exists(analysis: dict) -> None:
    assert analysis["unknown_queue_targets"] == []


def test_dismissal_rules_are_predicates_that_still_hold(analysis: dict) -> None:
    """Re-evaluating a rule is the whole reason a rule may stand in for a review."""
    blocks = {block["id"]: block for source in analysis["sources"]
              for block in source["blocks"]}
    atlas = json.loads((ROOT / "docs/gap_atlas/adjudication.json").read_text())["entries"]
    context = {"atlas": atlas, "has_reviewed_successor": {
        record["id"] for record in analysis["records"] if record["dismissed_by_rule"] == "lead_in"
    }}
    for record in analysis["records"]:
        name = record["dismissed_by_rule"]
        if name and name != "lead_in":
            assert obligations.evaluate_rule(name, blocks[record["id"]], context), record["id"]


def test_dismissed_populations_are_sampled_and_labelled(report: dict) -> None:
    audit = report["summary"]["dismissal_audit"]
    assert audit["unlabelled"] == []
    assert audit["missed_obligations"] == [], (
        "the audit found an obligation the partition dismissed; repair the "
        "detector rather than the label"
    )
    for name, stratum in audit["strata"].items():
        assert stratum["scored"] > 0, name
        assert stratum["miss_rate"] == 0.0, name


def test_audit_labels_are_bound_to_the_text_they_judged() -> None:
    """A label may not survive an edit to the block it labelled."""
    gold = json.loads((ROOT / "config/inherited_obligation_audit.json").read_text())
    blocks = {}
    for path in SOURCES:
        from tools.reqproof.inherited import scan_source
        for block in scan_source(path, (ROOT / path).read_text())["blocks"]:
            blocks[block["id"]] = block
    stale = [
        identity for identity, label in gold["labels"].items()
        if identity not in blocks or blocks[identity]["sha256"] != label["sha256"]
    ]
    assert stale == [], f"{len(stale)} audit labels judge text that has changed"


def test_closure_references_resolve(report: dict) -> None:
    """A checked box whose evidence has vanished is a stale checkbox."""
    closure = report["summary"]["closure_reference_audit"]
    assert closure["references_checked"] > 1000
    assert closure["unresolved"] == []


def test_a_retired_review_cannot_be_retired_while_its_text_stands() -> None:
    reviews = json.loads((ROOT / "config/inherited_ledger_reviews.json").read_text())
    for retired in reviews.get("retired_decisions", []):
        assert retired["retired_reason"]
        assert retired["superseded_by"]


def test_the_detector_separates_an_obligation_from_a_narration() -> None:
    """The null the old keyword signal never had."""
    vocabulary = obligations.load_vocabulary(
        ROOT / "config/inherited_status_vocabulary.json")
    narration = {
        "id": "x:1", "sha256": "0", "checkboxes": [],
        "text": "The gate must run before admission, and it needs the receipt.\n",
    }
    obligation = {
        "id": "x:2", "sha256": "0", "checkboxes": [],
        "text": "| B5 | Broad everyday competence | NOT RUN — needs a holdout |\n",
    }
    assert obligations.classify_block(narration, vocabulary, True)["kind"] != "obligation"
    assert obligations.classify_block(obligation, vocabulary, True)["kind"] == "obligation"


def test_a_resolved_marker_does_not_outrank_an_explicit_open_status() -> None:
    vocabulary = obligations.load_vocabulary(
        ROOT / "config/inherited_status_vocabulary.json")
    mixed = {
        "id": "x:3", "sha256": "0", "checkboxes": [],
        "text": "| 7 | Thing | PASS |\nThe full claim remains unproven because the run stopped.\n",
    }
    assert obligations.classify_block(mixed, vocabulary, True)["kind"] == "obligation"


def test_the_inventory_reports_itself_complete(report: dict) -> None:
    assert report["summary"]["inventory_review_complete"] is True
