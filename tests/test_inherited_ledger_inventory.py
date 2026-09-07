"""An inventory must retain obligations that do not resemble a task bullet."""

import json
from pathlib import Path

import pytest

from tools.reqproof.inherited import apply_reviews, scan_source


def test_inline_boxes_are_separate_obligations():
    result = scan_source("todo.md", "- [x] represent - [ ] inspect - [ ] construct\n")
    assert [box["checked"] for box in result["blocks"][0]["checkboxes"]] == [True, False, False]


def test_code_example_is_not_an_unchecked_obligation():
    result = scan_source("todo.md", "```text\n- [ ] example\n\n```\n\nActual prose.\n")
    assert not any(block["checkboxes"] for block in result["blocks"])
    assert "Actual prose." in result["blocks"][-1]["text"]


def test_no_signal_prose_is_retained_for_review():
    text = "# Title\n\nA failure has no special status word.\n\n| X | DONE |\n"
    result = scan_source("todo.md", text)
    kept = "".join(block["text"] for block in result["blocks"])
    assert [line for line in kept.splitlines() if line.strip()] == [
        line for line in text.splitlines() if line.strip()
    ]
    assert all(block["review_status"] == "unreviewed" for block in result["blocks"])


def review_for(block, **overrides):
    return {"source_id": block["id"], "source_sha256": block["sha256"],
            "status": "open", "reason": "Needs the named live run.",
            "mechanism": "live-acceptance", **overrides}


def test_group_keeps_each_source_obligation():
    source = scan_source("todo.md", "First obligation.\n\nSecond obligation.\n")
    decisions = [review_for(block) for block in source["blocks"]]
    grouped = apply_reviews([source], {"decisions": decisions})
    assert grouped == {"live-acceptance": ["todo.md:1", "todo.md:3"]}


@pytest.mark.parametrize("change", [
    {"source_sha256": "stale"}, {"status": "done"}, {"reason": ""},
    {"mechanism": ""}, {"status": "verified"}, {"status": "superseded"},
])
def test_bad_review_cannot_award_coverage(change):
    source = scan_source("todo.md", "Actual obligation.\n")
    with pytest.raises(ValueError):
        apply_reviews([source], {"decisions": [review_for(source["blocks"][0], **change)]})


def test_duplicate_review_cannot_inflate_coverage():
    source = scan_source("todo.md", "Actual obligation.\n")
    decision = review_for(source["blocks"][0])
    with pytest.raises(ValueError, match="duplicate"):
        apply_reviews([source], {"decisions": [decision, decision]})


def test_reviewed_unchecked_inventory_retains_every_inline_obligation():
    from tools.reqproof.inherited import SOURCES

    root = Path(__file__).resolve().parents[1]
    sources = [scan_source(path, (root / path).read_text()) for path in SOURCES]
    reviews = json.loads((root / "config/inherited_ledger_reviews.json").read_text())
    apply_reviews(sources, reviews)
    missing = [block["id"] for source in sources for block in source["blocks"]
               if any(not box["checked"] for box in block["checkboxes"])
               and block["review_status"] == "unreviewed"]
    assert missing == []
