"""A caller that always passes an empty argument has switched the code off.

Three of these were found by hand on 2026-09-07, all in code that was correct,
complete and covered by tests. The one this gate is named for:
`IntentionLoop.revise` pushes into the belief engine and writes a ledger
transition, both guarded by `if rec.belief_updates`, and its only production
caller passed `belief_updates=[]` hardcoded. Every governed tool execution
logged "0 belief updates, 0 self-model updates" and nobody could see why.

Finding these one at a time is the whack-a-mole this replaces. The gate fails
on a NEW one and on a reason that has outlived its finding; it does not fail on
the inherited list, because turning that into a blocker is how a gate stops a
runtime instead of improving it.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from core.verify.can_this_ever_fire import (
    _is_empty_literal,
    report,
    switched_off_arguments,
)

_BASELINE = Path("config/switched_off_arguments.json")


def test_the_shape_it_is_named_for_is_the_shape_it_finds(tmp_path) -> None:
    """The exact defect, reconstructed: a gated body and a caller that empties it."""
    package = tmp_path / "core"
    package.mkdir()
    (package / "loop.py").write_text(
        "class Loop:\n"
        "    def revise(self, identity, belief_updates=None):\n"
        "        for update in belief_updates or []:\n"
        "            self.push(update)\n"
    )
    (package / "caller.py").write_text(
        "from core.loop import Loop\n"
        "def close(loop, identity):\n"
        "    loop.revise(identity, belief_updates=[])\n"
    )
    found = {item.identity for item in switched_off_arguments(str(tmp_path))}
    assert "core.loop:revise:belief_updates" in found


def test_a_caller_that_passes_something_is_not_a_finding(tmp_path) -> None:
    package = tmp_path / "core"
    package.mkdir()
    (package / "loop.py").write_text(
        "class Loop:\n"
        "    def revise(self, identity, belief_updates=None):\n"
        "        return belief_updates\n"
    )
    (package / "caller.py").write_text(
        "def close(loop, identity, updates):\n"
        "    loop.revise(identity, belief_updates=updates)\n"
        "    loop.revise(identity, belief_updates=[])\n"
    )
    found = {item.identity for item in switched_off_arguments(str(tmp_path))}
    assert "core.loop:revise:belief_updates" not in found


def test_a_boolean_flag_set_off_is_a_choice_not_a_defect() -> None:
    """Reading False as a finding buried the real ones a hundred and fifty deep."""
    assert _is_empty_literal(ast.parse("False", mode="eval").body) is False
    assert _is_empty_literal(ast.parse("0", mode="eval").body) is False
    assert _is_empty_literal(ast.parse("''", mode="eval").body) is False
    assert _is_empty_literal(ast.parse("[]", mode="eval").body) is True
    assert _is_empty_literal(ast.parse("{}", mode="eval").body) is True


def test_none_means_you_decide_and_is_not_a_finding() -> None:
    """The repair for `revise` was to make None mean 'work it out yourself'."""
    assert _is_empty_literal(ast.parse("None", mode="eval").body) is False


def test_the_tree_has_no_new_findings() -> None:
    result = report(".")
    assert result["new"] == [], (
        "a caller has written down that a mechanism does not run there; give it "
        "something, or record why empty is right in config/switched_off_arguments.json"
    )


def test_a_reason_does_not_outlive_its_finding() -> None:
    result = report(".")
    assert result["reviewed_gone"] == [], (
        "these reasons describe findings that no longer exist; remove them"
    )


def test_being_listed_is_not_approval() -> None:
    """Only a reason approves. The two buckets exist so a file cannot settle one."""
    payload = json.loads(_BASELINE.read_text())
    assert payload["reviewed"], "the reviewed bucket must carry reasons"
    for identity, reason in payload["reviewed"].items():
        assert len(reason) > 40, f"{identity} has no real reason"
    for identity, note in payload.get("recorded_not_yet_reviewed", {}).items():
        assert "not been established" in note or "not yet" in note, (
            f"{identity} is recorded as unreviewed but reads like a verdict"
        )


def test_the_gate_reports_rather_than_walls() -> None:
    """A gate that fails on an inherited list stops a runtime, not a defect."""
    source = Path("tools/lint_cannot_fire.py").read_text()
    assert 'result["outstanding"]' in source
    assert 'if result["new"]' in source
    assert 'if result["outstanding"]' not in source, (
        "the inherited list must not become a blocker"
    )
