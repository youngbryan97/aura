"""Every tool says what it gives back, and the runtime checks that it did.

CrewAI requires structured schemas both ways. Aura declared the argument side
and left the result to whatever the caller happened to get: 82 tools, 78 of
them silent. Reading them, 72 returned a dict with `ok`, which is a real
shared contract rather than a convention somebody hoped for.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.skills.what_every_skill_gives_back import (
    THE_SHARED_RESULT,
    check_a_result,
    forget_everything,
    how_results_have_differed,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean():
    forget_everything()
    yield
    forget_everything()


# ---------------------------------------------------------- the contract


def test_the_shared_result_requires_only_what_every_skill_keeps():
    assert THE_SHARED_RESULT["required"] == ["ok"]
    assert set(THE_SHARED_RESULT["properties"]) == {"ok", "error", "summary"}


def test_the_shared_result_allows_extra_fields():
    """A schema claiming to be complete would be wrong for every skill that adds one.

    And a wrong schema is worse than none, because consumers act on it.
    """
    assert THE_SHARED_RESULT["additionalProperties"] is True
    assert check_a_result("a skill", THE_SHARED_RESULT, {"ok": True, "extra": 1}) == []


# ------------------------------------------------------------- the check


def test_a_result_that_matches_is_not_complained_about():
    assert check_a_result("clock", THE_SHARED_RESULT, {"ok": True}) == []
    assert how_results_have_differed() == {}


def test_a_missing_required_field_is_named():
    complaints = check_a_result("bad", THE_SHARED_RESULT, {"summary": "x"})
    assert "declared ok and did not give it" in complaints


def test_a_field_of_the_wrong_type_is_named():
    complaints = check_a_result("bad", THE_SHARED_RESULT, {"ok": "yes"})
    assert any("ok was declared a boolean" in one for one in complaints)


def test_a_result_that_is_not_an_object_is_named():
    complaints = check_a_result("bad", THE_SHARED_RESULT, "a string")
    assert complaints == ["declared an object and gave back str"]


def test_a_closed_schema_complains_about_fields_it_did_not_declare():
    closed = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    complaints = check_a_result("strict", closed, {"ok": True, "surprise": 1})
    assert any("did not declare: surprise" in one for one in complaints)


def test_the_check_never_raises_and_never_changes_the_result():
    result = {"ok": "not a bool"}
    check_a_result("bad", THE_SHARED_RESULT, result)
    assert result == {"ok": "not a bool"}


def test_a_skill_with_no_declaration_is_not_complained_about():
    assert check_a_result("undeclared", {}, {"anything": True}) == []
    assert check_a_result("undeclared", None, {"anything": True}) == []


def test_the_same_complaint_twice_is_counted_twice():
    for _ in range(3):
        check_a_result("bad", THE_SHARED_RESULT, {})
    assert how_results_have_differed()["bad"][
        "declared ok and did not give it"
    ] == 3


# ------------------------------------------------------------ the ratchet


def test_every_tool_says_what_it_gives_back():
    """78 of 82 did not. The baseline is zero and it only stays zero."""
    baseline = json.loads(
        (ROOT / "config" / "tool_contract_baseline.json").read_text("utf-8")
    )
    assert baseline["without_a_result"] == 0

    from tools.lint_tool_contracts import measure

    now = measure()
    assert now["without_a_result"] <= baseline["without_a_result"], (
        f"undeclared again: {now['which']['without_a_result']}"
    )


def test_the_one_skill_with_a_different_shape_declares_that_shape():
    """local_reference answers with `success`, and says so rather than lying."""
    from core.skills.local_reference import LocalReferenceSearchSkill

    declared = LocalReferenceSearchSkill.result_schema
    assert declared["required"] == ["success", "results"]
    assert "ok" not in declared["properties"]


def test_the_live_executor_checks_what_a_skill_gave_back():
    source = (
        ROOT / "core" / "runtime" / "action_executor.py"
    ).read_text("utf-8")
    assert "_check_what_the_skill_gave_back(engine, action_name, raw_result)" in source
