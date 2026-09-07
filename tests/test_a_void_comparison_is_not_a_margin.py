"""A margin over an unmatched comparison cannot certify anything.

`core/evaluation/matched_budget.py` states the rule plainly: differences on
outcome-determining dimensions make a comparison void — "not flagged — void".
The DNU battery computes exactly that report and writes it into BASELINES.json
beside the baseline pass rates, and currently declares the comparison VOID,
which is the correct scientific conclusion.

The final bundle validator then read the pass rates and never the report. Its
check 11 — full Aura must materially outperform the external baselines, the
substantive "more than a wrapper" proof — could be satisfied by a comparison
the repository itself had already voided.

That is the parity machinery's own defect class, one layer above where the
machinery was installed: the number and the reason it cannot be read sat in
the same file, and only the number was consulted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "agi"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from validate_dnu_final_bundle import why_the_margin_cannot_be_read  # noqa: E402


def test_a_void_comparison_refuses_the_margin():
    """The defect, stated as the rule it violated."""
    said = why_the_margin_cannot_be_read(
        {
            "_budget_parity": {
                "matched": False,
                "refusal_reason": "max_output_tokens differs: 160 vs unbounded",
                "what_would_match_them": "cap the live path at the same budget",
            }
        }
    )
    assert said
    assert "VOID" in said
    assert "max_output_tokens" in said
    assert "cap the live path" in said, "the refusal must say what would fix it"


def test_a_matched_comparison_lets_the_margin_be_read():
    """The gate must not be a way of never certifying anything."""
    assert why_the_margin_cannot_be_read({"_budget_parity": {"matched": True}}) == ""


def test_no_parity_report_is_a_refusal_and_not_a_pass():
    """Absence of a check is not a passed check."""
    assert why_the_margin_cannot_be_read({})
    assert why_the_margin_cannot_be_read({"_budget_parity": None})


def test_a_parity_report_of_the_wrong_shape_is_a_refusal():
    """A truthy value that is not a report cannot be read as one."""
    assert why_the_margin_cannot_be_read({"_budget_parity": "matched"})
    assert why_the_margin_cannot_be_read({"_budget_parity": []})


@pytest.mark.parametrize("said", [None, 0, "", "true", 1])
def test_only_the_boolean_true_counts_as_matched(said):
    """`matched: 1` and `matched: "true"` are data somebody wrote, not a verdict."""
    assert why_the_margin_cannot_be_read({"_budget_parity": {"matched": said}})


def test_the_validator_calls_the_gate_before_reading_the_margin():
    """Ordering is the whole point: the report is computed, then ignored."""
    import inspect

    import validate_dnu_final_bundle as validator

    source = inspect.getsource(validator.main)
    gate = source.index("why_the_margin_cannot_be_read")
    margin = source.index("did not materially outperform")
    assert gate < margin, "the margin is read before parity is checked"
