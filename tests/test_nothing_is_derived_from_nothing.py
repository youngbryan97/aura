"""A rating taken without the arguments describes no call at all.

One chain, three places, the same shape. A lease is taken against the real
arguments and recorded. Something later re-derives "what this call does" from
arguments it was not given, gets the worst case — because that is what you get
for reading an empty dict — and refuses the lease for disagreeing with it.

LIVE, 2026-08-27, in order:

* the lease check re-derived risk from an empty dict and got critical against a
  lease of medium;
* the will's `_derived_risk_level` did the same one layer up, which is where
  the critical was actually coming from;
* and `_derived_effect_scope` beside it resolved a tool's BLANKET scope — its
  most dangerous action — as "what this call does", which is how a read_only
  lease came to be refused for a state_mutation call it never made.

Each of the three was introduced to make two sides agree by construction. They
agree only when both sides have the same input.
"""

from __future__ import annotations

import pytest

from core.governance.will import _derived_effect_scope, _derived_risk_level


@pytest.mark.parametrize(
    "context",
    [
        {"tool": "code_repl"},
        {"tool": "code_repl", "authority_arguments": {}},
        {"tool": "code_repl", "authority_arguments": None},
        {"tool": "code_repl", "authority_arguments": "not a dict"},
        {},
    ],
)
def test_no_arguments_derives_nothing(context: dict) -> None:
    assert _derived_risk_level(context) == ""
    assert _derived_effect_scope(context) == ""


def test_real_arguments_still_derive_a_rating() -> None:
    """The derivation is the point; it just needs something to read."""
    context = {
        "tool": "code_repl",
        "authority_arguments": {"code": "print(sum(range(10)))"},
    }
    assert _derived_risk_level(context) == "medium"
    assert _derived_effect_scope(context)


def test_a_snippet_that_acts_still_derives_a_high_rating() -> None:
    context = {
        "tool": "code_repl",
        "authority_arguments": {"code": "import subprocess\nsubprocess.run(['ls'])"},
    }
    assert _derived_risk_level(context) in {"high", "critical"}


def test_the_lease_check_falls_back_to_the_record() -> None:
    """With nothing derived, the validator uses what the lease recorded.

    That is the other half: deriving nothing is only safe because the check
    knows what to do with nothing.
    """
    import inspect

    from core.executive import standing_authority

    source = inspect.getsource(standing_authority)
    assert "risk = record.risk_level" in source
