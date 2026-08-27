"""Authority to do something risky includes authority to do something safer.

LIVE, 2026-08-27: code_repl was leased before there was a snippet to read, so
it was leased at critical — the worst case, correctly, because nothing was
known yet. The call itself then carried a snippet that only computes and rated
medium, and the lease check refused it: `risk != record.risk_level`.

Rating a call MORE precisely made it fail, which is backwards. The scope test
one field above already allowed a narrower call, and the grant ceiling below
already used risk_at_most; only this comparison was an exact equality.
"""

from __future__ import annotations

import pytest

from core.executive.execution_policy import risk_at_most


@pytest.mark.parametrize(
    ("call", "lease"),
    [
        ("medium", "critical"),
        ("high", "critical"),
        ("low", "high"),
        ("medium", "medium"),
    ],
)
def test_a_call_no_riskier_than_its_lease_is_within_it(call: str, lease: str) -> None:
    assert risk_at_most(call, lease) is True


@pytest.mark.parametrize(
    ("call", "lease"),
    [
        ("critical", "medium"),
        ("critical", "high"),
        ("high", "low"),
    ],
)
def test_a_call_riskier_than_its_lease_is_still_refused(call: str, lease: str) -> None:
    """The lease bounds what may happen; that is the whole point of it."""
    assert risk_at_most(call, lease) is False


def test_the_lease_check_uses_the_ordering_rather_than_equality() -> None:
    """Read from the code, so the fix cannot silently revert."""
    import inspect

    from core.executive import standing_authority

    source = inspect.getsource(standing_authority)
    marker = "if risk != record.risk_level and not risk_at_most(risk, record.risk_level):"
    assert marker in source, "the lease risk check went back to an exact equality"


def test_the_refusal_names_both_risks() -> None:
    """A refusal naming only the rule cannot be diagnosed from outside.

    The scope check learned this already; the risk check said nothing about
    which two values disagreed, and finding that out took four live turns.
    """
    import inspect

    from core.executive import standing_authority

    source = inspect.getsource(standing_authority)
    assert 'standing_authority_risk_mismatch (lease={record.risk_level!r},' in source


def test_a_check_with_no_arguments_does_not_invent_a_rating() -> None:
    """Rating a call from an empty dict produces the worst case for want of input.

    LIVE, 2026-08-27: code_repl was leased at medium against a snippet that only
    computes; the lease check ran with no arguments in hand, derived critical
    from nothing, and refused the lease for disagreeing with it. The same turn
    had already failed the other way round the hour before, as the rating on
    each side got better independently.

    The comment above the fallback claims it "can only reproduce the recorded
    value for the same tool and the same arguments the lease was issued for" —
    true when there are arguments, false when there are none.
    """
    import inspect

    from core.executive import standing_authority

    source = inspect.getsource(standing_authority)
    assert "elif arguments:" in source, (
        "the lease check re-derives risk without checking it has anything to read"
    )
    assert "risk = record.risk_level" in source, (
        "with nothing to derive from, the record's own rating is the answer"
    )
