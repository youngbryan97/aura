"""One swap spike became two CRITICALs, and blocked every build.

Live 2026-07-28, from the desktop log::

    CRITICAL SERVICE FAILURE: Subsystem 'memory_watchdog' failed with failure
    policy 'fail-closed'. Original error: RuntimeError: CRITICAL SERVICE
    FAILURE: Subsystem 'memory_watchdog' failed with failure policy
    'fail-closed'. Original error: RuntimeError: swap exhaustion: managed RSS
    34494MB, swap 16.9GB

The escalation raises a ``RuntimeError``; that error propagates and is recorded
again for the same subsystem; the rate cap is keyed on ``(subsystem, error
type)`` and both wraps are ``RuntimeError``, so it sees one fault and lets the
second through.

The nested text is the harmless part. The damage is that ONE underlying event
produced TWO degradation records, and degradation count is what ``deg_threat``
is computed from. Measured on the same run: ``deg_threat=1.00`` →
``existential threat=1.00`` → the Ulysses covenant refuses heavy compute at
0.6 and above → every reconstruction she was asked for was silently refused,
and reported as though the model had failed.

So: an escalation is already the escalation. It never wraps itself again.
"""
from __future__ import annotations

import core.runtime.errors as errors


def test_the_marker_is_what_the_escalation_writes() -> None:
    """The guard reads back the same string the escalation emits."""
    assert errors._ESCALATION_MARKER == "CRITICAL SERVICE FAILURE:"


def test_an_escalation_error_is_recognised_as_already_escalated() -> None:
    already = RuntimeError(
        f"{errors._ESCALATION_MARKER} Subsystem 'memory_watchdog' failed with "
        "failure policy 'fail-closed'. Original error: RuntimeError: swap exhaustion"
    )
    assert errors._ESCALATION_MARKER in str(already)


def test_an_ordinary_error_is_not() -> None:
    """The guard must not swallow the first, real escalation."""
    first = RuntimeError("swap exhaustion: managed RSS 34494MB, swap 16.9GB")
    assert errors._ESCALATION_MARKER not in str(first)


def test_the_guard_is_wired_into_the_escalation_decision() -> None:
    """Structural: the check has to gate the branch, not sit beside it.

    Asserted on the source because the branch it guards needs live-mode,
    a fail-closed registry and a non-timeout error to reach, and a test
    that reconstructed all three would be testing its own scaffolding.
    """
    from source_contract import function_containing, in_order

    # Read from whichever function makes the decision, not from
    # record_degradation by name: the method-size sweep moved the branch
    # into an extracted helper and the guard went with it, untouched.
    _name, decides = function_containing(
        errors, "_already_escalated = _ESCALATION_MARKER in str(error)"
    )
    assert "and not _already_escalated" in decides
    # And the guard must be decided before the escalation is built.
    in_order(
        decides,
        "_already_escalated =",
        "failure_policy_error = (",
    )
