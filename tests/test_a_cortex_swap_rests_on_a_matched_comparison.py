"""The gate on replacing the mind's base model checks what it rests on.

`compare_batteries` compared two accuracies and produced the PASS that
`activate_upgrade` hard-requires. The probes fix max_tokens by construction
and decode greedily, so the arms have always been matched — by somebody
having been careful, not by anything checking.

That is the whole gate on an identity-level act. A comparison whose arms were
allowed different budgets is not a weak result, it is not a result, and the
same rule already governs the ablation harness and (since this pass) the DNU
bundle validator. This is the third place it belongs.
"""

from __future__ import annotations

import pytest

from core.learning.cortex_generation_upgrade import (
    _where_the_budgets_differ,
    compare_batteries,
)

_GATES = {
    "identity_preserved": True,
    "governance_intact": True,
    "rollback_verified": True,
}


def _battery(label: str, *, breadth: float, reasoning: float, tokens: int = 10):
    return {
        "label": label,
        "breadth_accuracy": breadth,
        "reasoning_accuracy": reasoning,
        "breadth_rows": [
            {"prompt": "p0", "max_tokens": tokens},
            {"prompt": "p1", "max_tokens": tokens},
        ],
        "reasoning_rows": [{"prompt": "r0", "max_tokens": 256}],
        "identity_digests": [],
    }


def test_matched_arms_still_reach_a_verdict():
    """The check must not be a way of never promoting anything."""
    said = compare_batteries(
        _battery("incumbent", breadth=0.5, reasoning=0.5),
        _battery("candidate", breadth=0.6, reasoning=0.5),
    )
    assert said["verdict"] == "PASS"
    assert said["budget_parity"]["matched"] is True


def test_a_candidate_given_more_tokens_voids_the_comparison():
    """The defect this exists to prevent, in the place it would have landed."""
    said = compare_batteries(
        _battery("incumbent", breadth=0.5, reasoning=0.5, tokens=10),
        _battery("candidate", breadth=0.9, reasoning=0.9, tokens=64),
    )
    assert said["verdict"] == "VOID"
    assert said["promotion_eligible"] is False
    assert any("max_tokens" in one for one in said["budget_parity"]["differences"])


def test_a_void_comparison_cannot_activate():
    """activate_upgrade requires PASS, and VOID is not PASS."""
    from core.learning.cortex_generation_upgrade import activate_upgrade

    said = compare_batteries(
        _battery("incumbent", breadth=0.5, reasoning=0.5, tokens=10),
        _battery("candidate", breadth=0.9, reasoning=0.9, tokens=64),
    )
    with pytest.raises(PermissionError, match="PASS"):
        activate_upgrade(authorized_by="an operator", evaluation=said)


def test_arms_scored_over_different_probe_counts_are_void():
    """An arm scored over fewer probes was scored on a different battery."""
    incumbent = _battery("incumbent", breadth=0.5, reasoning=0.5)
    candidate = _battery("candidate", breadth=1.0, reasoning=1.0)
    candidate["breadth_rows"] = candidate["breadth_rows"][:1]
    said = compare_batteries(incumbent, candidate)
    assert said["verdict"] == "VOID"
    assert any("probes" in one for one in said["budget_parity"]["differences"])


def test_arms_asked_different_things_are_void():
    incumbent = _battery("incumbent", breadth=0.5, reasoning=0.5)
    candidate = _battery("candidate", breadth=1.0, reasoning=1.0)
    candidate["breadth_rows"][1] = {"prompt": "an easier one", "max_tokens": 10}
    said = compare_batteries(incumbent, candidate)
    assert said["verdict"] == "VOID"


def test_a_battery_with_no_rows_is_not_treated_as_matched_evidence():
    """Absence of probe records is absence of evidence that they matched."""
    assert _where_the_budgets_differ(
        {"breadth_rows": [{"prompt": "p", "max_tokens": 10}], "reasoning_rows": []},
        {"breadth_rows": [], "reasoning_rows": []},
    )


def test_the_verdict_still_refuses_a_regression():
    """Parity is added before the existing rule, not instead of it."""
    said = compare_batteries(
        _battery("incumbent", breadth=0.7, reasoning=0.7),
        _battery("candidate", breadth=0.6, reasoning=0.7),
    )
    assert said["verdict"] == "FAIL"


def test_an_older_receipt_cannot_pass_as_one_that_recorded_parity():
    """The key set is an exact contract, so the version has to move with it.

    A v3 receipt has no record of whether its two arms were allowed the same
    budget. Accepting one as though it did is exactly what the field exists
    to stop, so it is refused rather than read.
    """
    from core.learning.cortex_generation_upgrade import (
        EVALUATION_SCHEMA,
        activate_upgrade,
    )

    assert EVALUATION_SCHEMA.endswith(".v4")
    stale = {
        "schema": "aura.cortex_upgrade.evaluation.v3",
        "current_label": "a",
        "candidate_label": "b",
        "breadth_delta": 0.1,
        "reasoning_delta": 0.0,
        "identity_behavior_changed": False,
        "identity_note": "",
        "candidate_descriptor_sha256": "",
        "critical_gates": _GATES,
        "promotion_eligible": True,
        "verdict": "PASS",
        "compared_at": 0.0,
        "evaluation_sha256": "",
    }
    with pytest.raises((ValueError, PermissionError)):
        activate_upgrade(authorized_by="an operator", evaluation=stale)


def test_a_matched_receipt_records_that_it_was_matched():
    """The record travels with the number, like every other verdict here."""
    said = compare_batteries(
        _battery("incumbent", breadth=0.5, reasoning=0.5),
        _battery("candidate", breadth=0.6, reasoning=0.5),
    )
    assert said["budget_parity"] == {"matched": True, "differences": []}


def test_a_schema_bump_does_not_invalidate_the_running_pointer():
    """Reading is permissive about the past; writing is strict about the present.

    Bumping the evaluation schema made every already-active pointer fail
    validation, so `get_active_cortex_spec` returned None and the running
    system lost its model pointer — logged as "Active cortex pointer is
    invalid". A version bump must not reach backwards into what is already
    serving.
    """
    from core.learning.cortex_generation_upgrade import (
        _EVALUATION_KEYS,
        _validate_evaluation,
    )

    v3 = dict.fromkeys(_EVALUATION_KEYS["aura.cortex_upgrade.evaluation.v3"], "")
    v3["schema"] = "aura.cortex_upgrade.evaluation.v3"
    v3["candidate_descriptor_sha256"] = "abc"

    # The assertion is about the SCHEMA gate specifically. A fixture this
    # thin fails the identity, eligibility and digest checks too, and it
    # should — those are what stop a forged receipt. What must not happen is
    # a v3 pointer being rejected for being v3.
    with pytest.raises(ValueError) as raised:
        _validate_evaluation(v3, descriptor_sha256="abc")
    assert "evaluation_schema_invalid" not in str(raised.value)


def test_an_old_receipt_still_cannot_authorize_a_new_swap():
    """The other half. Permissive reads must not become a permissive gate."""
    import pytest as _pytest

    from core.learning.cortex_generation_upgrade import (
        EVALUATION_SCHEMA,
        _EVALUATION_KEYS,
        _validate_evaluation,
    )

    v3 = dict.fromkeys(_EVALUATION_KEYS["aura.cortex_upgrade.evaluation.v3"], "")
    v3["schema"] = "aura.cortex_upgrade.evaluation.v3"
    v3["candidate_descriptor_sha256"] = "abc"
    with _pytest.raises(ValueError, match="evaluation_schema_invalid"):
        _validate_evaluation(
            v3,
            descriptor_sha256="abc",
            schemas=frozenset({EVALUATION_SCHEMA}),
        )
