"""A store that gained one row announced itself again.

core/brain/nonparametric_worker.py keeps a set of refusals already said out
loud, and its own comment gives the reason: "a refusal that cannot change
while the process runs is a standing fact, and a standing fact repeated
every turn is what makes a feed unreadable."

It keyed that set on the SENTENCE, and the sentence carries the entry
count. So "520 entries is too sparse" and "521 entries is too sparse" were
two different facts to it. Measured live 2026-09-18: five identical "520
entries is too sparse to steer a 5120-wide space (need 50,000)" warnings
in one boot, out of seven warnings the whole boot produced, from a set
that exists to make it one.

The count belongs in the message a reader sees. It does not belong in the
identity of the thing being deduplicated.
"""

from __future__ import annotations

from core.brain.nonparametric_worker import _refusal_condition

SPARSE = (
    "{n} entries is too sparse to steer a 5120-wide space (need 50,000); "
    "recall stays off until the store is dense enough for its neighbours to be near"
)


def test_a_growing_store_is_the_same_standing_fact():
    assert _refusal_condition(SPARSE.format(n=520)) == _refusal_condition(
        SPARSE.format(n=521)
    )
    assert _refusal_condition(SPARSE.format(n=408)) == _refusal_condition(
        SPARSE.format(n=50_000)
    )


def test_a_different_condition_is_still_its_own_fact():
    sparse = _refusal_condition(SPARSE.format(n=520))
    unrecallable = _refusal_condition(
        "only 3 of 520 entries carry a recallable token (need at least 25)"
    )
    assert sparse != unrecallable
    assert sparse == "too_sparse"
    assert unrecallable == "unrecallable_tokens"


def test_the_fraction_branch_is_the_same_condition_as_the_count_branch():
    """Both say the tokens are unrecallable; the numbers differ, the fact does not."""

    assert _refusal_condition(
        "only 3 of 520 entries carry a recallable token (need at least 25)"
    ) == _refusal_condition(
        "517 of 520 entries carry no recallable token (0.6% usable)"
    )


def test_an_unrecognised_refusal_still_deduplicates_on_itself():
    first = _refusal_condition("some new refusal nobody has classified")
    assert first
    assert first == _refusal_condition("some new refusal nobody has classified")


def test_empty_is_handled():
    assert _refusal_condition("") == ""


def test_the_worker_deduplicates_on_the_condition():
    import inspect

    from core.brain import nonparametric_worker

    source = inspect.getsource(nonparametric_worker)
    assert "condition = _refusal_condition(unusable)" in source
    assert "if condition not in _REPORTED_REFUSALS" in source
    assert "_REPORTED_REFUSALS.add(condition)" in source
