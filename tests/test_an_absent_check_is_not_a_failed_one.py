"""A missing receipt says nothing was checked, not that the answer was short.

LIVE 2026-08-29. A turn listed a directory, read an API reference, ran the
library's code and wrote a reply. It was refused with `authored_answer_incomplete`
beside `live_mind_controls_unbound`, at `completion_retries=0`. Nothing had been
cut off: both proofs were one receipt that was never bound, and the name sent
the investigation after a truncation that did not exist.
"""

from __future__ import annotations

import pytest

from interface.routes.chat import _only_soft_proofs_missing

pytestmark = pytest.mark.unit


def _hers(**missing) -> dict:
    """A contract whose text is genuinely hers, missing what the test names."""

    return {
        "engine_think_invoked": True,
        "cognitive_engine_reply_accepted": True,
        "cognitive_engine_reply_failed": False,
        "bounded_contract_used": False,
        "legacy_fallback_used": False,
        "authorship_replacement_applied": False,
        **missing,
    }


def test_nobody_checked_is_disclosed_rather_than_substituted() -> None:
    """The same category as the control bookkeeping already beside it."""

    assert _only_soft_proofs_missing(
        _hers(
            full_mind_missing_proofs=[
                "authored_answer_incomplete:nobody_checked",
                "live_mind_controls_unbound:not_bound",
            ]
        )
    )


@pytest.mark.parametrize(
    "cause",
    [
        "authored_answer_incomplete:generation_cut_off",
        "authored_answer_incomplete:retry_exhausted",
        "authored_answer_incomplete:semantically_short",
        "authored_answer_incomplete:semantic_contract_unmet",
        "authored_answer_incomplete",
    ],
)
def test_a_claim_about_the_answer_still_fails_closed(cause: str) -> None:
    """Only the absence is waived. Every cause that is about the text is not."""

    assert not _only_soft_proofs_missing(_hers(full_mind_missing_proofs=[cause]))


def test_authorship_is_never_waived_by_any_of_this() -> None:
    for forged in (
        {"authorship_replacement_applied": True},
        {"bounded_contract_used": True},
        {"legacy_fallback_used": True},
        {"cognitive_engine_reply_failed": True},
    ):
        assert not _only_soft_proofs_missing(
            _hers(
                full_mind_missing_proofs=["authored_answer_incomplete:nobody_checked"],
                **forged,
            )
        )


@pytest.mark.parametrize(
    ("flag", "named"),
    [
        ("completion_retry_exhausted", "retry_exhausted"),
        ("reply_generation_incomplete", "generation_cut_off"),
        ("semantic_completion_incomplete", "semantically_short"),
    ],
)
def test_the_contract_names_the_cause_it_found(monkeypatch, flag, named) -> None:
    """Each cause reaches the log under its own name."""

    from tests.test_full_mind_proof_degradation import (
        _force_full_mind_runtime,
        _green_trace,
        _payload,
    )
    from interface.routes import chat as chat_routes

    _force_full_mind_runtime(monkeypatch, chat_routes)
    trace = _green_trace()
    trace[flag] = True

    missing = _payload(chat_routes, trace)["full_mind_missing_proofs"]
    assert f"authored_answer_incomplete:{named}" in missing


class TestOnePolicyAtBothExits:
    """A proof disclosable at one exit and fatal at the next is not a policy.

    LIVE 2026-08-29: a turn that read a library's docs, wrote code against
    them, ran it and produced the trial balance was refused on
    authored_answer_incomplete:nobody_checked and
    live_mind_controls_unbound:not_applied — both on the soft list, both
    already disclosable at the other exit, on a turn whose ownership had just
    been proven.
    """

    def test_a_turn_missing_only_soft_proofs_can_serve_here_too(self) -> None:
        from interface.routes.chat import _authored_answer_can_serve

        assert _authored_answer_can_serve(
            _hers(
                full_mind_missing_proofs=[
                    "authored_answer_incomplete:nobody_checked",
                    "live_mind_controls_unbound:not_applied",
                ]
            )
        )

    def test_the_strong_proof_still_serves_on_its_own(self) -> None:
        from interface.routes.chat import _authored_answer_can_serve

        assert _authored_answer_can_serve(
            {"answer_delivery_proven": True, "authentic_cognitive_reply": True}
        )

    def test_a_claim_about_the_text_is_still_fatal_here(self) -> None:
        from interface.routes.chat import _authored_answer_can_serve

        assert not _authored_answer_can_serve(
            _hers(full_mind_missing_proofs=["authored_answer_incomplete:generation_cut_off"])
        )

    def test_repair_text_never_serves_by_this_route(self) -> None:
        from interface.routes.chat import _authored_answer_can_serve

        for forged in (
            {"authorship_replacement_applied": True},
            {"bounded_contract_used": True},
            {"legacy_fallback_used": True},
        ):
            assert not _authored_answer_can_serve(
                _hers(
                    full_mind_missing_proofs=[
                        "authored_answer_incomplete:nobody_checked"
                    ],
                    **forged,
                )
            )
