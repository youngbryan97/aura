"""Her answer existed. A gate threw it away and blamed the runtime.

Live, 2026-07-27, twice in one conversation:

    ✅ Cortex response received (len=199)
    ⚠️ Response confidence: degraded (same_answer_diff_prompt=True, ...)
    ⚠️ Required desktop full-mind contract was not proven; failing closed
       instead of serving partial/raw speech (path=cognitive_engine)

    -> "I couldn't get my full attention onto that one, and I'd rather tell you
        that than answer from half of it. Try me again in a moment."

The path was ``cognitive_engine``. The engine was invoked, the reply was
accepted, no repair machinery and no legacy fallback wrote a word of it. The
only unmet proof was ``confidence:degraded`` — she had recently said something
similar — and the full-mind contract requires "high". So a real answer was
replaced by an apology, which then became the previous answer and made the next
turn look repetitive too.

Three sibling gates — regenerate, recovery, candidate — already serve an
authentic reply with the shortfall disclosed rather than failing closed. This
one did not, because it evaluates the contract at the real confidence instead
of forcing "high", so ``authentic_cognitive_reply`` could never be true there.

The distinction the gate must keep making is authorship. Bounded-repair text
and legacy fallbacks must never speak in her voice; that stays fail-closed. Her
own words with a soft confidence reading are hers, and get served.
"""
from __future__ import annotations

import pytest

from interface.routes.chat import _authored_answer_can_serve, _only_soft_proofs_missing


def _contract(**overrides: object) -> dict:
    base = {
        "full_mind_missing_proofs": ["confidence:degraded"],
        "engine_think_invoked": True,
        "cognitive_engine_reply_accepted": True,
        "cognitive_engine_reply_failed": False,
        "bounded_contract_used": False,
        "legacy_fallback_used": False,
    }
    base.update(overrides)
    return base


def test_the_live_failure_now_serves_her_answer() -> None:
    assert _only_soft_proofs_missing(_contract())


@pytest.mark.parametrize(
    "confidence", ["confidence:degraded", "confidence:medium", "confidence:unset"]
)
def test_any_soft_confidence_reading_is_disclosed_not_substituted(confidence: str) -> None:
    assert _only_soft_proofs_missing(_contract(full_mind_missing_proofs=[confidence]))


# ── Authorship is never waived ─────────────────────────────────────────────

@pytest.mark.parametrize(
    "override",
    [
        {"bounded_contract_used": True},
        {"legacy_fallback_used": True},
        {"engine_think_invoked": False},
        {"cognitive_engine_reply_accepted": False},
        {"cognitive_engine_reply_failed": True},
    ],
)
def test_text_she_did_not_author_still_fails_closed(override: dict) -> None:
    """Theatre must never serve as Aura speech, whatever the confidence says."""
    assert not _only_soft_proofs_missing(_contract(**override))


@pytest.mark.parametrize(
    "proof",
    [
        "engine_reply_failed",
        "bounded_repair_authored_text",
        "legacy_fallback_authored_text",
        "response_path:bounded_repair",
        "latent_cortex_path_unproven",
        "subsystem:memory",
        "final_output_contract_unsatisfied",
    ],
)
def test_a_hard_proof_alongside_soft_confidence_still_fails_closed(proof: str) -> None:
    assert not _only_soft_proofs_missing(
        _contract(full_mind_missing_proofs=["confidence:degraded", proof])
    )


# ── State bookkeeping is not authorship ────────────────────────────────────
#
# ``live_mind_controls_unbound`` was in the hard list above, against this
# file's own thesis: the distinction the gate must keep making is AUTHORSHIP,
# and an unbound generation control says nothing about whose words these are.
#
# Live 2026-08-04 13:53 it cost a real answer. Asked to show a snippet of her
# code and say where it lives, she produced 1999 characters the quality pass
# marked ``assessment=ok``, and the turn was replaced with "I couldn't get my
# full attention onto that one." The same minute, at a sibling exit, the log
# reads "Desktop turn served with DEGRADED full-mind proof (authentic
# cognitive reply; missing: live_mind_controls_unbound)" — the identical
# proof, disclosed and served. A proof that is fatal at one exit and
# disclosable at the next is not a policy; it is which exit the turn took.


@pytest.mark.parametrize(
    "proof",
    [
        "live_mind_controls_unbound",
        "live_mind_snapshot_not_ready",
        "architecture_context_unbound",
    ],
)
def test_unbound_internal_state_is_disclosed_not_substituted(proof: str) -> None:
    assert _only_soft_proofs_missing(_contract(full_mind_missing_proofs=[proof]))


@pytest.mark.parametrize(
    "override",
    [
        {"bounded_contract_used": True},
        {"legacy_fallback_used": True},
        {"engine_think_invoked": False},
        {"cognitive_engine_reply_accepted": False},
    ],
)
def test_unbound_state_never_waives_authorship(override: dict) -> None:
    """Soft state proofs must not become a hole authorship falls through."""
    assert not _only_soft_proofs_missing(
        _contract(full_mind_missing_proofs=["live_mind_controls_unbound"], **override)
    )


def test_a_fully_proven_turn_is_not_routed_through_the_soft_path() -> None:
    """No missing proofs means the contract passed; this branch must not fire."""
    assert not _only_soft_proofs_missing(_contract(full_mind_missing_proofs=[]))


def test_a_malformed_contract_is_not_treated_as_permission() -> None:
    assert not _only_soft_proofs_missing(None)
    assert not _only_soft_proofs_missing("contract")


def test_authored_answer_survives_hard_certification_gaps() -> None:
    contract = _contract(
        full_mind_missing_proofs=["live_mind_controls_unbound", "subsystem:memory"],
        authentic_cognitive_reply=True,
        answer_delivery_proven=True,
    )

    assert _authored_answer_can_serve(contract)


def test_authored_answer_still_requires_completed_user_contract() -> None:
    """Without the delivery proof, what serves depends on WHY it is missing.

    The fixture sets answer_delivery_proven False with only soft proofs in the
    list, and those two cannot both be arbitrary: the delivery proof fails
    BECAUSE of the proofs that are missing. A hard one still refuses. A soft
    one is disclosed, which is the same policy the other exit has always
    applied — a proof disclosable at one exit and fatal at the next is an
    accident of which exit the turn took, not a rule.
    """

    assert not _authored_answer_can_serve(
        _contract(
            authentic_cognitive_reply=True,
            answer_delivery_proven=False,
            full_mind_missing_proofs=["engine_reply_failed"],
        )
    )
    assert not _authored_answer_can_serve(
        _contract(
            authentic_cognitive_reply=True,
            answer_delivery_proven=False,
            full_mind_missing_proofs=["authored_answer_incomplete:generation_cut_off"],
        )
    )
    # And authorship is never waived by the soft route.
    assert not _authored_answer_can_serve(
        _contract(
            authentic_cognitive_reply=True,
            answer_delivery_proven=False,
            authorship_replacement_applied=True,
        )
    )
    # Soft only, and hers: served with the degradation disclosed.
    assert _authored_answer_can_serve(
        _contract(authentic_cognitive_reply=True, answer_delivery_proven=False)
    )
