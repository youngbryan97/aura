from core.language.semantic_work import (
    INLINE_REPLY,
    SemanticWorkContract,
    build_semantic_work_contract,
)


def test_multipart_inline_answer_preserves_each_obligation_and_requires_deliberation():
    objective = (
        "Explain a shortest-path method in one complete response. Include: "
        "(1) its invariant, (2) numbered pseudocode, (3) a worked example, "
        "(4) two complexity bounds, and (5) a failure case with the right alternative."
    )

    contract = build_semantic_work_contract(objective)

    assert contract.delivery_mode == INLINE_REPLY
    assert contract.obligation_count >= 5
    assert any("one complete response" in item for item in contract.obligations)
    assert contract.requires_complete_reply is True
    assert contract.requires_deliberation is True
    assert contract.architecture_assistance_eligible is True
    expected = (
        "its invariant",
        "numbered pseudocode",
        "a worked example",
        "two complexity bounds",
        "a failure case with the right alternative",
    )
    assert all(
        any(fragment in obligation for obligation in contract.obligations)
        for fragment in expected
    )


def test_semantic_work_contract_is_domain_general_for_structured_explanation():
    objective = (
        "Walk me through this policy choice. Give the governing principle, "
        "compare two alternatives, and show a concrete counterexample."
    )

    contract = build_semantic_work_contract(objective)

    assert contract.delivery_mode == INLINE_REPLY
    assert contract.obligation_count == 3
    assert contract.requires_deliberation is True
    assert "policy" not in contract.decision_basis


def test_short_explanation_stays_on_low_latency_reply_path():
    contract = build_semantic_work_contract("Explain why leaves look green.")

    assert contract.delivery_mode == INLINE_REPLY
    assert contract.obligation_count == 1
    assert contract.requires_complete_reply is False
    assert contract.requires_deliberation is False
    assert contract.architecture_assistance_eligible is False


def test_external_effect_is_not_relabelled_as_inline_architecture_assistance():
    contract = build_semantic_work_contract(
        "Open Notes, write a paragraph, and export it as a PDF on my Desktop."
    )

    assert contract.delivery_mode != INLINE_REPLY
    assert contract.architecture_assistance_eligible is False


def test_semantic_work_contract_round_trips_as_typed_state():
    contract = SemanticWorkContract(
        delivery_mode=INLINE_REPLY,
        obligations=("first", "second"),
        obligation_count=2,
        requires_complete_reply=True,
        requires_deliberation=True,
        architecture_assistance_eligible=True,
        answer_token_floor=768,
        planning_token_estimate=512,
        decision_basis=("multipart", "answer_capacity"),
    )

    assert SemanticWorkContract.from_dict(contract.to_dict()) == contract
