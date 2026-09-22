"""Questions distinguish hypotheses without pretending their answer is known."""

import pytest
import asyncio
import json

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_inquiry import ProgramInquiry, plan_program_inquiries
from core.learning.semantic_program_portfolio import select_semantic_program_portfolio


def program(op):
    return Program(2, (Instruction(op, (0, 1)),))


def test_same_current_answer_leads_to_a_different_probe():
    proposals = {"add": program("add"), "multiply": program("mul")}
    inquiries = plan_program_inquiries(proposals, [(2, 2), (3, 2)], fuel=100_000)
    assert len(inquiries) == 1
    question = inquiries[0]
    assert question.inputs == (3, 2)
    assert question.expected_bits == pytest.approx(1, abs=1e-6)
    assert question.compatible_methods(observed_result=5) == ("add",)
    assert question.compatible_methods(observed_result=6) == ("multiply",)
    assert question.compatible_methods(observed_result=99) == ()
    assert question.to_dict()["observed_result"] is None
    assert question.to_dict()["correctness_authority"] is False


def test_duplicate_method_does_not_change_question_information():
    original = {"add": program("add"), "multiply": program("mul")}
    duplicate = {**original, "another_add": program("add")}
    first = plan_program_inquiries(original, [(3, 2)], fuel=100_000)[0]
    second = plan_program_inquiries(duplicate, [(3, 2), (3, 2)], fuel=100_000)
    assert len(second) == 1
    assert second[0].expected_bits == first.expected_bits


def test_undefined_execution_is_not_an_observation():
    assert not plan_program_inquiries(
        {"divide": program("idiv"), "add": program("add")}, [(2, 0)], fuel=100_000
    )


def test_portfolio_conflicts_reach_the_shared_question_planner():
    portfolio = select_semantic_program_portfolio(
        proposals={"add": program("add"), "multiply": program("mul")},
        provenance={"add": "a" * 64, "multiply": "b" * 64},
        public_inputs=(2, 2),
        observation_sha256="c" * 64,
        incumbent="add",
    )
    assert portfolio.plan_inquiries()
    assert portfolio.decision.selected == "add"


def test_missing_and_boolean_observations_are_not_integer_answers():
    question = plan_program_inquiries(
        {"add": program("add"), "multiply": program("mul")}, [(3, 2)], fuel=100_000
    )[0]
    for value in (None, True, "5"):
        with pytest.raises(ValueError):
            question.compatible_methods(observed_result=value)


def test_inquiry_identity_binds_the_hypotheses_not_only_probe_values():
    first = plan_program_inquiries(
        {"a": program("add"), "b": program("mul")}, [(3, 2)], fuel=100_000
    )[0]
    second = plan_program_inquiries(
        {"a": program("add"), "b": program("sub")}, [(3, 2)], fuel=100_000
    )[0]
    assert first.identity != second.identity


def test_pending_inquiry_roundtrips_without_becoming_an_observation():
    inquiry = plan_program_inquiries(
        {"add": program("add"), "multiply": program("mul")}, [(3, 2)], fuel=100_000
    )[0]
    restored = ProgramInquiry.from_dict(json.loads(json.dumps(inquiry.to_dict())))
    assert restored == inquiry
    assert restored.to_dict()["observed_result"] is None


@pytest.mark.parametrize("field,value", [
    ("inputs", [9, 2]), ("expected_bits", float("nan")),
    ("observed_result", 5), ("correctness_authority", True),
])
def test_pending_inquiry_rejects_corrupted_bindings(field, value):
    inquiry = plan_program_inquiries(
        {"add": program("add"), "multiply": program("mul")}, [(3, 2)], fuel=100_000
    )[0]
    payload = inquiry.to_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        ProgramInquiry.from_dict(payload)


def test_inquiry_survives_reopening_the_real_state_gateway(tmp_path):
    from core.state.state_gateway import ConcreteStateGateway

    inquiry = plan_program_inquiries(
        {"add": program("add"), "multiply": program("mul")}, [(3, 2)], fuel=100_000
    )[0]

    async def run():
        gateway = ConcreteStateGateway(root=tmp_path, governance_decide=lambda **_: True)
        receipt = await inquiry.retain(gateway)
        assert receipt.receipt_id
        reopened = ConcreteStateGateway(root=tmp_path)
        restored = await ProgramInquiry.restore(reopened, inquiry.identity)
        assert restored == inquiry
        assert await ProgramInquiry.restore(reopened, "a" * 64) is None
        assert restored.compatible_methods(observed_result=6) == ("multiply",)

    asyncio.run(run())


def test_portfolio_retains_disagreement_through_existing_state_owner(tmp_path):
    from core.state.state_gateway import ConcreteStateGateway

    portfolio = select_semantic_program_portfolio(
        proposals={"add": program("add"), "multiply": program("mul")},
        provenance={"add": "a" * 64, "multiply": "b" * 64},
        public_inputs=(2, 2), observation_sha256="c" * 64, incumbent="add",
    )

    async def run():
        gateway = ConcreteStateGateway(root=tmp_path, governance_decide=lambda **_: True)
        receipts = await portfolio.retain_inquiries(gateway)
        assert receipts
        reopened = ConcreteStateGateway(root=tmp_path)
        for receipt in receipts:
            restored = await ProgramInquiry.restore(reopened, receipt.key)
            assert restored.to_dict()["observation_required"] is True
        assert portfolio.decision.selected == "add"

    asyncio.run(run())
