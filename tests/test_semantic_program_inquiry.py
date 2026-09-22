"""Questions distinguish hypotheses without pretending their answer is known."""

import pytest
import asyncio
import json
from dataclasses import replace

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_inquiry import (
    ObservedProgramInquiry, ProgramInquiry, plan_program_inquiries,
)
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


def test_inquiry_identity_binds_request_not_only_programs_and_probe():
    proposals = {"add": program("add"), "multiply": program("mul")}
    first = plan_program_inquiries(proposals, [(3, 2)], fuel=100_000,
                                   source_sha256="a" * 64)[0]
    second = plan_program_inquiries(proposals, [(3, 2)], fuel=100_000,
                                    source_sha256="b" * 64)[0]
    assert first.identity != second.identity
    assert ProgramInquiry.from_dict(json.loads(json.dumps(first.to_dict()))) == first
    assert first.to_dict()["schema"] == "aura.semantic_program_inquiry.v2"
    portfolio = select_semantic_program_portfolio(
        proposals=proposals, incumbent="add",
        provenance={"add": "a" * 64, "multiply": "b" * 64},
        public_inputs=(2, 2), observation_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="different source request"):
        portfolio.reconcile_inquiry(second, observed_result=6,
                                    origin="user-example", ref="turn-1")


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


def test_inquiry_evidence_uses_shared_source_deduplication_and_probe_scope():
    inquiry = plan_program_inquiries(
        {"add": program("add"), "multiply": program("mul")}, [(3, 2)], fuel=100_000
    )[0]
    packets = inquiry.evidence_for(observed_result=5, origin="observed", ref="receipt-1")
    assert packets["add"].strength == 1
    assert packets["multiply"].strength == 0
    replay = inquiry.evidence_for(observed_result=5, origin="observed", ref="receipt-1")
    assert packets["add"].fuse(replay["add"]).mass == 1
    with pytest.raises(ValueError, match="different subjects"):
        packets["add"].fuse(packets["multiply"])


def test_inquiry_rejects_changed_predictions_even_when_probe_hash_matches():
    inquiry = plan_program_inquiries(
        {"add": program("add"), "multiply": program("mul")}, [(3, 2)], fuel=100_000
    )[0]
    payload = inquiry.to_dict()
    payload["predictions"]["add"] = '["integer", 9]'
    with pytest.raises(ValueError, match="digest"):
        ProgramInquiry.from_dict(payload)


def test_observed_feedback_changes_selection_without_an_evaluation_key():
    portfolio = select_semantic_program_portfolio(
        proposals={"add": program("add"), "multiply": program("mul")},
        provenance={"add": "a" * 64, "multiply": "b" * 64},
        public_inputs=(2, 2), observation_sha256="c" * 64, incumbent="add",
    )
    inquiry = plan_program_inquiries(dict(portfolio.proposals), [(3, 2)], fuel=100_000,
                                    source_sha256=portfolio.source_sha256)[0]
    assert portfolio.reconcile_inquiry(
        inquiry, observed_result=5, origin="user-example", ref="one"
    ).selected == "add"
    assert portfolio.reconcile_inquiry(
        inquiry, observed_result=6, origin="user-example", ref="two"
    ).selected == "multiply"
    assert portfolio.reconcile_inquiry(
        inquiry, observed_result=99, origin="user-example", ref="three"
    ) is None
    assert len(portfolio.proposals) == 2


def test_feedback_cannot_be_applied_to_changed_programs():
    portfolio = select_semantic_program_portfolio(
        proposals={"add": program("add"), "multiply": program("mul")},
        provenance={"add": "a" * 64, "multiply": "b" * 64},
        public_inputs=(2, 2), observation_sha256="c" * 64, incumbent="add",
    )
    wrong = plan_program_inquiries(
        {"add": program("sub"), "multiply": program("mul")}, [(3, 2)], fuel=100_000,
        source_sha256=portfolio.source_sha256,
    )[0]
    with pytest.raises(ValueError, match="different programs"):
        portfolio.reconcile_inquiry(wrong, observed_result=6, origin="observed", ref="one")


def test_selection_replays_predictions_instead_of_trusting_bound_names():
    portfolio = select_semantic_program_portfolio(
        proposals={"add": program("add"), "multiply": program("mul")},
        provenance={"add": "a" * 64, "multiply": "b" * 64},
        public_inputs=(2, 2), observation_sha256="c" * 64, incumbent="add",
    )
    inquiry = plan_program_inquiries(dict(portfolio.proposals), [(3, 2)], fuel=100_000,
                                    source_sha256=portfolio.source_sha256)[0]
    forged = replace(inquiry, predictions=(("add", '["integer", 6]'),
                                           ("multiply", '["integer", 5]')))
    with pytest.raises(ValueError, match="checked program execution"):
        portfolio.reconcile_inquiry(forged, observed_result=6, origin="observed", ref="one")


def test_history_does_not_resurrect_previously_refuted_programs():
    portfolio = select_semantic_program_portfolio(
        proposals={"add": program("add"), "multiply": program("mul"), "sub": program("sub")},
        provenance={"add": "a" * 64, "multiply": "b" * 64, "sub": "d" * 64},
        public_inputs=(2, 2), observation_sha256="c" * 64, incumbent="add",
    )
    questions = plan_program_inquiries(dict(portfolio.proposals), [(3, 2), (2, 2)],
                                       fuel=100_000, source_sha256=portfolio.source_sha256)
    by_inputs = {q.inputs: q for q in questions}
    history = [(by_inputs[3, 2], 6, "observed", "first"),
               (by_inputs[2, 2], 4, "observed", "second")]
    assert portfolio.reconcile_inquiries(history).selected == "multiply"
    assert portfolio.reconcile_inquiries(tuple(reversed(history))).selected == "multiply"
    assert portfolio.reconcile_inquiries(history + history).selected == "multiply"
    assert portfolio.reconcile_inquiries(history + [(by_inputs[3, 2], 5, "observed", "third")]) is None
    with pytest.raises(ValueError, match="conflicting feedback"):
        portfolio.reconcile_inquiries(history + [(by_inputs[3, 2], 5, "observed", "first")])
    assert portfolio.reconcile_inquiries([]) is portfolio.decision


def test_observed_feedback_survives_reopen_and_revises_the_portfolio(tmp_path):
    from core.state.state_gateway import ConcreteStateGateway

    portfolio = select_semantic_program_portfolio(
        proposals={"add": program("add"), "multiply": program("mul")},
        provenance={"add": "a" * 64, "multiply": "b" * 64},
        public_inputs=(2, 2), observation_sha256="c" * 64, incumbent="add",
    )
    inquiry = portfolio.plan_inquiries()[0]
    prediction = json.loads(dict(inquiry.predictions)["multiply"])[1]

    async def run():
        writer = ConcreteStateGateway(root=tmp_path, governance_decide=lambda **_: True)
        await inquiry.retain(writer)
        observed = ObservedProgramInquiry(inquiry, prediction, "user-example", "turn-1")
        await observed.retain(writer)
        reader = ConcreteStateGateway(root=tmp_path)
        restored = await ObservedProgramInquiry.restore_all(reader)
        assert restored == (observed,)
        assert (await portfolio.reconcile_retained_inquiries(reader)).selected == "multiply"
        return reader

    asyncio.run(run())


def test_retained_conflicting_outcomes_do_not_choose_a_candidate(tmp_path):
    from core.state.state_gateway import ConcreteStateGateway

    portfolio = select_semantic_program_portfolio(
        proposals={"add": program("add"), "multiply": program("mul")},
        provenance={"add": "a" * 64, "multiply": "b" * 64},
        public_inputs=(2, 2), observation_sha256="c" * 64, incumbent="add",
    )
    inquiry = portfolio.plan_inquiries()[0]

    async def run():
        gateway = ConcreteStateGateway(root=tmp_path, governance_decide=lambda **_: True)
        await inquiry.retain(gateway)
        await ObservedProgramInquiry(inquiry, 5, "user-example", "turn-1").retain(gateway)
        await ObservedProgramInquiry(inquiry, 6, "user-example", "turn-1").retain(gateway)
        reopened = ConcreteStateGateway(root=tmp_path)
        with pytest.raises(ValueError, match="conflicting feedback"):
            await portfolio.reconcile_retained_inquiries(reopened)

    asyncio.run(run())


def test_observed_record_rejects_changed_result_and_identity():
    inquiry = plan_program_inquiries(
        {"add": program("add"), "multiply": program("mul")}, [(3, 2)], fuel=100_000
    )[0]
    observed = ObservedProgramInquiry(inquiry, 5, "user-example", "turn-1")
    assert ObservedProgramInquiry.from_dict(json.loads(json.dumps(observed.to_dict()))) == observed
    changed = observed.to_dict()
    changed["observed_result"] = '["integer", 6]'
    with pytest.raises(ValueError, match="digest or bindings"):
        ObservedProgramInquiry.from_dict(changed)
