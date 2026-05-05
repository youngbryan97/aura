from core.environment.action_gateway import EnvironmentActionGateway
from core.environment.command import ActionIntent
from core.environment.modal import ModalState
from core.environment.simulation import SimulationBundle


def test_gateway_blocks_empty_unknown_and_modal_actions():
    gateway = EnvironmentActionGateway(legal_actions={"observe", "resolve_modal"})
    assert not gateway.approve(ActionIntent(name=""), context_id="ctx").approved
    assert not gateway.approve(ActionIntent(name="delete"), context_id="ctx").approved
    modal = ModalState(kind="prompt", text="Press return", legal_responses={" "}, safe_default=" ")
    assert not gateway.approve(ActionIntent(name="observe"), modal_state=modal, context_id="ctx").approved
    assert gateway.approve(ActionIntent(name="resolve_modal"), modal_state=modal, context_id="ctx").approved


def test_gateway_blocks_uncertain_irreversible_and_repeated_failures():
    gateway = EnvironmentActionGateway()
    intent = ActionIntent(name="submit", risk="irreversible", requires_authority=True)
    assert not gateway.approve(intent, uncertainty=0.9, authority_receipt_id="r", context_id="ctx").approved
    gateway.record_failure("move", "ctx")
    gateway.record_failure("move", "ctx")
    assert not gateway.approve(ActionIntent(name="move"), context_id="ctx").approved


def test_gateway_uses_simulator_worst_case_risk():
    gateway = EnvironmentActionGateway()
    intent = ActionIntent(name="use", risk="risky")
    sim = SimulationBundle(intent, [], worst_case_risk=0.9, expected_value=-0.5, uncertainty=0.4)
    assert not gateway.approve(intent, simulation=sim, context_id="ctx").approved
