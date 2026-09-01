from __future__ import annotations

from tools.run_semantic_neural_composition_canary import _reference
from tools.run_semantic_neural_composition_decode_canary import (
    ARMS,
    _arm_order,
    _cohort,
    _state_arms,
    _wrong_state_index,
)


def test_arm_order_is_complete_deterministic_and_rotated() -> None:
    orders = [_arm_order(f"task-{index}") for index in range(24)]

    assert all(set(order) == set(ARMS) and len(order) == len(ARMS) for order in orders)
    assert orders == [_arm_order(f"task-{index}") for index in range(24)]
    assert len({order[0] for order in orders}) == len(ARMS)


def test_all_causal_states_share_protocol_but_not_result() -> None:
    documents = _cohort(2026083201, 2)
    all_states = [_state_arms(document) for document in documents]
    document = documents[0]
    states = all_states[0]
    expected = _reference(document)

    treatment = states["treatment"]
    wrong = all_states[_wrong_state_index(all_states, 0)]["treatment"]
    assert treatment is not None and treatment.semantic_result == expected
    assert wrong is not None and wrong.semantic_result != expected
    for arm in ("additive_lesion", "multiplicative_lesion"):
        assert states[arm] is None or states[arm].semantic_result != expected
    retained = [state for state in states.values() if state is not None]
    assert {state.report for state in retained} == {tuple(document["report"])}
    assert all(state.receipt()["teacher_available"] is False for state in retained)
