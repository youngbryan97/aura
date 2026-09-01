from __future__ import annotations

from pathlib import Path

from tools.run_induced_neural_procedure_canary import task_set
from tools.run_induced_neural_procedure_decode_canary import (
    ARMS,
    _arm_order,
    _induction_basis,
    _states,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASIS = (
    REPO_ROOT
    / "artifacts/closeout/latent_cortex/"
    "induced_neural_procedure_canary_20260831"
)


def test_verified_basis_reconstructs_the_frozen_induced_program() -> None:
    identity, program = _induction_basis(BASIS)

    assert identity["program_sha"] == program.sha()
    assert identity["support_count"] == 16
    assert program.describe() == "idiv(add(in0, in1), in2)"


def test_state_arms_preserve_treatment_and_disrupt_causal_controls() -> None:
    _identity, program = _induction_basis(BASIS)
    tasks = task_set(8, seed=2026084802)

    for task in tasks:
        workflow, states = _states(program, tuple(int(value) for value in task.inputs))
        treatment = states["treatment"]
        lesion = states["coefficient_lesion"]
        wrong_input = states["matched_wrong_input"]

        assert workflow
        assert treatment is not None
        assert tuple(treatment.semantic_result.values()) == (task.output,)
        assert lesion is None or lesion.semantic_result != treatment.semantic_result
        assert wrong_input is not None
        assert wrong_input.semantic_result != treatment.semantic_result
        assert wrong_input.objective_sha256 != treatment.objective_sha256


def test_arm_order_is_complete_deterministic_and_counterbalanced() -> None:
    first = _arm_order("task-a")
    second = _arm_order("task-a")

    assert first == second
    assert set(first) == set(ARMS)
    assert len(first) == len(ARMS)
    assert len({_arm_order(f"task-{index}") for index in range(32)}) > 1
