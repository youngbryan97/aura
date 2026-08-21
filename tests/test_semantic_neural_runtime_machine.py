from __future__ import annotations

import pytest

from core.learning import semantic_neural_machine as measured_module
from core.learning.frontier_process_supervision import frontier_process_task_battery
from core.learning.public_frontier_action_compiler import compile_public_frontier_actions
from core.learning.semantic_neural_machine import SemanticNeuralMachine
from core.learning.semantic_neural_runtime_machine import SemanticNeuralRuntimeMachine


def _execute(machine, task):
    state = (0,) * 11
    receipts = []
    for action in compile_public_frontier_actions(task.prompt, task.family).values:
        transition = machine.transition(state, action)
        state = transition.next_state
        receipts.append(transition.receipt())
    return state, tuple(receipts)


def test_runtime_backend_is_receipt_exact_to_measured_mlx_tissue() -> None:
    tasks = frontier_process_task_battery(
        ("coding", "calibration", "misleading_premise", "scientific_inference"),
        (1, 2, 3),
        3,
        seed=2026082191,
    )
    measured = SemanticNeuralMachine()
    runtime = SemanticNeuralRuntimeMachine()

    for task in tasks:
        assert _execute(runtime, task) == _execute(measured, task)


def test_runtime_backend_submits_no_transition_scalar_work_to_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine = SemanticNeuralRuntimeMachine()
    task = frontier_process_task_battery(("calibration",), (3,), 1, seed=2026082192)[0]

    monkeypatch.setattr(
        measured_module.mx,
        "array",
        lambda *_args, **_kwargs: pytest.fail(
            "runtime scalar transition submitted work to the resident MLX queue"
        ),
    )

    state, receipts = _execute(machine, task)
    assert state[-1] == 1
    assert receipts
