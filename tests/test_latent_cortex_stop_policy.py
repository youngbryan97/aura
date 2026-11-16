"""Calibrated recurrent stopping, workload proof, and live controller wiring."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.engine import LatentCortexEngine  # noqa: E402
from core.brain.llm.latent_cortex.epistemic_state import (  # noqa: E402
    OperationKind,
)
from core.brain.llm.latent_cortex.recurrence import (  # noqa: E402
    HaltingController,
)
from core.brain.llm.latent_cortex.stop_gate import (  # noqa: E402
    LEARNED,
    StopContext,
    StopGateRuntime,
    build_stop_gate_receipt,
    validate_stop_gate_receipt,
)
from core.brain.llm.latent_cortex.types import (  # noqa: E402
    BranchConfig,
    ComputeBudget,
    CortexConfig,
    LatentOptConfig,
    RecurrenceConfig,
    WorkspaceConfig,
)
from core.learning.stop_policy import (  # noqa: E402
    MAX_STOP_ARTIFACT_BYTES,
    STOP_FEATURE_NAMES,
    StopPolicyHead,
    VerifiedStopExample,
    VerifiedStopTrajectory,
    canonical_sha256,
    certify_stop_workload,
    fit_stop_policy_head,
)
from core.learning.update_acceptance import (  # noqa: E402
    VerifiedTransitionExample,
    fit_update_acceptance_head,
)
from tests.fixtures.action_calibration import (  # noqa: E402
    certified_action_snapshot,
)

HIDDEN = 32
PROMPT = [5, 9, 17, 3, 42, 7, 11, 23, 2, 88]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _features(*, should_stop: bool, step_fraction: float = 0.5):
    if should_stop:
        values = {
            "step_fraction": step_fraction,
            "residual": 0.04,
            "residual_contraction_ratio": 0.35,
            "quality_probability": 0.94,
            "quality_uncertainty": 0.12,
            "evidence_improvement": 0.20,
            "verifier_score": 0.95,
            "verifier_delta": 0.02,
            "policy_uncertainty": 0.10,
            "expected_gain_lcb": 0.01,
            "expected_cost_ucb": 0.20,
            "expected_net_value": -0.19,
            "budget_remaining_fraction": 0.55,
            "proposal_accepted": 1.0,
            "quality_measured": 1.0,
            "evoc_measured": 1.0,
            "verifier_available": 1.0,
        }
    else:
        values = {
            "step_fraction": step_fraction,
            "residual": 0.42,
            "residual_contraction_ratio": 0.88,
            "quality_probability": 0.58,
            "quality_uncertainty": 0.84,
            "evidence_improvement": 0.03,
            "verifier_score": 0.45,
            "verifier_delta": 0.10,
            "policy_uncertainty": 0.72,
            "expected_gain_lcb": 0.48,
            "expected_cost_ucb": 0.12,
            "expected_net_value": 0.36,
            "budget_remaining_fraction": 0.82,
            "proposal_accepted": 1.0,
            "quality_measured": 1.0,
            "evoc_measured": 1.0,
            "verifier_available": 1.0,
        }
    assert tuple(values) == STOP_FEATURE_NAMES
    return values


def _examples(prefix: str, count: int):
    examples = []
    for index in range(count):
        should_stop = index % 2 == 0
        features = _features(
            should_stop=should_stop,
            step_fraction=0.65 if should_stop else 0.25,
        )
        features["residual"] += (index % 5 - 2) * 1e-4
        examples.append(
            VerifiedStopExample.from_values(
                example_id=f"{prefix}-example-{index}",
                task_id=f"{prefix}-task-{index}",
                features=features,
                should_stop=should_stop,
                verifier_receipt_sha256=_digest(f"{prefix}:receipt:{index}"),
            )
        )
    return examples


@pytest.fixture()
def fitted_head():
    return fit_stop_policy_head(_examples("train", 64), _examples("cal", 40))


@pytest.fixture(scope="module")
def tiny_model():
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=HIDDEN,
        num_hidden_layers=8,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


def test_stop_head_is_calibrated_pinned_and_stably_persisted(
    fitted_head,
    tmp_path,
):
    assert fitted_head.calibration["admitted"] is True
    assert fitted_head.calibration["false_stop_rate"] <= 0.10
    assert fitted_head.probability(_features(should_stop=True)) > fitted_head.threshold
    assert fitted_head.probability(_features(should_stop=False)) < fitted_head.threshold

    with pytest.raises(ValueError, match="end in .json"):
        fitted_head.save(tmp_path / "stop-head.npz")
    path = tmp_path / "stop-head.json"
    digest = fitted_head.save(path)
    loaded = StopPolicyHead.load(path, expected_sha256=digest)
    assert loaded.to_dict() == fitted_head.to_dict()
    with pytest.raises(ValueError, match="digest mismatch"):
        StopPolicyHead.load(path, expected_sha256="0" * 64)

    link = tmp_path / "linked.json"
    link.symlink_to(path)
    with pytest.raises(OSError):
        StopPolicyHead.load(link, expected_sha256=digest)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * (MAX_STOP_ARTIFACT_BYTES + 1))
    with pytest.raises(ValueError, match="size/type"):
        StopPolicyHead.load(
            oversized,
            expected_sha256=hashlib.sha256(oversized.read_bytes()).hexdigest(),
        )

    malformed = fitted_head.to_dict()
    malformed["means"] = {}
    malformed_path = tmp_path / "malformed.json"
    malformed_raw = json.dumps(
        malformed,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    malformed_path.write_bytes(malformed_raw)
    with pytest.raises(ValueError, match="artifact values"):
        StopPolicyHead.load(
            malformed_path,
            expected_sha256=hashlib.sha256(malformed_raw).hexdigest(),
        )


def test_stop_policy_uses_the_governed_artifact_writer(fitted_head, tmp_path, monkeypatch):
    from core.governance_context import get_active_governance
    from core.runtime.file_write_gateway import get_file_write_gateway

    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")
    gateway = get_file_write_gateway()
    write_bytes = gateway.write_bytes
    writes = []

    def write(path, payload, **kwargs):
        token = get_active_governance()
        assert token is not None and token.domain == "state_mutation"
        writes.append((path, payload, kwargs["source"]))
        write_bytes(path, payload, **kwargs)

    monkeypatch.setattr(gateway, "write_bytes", write)
    path = tmp_path / "new" / "stop-head.json"
    digest = fitted_head.save(path)
    expected = json.dumps(fitted_head.to_dict(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    assert writes == [(path, expected, "stop_policy.save")]
    assert path.read_bytes() == expected
    assert digest == hashlib.sha256(expected).hexdigest()
    assert path.stat().st_mode & 0o777 == 0o600
    assert StopPolicyHead.load(path, expected_sha256=digest).to_dict() == fitted_head.to_dict()


def test_stop_training_splits_are_task_disjoint_and_constructors_are_strict():
    shared = _examples("shared", 40)
    with pytest.raises(ValueError, match="overlap"):
        fit_stop_policy_head(shared, shared)
    with pytest.raises(ValueError, match="boolean"):
        VerifiedStopExample(
            example_id="invalid-label",
            task_id="task",
            features=tuple(0.0 for _ in STOP_FEATURE_NAMES),
            should_stop=1,  # type: ignore[arg-type]
            verifier_receipt_sha256=_digest("receipt"),
        )
    repeated_tasks = [
        replace(row, task_id=f"task-{index % 2}")
        for index, row in enumerate(_examples("repeated", 40))
    ]
    with pytest.raises(ValueError, match="eight unique tasks"):
        fit_stop_policy_head(repeated_tasks, _examples("cal-fresh", 40))


def test_heldout_workload_halts_easy_tasks_earlier_without_hard_regression(
    fitted_head,
):
    trajectories = []
    for index in range(8):
        easy = index < 4
        required_step = 2 if easy else 5
        steps = tuple(
            _features(
                should_stop=step >= required_step,
                step_fraction=step / 6.0,
            )
            for step in range(1, 7)
        )
        correct = tuple(step >= required_step for step in range(1, 7))
        trajectories.append(
            VerifiedStopTrajectory(
                task_id=f"heldout-{index}",
                difficulty="easy" if easy else "hard",
                steps=steps,
                correct_by_step=correct,
                required_step=required_step,
                verifier_receipt_sha256=_digest(f"heldout:{index}"),
            )
        )
    certificate = certify_stop_workload(fitted_head, trajectories)
    assert certificate["admitted"] is True
    assert certificate["easy_mean_step_reduction"] >= 1.0
    assert certificate["hard_premature_stops"] == 0
    assert certificate["hard_selected_accuracy"] == certificate["hard_baseline_accuracy"]
    overlapping = list(trajectories)
    overlapping[0] = replace(overlapping[0], task_id="train-task-0")
    with pytest.raises(ValueError, match="overlaps training"):
        certify_stop_workload(fitted_head, overlapping)
    with pytest.raises(ValueError, match="trajectory"):
        replace(
            trajectories[0],
            required_step=1,
        )


def _update_decision(*, should_stop: bool):
    features = {
        "anchor_distance_improvement": 0.22 if should_stop else 0.03,
        "evidence_distance_improvement": 0.18 if should_stop else 0.02,
    }
    return SimpleNamespace(
        probability=0.94 if should_stop else 0.58,
        accepted=True,
        features=features,
    )


def _context(*, measured: bool, should_stop: bool):
    return StopContext(
        action_step=1,
        max_steps=6,
        policy_uncertainty=0.10 if should_stop else 0.72,
        verifier_score=0.95 if should_stop else 0.45,
        verifier_delta=0.02 if should_stop else 0.10,
        expected_gain_lcb=0.01 if should_stop else 0.48,
        expected_cost_ucb=0.20 if should_stop else 0.12,
        quality_measured=measured,
        evoc_measured=measured,
        budget_remaining_fraction=0.55 if should_stop else 0.82,
    )


def test_live_controller_requires_measured_quality_and_evoc_before_stopping(
    fitted_head,
):
    gate = StopGateRuntime(
        mode=LEARNED,
        head=fitted_head,
        head_sha256="a" * 64,
    )
    unmeasured = gate.evaluate(
        step=2,
        residual=0.04,
        previous_residual=0.12,
        update_decision=_update_decision(should_stop=True),
        context=_context(measured=False, should_stop=True),
    )
    assert unmeasured.halt is False
    assert unmeasured.reason == "continue_unmeasured_evidence"

    controller = HaltingController(
        config=RecurrenceConfig(
            min_steps=2,
            max_steps=6,
            convergence_eps=1e-9,
        )
    )
    controller.stop_gate = gate
    state = mx.ones((1, 4, HIDDEN))
    first = controller.observe(
        0,
        state,
        residual=0.12,
        stop_context=_context(measured=True, should_stop=False),
        update_decision=_update_decision(should_stop=False),
    )
    assert first.should_halt is False
    second = controller.observe(
        1,
        state,
        residual=0.04,
        stop_context=_context(measured=True, should_stop=True),
        update_decision=_update_decision(should_stop=True),
    )
    assert second.should_halt is True
    assert second.reason == "learned_stop"
    assert controller.stop_trace[-1]["halt"] is True


def test_stop_receipt_recomputes_probabilities_and_rejects_rehashed_lies(
    fitted_head,
):
    gate = StopGateRuntime(
        mode=LEARNED,
        head=fitted_head,
        head_sha256="b" * 64,
    )
    controller = HaltingController(
        config=RecurrenceConfig(min_steps=1, max_steps=6, convergence_eps=1e-9)
    )
    controller.stop_gate = gate
    controller.residual_trail.append(0.12)
    decision = controller.observe(
        1,
        mx.ones((1, 4, HIDDEN)),
        residual=0.04,
        stop_context=_context(measured=True, should_stop=True),
        update_decision=_update_decision(should_stop=True),
    )
    assert decision.reason == "learned_stop"
    branch = SimpleNamespace(
        index=0,
        role="solver",
        halt_reason="learned_stop",
        steps=2,
        halting=controller,
    )
    update_features = {
        "anchor_distance_improvement": 0.22,
        "evidence_distance_improvement": 0.18,
    }
    update_acceptance = {
        "mode": "learned",
        "receipt_sha256": "c" * 64,
        "branches": [
            {
                "transitions": [
                    {
                        "branch_step": 1,
                        "probability": 0.94,
                        "accepted": True,
                        "features": update_features,
                    }
                ]
            }
        ],
    }
    loop_stability = {
        "receipt_sha256": "d" * 64,
        "loop_core": {"max_steps": 6},
        "branches": [
            {
                "transitions": [
                    {"branch_step": 0, "residual": 0.12},
                    {"branch_step": 1, "residual": 0.04},
                ]
            }
        ],
    }
    cognitive_action_trace = [
        {
            "decision": {
                "step_index": 1,
                "evidence": {
                    "gain_used": 0.01,
                    "cost_used": 0.20,
                    "measured": True,
                },
            },
            "state_signal": {
                "uncertainty": 0.10,
                "verifier_score": 0.95,
                "verifier_delta": 0.02,
                "budget_remaining_fraction": 0.55,
            },
        }
    ]
    receipt = build_stop_gate_receipt(
        branches=[branch],
        gate=gate,
        update_acceptance=update_acceptance,
        loop_stability=loop_stability,
        cognitive_action_trace=cognitive_action_trace,
    )
    assert receipt["head_was_causal"] is True

    forged = copy.deepcopy(receipt)
    forged["branches"][0]["decisions"][0]["probability"] = 0.01
    payload = {key: value for key, value in forged.items() if key != "receipt_sha256"}
    forged["receipt_sha256"] = canonical_sha256(payload)
    with pytest.raises(ValueError, match="differs from its head"):
        validate_stop_gate_receipt(
            forged,
            expected_gate=gate,
            update_acceptance=update_acceptance,
            loop_stability=loop_stability,
            cognitive_action_trace=cognitive_action_trace,
        )

    malformed_source = copy.deepcopy(update_acceptance)
    del malformed_source["branches"][0]["transitions"][0]["features"]["anchor_distance_improvement"]
    with pytest.raises(ValueError, match="source values"):
        validate_stop_gate_receipt(
            receipt,
            expected_gate=gate,
            update_acceptance=malformed_source,
            loop_stability=loop_stability,
            cognitive_action_trace=cognitive_action_trace,
        )


def _fitted_update_head(observed_rows):
    positive = observed_rows[0]["features"]
    negative = observed_rows[-1]["features"]

    def rows(prefix, count):
        examples = []
        for index in range(count):
            improved = index % 2 == 0
            features = dict(positive if improved else negative)
            features["proposal_residual"] += (index % 7 - 3) * 1e-5
            examples.append(
                VerifiedTransitionExample.from_values(
                    example_id=f"{prefix}-{index}",
                    features=features,
                    improved=improved,
                    verifier_receipt_sha256=_digest(f"{prefix}:update:{index}"),
                )
            )
        return examples

    return fit_update_acceptance_head(rows("train-live", 64), rows("cal-live", 40))


def _evoc_stop_head():
    def rows(prefix, count):
        examples = []
        for index in range(count):
            should_stop = index % 2 == 0
            features = _features(should_stop=True, step_fraction=0.5)
            features.update(
                {
                    "residual": 0.2,
                    "residual_contraction_ratio": 0.8,
                    "quality_probability": 0.5,
                    "quality_uncertainty": 0.5,
                    "evidence_improvement": 0.0,
                    "verifier_score": 0.0,
                    "verifier_delta": 0.0,
                    "policy_uncertainty": 0.5,
                    "expected_gain_lcb": -0.05 if should_stop else 0.45,
                    "expected_cost_ucb": 0.20 if should_stop else 0.10,
                    "expected_net_value": -0.25 if should_stop else 0.35,
                    "budget_remaining_fraction": 0.7,
                    "verifier_available": 0.0,
                }
            )
            examples.append(
                VerifiedStopExample.from_values(
                    example_id=f"{prefix}-{index}",
                    task_id=f"{prefix}-task-{index}",
                    features=features,
                    should_stop=should_stop,
                    verifier_receipt_sha256=_digest(f"{prefix}:stop:{index}"),
                )
            )
        return examples

    return fit_stop_policy_head(rows("train-evoc", 64), rows("cal-evoc", 40))


def _measured_negative_value_evidence():
    return certified_action_snapshot(
        bucket="test|none|short|s:mid|u:mid",
        cells={action: (-0.05, -0.04, -0.03, 0.20) for action in OperationKind},
    )


def _engine_config(*, update_gate, halting=None):
    return CortexConfig(
        workspace=WorkspaceConfig(n_slots=4, seed=17),
        recurrence=RecurrenceConfig(
            min_steps=2,
            max_steps=6,
            convergence_eps=1e-9,
        ),
        branches=BranchConfig(n_branches=1),
        latent_opt=LatentOptConfig(enabled=False),
        decode_max_tokens=4,
        update_gate=update_gate,
        halting=halting,
    )


def test_real_tiny_qwen_stop_policy_is_causal_under_measured_equal_evidence(
    tiny_model,
    tmp_path,
    monkeypatch,
):
    evidence, root_pem = _measured_negative_value_evidence()
    root_path = tmp_path / "action-calibration-root.pem"
    root_path.write_bytes(root_pem)
    monkeypatch.setenv(
        "AURA_RLC_ACTION_CALIBRATION_TRUST_ROOT",
        str(root_path),
    )
    observed = LatentCortexEngine(
        tiny_model,
        config=_engine_config(update_gate=None),
    ).reason(
        token_ids=PROMPT,
        budget=ComputeBudget(),
        action_policy_evidence=evidence,
    )
    assert observed.ok is True
    observed_rows = observed.receipt.update_acceptance["branches"][0]["transitions"]
    assert len(observed_rows) >= 2
    update_head = _fitted_update_head(observed_rows)
    update_path = tmp_path / "update-head.npz"
    update_digest = update_head.save(update_path)
    update_config = {
        "mode": "learned",
        "head_path": str(update_path),
        "head_sha256": update_digest,
    }
    baseline = LatentCortexEngine(
        tiny_model,
        config=_engine_config(update_gate=update_config),
    ).reason(
        token_ids=PROMPT,
        budget=ComputeBudget(),
        action_policy_evidence=evidence,
    )

    stop_head = _evoc_stop_head()
    stop_path = tmp_path / "stop-head.json"
    stop_digest = stop_head.save(stop_path)
    learned = LatentCortexEngine(
        tiny_model,
        config=_engine_config(
            update_gate=update_config,
            halting={
                "mode": "learned",
                "head_path": str(stop_path),
                "head_sha256": stop_digest,
            },
        ),
    ).reason(
        token_ids=PROMPT,
        budget=ComputeBudget(),
        action_policy_evidence=evidence,
    )

    assert baseline.ok is True and learned.ok is True
    assert learned.receipt.halting["head_was_causal"] is True
    assert learned.receipt.halting["learned_halts"] == 1
    assert learned.receipt.steps_taken < baseline.receipt.steps_taken
    assert learned.receipt.halting_reason.startswith("learned_stop")
    assert learned.receipt.budget["spent_layer_apps"] < baseline.receipt.budget["spent_layer_apps"]
    assert (
        learned.receipt.recurrent_grounding["selected_transition_count"]
        < baseline.receipt.recurrent_grounding["selected_transition_count"]
    )
