from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.learning.recurrent_sft_behavior_canaries import (
    RecurrentSFTBehaviorCanaryError,
    build_generated_behavior_canaries,
    build_generated_behavior_generation_contract,
    generated_behavior_verdict,
    grade_generated_behavior_text,
)
from core.learning.recurrent_sft_evaluation import (
    build_regression_canary_rows,
    regression_canary_verdict,
)
from core.learning.recurrent_sft_falsification import (
    ALL_ARMS,
    BASE_ARM,
    CONTROL_ARMS,
    TRAINED_ARM,
    build_falsification_verdict,
    sha256_json,
)
from core.learning.recurrent_sft_kernel_probe import build_kernel_probe_spec
from core.learning.recurrent_sft_sampling import FAMILY_BALANCED_SAMPLER
from core.learning.structured_sft_research_state import CHECKPOINT_SCHEMA
from tools import verify_recurrent_sft_falsification as verifier


def _contract(tmp_path: Path) -> tuple[dict, dict]:
    profile = tmp_path / "evaluator.sb"
    profile.write_text("(version 1)\n(deny default)\n")
    source_closure = {
        "schema": "source",
        "files": [],
        "closure_sha256": "1" * 64,
    }
    custody = {"candidate": {}, "evaluator": {}, "custody": {}}
    command = [
        "/usr/bin/sandbox-exec",
        "-f",
        str(profile),
        "/usr/bin/python3",
        "evaluator.py",
    ]
    environment = {"PYTHONHASHSEED": "0"}
    targets = {}
    for role in (
        "evaluator_read",
        "production_write",
        "resident_read",
        "training_write",
    ):
        target = tmp_path / role
        target.write_bytes(role.encode("ascii"))
        targets[role] = target
    kernel_probe = build_kernel_probe_spec(
        sandbox_executable=Path("/usr/bin/sandbox-exec"),
        profile=profile,
        python=Path("/usr/bin/python3"),
        targets=targets,
    )
    body = {
        "source_closure": source_closure,
        "authority_sha256": "2" * 64,
        "model_identity_sha256": "3" * 64,
        "execution_spec_sha256": "4" * 64,
        "custody_binding_sha256": "5" * 64,
        "custody_bindings": custody,
        "network": "kernel_denied",
        "process_fork": "kernel_denied",
        "evaluator_access": True,
        "training_write_access": False,
        "resident_checkpoint_access": False,
        "production_write_access": False,
        "resume_contract": "none",
        "profile_path": str(profile),
        "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
        "sandbox_executable_sha256": hashlib.sha256(
            Path("/usr/bin/sandbox-exec").read_bytes()
        ).hexdigest(),
        "command": command,
        "command_sha256": sha256_json(command),
        "environment": environment,
        "environment_sha256": sha256_json(environment),
        "kernel_probe": kernel_probe,
        "kernel_probe_path": str(tmp_path / "kernel_probe.json"),
    }
    contract = {**body, "contract_sha256": sha256_json(body)}
    report = {
        "containment_contract_sha256": contract["contract_sha256"],
        "source_closure": source_closure,
        "authority_sha256": "2" * 64,
        "model_identity_sha256": "3" * 64,
        "execution_spec_sha256": "4" * 64,
        "custody_binding_sha256": "5" * 64,
        "custody": custody,
    }
    return contract, report


def test_contract_replays_execution_and_custody_bindings(tmp_path: Path) -> None:
    contract, report = _contract(tmp_path)
    verifier._verify_contract(contract, report)

    report["custody_binding_sha256"] = "6" * 64
    with pytest.raises(
        verifier.RecurrentSFTFalsificationVerificationError,
        match="contract_invalid",
    ):
        verifier._verify_contract(contract, report)


def test_binding_rejects_symlink(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"bound")
    link = tmp_path / "link"
    link.symlink_to(artifact)
    with pytest.raises(
        verifier.RecurrentSFTFalsificationVerificationError,
        match="symlink_rejected",
    ):
        verifier._verify_binding(
            {
                "path": str(link),
                "sha256": hashlib.sha256(b"bound").hexdigest(),
                "size_bytes": 5,
            },
            role="test",
        )


def test_receipt_requires_exact_contained_command(tmp_path: Path) -> None:
    contract, _report = _contract(tmp_path)
    receipt = {
        "returncode": 0,
        "timed_out": False,
        "process_group_empty": True,
        "duration_s": 1.0,
        "status": "passed",
        "restart_count": 0,
        "containment_verified": True,
        "command": contract["command"],
        "executed_command": contract["command"],
        "command_sha256": contract["command_sha256"],
        "lineage_empty": True,
    }
    verifier._verify_receipt(receipt, contract=contract)

    receipt["executed_command"] = ["/bin/echo", "substituted"]
    with pytest.raises(
        verifier.RecurrentSFTFalsificationVerificationError,
        match="detached_receipt_invalid",
    ):
        verifier._verify_receipt(receipt, contract=contract)


def _transfer_observations(*, loss: float, improve: bool) -> list[dict]:
    rows = []
    for index in range(8):
        rows.append(
            {
                "example_id": f"{index + 1:064x}",
                "family": "logic" if index < 4 else "tool",
                "loss": loss + index * 0.001,
                "target_top1": (
                    [True, True, True, True] if improve else [False, True, False, True]
                ),
                "generated_correct": improve,
            }
        )
    return rows


def _likelihood_observations(*, loss: float) -> list[dict]:
    return [
        {
            "example_id": row["_meta"]["example_id"],
            "family": row["_meta"]["family"],
            "loss": loss,
            "target_top1": [True, False],
            "generated_correct": None,
        }
        for row in build_regression_canary_rows()
    ]


def _behavior_observations(
    *,
    arm: str,
    adapter_fingerprint: str | None,
    contract_sha256: str,
) -> list[dict]:
    rows = []
    for case in build_generated_behavior_canaries():
        if case["name"] == "runtime_perspective_boundary":
            text = "My runtime is software. Bryan is physical. I am not him."
        elif case["name"] == "approval_bypass_email":
            text = "I will not bypass approval."
        else:
            text = " ".join(group[0] for group in case["required_groups"])
        tokens = [1, 2, 3]
        rows.append(
            {
                "case_id": case["case_id"],
                "family": case["family"],
                "arm": arm,
                "prompt_sha256": hashlib.sha256(case["prompt"].encode("utf-8")).hexdigest(),
                "generation_contract_sha256": contract_sha256,
                "engine_ok": True,
                "engine_reason": "",
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "tokens": tokens,
                "token_count": len(tokens),
                "tokens_sha256": hashlib.sha256(
                    json.dumps(tokens, separators=(",", ":")).encode("ascii")
                ).hexdigest(),
                "decode_termination": "eos",
                "fallback_used": False,
                "adapter_active": arm == TRAINED_ARM,
                "adapter_fingerprint": adapter_fingerprint,
                "params_before": "c" * 64,
                "params_after": "c" * 64,
                "params_unchanged": True,
                "grade": grade_generated_behavior_text(case, text),
            }
        )
    return rows


def _decision_report() -> dict:
    transfer = {
        BASE_ARM: _transfer_observations(loss=1.0, improve=False),
        TRAINED_ARM: _transfer_observations(loss=0.6, improve=True),
        "sham_labels": _transfer_observations(loss=0.95, improve=False),
        "shuffled_traces": _transfer_observations(loss=0.9, improve=False),
        "syntax_only": _transfer_observations(loss=0.85, improve=False),
    }
    base_likelihood = _likelihood_observations(loss=1.0)
    trained_likelihood = _likelihood_observations(loss=0.99)
    contract = build_generated_behavior_generation_contract(execution_spec_sha256="e" * 64)
    trained_fingerprint = "b" * 64
    behavior = {
        BASE_ARM: _behavior_observations(
            arm=BASE_ARM,
            adapter_fingerprint=None,
            contract_sha256=contract["contract_sha256"],
        ),
        TRAINED_ARM: _behavior_observations(
            arm=TRAINED_ARM,
            adapter_fingerprint=trained_fingerprint,
            contract_sha256=contract["contract_sha256"],
        ),
    }
    behavior_result = generated_behavior_verdict(
        behavior[BASE_ARM],
        behavior[TRAINED_ARM],
        expected_generation_contract_sha256=contract["contract_sha256"],
        expected_trained_adapter_fingerprint=trained_fingerprint,
    )
    return {
        "execution_spec_sha256": "e" * 64,
        "observations": transfer,
        "falsification": build_falsification_verdict(transfer),
        "regression_likelihood_canary_observations": {
            BASE_ARM: base_likelihood,
            TRAINED_ARM: trained_likelihood,
        },
        "regression_likelihood_canary_verdict": regression_canary_verdict(
            base_likelihood,
            trained_likelihood,
        ),
        "generated_behavior_canary_count": len(build_generated_behavior_canaries()),
        "generated_behavior_generation_contract": contract,
        "generated_behavior_generation_contract_sha256": contract["contract_sha256"],
        "generated_behavior_canary_observations": behavior,
        "generated_behavior_canary_verdict": behavior_result,
        "generated_behavior_regression_tested": True,
        "adapter_fingerprints": {
            BASE_ARM: None,
            TRAINED_ARM: trained_fingerprint,
            **{arm: "d" * 64 for arm in CONTROL_ARMS},
        },
        "ordinary_lexical_hashes": {arm: "f" * 64 for arm in ALL_ARMS},
        "ordinary_lexical_invariance_proven": True,
        "base_weights_unchanged": True,
        "all_small_checkpoint_gates_passed": True,
        "status": "small_checkpoint_transfer_with_all_regression_gates_passed",
        "production_effect": False,
        "promotion_allowed": False,
        "claims_not_supported": [
            "broad_reasoning_gain",
            "frontier_performance",
            "resident_32b_result",
            "production_promotion",
            "wow_signal",
        ],
    }


def test_decision_replay_includes_generated_behavior_gate() -> None:
    report = _decision_report()
    replay = verifier._verify_decisions(report)
    assert replay["all_small_checkpoint_gates_passed"] is True
    assert replay["generated_behavior_canaries"]["passed"] is True


def test_decision_replay_rejects_forged_generated_grade() -> None:
    report = _decision_report()
    report["generated_behavior_canary_observations"][TRAINED_ARM][0]["grade"]["passed"] = False
    with pytest.raises(
        RecurrentSFTBehaviorCanaryError,
        match="grade_replay_mismatch",
    ):
        verifier._verify_decisions(report)


def test_generated_behavior_failure_forces_final_gate_failure() -> None:
    report = _decision_report()
    case = build_generated_behavior_canaries()[0]
    observation = report["generated_behavior_canary_observations"][TRAINED_ARM][0]
    text = "unsupported answer"
    observation["text"] = text
    observation["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    observation["grade"] = grade_generated_behavior_text(case, text)
    report["generated_behavior_canary_verdict"] = generated_behavior_verdict(
        report["generated_behavior_canary_observations"][BASE_ARM],
        report["generated_behavior_canary_observations"][TRAINED_ARM],
        expected_generation_contract_sha256=report["generated_behavior_generation_contract_sha256"],
        expected_trained_adapter_fingerprint=report["adapter_fingerprints"][TRAINED_ARM],
    )
    report["all_small_checkpoint_gates_passed"] = False
    report["status"] = "small_checkpoint_transfer_not_proven"
    replay = verifier._verify_decisions(report)
    assert replay["all_small_checkpoint_gates_passed"] is False
    assert replay["generated_behavior_canaries"]["right_to_wrong"] == 1


def test_reference_initialization_binds_balanced_checkpoint_and_report() -> None:
    initial_sha256 = "a" * 64
    authority = {"trainer": {"sampler": FAMILY_BALANCED_SAMPLER}}
    trained = {"initial_adapter_sha256": initial_sha256}
    checkpoint = {"initial_adapter_sha256": initial_sha256}

    assert (
        verifier._verify_reference_initialization(
            trained,
            checkpoint,
            authority=authority,
        )
        == initial_sha256
    )

    checkpoint["initial_adapter_sha256"] = "b" * 64
    with pytest.raises(
        verifier.RecurrentSFTFalsificationVerificationError,
        match="reference_initialization_mismatch",
    ):
        verifier._verify_reference_initialization(
            trained,
            checkpoint,
            authority=authority,
        )


def test_reference_initialization_rejects_unexpected_legacy_commitment() -> None:
    with pytest.raises(
        verifier.RecurrentSFTFalsificationVerificationError,
        match="unexpected_reference_initialization",
    ):
        verifier._verify_reference_initialization(
            {"initial_adapter_sha256": "a" * 64},
            {"initial_adapter_sha256": None},
            authority={"trainer": {"sampler": "legacy_sampler.v1"}},
        )


def test_reference_checkpoint_state_strips_exact_completion_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}
    monkeypatch.setattr(
        verifier,
        "validate_checkpoint_state",
        lambda state: observed.update(state),
    )
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "adapter": {"path": "adapter"},
        "optimizer": {"path": "optimizer"},
        "checkpoint_id": "step-00000001-test",
        "created_unix": 1.0,
        "state_field": "bound",
    }

    assert verifier._validate_reference_checkpoint_state(checkpoint) == {
        "state_field": "bound"
    }
    assert observed == {"state_field": "bound"}

    checkpoint["schema"] = "forged"
    with pytest.raises(
        verifier.RecurrentSFTFalsificationVerificationError,
        match="reference_checkpoint_schema_invalid",
    ):
        verifier._validate_reference_checkpoint_state(checkpoint)


def test_decision_replay_rejects_generation_contract_or_arm_drift() -> None:
    report = _decision_report()
    report["generated_behavior_generation_contract"]["decode"]["temperature"] = 0.2
    with pytest.raises(
        verifier.RecurrentSFTFalsificationVerificationError,
        match="behavior_contract_invalid",
    ):
        verifier._verify_decisions(report)

    report = _decision_report()
    report["generated_behavior_canary_observations"][BASE_ARM][0]["arm"] = TRAINED_ARM
    with pytest.raises(
        RecurrentSFTBehaviorCanaryError,
        match="observation_invalid",
    ):
        verifier._verify_decisions(report)
