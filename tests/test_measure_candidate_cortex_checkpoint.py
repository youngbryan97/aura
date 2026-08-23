from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.learning.candidate_cortex_measurement import CandidateCortexMeasurementError
from core.learning.candidate_cortex_training import document_sha256
from tools import measure_candidate_cortex_checkpoint as measure


def _write_adapter_config(path: Path, **changes: object) -> None:
    config = {
        "fine_tune_type": "lora",
        "num_layers": -1,
        "lora_parameters": {
            "dropout": 0.0,
            "keys": ["self_attn.q_proj", "mlp.down_proj"],
            "rank": 32,
            "scale": 20.0,
        },
    }
    config.update(changes)
    path.mkdir()
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")


def test_adapter_spec_preserves_exact_mapping_contract(tmp_path: Path) -> None:
    root = tmp_path / "adapter"
    _write_adapter_config(root)

    layers, parameters, use_dora = measure._adapter_spec(root)

    assert layers == -1
    assert parameters == {
        "dropout": 0.0,
        "keys": ["self_attn.q_proj", "mlp.down_proj"],
        "rank": 32,
        "scale": 20.0,
    }
    assert use_dora is False


@pytest.mark.parametrize(
    "changes",
    [
        {"fine_tune_type": "full"},
        {"num_layers": True},
        {"lora_parameters": {"dropout": 0.0, "rank": 32, "scale": 20.0}},
        {
            "lora_parameters": {
                "dropout": 0.0,
                "keys": ["self_attn.q_proj", "self_attn.q_proj"],
                "rank": 32,
                "scale": 20.0,
            }
        },
    ],
)
def test_adapter_spec_rejects_ambiguous_geometry(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    root = tmp_path / "adapter"
    _write_adapter_config(root, **changes)
    with pytest.raises(CandidateCortexMeasurementError):
        measure._adapter_spec(root)


def test_jsonl_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"messages":[],"messages":[]}\n', encoding="utf-8")
    with pytest.raises(
        CandidateCortexMeasurementError, match="measurement_input_duplicate_key"
    ):
        measure._jsonl(path)


def test_loss_pairing_is_order_independent_but_identity_exact() -> None:
    baseline = [
        {"sample_id": "b", "domain": "voice", "nll_sum": 2.0, "tokens": 2},
        {"sample_id": "a", "domain": "identity", "nll_sum": 3.0, "tokens": 3},
    ]
    candidate = [
        {"sample_id": "a", "domain": "identity", "nll_sum": 1.0, "tokens": 3},
        {"sample_id": "b", "domain": "voice", "nll_sum": 1.5, "tokens": 2},
    ]

    rows = measure._pair_losses(baseline, candidate)

    assert [row["sample_id"] for row in rows] == ["a", "b"]
    assert rows[0]["baseline_nll_sum"] == 3.0
    assert rows[0]["candidate_nll_sum"] == 1.0


def test_behavior_pairing_rejects_evaluator_drift() -> None:
    baseline = [
        {
            "probe_id": "p",
            "family": "grounding",
            "passed": True,
            "evaluator_sha256": "a" * 64,
        }
    ]
    candidate = [
        {
            "probe_id": "p",
            "family": "grounding",
            "passed": True,
            "evaluator_sha256": "b" * 64,
        }
    ]
    with pytest.raises(
        CandidateCortexMeasurementError, match="measurement_behavior_pairing_invalid"
    ):
        measure._pair_behaviors(baseline, candidate)


def _plan() -> dict[str, object]:
    return {
        "plan_sha256": "1" * 64,
        "model": {"descriptor_sha256": "2" * 64},
        "dataset": {"receipt_sha256": "3" * 64},
    }


def test_baseline_document_is_bound_and_reusable() -> None:
    plan = _plan()
    baseline = measure._baseline_document(
        plan=plan,
        contract_sha256="4" * 64,
        persona=[{"sample_id": "p", "nll_sum": 1.0, "tokens": 2}],
        retention=[{"sample_id": "r", "nll_sum": 2.0, "tokens": 3}],
        behavior=[{"probe_id": "b", "passed": True}],
    )

    assert (
        measure._validate_baseline_document(
            baseline, plan=plan, contract_sha256="4" * 64
        )["baseline_sha256"]
        == baseline["baseline_sha256"]
    )


def test_baseline_document_rejects_tampering_and_contract_drift() -> None:
    plan = _plan()
    baseline = measure._baseline_document(
        plan=plan,
        contract_sha256="4" * 64,
        persona=[{"sample_id": "p", "nll_sum": 1.0, "tokens": 2}],
        retention=[{"sample_id": "r", "nll_sum": 2.0, "tokens": 3}],
        behavior=[{"probe_id": "b", "passed": True}],
    )
    baseline["persona"][0]["nll_sum"] = 9.0
    with pytest.raises(
        CandidateCortexMeasurementError, match="measurement_baseline_identity_invalid"
    ):
        measure._validate_baseline_document(
            baseline, plan=plan, contract_sha256="4" * 64
        )

    baseline["baseline_sha256"] = document_sha256(
        {key: value for key, value in baseline.items() if key != "baseline_sha256"}
    )
    with pytest.raises(
        CandidateCortexMeasurementError, match="measurement_baseline_identity_invalid"
    ):
        measure._validate_baseline_document(
            baseline, plan=plan, contract_sha256="5" * 64
        )
