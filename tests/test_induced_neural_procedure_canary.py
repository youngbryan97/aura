from __future__ import annotations

from tools.run_induced_neural_procedure_canary import _sha, run_canary
from tools.verify_induced_neural_procedure_canary import verify


def test_powered_shape_induces_transfers_and_survives_controls() -> None:
    report = run_canary(task_count=24, null_runs=5)
    assert report["admitted"] is True
    assert report["program"]["depth"] >= 2
    assert report["single_primitive_shortcut"] is False
    assert report["null_found"] == 0
    assert report["counts"] == {
        "treatment_exact": 24,
        "coefficient_lesion_disrupted": 24,
        "wrong_input_disrupted": 24,
        "no_procedure_exact": report["counts"]["no_procedure_exact"],
    }
    assert report["counts"]["no_procedure_exact"] <= 2

    verification = verify(report)
    assert verification["verified"] is True
    assert verification["program_sha"] == report["program"]["sha"]


def test_independent_replay_rejects_a_resealed_row_change() -> None:
    report = run_canary(task_count=24, null_runs=5)
    report["rows"][0]["observed"] += 1
    report["task_set_sha256"] = _sha(report["rows"])
    body = {key: value for key, value in report.items() if key != "receipt_sha256"}
    report["receipt_sha256"] = _sha(body)

    try:
        verify(report)
    except ValueError as exc:
        assert "replay differs" in str(exc)
    else:  # pragma: no cover - the verifier must fail closed.
        raise AssertionError("resealed row tamper passed independent replay")
