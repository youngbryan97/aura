from types import SimpleNamespace

import pytest

from tools.evaluate_semantic_grounded_native import PublicDecodeArm, compare, digest


class FixedPublicDecoder:
    def __init__(self, outputs):
        self.outputs, self.calls = outputs, []
        self.last_receipt = None

    def decode(self, **public):
        assert set(public) == {"source_token_ids", "hidden_states", "public_inputs",
            "source_text_sha256", "model_basis_sha256", "search_time_limit_s"}
        self.calls.append(public)
        result = self.outputs[public["source_text_sha256"]]
        self.last_receipt = {"source_id": public["source_text_sha256"], "target_available_to_decoder": False}
        return SimpleNamespace(ir=result, refusal=None if result is not None else "fixture_refusal")


def cases():
    from tests.test_semantic_program_shared_transducer import _examples
    return _examples()[:2]


def arms(items, *, joint_failure=False):
    result = {}
    for name in ("source_parent", "global_chart", "joint_native"):
        outputs = {item.ir.source_text_sha256: item.ir for item in items}
        if name != "joint_native":
            outputs[items[1].ir.source_text_sha256] = None
        elif joint_failure:
            outputs[items[0].ir.source_text_sha256] = None
        result[name] = PublicDecodeArm(FixedPublicDecoder(outputs), receipt="a" * 64, search_seconds=2.)
    return result


def test_paired_screen_passes_only_public_inputs_and_does_not_close_g03():
    items = cases()
    candidates = arms(items)
    result = compare(candidates, items)
    assert result["advance_development"] and result["paired_regressions"] == {"source_parent": 0, "global_chart": 0}
    assert result["comparison"]["candidates"]["joint_native"]["program_equivalent"] == 2
    assert not result["held_used_for_fit_or_checkpoint_selection"]
    assert not result["qualification_evidence"] and not result["serving_authority"]
    for name, arm in candidates.items():
        assert len(arm.decoder.calls) == 2
        assert all(call["search_time_limit_s"] == 2. for call in arm.decoder.calls)
        assert all(row["joint_receipt"]["target_available_to_decoder"] is False
            for row in result["decodes"][name].values())


def test_screen_records_control_regressions_and_refuses_advance():
    items = cases()
    result = compare(arms(items, joint_failure=True), items)
    assert not result["advance_development"]
    assert result["paired_regressions"] == {"source_parent": 1, "global_chart": 1}
    assert result["comparison"]["candidates"]["joint_native"]["equivalent_gains"] == 1
    assert result["comparison"]["candidates"]["joint_native"]["equivalent_regressions"] == 1


def test_screen_requires_the_declared_controls_and_nonempty_unique_cases():
    items = cases()
    with pytest.raises(ValueError, match="three declared arms"):
        compare({"joint_native": arms(items)["joint_native"]}, items)
    with pytest.raises(ValueError, match="unique validation"):
        compare(arms(items), ())
    with pytest.raises(ValueError, match="unique validation"):
        compare(arms(items), (items[0], items[0]))


def measured_report():
    items = cases()
    comparison = compare(arms(items), items)
    fit_body = {"selected_step": 4, "learned_checkpoint_selected": True,
        "fit_receipt_sha256": "b" * 64, "weights_sha256": "c" * 64}
    fit = {**fit_body, "receipt_sha256": digest(fit_body)}
    plan_body = {"schema": "aura.grounded_native_development_plan.v1", "fit_verification": fit,
        "held_ids": [item.ir.source_text_sha256 for item in items],
        "arms": ["source_parent", "global_chart", "joint_native"],
        "cohort": "previously_exposed_source_bank_development",
        "held_controls_fit_or_checkpoint_selection": False, "serving_authority": False}
    plan = {**plan_body, "plan_sha256": digest(plan_body)}
    for name, metrics in comparison["comparison"]["candidates"].items():
        metrics["transducer_receipt_sha256"] = digest({"arm": name, "plan": plan["plan_sha256"]})
    seal(comparison["comparison"], "report_sha256")
    for identity, decoded in comparison["decodes"]["joint_native"].items():
        decoded["joint_receipt"] = {"schema": "aura.grounded_native_chart_decode.v1", "source_id": identity,
            "fit_receipt_sha256": fit["fit_receipt_sha256"], "weights_sha256": fit["weights_sha256"],
            "selected_step": 4, "learned_checkpoint_selected": True, "source_only_native_capture": True,
            "target_available_to_decoder": False, "serving_authority": False,
            "refusal": None, "selected_chart": {"status": "bound"}}
    report = {"schema": "aura.grounded_native_development.v1", "plan": plan, **comparison,
        "g03_complete": False, "fresh_transfer_proven": False}
    seal(report, "receipt_sha256")
    return report, plan, fit


def seal(value, key):
    value[key] = digest({k: v for k, v in value.items() if k != key})


def test_independent_report_check_keeps_artifact_verification_separate_from_replay():
    from tools.verify_semantic_grounded_evaluation import verify_report
    result = verify_report(*measured_report())
    assert result["advance_development"] and result["artifacts_verified"]
    assert result["measured_requests"] == 2
    assert not result["semantic_interpretations_independently_replayed"]
    assert not result["g03_complete"] and not result["qualification_evidence"]


@pytest.mark.parametrize("fault", ["checksum", "count", "source", "step", "target", "verdict", "proof"])
def test_independent_report_rejects_rehashed_semantic_contract_drift(fault):
    from tools.verify_semantic_grounded_evaluation import verify_report
    report, plan, fit = measured_report()
    if fault == "count":
        report["comparison"]["candidates"]["joint_native"]["program_equivalent"] = 1
        seal(report["comparison"], "report_sha256")
    elif fault in {"source", "step", "target"}:
        row = next(iter(report["decodes"]["joint_native"].values()))["joint_receipt"]
        row[{"source": "source_id", "step": "selected_step", "target": "target_available_to_decoder"}[fault]] = {
            "source": "d" * 64, "step": 0, "target": True}[fault]
    elif fault == "verdict":
        report["advance_development"] = False
    elif fault == "proof":
        report["g03_complete"] = True
    if fault != "checksum":
        seal(report, "receipt_sha256")
    else:
        report["receipt_sha256"] = "e" * 64
    with pytest.raises(ValueError):
        verify_report(report, plan, fit)


def test_per_request_progress_reports_started_and_measured_completion():
    items = cases()
    progress = []
    arm = PublicDecodeArm(FixedPublicDecoder({items[0].ir.source_text_sha256: items[0].ir}),
        receipt="a" * 64, search_seconds=2., name="joint_native", progress=progress.append)
    item = items[0]
    arm.decode(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
        public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=item.ir.model_basis_receipt_sha256)
    assert [row["stage"] for row in progress] == ["grounded_decode_started", "grounded_decode_completed"]
    assert progress[-1]["elapsed_seconds"] >= 0 and progress[-1]["program_sha256"] is not None
