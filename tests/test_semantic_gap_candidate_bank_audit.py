"""The larger G03 bank audit freezes search before admitting labels."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from tools.diagnose_semantic_gap_candidate_bank import (
    _audit_row,
    _frozen_methods,
    _plan,
    _summarize,
    _verify_inputs,
)


def _gaps():
    return {"schema": "aura.semantic_portfolio_gap_diagnosis.v1",
            "validation_labels_used_only_for_diagnosis": True,
            "summary": {"candidate_unavailable": 1, "selection_miss": 1},
            "failures": [{"source": "a", "kind": "candidate_unavailable"},
                         {"source": "b", "kind": "selection_miss"}]}


def test_plan_binds_population_model_and_finite_search_allowances():
    model = SimpleNamespace(receipt_sha256="model", model_basis_sha256="basis")
    options = dict(gap_sha="gap", model=model, source_report_sha="source",
                   candidate_report_sha="candidate",
                   implementation={"source": "commit"}, max_charts=8,
                   max_graphs=4, solve_seconds=2., composition_budget=8,
                   composition_examined_limit=256)
    plan = _plan(_gaps(), **options)
    assert plan["sources"] == ("a", "b")
    assert plan["validation_labels_not_sent_to_decoder"] is True
    assert plan["serving_authority"] is False
    assert plan["plan_sha256"] != _plan(_gaps(), **{**options, "max_charts": 9})["plan_sha256"]
    with pytest.raises(ValueError, match="bounds"):
        _plan(_gaps(), **{**options, "max_graphs": 0})
    with pytest.raises(ValueError, match="development failures"):
        _plan({**_gaps(), "validation_labels_used_only_for_diagnosis": False}, **options)


def test_decode_receives_observation_but_not_source_target(monkeypatch):
    import core.learning.semantic_failure_diagnosis as diagnosis_module

    item = SimpleNamespace(
        ir=SimpleNamespace(source_token_ids=(1, 2), source_text_sha256="source",
                           model_basis_receipt_sha256="basis", gold_program="secret"),
        public_inputs=(2, 3), hidden_states="hidden", split="validation",
    )
    seen = []

    def decode_candidates(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(receipt={"search_complete": False}, validate=lambda: None)

    monkeypatch.setattr(diagnosis_module, "diagnose_semantic_candidate_bank",
                        lambda bank, source: {"failure_stage": "incomplete_search"})
    model = SimpleNamespace(decode_candidates=decode_candidates)
    plan = {"plan_sha256": "p", "max_charts": 8, "max_graphs_per_chart": 4,
            "solve_seconds": 2.}
    row = _audit_row(item, model, plan)
    assert row["source"] == "source"
    assert row["diagnosis"]["failure_stage"] == "incomplete_search"
    assert seen == [{"source_token_ids": (1, 2), "hidden_states": "hidden",
                     "public_inputs": (2, 3), "source_text_sha256": "source",
                     "model_basis_sha256": "basis", "max_charts": 8,
                     "max_graphs_per_chart": 4, "solve_time_limit_s": 2.}]


def test_partial_audit_cannot_be_reported_as_a_complete_population():
    plan = {"sources": ("a", "b")}
    rows = [{"diagnosis": {"failure_stage": "selection", "correct_reachable": True,
                           "selected_semantic_status": "different"},
             "mixed": {"correct_reachable": True}}]
    report = _summarize(rows, plan)
    assert report["observed"] == 1
    assert report["population"] == 2
    assert report["complete_population"] is False
    assert report["failure_stage_counts"] == {"selection": 1}
    assert report["correct_reachable_counts"] == {"True": 1}
    assert report["mixed_reachable_counts"] == {"True": 1}
    assert report["selected_equivalent"] == 0


def test_audit_inputs_require_original_candidate_manifest_and_target():
    model = SimpleNamespace(receipt_sha256="model")
    source = {"representation_compatibility": {
        "source_feature_manifest_sha256s": {"family": "manifest"}}}
    gap = {"source_manifest_sha256s": {"family": "manifest"},
           "failures": [{"source": "source", "gold_program": {"instructions": []}}]}
    item = SimpleNamespace(ir=SimpleNamespace(
        to_program=lambda: SimpleNamespace(to_dict=lambda: {"instructions": []})))
    args = (gap, source, {"candidate": "model"}, model, {"source": item})
    _verify_inputs(*args)
    with pytest.raises(ValueError, match="candidate report"):
        _verify_inputs(gap, source, {"candidate": "other"}, model, {"source": item})
    with pytest.raises(ValueError, match="feature manifests"):
        _verify_inputs({**gap, "source_manifest_sha256s": {}}, source,
                       {"candidate": "model"}, model, {"source": item})
    with pytest.raises(ValueError, match="gap target"):
        _verify_inputs(gap, source, {"candidate": "model"}, model, {})


def test_frozen_methods_use_the_original_target_blind_graph(tmp_path):
    from core.learning.procedure_induction import Instruction, Program

    program = Program(2, (Instruction("sub", (0, 1)),))
    row = {"source": "source", "arm": "context", "program": program.to_dict(),
           "original_program": program.to_dict(), "runtime_input_spans": [[1, 2], [3, 4]]}
    path = tmp_path / "report.json"
    raw = json.dumps({"rows": [row]}).encode()
    path.write_bytes(raw)
    gaps = {"graph_report_sha256s": {str(path): hashlib.sha256(raw).hexdigest()},
            "failures": [{"source": "source", "method_programs": {"context": program.to_dict()}}]}
    items = {"source": SimpleNamespace(public_inputs=(5, 5))}
    methods = _frozen_methods(gaps, items)
    assert methods["source"]["context"].program == program
    assert methods["source"]["context"].source_sha256 == "source"
    path.write_text('{"rows": []}')
    with pytest.raises(ValueError, match="digest differs"):
        _frozen_methods(gaps, items)
