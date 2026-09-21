"""Separate score changes from search changes on a frozen, target-blind bank."""

import math
from types import SimpleNamespace

from core.learning.semantic_argument_optimization import ArgumentOptimizationIncompleteError
from core.learning.semantic_candidate_bank import candidate_observation_identity
from core.learning.semantic_joint_graph_learning import (
    graph_selection_key, preferred_semantic_graph, score_annotated_graph,
)
from core.learning.semantic_program_campaign import _sha
from core.learning.semantic_program_ir import normalize_semantic_value
from core.learning.semantic_program_transducer import _hidden_array
from typing import Any


def rescore_semantic_candidate_bank(
    bank: Any,
    model: Any,
    *,
    source_token_ids: Any,
    hidden_states: Any,
    public_inputs: Any,
    source_text_sha256: Any,
    model_basis_sha256: Any,
    solve_time_limit_s: float=10.0,
    progress: Any=None,
) -> dict[str, Any]:
    """Replay both scorers through this function without adding target programs.

    Program bindings and source spans are frozen; the scorer optimizes its own
    latent mentions/definitions through the same chart used in ordinary decode.
    This is a fixed-bank intervention, not another serving path or a proof that
    the bounded bank exhausts the model's complete grammar.
    """
    if (type(solve_time_limit_s) not in (int, float)
            or not math.isfinite(solve_time_limit_s) or solve_time_limit_s <= 0):
        raise ValueError("bank replay allowance must be finite and positive")
    bank.validate()
    if bank.receipt.get("schema") != "aura.semantic_candidate_bank.v2":
        raise ValueError("bank replay requires execution-ordered operation spans")
    if (bank.receipt["source_text_sha256"] != source_text_sha256
            or bank.receipt["model_basis_sha256"] != model_basis_sha256
            or model.model_basis_sha256 != model_basis_sha256):
        raise ValueError("bank replay source or model basis differs")
    hidden = _hidden_array(hidden_states, expected_width=model.hidden_size)
    inputs = tuple(normalize_semantic_value(value) for value in public_inputs)
    tokens = tuple(source_token_ids)
    observed = candidate_observation_identity(tokens, inputs, hidden)
    if bank.receipt.get("observation_sha256") != observed:
        raise ValueError("bank replay observation differs")
    if len(bank.input_spans) != len(inputs):
        raise ValueError("bank input geometry differs")
    for span in bank.input_spans:
        span.validate_bound(len(tokens))
    # This view deliberately has no IR, source annotation, split or target.
    item = SimpleNamespace(hidden_states=hidden, public_inputs=inputs)
    body = {"schema": "aura.semantic_bank_replay.v1",
            "bank_receipt_sha256": bank.receipt["receipt_sha256"],
            "bank_transducer_receipt_sha256": bank.receipt["transducer_receipt_sha256"],
            "scorer_transducer_receipt_sha256": model.receipt_sha256,
            "observation_sha256": observed, "model_basis_sha256": model_basis_sha256,
            "source_text_sha256": source_text_sha256,
            "selection_policy": model.training_receipt.get("operation_assignment_policy", "first_feasible_v1"),
            "solve_time_limit_s": solve_time_limit_s,
            "candidate_generation_repeated": False, "source_annotations_available": False,
            "expected_answer_available": False, "serving_authority": False,
            "bank_search_complete": bank.receipt["search_complete"],
            "tie_policy": "first_in_frozen_bank", "rows": []}
    incumbent, best_index = None, None
    incomplete = False
    for index, candidate in enumerate(bank.candidates):
        if (candidate.program.n_inputs != len(inputs)
                or len(candidate.operation_spans) != len(candidate.program.instructions)
                or len(set(candidate.operation_spans)) != len(candidate.operation_spans)):
            raise ValueError("bank candidate source geometry differs")
        for span in candidate.operation_spans:
            span.validate_bound(len(tokens))
        instructions = tuple(SimpleNamespace(op=ins.op, args=ins.args, operation_span=span)
            for ins, span in zip(candidate.program.instructions, candidate.operation_spans, strict=True))
        row = {"index": index, "candidate_sha256": _sha(candidate.to_dict()),
               "program_sha256": candidate.program.sha()}
        if progress:
            progress({"stage": "bank_candidate_replay", "index": index, "total": len(bank.candidates)})
        try:
            graph = score_annotated_graph(model, item, instructions, bank.input_spans,
                                          source_token_ids=tokens,
                                          solve_time_limit_s=solve_time_limit_s)
        except ArgumentOptimizationIncompleteError as exc:
            incomplete = True
            row.update(status="incomplete", reason=str(exc))
        else:
            if graph is None:
                row["status"] = "infeasible_under_scorer"
            else:
                values = {key: graph[key] for key in ("score", "operation_score", "argument_score")}
                if not all(math.isfinite(value) for value in values.values()):
                    raise ValueError("bank replay score is nonfinite")
                if graph["program"] != candidate.program:
                    raise ValueError("bank replay changed the frozen program")
                row.update(status="scored", **values, selection_key=graph_selection_key(model, graph))
                if incumbent is None or preferred_semantic_graph(model, graph, incumbent):
                    incumbent, best_index = graph, index
        body["rows"].append(row)
    body.update(scoring_complete=not incomplete, best_observed_index=best_index,
                selected_index=best_index if not incomplete else None)
    return {**body, "receipt_sha256": _sha(body)}


def compare_semantic_candidate_banks(
    parent: Any,
    candidate: Any,
    item: Any,
    *,
    max_charts: int=2,
    max_graphs_per_chart: int=2,
    solve_time_limit_s: float=3.0,
    progress: Any=None,
) -> Any:
    """A two-by-two search/scorer intervention, labeled only after both banks exist."""
    from core.learning.semantic_failure_diagnosis import diagnose_semantic_candidate_bank

    if item.split not in {"train", "validation"}:
        raise ValueError("bank comparison requires a development observation")
    if (parent.model_basis_sha256 != candidate.model_basis_sha256
            or parent.input_grounding != candidate.input_grounding):
        raise ValueError("bank comparison representation differs")
    models = {"parent": parent, "candidate": candidate}
    options = dict(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
        public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=item.ir.model_basis_receipt_sha256,
        solve_time_limit_s=solve_time_limit_s)
    banks = {name: model.decode_candidates(**options, max_charts=max_charts,
        max_graphs_per_chart=max_graphs_per_chart, progress=progress) for name, model in models.items()}
    body = {"schema": "aura.semantic_bank_comparison.v1", "split": item.split,
            "source_text_sha256": item.ir.source_text_sha256,
            "banks": {name: bank.receipt for name, bank in banks.items()},
            "replays": {}, "diagnoses": {}, "serving_authority": False,
            "learning_performed": False, "fresh_transfer_claim": False,
            "labels_used_after_both_banks_completed": True}
    for name, bank in banks.items():
        if bank.selected.ir is None:
            body["replays"][name] = {"status": "bank_decode_unavailable"}
            continue
        replays = {scorer: rescore_semantic_candidate_bank(bank, model, **options, progress=progress)
                   for scorer, model in models.items()}
        diagnosis = diagnose_semantic_candidate_bank(bank, item)
        statuses = {row["program_sha256"]: row["status"] for row in diagnosis["comparisons"]}
        body["diagnoses"][name] = diagnosis
        body["replays"][name] = {"status": "measured", "scorers": replays,
            "selected_semantic_status": {scorer: (statuses[bank.candidates[replay["selected_index"]].program.sha()]
                if replay["selected_index"] is not None else "unmeasured")
                for scorer, replay in replays.items()}}
    def identities(bank: Any) -> set[Any]:
        return {_sha({"program": value.program.to_dict(),
                      "operation_spans": [span.to_dict() for span in value.operation_spans]})
                for value in bank.candidates}
    left, right = identities(banks["parent"]), identities(banks["candidate"])
    body["candidate_overlap"] = {"shared": sorted(left & right), "parent_only": sorted(left - right),
                                  "candidate_only": sorted(right - left),
                                  "comparison_scope": "observed_banks_not_complete_grammar"}
    return {**body, "receipt_sha256": _sha(body)}
