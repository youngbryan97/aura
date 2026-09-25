"""Inspect the deployed semantic grammar without supplying a target to search."""

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from core.learning.procedure_induction import Program
from core.learning.semantic_argument_optimization import ArgumentOptimizationIncompleteError
from core.learning.semantic_graph_counterexamples import (
    argument_graph_order,
    argument_graph_program,
)
from core.learning.semantic_operation_search import (
    OperationChartSearch,
    OperationSearchIncompleteError,
)
from core.learning.semantic_program_campaign import _sha
from core.learning.semantic_program_ir import TokenSpan, normalize_semantic_value
from core.learning.semantic_program_transducer import SemanticTransductionOutcome, _hidden_array
from core.learning.semantic_program_transducer_fitting import _assign_typed_arguments
from core.verify.invariants import invariant


@dataclass(frozen=True)
class SemanticCandidate:
    """A program and its source evidence, all in execution order."""
    program: Program
    joint_score: float | None
    operation_spans: tuple[TokenSpan, ...]
    chart_index: int | None
    graph_index: int | None
    argument_spans: tuple[tuple[TokenSpan, ...], ...] | None = None
    definition_spans: tuple[tuple[TokenSpan, ...], ...] | None = None
    definition_provenance: str = "unavailable"

    def __post_init__(self) -> None:
        if len(self.operation_spans) != self.program.depth:
            raise ValueError("candidate operation evidence differs from program")
        if (self.definition_provenance not in {
                "unavailable", "register_anchor", "optimizer_selected"}
                or (self.definition_spans is None) !=
                (self.definition_provenance == "unavailable")):
            raise ValueError("candidate definition evidence lacks its origin")
        for name, rows in (("argument", self.argument_spans),
                           ("definition", self.definition_spans)):
            if rows is not None and (len(rows) != self.program.depth or any(
                    len(row) != len(ins.args) or any(not isinstance(span, TokenSpan)
                                                    for span in row)
                    for row, ins in zip(rows, self.program.instructions, strict=True))):
                raise ValueError(f"candidate {name} evidence differs from program")

    @classmethod
    def from_argument_graph(
        cls,
        nodes: tuple[Any, ...],
        arguments: tuple[Any, ...],
        *,
        n_inputs: int,
        joint_score: float,
        chart_index: int,
        graph_index: int,
        argument_spans: tuple[tuple[TokenSpan, ...], ...] | None = None,
        definition_spans: tuple[tuple[TokenSpan, ...], ...] | None = None,
    ) -> Any:
        order = argument_graph_order(nodes, arguments, n_inputs=n_inputs)
        return cls(argument_graph_program(nodes, arguments, n_inputs=n_inputs), joint_score,
                   tuple(nodes[index].span for index in order), chart_index, graph_index,
                   tuple(argument_spans[index] for index in order)
                   if argument_spans is not None else None,
                   tuple(definition_spans[index] for index in order)
                   if definition_spans is not None else None,
                   "optimizer_selected" if definition_spans is not None else "unavailable")

    def to_dict(self) -> dict[str, Any]:
        return {"program": self.program.to_dict(), "program_sha256": self.program.sha(),
                "joint_score": self.joint_score,
                "operation_spans": [span.to_dict() for span in self.operation_spans],
                "chart_index": self.chart_index, "graph_index": self.graph_index,
                "argument_spans": ([[span.to_dict() for span in row] for row in self.argument_spans]
                                   if self.argument_spans is not None else None),
                "definition_spans": ([[span.to_dict() for span in row] for row in self.definition_spans]
                                     if self.definition_spans is not None else None),
                "definition_provenance": self.definition_provenance}


@dataclass(frozen=True)
class SemanticCandidateBank:
    selected: SemanticTransductionOutcome
    candidates: tuple[SemanticCandidate, ...]
    input_spans: tuple[TokenSpan, ...]
    receipt: dict

    def meaning_hypotheses(self):
        """Expose source-bound proposal meanings without claiming a posterior."""
        from core.learning.semantic_meaning_hypothesis import meaning_hypotheses_from_bank

        return meaning_hypotheses_from_bank(self)

    def validate(self) -> None:
        """Reject modified payloads before they can become diagnostic evidence."""
        body = {key: value for key, value in self.receipt.items() if key != "receipt_sha256"}
        if self.receipt.get("receipt_sha256") != _sha(body):
            raise ValueError("candidate bank receipt integrity mismatch")
        selected = self.selected.ir.to_program().sha() if self.selected.ir is not None else None
        if (body.get("candidates") != [candidate.to_dict() for candidate in self.candidates]
                or body.get("input_spans") != [span.to_dict() for span in self.input_spans]
                or body.get("selected_program_sha256") != selected
                or body.get("selected_refusal") != self.selected.refusal
                or body.get("selected_search_interrupted") != self.selected.search_interrupted):
            raise ValueError("candidate bank payload differs from receipt")
        if body.get("search_complete") is True and (
                body.get("operation_inventory_exhausted") is not True
                or body.get("operation_search_complete") is not True
                or any(row.get("search_complete") is not True for row in body["charts"])):
            raise ValueError("candidate bank completion lacks exhausted search")


def candidate_observation_identity(
    tokens: tuple[Any, ...],
    inputs: tuple[Any, ...],
    hidden: Any,
) -> Any:
    """Bind every decoded observation, not only its caller-supplied text hash."""
    return _sha({"tokens": tuple(tokens), "inputs": tuple(inputs),
        "hidden_dtype": hidden.dtype.str, "hidden_shape": list(hidden.shape),
        "hidden_sha256": hashlib.sha256(hidden.tobytes(order="C")).hexdigest()})


def _register_definition_anchors(ir: Any) -> tuple[tuple[TokenSpan, ...], ...]:
    """Distinguish known IR register anchors from optimizer-selected definitions."""
    anchors = list(ir.input_spans)
    rows = []
    for instruction in ir.instructions:
        rows.append(tuple(anchors[reference] for reference in instruction.args))
        anchors.append(instruction.operation_span)
    return tuple(rows)


def decode_semantic_candidates(
    model: Any,
    *,
    source_token_ids: Any,
    hidden_states: Any,
    public_inputs: Any,
    source_text_sha256: Any,
    model_basis_sha256: Any,
    max_charts: int=16,
    max_graphs_per_chart: int=16,
    solve_time_limit_s: float=10.0,
    progress: Any=None,
) -> Any:
    """Retain bounded alternatives from the same builders used by ``decode``.

    The incumbent answer is preserved, not selected again from this bank.
    Search completion refers only to the model's declared bounded grammar.
    Legacy beams cannot certify that grammar's exhaustion. No target program,
    expected value, family label or source annotation enters candidate search.
    """
    if any(type(value) is not int or value < 1 for value in (max_charts, max_graphs_per_chart)):
        raise ValueError("candidate bank allowances must be positive integers")
    if (type(solve_time_limit_s) not in (int, float) or not math.isfinite(solve_time_limit_s)
            or solve_time_limit_s <= 0):
        raise ValueError("candidate solve allowance must be finite and positive")
    source_token_ids, public_inputs = tuple(source_token_ids), tuple(public_inputs)
    if progress:
        progress({"stage": "ordinary_decode"})
    outcome = model.decode(source_token_ids=source_token_ids, hidden_states=hidden_states,
        public_inputs=public_inputs, source_text_sha256=source_text_sha256,
        model_basis_sha256=model_basis_sha256, search_time_limit_s=solve_time_limit_s)
    body = {"schema": "aura.semantic_candidate_bank.v3", "source_text_sha256": source_text_sha256,
            "model_basis_sha256": model_basis_sha256, "transducer_receipt_sha256": model.receipt_sha256,
            "expected_answer_available": False, "source_annotations_available": False,
            "serving_authority": False, "selection_changed": False,
            "max_charts": max_charts, "max_graphs_per_chart": max_graphs_per_chart,
            "solve_time_limit_s": solve_time_limit_s, "charts": [],
            "operation_inventory_exhausted": False, "operation_search_complete": False,
            "search_complete": False, "selected_refusal": outcome.refusal}
    body["selected_search_interrupted"] = outcome.search_interrupted
    candidates = []
    spans = ()

    def finish(reason: str) -> Any:
        body["limit_reason"] = reason
        body["candidates"] = [candidate.to_dict() for candidate in candidates]
        body["input_spans"] = [span.to_dict() for span in spans]
        body["selected_program_sha256"] = outcome.ir.to_program().sha() if outcome.ir is not None else None
        return SemanticCandidateBank(outcome, tuple(candidates), tuple(spans),
            {**body, "receipt_sha256": _sha(body)})

    # Search interruption is not proof that no graph exists. Keep the public
    # refusal while inspecting diagnostic alternatives; pre-search failures stay closed.
    if outcome.ir is None and not outcome.search_interrupted:
        return finish("ordinary_decode_unavailable")
    if outcome.ir is not None:
        spans = outcome.ir.input_spans
        candidates.append(SemanticCandidate(outcome.ir.to_program(), None,
            tuple(ins.operation_span for ins in outcome.ir.instructions), None, None,
            tuple(ins.argument_spans for ins in outcome.ir.instructions),
            _register_definition_anchors(outcome.ir), "register_anchor"))
    inputs = tuple(normalize_semantic_value(value) for value in public_inputs)
    hidden = _hidden_array(hidden_states, expected_width=model.hidden_size)
    tokens = tuple(source_token_ids)
    body["observation_sha256"] = candidate_observation_identity(tokens, inputs, hidden)
    if model.training_receipt.get("argument_search_strategy") != "global_constraint_v1":
        return finish("argument_inventory_unsupported")
    try:
        spans, _, argument_scores, charts = model._runtime_operation_charts(
            tokens, hidden, inputs, model.inference_step_limit(len(inputs)))
    except OperationSearchIncompleteError as exc:
        return finish(str(exc))
    if outcome.ir is not None and tuple(spans) != outcome.ir.input_spans:
        raise ValueError("candidate search changed ordinary input grounding")
    charts = iter(charts)
    relation_scores, relation_vectors = {}, {}
    definition_scores = model.definition_pointer.score_sequence(hidden)
    reasons = []
    try:
        for chart_index in range(max_charts + 1):
            nodes = next(charts, None)
            if nodes is None:
                body["operation_inventory_exhausted"] = True
                body["operation_search_complete"] = isinstance(charts, OperationChartSearch) and charts.complete
                if not body["operation_search_complete"]:
                    reasons.append("operation_beam_not_complete")
                break
            if chart_index == max_charts:
                reasons.append("operation_chart_limit")
                break
            captured = []
            if progress:
                progress({"stage": "candidate_chart", "chart": chart_index})
            row = {"operations": [{"op": node.operation, "span": node.span.to_dict()} for node in nodes],
                   "examined_graphs": 0, "search_complete": not captured}
            body["charts"].append(row)
            try:
                _assign_typed_arguments(model=model, hidden=hidden, inputs=inputs, input_spans=spans,
                    source_token_ids=tokens,
                    operation_nodes=nodes, argument_pointer_scores=argument_scores,
                    relation_score_cache=relation_scores, relation_vector_cache=relation_vectors,
                    definition_pointer_scores=definition_scores, chart_observer=captured.append,
                    build_only=True)
            except ArgumentOptimizationIncompleteError as exc:
                row["interruption"] = str(exc)
                row["search_complete"] = False
                reasons.append("argument_chart_construction_incomplete")
                continue
            row["search_complete"] = not captured
            if not captured:
                continue
            excluded = []
            try:
                for graph_index in range(max_graphs_per_chart + 1):
                    if progress:
                        progress({"stage": "candidate_graph", "chart": chart_index, "graph": graph_index})
                    selected_options = []
                    result = captured[0].solve(excluded_graphs=excluded,
                                               selection_observer=selected_options.append,
                                               time_limit_s=solve_time_limit_s)
                    if result is None:
                        row["search_complete"] = True
                        break
                    if graph_index == max_graphs_per_chart:
                        reasons.append("argument_graph_limit")
                        break
                    score, arguments, mentions, _dependencies = result
                    if len(selected_options) != 1:
                        raise ValueError("candidate graph omitted its selected source evidence")
                    definition_spans = (tuple(tuple(captured[0].definition_options[node][role][index]
                                                       for role, index in enumerate(indices))
                                              for node, indices in enumerate(selected_options[0]))
                                        if captured[0].definition_options is not None else None)
                    excluded.append(arguments)
                    program = argument_graph_program(nodes, arguments, n_inputs=len(inputs))
                    joint_score = score + sum(node.score for node in nodes) - model.operation_length_penalty * len(nodes)
                    if not math.isfinite(joint_score):
                        raise ValueError("candidate graph score is nonfinite")
                    candidates.append(SemanticCandidate.from_argument_graph(nodes, arguments,
                        n_inputs=len(inputs), joint_score=joint_score,
                        chart_index=chart_index, graph_index=graph_index,
                        argument_spans=mentions, definition_spans=definition_spans))
                    row["examined_graphs"] += 1
                    if progress:
                        progress({"stage": "candidate_retained", "chart": chart_index,
                                  "graph": graph_index, "program_sha256": program.sha()})
            except ArgumentOptimizationIncompleteError as exc:
                row["interruption"] = str(exc)
                reasons.append("argument_search_incomplete")
    except OperationSearchIncompleteError as exc:
        body["interruption"] = str(exc)
        reasons.append("operation_search_incomplete")
    body["search_complete"] = body["operation_search_complete"] and all(
        row["search_complete"] for row in body["charts"])
    return finish(",".join(dict.fromkeys(reasons)) or None)


@invariant("learning.semantic_candidate_spans_follow_program_order", scope="learning",
           owner="core/learning/semantic_candidate_bank.py", observational=False)
def _candidate_span_order() -> tuple:
    from types import SimpleNamespace

    nodes = (SimpleNamespace(operation="sub", span=TokenSpan(2, 3)),
             SimpleNamespace(operation="add", span=TokenSpan(9, 10)))
    candidate = SemanticCandidate.from_argument_graph(nodes, ((3, 0), (0, 1)),
        n_inputs=2, joint_score=1., chart_index=0, graph_index=0)
    assert candidate.program.instructions[0].op == "add"
    assert candidate.operation_spans == (nodes[1].span, nodes[0].span)
    return ()
