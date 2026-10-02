"""Source-only complete-program contrasts under the runtime's factor score.

The partition is over witnessed, bounded alternatives, not every program.
Unknown semantic comparisons remain unlabelled rather than becoming negatives.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from itertools import islice

import mlx.core as mx

from core.learning.semantic_conditioned_relations import (
    conditional_choice_scores,
    conditioned_selected_scores,
)
from core.learning.semantic_context_binding import BindingContext, BindingRole, ContextReferent
from core.learning.semantic_program_floor import semantic_primitive_type_signature
from core.learning.semantic_relational_pointer import semantic_role_features


@dataclass(frozen=True)
class ProgramGraphChoice:
    indices: tuple
    baseline_score: float
    comparison: dict


@dataclass(frozen=True)
class ProgramChartChoices:
    nodes: tuple
    chart: object
    graphs: tuple[ProgramGraphChoice, ...]

    def receipt(self):
        chart = self.chart
        return {"operations": [(node.operation, node.span.start, node.span.end) for node in self.nodes],
            "options": [[[(score, register, span.start, span.end) for score, register, span in slot]
                for slot in node] for node in chart.options],
            "definitions": None if chart.definition_options is None else
                [[[None if span is None else (span.start, span.end) for span in slot]
                    for slot in node] for node in chart.definition_options],
            "definition_scores": sorted((register, span.start, span.end, score)
                for (register, span), score in (chart.definition_scores or {}).items()),
            "normalizer": chart.choice_log_normalizer,
            "n_inputs": chart.n_inputs, "register_contract": chart.contract.to_dict(),
            "graphs": [{"indices": graph.indices, "baseline_score": graph.baseline_score,
                "comparison": graph.comparison} for graph in self.graphs]}


def program_chart_edge_scores(pointer, states, proposal, *, source_id, input_spans, inputs,
                              nuisance_projection=None, batch_size=16):
    """Condition each public mention/definition exactly as the runtime does."""
    nodes, chart = proposal.nodes, proposal.chart
    if (states.ndim != 3 or states.shape[1:] != (pointer.depths, pointer.hidden_width)
            or type(batch_size) is not int or not 1 <= batch_size <= 64
            or len(nodes) != len(chart.options) or len(input_spans) != chart.n_inputs
            or len(inputs) != chart.n_inputs):
        raise ValueError("program objective and runtime chart geometry differ")
    signatures = [semantic_primitive_type_signature(node.operation) for node in nodes]
    if any(signature is None for signature in signatures):
        raise ValueError("complete program needs typed operation roles")
    anchors = (*input_spans, *(node.span for node in nodes))
    types = tuple("integer_sequence" if isinstance(value, tuple) else "integer" for value in inputs)
    types += tuple(signature[1] for signature in signatures)
    records = tuple(ContextReferent(source_id, f"register:{index}", types[index],
        f"source-token-span:{span.start}:{span.end}", scope=("program",)) for index, span in enumerate(anchors))
    context = BindingContext(records, {name: None for name in types})
    roles, operations, mentions, locations = [], [], [], []
    observed = {}
    def observe(span):
        span.validate_bound(len(states))
        if span not in observed:
            value = mx.mean(states[span.start:span.end], axis=0)
            if nuisance_projection is not None:
                basis = mx.array(nuisance_projection.basis.tolist(), dtype=value.dtype)
                value = value - (value @ basis) @ basis.T
            observed[span] = value
        return observed[span]
    for step, (node, slots, signature) in enumerate(zip(nodes, chart.options, signatures, strict=True)):
        if len(slots) != len(signature[0]) or any(not pool for pool in slots):
            raise ValueError("complete program has an empty or untyped argument slot")
        for slot, (pool, required) in enumerate(zip(slots, signature[0], strict=True)):
            role = BindingRole(f"argument:{step}:{slot}", f"{node.operation}:operand:{slot}", required,
                scope=("program",), referents=tuple(record.key for index, record in enumerate(records)
                    if index != chart.n_inputs + step))
            roles.append(role)
            operations.append(observe(node.span))
            weights = mx.softmax(mx.array([score for score, _register, _span in pool]))
            mentions.append(mx.sum(mx.stack([observe(span) for _score, _register, span in pool])
                * weights[:, None, None], axis=0))
            for index, (_score, register, span) in enumerate(pool):
                if (type(register) is not int or not 0 <= register < len(records)
                        or not context.eligible(role, records[register])):
                    raise ValueError("complete program has an inadmissible public register")
                definition = None if chart.definition_options is None else chart.definition_options[step][slot][index]
                locations.append((len(roles) - 1, register, span, definition))
    allowed = mx.array([[context.eligible(role, record) for record in records] for role in roles])
    adjacency = mx.zeros((len(roles) + len(records),) * 2)
    adjacency[:len(roles), len(roles):] = allowed.astype(mx.float32)
    adjacency[len(roles):, :len(roles)] = allowed.T.astype(mx.float32)
    features = semantic_role_features(roles)
    def encode(values, kind):
        gate = pointer.lora_depth_filler if kind == "candidate" else pointer.lora_depth_query
        return pointer._pool(mx.stack(values), gate) @ getattr(pointer, "lora_" + kind)
    operations = encode(operations, "operation")
    mentions = encode(mentions, "mention")
    candidates = encode([observe(span) for span in anchors], "candidate")
    learned = []
    for start in range(0, len(locations), batch_size):
        batch = locations[start:start + batch_size]
        rows = mx.array([row for row, *_rest in batch], dtype=mx.int32)
        columns = mx.array([column for _row, column, *_rest in batch], dtype=mx.int32)
        index = mx.arange(len(batch))
        m = mx.broadcast_to(mentions, (len(batch), *mentions.shape))
        c = mx.broadcast_to(candidates, (len(batch), *candidates.shape))
        m[index, rows] = encode([observe(span) for _row, _column, span, _definition in batch], "mention")
        c[index, columns] = encode([observe(anchors[column] if definition is None else definition)
            for _row, column, _span, definition in batch], "candidate")
        learned.append(conditioned_selected_scores(pointer,
            mx.broadcast_to(operations, (len(batch), *operations.shape)), m, c,
            rows=rows, columns=columns, adjacency=adjacency, role_features=features))
    return mx.concatenate(learned)


@dataclass(frozen=True)
class GroundedProgramSupervision:
    source_id: str
    input_spans: tuple
    inputs: tuple
    charts: tuple[ProgramChartChoices, ...]
    mining: dict

    def receipt(self):
        return {"source_id": self.source_id, "input_spans": [(span.start, span.end) for span in self.input_spans],
            "inputs": self.inputs, "charts": [chart.receipt() for chart in self.charts], "mining": self.mining}

    @classmethod
    def from_receipt(cls, value):
        from core.learning.semantic_argument_chart import ScoredArgumentChart
        from core.learning.semantic_program_ir import TokenSpan
        from core.learning.semantic_program_transducer_fitting import (
            RegisterUseContract,
            _OperationNode,
        )

        proposals = []
        for row in value["charts"]:
            options = tuple(tuple(tuple((score, register, TokenSpan(start, end))
                for score, register, start, end in slot) for slot in node) for node in row["options"])
            definitions = None if row["definitions"] is None else tuple(tuple(tuple(
                None if span is None else TokenSpan(*span) for span in slot) for slot in node)
                for node in row["definitions"])
            chart = ScoredArgumentChart(options, row["n_inputs"], RegisterUseContract(**row["register_contract"]),
                definition_options=definitions, definition_scores=None if not row["definition_scores"] else {
                    (register, TokenSpan(start, end)): score
                    for register, start, end, score in row["definition_scores"]},
                choice_log_normalizer=row["normalizer"])
            nodes = tuple(_OperationNode(TokenSpan(start, end), label, 0., 0., 1.)
                for label, start, end in row["operations"])
            graphs = tuple(ProgramGraphChoice(tuple(tuple(node) for node in graph["indices"]),
                graph["baseline_score"], graph["comparison"]) for graph in row["graphs"])
            proposals.append(ProgramChartChoices(nodes, chart, graphs))
        return cls(value["source_id"], tuple(TokenSpan(*span) for span in value["input_spans"]),
            tuple(tuple(item) if isinstance(item, list) else item for item in value["inputs"]),
            tuple(proposals), value["mining"])

    def validate(self, operation_field, token_count):
        from core.learning.semantic_native_operation_field import operation_span_inventory

        spans = set(operation_span_inventory(token_count, operation_field.max_span_tokens, self.input_spans))
        positives, negatives = 0, 0
        if not self.source_id or not self.charts or len(self.input_spans) != len(self.inputs):
            raise ValueError("complete program needs source-qualified alternatives")
        for proposal in self.charts:
            chart = proposal.chart
            if (not 1 <= len(proposal.nodes) <= operation_field.max_steps
                    or chart.n_inputs != len(self.inputs) or len(chart.options) != len(proposal.nodes)
                    or any(node.span not in spans or node.operation not in operation_field.labels
                        for node in proposal.nodes)):
                raise ValueError("complete program is outside the native operation grammar")
            for graph in proposal.graphs:
                if (graph.comparison.get("status") not in {"equivalent", "different"}
                        or len(graph.indices) != len(chart.options)
                        or not math.isfinite(graph.baseline_score)):
                    raise ValueError("complete program has unproved labels or nonfinite score")
                score, definitions = -chart.choice_log_normalizer, set()
                for step, (node, indices) in enumerate(zip(chart.options, graph.indices, strict=True)):
                    if len(node) != len(indices):
                        raise ValueError("complete program role indices differ")
                    for slot, (pool, index) in enumerate(zip(node, indices, strict=True)):
                        if type(index) is not int or not 0 <= index < len(pool):
                            raise ValueError("complete program option identity absent")
                        value, register, _span = pool[index]
                        score += value
                        if chart.definition_options is not None:
                            span = chart.definition_options[step][slot][index]
                            if span is not None:
                                definitions.add((register, span))
                score += sum((chart.definition_scores or {}).get(key, 0.) for key in definitions)
                if abs(score - graph.baseline_score) > 1e-5 * (1. + abs(score)):
                    raise ValueError("complete program baseline differs from selected public factors")
                chart.certify_selection(graph.indices)
                positives += graph.comparison["status"] == "equivalent"
                negatives += graph.comparison["status"] == "different"
        if not positives:
            raise ValueError("complete-program fitting needs a proven positive")

    def source_loss(self, pointer, operation_field, states, *, nuisance_projection=None):
        graph_scores, positives = [], []
        operation_spans = tuple(dict.fromkeys(node.span for proposal in self.charts for node in proposal.nodes))
        operation_energy = operation_field(states, operation_spans)
        span_indices = {span: index for index, span in enumerate(operation_spans)}
        labels = {label: index for index, label in enumerate(operation_field.labels)}
        for proposal in self.charts:
            edges = program_chart_edge_scores(pointer, states, proposal, source_id=self.source_id,
                input_spans=self.input_spans, inputs=self.inputs, nuisance_projection=nuisance_projection)
            offsets, updates, offset = [], [], 0
            for node in proposal.chart.options:
                node_offsets = []
                for slot in node:
                    baseline = mx.array([score for score, _register, _span in slot])
                    updated = conditional_choice_scores(baseline, edges[offset:offset + len(slot)])
                    updates.append(updated - baseline)
                    node_offsets.append(len(updates) - 1)
                    offset += len(slot)
                offsets.append(node_offsets)
            op_score = sum((operation_energy[span_indices[node.span], labels[node.operation]]
                for node in proposal.nodes), mx.array(0.))
            for graph in proposal.graphs:
                delta = sum((updates[offsets[step][slot]][index]
                    for step, node in enumerate(graph.indices) for slot, index in enumerate(node)), mx.array(0.))
                if graph.comparison["status"] == "equivalent":
                    positives.append(len(graph_scores))
                graph_scores.append(op_score + graph.baseline_score + delta)
        if not positives:
            raise ValueError("complete-program fitting needs a proven positive")
        scores = mx.stack(graph_scores)
        return mx.logsumexp(scores) - mx.logsumexp(scores[mx.array(positives, dtype=mx.int32)])


def _source_program_context(parent, item, max_charts):
    from core.learning.semantic_graph_counterexamples import argument_graph_program
    from core.learning.semantic_joint_graph_learning import align_source_input_registers
    from core.learning.semantic_program_transducer_fitting import _OperationNode

    if item.split != "train" or type(max_charts) is not int or max_charts < 1:
        raise ValueError("complete-program mining requires bounded source-training inputs")
    parent = parent.with_global_constraint_arguments().with_conditional_argument_choices()
    limit = parent.inference_step_limit(len(item.public_inputs))
    if limit is None:
        raise ValueError("complete-program public input count unsupported")
    spans, _scores, arguments, public = parent._runtime_operation_charts(
        item.ir.source_token_ids, item.hidden_states, item.public_inputs, limit)
    public = tuple(islice(public, max_charts))
    instructions, _mapping = align_source_input_registers(item, spans)
    order = sorted(range(len(instructions)), key=lambda index: instructions[index].operation_span.start)
    remap = {len(spans) + old: len(spans) + new for new, old in enumerate(order)}
    target_args = tuple(tuple(register if register < len(spans) else remap[register]
        for register in instructions[index].args) for index in order)
    target_nodes = tuple(_OperationNode(instructions[index].operation_span, instructions[index].op, 0., 0., 1.)
        for index in order)
    target = argument_graph_program(target_nodes, target_args, n_inputs=len(spans))
    return parent, spans, arguments, public, target_nodes, target_args, target


def revalidate_grounded_program_supervision(parent, item, programs, *, max_charts, max_graphs):
    """Rebuild public factors and reprove stored meanings before changing custody."""
    from core.learning.semantic_graph_counterexamples import (
        argument_graph_program,
        compare_program_meanings,
        counterfactual_inputs,
    )
    from core.learning.semantic_program_transducer_fitting import _assign_typed_arguments

    parent, spans, arguments, public, target_nodes, _args, target = _source_program_context(parent, item, max_charts)
    signature = lambda nodes: tuple((node.operation, node.span) for node in nodes)
    allowed = {signature(nodes) for nodes in (*public, target_nodes)}
    if (programs.source_id != item.ir.source_text_sha256 or programs.input_spans != spans
            or programs.inputs != item.public_inputs or len(programs.charts) > len(allowed)
            or len({signature(proposal.nodes) for proposal in programs.charts}) != len(programs.charts)):
        raise ValueError("retained program pool differs from current source context")
    probes = counterfactual_inputs(item.public_inputs)
    proposals, positives, negatives = [], 0, 0
    for proposal in programs.charts:
        if signature(proposal.nodes) not in allowed or not 1 <= len(proposal.graphs) <= max_graphs + 1:
            raise ValueError("retained program pool exceeds current public proposal bounds")
        captured = []
        _assign_typed_arguments(model=parent, hidden=item.hidden_states, inputs=item.public_inputs,
            source_token_ids=item.ir.source_token_ids, input_spans=spans, operation_nodes=proposal.nodes,
            argument_pointer_scores=arguments, chart_observer=captured.append, build_only=True)
        if len(captured) != 1:
            raise ValueError("retained program chart is absent from current public grammar")
        chart = captured[0]
        if (ProgramChartChoices(proposal.nodes, chart, ()).receipt()
                != replace(proposal, graphs=()).receipt()):
            raise ValueError("retained program factors differ from current public chart")
        graphs = []
        for graph in proposal.graphs:
            score, registers, _mentions, _dependencies = chart.certify_selection(graph.indices)
            if abs(score - graph.baseline_score) > 1e-5 * (1. + abs(score)):
                raise ValueError("retained program baseline differs from selected public factors")
            program = argument_graph_program(proposal.nodes, registers, n_inputs=len(spans))
            comparison = compare_program_meanings(target, program, probes)
            if (comparison["status"] not in {"equivalent", "different"}
                    or comparison["status"] != graph.comparison.get("status")):
                raise ValueError("retained program meaning was not independently reproved")
            positives += comparison["status"] == "equivalent"
            negatives += comparison["status"] == "different"
            graphs.append(replace(graph, comparison=comparison))
        proposals.append(replace(proposal, chart=chart, graphs=tuple(graphs)))
    if not positives:
        raise ValueError("retained program pool has no proven positive")
    mining = {**programs.mining, "proven_positive_graphs": positives, "witnessed_negative_graphs": negatives,
        "source_factors_rebuilt": True, "graph_constraints_rechecked": True,
        "program_meanings_reproved": True, "optimizer_search_repeated": False}
    return replace(programs, charts=tuple(proposals), mining=mining)


def mine_grounded_program_supervision(parent, item, *, max_charts=4, max_graphs=4, max_seconds=5.):
    """Freeze public competitors first; use source annotations only for labels.

    The source target chart is added for fitting and explicitly recorded. It
    is never supplied to public decoding. A finite probe cannot certify two
    programs equal; such undecided comparisons are excluded from this loss.
    """
    from core.learning.semantic_graph_counterexamples import (
        argument_graph_program,
        compare_program_meanings,
        counterfactual_inputs,
    )
    from core.learning.semantic_program_transducer_fitting import _assign_typed_arguments

    if type(max_graphs) is not int or max_graphs < 1 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("complete-program mining requires bounded source-training inputs")
    deadline = time.monotonic() + max_seconds
    def remaining():
        allowance = deadline - time.monotonic()
        if allowance <= 0:
            raise TimeoutError("complete-program source mining exhausted its allowance")
        return allowance
    parent, spans, arguments, public, target_nodes, target_args, target = _source_program_context(parent, item, max_charts)
    signature = lambda nodes: tuple((node.operation, node.span) for node in nodes)
    public_signatures = {signature(nodes) for nodes in public}
    # Source feasibility is required; further competitors are bounded witnesses.
    # The public proposal bank above is already frozen before reading the target.
    sets = dict((signature(nodes), nodes) for nodes in (target_nodes, *public))
    proposals, unknown, negatives, positives = [], 0, 0, 0
    searches, complete = [], True
    probes = counterfactual_inputs(item.public_inputs)
    for nodes in sets.values():
        if time.monotonic() >= deadline:
            complete = False
            searches.append({"phase": "chart_start", "status": "budget_exhausted"})
            break
        captured = []
        _assign_typed_arguments(model=parent, hidden=item.hidden_states, inputs=item.public_inputs,
            source_token_ids=item.ir.source_token_ids, input_spans=spans, operation_nodes=nodes,
            argument_pointer_scores=arguments, chart_observer=captured.append, build_only=True)
        if not captured:
            continue
        chart = captured[0]
        selections, excluded = [], []
        if signature(nodes) == signature(target_nodes):
            restricted = chart.restrict_arguments(target_args)
            indices = []
            result = restricted.solve(selection_observer=indices.append, time_limit_s=remaining())
            if result is None:
                raise ValueError("source meaning is unreachable in the complete public argument grammar")
            mapped = []
            for step, node in enumerate(indices[0]):
                slots = []
                for slot, index in enumerate(node):
                    option = restricted.options[step][slot][index]
                    definition = None if restricted.definition_options is None else restricted.definition_options[step][slot][index]
                    slots.append(next(i for i, candidate in enumerate(chart.options[step][slot])
                        if candidate == option and (chart.definition_options is None
                            or chart.definition_options[step][slot][i] == definition)))
                mapped.append(tuple(slots))
            selections.append((result, tuple(mapped)))
            excluded.append(result[1])
        for _ in range(max_graphs):
            indices = []
            from core.learning.semantic_argument_optimization import (
                ArgumentOptimizationIncompleteError,
            )

            try:
                result = chart.solve(excluded_graphs=excluded, selection_observer=indices.append,
                    time_limit_s=remaining())
            except (TimeoutError, ArgumentOptimizationIncompleteError) as exc:
                complete = False
                searches.append({"operations": [(node.operation, node.span.start, node.span.end)
                    for node in nodes], "phase": "competitor_search", "status": "incomplete",
                    "reason": str(exc), "completed_graphs_retained": len(selections)})
                break
            if result is None:
                break
            selections.append((result, indices[0]))
            excluded.append(result[1])
        graphs = []
        for result, indices in selections:
            program = argument_graph_program(nodes, result[1], n_inputs=len(spans))
            comparison = compare_program_meanings(target, program, probes)
            if comparison["status"] == "unknown":
                unknown += 1
                continue
            positives += comparison["status"] == "equivalent"
            negatives += comparison["status"] == "different"
            graphs.append(ProgramGraphChoice(indices, result[0], comparison))
        if graphs:
            proposals.append(ProgramChartChoices(nodes, chart, tuple(graphs)))
        if not complete:
            break
    mining = {"schema": "aura.grounded_program_mining.v1", "max_charts": max_charts,
        "max_graphs_per_chart": max_graphs, "public_charts": len(public),
        "source_target_chart_added": signature(target_nodes) not in public_signatures,
        "proven_positive_graphs": positives, "witnessed_negative_graphs": negatives,
        "informative_complete_program_contrast": bool(negatives),
        "requested_searches_completed": complete, "unfinished_searches": searches,
        "unknown_comparisons_excluded": unknown, "complete_grammar_partition": False,
        "target_available_to_runtime": False, "serving_authority": False}
    if not positives:
        raise ValueError("source complete-program pool has no proven positive")
    return GroundedProgramSupervision(item.ir.source_text_sha256, spans, item.public_inputs, tuple(proposals), mining)


def program_supervision_digest(sources):
    return hashlib.sha256(json.dumps([sources[key].receipt() for key in sorted(sources)],
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def program_objective_contract(sources, weight):
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("complete-program objective needs a positive finite weight")
    return {"schema": "aura.grounded_complete_program_objective.v1",
        "source_ids": sorted(sources), "source_pool_sha256": program_supervision_digest(sources),
        "weight": weight, "score": "native_operation_plus_conditioned_public_argument_factors",
        "partition": "bounded_witnessed_complete_program_pool", "runtime_score_function_shared": True,
        "all_programs_covered": False, "whole_program_calibration_proven": False,
        "mining": [{"source_id": key, **sources[key].mining} for key in sorted(sources)]}
