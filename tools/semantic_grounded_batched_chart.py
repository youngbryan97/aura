"""Batch the fitted pointer's public alternatives without changing its function.

Training code and checkpoint custody stay unchanged. This is an explicit
research replay execution variant, not a replacement checkpoint or serving path.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import mlx.core as mx

from core.learning.semantic_argument_optimization import ArgumentOptimizationIncompleteError
from core.learning.semantic_context_binding import BindingContext, BindingRole, ContextReferent
from core.learning.semantic_grounded_binding_engine import (
    GroundedBindingEvidence,
)
from core.learning.semantic_grounded_chart_bridge import GroundedBindingChartSolver
from core.learning.semantic_program_floor import semantic_primitive_type_signature
from core.learning.semantic_program_transducer_fitting import _operation_order
from core.learning.semantic_relational_pointer import semantic_role_features
from tools.semantic_grounded_score_execution import (
    ObservationEncoder,
    conditional_chart_upper_bound,
    conditional_role_update,
    reduce_dominated_mentions,
)


def execution_contract(score_policy="raw"):
    if score_policy not in {"raw", "conditional_likelihood"}:
        raise ValueError("undeclared grounded relation score policy")
    helper = Path(__file__).with_name("semantic_grounded_score_execution.py")
    return {"schema": "aura.grounded_batched_chart_execution.v5",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "pointer_function": "factored_fitted_pointer_selected_edges", "alternative_batch_size": 16,
        "all_graph_message_nodes_retained": True,
        "candidate_pruning": "certified_same_register_definition_token_subset_only",
        "chart_pruning": "baseline_partition_bound_v1" if score_policy == "conditional_likelihood" else "none",
        "relation_score_policy": score_policy,
        "checkpoint_mutation": False, "qualification_evidence": False,
        "serving_authority": False}


def conditioned_scores(pointer, evidence, alternatives, *, projection=None, batch_size=16, remaining=None, encoder=None):
    """Evaluate every (role, candidate, mention, definition) condition separately.

    Each graph gets its own complete workspace and the same contextual mask.
    Only unrequested final pointwise edge outputs are omitted; graph messages
    remain complete and the fitted relation function is unchanged.
    Batch size bounds transient replicated native states, not search reach.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 64 or not alternatives:
        raise ValueError("conditioned pointer needs nonempty alternatives and a bounded batch")
    evidence.arrays()
    encoder = ObservationEncoder(pointer, projection) if encoder is None else encoder
    operations = encoder.encode([evidence.operations[role.identity] for role in evidence.roles], "operation")
    mentions = encoder.encode([evidence.mentions[role.identity] for role in evidence.roles], "mention")
    candidates = encoder.encode([evidence.candidates[record.key] for record in evidence.context.referents], "candidate")
    roles = semantic_role_features(evidence.roles)
    node_count = len(evidence.roles) + len(evidence.context.referents)
    if evidence.adjacency is not None and (evidence.adjacency.shape != (node_count, node_count)
            or not mx.all(mx.isfinite(evidence.adjacency) & (evidence.adjacency >= 0)).item()):
        raise ValueError("relational pointer needs finite nonnegative graph evidence with aligned identities")
    values = []
    for start in range(0, len(alternatives), batch_size):
        if remaining is not None:
            remaining()
        selected = alternatives[start:start + batch_size]
        if any(type(row) is not int or not 0 <= row < len(evidence.roles)
                or type(column) is not int or not 0 <= column < len(evidence.context.referents)
                or not evidence.context.eligible(evidence.roles[row], evidence.context.referents[column])
                for row, column, _mention, _definition in selected):
            raise ValueError("batched pointer alternative has an inadmissible identity")
        count = len(selected)
        batch_mentions = mx.broadcast_to(mentions, (count, *mentions.shape))
        batch_candidates = mx.broadcast_to(candidates, (count, *candidates.shape))
        rows = mx.array([row for row, _column, _mention, _definition in selected], dtype=mx.int32)
        columns = mx.array([column for _row, column, _mention, _definition in selected], dtype=mx.int32)
        index = mx.arange(count)
        batch_mentions[index, rows] = encoder.encode([mention for _row, _column, mention, _definition in selected], "mention")
        definitions = encoder.encode([evidence.candidates[evidence.context.referents[column].key]
            if definition is None else definition for _row, column, _mention, definition in selected], "candidate")
        batch_candidates[index, columns] = definitions
        if not mx.all(mx.isfinite(batch_mentions)).item() or not mx.all(mx.isfinite(batch_candidates)).item():
            raise ValueError("batched pointer needs finite conditioned observations")
        scores = encoder.selected_scores(mx.broadcast_to(operations, (count, *operations.shape)),
            batch_mentions, batch_candidates, rows=rows, columns=columns,
            adjacency=evidence.adjacency, role_features=roles)
        measured = scores.tolist()
        if any(not math.isfinite(value) for value in measured):
            raise ValueError("batched pointer produced nonfinite relation evidence")
        values.extend(measured)
    return tuple(values)


class BatchedGroundedBindingChartSolver(GroundedBindingChartSolver):
    def __init__(self, *args, score_policy="raw", length_penalty=None, **kwargs):
        super().__init__(*args, **kwargs)
        execution_contract(score_policy)
        self.score_policy = score_policy
        if length_penalty is not None and (not math.isfinite(length_penalty) or length_penalty < 0):
            raise ValueError("joint chart bound needs the unchanged nonnegative operation penalty")
        self.length_penalty = length_penalty
        self.best_joint_score = -math.inf
        self.observed_spans = {}
        self.encoder = ObservationEncoder(self.engine.pointer, self.engine.nuisance_projection)

    def __call__(self, chart, *, operation_nodes, input_spans, inputs, source_id, time_limit_s=None):
        if source_id != self.source_id or len(operation_nodes) != len(chart.options) or len(inputs) != chart.n_inputs:
            raise ValueError("batched chart belongs to a different public source")
        allowance = self.max_seconds if time_limit_s is None else min(self.max_seconds, time_limit_s)
        deadline = time.monotonic() + allowance
        def remaining():
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise ArgumentOptimizationIncompleteError("grounded_chart_bridge_budget_exhausted")
            return seconds
        def observe(span):
            remaining()
            if span not in self.observed_spans:
                span.validate_bound(len(self.depth_states))
                self.observed_spans[span] = mx.mean(self.depth_states[span.start:span.end], axis=0)
            return self.observed_spans[span]
        operation_score, bound_evidence = None, {}
        if self.score_policy == "conditional_likelihood" and self.length_penalty is not None:
            operation_score = sum(node.score for node in operation_nodes) - self.length_penalty * len(operation_nodes)
            if not math.isfinite(operation_score):
                raise ValueError("joint chart bound needs finite operation scores")
            upper = conditional_chart_upper_bound(chart) + operation_score
            finite_incumbent = math.isfinite(self.best_joint_score)
            guard = 1e-6 * (1. + abs(upper) + abs(self.best_joint_score)) if finite_incumbent and math.isfinite(upper) else 0.
            bound_evidence = {"chart_upper_bound": upper if math.isfinite(upper) else None,
                "operation_chart_score": operation_score,
                "incumbent_joint_score": self.best_joint_score if finite_incumbent else None,
                "bound_roundoff_guard": guard}
            if finite_incumbent and math.isfinite(upper) and upper + guard < self.best_joint_score:
                self.last_resolution = {"status": "certified_pruned", "margin": None, "source_id": source_id,
                    "all_options_retained": False, **bound_evidence, "role_updates": (), "solver_option_count": 0,
                    "relation_score_policy": self.score_policy,
                    "reason": "complete_chart_cannot_exceed_verified_incumbent"}
                self.resolutions.append(self.last_resolution)
                return None
        signatures = [semantic_primitive_type_signature(node.operation) for node in operation_nodes]
        if any(signature is None for signature in signatures):
            raise ValueError("batched chart needs typed public operations")
        anchors = (*input_spans, *(node.span for node in operation_nodes))
        types = tuple("integer_sequence" if isinstance(value, tuple) else "integer" for value in inputs)
        types += tuple(signature[1] for signature in signatures)
        records = tuple(ContextReferent(source_id, f"register:{index}", types[index],
            f"source-token-span:{span.start}:{span.end}", scope=("program",)) for index, span in enumerate(anchors))
        context = BindingContext(records, {name: None for name in types})
        register_keys = {index: record.key for index, record in enumerate(records)}
        roles, operations, mentions, addresses = [], {}, {}, []
        for step, (node, slots, signature) in enumerate(zip(operation_nodes, chart.options, signatures, strict=True)):
            if len(slots) != len(signature[0]) or any(not pool for pool in slots):
                self.last_resolution = {"status": "infeasible", "margin": None}
                return None
            for slot, (pool, required) in enumerate(zip(slots, signature[0], strict=True)):
                role = BindingRole(f"argument:{step}:{slot}", f"{node.operation}:operand:{slot}", required,
                    scope=("program",), referents=tuple(key for register, key in register_keys.items()
                        if register != chart.n_inputs + step))
                roles.append(role)
                operations[role.identity] = observe(node.span)
                weights = mx.softmax(mx.array([score for score, _register, _span in pool]))
                mentions[role.identity] = mx.sum(mx.stack([observe(span) for _score, _register, span in pool])
                    * weights[:, None, None], axis=0)
                addresses.append((step, slot))
        candidates = {record.key: observe(span) for record, span in zip(records, anchors, strict=True)}
        allowed = mx.array([[context.eligible(role, record) for record in records] for role in roles])
        adjacency = mx.zeros((len(roles) + len(records),) * 2)
        adjacency[:len(roles), len(roles):] = allowed.astype(mx.float32)
        adjacency[len(roles):, :len(roles)] = allowed.T.astype(mx.float32)
        evidence = GroundedBindingEvidence(source_id, context, tuple(roles), operations, mentions, candidates, adjacency)
        alternatives, locations = [], []
        for row, ((step, slot), role) in enumerate(zip(addresses, roles, strict=True)):
            for option_index, (_score, register, span) in enumerate(chart.options[step][slot]):
                if type(register) is not int or register not in register_keys or not context.eligible(role, records[register]):
                    raise ValueError("batched chart contains an inadmissible register")
                definition = None if chart.definition_options is None else chart.definition_options[step][slot][option_index]
                alternatives.append((row, register, observe(span), None if definition is None else observe(definition)))
                locations.append((step, slot, option_index, role, register, span))
        learned = conditioned_scores(self.engine.pointer, evidence, alternatives,
            projection=self.engine.nuisance_projection, remaining=remaining, encoder=self.encoder)
        conditioned = [[[] for _slot in node] for node in chart.options]
        for value, (step, slot, *_location) in zip(learned, locations, strict=True):
            conditioned[step][slot].append(value)
        policies, updated = {}, [[[] for _slot in node] for node in chart.options]
        for step, node in enumerate(chart.options):
            for slot, pool in enumerate(node):
                if self.score_policy == "conditional_likelihood":
                    values, policy = conditional_role_update([option[0] for option in pool],
                        conditioned[step][slot], weight=self.engine.evidence_weight)
                    policies[step, slot] = policy
                else:
                    values = tuple(option[0] + self.engine.evidence_weight * value
                        for option, value in zip(pool, conditioned[step][slot], strict=True))
                updated[step][slot] = [(value, register, span)
                    for value, (_baseline, register, span) in zip(values, pool, strict=True)]
        edge_receipts = []
        for value, (step, slot, option_index, role, register, span) in zip(learned, locations, strict=True):
            score = chart.options[step][slot][option_index][0]
            combined = updated[step][slot][option_index][0]
            edge_receipts.append({"operation_id": step, "operation": operation_nodes[step].operation,
                "role_instance": role.identity, "role_id": role.role, "required_type": role.type_name,
                "candidate_id": register_keys[register], "candidate_source": records[register].source,
                "mention_span": (span.start, span.end), "baseline_score": score,
                "learned_relation_score": value, "evidence_weight": self.engine.evidence_weight,
                "combined_score": combined})
        augmented = replace(chart, options=tuple(tuple(tuple(pool) for pool in node) for node in updated),
            option_factors=None, option_relation_evidence=None)
        augmented, witnesses, margin_adjustment = reduce_dominated_mentions(augmented, remaining=remaining)
        nested_roles, offset = [], 0
        for node in chart.options:
            nested_roles.append(tuple(roles[offset:offset + len(node)]))
            offset += len(node)
        resolution = augmented.solve_grounded(context, tuple(nested_roles), register_keys,
            minimum_margin=self.minimum_margin + margin_adjustment, time_limit_s=remaining())
        self.last_resolution = {"status": resolution.status, "margin": resolution.margin, "source_id": source_id,
            **bound_evidence,
            "all_options_retained": True, "role_bindings": resolution.bindings, "edge_evidence": edge_receipts,
            "margin_is_probability": False, "pointer_execution": "factored_conditioned_selected_edges_v4",
            "relation_score_policy": self.score_policy,
            "role_updates": tuple({"operation_id": step, "slot": slot, **policy}
                for (step, slot), policy in policies.items()),
            "dominated_mention_witnesses": witnesses, "ambiguity_margin_adjustment": margin_adjustment,
            "solver_option_count": sum(len(slot) for node in augmented.options for slot in node)}
        if resolution.assignment is not None:
            self.last_resolution["graph_signature"] = tuple(sorted((node.operation, node.span.start, node.span.end,
                tuple((anchors[register].start, anchors[register].end) for register in values))
                for node, values in zip(operation_nodes, resolution.assignment[1], strict=True)))
            self.last_resolution["argument_graph_score"] = resolution.assignment[0]
            if resolution.status == "bound" and operation_score is not None:
                _score, arguments, _spans, dependencies = resolution.assignment
                order = _operation_order(dependencies, operation_nodes, require_connected=True)
                referenced = {dependency for values in dependencies for dependency in values}
                sinks = tuple(index for index in range(len(operation_nodes)) if index not in referenced)
                accepted = (order is not None and len(sinks) == 1 and chart.contract.accepts_complete(
                    Counter(register for values in arguments for register in values), n_inputs=chart.n_inputs,
                    operation_count=len(operation_nodes), sink=sinks[0]))
                self.last_resolution["parent_assignment_checks_passed"] = accepted
                if accepted:
                    joint_score = resolution.assignment[0] + operation_score
                    if not math.isfinite(joint_score):
                        raise ValueError("joint chart incumbent needs a finite complete score")
                    self.best_joint_score = max(self.best_joint_score, joint_score)
        self.resolutions.append(self.last_resolution)
        return resolution.assignment if resolution.status == "bound" else None


class BatchedNativeChartDecoder:
    def __init__(self, verified_decoder, *, score_policy="raw"):
        self.owner = verified_decoder
        execution_contract(score_policy)
        self.score_policy = score_policy
        self.last_receipt = None

    def decode(self, *, source_token_ids, hidden_states, public_inputs, source_text_sha256,
               model_basis_sha256, search_time_limit_s=10.):
        import json

        from tools.probe_semantic_native_prefix_branches import installed_arithmetic_basis
        from tools.semantic_grounded_native_decode import selected_chart_receipt

        self.last_receipt = None
        owner, tokens = self.owner, tuple(source_token_ids)
        if owner.engine.operation_field is not None and self.score_policy != "conditional_likelihood":
            raise ValueError("native operation field requires the conditional binding score policy")
        if (not tokens or len(tokens) > owner.plan["max_tokens"]
                or any(type(token) is not int or token < 0 for token in tokens)
                or type(search_time_limit_s) not in (int, float)
                or not math.isfinite(search_time_limit_s) or search_time_limit_s <= 0
                or model_basis_sha256 != owner.parent.model_basis_sha256
                or owner.plan["installed_arithmetic"] != installed_arithmetic_basis()):
            raise ValueError("batched chart needs unchanged arithmetic and bounded public inputs")
        started = time.monotonic()
        hidden = owner.prefix.capture(mx.array([tokens], dtype=mx.int32))
        states = mx.stack(owner.suffix.layer_states(hidden, tuple(range(owner.plan["suffix_layers"]))), axis=2)[0]
        mx.eval(states)
        remaining = search_time_limit_s - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("batched chart acquisition exhausted its allowance")
        bridge = BatchedGroundedBindingChartSolver(owner.engine, source_text_sha256, states,
            max_seconds=remaining, score_policy=self.score_policy, length_penalty=owner.parent.operation_length_penalty)
        proposer = (owner.engine.operation_field.proposal(source_text_sha256, states)
                    if owner.engine.operation_field is not None else None)
        outcome = owner.parent.decode(source_token_ids=tokens, hidden_states=hidden_states, public_inputs=public_inputs,
            source_text_sha256=source_text_sha256, model_basis_sha256=model_basis_sha256,
            search_time_limit_s=remaining, binding_chart_solver=bridge, operation_chart_proposer=proposer)
        self.last_receipt = {"schema": "aura.grounded_native_chart_decode.v1",
            "fit_receipt_sha256": owner.verification["fit_receipt_sha256"],
            "weights_sha256": owner.verification["weights_sha256"], "source_id": source_text_sha256,
            "source_tokens_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            "native_state_shape": list(states.shape), "source_only_native_capture": True,
            "parent_receipt_sha256": owner.parent.receipt_sha256, "elapsed_seconds": time.monotonic() - started,
            "selected_chart": selected_chart_receipt(bridge.resolutions, outcome),
            "examined_charts": len(bridge.resolutions), "refusal": outcome.refusal,
            "selected_step": owner.verification["selected_step"],
            "learned_checkpoint_selected": owner.verification["learned_checkpoint_selected"],
            "operation_proposal": proposer.last_receipt if proposer is not None else None,
            "target_available_to_decoder": False, "execution": execution_contract(self.score_policy),
            "chart_diagnostics": tuple({key: resolution.get(key) for key in (
                "status", "margin", "graph_signature", "argument_graph_score", "relation_score_policy",
                "role_updates", "solver_option_count", "chart_upper_bound", "operation_chart_score",
                "incumbent_joint_score", "bound_roundoff_guard", "parent_assignment_checks_passed",
                "reason")} for resolution in bridge.resolutions),
            "serving_authority": False}
        return outcome
