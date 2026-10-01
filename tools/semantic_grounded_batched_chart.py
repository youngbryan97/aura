"""Batch the fitted pointer's public alternatives without changing its function.

Training code and checkpoint custody stay unchanged. This is an explicit
research replay execution variant, not a replacement checkpoint or serving path.
"""

from __future__ import annotations

import hashlib
import math
import time
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
from core.learning.semantic_relational_pointer import semantic_role_features


def execution_contract():
    return {"schema": "aura.grounded_batched_chart_execution.v1",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pointer_function": "unchanged_fitted_pointer_vmap", "alternative_batch_size": 16,
        "candidate_pruning": False, "checkpoint_mutation": False, "qualification_evidence": False,
        "serving_authority": False}


def conditioned_scores(pointer, evidence, alternatives, *, projection=None, batch_size=16, remaining=None):
    """Evaluate every (role, candidate, mention, definition) condition separately.

    The original pointer is vmapped, not approximated by a second scorer.
    Each graph gets its own complete workspace and the same contextual mask.
    Batch size bounds transient replicated native states, not search reach.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 64 or not alternatives:
        raise ValueError("conditioned pointer needs nonempty alternatives and a bounded batch")
    operations, mentions, candidates = evidence.arrays()[0]
    roles = semantic_role_features(evidence.roles)
    allowed = mx.array([[evidence.context.eligible(role, record) for record in evidence.context.referents]
        for role in evidence.roles], dtype=mx.bool_)
    projection_basis = None if projection is None else mx.array(projection.basis.tolist())
    def project(value):
        if projection_basis is None:
            return value
        if value.shape[-1] != projection_basis.shape[0]:
            raise ValueError("batched grounded projection width differs")
        basis = projection_basis.astype(value.dtype)
        return value - (value @ basis) @ basis.T
    operations = project(operations)
    scorer = mx.vmap(lambda operation, mention, candidate: pointer(operation, mention, candidate,
        adjacency=evidence.adjacency, allowed=allowed, role_features=roles))
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
        batch_mentions[index, rows] = mx.stack([mention for _row, _column, mention, _definition in selected])
        batch_candidates[index, columns] = mx.stack([candidates[column] if definition is None else definition
            for _row, column, _mention, definition in selected])
        batch_mentions, batch_candidates = project(batch_mentions), project(batch_candidates)
        if not mx.all(mx.isfinite(batch_mentions)).item() or not mx.all(mx.isfinite(batch_candidates)).item():
            raise ValueError("batched pointer needs finite conditioned observations")
        scores = scorer(mx.broadcast_to(operations, (count, *operations.shape)), batch_mentions, batch_candidates)
        measured = scores[index, rows, columns].tolist()
        if any(not math.isfinite(value) for value in measured):
            raise ValueError("batched pointer produced nonfinite relation evidence")
        values.extend(measured)
    return tuple(values)


class BatchedGroundedBindingChartSolver(GroundedBindingChartSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observed_spans = {}

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
            projection=self.engine.nuisance_projection, remaining=remaining)
        updated = [[[] for _slot in node] for node in chart.options]
        edge_receipts = []
        for value, (step, slot, option_index, role, register, span) in zip(learned, locations, strict=True):
            score = chart.options[step][slot][option_index][0]
            combined = score + self.engine.evidence_weight * value
            updated[step][slot].append((combined, register, span))
            edge_receipts.append({"operation_id": step, "operation": operation_nodes[step].operation,
                "role_instance": role.identity, "role_id": role.role, "required_type": role.type_name,
                "candidate_id": register_keys[register], "candidate_source": records[register].source,
                "mention_span": (span.start, span.end), "baseline_score": score,
                "learned_relation_score": value, "evidence_weight": self.engine.evidence_weight,
                "combined_score": combined})
        augmented = replace(chart, options=tuple(tuple(tuple(pool) for pool in node) for node in updated),
            option_factors=None, option_relation_evidence=None)
        nested_roles, offset = [], 0
        for node in chart.options:
            nested_roles.append(tuple(roles[offset:offset + len(node)]))
            offset += len(node)
        resolution = augmented.solve_grounded(context, tuple(nested_roles), register_keys,
            minimum_margin=self.minimum_margin, time_limit_s=remaining())
        self.last_resolution = {"status": resolution.status, "margin": resolution.margin, "source_id": source_id,
            "all_options_retained": True, "role_bindings": resolution.bindings, "edge_evidence": edge_receipts,
            "margin_is_probability": False, "pointer_execution": "vmap_conditioned_alternatives_v1"}
        if resolution.assignment is not None:
            self.last_resolution["graph_signature"] = tuple(sorted((node.operation, node.span.start, node.span.end,
                tuple((anchors[register].start, anchors[register].end) for register in values))
                for node, values in zip(operation_nodes, resolution.assignment[1], strict=True)))
            self.last_resolution["argument_graph_score"] = resolution.assignment[0]
        self.resolutions.append(self.last_resolution)
        return resolution.assignment if resolution.status == "bound" else None


class BatchedNativeChartDecoder:
    def __init__(self, verified_decoder):
        self.owner = verified_decoder
        self.last_receipt = None

    def decode(self, *, source_token_ids, hidden_states, public_inputs, source_text_sha256,
               model_basis_sha256, search_time_limit_s=10.):
        import json

        from tools.probe_semantic_native_prefix_branches import installed_arithmetic_basis
        from tools.semantic_grounded_native_decode import selected_chart_receipt

        self.last_receipt = None
        owner, tokens = self.owner, tuple(source_token_ids)
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
        bridge = BatchedGroundedBindingChartSolver(owner.engine, source_text_sha256, states, max_seconds=remaining)
        outcome = owner.parent.decode(source_token_ids=tokens, hidden_states=hidden_states, public_inputs=public_inputs,
            source_text_sha256=source_text_sha256, model_basis_sha256=model_basis_sha256,
            search_time_limit_s=remaining, binding_chart_solver=bridge)
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
            "target_available_to_decoder": False, "execution": execution_contract(), "serving_authority": False}
        return outcome
