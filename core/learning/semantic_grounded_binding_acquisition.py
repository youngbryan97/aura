"""Reuse acquired source states and proposed spans for contextual binding.

Evidence construction never reads an instruction's target argument registers.
The separate supervision constructor may read them on source fit folds only.
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from core.learning.semantic_context_binding import BindingContext, BindingRole, ContextReferent
from core.learning.semantic_grounded_binding_engine import (
    GroundedBindingEvidence,
    GroundedBindingSupervision,
)
from core.learning.semantic_program_floor import semantic_primitive_type_signature


def grounded_evidence_from_source_example(item, *, channels=("middle_causal_hidden",)):
    """At inference, item.ir must contain the parser's proposed frame/spans.

    Known operation names/types constrain candidates, not their gold binding.
    Cached channels are used as recorded; one depth is never copied and
    relabeled as several measured native depths.
    """
    widths = dict(zip(item.hidden_channels, item.hidden_channel_widths, strict=True))
    if (not channels or len(set(channels)) != len(channels) or any(name not in widths for name in channels)
            or len({widths[name] for name in channels}) != 1
            or sum(item.hidden_channel_widths) != item.hidden_states.shape[1]):
        raise ValueError("grounded acquisition needs recorded equal-width source channels")
    offsets, start = {}, 0
    for name, width in zip(item.hidden_channels, item.hidden_channel_widths, strict=True):
        offsets[name] = start, start + width
        start += width
    ir, namespace = item.ir, item.ir.source_text_sha256
    signatures = tuple(semantic_primitive_type_signature(instruction.op) for instruction in ir.instructions)
    if any(signature is None for signature in signatures):
        raise ValueError("grounded acquisition needs typed proposed operations")
    anchors = (item.register_definition_spans or
               (*ir.input_spans, *(instruction.operation_span for instruction in ir.instructions)))
    types = tuple("integer_sequence" if isinstance(value, tuple) else "integer"
                  for value in item.public_inputs) + tuple(signature[1] for signature in signatures)
    if len(anchors) != len(types) or len(item.public_inputs) != ir.n_inputs:
        raise ValueError("grounded register definitions differ from the proposed source graph")
    keys = {index: (namespace, f"register:{index}") for index in range(len(anchors))}
    records = tuple(ContextReferent(namespace, keys[index][1], types[index],
        f"source-token-span:{span.start}:{span.end}", scope=("program",)) for index, span in enumerate(anchors))
    context = BindingContext(records, {kind: None for kind in types})

    def observe(span):
        span.validate_bound(len(item.hidden_states))
        return mx.array(np.stack([np.mean(item.hidden_states[span.start:span.end, slice(*offsets[name])], axis=0)
                                  for name in channels]).astype(np.float32))

    roles, operations, mentions = [], {}, {}
    for step, (instruction, signature) in enumerate(zip(ir.instructions, signatures, strict=True)):
        if len(instruction.argument_spans) != len(signature[0]):
            raise ValueError("grounded proposed argument spans differ from operation arity")
        for slot, (span, required) in enumerate(zip(instruction.argument_spans, signature[0], strict=True)):
            identity = f"argument:{step}:{slot}"
            # Program proposals may be unordered DAGs. Do not impose source
            # token order as semantic time; the existing SSA solver owns it.
            candidates = tuple(key for index, key in keys.items() if index != ir.n_inputs + step)
            roles.append(BindingRole(identity, f"{instruction.op}:operand:{slot}", required,
                                     scope=("program",), referents=candidates))
            operations[identity], mentions[identity] = observe(instruction.operation_span), observe(span)
    candidates = {keys[index]: observe(span) for index, span in enumerate(anchors)}
    count = len(roles) + len(records)
    adjacency = np.zeros((count, count), dtype=np.float32)
    for row, role in enumerate(roles):
        for column, record in enumerate(records):
            if context.eligible(role, record):
                adjacency[row, len(roles) + column] = adjacency[len(roles) + column, row] = 1.
    evidence = GroundedBindingEvidence(namespace, context, tuple(roles), operations, mentions,
                                      candidates, mx.array(adjacency))
    evidence.arrays()
    return evidence, keys


def grounded_supervision_from_source_example(item, *, channels=("middle_causal_hidden",), graph_limit=32,
                                            equivalence_supervision=False):
    """Read source-only teacher identities, never expected execution values."""
    if (item.split not in {"train", "validation"} or type(graph_limit) is not int or graph_limit < 1
            or type(equivalence_supervision) is not bool):
        raise ValueError("grounded source supervision cannot consume test targets")
    evidence, keys = grounded_evidence_from_source_example(item, channels=channels)
    positives = tuple((keys[register],) for instruction in item.ir.instructions for register in instruction.args)
    target = tuple(positive[0] for positive in positives)
    alternatives = [target]
    positive_graphs = [0]
    from core.learning.semantic_graph_counterexamples import (
        argument_graph_order,
        argument_graph_program,
        compare_program_meanings,
        counterfactual_inputs,
    )
    nodes = tuple(SimpleNamespace(operation=instruction.op, span=instruction.operation_span)
                  for instruction in item.ir.instructions)
    def arguments_for(graph):
        registers = {key: index for index, key in keys.items()}
        cursor, arguments = 0, []
        for instruction in item.ir.instructions:
            arity = len(instruction.argument_spans)
            arguments.append(tuple(registers[key] for key in graph[cursor:cursor + arity]))
            cursor += arity
        return tuple(arguments)

    def feasible(graph):
        try:
            argument_graph_order(nodes, arguments_for(graph), n_inputs=item.ir.n_inputs)
        except ValueError:
            return False  # not a failure: a graph with no order is the answer no
        return True
    if not feasible(target):
        raise ValueError("source-positive binding graph violates the existing SSA contract")

    def append_graph(graph):
        if graph in alternatives or not feasible(graph) or len(alternatives) >= graph_limit:
            return
        if equivalence_supervision:
            program = argument_graph_program(nodes, arguments_for(graph), n_inputs=item.ir.n_inputs)
            comparison = compare_program_meanings(item.ir.to_program(), program,
                counterfactual_inputs(item.public_inputs, count=8))
            if comparison["status"] == "equivalent":
                positive_graphs.append(len(alternatives))
            elif comparison["status"] != "different" or comparison.get("witness") is None:
                return
        alternatives.append(graph)

    # Joint role reversals can remain valid where changing either edge alone
    # disconnects the graph. Include them before less informative local edits.
    offset = 0
    for instruction in item.ir.instructions:
        arity = len(instruction.argument_spans)
        for left in range(arity):
            for right in range(left + 1, arity):
                graph = list(target)
                graph[offset + left], graph[offset + right] = graph[offset + right], graph[offset + left]
                if all(evidence.context.eligible(role, next(record for record in evidence.context.referents
                       if record.key == key)) for role, key in zip(evidence.roles, graph, strict=True)):
                    append_graph(tuple(graph))
        offset += arity
    for row, role in enumerate(evidence.roles):
        if len(alternatives) >= graph_limit:
            break
        for record in evidence.context.referents:
            if record.key != target[row] and evidence.context.eligible(role, record):
                changed = list(target)
                changed[row] = record.key
                graph = tuple(changed)
                append_graph(graph)
                if len(alternatives) >= graph_limit:
                    break
        if len(alternatives) >= graph_limit:
            break
    if equivalence_supervision:
        positives = tuple(tuple(dict.fromkeys(alternatives[index][row] for index in positive_graphs))
                          for row in range(len(evidence.roles)))
    supervision = GroundedBindingSupervision(evidence, positives, tuple(alternatives), tuple(positive_graphs),
                                             environment=item.construction_id.split(":", 1)[0])
    supervision.indices()
    return supervision


def grounded_source_equivariance_pairs(items, examples):
    """Use existing same-relation pairing, with stricter explicit slot alignment.

    A shared canonical polynomial is insufficient to align arbitrary nodes.
    Here the exact program, input values, node arities and admitted identities
    must match. Unsupported correspondences remain absent, not guessed.
    """
    from core.learning.semantic_counterfactual_corpus import cross_construction_relation_partners
    from core.learning.semantic_grounded_binding_engine import GroundedEquivariancePair

    items = tuple(items)
    records = {item.ir.source_text_sha256: item for item in items}
    sources = {item.evidence.source_id: item for item in examples}
    if set(records) != set(sources) or any(item.split != "train" for item in items):
        raise ValueError("grounded equivariance may use only the exact source fit cohort")
    partners = cross_construction_relation_partners(items)
    pairs, seen = [], set()
    for left, right in sorted(partners.items()):
        identity = tuple(sorted((left, right)))
        if identity in seen:
            continue
        seen.add(identity)
        first, second = records[left], records[right]
        if first.ir.to_program() != second.ir.to_program() or first.public_inputs != second.public_inputs:
            continue
        a, b = sources[left].evidence, sources[right].evidence
        if ([role.role for role in a.roles] != [role.role for role in b.roles]
                or len(a.context.referents) != len(b.context.referents)):
            continue
        pair = GroundedEquivariancePair(left, right,
            tuple((role.identity, other.identity) for role, other in zip(a.roles, b.roles, strict=True)),
            tuple((record.key, other.key) for record, other in zip(a.context.referents, b.context.referents, strict=True)))
        pair.orders(sources)
        pairs.append(pair)
    return tuple(pairs)


def resolve_source_argument_chart(engine, item, chart, *, channels=("middle_causal_hidden",), **options):
    """Rank the actual proposal chart, retaining all of its structural gates."""
    evidence, keys = grounded_evidence_from_source_example(item, channels=channels)
    return engine.resolve_chart(chart, evidence, keys, **options)
