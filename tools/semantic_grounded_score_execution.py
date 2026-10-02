"""Factor unchanged pointer projections and certify dominated mention removal."""

import math
from collections import defaultdict
from dataclasses import replace

import mlx.core as mx

from core.learning.semantic_relational_pointer import RelationalBindingPointer


def conditional_role_update(baseline, learned, *, weight):
    """Normalize a multiplicative update of the complete baseline choice pool.

    u_i = b_i + w*s_i - (logsumexp(b+w*s) - logsumexp(b)).
    The fitted choice loss cannot identify a constant offset in one role's
    logits. That offset cancels here. Uniform evidence is exactly neutral,
    and the original slot's partition mass stays unchanged across charts.
    No target, selected answer or construction label defines this correction.
    """
    baseline, learned = tuple(baseline), tuple(learned)
    if (not baseline or len(baseline) != len(learned) or not math.isfinite(weight)
            or weight < 0 or not all(map(math.isfinite, (*baseline, *learned)))):
        raise ValueError("conditional update needs complete finite choice evidence")
    # Center before scaling so large, meaningless learned offsets do not
    # incur cancellation against an equally large partition function.
    center = max(learned)
    shifted = tuple(base + weight * (score - center) for base, score in zip(baseline, learned, strict=True))
    def log_partition(values):
        maximum = max(values)
        return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))
    correction = log_partition(shifted) - log_partition(baseline)
    updated = tuple(value - correction for value in shifted)
    if not all(map(math.isfinite, updated)):
        raise ValueError("conditional update overflowed finite choice evidence")
    return updated, {"policy": "conditional_likelihood_v1", "choice_count": len(baseline),
        "learned_center": center, "centered_log_partition_delta": correction,
        "weight": weight, "uniform_evidence_is_neutral": True, "target_available": False}


def conditional_chart_upper_bound(chart):
    """Bound any normalized learned update without looking at its logits.

    Each updated choice is at most the unchanged baseline log-partition.
    Relax all assignment constraints and add each register's positive maximum
    definition score once. The original chart normalizer still applies.
    This bound is invalid for an unnormalized additive learned score.
    """
    partitions = []
    for node in chart.options:
        for slot in node:
            values = tuple(option[0] for option in slot)
            if not values:
                return -math.inf
            if not all(map(math.isfinite, values)):
                raise ValueError("conditional chart bound needs finite baseline choices")
            maximum = max(values)
            partitions.append(maximum + math.log(math.fsum(math.exp(value - maximum) for value in values)))
    definitions = {}
    for (register, _span), score in (chart.definition_scores or {}).items():
        if not math.isfinite(score):
            raise ValueError("conditional chart bound needs finite definition scores")
        definitions[register] = max(definitions.get(register, 0.), score)
    return math.fsum((*partitions, *definitions.values(), -chart.choice_log_normalizer))


class PooledPointer(RelationalBindingPointer):
    """Replay the original relation function after its independent projections."""
    def __init__(self, owner):
        import mlx.nn as nn

        nn.Module.__init__(self)
        width = owner.relation_width
        self.hidden_width, self.depths = width, 1
        self.relation_width, self.rounds = width, owner.rounds
        self.role_queries, self.feature_blocks = owner.role_queries, owner.feature_blocks
        self.lora_depth_query = self.lora_depth_filler = mx.zeros((width, 1))
        self.lora_operation = self.lora_mention = self.lora_candidate = mx.eye(width)
        for name in ("lora_message", "lora_update", "lora_b"):
            setattr(self, name, getattr(owner, name))
        if owner.role_queries:
            self.lora_role_query = owner.lora_role_query


class ObservationEncoder:
    """Source-local immutable observations share only their independent encoding.

    The learned depth gate and nuisance projection run before the learned
    operation/mention/candidate matrix, exactly as in the original pointer.
    Graph messages and role-conditioned interactions remain separate per
    alternative. Cache entries retain their input object to prevent id reuse.
    """
    def __init__(self, pointer, projection=None):
        self.pointer = pointer
        self.basis = None if projection is None else mx.array(projection.basis.tolist())
        self.cache = {}
        self.replay = PooledPointer(pointer)

    def encode(self, values, kind):
        if kind not in {"operation", "mention", "candidate"} or not values:
            raise ValueError("pointer encoding needs a declared observation kind")
        missing = {id(value): value for value in values if (kind, id(value)) not in self.cache}
        if missing:
            observed = mx.stack(list(missing.values()))
            if self.basis is not None:
                if observed.shape[-1] != self.basis.shape[0]:
                    raise ValueError("factored grounded projection width differs")
                basis = self.basis.astype(observed.dtype)
                observed = observed - (observed @ basis) @ basis.T
            owner = self.pointer
            gate = owner.lora_depth_filler if kind == "candidate" else owner.lora_depth_query
            encoded = owner._pool(observed, gate) @ getattr(owner, "lora_" + kind)
            if not mx.all(mx.isfinite(encoded)).item():
                raise ValueError("factored pointer needs finite conditioned observations")
            for index, (identity, value) in enumerate(missing.items()):
                self.cache[kind, identity] = (value, encoded[index])
        return mx.stack([self.cache[kind, id(value)][1] for value in values])[:, None, :]

    def selected_scores(self, operations, mentions, candidates, *, rows, columns,
                        adjacency, role_features):
        """Read one edge per independent workspace after every graph update.

        Only the final pointwise relation features are gathered early. Every
        role and candidate still participates in the original message rounds.
        Those features do not feed another round or affect any other edge.
        Observations have already passed the original depth and matrix maps.
        """
        import mlx.nn as nn

        owner = self.pointer
        operation, mention, candidate = operations[:, :, 0, :], mentions[:, :, 0, :], candidates[:, :, 0, :]
        role_count = operation.shape[1]
        role_query = role_features @ owner.lora_role_query if owner.role_queries else None
        nodes = mx.concatenate([operation + mention + (0. if role_query is None else role_query), candidate], axis=1)
        if adjacency is not None:
            degree = mx.sum(adjacency.astype(mx.float32), axis=-1, keepdims=True)
            weights = adjacency.astype(mx.float32) / mx.maximum(degree, 1.)
            for _ in range(owner.rounds):
                message = weights @ (nodes @ owner.lora_message)
                update = mx.tanh(mx.concatenate([nodes, message], axis=-1) @ owner.lora_update)
                nodes = nodes + mx.where(degree > 0, update, 0.)
        index = mx.arange(operation.shape[0])
        o, m = operation[index, rows], mention[index, rows]
        c, contextual = nodes[index, role_count + columns], nodes[index, rows]
        if role_query is None:
            blocks = [o, m, c, o * m, o * c, m * c, o * m * c, contextual - c]
        else:
            r = role_query[rows]
            blocks = [o, r, m, c, o * r, o * m, o * c, r * m, r * c, m * c,
                o * r * m, o * r * c, o * m * c, r * m * c, o * r * m * c, contextual - c]
        return (nn.silu(mx.concatenate(blocks, axis=-1)) @ owner.lora_b).squeeze(-1)


def reduce_dominated_mentions(chart, *, remaining=None):
    """Remove only no-better choices with a superset of occupied mention tokens.

    A replacement must have the same slot, register and definition. It thus
    preserves use counts, definition consistency, dependencies and every
    excluded register graph. Its smaller token interval cannot create an
    overlap. All original choices have already received their learned score.
    Pair factors and retained factor observers are deliberately unsupported.
    """
    if chart.option_factors is not None or chart.option_relation_evidence is not None:
        raise ValueError("certified mention reduction needs plain scored options")
    options, definitions, witnesses = [], [], []
    original_scale = sum(abs(option[0]) for node in chart.options for slot in node for option in slot)
    for node_index, node in enumerate(chart.options):
        output_node, output_definitions = [], []
        for slot_index, slot in enumerate(node):
            if remaining is not None:
                remaining()
            names = (None,) * len(slot) if chart.definition_options is None else chart.definition_options[node_index][slot_index]
            groups = defaultdict(list)
            for index, (_score, register, _span) in enumerate(slot):
                groups[register, names[index]].append(index)
            retained = []
            for group in groups.values():
                ordered = sorted(group, key=lambda index: (
                    slot[index][2].end - slot[index][2].start, -slot[index][0], index))
                frontier = []
                for index in ordered:
                    score, _register, span = slot[index]
                    witness = next((other for other in frontier if slot[other][0] >= score
                        and span.start <= slot[other][2].start and slot[other][2].end <= span.end), None)
                    if witness is None:
                        frontier.append(index)
                        retained.append(index)
                    else:
                        witnesses.append((node_index, slot_index, index, witness))
            indices = sorted(retained)
            output_node.append(tuple(slot[index] for index in indices))
            output_definitions.append(tuple(names[index] for index in indices))
        options.append(tuple(output_node))
        definitions.append(tuple(output_definitions))
    reduced = replace(chart, options=tuple(options),
        definition_options=None if chart.definition_options is None else tuple(definitions))
    reduced_scale = sum(abs(option[0]) for node in reduced.options for slot in node for option in slot)
    # The contextual solve scales ambiguity tolerance by every option score.
    # Restore the original tolerance after removing redundant realizations.
    margin_adjustment = 1e-7 * max(0., original_scale - reduced_scale)
    return reduced, tuple(witnesses), margin_adjustment
