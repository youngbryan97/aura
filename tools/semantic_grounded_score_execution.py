"""Factor unchanged pointer projections and certify dominated mention removal."""

from collections import defaultdict
from dataclasses import replace

import mlx.core as mx

from core.learning.semantic_relational_pointer import RelationalBindingPointer


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
