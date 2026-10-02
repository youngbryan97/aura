"""Immutable scored charts and offline attribution of semantic graph failures."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from core.learning import semantic_argument_optimization
from core.learning.semantic_program_ir import TokenSpan
from core.verify.invariants import invariant

if TYPE_CHECKING:
    from core.learning.semantic_program_transducer_fitting import RegisterUseContract


def select_operation_argument_graph(
    charts: Any,
    assign: Any,
    *,
    length_penalty: Any,
    joint: bool=False,
    bounded_assign: Any=None,
) -> Any:
    """Select complete graphs under a declared factor score, or replay legacy order."""
    selected, best = None, -math.inf
    for chart in charts:
        operation_score = sum(node.score for node in chart) - length_penalty * len(chart)
        candidate = (
            bounded_assign(chart, best - operation_score)
            if joint and bounded_assign is not None else assign(chart)
        )
        if candidate is None:
            continue
        if not joint:
            return candidate
        score = candidate.score + operation_score
        if not math.isfinite(score):
            raise ValueError('nonfinite joint operation-argument score')
        if score > best:
            selected, best = candidate, score
    return selected


@dataclass(frozen=True)
class ScoredArgumentChart:
    options: tuple
    n_inputs: int
    contract: RegisterUseContract
    definition_options: tuple | None = None
    definition_scores: Mapping[tuple[int, TokenSpan], float] | None = None
    prune_dominated: bool = False
    option_factors: tuple | None = None
    option_relation_evidence: tuple | None = None
    choice_log_normalizer: float = 0.

    def __post_init__(self) -> None:
        if not math.isfinite(self.choice_log_normalizer):
            raise ValueError("argument choice normalizer must be finite")
        object.__setattr__(self, "options", tuple(
            tuple(tuple(tuple(option) for option in slot) for slot in node)
            for node in self.options
        ))
        if self.definition_options is not None:
            object.__setattr__(self, "definition_options", tuple(
                tuple(tuple(slot) for slot in node) for node in self.definition_options
            ))
        if self.definition_scores is not None:
            object.__setattr__(self, "definition_scores", MappingProxyType(dict(self.definition_scores)))
        if self.option_factors is not None:
            factors = tuple(tuple(tuple(tuple(float(value) for value in row) for row in slot)
                                  for slot in node) for node in self.option_factors)
            widths = {len(row) for node in factors for slot in node for row in slot}
            if len(factors) != len(self.options) or any(
                len(node) != len(options) or any(
                    len(slot) != len(choices) or any(not all(map(math.isfinite, row))
                                                   for row in slot)
                    for slot, choices in zip(node, options)
                ) for node, options in zip(factors, self.options)
            ) or len(widths) != 1 or next(iter(widths)) not in (4, 5):
                raise ValueError("argument score factors differ from chart")
            object.__setattr__(self, "option_factors", factors)
        if self.option_relation_evidence is not None:
            evidence = tuple(tuple(tuple(slot) for slot in node) for node in self.option_relation_evidence)
            if len(evidence) != len(self.options) or any(
                len(node) != len(options) or any(len(slot) != len(choices)
                    for slot, choices in zip(node, options))
                for node, options in zip(evidence, self.options)
            ):
                raise ValueError("relation evidence differs from chart")
            object.__setattr__(self, "option_relation_evidence", evidence)

    def score_upper_bound(self) -> float:
        """Relax consistency and overlap, retaining every potentially positive term."""
        maxima = []
        for node in self.options:
            for slot in node:
                if not slot:
                    return -math.inf
                maxima.append(max(option[0] for option in slot))
        per_register = {}
        for (register, _span), score in (self.definition_scores or {}).items():
            per_register[register] = max(per_register.get(register, 0.0), score)
        return math.fsum((*maxima, *per_register.values(), -self.choice_log_normalizer))

    def with_conditional_choices(self) -> Any:
        """Normalize the full local choice pools before restricting any target.

        This is a product of local categorical factors, not the partition
        function of the globally constrained graph. Constraints still select
        the highest-scoring valid assignment under that product.
        """
        from scipy.special import logsumexp

        slots = [slot for node in self.options for slot in node]
        if not slots or any(not slot for slot in slots):
            raise ValueError("conditional argument choices require nonempty slots")
        normalizer = math.fsum(float(logsumexp([option[0] for option in slot])) for slot in slots)
        return replace(self, choice_log_normalizer=normalizer)

    def solve(
        self,
        *,
        excluded_arguments: Any=None,
        excluded_graphs: tuple[Any, ...]=(),
        selection_observer: Any=None,
        time_limit_s: Any=None,
    ) -> Any:
        result = semantic_argument_optimization.optimize_argument_chart(
            self.options, n_inputs=self.n_inputs, contract=self.contract,
            definition_options=self.definition_options, definition_scores=self.definition_scores,
            prune_dominated=self.prune_dominated,
            excluded_arguments=excluded_arguments,
            excluded_graphs=excluded_graphs,
            time_limit_s=time_limit_s,
            selection_observer=selection_observer,
        )
        return (result[0] - self.choice_log_normalizer, *result[1:]) if result is not None else None

    def certify_selection(self, indices: Sequence[Sequence[int]]) -> Any:
        """Check a supplied integer graph without claiming it maximizes the score."""
        if len(indices) != len(self.options) or not self.options or self.n_inputs < 1:
            raise ValueError("selected graph dimensions differ from chart")
        if self.definition_options is not None and (len(self.definition_options) != len(self.options)
                or any(len(labels) != len(node) or any(len(names) != len(pool)
                    for names, pool in zip(labels, node))
                    for labels, node in zip(self.definition_options, self.options))):
            raise ValueError("selected graph definition dimensions differ from chart")
        if self.definition_scores is not None and self.definition_options is None:
            raise ValueError("selected graph definition labels absent")
        arguments, spans, contributions, definitions = [], [], [], {}
        for step, (node, selected) in enumerate(zip(self.options, indices, strict=True)):
            if not node or len(node) != len(selected):
                raise ValueError("selected graph role dimensions differ from chart")
            registers, mentions = [], []
            for slot, (pool, index) in enumerate(zip(node, selected, strict=True)):
                if type(index) is not int or not 0 <= index < len(pool):
                    raise ValueError("selected graph option identity absent")
                score, register, span = pool[index]
                if (not math.isfinite(score) or type(register) is not int
                        or not 0 <= register < self.n_inputs + len(self.options)
                        or register == self.n_inputs + step):
                    raise ValueError("selected graph has an invalid typed option")
                registers.append(register)
                mentions.append(span)
                contributions.append(score)
                if self.definition_options is not None:
                    definition = self.definition_options[step][slot][index]
                    if register in definitions and definitions[register] != definition:
                        raise ValueError("selected graph changes a register definition")
                    definitions[register] = definition
            if self.contract.distinct_arguments and len(set(registers)) != len(registers):
                raise ValueError("selected graph repeats a distinct operand")
            arguments.append(tuple(registers))
            spans.append(tuple(mentions))
        mentions = sorted((span for row in spans for span in row), key=lambda span: span.start)
        if any(left.end > right.start for left, right in zip(mentions, mentions[1:])):
            raise ValueError("selected graph has overlapping argument mentions")
        dependencies = tuple(tuple(sorted({register - self.n_inputs for register in row
            if register >= self.n_inputs})) for row in arguments)
        unresolved = set(range(len(self.options)))
        while unresolved:
            ready = {step for step in unresolved if not unresolved.intersection(dependencies[step])}
            if not ready:
                raise ValueError("selected graph has a dependency cycle")
            unresolved.difference_update(ready)
        used = {dependency for row in dependencies for dependency in row}
        sinks = set(range(len(self.options))) - used
        counts = Counter(register for row in arguments for register in row)
        if len(sinks) != 1 or not self.contract.accepts_complete(counts,
                n_inputs=self.n_inputs, operation_count=len(self.options), sink=next(iter(sinks))):
            raise ValueError("selected graph violates connected register-use bounds")
        if self.definition_scores is not None:
            keys = [(register, span) for register, span in definitions.items()]
            if any(key not in self.definition_scores or not math.isfinite(self.definition_scores[key])
                   for key in keys):
                raise ValueError("selected graph definition score absent or nonfinite")
            contributions.extend(self.definition_scores[key] for key in keys)
        return (math.fsum((*contributions, -self.choice_log_normalizer)), tuple(arguments),
                tuple(spans), dependencies)

    def solve_with_factors(
        self,
        *,
        excluded_arguments: Any=None,
        excluded_graphs: tuple[Any, ...]=(),
        time_limit_s: Any=None,
        relation_observer: Any=None,
    ) -> Any:
        """Return the exact selected factor sums, including latent definitions."""
        if self.option_factors is None:
            raise ValueError("argument chart did not retain score factors")
        selected = []
        result = self.solve(excluded_arguments=excluded_arguments, excluded_graphs=excluded_graphs,
                            selection_observer=selected.append, time_limit_s=time_limit_s)
        if result is None:
            return None
        rows = [self.option_factors[node][position][index]
                for node, positions in enumerate(selected[0]) for position, index in enumerate(positions)]
        if relation_observer is not None:
            if self.option_relation_evidence is None:
                raise ValueError("chart did not retain relation evidence")
            relation_observer(tuple(self.option_relation_evidence[node][position][index]
                for node, positions in enumerate(selected[0]) for position, index in enumerate(positions)))
        return result, tuple(math.fsum(row[column] for row in rows)
                             for column in range(len(rows[0])))

    def solve_grounded(self, context, roles, register_keys, **options):
        """Resolve contextual identities with the chart's existing graph constraints."""
        from core.learning.semantic_context_binding import solve_grounded_argument_chart

        return solve_grounded_argument_chart(self, context, roles, register_keys, **options)

    def restrict_arguments(self, targets: Sequence[Sequence[int]]) -> ScoredArgumentChart:
        """Keep every mention realizing a supplied training/diagnostic graph."""
        if len(targets) != len(self.options) or any(
            len(target) != len(node) for target, node in zip(targets, self.options, strict=True)
        ) or any(type(register) is not int or not 0 <= register < self.n_inputs + len(self.options)
                 for node in targets for register in node):
            raise ValueError("target arguments differ from chart")
        options, definitions, factors, evidence = [], [], [], []
        for node_index, (node, target) in enumerate(zip(self.options, targets, strict=True)):
            rows, labels, scores, evidence_rows = [], [], [], []
            for slot_index, (slot, register) in enumerate(zip(node, target, strict=True)):
                indices = [index for index, option in enumerate(slot) if option[1] == register]
                rows.append(tuple(slot[index] for index in indices))
                if self.option_factors is not None:
                    scores.append(tuple(self.option_factors[node_index][slot_index][index] for index in indices))
                if self.option_relation_evidence is not None:
                    evidence_rows.append(tuple(self.option_relation_evidence[node_index][slot_index][index]
                                               for index in indices))
                if self.definition_options is not None:
                    labels.append(tuple(self.definition_options[node_index][slot_index][index]
                                        for index in indices))
            options.append(tuple(rows))
            definitions.append(tuple(labels))
            factors.append(tuple(scores))
            evidence.append(tuple(evidence_rows))
        return ScoredArgumentChart(tuple(options), self.n_inputs, self.contract,
            tuple(definitions) if self.definition_options is not None else None,
            self.definition_scores, self.prune_dominated,
            tuple(factors) if self.option_factors is not None else None,
            tuple(evidence) if self.option_relation_evidence is not None else None,
            self.choice_log_normalizer)

    def diagnose_target(self, targets: Sequence[Sequence[int]]) -> dict:
        """Measure target reachability without altering the production choice.

This method consumes labeled arguments and is an offline diagnostic. A target
may be absent from the proposals, excluded by a joint constraint, or feasible
but outranked. These cases need different repairs.
"""

        if len(targets) != len(self.options) or any(
            len(target) != len(node) for target, node in zip(targets, self.options, strict=True)
        ):
            raise ValueError("target shape differs from the scored chart")
        if any(
            type(register) is not int or not 0 <= register < self.n_inputs + len(self.options)
            for node in targets for register in node
        ):
            raise ValueError("target register is invalid")
        restricted, labels, missing = [], [], []
        for node_index, (node, target) in enumerate(zip(self.options, targets, strict=True)):
            choices, definitions = [], []
            for slot_index, (slot, register) in enumerate(zip(node, target, strict=True)):
                indices = [index for index, option in enumerate(slot) if option[1] == register]
                if not indices:
                    missing.append((node_index, slot_index, register))
                choices.append(tuple(slot[index] for index in indices))
                if self.definition_options is not None:
                    definitions.append(tuple(
                        self.definition_options[node_index][slot_index][index] for index in indices
                    ))
            restricted.append(tuple(choices))
            labels.append(tuple(definitions))
        selected = self.solve()
        target = None if missing else ScoredArgumentChart(
            tuple(restricted), self.n_inputs, self.contract,
            tuple(labels) if self.definition_options is not None else None, self.definition_scores,
            prune_dominated=self.prune_dominated,
            choice_log_normalizer=self.choice_log_normalizer,
        ).solve()
        without_definition_consistency = None
        if not missing and target is None and self.definition_options is not None:
            without_definition_consistency = ScoredArgumentChart(
                tuple(restricted), self.n_inputs, self.contract,
                prune_dominated=self.prune_dominated,
            ).solve() is not None
        without_mention_exclusivity = None
        if not missing and target is None:
            relaxed, ordinal = [], 0
            for node in restricted:
                slots = []
                for slot in node:
                    slots.append(tuple((score, register, TokenSpan(ordinal, ordinal + 1))
                                       for score, register, _span in slot))
                    ordinal += 1
                relaxed.append(tuple(slots))
            without_mention_exclusivity = ScoredArgumentChart(
                tuple(relaxed), self.n_inputs, self.contract,
                tuple(labels) if self.definition_options is not None else None, self.definition_scores,
                prune_dominated=self.prune_dominated,
            ).solve() is not None
        if missing:
            cause = "target_register_not_proposed"
        elif target is None:
            cause = "target_excluded_by_joint_constraints"
        elif selected is None:
            raise AssertionError("a restricted target cannot be feasible in an infeasible chart")
        elif selected[1] == tuple(tuple(node) for node in targets):
            cause = "target_selected"
        else:
            cause = "target_feasible_but_not_selected"
        return {
            "diagnostic_only": True, "oracle_target_supplied": True,
            "serving_authority": False, "cause": cause,
            "missing_slots": missing, "target_feasible": target is not None,
            "selected_score": selected[0] if selected is not None else None,
            "target_score": target[0] if target is not None else None,
            "target_score_gap": selected[0] - target[0] if target is not None else None,
            "target_without_definition_consistency_feasible": without_definition_consistency,
            "target_without_mention_exclusivity_feasible": without_mention_exclusivity,
            "selected_arguments": selected[1] if selected is not None else None,
        }


@invariant("learning.argument_selection_certificate_checks_global_constraints", scope="learning",
           owner="core/learning/semantic_argument_chart.py", observational=False)
def _selection_certificate_invariant():
    from core.learning.semantic_program_transducer_fitting import RegisterUseContract

    chart = ScoredArgumentChart(((((2., 0, TokenSpan(0, 1)),), ((3., 1, TokenSpan(2, 3)),)),),
        2, RegisterUseContract(1, 1, 0, 1, True))
    assert chart.certify_selection(((0, 0),))[0] == 5.
    invalid = replace(chart, options=((((2., 0, TokenSpan(0, 2)),), ((3., 1, TokenSpan(1, 3)),)),))
    try:
        invalid.certify_selection(((0, 0),))
    except ValueError:
        return ()
    raise AssertionError("overlapping mentions passed the graph certificate")
