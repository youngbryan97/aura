"""Immutable scored charts and offline attribution of semantic graph failures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from core.learning import semantic_argument_optimization
from core.learning.semantic_program_ir import TokenSpan

if TYPE_CHECKING:
    from core.learning.semantic_program_transducer_fitting import RegisterUseContract


@dataclass(frozen=True)
class ScoredArgumentChart:
    options: tuple
    n_inputs: int
    contract: RegisterUseContract
    definition_options: tuple | None = None
    definition_scores: Mapping[tuple[int, TokenSpan], float] | None = None

    def __post_init__(self) -> None:
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

    def solve(self):
        return semantic_argument_optimization.optimize_argument_chart(
            self.options, n_inputs=self.n_inputs, contract=self.contract,
            definition_options=self.definition_options, definition_scores=self.definition_scores,
        )

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
        ).solve()
        without_definition_consistency = None
        if not missing and target is None and self.definition_options is not None:
            without_definition_consistency = ScoredArgumentChart(
                tuple(restricted), self.n_inputs, self.contract,
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
