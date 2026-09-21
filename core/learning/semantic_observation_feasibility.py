"""Detect incompatible semantic supervision for identical decoder observations."""

from collections import defaultdict
from itertools import combinations

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_candidate_bank import candidate_observation_identity
from core.learning.semantic_graph_counterexamples import (
    compare_program_meanings,
    counterfactual_inputs,
)
from core.learning.semantic_joint_graph_learning import align_source_input_registers
from core.learning.semantic_program_campaign import _sha
from core.learning.semantic_program_transducer import _hidden_array


def audit_observation_feasibility(examples):
    """Audit development labels; no labels enter prediction or parameter fitting.

    Source hashes and family names are not decoder information. Model basis,
    tokens, public inputs and normalized hidden states are. A witnessed
    difference refutes deterministic recovery of both intended programs from
    that same information, not correctness of every possible public answer.
    """
    groups = defaultdict(list)
    count = 0
    for item in examples:
        if item.split not in {"train", "validation"}:
            raise ValueError("feasibility audit excludes sealed test observations")
        observation = candidate_observation_identity(item.ir.source_token_ids,
            item.public_inputs, _hidden_array(item.hidden_states))
        groups[(item.ir.model_basis_receipt_sha256, observation)].append(item)
        count += 1
    comparisons = []
    for (basis, observation), items in groups.items():
        for left, right in combinations(items, 2):
            try:
                aligned, _ = align_source_input_registers(right, left.ir.input_spans)
            except ValueError as exc:
                comparison = {"status": "unknown", "reason": str(exc)}
            else:
                target = Program(len(right.public_inputs), tuple(
                    Instruction(ins.op, ins.args) for ins in aligned))
                comparison = compare_program_meanings(left.ir.to_program(), target,
                    counterfactual_inputs(left.public_inputs))
            comparisons.append({"observation": observation, "basis": basis,
                "sources": [left.ir.source_text_sha256, right.ir.source_text_sha256],
                "comparison": comparison})
    body = {"schema": "aura.semantic_observation_feasibility.v1",
        "observations": count, "distinct_observations": len(groups),
        "duplicate_groups": sum(len(items) > 1 for items in groups.values()),
        "comparisons": comparisons,
        "contradictions": sum(row["comparison"]["status"] == "different" for row in comparisons),
        "unresolved": sum(row["comparison"]["status"] == "unknown" for row in comparisons),
        "scope": "identical_observation_intended_program_consistency",
        "learnability_proven": False, "test_examples_used": 0, "serving_authority": False}
    return {**body, "receipt_sha256": _sha(body)}
