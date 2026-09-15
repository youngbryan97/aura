"""Source-supervised graph contrasts require a distinguishing computation.

Search keeps alternate correct computations and unknown equivalence separate
from demonstrated errors. This module is for training and diagnostics; source
targets and counterfactual probes are never runtime answer inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import random

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_argument_chart import ScoredArgumentChart
from core.learning.semantic_argument_optimization import ArgumentOptimizationIncompleteError
from core.learning.semantic_program_campaign import _sha
from core.learning.semantic_program_floor import (
    compile_source_independent_program_to_floor,
    execute_semantic_floor_program,
    semantic_program_polynomial_key,
    semantic_programs_structurally_equivalent,
)
from core.learning.semantic_program_transducer_fitting import _operation_order
from core.verify.invariants import invariant


def counterfactual_inputs(inputs, *, count=32, seed=0):
    """Vary typed public values without reading operations, names or answers."""
    if type(count) is not int or count < 0:
        raise ValueError("counterfactual count must be nonnegative")
    if any(type(value) is not int and not (
        isinstance(value, tuple) and all(type(item) is int for item in value)
    ) for value in inputs):
        raise ValueError("counterfactual inputs require integer or integer-sequence values")
    rng = random.Random(seed)
    integer_values = (-3, -1, 0, 1, 2, 5, 11)
    sequence_values = ((), (0,), (1, -1), (2, 2, 3), (5, 1, -3, 2))
    probes = [tuple(inputs)]
    for _ in range(count):
        probes.append(tuple(rng.choice(sequence_values if isinstance(value, tuple) else integer_values)
                            for value in inputs))
    return tuple(dict.fromkeys(probes))


def argument_graph_program(nodes, arguments, *, n_inputs: int) -> Program:
    """Normalize a selected DAG using the decoder's existing topological rule."""
    if len(nodes) != len(arguments) or any(
        type(register) is not int or not 0 <= register < n_inputs + len(nodes)
        for row in arguments for register in row
    ):
        raise ValueError("argument graph dimensions differ")
    dependencies = tuple(tuple(sorted({r - n_inputs for r in row if r >= n_inputs})) for row in arguments)
    order = _operation_order(dependencies, nodes, require_connected=True)
    if order is None:
        raise ValueError("argument graph has no connected topological schedule")
    remap = {n_inputs + old: n_inputs + new for new, old in enumerate(order)}
    return Program(n_inputs, tuple(Instruction(nodes[index].operation,
        tuple(r if r < n_inputs else remap[r] for r in arguments[index])) for index in order))


def compare_program_meanings(target: Program, alternative: Program, probes: Sequence[tuple], *, fuel=100_000) -> dict:
    """Prove a supported symmetry or find a replayable different-output witness."""
    if target.n_inputs != alternative.n_inputs:
        raise ValueError("semantic contrast public input geometry differs")
    if semantic_programs_structurally_equivalent(target, alternative):
        return {"status": "equivalent", "method": "floor_structural_symmetry_v1"}
    key = semantic_program_polynomial_key(target)
    if key is not None and key == semantic_program_polynomial_key(alternative):
        return {"status": "equivalent", "method": "floor_integer_polynomial_v1",
                "normal_form_sha256": _sha(key)}
    observations = []
    for values in probes:
        outputs, receipts, failures = [], [], []
        for program in (target, alternative):
            try:
                compiled = compile_source_independent_program_to_floor(program, tuple(values),
                    provenance_receipt_sha256=_sha({"program": program.sha(), "purpose": "training_contrast"}))
                result = execute_semantic_floor_program(compiled, fuel=fuel)
            except (ValueError, TypeError, RuntimeError, ArithmeticError) as exc:
                outputs.append(None)
                receipts.append(None)
                failures.append(f"{type(exc).__name__}:{exc}")
            else:
                outputs.append(result.result)
                receipts.append(result.receipt)
                failures.append(None)
        observation = {"inputs": list(values), "outputs": outputs, "execution_receipts": receipts,
                       "failures": failures}
        if not any(failures) and outputs[0] != outputs[1]:
            return {"status": "different", "method": "universal_floor_counterexample_v1",
                    "witness": observation, "witness_sha256": _sha(observation)}
        observations.append(observation)
    return {"status": "unknown", "method": "finite_probes_without_distinction",
            "probes_checked": len(observations), "observations_sha256": _sha(observations),
            "failed_probes": sum(any(row["failures"]) for row in observations)}


@dataclass(frozen=True)
class GraphCounterexampleSearch:
    positive: tuple | None
    negative: tuple | None
    receipt: dict


def find_graph_counterexample(chart: ScoredArgumentChart, nodes, target_arguments, *, probes=(), max_graphs=128,
                              progress=None, solve_time_limit_s=None):
    """Find the highest-scoring witnessed error; unresolved meanings prevent that claim."""
    if type(max_graphs) is not int or max_graphs < 1:
        raise ValueError("counterexample graph allowance must be positive")
    target_arguments = tuple(tuple(row) for row in target_arguments)
    receipt = {"schema": "aura.semantic_graph_counterexample.v1", "serving_authority": False,
               "source_target_available": True, "max_graphs": max_graphs, "examined": [],
               "solve_time_limit_s": solve_time_limit_s,
               "search_complete": False, "highest_incorrect_proven": False}
    if progress:
        progress({"stage": "positive_graph"})
    try:
        positive = chart.restrict_arguments(target_arguments).solve_with_factors(time_limit_s=solve_time_limit_s)
    except ArgumentOptimizationIncompleteError as exc:
        return GraphCounterexampleSearch(None, None, {**receipt, "status": "positive_search_incomplete", "reason": str(exc)})
    if positive is None:
        return GraphCounterexampleSearch(None, None, {**receipt, "status": "target_unreachable"})
    target = argument_graph_program(nodes, target_arguments, n_inputs=chart.n_inputs)
    excluded = [target_arguments]
    unresolved = False
    for _ in range(max_graphs):
        if progress:
            progress({"stage": "alternative_graph", "excluded_graphs": len(excluded)})
        try:
            candidate = chart.solve_with_factors(excluded_graphs=excluded, time_limit_s=solve_time_limit_s)
        except ArgumentOptimizationIncompleteError as exc:
            return GraphCounterexampleSearch(positive, None, {**receipt, "status": "alternative_search_incomplete", "reason": str(exc)})
        if candidate is None:
            receipt.update(status="equivalence_unresolved" if unresolved else "no_incorrect_graph",
                           search_complete=True)
            return GraphCounterexampleSearch(positive, None, receipt)
        arguments = candidate[0][1]
        program = argument_graph_program(nodes, arguments, n_inputs=chart.n_inputs)
        comparison = compare_program_meanings(target, program, probes)
        receipt["examined"].append({"arguments": arguments, "score": candidate[0][0],
                                     "comparison": comparison})
        if comparison["status"] == "different":
            receipt.update(status="counterexample", highest_incorrect_proven=not unresolved)
            return GraphCounterexampleSearch(positive, candidate, receipt)
        if comparison["status"] == "unknown":
            unresolved = True
        elif candidate[0][0] > positive[0][0]:
            positive = candidate
        excluded.append(arguments)
    receipt["status"] = "search_incomplete"
    return GraphCounterexampleSearch(positive, None, receipt)


@invariant("learning.semantic_counterexamples_do_not_train_against_commutativity", scope="learning",
           owner="core/learning/semantic_graph_counterexamples.py", observational=False)
def _semantic_counterexample_symmetry() -> tuple:
    target = Program(2, (Instruction("add", (0, 1)),))
    same = Program(2, (Instruction("add", (1, 0)),))
    different = Program(2, (Instruction("sub", (1, 0)),))
    assert compare_program_meanings(target, same, [(3, 7)])["status"] == "equivalent"
    assert compare_program_meanings(target, different, [(3, 7)])["status"] == "different"
    return ()
