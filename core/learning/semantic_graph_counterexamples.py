"""Source-supervised graph contrasts require a distinguishing computation.

Search keeps alternate correct computations and unknown equivalence separate
from demonstrated errors. This module is for training and diagnostics; source
targets and counterfactual probes are never runtime answer inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import random

from core.cognition.the_floor_she_stands_on import Stuck
from core.learning.procedure_induction import Instruction, Program, _UNDEFINED
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


def _observe_program_domain(program, values, *, fuel):
    """Require reference/floor agreement before admitting a value or domain fact."""
    reference = program.run(values)
    observation = {"status": "error", "result": None, "execution_receipt": None,
                   "compiled_receipt": None, "failure": None,
                   "reference_defined": reference is not _UNDEFINED}
    try:
        compiled = compile_source_independent_program_to_floor(program, tuple(values),
            provenance_receipt_sha256=_sha({"program": program.sha(), "purpose": "training_contrast"}))
        observation["compiled_receipt"] = compiled.receipt
        result = execute_semantic_floor_program(compiled, fuel=fuel)
    except Stuck as exc:
        if reference is _UNDEFINED and observation["compiled_receipt"] is not None:
            observation["status"] = "undefined"
        observation["failure"] = f"{type(exc).__name__}:{exc}"
    except (ValueError, TypeError, RuntimeError, ArithmeticError) as exc:
        observation["failure"] = f"{type(exc).__name__}:{exc}"
    else:
        observation.update(result=result.result, execution_receipt=result.receipt)
        if reference is not _UNDEFINED and result.result == reference:
            observation["status"] = "value"
        else:
            observation["failure"] = "reference_floor_disagreement"
    return observation


def compare_program_meanings(target: Program, alternative: Program, probes: Sequence[tuple], *, fuel=100_000) -> dict:
    """Prove a supported symmetry or witness different values or defined domains."""
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
        outcomes = [_observe_program_domain(program, values, fuel=fuel) for program in (target, alternative)]
        statuses = [row["status"] for row in outcomes]
        outputs = [row["result"] for row in outcomes]
        observation = {"inputs": list(values), "outputs": outputs,
                       "execution_receipts": [row["execution_receipt"] for row in outcomes],
                       "failures": [row["failure"] for row in outcomes], "outcomes": outcomes}
        if "error" not in statuses and (statuses[0] != statuses[1] or (
                statuses[0] == "value" and outputs[0] != outputs[1])):
            return {"status": "different", "method": "universal_floor_counterexample_v2",
                    "distinction": "domain" if statuses[0] != statuses[1] else "value",
                    "witness": observation, "witness_sha256": _sha(observation)}
        observations.append(observation)
    return {"status": "unknown", "method": "finite_probes_without_distinction",
            "probes_checked": len(observations), "observations_sha256": _sha(observations),
            "failed_probes": sum(any(item["status"] == "error" for item in row["outcomes"])
                                 for row in observations),
            "jointly_undefined_probes": sum(all(item["status"] == "undefined" for item in row["outcomes"])
                                            for row in observations)}


@dataclass(frozen=True)
class GraphCounterexampleSearch:
    positive: tuple | None
    negative: tuple | None
    receipt: dict
    positive_evidence: tuple = ()
    negative_evidence: tuple = ()


def uniform_reduction_equivalence(chart, nodes):
    """Prove all feasible linear-use monoid trees have one output meaning.

    The argument solver enforces acyclicity and a single sink. With each
    public input and nonsink intermediate used once, every feasible graph is
    a tree containing every input exactly once. Integer addition and
    multiplication are associative and commutative on this total domain.
    """
    contract = chart.contract
    if (not nodes or len(nodes) != len(chart.options) or chart.n_inputs != len(nodes) + 1
            or any(len(row) != 2 for row in chart.options)
            or len({node.operation for node in nodes}) != 1
            or nodes[0].operation not in {"add", "mul"}
            or (contract.input_min_uses, contract.input_max_uses,
                contract.intermediate_min_uses, contract.intermediate_max_uses) != (1, 1, 1, 1)):
        return None
    return {"method": "linear_use_integer_monoid_v1", "operation": nodes[0].operation,
            "input_count": chart.n_inputs, "operation_count": len(nodes),
            "acyclic_single_sink": True, "each_input_and_nonsink_used_once": True,
            "claim": "output_equality_only"}


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
    positive_evidence = []
    try:
        positive = chart.restrict_arguments(target_arguments).solve_with_factors(time_limit_s=solve_time_limit_s,
            relation_observer=positive_evidence.append if chart.option_relation_evidence is not None else None)
    except ArgumentOptimizationIncompleteError as exc:
        return GraphCounterexampleSearch(None, None, {**receipt, "status": "positive_search_incomplete", "reason": str(exc)})
    if positive is None:
        return GraphCounterexampleSearch(None, None, {**receipt, "status": "target_unreachable"})
    target = argument_graph_program(nodes, target_arguments, n_inputs=chart.n_inputs)
    class_proof = uniform_reduction_equivalence(chart, nodes)
    if class_proof is not None:
        receipt.update(status="no_incorrect_graph", search_complete=True, equivalence_class_proof=class_proof)
        return GraphCounterexampleSearch(positive, None, receipt,
                                         positive_evidence[0] if positive_evidence else ())
    return find_program_counterexample(chart, nodes, target, probes=probes, max_graphs=max_graphs,
        progress=progress, solve_time_limit_s=solve_time_limit_s, positive=positive,
        positive_evidence=positive_evidence[0] if positive_evidence else (), excluded_graphs=(target_arguments,))


def find_program_counterexample(chart: ScoredArgumentChart, nodes, target: Program, *, probes=(), max_graphs=128,
                                progress=None, solve_time_limit_s=None, positive=None,
                                positive_evidence=(), excluded_graphs=()):
    """Search a runtime chart even when its operations differ from the source target."""
    if type(max_graphs) is not int or max_graphs < 1:
        raise ValueError("counterexample graph allowance must be positive")
    if target.n_inputs != chart.n_inputs:
        raise ValueError("semantic contrast public input geometry differs")
    receipt = {"schema": "aura.semantic_graph_counterexample.v1", "serving_authority": False,
               "source_target_available": True, "max_graphs": max_graphs, "examined": [],
               "solve_time_limit_s": solve_time_limit_s,
               "search_complete": False, "highest_incorrect_proven": False}
    excluded = list(excluded_graphs)
    unresolved = False
    best_evidence = positive_evidence
    for _ in range(max_graphs):
        if progress:
            progress({"stage": "alternative_graph", "excluded_graphs": len(excluded)})
        try:
            candidate_evidence = []
            candidate = chart.solve_with_factors(excluded_graphs=excluded, time_limit_s=solve_time_limit_s,
                relation_observer=candidate_evidence.append if chart.option_relation_evidence is not None else None)
        except ArgumentOptimizationIncompleteError as exc:
            return GraphCounterexampleSearch(positive, None, {**receipt, "status": "alternative_search_incomplete", "reason": str(exc)}, best_evidence)
        if candidate is None:
            receipt.update(status="equivalence_unresolved" if unresolved else "no_incorrect_graph",
                           search_complete=True)
            return GraphCounterexampleSearch(positive, None, receipt, best_evidence)
        arguments = candidate[0][1]
        program = argument_graph_program(nodes, arguments, n_inputs=chart.n_inputs)
        comparison = compare_program_meanings(target, program, probes)
        receipt["examined"].append({"arguments": arguments, "score": candidate[0][0],
                                     "comparison": comparison})
        if comparison["status"] == "different":
            receipt.update(status="counterexample", highest_incorrect_proven=not unresolved)
            return GraphCounterexampleSearch(positive, candidate, receipt, best_evidence,
                                            candidate_evidence[0] if candidate_evidence else ())
        if comparison["status"] == "unknown":
            unresolved = True
        elif positive is None or candidate[0][0] > positive[0][0]:
            positive = candidate
            best_evidence = candidate_evidence[0] if candidate_evidence else ()
        class_proof = uniform_reduction_equivalence(chart, nodes)
        if comparison["status"] == "equivalent" and class_proof is not None:
            receipt.update(status="no_incorrect_graph", search_complete=True,
                           equivalence_class_proof=class_proof)
            return GraphCounterexampleSearch(positive, None, receipt, best_evidence)
        excluded.append(arguments)
    receipt["status"] = "search_incomplete"
    return GraphCounterexampleSearch(positive, None, receipt, best_evidence)


@invariant("learning.semantic_counterexamples_do_not_train_against_commutativity", scope="learning",
           owner="core/learning/semantic_graph_counterexamples.py", observational=False)
def _semantic_counterexample_symmetry() -> tuple:
    target = Program(2, (Instruction("add", (0, 1)),))
    same = Program(2, (Instruction("add", (1, 0)),))
    different = Program(2, (Instruction("sub", (1, 0)),))
    assert compare_program_meanings(target, same, [(3, 7)])["status"] == "equivalent"
    assert compare_program_meanings(target, different, [(3, 7)])["status"] == "different"
    return ()
