"""Measure learned program reuse through the common procedure registry."""

from core.cognition.procedure import ProcedureRegistry
from core.cognition.the_floor_she_stands_on import Stuck
from core.learning.procedure_induction import _UNDEFINED
from core.learning.semantic_graph_counterexamples import compare_program_meanings, counterfactual_inputs
from core.learning.semantic_procedure_currency import from_semantic_program, execute_semantic_procedure
from core.learning.semantic_program_campaign import _sha
from typing import Any


def evaluate_learned_procedure_reuse(
    model: Any,
    examples: Any,
    *,
    split: Any,
    probe_count: int=32,
    seed: int=0,
    progress: Any=None,
) -> dict[str, Any]:
    """Keep task correctness separate from faithful lowering and new-value execution.

    The decoder and registry receive no target program or expected answer.
    Targets belong only to this offline evaluator. Finite successful probes
    never establish equivalence for an unsupported program class.
    """
    selected = tuple(item for item in examples if item.split == split)
    ids = tuple(item.ir.source_text_sha256 for item in selected)
    if not selected or len(set(ids)) != len(ids):
        raise ValueError("reuse evaluation needs a nonempty unique split")
    if type(probe_count) is not int or probe_count < 1:
        raise ValueError("reuse evaluation needs positive fresh-value probes")
    registry, rows = ProcedureRegistry(), []
    for item in selected:
        outcome = model.decode(
            source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
            public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
            model_basis_sha256=item.ir.model_basis_receipt_sha256,
        )
        row = {"source_sha256": item.ir.source_text_sha256, "accepted": outcome.ir is not None,
               "refusal": outcome.refusal, "probes": [], "equivalence": None}
        if outcome.ir is not None:
            predicted, target = outcome.ir.to_program(), item.ir.to_program()
            probes = tuple(values for values in counterfactual_inputs(
                item.public_inputs, count=probe_count, seed=seed,
            ) if values != tuple(item.public_inputs))
            row["equivalence"] = compare_program_meanings(target, predicted, probes)
            procedure = from_semantic_program(outcome.ir, registry=registry)
            row["procedure_id"] = procedure.procedure_id
            row["program_sha256"] = procedure.program.program_sha256
            row["procedure_receipt"] = procedure.program.receipt()
            for values in probes:
                predicted_value, target_value = predicted.run(values), target.run(values)
                reference_defined, target_defined = predicted_value is not _UNDEFINED, target_value is not _UNDEFINED
                observation = {"inputs": values, "lowering_correct": False,
                    "task_correct": False, "answer_correct": False, "error": None,
                    "reference_defined": reference_defined, "target_defined": target_defined,
                    "execution_status": "error"}
                try:
                    state = {f"semantic:argument:{index}": value for index, value in enumerate(values)}
                    execution = execute_semantic_procedure(procedure, state)
                    observation.update(
                        result=execution.result,
                        lowering_correct=reference_defined and execution.result == predicted_value,
                        task_correct=target_defined and execution.result == target_value,
                        answer_correct=target_defined and execution.result == target_value,
                        execution_status="value",
                        execution_receipt=execution.receipt,
                    )
                except Stuck as exc:
                    observation.update(execution_status="undefined",
                        lowering_correct=not reference_defined, task_correct=not target_defined,
                        error=f"{type(exc).__name__}:{exc}")
                except (ValueError, TypeError, RuntimeError, ArithmeticError) as exc:
                    observation["error"] = f"{type(exc).__name__}:{exc}"
                row["probes"].append(observation)
        rows.append(row)
        if progress is not None:
            progress({"stage": "learned_procedure_reuse", "completed": len(rows), "total": len(selected)})
    body = {
        "schema": "aura.learned_procedure_reuse.v2", "split": split,
        "transducer_receipt_sha256": model.receipt_sha256,
        "source_ids_sha256": _sha(ids), "seed": seed, "probe_count": probe_count,
        "total": len(rows), "accepted": sum(row["accepted"] for row in rows),
        "proved_equivalent": sum((row["equivalence"] or {}).get("status") == "equivalent" for row in rows),
        "unique_procedures": len({row["procedure_id"] for row in rows if row["accepted"]}),
        "fresh_probes": sum(len(row["probes"]) for row in rows),
        "lowering_failures": sum(not probe["lowering_correct"] for row in rows for probe in row["probes"]),
        "task_failures": sum(not probe["task_correct"] for row in rows for probe in row["probes"]),
        "defined_task_probes": sum(probe["target_defined"] for row in rows for probe in row["probes"]),
        "correct_task_answers": sum(probe["answer_correct"] for row in rows for probe in row["probes"]),
        "matched_domain_rejections": sum(probe["execution_status"] == "undefined" and not probe["target_defined"]
            for row in rows for probe in row["probes"]),
        "execution_errors": sum(probe["execution_status"] == "error" for row in rows for probe in row["probes"]),
        "rows": rows, "serving_authority": False, "registry_scope": "evaluation_local",
        "target_available_to_decoder": False, "target_available_to_registry": False,
        "finite_probes_prove_equivalence": False,
    }
    return {**body, "receipt_sha256": _sha(body)}
