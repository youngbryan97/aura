"""Measure learned program reuse through the common procedure registry."""

from core.cognition.procedure import ProcedureRegistry
from core.learning.semantic_graph_counterexamples import compare_program_meanings, counterfactual_inputs
from core.learning.semantic_procedure_currency import from_semantic_program, execute_semantic_procedure
from core.learning.semantic_program_campaign import _sha


def evaluate_learned_procedure_reuse(model, examples, *, split, probe_count=32, seed=0, progress=None):
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
                observation = {"inputs": values, "lowering_correct": False,
                               "task_correct": False, "error": None}
                try:
                    state = {f"semantic:argument:{index}": value for index, value in enumerate(values)}
                    execution = execute_semantic_procedure(procedure, state)
                    observation.update(
                        result=execution.result,
                        lowering_correct=execution.result == predicted.run(values),
                        task_correct=execution.result == target.run(values),
                        execution_receipt=execution.receipt,
                    )
                except (ValueError, TypeError, RuntimeError, ArithmeticError) as exc:
                    observation["error"] = f"{type(exc).__name__}:{exc}"
                row["probes"].append(observation)
        rows.append(row)
        if progress is not None:
            progress({"stage": "learned_procedure_reuse", "completed": len(rows), "total": len(selected)})
    body = {
        "schema": "aura.learned_procedure_reuse.v1", "split": split,
        "transducer_receipt_sha256": model.receipt_sha256,
        "source_ids_sha256": _sha(ids), "seed": seed, "probe_count": probe_count,
        "total": len(rows), "accepted": sum(row["accepted"] for row in rows),
        "proved_equivalent": sum((row["equivalence"] or {}).get("status") == "equivalent" for row in rows),
        "unique_procedures": len({row["procedure_id"] for row in rows if row["accepted"]}),
        "fresh_probes": sum(len(row["probes"]) for row in rows),
        "lowering_failures": sum(not probe["lowering_correct"] for row in rows for probe in row["probes"]),
        "task_failures": sum(not probe["task_correct"] for row in rows for probe in row["probes"]),
        "rows": rows, "serving_authority": False, "registry_scope": "evaluation_local",
        "target_available_to_decoder": False, "target_available_to_registry": False,
        "finite_probes_prove_equivalence": False,
    }
    return {**body, "receipt_sha256": _sha(body)}
