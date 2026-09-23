"""Attribute a fixed candidate bank against separately supplied source evidence."""

from typing import Any

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_candidate_bank import candidate_observation_identity
from core.learning.semantic_graph_counterexamples import (
    ProgramObservationCache,
    compare_program_meanings,
    counterfactual_inputs,
)
from core.learning.semantic_joint_graph_learning import align_source_input_registers
from core.learning.semantic_program_campaign import _sha
from core.learning.semantic_program_transducer import _hidden_array


def diagnose_semantic_candidate_bank(
    bank: Any,
    item: Any,
    *,
    execution_correct: Any=None,
    emission_correct: Any=None,
) -> Any:
    """Compare after generation; labels never change the search or its selection.

    Exhausted search with unknown equivalence cannot establish unreachability.
    Neither semantic equivalence nor missing downstream observations establish
    successful execution and public emission.
    """
    if any(value is not None and type(value) is not bool
           for value in (execution_correct, emission_correct)):
        raise ValueError("downstream observations must be booleans or unmeasured")
    bank.validate()
    if (bank.receipt["source_text_sha256"] != item.ir.source_text_sha256
            or bank.receipt["model_basis_sha256"] != item.ir.model_basis_receipt_sha256):
        raise ValueError("candidate bank and diagnostic source identity differ")
    body = {"schema": "aura.semantic_candidate_diagnosis.v2",
            "bank_receipt_sha256": bank.receipt["receipt_sha256"], "split": item.split,
            "source_text_sha256": item.ir.source_text_sha256,
            "diagnostic_only": True, "source_target_available": True, "serving_authority": False,
            "execution_correct": execution_correct, "emission_correct": emission_correct,
            "search_complete": bank.receipt["search_complete"], "comparisons": [],
            "correct_reachable": None, "selected_semantic_status": "unmeasured"}

    def finish(stage: str) -> dict[str, Any]:
        body["failure_stage"] = stage
        return {**body, "receipt_sha256": _sha(body)}

    if not bank.candidates:
        return finish("decode_unavailable")
    observed = candidate_observation_identity(item.ir.source_token_ids, item.public_inputs,
                                               _hidden_array(item.hidden_states))
    if bank.receipt.get("observation_sha256") != observed:
        raise ValueError("candidate bank observation differs from diagnostic input")
    try:
        instructions, _ = align_source_input_registers(item, bank.input_spans)
    except ValueError as exc:
        body["failure"] = str(exc)
        return finish("grounding")
    target = Program(len(item.public_inputs), tuple(Instruction(ins.op, ins.args) for ins in instructions))
    cache = ProgramObservationCache()
    comparisons = {}
    probes = counterfactual_inputs(item.public_inputs)
    for candidate in bank.candidates:
        key = candidate.program.sha()
        if key not in comparisons:
            comparisons[key] = compare_program_meanings(target, candidate.program, probes, observation_cache=cache)
            body["comparisons"].append({"program_sha256": key, **comparisons[key]})
    selected = bank.selected.ir.to_program().sha() if bank.selected.ir is not None else None
    if selected is not None and selected not in comparisons:
        raise ValueError("candidate bank omitted its selected program")
    status = comparisons[selected]["status"] if selected is not None else "unavailable"
    reachable = any(row["status"] == "equivalent" for row in comparisons.values())
    unknown = any(row["status"] == "unknown" for row in comparisons.values())
    source_ops = tuple(ins.op for ins in target.instructions)
    source_spans = tuple(ins.operation_span for ins in item.ir.instructions)

    def source_view(candidate: Any) -> bool:
        return (tuple(ins.op for ins in candidate.program.instructions) == source_ops
                and len(candidate.operation_spans) == len(source_spans)
                and all(source.start < observed.end and observed.start < source.end
                        for source, observed in zip(source_spans, candidate.operation_spans,
                                                    strict=True)))

    equivalent = [candidate for candidate in bank.candidates
                  if comparisons[candidate.program.sha()]["status"] == "equivalent"]
    selected_view = bank.candidates[0] if selected is not None else None
    body["operation_view"] = {
        "source_spans": [span.to_dict() for span in source_spans],
        "selected_spans": None if selected_view is None else
            [span.to_dict() for span in selected_view.operation_spans],
        "selected_source_view_aligned": None if selected_view is None else source_view(selected_view),
        "equivalent_candidates": len(equivalent),
        "equivalent_source_view_aligned": sum(source_view(candidate) for candidate in equivalent),
        "equivalent_exact_source_spans": sum(
            source_view(candidate) and candidate.operation_spans == source_spans
            for candidate in equivalent),
        "diagnostic_only": True,
    }
    body.update(selected_semantic_status=status, correct_reachable=reachable or
        (None if unknown or not body["search_complete"] else False),
        target_program_sha256=target.sha(), floor_observation_reuse=cache.statistics())
    if not reachable:
        return finish("verification_unknown" if unknown else
                      "reachability" if body["search_complete"] else "incomplete_search")
    if status != "equivalent":
        return finish("selection" if status == "different" else "verification_unknown")
    if execution_correct is not True:
        return finish("execution" if execution_correct is False else "execution_unmeasured")
    if emission_correct is not True:
        return finish("emission" if emission_correct is False else "emission_unmeasured")
    return finish("success")
