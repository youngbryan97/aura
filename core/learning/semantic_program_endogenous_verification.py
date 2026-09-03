"""Evaluate an endogenous semantic bridge on immutable resident observations."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_compositional_transducer import (
    CompositionalSemanticProgramTransducer,
)
from core.learning.semantic_program_corpus import SemanticProgramExample
from core.learning.semantic_program_feature_materialization import (
    LoadedSemanticFeatureBundle,
    tokenize_with_offsets,
)
from core.learning.semantic_program_runtime import (
    SemanticProgramDecodeRejectedError,
    execute_compositional_semantic_observation,
)
from core.learning.semantic_public_inputs import semantic_public_token_inputs

ENDOGENOUS_SEMANTIC_EVALUATION_SCHEMA: Final = (
    "aura.semantic_program_endogenous_evaluation.v1"
)
ENDOGENOUS_SEMANTIC_VERIFICATION_SCHEMA: Final = (
    "aura.semantic_program_endogenous_verification.v1"
)
_SPLITS: Final = frozenset({"validation", "test"})
ENDOGENOUS_SEMANTIC_VERIFICATION_SOURCES: Final = (
    "core/learning/semantic_input_grounding.py",
    "core/learning/semantic_program_compositional_transducer.py",
    "core/learning/semantic_program_endogenous_verification.py",
    "core/learning/semantic_program_feature_materialization.py",
    "core/learning/semantic_program_floor.py",
    "core/learning/semantic_program_ir.py",
    "core/learning/semantic_program_runtime.py",
    "core/learning/semantic_public_inputs.py",
    "tools/verify_endogenous_compositional_semantic_runtime.py",
)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _canonical_expected_program(example: SemanticProgramExample) -> Program:
    """Re-index public registers by source order, leaving computed SSA ids stable."""

    input_count = len(example.inputs)
    source_order = sorted(
        range(input_count),
        key=lambda index: (
            example.input_spans[index].start,
            example.input_spans[index].end,
        ),
    )
    old_to_new = {old: new for new, old in enumerate(source_order)}
    return Program(
        n_inputs=input_count,
        instructions=tuple(
            Instruction(
                annotated.instruction.op,
                tuple(
                    old_to_new[argument] if argument < input_count else argument
                    for argument in annotated.instruction.args
                ),
            )
            for annotated in example.instructions
        ),
    )


def _source_order_annotations(
    example: SemanticProgramExample,
) -> tuple[tuple[Any, int, int], ...]:
    return tuple(
        sorted(
            (
                (value, span.start, span.end)
                for value, span in zip(
                    example.inputs,
                    example.input_spans,
                    strict=True,
                )
            ),
            key=lambda item: (item[1], item[2]),
        )
    )


@dataclass(frozen=True, slots=True)
class EndogenousSemanticEvaluation:
    arm: str
    total: int
    accepted: int
    public_input_exact: int
    program_exact: int
    answer_exact: int
    rows: tuple[dict[str, Any], ...]
    receipt: dict[str, Any]

    def __post_init__(self) -> None:
        body = {key: value for key, value in self.receipt.items() if key != "receipt_sha256"}
        if (
            self.receipt.get("schema") != ENDOGENOUS_SEMANTIC_EVALUATION_SCHEMA
            or self.receipt.get("receipt_sha256") != _sha(body)
            or self.receipt.get("total") != self.total
            or self.receipt.get("accepted") != self.accepted
            or self.receipt.get("public_input_exact") != self.public_input_exact
            or self.receipt.get("program_exact") != self.program_exact
            or self.receipt.get("answer_exact") != self.answer_exact
        ):
            raise ValueError("endogenous semantic evaluation receipt is invalid")


def evaluate_endogenous_semantic_bridge(
    *,
    model: CompositionalSemanticProgramTransducer,
    bundle: LoadedSemanticFeatureBundle,
    corpus: Sequence[SemanticProgramExample],
    tokenizer: Any,
    expected_representation_basis_sha256: str,
    arm: str,
) -> EndogenousSemanticEvaluation:
    """Decode held-out observations after recovering values solely from their text."""

    examples = {item.example_id: item for item in corpus}
    held = tuple(
        record for record in bundle.examples if record.metadata.get("split") in _SPLITS
    )
    if not held or any(record.metadata.get("example_id") not in examples for record in held):
        raise ValueError("endogenous semantic held-out corpus differs from its bundle")

    rows: list[dict[str, Any]] = []
    counts = {
        "accepted": 0,
        "public_input_exact": 0,
        "program_exact": 0,
        "answer_exact": 0,
    }
    for record in held:
        example = examples[str(record.metadata["example_id"])]
        local_tokens, offsets = tokenize_with_offsets(tokenizer, example.source_text)
        if local_tokens != record.token_ids.tolist():
            raise ValueError("endogenous semantic tokenizer differs from retained observation")
        expected_program = _canonical_expected_program(example)
        expected_values = tuple(
            item[0] for item in _source_order_annotations(example)
        )
        expected_answer = expected_program.run(expected_values)
        parsed_inputs = semantic_public_token_inputs(example.source_text, offsets)
        observed_annotations = tuple(
            (
                literal.value,
                literal.character_start,
                literal.character_end,
            )
            for literal in parsed_inputs.literals
        )
        accepted = False
        public_input_exact = observed_annotations == _source_order_annotations(example)
        program_exact = False
        answer_exact = False
        refusal = ""
        runtime_receipt_sha256 = ""
        try:
            outcome = execute_compositional_semantic_observation(
                model=model,
                source_text=example.source_text,
                source_token_ids=local_tokens,
                offset_mapping=offsets,
                hidden_states=record.hidden_states,
                worker_model_basis=record.metadata["worker_receipt"]["model_basis"],
                expected_representation_basis_sha256=(
                    expected_representation_basis_sha256
                ),
            )
            accepted = True
            program_exact = outcome.ir.to_program() == expected_program
            answer_exact = outcome.execution.result == expected_answer
            runtime_receipt_sha256 = outcome.receipt["receipt_sha256"]
        except SemanticProgramDecodeRejectedError as exc:
            refusal = f"{type(exc).__name__}:{exc}"
        values = {
            "accepted": accepted,
            "public_input_exact": public_input_exact,
            "program_exact": program_exact,
            "answer_exact": answer_exact,
        }
        for key, value in values.items():
            counts[key] += int(value)
        rows.append(
            {
                "example_id": example.example_id,
                "split": example.split,
                "source_text_sha256": hashlib.sha256(
                    example.source_text.encode("utf-8")
                ).hexdigest(),
                "runtime_receipt_sha256": runtime_receipt_sha256,
                "refusal": refusal,
                **values,
            }
        )

    body = {
        "schema": ENDOGENOUS_SEMANTIC_EVALUATION_SCHEMA,
        "arm": arm,
        "feature_manifest_sha256": bundle.manifest["manifest_sha256"],
        "transducer_receipt_sha256": model.receipt_sha256,
        "representation_basis_sha256": expected_representation_basis_sha256,
        "total": len(held),
        **counts,
        "rows": rows,
        "public_input_recovery": "exact_source_parser",
        "oracle_public_values_available_to_decode": False,
        "family_available_to_decode": False,
        "expected_program_available_to_decode": False,
        "expected_answer_available_to_decode": False,
        "verifier_trace_available_to_decode": False,
        "generated_text_available": False,
        "serving_authority": False,
    }
    receipt = {**body, "receipt_sha256": _sha(body)}
    return EndogenousSemanticEvaluation(
        arm=arm,
        total=len(held),
        rows=tuple(rows),
        receipt=receipt,
        **counts,
    )


def _paired_exact(
    treatment_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    treatment = {str(row["example_id"]): row.get(metric) is True for row in treatment_rows}
    control = {str(row["example_id"]): row.get(metric) is True for row in control_rows}
    if treatment.keys() != control.keys():
        raise ValueError("endogenous semantic paired task sets differ")
    treatment_only = sum(treatment[key] and not control[key] for key in treatment)
    control_only = sum(control[key] and not treatment[key] for key in treatment)
    discordant = treatment_only + control_only
    numerator = (
        sum(
            math.comb(discordant, successes)
            for successes in range(treatment_only, discordant + 1)
        )
        if discordant
        else 1
    )
    denominator = 2**discordant if discordant else 1
    divisor = math.gcd(numerator, denominator)
    return {
        "metric": metric,
        "treatment_only": treatment_only,
        "control_only": control_only,
        "discordant": discordant,
        "one_sided_exact_p_numerator": numerator // divisor,
        "one_sided_exact_p_denominator": denominator // divisor,
        "one_sided_exact_p": numerator / denominator,
    }


def verify_endogenous_semantic_bridge(
    *,
    treatment: EndogenousSemanticEvaluation,
    lesion: EndogenousSemanticEvaluation,
    source_verification_sha256: str,
    source_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    """Seal a causal bridge result without granting it serving authority."""

    if (
        treatment.arm != "treatment"
        or lesion.arm == "treatment"
        or treatment.total != lesion.total
        or treatment.total < 1
        or len(source_verification_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_verification_sha256)
        or set(source_sha256s) != set(ENDOGENOUS_SEMANTIC_VERIFICATION_SOURCES)
        or any(
            not isinstance(path, str)
            or not path
            or not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for path, value in source_sha256s.items()
        )
        or treatment.receipt.get("feature_manifest_sha256")
        != lesion.receipt.get("feature_manifest_sha256")
        or treatment.receipt.get("representation_basis_sha256")
        != lesion.receipt.get("representation_basis_sha256")
    ):
        raise ValueError("endogenous semantic evaluation counts differ")
    paired = {
        metric: _paired_exact(treatment.rows, lesion.rows, metric=metric)
        for metric in ("program_exact", "answer_exact")
    }
    verified = bool(
        treatment.public_input_exact == treatment.total
        and treatment.answer_exact > lesion.answer_exact
        and paired["answer_exact"]["one_sided_exact_p"] < 0.05
    )
    body = {
        "schema": ENDOGENOUS_SEMANTIC_VERIFICATION_SCHEMA,
        "verified": verified,
        "treatment_receipt_sha256": treatment.receipt["receipt_sha256"],
        "lesion_receipt_sha256": lesion.receipt["receipt_sha256"],
        "source_verification_sha256": source_verification_sha256,
        "source_sha256s": dict(sorted(source_sha256s.items())),
        "paired_exact_tests": paired,
        "public_input_exact": treatment.public_input_exact,
        "total": treatment.total,
        "treatment_program_exact": treatment.program_exact,
        "lesion_program_exact": lesion.program_exact,
        "treatment_answer_exact": treatment.answer_exact,
        "lesion_answer_exact": lesion.answer_exact,
        "fit_or_refit_calls": 0,
        "public_input_recovery": "exact_source_parser",
        "oracle_public_values_available_to_decode": False,
        "family_available_to_decode": False,
        "expected_program_available_to_decode": False,
        "expected_answer_available_to_decode": False,
        "verifier_trace_available_to_decode": False,
        "generated_text_available": False,
        "serving_authority": False,
        "claim_boundary": (
            "bounded resident-27B endogenous public-input semantic execution; "
            "no open-domain, frontier-reasoning, or serving claim"
        ),
    }
    return {**body, "verification_sha256": _sha(body)}


__all__ = [
    "ENDOGENOUS_SEMANTIC_EVALUATION_SCHEMA",
    "ENDOGENOUS_SEMANTIC_VERIFICATION_SCHEMA",
    "ENDOGENOUS_SEMANTIC_VERIFICATION_SOURCES",
    "EndogenousSemanticEvaluation",
    "evaluate_endogenous_semantic_bridge",
    "verify_endogenous_semantic_bridge",
]
