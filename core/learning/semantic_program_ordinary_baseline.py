"""Ordinary resident-model control for frozen semantic-program replication."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

from core.brain.llm.latent_cortex.experiments import extract_final_numeric_claim
from core.learning.semantic_program_corpus import (
    SemanticProgramExample,
    build_semantic_program_natural_weave_replication_corpus,
)
from core.learning.semantic_program_frozen_path_replication import (
    FROZEN_PATH_PREREGISTRATION_SCHEMA,
    FROZEN_PATH_RESULT_SCHEMA,
)
from core.learning.semantic_program_path_ensemble_replication import paired_exact_test

ORDINARY_BASELINE_RESULT_SCHEMA: Final = (
    "aura.semantic_program_ordinary_resident_baseline.v1"
)
_INTEGER_TOKEN = re.compile(r"^[+-]?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?$")
_FRACTION_TOKEN = re.compile(r"^[+-]?\d[\d,]*(?:\.\d+)?/[+-]?\d[\d,]*(?:\.\d+)?$")
_SOURCE_FILES: Final = (
    "core/brain/llm/chat_format.py",
    "core/brain/llm/latent_cortex/experiment_tasks.py",
    "core/brain/llm/latent_cortex/experiments.py",
    "core/brain/llm/model_artifact_profile.py",
    "core/learning/semantic_program_corpus.py",
    "core/learning/semantic_program_frozen_path_replication.py",
    "core/learning/semantic_program_ordinary_baseline.py",
    "core/learning/semantic_program_path_ensemble_replication.py",
    "tools/run_semantic_program_ordinary_baseline.py",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def verify_embedded_receipt(value: Mapping[str, Any], *, field: str) -> None:
    claimed = value.get(field)
    body = dict(value)
    body.pop(field, None)
    if not isinstance(claimed, str) or claimed != canonical_sha256(body):
        raise ValueError(f"{field} differs")


def source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    files = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in _SOURCE_FILES
    }
    body = {
        "schema": "aura.semantic_program_ordinary_baseline_source_identity.v1",
        "files": files,
    }
    return {**body, "source_sha256": canonical_sha256(body)}


def _decimal_fraction(value: str) -> Fraction:
    decimal = Decimal(value.replace(",", ""))
    if not decimal.is_finite():
        raise ValueError("numeric claim is not finite")
    return Fraction(decimal)


def parse_integral_numeric_claim(text: str) -> int | None:
    """Return the final answer-shaped value only when it is an exact integer."""

    claim = extract_final_numeric_claim(text)
    if not claim:
        return None
    try:
        if _FRACTION_TOKEN.fullmatch(claim):
            numerator, denominator = claim.split("/", 1)
            divisor = _decimal_fraction(denominator)
            if divisor == 0:
                return None
            value = _decimal_fraction(numerator) / divisor
        elif _INTEGER_TOKEN.fullmatch(claim):
            value = _decimal_fraction(claim)
        else:
            return None
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return int(value) if value.denominator == 1 else None


def target_examples(
    preregistration: Mapping[str, Any],
) -> tuple[SemanticProgramExample, ...]:
    corpus = preregistration.get("replication_corpus")
    if not isinstance(corpus, Mapping):
        raise ValueError("ordinary baseline corpus contract is missing")
    return build_semantic_program_natural_weave_replication_corpus(
        seed=int(corpus["seed"]),
        examples_per_schema_domain=int(corpus["examples_per_schema_domain"]),
    )


def verify_ordinary_baseline_preflight(
    *,
    preregistration: Mapping[str, Any],
    mechanism_result: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    max_tokens: int,
) -> tuple[tuple[SemanticProgramExample, ...], tuple[dict[str, Any], ...]]:
    """Bind the deferred control to the passed mechanism cohort and model."""

    verify_embedded_receipt(mechanism_result, field="result_sha256")
    if (
        preregistration.get("schema") != FROZEN_PATH_PREREGISTRATION_SCHEMA
        or mechanism_result.get("schema") != FROZEN_PATH_RESULT_SCHEMA
        or mechanism_result.get("verdict") != "PASS_MECHANISM_READY_FOR_ORDINARY_BASELINE"
        or mechanism_result.get("ordinary_resident_27b_decode")
        != {"status": "DEFERRED_READY", "model_load_or_decode_calls": 0}
        or preregistration.get("ordinary_decode_deferred_until_mechanism_pass") is not True
        or not 320 <= max_tokens <= 2048
        or descriptor.get("schema") != "aura.model_artifact_descriptor.v1"
    ):
        raise ValueError("ordinary baseline preflight contract differs")
    arms = mechanism_result.get("arms")
    if not isinstance(arms, Mapping):
        raise ValueError("mechanism result arms are missing")
    treatment = arms.get("frozen_transducer")
    if not isinstance(treatment, Mapping) or not isinstance(treatment.get("rows"), list):
        raise ValueError("mechanism treatment rows are missing")
    treatment_rows = tuple(dict(row) for row in treatment["rows"])
    examples = target_examples(preregistration)
    examples_by_id = {
        hashlib.sha256(item.source_text.encode("utf-8")).hexdigest(): item
        for item in examples
    }
    observed_ids = tuple(str(row.get("source_text_sha256")) for row in treatment_rows)
    if (
        len(examples_by_id) != 48
        or len(observed_ids) != 48
        or len(set(observed_ids)) != 48
        or set(observed_ids) != set(examples_by_id)
    ):
        raise ValueError("ordinary baseline task cohort differs from mechanism result")
    ordered_examples = tuple(examples_by_id[source_id] for source_id in observed_ids)
    return ordered_examples, treatment_rows


def ordinary_result_row(
    example: SemanticProgramExample,
    *,
    response_text: str,
    raw_output_sha256: str,
    prompt_tokens_sha256: str,
    prompt_token_count: int,
    generated_token_count: int,
    termination: str,
    native_thinking_boundary_closed: bool,
    model_descriptor_sha256: str,
) -> dict[str, Any]:
    source_sha = hashlib.sha256(example.source_text.encode("utf-8")).hexdigest()
    expected = example.program.run(example.inputs)
    parsed = parse_integral_numeric_claim(response_text)
    return {
        "source_text_sha256": source_sha,
        "construction_id": example.construction_id,
        "topology_id": example.topology_id,
        "split": example.split,
        "response_text": response_text,
        "response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
        "raw_output_sha256": raw_output_sha256,
        "prompt_tokens_sha256": prompt_tokens_sha256,
        "prompt_token_count": int(prompt_token_count),
        "generated_token_count": int(generated_token_count),
        "termination": termination,
        "native_thinking_boundary_closed": bool(native_thinking_boundary_closed),
        "parsed_integer": parsed,
        "answer_exact": parsed == expected,
        "model_descriptor_sha256": model_descriptor_sha256,
    }


def best_possible_product_test(
    treatment_rows: Sequence[Mapping[str, Any]],
    ordinary_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the strongest paired result still possible after partial decoding."""

    if len(ordinary_rows) > len(treatment_rows):
        raise ValueError("ordinary rows exceed the treatment cohort")
    hypothetical = [dict(row) for row in ordinary_rows]
    hypothetical.extend(
        {"answer_exact": False, "source_text_sha256": row["source_text_sha256"]}
        for row in treatment_rows[len(ordinary_rows) :]
    )
    return paired_exact_test(treatment_rows, hypothetical)


def product_bar_is_reachable(
    treatment_rows: Sequence[Mapping[str, Any]],
    ordinary_rows: Sequence[Mapping[str, Any]],
) -> bool:
    best = best_possible_product_test(treatment_rows, ordinary_rows)
    return bool(
        best["treatment_only"] > best["control_only"]
        and best["one_sided_exact_p"] < 0.05
    )


def adjudicate_ordinary_product_bar(
    treatment_rows: Sequence[Mapping[str, Any]],
    ordinary_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(treatment_rows) != 48 or len(ordinary_rows) != 48:
        raise ValueError("ordinary product adjudication requires all 48 tasks")
    treatment_ids = tuple(str(row.get("source_text_sha256")) for row in treatment_rows)
    ordinary_ids = tuple(str(row.get("source_text_sha256")) for row in ordinary_rows)
    if treatment_ids != ordinary_ids or len(set(treatment_ids)) != 48:
        raise ValueError("ordinary and treatment task identities differ")
    paired = paired_exact_test(treatment_rows, ordinary_rows)
    treatment_exact = sum(row.get("answer_exact") is True for row in treatment_rows)
    ordinary_exact = sum(row.get("answer_exact") is True for row in ordinary_rows)
    ordinary_incumbents_retained = all(
        isinstance(row.get("response_text"), str)
        and isinstance(row.get("response_sha256"), str)
        and len(str(row["response_sha256"])) == 64
        for row in ordinary_rows
        if row.get("answer_exact") is True
    )
    passed = bool(
        treatment_exact > ordinary_exact
        and paired["treatment_only"] > paired["control_only"]
        and paired["one_sided_exact_p"] < 0.05
        and ordinary_incumbents_retained
    )
    return {
        "treatment_answer_exact": treatment_exact,
        "ordinary_answer_exact": ordinary_exact,
        "paired_exact_test": paired,
        "ordinary_exact_incumbents_retained": ordinary_incumbents_retained,
        "composition_policy": {
            "policy": "ordinary_incumbent_with_independently_verified_promotion_only",
            "ordinary_response_is_immutable_incumbent": True,
            "semantic_replacement_authorized_by_this_evaluator": False,
        },
        "product_bar_pass": passed,
    }


__all__ = [
    "ORDINARY_BASELINE_RESULT_SCHEMA",
    "adjudicate_ordinary_product_bar",
    "best_possible_product_test",
    "canonical_bytes",
    "canonical_sha256",
    "ordinary_result_row",
    "parse_integral_numeric_claim",
    "product_bar_is_reachable",
    "source_identity",
    "target_examples",
    "verify_embedded_receipt",
    "verify_ordinary_baseline_preflight",
]
