"""Independent direct-versus-floor replay for learned semantic programs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Final

from core.learning.semantic_program_execution import execute_semantic_program
from core.learning.semantic_program_floor import (
    compile_semantic_program_to_floor,
    execute_semantic_floor_program,
    semantic_floor_primitive_coverage,
)
from core.learning.semantic_program_shared_transducer import (
    SharedSemanticProgramTransducer,
)
from core.learning.semantic_program_transducer import SemanticTransducerTrainingExample

SEMANTIC_PROGRAM_FLOOR_VERIFICATION_SCHEMA: Final = (
    "aura.semantic_program_floor_equivalence_verification.v1"
)
SEMANTIC_PROGRAM_FLOOR_VERIFICATION_SOURCES: Final = (
    "core/cognition/the_floor_she_stands_on.py",
    "core/learning/procedure_induction.py",
    "core/learning/semantic_program_execution.py",
    "core/learning/semantic_program_floor.py",
    "core/learning/semantic_program_floor_verification.py",
    "core/learning/semantic_program_ir.py",
    "core/learning/semantic_program_shared_transducer.py",
    "tools/verify_semantic_program_floor_equivalence.py",
)


def _sha(value: Any) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _outcome(ir: Any, public_inputs: tuple[Any, ...], *, floor: bool) -> tuple[str, Any]:
    try:
        if floor:
            result = execute_semantic_floor_program(
                compile_semantic_program_to_floor(ir, public_inputs)
            ).result
        else:
            result = execute_semantic_program(ir, public_inputs).result
    except (ArithmeticError, RuntimeError, TypeError, ValueError):
        return "refused", None
    return "value", result


def verify_semantic_program_floor_equivalence(
    model: SharedSemanticProgramTransducer,
    examples: Sequence[SemanticTransducerTrainingExample],
    *,
    feature_manifest_sha256s: Mapping[str, str],
    source_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    """Replay test programs through both engines without answers or fitting."""

    if not isinstance(model, SharedSemanticProgramTransducer):
        raise TypeError("floor verification requires a shared semantic transducer")
    if set(source_sha256s) != set(SEMANTIC_PROGRAM_FLOOR_VERIFICATION_SOURCES) or any(
        not _is_sha256(value) for value in source_sha256s.values()
    ):
        raise ValueError("semantic floor verification source inventory differs")
    if not feature_manifest_sha256s or any(
        not _is_sha256(value) for value in feature_manifest_sha256s.values()
    ):
        raise ValueError("semantic floor feature manifests differ")
    selected = tuple(item for item in examples if item.split == "test")
    if not selected:
        raise ValueError("semantic floor verification has no test examples")

    accepted = 0
    agreements = 0
    value_agreements = 0
    refusal_agreements = 0
    by_family: dict[str, dict[str, int]] = defaultdict(
        lambda: {"accepted": 0, "agreements": 0, "value_agreements": 0, "refusal_agreements": 0}
    )
    rows: list[dict[str, Any]] = []
    for item in selected:
        decoded = model.decode(
            source_token_ids=item.ir.source_token_ids,
            hidden_states=item.hidden_states,
            public_inputs=item.public_inputs,
            source_text_sha256=item.ir.source_text_sha256,
            model_basis_sha256=item.ir.model_basis_receipt_sha256,
        )
        predicted = decoded.ir
        family = item.construction_id.partition(":")[0]
        if predicted is None:
            rows.append(
                {
                    "source_text_sha256": item.ir.source_text_sha256,
                    "family": family,
                    "accepted": False,
                    "agreement": False,
                    "refusal": decoded.refusal,
                }
            )
            continue
        accepted += 1
        by_family[family]["accepted"] += 1
        direct = _outcome(predicted, item.public_inputs, floor=False)
        floor = _outcome(predicted, item.public_inputs, floor=True)
        agreement = direct == floor
        value_agreement = agreement and direct[0] == "value"
        refusal_agreement = agreement and direct[0] == "refused"
        agreements += int(agreement)
        value_agreements += int(value_agreement)
        refusal_agreements += int(refusal_agreement)
        by_family[family]["agreements"] += int(agreement)
        by_family[family]["value_agreements"] += int(value_agreement)
        by_family[family]["refusal_agreements"] += int(refusal_agreement)
        rows.append(
            {
                "source_text_sha256": item.ir.source_text_sha256,
                "family": family,
                "accepted": True,
                "agreement": agreement,
                "direct_status": direct[0],
                "floor_status": floor[0],
                "result_sha256": _sha(
                    list(direct[1]) if isinstance(direct[1], tuple) else direct[1]
                )
                if value_agreement
                else None,
            }
        )
    coverage = semantic_floor_primitive_coverage()
    if agreements != accepted or accepted != len(selected) or not coverage["complete"]:
        raise ValueError("semantic floor equivalence is not complete")
    body = {
        "schema": SEMANTIC_PROGRAM_FLOOR_VERIFICATION_SCHEMA,
        "verified": True,
        "transducer_receipt_sha256": model.receipt_sha256,
        "feature_manifest_sha256s": dict(sorted(feature_manifest_sha256s.items())),
        "source_sha256s": dict(sorted(source_sha256s.items())),
        "test_total": len(selected),
        "accepted": accepted,
        "agreements": agreements,
        "value_agreements": value_agreements,
        "refusal_agreements": refusal_agreements,
        "by_family": {key: by_family[key] for key in sorted(by_family)},
        "primitive_coverage": coverage,
        "rows_sha256": _sha(rows),
        "fit_or_refit_calls": 0,
        "expected_answers_available": False,
        "family_router_present": False,
        "serving_authority": False,
        "claim_boundary": (
            "the frozen shared semantic transducer's accepted test programs have "
            "identical outcomes under the closed exact executor and universal "
            "metered floor; no unseen-schema, serving, or broad reasoning claim"
        ),
    }
    return {**body, "verification_sha256": _sha(body)}


__all__ = [
    "SEMANTIC_PROGRAM_FLOOR_VERIFICATION_SCHEMA",
    "SEMANTIC_PROGRAM_FLOOR_VERIFICATION_SOURCES",
    "verify_semantic_program_floor_equivalence",
]
