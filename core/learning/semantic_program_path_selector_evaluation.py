"""Development-only causal evaluation of a frozen calibrated path selector."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any, Final

from core.evidence.calibrated_candidate_selector import CalibratedCandidateSelector
from core.evidence.necessary_condition_selector import PairwiseSelectionEvidence
from core.learning.semantic_program_execution import execute_semantic_program
from core.learning.semantic_program_path_ensemble import (
    SemanticProgramPathEnsemble,
    semantic_path_selection_values,
)
from core.learning.semantic_program_transducer import (
    SemanticTransducerTrainingExample,
    SemanticTransductionOutcome,
)

SEMANTIC_PATH_SELECTOR_DEVELOPMENT_RESULT_SCHEMA: Final = (
    "aura.semantic_program_path_selector_development_result.v1"
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


def _correct(
    item: SemanticTransducerTrainingExample,
    outcome: SemanticTransductionOutcome,
) -> bool:
    if outcome.ir is None:
        return False
    try:
        actual = execute_semantic_program(outcome.ir, item.public_inputs).result
    except (RuntimeError, TypeError, ValueError):
        return False
    return actual == item.ir.to_program().run(item.public_inputs)


def _one_sided_exact_p(*, treatment_only: int, control_only: int) -> float:
    discordant = treatment_only + control_only
    if discordant == 0:
        return 1.0
    return sum(
        math.comb(discordant, successes)
        for successes in range(treatment_only, discordant + 1)
    ) / (2**discordant)


def evaluate_calibrated_path_selector_development(
    *,
    ensemble: SemanticProgramPathEnsemble,
    examples: Sequence[SemanticTransducerTrainingExample],
    target_status: str,
) -> dict[str, Any]:
    """Test frozen arbitration without granting fresh-replication authority."""
    if not isinstance(ensemble.selector, CalibratedCandidateSelector):
        raise ValueError("development evaluation requires a calibrated selector")
    items = tuple(examples)
    if (
        not items
        or target_status not in {"EXPOSED_DEVELOPMENT_TARGET", "FRESH_RESERVED_TARGET"}
        or len({item.ir.source_text_sha256 for item in items}) != len(items)
        or any(
            item.ir.model_basis_receipt_sha256 != ensemble.model_basis_sha256
            for item in items
        )
    ):
        raise ValueError("selector development target contract is invalid")
    started = time.monotonic()
    rows = []
    for item in items:
        arbitrated = ensemble.decode_with_receipt(
            source_token_ids=item.ir.source_token_ids,
            hidden_states=item.hidden_states,
            public_inputs=item.public_inputs,
            source_text_sha256=item.ir.source_text_sha256,
            model_basis_sha256=item.ir.model_basis_receipt_sha256,
        )
        evidence = PairwiseSelectionEvidence.from_mappings(
            incumbent=semantic_path_selection_values(arbitrated.incumbent),
            challenger=semantic_path_selection_values(arbitrated.challenger),
            packet=arbitrated.decision.evidence,
        )
        lesion = ensemble.selector.necessary.select(
            incumbent="incumbent",
            challenger="challenger",
            evidence=evidence,
        )
        lesion_outcome = (
            arbitrated.challenger
            if lesion.selected == "challenger"
            else arbitrated.incumbent
        )
        incumbent_values = evidence.values("incumbent")
        challenger_values = evidence.values("challenger")
        incumbent_quality = ensemble.selector.scorer.predict(incumbent_values)
        challenger_quality = ensemble.selector.scorer.predict(challenger_values)
        rows.append(
            {
                "source_text_sha256": item.ir.source_text_sha256,
                "split": item.split,
                "construction_id": item.construction_id,
                "topology_id": item.topology_id,
                "incumbent_correct": _correct(item, arbitrated.incumbent),
                "challenger_correct": _correct(item, arbitrated.challenger),
                "selected_correct": _correct(item, arbitrated.outcome),
                "necessary_lesion_correct": _correct(item, lesion_outcome),
                "selected_path": arbitrated.decision.selected,
                "selection_reason": arbitrated.decision.receipt["reason"],
                "necessary_lesion_path": lesion.selected,
                "incumbent_quality": incumbent_quality,
                "challenger_quality": challenger_quality,
                "quality_delta": challenger_quality - incumbent_quality,
                "switch_margin": ensemble.selector.switch_margin,
                "incumbent_selection_values": incumbent_values,
                "challenger_selection_values": challenger_values,
            }
        )
    improvements = sum(
        row["selected_correct"] and not row["incumbent_correct"] for row in rows
    )
    regressions = sum(
        row["incumbent_correct"] and not row["selected_correct"] for row in rows
    )
    treatment_only = sum(
        row["selected_correct"] and not row["necessary_lesion_correct"] for row in rows
    )
    lesion_only = sum(
        row["necessary_lesion_correct"] and not row["selected_correct"] for row in rows
    )
    p_value = _one_sided_exact_p(
        treatment_only=treatment_only,
        control_only=lesion_only,
    )
    mechanism_pass = bool(
        improvements >= 5
        and regressions == 0
        and treatment_only >= 5
        and lesion_only == 0
        and p_value < 0.05
    )
    body = {
        "schema": SEMANTIC_PATH_SELECTOR_DEVELOPMENT_RESULT_SCHEMA,
        "target_status": target_status,
        "claim_authority": (
            "DEVELOPMENT_ONLY_TARGET_ALREADY_EXPOSED"
            if target_status == "EXPOSED_DEVELOPMENT_TARGET"
            else "FRESH_TARGET_REQUIRES_SEPARATE_PREREGISTERED_ADJUDICATION"
        ),
        "ensemble_receipt_sha256": ensemble.receipt_sha256,
        "selector_receipt_sha256": ensemble.selector.receipt_sha256,
        "tasks": len(rows),
        "incumbent_correct": sum(row["incumbent_correct"] for row in rows),
        "challenger_correct": sum(row["challenger_correct"] for row in rows),
        "selected_correct": sum(row["selected_correct"] for row in rows),
        "necessary_lesion_correct": sum(
            row["necessary_lesion_correct"] for row in rows
        ),
        "improvements_over_incumbent": improvements,
        "regressions_from_incumbent": regressions,
        "selected_only_vs_necessary_lesion": treatment_only,
        "necessary_lesion_only": lesion_only,
        "one_sided_exact_p": p_value,
        "selection_reasons": dict(
            sorted(Counter(row["selection_reason"] for row in rows).items())
        ),
        "mechanism_pass": mechanism_pass,
        "verdict": (
            "PASS_DEVELOPMENT_SELECTOR_MECHANISM"
            if mechanism_pass
            else "FAIL_DEVELOPMENT_SELECTOR_MECHANISM"
        ),
        "rows": rows,
        "wall_time_s": time.monotonic() - started,
        "expected_answers_available_to_paths_or_selector": False,
        "expected_answers_available_to_evaluator": True,
        "serving_authority": False,
    }
    return {**body, "result_sha256": _sha(body)}


__all__ = [
    "SEMANTIC_PATH_SELECTOR_DEVELOPMENT_RESULT_SCHEMA",
    "evaluate_calibrated_path_selector_development",
]
