"""Fresh evaluation and causal lesions for a frozen semantic path ensemble."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any, Final

from core.evidence.necessary_condition_selector import PairwiseSelectionEvidence
from core.learning.semantic_program_basis import (
    bind_examples_to_compatible_training_session,
    establish_semantic_representation_compatibility,
)
from core.learning.semantic_program_campaign import training_examples_from_feature_bundle
from core.learning.semantic_program_corpus import (
    build_semantic_program_corpus,
    build_semantic_program_fork_join_corpus,
    build_semantic_program_natural_alias_source_corpus,
    build_semantic_program_natural_branch_replication_corpus,
    build_semantic_program_natural_identity_source_corpus,
    build_semantic_program_natural_replication_corpus,
    build_semantic_program_natural_request_corpus,
    build_semantic_program_natural_source_corpus,
    build_semantic_program_sequence_cataphoric_corpus,
    build_semantic_program_sequence_reserved_alias_corpus,
    build_semantic_program_sequence_role_binding_corpus,
)
from core.learning.semantic_program_execution import execute_semantic_program
from core.learning.semantic_program_feature_materialization import (
    NATURAL_BRANCH_REPLICATION_CORPUS_KIND,
    LoadedSemanticFeatureBundle,
)
from core.learning.semantic_program_natural_transfer import procedure_schema_signature
from core.learning.semantic_program_path_ensemble import (
    EXECUTABLE_PROGRAM_CONDITION,
    SemanticProgramPathEnsemble,
    semantic_path_selection_values,
)
from core.learning.semantic_program_transducer import (
    SemanticTransducerTrainingExample,
    SemanticTransductionOutcome,
)

PATH_ENSEMBLE_REPLICATION_PREREGISTRATION_SCHEMA: Final = (
    "aura.semantic_program_path_ensemble_replication_preregistration.v1"
)
PATH_ENSEMBLE_REPLICATION_RESULT_SCHEMA: Final = (
    "aura.semantic_program_path_ensemble_replication_result.v1"
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _one_sided_exact_p(*, treatment_only: int, control_only: int) -> float:
    discordant = treatment_only + control_only
    if discordant == 0:
        return 1.0
    return sum(
        math.comb(discordant, successes)
        for successes in range(treatment_only, discordant + 1)
    ) / (2**discordant)


def paired_exact_test(
    treatment_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str = "answer_exact",
) -> dict[str, Any]:
    """Recount a paired exact test from task rows."""

    if metric not in {"answer_exact", "program_exact"}:
        raise ValueError("path ensemble paired metric is unsupported")
    treatment = {
        str(row["source_text_sha256"]): row.get(metric) is True for row in treatment_rows
    }
    control = {
        str(row["source_text_sha256"]): row.get(metric) is True for row in control_rows
    }
    if not treatment or treatment.keys() != control.keys():
        raise ValueError("path ensemble paired arms contain different tasks")
    treatment_only = sum(treatment[key] and not control[key] for key in treatment)
    control_only = sum(control[key] and not treatment[key] for key in treatment)
    return {
        "metric": metric,
        "treatment_only": treatment_only,
        "control_only": control_only,
        "discordant": treatment_only + control_only,
        "one_sided_exact_p": _one_sided_exact_p(
            treatment_only=treatment_only,
            control_only=control_only,
        ),
    }


def verify_branch_replication_preflight(
    *,
    bundle: LoadedSemanticFeatureBundle,
    training_manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    ensemble: SemanticProgramPathEnsemble,
) -> tuple[dict[str, Any], tuple[SemanticTransducerTrainingExample, ...]]:
    """Bind the generated corpus and feature basis to the frozen contract."""

    corpus_contract = preregistration.get("corpus")
    frozen_paths = preregistration.get("frozen_paths")
    config = bundle.manifest.get("config")
    target = build_semantic_program_natural_branch_replication_corpus()
    if (
        preregistration.get("schema")
        != PATH_ENSEMBLE_REPLICATION_PREREGISTRATION_SCHEMA
        or preregistration.get("preregistered_before_generator_implementation") is not True
        or preregistration.get("preregistered_before_target_generation") is not True
        or not isinstance(corpus_contract, Mapping)
        or not isinstance(frozen_paths, Mapping)
        or not isinstance(config, Mapping)
        or config.get("corpus_kind") != NATURAL_BRANCH_REPLICATION_CORPUS_KIND
        or config.get("seed") != corpus_contract.get("seed")
        or config.get("examples_per_operation_pair")
        != corpus_contract.get("examples_per_schema_domain")
        or config.get("max_examples") != corpus_contract.get("max_examples")
        or bundle.manifest.get("complete") is not True
        or bundle.manifest.get("example_count") != len(target)
        or frozen_paths.get("model_basis_sha256") != ensemble.model_basis_sha256
        or frozen_paths.get("incumbent", {}).get("receipt_sha256")
        != ensemble.incumbent.receipt_sha256
        or frozen_paths.get("challenger", {}).get("receipt_sha256")
        != ensemble.challenger.receipt_sha256
    ):
        raise ValueError("path ensemble replication contract differs")

    raw_examples = training_examples_from_feature_bundle(
        bundle,
        required_splits=frozenset({"validation", "test"}),
    )
    incumbent_compatibility = establish_semantic_representation_compatibility(
        model=ensemble.incumbent,
        training_manifest=training_manifest,
        replication_manifest=bundle.manifest,
    )
    challenger_compatibility = establish_semantic_representation_compatibility(
        model=ensemble.challenger,
        training_manifest=training_manifest,
        replication_manifest=bundle.manifest,
    )
    examples = bind_examples_to_compatible_training_session(
        raw_examples,
        compatibility=incumbent_compatibility,
    )
    expected_graph = tuple(
        tuple(int(value) for value in pair)
        for pair in corpus_contract["operation_graph"]["arguments"]
    )
    expected_topologies = set(corpus_contract["schemas"])
    if (
        len(examples) != corpus_contract["max_examples"]
        or {item.topology_id for item in examples} != expected_topologies
        or any(
            tuple(step.args for step in item.ir.instructions) != expected_graph
            or item.ir.report_value != corpus_contract["operation_graph"]["report_value"]
            or item.ir.model_basis_receipt_sha256 != ensemble.model_basis_sha256
            for item in examples
        )
    ):
        raise ValueError("path ensemble replication graph or model basis differs")

    prior = (
        *build_semantic_program_natural_request_corpus(),
        *build_semantic_program_natural_replication_corpus(),
        *build_semantic_program_natural_source_corpus(),
        *build_semantic_program_natural_alias_source_corpus(),
        *build_semantic_program_natural_identity_source_corpus(),
    )
    fit = (
        *build_semantic_program_corpus(),
        *build_semantic_program_fork_join_corpus(),
        *build_semantic_program_sequence_cataphoric_corpus(),
        *build_semantic_program_sequence_reserved_alias_corpus(),
        *build_semantic_program_sequence_role_binding_corpus(),
        *prior,
    )
    target_texts = {item.source_text for item in target}
    target_constructions = {item.construction_id for item in target}
    target_schemas = {procedure_schema_signature(item) for item in target}
    if (
        target_texts & {item.source_text for item in prior}
        or target_constructions & {item.construction_id for item in prior}
        or target_schemas & {procedure_schema_signature(item) for item in fit}
    ):
        raise ValueError("path ensemble replication overlaps earlier evidence")
    return ({
        "example_count": len(examples),
        "feature_manifest_sha256": bundle.manifest["manifest_sha256"],
        "corpus_sha256": bundle.manifest["corpus_sha256"],
        "model_basis_sha256": ensemble.model_basis_sha256,
        "target_source_text_overlap": 0,
        "target_construction_overlap": 0,
        "target_procedure_schema_overlap": 0,
        "incumbent_representation_compatibility": incumbent_compatibility,
        "challenger_representation_compatibility": challenger_compatibility,
        "expected_answers_available_to_paths_or_selector": False,
    }, examples)


def _outcome_row(
    item: SemanticTransducerTrainingExample,
    outcome: SemanticTransductionOutcome,
    *,
    arm: str,
    selected_path: str,
    selector_reason: str,
) -> dict[str, Any]:
    predicted = outcome.ir
    predicted_answer: Any = None
    if predicted is not None:
        try:
            predicted_answer = execute_semantic_program(predicted, item.public_inputs).result
        except (RuntimeError, TypeError, ValueError):
            predicted_answer = None
    expected_answer = item.ir.to_program().run(item.public_inputs)
    return {
        "source_text_sha256": item.ir.source_text_sha256,
        "construction_id": item.construction_id,
        "topology_id": item.topology_id,
        "split": item.split,
        "arm": arm,
        "selected_path": selected_path,
        "selector_reason": selector_reason,
        "refusal": outcome.refusal,
        "accepted": predicted is not None,
        "program_exact": bool(
            predicted is not None and predicted.to_program() == item.ir.to_program()
        ),
        "answer_emitted": predicted_answer is not None,
        "answer_exact": bool(
            predicted_answer is not None and predicted_answer == expected_answer
        ),
    }


def _arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "accepted": sum(row.get("accepted") is True for row in rows),
        "program_exact": sum(row.get("program_exact") is True for row in rows),
        "answer_emitted": sum(row.get("answer_emitted") is True for row in rows),
        "answer_exact": sum(row.get("answer_exact") is True for row in rows),
        "rows": [dict(row) for row in rows],
    }


def evaluate_path_ensemble_replication(
    *,
    bundle: LoadedSemanticFeatureBundle,
    training_manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    ensemble: SemanticProgramPathEnsemble,
) -> dict[str, Any]:
    """Run both paths once and derive the treatment and selector lesions."""

    started = time.monotonic()
    preflight, examples = verify_branch_replication_preflight(
        bundle=bundle,
        training_manifest=training_manifest,
        preregistration=preregistration,
        ensemble=ensemble,
    )
    rows: dict[str, list[dict[str, Any]]] = {
        "frozen_incumbent": [],
        "frozen_challenger": [],
        "necessary_condition_ensemble": [],
        "forced_incumbent_selector_lesion": [],
        "challenger_evidence_absent_lesion": [],
    }
    for item in examples:
        arbitrated = ensemble.decode_with_receipt(
            source_token_ids=item.ir.source_token_ids,
            hidden_states=item.hidden_states,
            public_inputs=item.public_inputs,
            source_text_sha256=item.ir.source_text_sha256,
            model_basis_sha256=item.ir.model_basis_receipt_sha256,
        )
        rows["frozen_incumbent"].append(
            _outcome_row(
                item,
                arbitrated.incumbent,
                arm="frozen_incumbent",
                selected_path="incumbent",
                selector_reason="standalone_path",
            )
        )
        rows["frozen_challenger"].append(
            _outcome_row(
                item,
                arbitrated.challenger,
                arm="frozen_challenger",
                selected_path="challenger",
                selector_reason="standalone_path",
            )
        )
        treatment = _outcome_row(
            item,
            arbitrated.outcome,
            arm="necessary_condition_ensemble",
            selected_path=arbitrated.decision.selected,
            selector_reason=str(arbitrated.decision.receipt["reason"]),
        )
        treatment["incumbent_ir_available"] = arbitrated.incumbent.ir is not None
        treatment["challenger_ir_available"] = arbitrated.challenger.ir is not None
        rows["necessary_condition_ensemble"].append(treatment)
        rows["forced_incumbent_selector_lesion"].append(
            _outcome_row(
                item,
                arbitrated.incumbent,
                arm="forced_incumbent_selector_lesion",
                selected_path="incumbent",
                selector_reason="selector_forced_to_incumbent",
            )
        )
        evidence_lesion = ensemble.selector.select(
            incumbent="incumbent",
            challenger="challenger",
            evidence=PairwiseSelectionEvidence.from_mappings(
                incumbent=semantic_path_selection_values(arbitrated.incumbent),
                challenger={EXECUTABLE_PROGRAM_CONDITION: 0.0},
                packet=arbitrated.decision.evidence,
            ),
        )
        lesion_outcome = (
            arbitrated.challenger
            if evidence_lesion.selected == "challenger"
            else arbitrated.incumbent
        )
        rows["challenger_evidence_absent_lesion"].append(
            _outcome_row(
                item,
                lesion_outcome,
                arm="challenger_evidence_absent_lesion",
                selected_path=evidence_lesion.selected,
                selector_reason=str(evidence_lesion.receipt["reason"]),
            )
        )

    treatment_rows = rows["necessary_condition_ensemble"]
    incumbent_rows = rows["frozen_incumbent"]
    improvements = [
        row
        for row, incumbent in zip(treatment_rows, incumbent_rows, strict=True)
        if row["answer_exact"] and not incumbent["answer_exact"]
    ]
    regressions = [
        row
        for row, incumbent in zip(treatment_rows, incumbent_rows, strict=True)
        if incumbent["answer_exact"] and not row["answer_exact"]
    ]
    paired = {
        lesion: paired_exact_test(treatment_rows, rows[lesion])
        for lesion in (
            "forced_incumbent_selector_lesion",
            "challenger_evidence_absent_lesion",
        )
    }
    causal_improvements = all(
        row["selected_path"] == "challenger"
        and row["incumbent_ir_available"] is False
        and row["challenger_ir_available"] is True
        and row["selector_reason"] == "challenger_repairs_necessary_condition_failure"
        for row in improvements
    )
    mechanism_pass = (
        not regressions
        and len(improvements) >= 5
        and causal_improvements
        and all(test["one_sided_exact_p"] < 0.05 for test in paired.values())
    )
    body = {
        "schema": PATH_ENSEMBLE_REPLICATION_RESULT_SCHEMA,
        "claim_boundary": preregistration["claim_boundary"],
        "preregistration_source_commit": preregistration[
            "source_commit_before_preregistration"
        ],
        "preflight": preflight,
        "ensemble_receipt_sha256": ensemble.receipt_sha256,
        "arms": {name: _arm(value) for name, value in rows.items()},
        "paired_exact_tests": paired,
        "improvements_over_incumbent": len(improvements),
        "regressions_from_incumbent": len(regressions),
        "causal_improvement_contract_satisfied": causal_improvements,
        "ordinary_resident_27b_decode": {
            "status": "DEFERRED_UNTIL_MECHANISM_PASS" if mechanism_pass else "ABORTED",
            "model_load_or_decode_calls": 0,
        },
        "verdict": (
            "PASS_MECHANISM_READY_FOR_ORDINARY_BASELINE"
            if mechanism_pass
            else "FAIL_PREREGISTERED_MECHANISM_BAR"
        ),
        "wall_time_s": time.monotonic() - started,
        "expected_answers_available_to_paths_or_selector": False,
        "serving_authority": False,
    }
    return {**body, "result_sha256": _sha(body)}


__all__ = [
    "PATH_ENSEMBLE_REPLICATION_PREREGISTRATION_SCHEMA",
    "PATH_ENSEMBLE_REPLICATION_RESULT_SCHEMA",
    "evaluate_path_ensemble_replication",
    "paired_exact_test",
    "verify_branch_replication_preflight",
]
