"""Preregistered causal replication for calibrated semantic-path arbitration."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any, Final

from core.evidence.calibrated_candidate_selector import CalibratedCandidateSelector
from core.evidence.necessary_condition_selector import PairwiseSelectionEvidence
from core.learning.semantic_program_basis import (
    bind_examples_to_compatible_training_session,
    establish_semantic_representation_compatibility,
)
from core.learning.semantic_program_campaign import training_examples_from_feature_bundle
from core.learning.semantic_program_corpus import (
    NATURAL_WEAVE_REPLICATION_DOMAINS,
    build_semantic_program_corpus,
    build_semantic_program_fork_join_corpus,
    build_semantic_program_natural_alias_source_corpus,
    build_semantic_program_natural_branch_replication_corpus,
    build_semantic_program_natural_identity_source_corpus,
    build_semantic_program_natural_replication_corpus,
    build_semantic_program_natural_request_corpus,
    build_semantic_program_natural_source_corpus,
    build_semantic_program_natural_weave_replication_corpus,
    build_semantic_program_sequence_cataphoric_corpus,
    build_semantic_program_sequence_reserved_alias_corpus,
    build_semantic_program_sequence_role_binding_corpus,
)
from core.learning.semantic_program_execution import execute_semantic_program
from core.learning.semantic_program_feature_materialization import (
    NATURAL_WEAVE_REPLICATION_CORPUS_KIND,
    LoadedSemanticFeatureBundle,
)
from core.learning.semantic_program_natural_transfer import procedure_schema_signature
from core.learning.semantic_program_path_ensemble import (
    SemanticProgramPathEnsemble,
    semantic_path_selection_values,
)
from core.learning.semantic_program_path_ensemble_replication import paired_exact_test
from core.learning.semantic_program_transducer import (
    SemanticTransducerTrainingExample,
    SemanticTransductionOutcome,
)

CALIBRATED_PATH_REPLICATION_PREREGISTRATION_SCHEMA: Final = (
    "aura.semantic_program_calibrated_path_replication_preregistration.v1"
)
CALIBRATED_PATH_REPLICATION_RESULT_SCHEMA: Final = (
    "aura.semantic_program_calibrated_path_replication_result.v1"
)
_ARM_NAMES: Final = (
    "calibrated_path_ensemble",
    "frozen_incumbent",
    "frozen_challenger",
    "necessary_condition_selector_lesion",
    "forced_incumbent_selector_lesion",
    "source_only_calibration_control",
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


def _prior_corpus() -> tuple[Any, ...]:
    return (
        *build_semantic_program_natural_request_corpus(),
        *build_semantic_program_natural_replication_corpus(),
        *build_semantic_program_natural_source_corpus(),
        *build_semantic_program_natural_alias_source_corpus(),
        *build_semantic_program_natural_identity_source_corpus(),
        *build_semantic_program_natural_branch_replication_corpus(),
    )


def _fit_and_prior_corpus() -> tuple[Any, ...]:
    return (
        *build_semantic_program_corpus(),
        *build_semantic_program_fork_join_corpus(),
        *build_semantic_program_sequence_cataphoric_corpus(),
        *build_semantic_program_sequence_reserved_alias_corpus(),
        *build_semantic_program_sequence_role_binding_corpus(),
        *_prior_corpus(),
    )


def _require_calibrated(ensemble: SemanticProgramPathEnsemble) -> CalibratedCandidateSelector:
    selector = ensemble.selector
    if not isinstance(selector, CalibratedCandidateSelector):
        raise ValueError("calibrated path replication requires a calibrated selector")
    return selector


def verify_calibrated_path_replication_preflight(
    *,
    bundle: LoadedSemanticFeatureBundle,
    training_manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    ensemble: SemanticProgramPathEnsemble,
    source_only_control: SemanticProgramPathEnsemble,
) -> tuple[dict[str, Any], tuple[SemanticTransducerTrainingExample, ...]]:
    """Bind fresh features, frozen selectors, and unseen procedure geometry."""

    corpus_contract = preregistration.get("corpus")
    frozen = preregistration.get("frozen_ensemble")
    source_frozen = preregistration.get("frozen_source_only_control")
    correction = preregistration.get("post_implementation_contract_corrections")
    config = bundle.manifest.get("config")
    mixed_selector = _require_calibrated(ensemble)
    source_selector = _require_calibrated(source_only_control)
    if not all(
        isinstance(value, Mapping) for value in (corpus_contract, frozen, source_frozen, config)
    ):
        raise ValueError("calibrated replication contract is incomplete")
    assert isinstance(corpus_contract, Mapping)
    assert isinstance(frozen, Mapping)
    assert isinstance(source_frozen, Mapping)
    assert isinstance(config, Mapping)
    if (
        preregistration.get("schema") != CALIBRATED_PATH_REPLICATION_PREREGISTRATION_SCHEMA
        or preregistration.get("preregistered_before_generator_implementation") is not True
        or preregistration.get("preregistered_before_target_generation") is not True
        or preregistration.get("ordinary_decode_deferred_until_mechanism_pass") is not True
        or preregistration.get("serving_authority") is not False
        or preregistration.get("arms") != [*_ARM_NAMES, "ordinary_resident_27b_decode"]
        or correction
        != [
            {
                "field": "corpus.operation_graph.dependencies[3]",
                "original": [2, 1],
                "corrected": [1, 2],
                "reason": (
                    "Dependency identities are canonicalized in ascending producer order. "
                    "Operation argument order remains [8, 7], so program semantics are unchanged."
                ),
                "target_examples_generated_before_correction": 0,
            }
        ]
        or config.get("corpus_kind") != NATURAL_WEAVE_REPLICATION_CORPUS_KIND
        or config.get("seed") != corpus_contract.get("seed")
        or config.get("examples_per_operation_pair")
        != corpus_contract.get("examples_per_schema_domain")
        or config.get("max_examples") != corpus_contract.get("max_examples")
        or bundle.manifest.get("complete") is not True
        or frozen.get("receipt_sha256") != ensemble.receipt_sha256
        or frozen.get("selector_receipt_sha256") != mixed_selector.receipt_sha256
        or frozen.get("calibration_report_sha256")
        != ensemble.composition_receipt.get("calibration_report_sha256")
        or frozen.get("model_basis_sha256") != ensemble.model_basis_sha256
        or frozen.get("incumbent_receipt_sha256") != ensemble.incumbent.receipt_sha256
        or frozen.get("challenger_receipt_sha256") != ensemble.challenger.receipt_sha256
        or source_frozen.get("receipt_sha256") != source_only_control.receipt_sha256
        or source_frozen.get("selector_receipt_sha256") != source_selector.receipt_sha256
        or source_frozen.get("calibration_report_sha256")
        != source_only_control.composition_receipt.get("calibration_report_sha256")
        or source_only_control.model_basis_sha256 != ensemble.model_basis_sha256
        or source_only_control.incumbent.receipt_sha256 != ensemble.incumbent.receipt_sha256
        or source_only_control.challenger.receipt_sha256 != ensemble.challenger.receipt_sha256
    ):
        raise ValueError("calibrated path replication contract differs")

    target = build_semantic_program_natural_weave_replication_corpus(
        seed=int(corpus_contract["seed"]),
        examples_per_schema_domain=int(corpus_contract["examples_per_schema_domain"]),
    )
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
    expected_arguments = tuple(
        tuple(int(value) for value in pair)
        for pair in corpus_contract["operation_graph"]["arguments"]
    )
    expected_dependencies = tuple(
        tuple(int(value) for value in row)
        for row in corpus_contract["operation_graph"]["dependencies"]
    )
    expected_by_source = {
        hashlib.sha256(item.source_text.encode("utf-8")).hexdigest(): item for item in target
    }
    observed_by_source = {item.ir.source_text_sha256: item for item in examples}
    if (
        len(target) != corpus_contract.get("max_examples")
        or bundle.manifest.get("example_count") != len(target)
        or len(observed_by_source) != len(examples)
        or observed_by_source.keys() != expected_by_source.keys()
        or set(NATURAL_WEAVE_REPLICATION_DOMAINS)
        != set(corpus_contract.get("domain_inventory", ()))
        or {item.topology_id for item in target} != set(corpus_contract.get("schemas", ()))
        or any(
            item.public_inputs != expected_by_source[source].inputs
            or tuple(step.args for step in item.ir.instructions) != expected_arguments
            or tuple(step.depends_on for step in item.ir.instructions) != expected_dependencies
            or item.ir.report_value != corpus_contract["operation_graph"]["report_value"]
            or item.ir.model_basis_receipt_sha256 != ensemble.model_basis_sha256
            for source, item in observed_by_source.items()
        )
    ):
        raise ValueError("calibrated replication graph, inputs, or model basis differs")

    prior = _prior_corpus()
    fit_and_prior = _fit_and_prior_corpus()
    target_texts = {item.source_text for item in target}
    target_constructions = {item.construction_id for item in target}
    target_topologies = {item.topology_id for item in target}
    target_schemas = {procedure_schema_signature(item) for item in target}
    if (
        target_texts & {item.source_text for item in prior}
        or target_constructions & {item.construction_id for item in prior}
        or target_topologies & {item.topology_id for item in prior}
        or target_schemas & {procedure_schema_signature(item) for item in fit_and_prior}
    ):
        raise ValueError("calibrated replication overlaps earlier evidence")
    preflight = {
        "example_count": len(examples),
        "feature_manifest_sha256": bundle.manifest["manifest_sha256"],
        "corpus_sha256": bundle.manifest["corpus_sha256"],
        "model_basis_sha256": ensemble.model_basis_sha256,
        "target_source_text_overlap": 0,
        "target_construction_overlap": 0,
        "target_topology_overlap": 0,
        "target_procedure_schema_overlap": 0,
        "public_input_recovery_exact": True,
        "incumbent_representation_compatibility": incumbent_compatibility,
        "challenger_representation_compatibility": challenger_compatibility,
        "mixed_selector_receipt_sha256": mixed_selector.receipt_sha256,
        "source_only_selector_receipt_sha256": source_selector.receipt_sha256,
        "path_decodes_per_task": 2,
        "target_labels_available_to_paths_or_selectors": False,
        "text_or_domain_identity_available_to_selectors": False,
    }
    return preflight, examples


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
    expected_program = item.ir.to_program()
    expected_answer = expected_program.run(item.public_inputs)
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
        "program_exact": bool(predicted is not None and predicted.to_program() == expected_program),
        "answer_emitted": predicted_answer is not None,
        "answer_exact": bool(predicted_answer is not None and predicted_answer == expected_answer),
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


def adjudicate_calibrated_path_replication(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Apply the frozen success bar to already measured, task-matched arms."""

    if set(rows) != set(_ARM_NAMES) or any(not rows[name] for name in _ARM_NAMES):
        raise ValueError("calibrated replication arms are incomplete")
    identities = {
        name: tuple(str(row.get("source_text_sha256")) for row in rows[name]) for name in _ARM_NAMES
    }
    first = identities[_ARM_NAMES[0]]
    if len(set(first)) != len(first) or any(value != first for value in identities.values()):
        raise ValueError("calibrated replication arms contain different tasks")

    treatment = rows["calibrated_path_ensemble"]
    incumbent = rows["frozen_incumbent"]
    improvements = [
        row
        for row, control in zip(treatment, incumbent, strict=True)
        if row.get("answer_exact") is True and control.get("answer_exact") is not True
    ]
    regressions = [
        row
        for row, control in zip(treatment, incumbent, strict=True)
        if control.get("answer_exact") is True and row.get("answer_exact") is not True
    ]
    paired = {
        control: paired_exact_test(treatment, rows[control])
        for control in (
            "forced_incumbent_selector_lesion",
            "necessary_condition_selector_lesion",
        )
    }
    causal_improvements = all(
        row.get("selected_path") == "challenger"
        and row.get("challenger_answer_exact") is True
        and row.get("incumbent_answer_exact") is not True
        for row in improvements
    )
    mechanism_pass = (
        not regressions
        and len(improvements) >= 5
        and causal_improvements
        and all(test["one_sided_exact_p"] < 0.05 for test in paired.values())
    )
    return {
        "paired_exact_tests": paired,
        "improvements_over_incumbent": len(improvements),
        "regressions_from_incumbent": len(regressions),
        "causal_improvement_contract_satisfied": causal_improvements,
        "mechanism_pass": mechanism_pass,
    }


def evaluate_calibrated_path_replication(
    *,
    bundle: LoadedSemanticFeatureBundle,
    training_manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    ensemble: SemanticProgramPathEnsemble,
    source_only_control: SemanticProgramPathEnsemble,
) -> dict[str, Any]:
    """Decode each path once, then derive every preregistered neural arm."""

    started = time.monotonic()
    preflight, examples = verify_calibrated_path_replication_preflight(
        bundle=bundle,
        training_manifest=training_manifest,
        preregistration=preregistration,
        ensemble=ensemble,
        source_only_control=source_only_control,
    )
    mixed_selector = _require_calibrated(ensemble)
    source_selector = _require_calibrated(source_only_control)
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in _ARM_NAMES}
    for item in examples:
        arbitrated = ensemble.decode_with_receipt(
            source_token_ids=item.ir.source_token_ids,
            hidden_states=item.hidden_states,
            public_inputs=item.public_inputs,
            source_text_sha256=item.ir.source_text_sha256,
            model_basis_sha256=item.ir.model_basis_receipt_sha256,
        )
        incumbent_values = semantic_path_selection_values(arbitrated.incumbent)
        challenger_values = semantic_path_selection_values(arbitrated.challenger)
        evidence = PairwiseSelectionEvidence.from_mappings(
            incumbent=incumbent_values,
            challenger=challenger_values,
            packet=arbitrated.decision.evidence,
        )
        necessary_decision = mixed_selector.necessary.select(
            incumbent="incumbent",
            challenger="challenger",
            evidence=evidence,
        )
        source_decision = source_selector.select(
            incumbent="incumbent",
            challenger="challenger",
            evidence=evidence,
        )
        decisions = {
            "calibrated_path_ensemble": arbitrated.decision,
            "necessary_condition_selector_lesion": necessary_decision,
            "source_only_calibration_control": source_decision,
        }
        outcomes = {
            "frozen_incumbent": arbitrated.incumbent,
            "frozen_challenger": arbitrated.challenger,
            "forced_incumbent_selector_lesion": arbitrated.incumbent,
            **{
                arm: (
                    arbitrated.challenger
                    if decision.selected == "challenger"
                    else arbitrated.incumbent
                )
                for arm, decision in decisions.items()
            },
        }
        incumbent_probe = _outcome_row(
            item,
            arbitrated.incumbent,
            arm="frozen_incumbent",
            selected_path="incumbent",
            selector_reason="standalone_path",
        )
        challenger_probe = _outcome_row(
            item,
            arbitrated.challenger,
            arm="frozen_challenger",
            selected_path="challenger",
            selector_reason="standalone_path",
        )
        for arm in _ARM_NAMES:
            if arm == "frozen_incumbent":
                row = incumbent_probe
            elif arm == "frozen_challenger":
                row = challenger_probe
            else:
                decision = decisions.get(arm)
                selected_path = decision.selected if decision is not None else "incumbent"
                reason = (
                    str(decision.receipt["reason"])
                    if decision is not None
                    else "selector_forced_to_incumbent"
                )
                row = _outcome_row(
                    item,
                    outcomes[arm],
                    arm=arm,
                    selected_path=selected_path,
                    selector_reason=reason,
                )
            row["incumbent_ir_available"] = arbitrated.incumbent.ir is not None
            row["challenger_ir_available"] = arbitrated.challenger.ir is not None
            row["incumbent_answer_exact"] = incumbent_probe["answer_exact"]
            row["challenger_answer_exact"] = challenger_probe["answer_exact"]
            rows[arm].append(row)

    adjudication = adjudicate_calibrated_path_replication(rows)
    mechanism_pass = adjudication["mechanism_pass"]
    body = {
        "schema": CALIBRATED_PATH_REPLICATION_RESULT_SCHEMA,
        "claim_boundary": preregistration["claim_boundary"],
        "preregistration_source_commit": preregistration["source_commit_before_preregistration"],
        "preflight": preflight,
        "mixed_ensemble_receipt_sha256": ensemble.receipt_sha256,
        "source_only_ensemble_receipt_sha256": source_only_control.receipt_sha256,
        "arms": {name: _arm(value) for name, value in rows.items()},
        **{key: value for key, value in adjudication.items() if key != "mechanism_pass"},
        "ordinary_resident_27b_decode": {
            "status": "DEFERRED_READY" if mechanism_pass else "ABORTED",
            "model_load_or_decode_calls": 0,
        },
        "verdict": (
            "PASS_MECHANISM_READY_FOR_ORDINARY_BASELINE"
            if mechanism_pass
            else "FAIL_PREREGISTERED_MECHANISM_BAR"
        ),
        "wall_time_s": time.monotonic() - started,
        "path_decode_calls": 2 * len(examples),
        "target_labels_available_to_paths_or_selectors": False,
        "serving_authority": False,
    }
    return {**body, "result_sha256": _sha(body)}


__all__ = [
    "CALIBRATED_PATH_REPLICATION_PREREGISTRATION_SCHEMA",
    "CALIBRATED_PATH_REPLICATION_RESULT_SCHEMA",
    "adjudicate_calibrated_path_replication",
    "evaluate_calibrated_path_replication",
    "verify_calibrated_path_replication_preflight",
]
