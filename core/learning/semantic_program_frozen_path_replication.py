"""Causal replication of one frozen semantic-program transducer."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from core.learning.semantic_program_basis import (
    bind_examples_to_compatible_training_session,
    establish_semantic_representation_compatibility,
)
from core.learning.semantic_program_campaign import training_examples_from_feature_bundle
from core.learning.semantic_program_compositional_transducer import (
    CompositionalSemanticProgramTransducer,
)
from core.learning.semantic_program_corpus import (
    build_semantic_program_natural_alias_source_corpus,
    build_semantic_program_natural_weave_replication_corpus,
)
from core.learning.semantic_program_evaluation import shuffle_hidden_tokens
from core.learning.semantic_program_execution import execute_semantic_program
from core.learning.semantic_program_feature_materialization import (
    NATURAL_WEAVE_REPLICATION_CORPUS_KIND,
    LoadedSemanticFeatureBundle,
)
from core.learning.semantic_program_path_ensemble_replication import paired_exact_test
from core.learning.semantic_program_transducer import (
    SemanticTransducerTrainingExample,
    SemanticTransductionOutcome,
)

FROZEN_PATH_PREREGISTRATION_SCHEMA: Final = (
    "aura.semantic_program_frozen_path_replication_preregistration.v1"
)
FROZEN_PATH_RESULT_SCHEMA: Final = "aura.semantic_program_frozen_path_replication.v1"
_ARMS: Final = ("frozen_transducer", "coefficient_lesion", "hidden_token_shuffle")
_SOURCE_FILES: Final = (
    "core/learning/semantic_program_basis.py",
    "core/learning/semantic_program_compositional_transducer.py",
    "core/learning/semantic_program_evaluation.py",
    "core/learning/semantic_program_execution.py",
    "core/learning/semantic_program_feature_materialization.py",
    "core/learning/semantic_program_frozen_path_replication.py",
    "core/learning/semantic_program_ir.py",
    "core/learning/semantic_program_path_ensemble.py",
    "core/learning/semantic_program_path_ensemble_replication.py",
    "tools/run_semantic_program_frozen_path_replication.py",
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


def _source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    files = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in _SOURCE_FILES
    }
    body = {
        "schema": "aura.semantic_program_frozen_path_source_identity.v1",
        "files": files,
    }
    return {**body, "source_sha256": _sha(body)}


def _outcome_row(
    item: SemanticTransducerTrainingExample,
    outcome: SemanticTransductionOutcome,
    *,
    arm: str,
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
        "refusal": outcome.refusal,
        "accepted": predicted is not None,
        "program_exact": bool(predicted is not None and predicted.to_program() == expected_program),
        "answer_emitted": predicted_answer is not None,
        "answer_exact": bool(predicted_answer is not None and predicted_answer == expected_answer),
        "transducer_receipt_sha256": (
            predicted.transducer_receipt_sha256 if predicted is not None else None
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


def verify_frozen_path_replication_preflight(
    *,
    bundle: LoadedSemanticFeatureBundle,
    training_manifest: Mapping[str, Any],
    discovery_manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    model: CompositionalSemanticProgramTransducer,
) -> tuple[dict[str, Any], tuple[SemanticTransducerTrainingExample, ...]]:
    """Bind one frozen transducer to a fresh, source-disjoint feature cohort."""

    frozen = preregistration.get("frozen_transducer")
    training = preregistration.get("training_evidence")
    corpus_contract = preregistration.get("replication_corpus")
    config = bundle.manifest.get("config")
    if not all(
        isinstance(value, Mapping) for value in (frozen, training, corpus_contract, config)
    ):
        raise ValueError("frozen path replication contract is incomplete")
    assert isinstance(frozen, Mapping)
    assert isinstance(training, Mapping)
    assert isinstance(corpus_contract, Mapping)
    assert isinstance(config, Mapping)
    if (
        preregistration.get("schema") != FROZEN_PATH_PREREGISTRATION_SCHEMA
        or preregistration.get("ordinary_decode_deferred_until_mechanism_pass") is not True
        or preregistration.get("target_labels_available_to_transducer_or_controls") is not False
        or preregistration.get("serving_authority") is not False
        or preregistration.get("arms") != [*_ARMS, "ordinary_resident_27b_decode"]
        or frozen.get("path") != "challenger"
        or frozen.get("schema") != model.schema
        or frozen.get("receipt_sha256") != model.receipt_sha256
        or frozen.get("coefficient_sha256")
        != model.training_receipt.get("coefficient_sha256")
        or frozen.get("model_basis_sha256") != model.model_basis_sha256
        or frozen.get("coefficients_or_hyperparameters_may_change") is not False
        or training.get("manifest_sha256") != training_manifest.get("manifest_sha256")
        or training.get("corpus_sha256") != training_manifest.get("corpus_sha256")
        or config.get("corpus_kind") != NATURAL_WEAVE_REPLICATION_CORPUS_KIND
        or config.get("seed") != corpus_contract.get("seed")
        or config.get("examples_per_operation_pair")
        != corpus_contract.get("examples_per_schema_domain")
        or config.get("max_examples") != corpus_contract.get("max_examples")
        or bundle.manifest.get("complete") is not True
        or bundle.manifest.get("manifest_sha256")
        == discovery_manifest.get("manifest_sha256")
        or bundle.manifest.get("corpus_sha256") == discovery_manifest.get("corpus_sha256")
    ):
        raise ValueError("frozen path replication contract differs")

    target = build_semantic_program_natural_weave_replication_corpus(
        seed=int(corpus_contract["seed"]),
        examples_per_schema_domain=int(corpus_contract["examples_per_schema_domain"]),
    )
    discovery = build_semantic_program_natural_weave_replication_corpus(
        seed=3141592653,
        examples_per_schema_domain=int(corpus_contract["examples_per_schema_domain"]),
    )
    source = build_semantic_program_natural_alias_source_corpus(
        seed=int(training_manifest["config"]["seed"]),
        examples_per_schema_domain=int(
            training_manifest["config"]["examples_per_operation_pair"]
        ),
    )
    raw = training_examples_from_feature_bundle(
        bundle,
        required_splits=frozenset({"validation", "test"}),
    )
    compatibility = establish_semantic_representation_compatibility(
        model=model,
        training_manifest=training_manifest,
        replication_manifest=bundle.manifest,
    )
    examples = bind_examples_to_compatible_training_session(raw, compatibility=compatibility)
    target_by_sha = {
        hashlib.sha256(item.source_text.encode("utf-8")).hexdigest(): item for item in target
    }
    observed_by_sha = {item.ir.source_text_sha256: item for item in examples}
    target_texts = {item.source_text for item in target}
    if (
        len(target) != 48
        or len(examples) != len(target)
        or len(observed_by_sha) != len(examples)
        or observed_by_sha.keys() != target_by_sha.keys()
        or target_texts & {item.source_text for item in discovery}
        or target_texts & {item.source_text for item in source}
        or any(
            item.public_inputs != target_by_sha[source_sha].inputs
            or item.ir.to_program() != target_by_sha[source_sha].program
            or item.ir.model_basis_receipt_sha256 != model.model_basis_sha256
            for source_sha, item in observed_by_sha.items()
        )
    ):
        raise ValueError("frozen path target identity, program, or overlap differs")
    preflight = {
        "example_count": len(examples),
        "feature_manifest_sha256": bundle.manifest["manifest_sha256"],
        "corpus_sha256": bundle.manifest["corpus_sha256"],
        "discovery_feature_manifest_sha256": discovery_manifest["manifest_sha256"],
        "training_feature_manifest_sha256": training_manifest["manifest_sha256"],
        "transducer_receipt_sha256": model.receipt_sha256,
        "coefficient_sha256": model.training_receipt["coefficient_sha256"],
        "representation_compatibility": compatibility,
        "training_source_text_overlap": 0,
        "discovery_source_text_overlap": 0,
        "target_labels_available_to_transducer_or_controls": False,
        "source_text_or_domain_identity_available_to_transducer": False,
    }
    return preflight, examples


def adjudicate_frozen_path_replication(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    treatment_receipt_sha256: str,
    minimum_exact: int = 18,
) -> dict[str, Any]:
    """Apply the frozen causal bar to task-matched neural arms."""

    if set(rows) != set(_ARMS) or any(len(rows[arm]) != 48 for arm in _ARMS):
        raise ValueError("frozen path replication arms are incomplete")
    identities = {
        arm: tuple(str(row.get("source_text_sha256")) for row in rows[arm]) for arm in _ARMS
    }
    first = identities[_ARMS[0]]
    if len(set(first)) != len(first) or any(value != first for value in identities.values()):
        raise ValueError("frozen path replication arms contain different tasks")
    treatment = rows["frozen_transducer"]
    paired = {
        control: paired_exact_test(treatment, rows[control])
        for control in ("coefficient_lesion", "hidden_token_shuffle")
    }
    causal_wins = all(
        row.get("transducer_receipt_sha256") == treatment_receipt_sha256
        for control in ("coefficient_lesion", "hidden_token_shuffle")
        for row, control_row in zip(treatment, rows[control], strict=True)
        if row.get("answer_exact") is True and control_row.get("answer_exact") is not True
    )
    treatment_exact = sum(row.get("answer_exact") is True for row in treatment)
    mechanism_pass = bool(
        treatment_exact >= minimum_exact
        and causal_wins
        and all(
            test["treatment_only"] > test["control_only"]
            and test["one_sided_exact_p"] < 0.05
            for test in paired.values()
        )
    )
    return {
        "treatment_answer_exact": treatment_exact,
        "minimum_exact": minimum_exact,
        "paired_exact_tests": paired,
        "causal_treatment_receipt_contract_satisfied": causal_wins,
        "mechanism_pass": mechanism_pass,
    }


def evaluate_frozen_path_replication(
    *,
    bundle: LoadedSemanticFeatureBundle,
    training_manifest: Mapping[str, Any],
    discovery_manifest: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    model: CompositionalSemanticProgramTransducer,
    progress: Callable[[int, int, Mapping[str, int]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen path and matched causal lesions without refitting."""

    started = time.monotonic()
    preflight, examples = verify_frozen_path_replication_preflight(
        bundle=bundle,
        training_manifest=training_manifest,
        discovery_manifest=discovery_manifest,
        preregistration=preregistration,
        model=model,
    )
    lesion = model.coefficient_lesion()
    rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in _ARMS}
    for index, item in enumerate(examples, 1):
        variants = {
            "frozen_transducer": (model, item.hidden_states),
            "coefficient_lesion": (lesion, item.hidden_states),
            "hidden_token_shuffle": (
                model,
                shuffle_hidden_tokens(item.hidden_states, item.ir.source_text_sha256),
            ),
        }
        for arm, (candidate, hidden) in variants.items():
            outcome = candidate.decode(
                source_token_ids=item.ir.source_token_ids,
                hidden_states=hidden,
                public_inputs=item.public_inputs,
                source_text_sha256=item.ir.source_text_sha256,
                model_basis_sha256=item.ir.model_basis_receipt_sha256,
            )
            rows[arm].append(_outcome_row(item, outcome, arm=arm))
        if progress is not None:
            progress(
                index,
                len(examples),
                {
                    arm: sum(row["answer_exact"] is True for row in rows[arm])
                    for arm in _ARMS
                },
            )
    adjudication = adjudicate_frozen_path_replication(
        rows,
        treatment_receipt_sha256=model.receipt_sha256,
    )
    mechanism_pass = adjudication["mechanism_pass"]
    body = {
        "schema": FROZEN_PATH_RESULT_SCHEMA,
        "claim_boundary": preregistration["claim_boundary"],
        "preregistration_source_commit": preregistration["source_commit_before_preregistration"],
        "preflight": preflight,
        "source_identity": _source_identity(),
        "arms": {arm: _arm(rows[arm]) for arm in _ARMS},
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
        "decode_calls": len(examples) * len(_ARMS),
        "wall_time_s": time.monotonic() - started,
        "serving_authority": False,
    }
    return {**body, "result_sha256": _sha(body)}


__all__ = [
    "FROZEN_PATH_PREREGISTRATION_SCHEMA",
    "FROZEN_PATH_RESULT_SCHEMA",
    "adjudicate_frozen_path_replication",
    "evaluate_frozen_path_replication",
    "verify_frozen_path_replication_preflight",
]
