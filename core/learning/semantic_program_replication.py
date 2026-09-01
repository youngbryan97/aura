"""Frozen fresh-cohort replication for the learned semantic transducer.

The evaluator loads a separately materialized synthetic cohort, reconstructs
that cohort from its committed config, and evaluates one supplied transducer.
It never fits, refits, or updates coefficients. Matched controls either move
the observed token states or remove dependence on them.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from core.learning.semantic_program_basis import (
    bind_examples_to_compatible_training_session,
    establish_semantic_representation_compatibility,
)
from core.learning.semantic_program_campaign import (
    training_examples_from_feature_bundle,
)
from core.learning.semantic_program_evaluation import (
    coefficient_lesion,
    evaluate_semantic_program_transducer,
    shuffle_hidden_tokens,
)
from core.learning.semantic_program_feature_materialization import (
    LoadedSemanticFeatureBundle,
    load_standard_semantic_feature_bundle,
)
from core.learning.semantic_program_transducer import (
    semantic_program_transducer_from_dict,
)
from core.runtime.file_read_gateway import read_stable_bytes

SEMANTIC_PROGRAM_REPLICATION_SCHEMA: Final = (
    "aura.semantic_program_fresh_cohort_replication.v1"
)
SEMANTIC_PROGRAM_REPLICATION_SOURCES: Final = (
    "core/brain/llm/latent_cortex/runtime_identity.py",
    "core/learning/semantic_program_basis.py",
    "core/learning/semantic_program_campaign.py",
    "core/learning/semantic_program_corpus.py",
    "core/learning/semantic_program_evaluation.py",
    "core/learning/semantic_program_execution.py",
    "core/learning/semantic_program_feature_materialization.py",
    "core/learning/semantic_program_ir.py",
    "core/learning/semantic_program_replication.py",
    "core/learning/semantic_program_transducer.py",
    "tools/run_semantic_program_replication.py",
)
_SPLITS: Final = ("train", "validation", "test")
_CONTROLS: Final = ("hidden_token_shuffle", "coefficient_lesion")
_METRICS: Final = ("program_exact", "answer_exact")


class SemanticProgramReplicationError(RuntimeError):
    """The frozen replication contract could not be established."""


@dataclass(frozen=True, slots=True)
class FrozenTrainingCohort:
    """Caller-supplied identity of the cohort used to train the frozen model."""

    feature_manifest_sha256: str
    example_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not _is_sha256(self.feature_manifest_sha256)
            or not self.example_ids
            or len(set(self.example_ids)) != len(self.example_ids)
            or any(not isinstance(value, str) or not value for value in self.example_ids)
        ):
            raise ValueError("semantic replication training cohort identity is invalid")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise SemanticProgramReplicationError(
            "semantic replication evidence is not canonical JSON"
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_reconstructed_semantic_cohort(
    bundle_directory: Path,
) -> LoadedSemanticFeatureBundle:
    """Load a strict bundle after the canonical seeded-corpus reconstruction."""

    return load_standard_semantic_feature_bundle(bundle_directory)


def semantic_replication_source_sha256s(repo_root: Path) -> dict[str, str]:
    """Hash every implementation source that can change the replication result."""

    root = repo_root.expanduser().resolve(strict=True)
    result: dict[str, str] = {}
    for relative in SEMANTIC_PROGRAM_REPLICATION_SOURCES:
        payload = read_stable_bytes(root / relative, max_bytes=4 * 1024 * 1024)
        result[relative] = hashlib.sha256(payload).hexdigest()
    return result


def _exact_one_sided_pair(
    treatment_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    if metric not in _METRICS:
        raise ValueError("semantic replication paired metric is unsupported")
    treatment = {
        str(row["source_text_sha256"]): row.get(metric) is True
        for row in treatment_rows
    }
    control = {
        str(row["source_text_sha256"]): row.get(metric) is True
        for row in control_rows
    }
    if treatment.keys() != control.keys() or len(treatment) != len(treatment_rows):
        raise SemanticProgramReplicationError(
            "semantic replication paired arms contain different tasks"
        )
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
    common = math.gcd(numerator, denominator)
    numerator //= common
    denominator //= common
    return {
        "metric": metric,
        "treatment_only": treatment_only,
        "control_only": control_only,
        "discordant": discordant,
        "one_sided_exact_p_numerator": numerator,
        "one_sided_exact_p_denominator": denominator,
        "one_sided_exact_p": numerator / denominator,
    }


def _pooled_rows(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    arm: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in _SPLITS:
        value = arms[f"{arm}:{split}"].get("rows")
        if not isinstance(value, list):
            raise SemanticProgramReplicationError(
                "semantic replication arm rows are malformed"
            )
        rows.extend(value)
    identities = [str(row.get("source_text_sha256")) for row in rows]
    if len(set(identities)) != len(identities):
        raise SemanticProgramReplicationError(
            "semantic replication source identities are not unique"
        )
    return rows


def _assert_fresh_cohort(
    bundle: LoadedSemanticFeatureBundle,
    training_cohort: FrozenTrainingCohort,
) -> tuple[str, ...]:
    manifest_sha256 = bundle.manifest.get("manifest_sha256")
    if not _is_sha256(manifest_sha256):
        raise SemanticProgramReplicationError(
            "semantic replication feature manifest identity is invalid"
        )
    if manifest_sha256 == training_cohort.feature_manifest_sha256:
        raise SemanticProgramReplicationError(
            "semantic replication reused the training feature manifest"
        )
    replication_ids = tuple(
        str(example.metadata.get("example_id") or "") for example in bundle.examples
    )
    if (
        not replication_ids
        or any(not value for value in replication_ids)
        or set(replication_ids) & set(training_cohort.example_ids)
    ):
        raise SemanticProgramReplicationError(
            "semantic replication cohort overlaps model training examples"
        )
    return replication_ids


def evaluate_frozen_semantic_replication(
    bundle: LoadedSemanticFeatureBundle,
    *,
    trained_model_payload: Any,
    training_cohort: FrozenTrainingCohort,
    training_manifest: Mapping[str, Any],
    source_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    """Evaluate one already-trained transducer without any optimization call."""

    if (
        not isinstance(source_sha256s, Mapping)
        or set(source_sha256s) != set(SEMANTIC_PROGRAM_REPLICATION_SOURCES)
        or any(not _is_sha256(value) for value in source_sha256s.values())
    ):
        raise SemanticProgramReplicationError(
            "semantic replication source identity is invalid"
        )
    replication_ids = _assert_fresh_cohort(bundle, training_cohort)
    model = semantic_program_transducer_from_dict(trained_model_payload)
    model_payload_before = model.to_dict()
    model_sha256 = _sha(model_payload_before)
    if training_manifest.get("manifest_sha256") != (
        training_cohort.feature_manifest_sha256
    ):
        raise SemanticProgramReplicationError(
            "semantic replication training manifest identity differs"
        )
    compatibility = establish_semantic_representation_compatibility(
        model=model,
        training_manifest=training_manifest,
        replication_manifest=bundle.manifest,
    )
    examples = bind_examples_to_compatible_training_session(
        training_examples_from_feature_bundle(bundle),
        compatibility=compatibility,
    )
    lesion = coefficient_lesion(model)
    arms: dict[str, dict[str, Any]] = {}
    for split in _SPLITS:
        arms[f"treatment:{split}"] = evaluate_semantic_program_transducer(
            model,
            examples,
            split=split,
            arm="treatment",
        ).to_dict()
        arms[f"hidden_token_shuffle:{split}"] = (
            evaluate_semantic_program_transducer(
                model,
                examples,
                split=split,
                arm="hidden_token_shuffle",
                hidden_transform=shuffle_hidden_tokens,
            ).to_dict()
        )
        arms[f"coefficient_lesion:{split}"] = evaluate_semantic_program_transducer(
            lesion,
            examples,
            split=split,
            arm="coefficient_lesion",
        ).to_dict()

    paired: dict[str, dict[str, Any]] = {}
    for control in _CONTROLS:
        for split in (*_SPLITS, "pooled"):
            treatment_rows = (
                _pooled_rows(arms, arm="treatment")
                if split == "pooled"
                else arms[f"treatment:{split}"]["rows"]
            )
            control_rows = (
                _pooled_rows(arms, arm=control)
                if split == "pooled"
                else arms[f"{control}:{split}"]["rows"]
            )
            for metric in _METRICS:
                paired[f"{control}:{split}:{metric}"] = _exact_one_sided_pair(
                    treatment_rows,
                    control_rows,
                    metric=metric,
                )

    model_payload_after = model.to_dict()
    if model_payload_after != model_payload_before:
        raise SemanticProgramReplicationError(
            "semantic replication mutated the supplied transducer"
        )
    body = {
        "schema": SEMANTIC_PROGRAM_REPLICATION_SCHEMA,
        "fresh_cohort": True,
        "training_feature_manifest_sha256": training_cohort.feature_manifest_sha256,
        "replication_feature_manifest_sha256": bundle.manifest["manifest_sha256"],
        "training_example_count_declared": len(training_cohort.example_ids),
        "replication_example_count": len(replication_ids),
        "training_replication_example_overlap": 0,
        "replication_config_sha256": bundle.manifest["config_sha256"],
        "model_basis_sha256": model.model_basis_sha256,
        "transducer_receipt_sha256": model.receipt_sha256,
        "trained_model_sha256": model_sha256,
        "trained_model_unchanged": True,
        "representation_compatibility": compatibility,
        "fitting_calls": 0,
        "refitting_calls": 0,
        "arms": arms,
        "paired_exact_tests": paired,
        "source_sha256s": dict(sorted(source_sha256s.items())),
        "expected_answers_available_to_training": False,
        "expected_answers_available_to_evaluation": True,
        "verifier_traces_available": False,
        "generated_compiler_text_available": False,
        "serving_authority": False,
        "claim_boundary": (
            "bounded fresh synthetic semantic-program cohort on a function-identical "
            "frozen transducer across worker sessions; no broad-domain claim"
        ),
    }
    return {**body, "report_sha256": _sha(body)}


def run_frozen_semantic_replication(
    bundle_directory: Path,
    *,
    trained_model_payload: Any,
    training_cohort: FrozenTrainingCohort,
    training_manifest: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Load verified inputs and run the CPU-only frozen replication."""

    bundle = load_reconstructed_semantic_cohort(bundle_directory)
    sources = semantic_replication_source_sha256s(repo_root)
    return evaluate_frozen_semantic_replication(
        bundle,
        trained_model_payload=trained_model_payload,
        training_cohort=training_cohort,
        training_manifest=training_manifest,
        source_sha256s=sources,
    )


__all__ = [
    "SEMANTIC_PROGRAM_REPLICATION_SCHEMA",
    "SEMANTIC_PROGRAM_REPLICATION_SOURCES",
    "FrozenTrainingCohort",
    "SemanticProgramReplicationError",
    "evaluate_frozen_semantic_replication",
    "load_reconstructed_semantic_cohort",
    "run_frozen_semantic_replication",
    "semantic_replication_source_sha256s",
]
