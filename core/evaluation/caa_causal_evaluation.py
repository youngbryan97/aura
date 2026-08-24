"""Independent replay and adjudication for model-bound CAA generations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from core.evaluation.steering_ab import (
    REQUIRED_CONDITIONS,
    SPECIFICITY_CONTROLS,
    affect_target_score,
    analyze_steering_ab,
)
from core.learning.cortex_migration_authority import CAA_EVALUATION_SCHEMA

VERIFIER_SCHEMA: Final = "aura.caa.independent_replay.v1"


class CAACausalEvaluationError(ValueError):
    """The campaign cannot support a causal steering claim."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CAACausalEvaluationError("caa_evidence_not_canonical") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_vector_generation(
    metadata: Mapping[str, Any],
    *,
    generation_dir: Path,
    model_descriptor_sha256: str,
) -> dict[str, Any]:
    """Reopen every vector and prove one exact extraction generation."""

    identity = metadata.get("model_identity")
    extraction = metadata.get("extraction_contract")
    vectors = metadata.get("vector_files")
    if (
        not isinstance(identity, Mapping)
        or identity.get("model_descriptor_sha256") != model_descriptor_sha256
        or not isinstance(extraction, Mapping)
        or not isinstance(vectors, Sequence)
        or isinstance(vectors, (str, bytes))
        or not vectors
    ):
        raise CAACausalEvaluationError("caa_vector_generation_invalid")

    normalized_vectors: list[dict[str, Any]] = []
    root = generation_dir.expanduser().resolve(strict=True)
    for raw in vectors:
        if not isinstance(raw, Mapping) or set(raw) != {
            "name",
            "size_bytes",
            "sha256",
        }:
            raise CAACausalEvaluationError("caa_vector_manifest_invalid")
        name = raw.get("name")
        size = raw.get("size_bytes")
        digest = raw.get("sha256")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or type(size) is not int
            or size <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise CAACausalEvaluationError("caa_vector_manifest_invalid")
        path = (root / name).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise CAACausalEvaluationError("caa_vector_path_escape") from exc
        if path.stat().st_size != size or file_sha256(path) != digest:
            raise CAACausalEvaluationError("caa_vector_generation_drift")
        normalized_vectors.append(
            {"name": name, "size_bytes": size, "sha256": digest}
        )

    generation_sha256 = canonical_sha256(
        {
            "extraction_contract_sha256": extraction.get(
                "extraction_contract_sha256"
            ),
            "vector_files": normalized_vectors,
        }
    )
    if metadata.get("generation_sha256") != generation_sha256:
        raise CAACausalEvaluationError("caa_generation_digest_invalid")
    return {
        "model_descriptor_sha256": model_descriptor_sha256,
        "generation_sha256": generation_sha256,
        "vector_count": len(normalized_vectors),
    }


def _condition_outputs(result: Mapping[str, Any]) -> dict[str, list[str]]:
    raw = result.get("condition_outputs")
    required = {*REQUIRED_CONDITIONS, *SPECIFICITY_CONTROLS}
    if not isinstance(raw, Mapping) or not required.issubset(raw):
        raise CAACausalEvaluationError("caa_replay_outputs_incomplete")
    outputs: dict[str, list[str]] = {}
    sample_count: int | None = None
    for name in sorted(required):
        values = raw.get(name)
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
        ):
            raise CAACausalEvaluationError("caa_replay_outputs_invalid")
        normalized = [str(value) for value in values]
        if sample_count is None:
            sample_count = len(normalized)
        if len(normalized) != sample_count or any(not value.strip() for value in normalized):
            raise CAACausalEvaluationError("caa_replay_outputs_invalid")
        outputs[name] = normalized
    if sample_count is None or sample_count < 24:
        raise CAACausalEvaluationError("caa_replay_sample_too_small")
    return outputs


def replay_campaign(result: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute all target scores, effects, controls, and paired outcomes."""

    outputs = _condition_outputs(result)
    scores = {
        name: [affect_target_score(text) for text in values]
        for name, values in outputs.items()
    }
    recorded_scores = result.get("target_scores")
    if not isinstance(recorded_scores, Mapping):
        raise CAACausalEvaluationError("caa_recorded_scores_missing")
    for name, expected in scores.items():
        if list(recorded_scores.get(name) or ()) != expected:
            raise CAACausalEvaluationError("caa_recorded_scores_mismatch")

    report = analyze_steering_ab(
        outputs,
        target_scores=scores,
        n_resamples=5000,
        seed=42,
    )
    baseline = scores["baseline"]
    treatment = scores["steered_black_box"]
    matched_control = scores["baseline_replicate"]

    def wins(values: Sequence[float]) -> int:
        return sum(
            1
            for candidate, reference in zip(values, baseline, strict=True)
            if candidate > reference
        )

    treatment_successes = wins(treatment)
    matched_control_successes = wins(matched_control)
    lesion_successes = {
        name: wins(scores[name]) for name in sorted(SPECIFICITY_CONTROLS)
    }

    tasks = result.get("held_out_tasks")
    trials_per_task = result.get("n_trials_per_task")
    if (
        not isinstance(tasks, Sequence)
        or isinstance(tasks, (str, bytes))
        or not tasks
        or type(trials_per_task) is not int
        or trials_per_task <= 0
        or len(tasks) * trials_per_task != len(baseline)
    ):
        raise CAACausalEvaluationError("caa_trial_layout_invalid")
    no_regression = True
    task_deltas: dict[str, float] = {}
    for index, task in enumerate(tasks):
        start = index * trials_per_task
        stop = start + trials_per_task
        delta = sum(treatment[start:stop]) - sum(baseline[start:stop])
        task_deltas[str(task)] = float(delta / trials_per_task)
        if delta < 0:
            no_regression = False

    causal_effect_positive = bool(
        report.passes_adversarial_control
        and treatment_successes > matched_control_successes
        and all(value < treatment_successes for value in lesion_successes.values())
    )
    return {
        "sample_count": len(baseline),
        "treatment_successes": treatment_successes,
        "matched_control_successes": matched_control_successes,
        "lesion_successes": lesion_successes,
        "no_regression": no_regression,
        "causal_effect_positive": causal_effect_positive,
        "passes_adversarial_control": report.passes_adversarial_control,
        "unmet_requirements": list(report.unmet_requirements()),
        "task_target_deltas": task_deltas,
        "analysis": report.to_dict(),
    }


def build_independent_verifier_evidence(
    *,
    result: Mapping[str, Any],
    result_sha256: str,
    metadata: Mapping[str, Any],
    metadata_sha256: str,
    generation_dir: Path,
) -> dict[str, Any]:
    model_descriptor_sha256 = str(result.get("model_descriptor_sha256") or "")
    generation = validate_vector_generation(
        metadata,
        generation_dir=generation_dir,
        model_descriptor_sha256=model_descriptor_sha256,
    )
    replay = replay_campaign(result)
    body = {
        "schema": VERIFIER_SCHEMA,
        "verifier": {"name": "aura-caa-independent-replay", "version": "1"},
        "result_sha256": result_sha256,
        "metadata_sha256": metadata_sha256,
        **generation,
        "replay": replay,
        "verified": bool(
            replay["passes_adversarial_control"]
            and replay["causal_effect_positive"]
            and replay["no_regression"]
        ),
    }
    return {**body, "verification_sha256": canonical_sha256(body)}


def build_causal_evaluation(
    *,
    result: Mapping[str, Any],
    metadata: Mapping[str, Any],
    verifier_evidence: Mapping[str, Any],
    verifier_evidence_sha256: str,
) -> dict[str, Any]:
    """Issue the strict authority input from independently replayed evidence."""

    if (
        verifier_evidence.get("schema") != VERIFIER_SCHEMA
        or verifier_evidence.get("verified") is not True
        or verifier_evidence.get("verification_sha256")
        != canonical_sha256(
            {
                key: value
                for key, value in verifier_evidence.items()
                if key != "verification_sha256"
            }
        )
        or verifier_evidence.get("model_descriptor_sha256")
        != result.get("model_descriptor_sha256")
        or verifier_evidence.get("generation_sha256")
        != metadata.get("generation_sha256")
    ):
        raise CAACausalEvaluationError("caa_independent_verifier_invalid")
    replay = verifier_evidence.get("replay")
    verifier = verifier_evidence.get("verifier")
    if not isinstance(replay, Mapping) or not isinstance(verifier, Mapping):
        raise CAACausalEvaluationError("caa_independent_verifier_invalid")
    body = {
        "schema": CAA_EVALUATION_SCHEMA,
        "model_descriptor_sha256": result["model_descriptor_sha256"],
        "generation_sha256": metadata["generation_sha256"],
        "verdict": "PASS",
        "qualified": True,
        "sample_count": replay["sample_count"],
        "treatment_successes": replay["treatment_successes"],
        "matched_control_successes": replay["matched_control_successes"],
        "lesion_successes": dict(replay["lesion_successes"]),
        "no_regression": replay["no_regression"],
        "causal_effect_positive": replay["causal_effect_positive"],
        "independent_verifier": {
            "name": verifier["name"],
            "version": verifier["version"],
            "evidence_sha256": verifier_evidence_sha256,
        },
    }
    if (
        body["no_regression"] is not True
        or body["causal_effect_positive"] is not True
    ):
        raise CAACausalEvaluationError("caa_campaign_not_qualified")
    return {**body, "evaluation_sha256": canonical_sha256(body)}


__all__ = [
    "CAACausalEvaluationError",
    "VERIFIER_SCHEMA",
    "build_causal_evaluation",
    "build_independent_verifier_evidence",
    "canonical_sha256",
    "file_sha256",
    "replay_campaign",
    "validate_vector_generation",
]
