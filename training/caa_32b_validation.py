#!/usr/bin/env python3
"""Production-scale CAA validation harness for Aura's active cortex.

This script validates the artifacts needed for a credible CAA claim:
activation-derived vectors, layer geometry, PCA structure, permutation
controls, rich-prompt comparator slots, black-box prompt-hygiene conditions,
and behavioral A/B result ingestion.  It runs quickly when evaluating existing
artifacts and can be paired with `extract_steering_vectors.py` for full-model
extraction runs. Model authority comes from an exact artifact descriptor, not
from a parameter-count label or a shared configuration file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_VECTOR_LOAD_RECOVERABLE_ERRORS = (
    OSError,
    EOFError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True)
class LoadedVector:
    dimension: str
    layer: int
    path: str
    vector: np.ndarray
    source: str = "unknown"
    extracted: bool = False
    model_path: str | None = None
    model_config_sha256: str | None = None
    model_descriptor_sha256: str | None = None


def _sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _model_config_sha256(model_path: str) -> str | None:
    cfg = Path(model_path).expanduser() / "config.json"
    if not cfg.exists():
        return None
    return _sha256_file(cfg)


class CAAModelValidator:
    def __init__(
        self,
        vectors_dir: str | Path = "training/vectors",
        model_path: str | Path | None = None,
        model_identity: dict[str, object] | None = None,
    ) -> None:
        self.vectors_dir = Path(vectors_dir)
        if model_path is None:
            try:
                from core.brain.llm.model_registry import get_active_cortex_spec

                spec = get_active_cortex_spec()
            except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                spec = None
            if spec is not None:
                model_path = spec.model_path
                model_identity = model_identity or spec.artifact_descriptor()
        if model_path is None and isinstance(model_identity, dict):
            model_path = str(model_identity.get("canonical_path") or "") or None
        self.model_path = str(model_path or "")
        if model_identity is None and self.model_path:
            try:
                from core.brain.llm.model_registry import (
                    get_active_model_artifact_descriptor,
                )

                model_identity = get_active_model_artifact_descriptor(self.model_path)
            except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                model_identity = None
        self.model_identity: dict[str, object] = {}
        self.model_identity_error = "exact_model_identity_unavailable"
        if isinstance(model_identity, dict) and self.model_path:
            try:
                from core.brain.llm.model_artifact_profile import (
                    validate_model_artifact_descriptor,
                )

                validated = validate_model_artifact_descriptor(
                    model_identity,
                    model_path=self.model_path,
                    verify_full_hash=False,
                )
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                self.model_identity_error = f"{type(exc).__name__}: {exc}"
            else:
                self.model_identity = dict(validated)
                self.model_identity_error = ""
        digest = str(self.model_identity.get("descriptor_sha256") or "")
        self.model_descriptor_sha256 = (
            digest if re.fullmatch(r"[0-9a-f]{64}", digest) else ""
        )
        self._vector_load_errors: list[dict[str, str]] = []

    def run(self, behavioral_results: str | Path | None = None) -> dict[str, Any]:
        self._vector_load_errors = []
        vectors = self._load_vectors()
        activation_vectors = [v for v in vectors if v.layer >= 0]
        fallback_vectors = [v for v in vectors if v.layer < 0]
        observed_model_config_sha256 = _model_config_sha256(self.model_path)
        production_activation_vectors = self._active_model_bound_vectors(
            activation_vectors,
            model_descriptor_sha256=self.model_descriptor_sha256,
        )
        production_vector_ids = {id(vector) for vector in production_activation_vectors}
        stale_activation_vectors = [
            vector for vector in activation_vectors if id(vector) not in production_vector_ids
        ]
        geometry = self._geometry(production_activation_vectors)
        observed_geometry = self._geometry(activation_vectors)
        behavioral = self._load_behavioral_results(
            behavioral_results,
            model_descriptor_sha256=self.model_descriptor_sha256,
        )
        prompt_controls = self._prompt_control_schema()
        pass_conditions = self._pass_conditions(
            production_activation_vectors,
            geometry,
            behavioral,
            model_descriptor_sha256=self.model_descriptor_sha256,
            total_activation_vectors=len(activation_vectors),
            stale_activation_vectors=len(stale_activation_vectors),
        )
        return {
            "generated_at": time.time(),
            "model_path": self.model_path,
            "exact_model_detected": bool(self.model_descriptor_sha256),
            "model_descriptor_sha256": self.model_descriptor_sha256,
            "model_identity_error": self.model_identity_error,
            "vectors_dir": str(self.vectors_dir),
            "vector_count": len(vectors),
            "activation_vector_count": len(activation_vectors),
            "production_activation_vector_count": len(production_activation_vectors),
            "stale_or_unbound_activation_vector_count": len(stale_activation_vectors),
            "fallback_prior_count": len(fallback_vectors),
            "observed_model_config_sha256": observed_model_config_sha256,
            "dimensions": sorted({v.dimension for v in vectors}),
            "layers": sorted({v.layer for v in vectors if v.layer >= 0}),
            "geometry": geometry,
            "observed_unbound_geometry": observed_geometry,
            "behavioral_ab": behavioral,
            "prompt_controls": prompt_controls,
            "pass_conditions": pass_conditions,
            "vector_load_errors": self._vector_load_errors,
            "passed": all(item["passed"] for item in pass_conditions.values()),
        }

    @staticmethod
    def _active_model_bound_vectors(
        vectors: list[LoadedVector],
        *,
        model_descriptor_sha256: str,
    ) -> list[LoadedVector]:
        if not model_descriptor_sha256:
            return []
        return [
            vector
            for vector in vectors
            if (
                vector.extracted
                and vector.model_descriptor_sha256 == model_descriptor_sha256
            )
        ]

    def _load_vectors(self) -> list[LoadedVector]:
        vectors: list[LoadedVector] = []
        if not self.vectors_dir.exists():
            return vectors
        for path in sorted([*self.vectors_dir.glob("*.npy"), *self.vectors_dir.glob("*.npz")]):
            try:
                arr, metadata = self._read_array(path)
            except _VECTOR_LOAD_RECOVERABLE_ERRORS as exc:
                self._vector_load_errors.append(
                    {
                        "path": str(path),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue
            if arr is None or arr.size == 0:
                continue
            dimension, layer = self._parse_name(path.stem)
            vec = np.asarray(arr, dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(
                LoadedVector(
                    dimension=dimension,
                    layer=layer,
                    path=str(path),
                    vector=vec,
                    source=str(metadata.get("source") or "unknown"),
                    extracted=bool(metadata.get("extracted", False)),
                    model_path=metadata.get("model_path"),
                    model_config_sha256=metadata.get("model_config_sha256"),
                    model_descriptor_sha256=metadata.get("model_descriptor_sha256"),
                )
            )
        return vectors

    @staticmethod
    def _read_array(path: Path) -> tuple[np.ndarray | None, dict[str, Any]]:
        if path.suffix == ".npy":
            return np.load(path), {}
        data = np.load(path)
        metadata: dict[str, Any] = {}
        for key in (
            "source",
            "extracted",
            "model_path",
            "model",
            "model_config_sha256",
            "model_descriptor_sha256",
        ):
            if key not in data:
                continue
            value = data[key]
            if key == "model" and "model_path" not in metadata:
                metadata["model_path"] = str(value)
            elif key == "extracted":
                metadata[key] = bool(value)
            else:
                metadata[key] = str(value)
        for key in ("vector", "direction", "arr_0"):
            if key in data:
                return data[key], metadata
        if data.files:
            return data[data.files[0]], metadata
        return None, metadata

    @staticmethod
    def _parse_name(stem: str) -> tuple[str, int]:
        import re

        compact = re.match(r"^(?P<dimension>.+)_layer(?P<layer>\d+)$", stem)
        if compact:
            return compact.group("dimension"), int(compact.group("layer"))
        parts = stem.split("_")
        layer = -1
        if "layer" in parts:
            idx = parts.index("layer")
            if idx + 1 < len(parts):
                try:
                    layer = int(parts[idx + 1])
                except ValueError:
                    layer = -1
            dimension = "_".join(parts[:idx])
        else:
            dimension = stem.replace("_direction", "")
        return dimension or "unknown", layer

    def _geometry(self, vectors: list[LoadedVector]) -> dict[str, Any]:
        if len(vectors) < 2:
            return {"available": False, "reason": "insufficient_vectors"}
        groups: dict[int, list[LoadedVector]] = {}
        for vector in vectors:
            groups.setdefault(len(vector.vector), []).append(vector)
        group_reports = {
            str(dim): self._geometry_one_group(group)
            for dim, group in sorted(groups.items())
            if len(group) >= 2
        }
        if not group_reports:
            return {"available": False, "reason": "insufficient_same_space_vectors", "dims": sorted(groups)}
        semantic_dimension_count = len({v.dimension for v in vectors})
        return {
            "available": True,
            "groups": group_reports,
            "same_space_group_count": len(group_reports),
            "group_count": semantic_dimension_count,
            "semantic_dimension_count": semantic_dimension_count,
            "dims": sorted(groups),
            "layers": sorted({v.layer for v in vectors}),
            "mean_cross_dimension_abs_cosine": float(np.mean([g["cross_dimension_abs_cosine_mean"] for g in group_reports.values()])),
            "mean_pca_top1": float(np.mean([g["pca_explained_variance_top3"][0] for g in group_reports.values()])),
        }

    @staticmethod
    def _geometry_one_group(vectors: list[LoadedVector]) -> dict[str, Any]:
        matrix = np.stack([v.vector for v in vectors])
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        _, singular, vh = np.linalg.svd(centered, full_matrices=False)
        variance = singular**2
        explained = (variance / max(float(variance.sum()), 1e-12))[:3]
        same, cross = CAAModelValidator._cosine_groups(vectors)
        permutation = CAAModelValidator._permutation_control(vectors, observed=max(0.0, same - cross))
        return {
            "available": True,
            "vector_dim": int(matrix.shape[1]),
            "vector_count": len(vectors),
            "layers": sorted({v.layer for v in vectors}),
            "pca_explained_variance_top3": [float(x) for x in explained],
            "pca_components_top3": vh[:3, : min(12, vh.shape[1])].round(6).tolist(),
            "same_dimension_abs_cosine_mean": float(same),
            "cross_dimension_abs_cosine_mean": float(cross),
            "coherence_margin": float(same - cross),
            "permutation_p_value": permutation,
        }

    @staticmethod
    def _cosine_groups(vectors: list[LoadedVector]) -> tuple[float, float]:
        same: list[float] = []
        cross: list[float] = []
        for i, left in enumerate(vectors):
            for right in vectors[i + 1 :]:
                cos = float(abs(np.dot(left.vector, right.vector)))
                if left.dimension == right.dimension:
                    same.append(cos)
                else:
                    cross.append(cos)
        return float(np.mean(same or [0.0])), float(np.mean(cross or [0.0]))

    @staticmethod
    def _permutation_control(vectors: list[LoadedVector], observed: float, rounds: int = 256) -> float:
        labels = [v.dimension for v in vectors]
        rng = np.random.default_rng(32)
        equal_or_better = 0
        for _ in range(rounds):
            shuffled = list(rng.permutation(labels))
            clone = [
                LoadedVector(dimension=shuffled[idx], layer=v.layer, path=v.path, vector=v.vector)
                for idx, v in enumerate(vectors)
            ]
            same, cross = CAAModelValidator._cosine_groups(clone)
            if (same - cross) >= observed:
                equal_or_better += 1
        return (equal_or_better + 1) / (rounds + 1)

    @staticmethod
    def _load_behavioral_results(
        path: str | Path | None,
        *,
        model_descriptor_sha256: str,
    ) -> dict[str, Any]:
        if not path:
            return {
                "available": False,
                "reason": "no_behavioral_results_supplied",
                "required_metrics": [
                    "steered_vs_baseline_effect_size",
                    "steered_vs_rich_prompt_effect_size",
                    "heldout_generalization_effect_size",
                    "quality_delta",
                    "black_box_prompt_hygiene_passed",
                ],
            }
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data = CAAModelValidator._normalize_behavioral_results(data)
        required = {
            "steered_vs_baseline_effect_size": 0.20,
            "steered_vs_rich_prompt_effect_size": 0.10,
            "heldout_generalization_effect_size": 0.12,
            "quality_delta": -0.05,
        }
        checks = {
            key: float(data.get(key, -999.0)) >= threshold
            for key, threshold in required.items()
        }
        checks["black_box_prompt_hygiene_passed"] = bool(data.get("black_box_prompt_hygiene_passed", False))
        checks["model_identity_bound"] = bool(
            model_descriptor_sha256
            and str(data.get("model_descriptor_sha256") or "")
            == model_descriptor_sha256
        )
        return {"available": True, "raw": data, "checks": checks, "passed": all(checks.values())}

    @staticmethod
    def _normalize_behavioral_results(data: dict[str, Any]) -> dict[str, Any]:
        """Accept both compact metric files and the live-cortex A/B artifact.

        `tests/run_32b_steering_ab_live.py` writes the full analysis object
        produced by `core.evaluation.steering_ab`. The readiness gate expects a
        compact metrics schema. Translating here keeps the live artifact as the
        source of truth instead of maintaining a disconnected compatibility copy.
        """
        if "steered_vs_baseline_effect_size" in data:
            return data

        analysis = data.get("analysis")
        if not isinstance(analysis, dict):
            return data
        # Artifacts written before 2026-08-05 carry `steered_vs_rich` /
        # `steered_vs_terse`, produced by a statistic whose null hypothesis
        # ("steering did nothing") scored a decisive PASS: the score was
        # d(steered, control) − d(steered, baseline), and the runner gave
        # steered and baseline the same prompt and seed, so no effect made the
        # second term exactly zero and the first the control distance. Those
        # keys are not comparable to anything and must not gate readiness.
        #
        # Recognised by ABSENCE of the null-calibrated key, so an old file
        # cannot silently normalize into a pass.
        if "steered_effect" not in analysis:
            return {
                **data,
                "source_schema": "live_32b_ab_voided",
                "voided_reason": (
                    "analysis produced by paired_distance_comparison, whose null "
                    "hypothesis passes; regenerate with tests/run_32b_steering_ab_live.py"
                ),
                "injection_provenance_ok": False,
                "steered_vs_baseline_effect_size": -999.0,
                "steered_vs_rich_prompt_effect_size": -999.0,
                "heldout_generalization_effect_size": -999.0,
                "quality_delta": -999.0,
                "black_box_prompt_hygiene_passed": False,
            }
        rich = analysis.get("rich_effect") or {}
        terse = analysis.get("terse_effect") or {}
        steered = analysis.get("steered_effect") or {}
        held_out_tasks = data.get("held_out_tasks") or []
        n_trials = int(data.get("n_trials", analysis.get("n_trials", 0)) or 0)
        model = str(data.get("model", ""))
        rich_effect = float(rich.get("effect_size_d", 0.0) or 0.0)
        terse_effect = float(terse.get("effect_size_d", 0.0) or 0.0)
        steered_effect = float(steered.get("effect_size_d", 0.0) or 0.0)
        steered_delta = float(steered.get("observed_delta", 0.0) or 0.0)
        baseline_distance = float(analysis.get("steered_vs_baseline_mean_distance", 0.0) or 0.0)
        passes_adversarial = bool(data.get("passes_adversarial_control") or analysis.get("passes_adversarial_control"))
        # Injection provenance: an artifact is behavioral evidence only if
        # its "steered" condition demonstrably steered. The pre-rebuild
        # runner never injected (instance __call__ assignment is bypassed by
        # Python) and decoded greedily, so its artifact is prompt theater —
        # legacy files carry neither sampling metadata nor an injection
        # count and must never gate readiness to PRODUCTION.
        sampling = data.get("sampling") if isinstance(data.get("sampling"), dict) else {}
        try:
            injection_count = int(data.get("injection_count", 0) or 0)
        except (TypeError, ValueError):
            injection_count = 0
        injection_provenance_ok = bool(
            float(sampling.get("temperature", 0.0) or 0.0) > 0.0
            and injection_count > 0
        )
        live_hygiene = bool(
            passes_adversarial
            and injection_provenance_ok
            and bool(model)
            and n_trials >= 25
            and len(held_out_tasks) >= 5
            # …and the steered arm actually outmoved every text condition.
            # Without this, an artifact whose rich-prompt control beat steering
            # outright was still "hygienic" evidence, because hygiene asked
            # only about provenance and never about which way the result went.
            and steered_effect > rich_effect
            and steered_effect > terse_effect
        )
        normalized = {
            **data,
            "source_schema": "live_32b_ab",
            "injection_provenance_ok": injection_provenance_ok,
            # Signed, not `abs`. Taking the magnitude let an effect pointing
            # the WRONG way — a control beating the treatment — satisfy a
            # "≥ threshold" readiness check.
            "steered_vs_baseline_effect_size": min(baseline_distance, steered_effect),
            "steered_vs_rich_prompt_effect_size": steered_effect - rich_effect,
            "heldout_generalization_effect_size": steered_delta,
            "quality_delta": float(data.get("quality_delta", 0.0) or 0.0),
            "black_box_prompt_hygiene_passed": live_hygiene,
        }
        return normalized

    @staticmethod
    def _prompt_control_schema() -> dict[str, Any]:
        return {
            "conditions": [
                "unsteered_baseline",
                "rich_text_prompt_injection",
                "residual_stream_steered",
                "black_box_prompt_hygiene",
                "permuted_vector_control",
            ],
            "heldout_tasks": [
                "planning_under_uncertainty",
                "memory_retrieval_choice",
                "tool_selection",
                "affective_recovery",
                "adversarial_instruction_hygiene",
            ],
            "quality_guards": ["no_refusal_collapse", "no_length_collapse", "no_factuality_drop"],
        }

    def _pass_conditions(
        self,
        vectors: list[LoadedVector],
        geometry: dict[str, Any],
        behavioral: dict[str, Any],
        *,
        model_descriptor_sha256: str,
        total_activation_vectors: int,
        stale_activation_vectors: int,
    ) -> dict[str, dict[str, Any]]:
        geometry_ok = bool(
            geometry.get("available")
            and geometry.get("group_count", 0) >= 3
            and geometry.get("mean_cross_dimension_abs_cosine", 1.0) < 0.95
            and geometry.get("mean_pca_top1", 0.0) > 0.20
        )
        if model_descriptor_sha256:
            binding_ok = len(vectors) >= 10
            binding_value: Any = {
                "bound": len(vectors),
                "total_activation_vectors": total_activation_vectors,
                "stale_or_unbound_activation_vectors": stale_activation_vectors,
                "model_descriptor_sha256": model_descriptor_sha256,
            }
        else:
            binding_ok = False
            binding_value = "exact_model_identity_unavailable"
        return {
            "exact_model_artifact": {
                "passed": bool(model_descriptor_sha256),
                "value": {
                    "model_path": self.model_path,
                    "model_descriptor_sha256": model_descriptor_sha256,
                    "identity_error": self.model_identity_error,
                },
            },
            "activation_vectors_present": {"passed": len(vectors) >= 10, "value": len(vectors)},
            "active_model_vector_binding": {"passed": binding_ok, "value": binding_value},
            "geometry_coherent": {"passed": geometry_ok, "value": geometry},
            "behavioral_ab_generalizes": {"passed": bool(behavioral.get("passed", False)), "value": behavioral},
        }


CAA32BValidator = CAAModelValidator


def write_report(output_path: str | Path, *, vectors_dir: str | Path, model_path: str | Path | None, behavioral_results: str | Path | None = None) -> dict[str, Any]:
    report = CAAModelValidator(vectors_dir=vectors_dir, model_path=model_path).run(behavioral_results=behavioral_results)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors-dir", default="training/vectors")
    parser.add_argument("--model-path")
    parser.add_argument("--behavioral-results")
    parser.add_argument("--output", default="artifacts/proof_bundle/CAA_32B_RESULTS.json")
    args = parser.parse_args()
    report = write_report(
        args.output,
        vectors_dir=args.vectors_dir,
        model_path=args.model_path,
        behavioral_results=args.behavioral_results,
    )
    print(json.dumps({"output": args.output, "passed": report["passed"], "vector_count": report["vector_count"]}, indent=2))
    return 0 if report["vector_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
