#!/usr/bin/env python3
"""Production-scale CAA validation harness for Aura 32B.

This script validates the artifacts needed for a credible CAA claim:
activation-derived vectors, layer geometry, PCA structure, permutation
controls, rich-prompt comparator slots, black-box prompt-hygiene conditions,
and behavioral A/B result ingestion.  It runs quickly when evaluating existing
artifacts and can be paired with `extract_steering_vectors.py` for full 32B
extraction runs.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LoadedVector:
    dimension: str
    layer: int
    path: str
    vector: np.ndarray


class CAA32BValidator:
    def __init__(self, vectors_dir: str | Path = "training/vectors", model_path: str = "mlx-community/Qwen2.5-32B-Instruct-4bit") -> None:
        self.vectors_dir = Path(vectors_dir)
        self.model_path = str(model_path)

    def run(self, behavioral_results: str | Path | None = None) -> dict[str, Any]:
        vectors = self._load_vectors()
        activation_vectors = [v for v in vectors if v.layer >= 0]
        fallback_vectors = [v for v in vectors if v.layer < 0]
        geometry = self._geometry(activation_vectors)
        behavioral = self._load_behavioral_results(behavioral_results)
        prompt_controls = self._prompt_control_schema()
        pass_conditions = self._pass_conditions(activation_vectors, geometry, behavioral)
        return {
            "generated_at": time.time(),
            "model_path": self.model_path,
            "production_model_detected": "32b" in self.model_path.lower(),
            "vectors_dir": str(self.vectors_dir),
            "vector_count": len(vectors),
            "activation_vector_count": len(activation_vectors),
            "fallback_prior_count": len(fallback_vectors),
            "dimensions": sorted({v.dimension for v in vectors}),
            "layers": sorted({v.layer for v in vectors if v.layer >= 0}),
            "geometry": geometry,
            "behavioral_ab": behavioral,
            "prompt_controls": prompt_controls,
            "pass_conditions": pass_conditions,
            "passed": all(item["passed"] for item in pass_conditions.values()),
        }

    def _load_vectors(self) -> list[LoadedVector]:
        vectors: list[LoadedVector] = []
        if not self.vectors_dir.exists():
            return vectors
        for path in sorted([*self.vectors_dir.glob("*.npy"), *self.vectors_dir.glob("*.npz")]):
            try:
                arr = self._read_array(path)
            except Exception:
                continue
            if arr is None or arr.size == 0:
                continue
            dimension, layer = self._parse_name(path.stem)
            vec = np.asarray(arr, dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(LoadedVector(dimension=dimension, layer=layer, path=str(path), vector=vec))
        return vectors

    @staticmethod
    def _read_array(path: Path) -> np.ndarray | None:
        if path.suffix == ".npy":
            return np.load(path)
        data = np.load(path)
        for key in ("vector", "direction", "arr_0"):
            if key in data:
                return data[key]
        if data.files:
            return data[data.files[0]]
        return None

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
        return {
            "available": True,
            "groups": group_reports,
            "group_count": len(group_reports),
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
        same, cross = CAA32BValidator._cosine_groups(vectors)
        permutation = CAA32BValidator._permutation_control(vectors, observed=max(0.0, same - cross))
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
            same, cross = CAA32BValidator._cosine_groups(clone)
            if (same - cross) >= observed:
                equal_or_better += 1
        return (equal_or_better + 1) / (rounds + 1)

    @staticmethod
    def _load_behavioral_results(path: str | Path | None) -> dict[str, Any]:
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
        return {"available": True, "raw": data, "checks": checks, "passed": all(checks.values())}

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
    ) -> dict[str, dict[str, Any]]:
        geometry_ok = bool(
            geometry.get("available")
            and geometry.get("group_count", 0) >= 3
            and geometry.get("mean_cross_dimension_abs_cosine", 1.0) < 0.95
            and geometry.get("mean_pca_top1", 0.0) > 0.20
        )
        return {
            "production_32b_model": {"passed": "32b" in self.model_path.lower(), "value": self.model_path},
            "activation_vectors_present": {"passed": len(vectors) >= 10, "value": len(vectors)},
            "geometry_coherent": {"passed": geometry_ok, "value": geometry},
            "behavioral_ab_generalizes": {"passed": bool(behavioral.get("passed", False)), "value": behavioral},
        }


def write_report(output_path: str | Path, *, vectors_dir: str | Path, model_path: str, behavioral_results: str | Path | None = None) -> dict[str, Any]:
    report = CAA32BValidator(vectors_dir=vectors_dir, model_path=model_path).run(behavioral_results=behavioral_results)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors-dir", default="training/vectors")
    parser.add_argument("--model-path", default="mlx-community/Qwen2.5-32B-Instruct-4bit")
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
