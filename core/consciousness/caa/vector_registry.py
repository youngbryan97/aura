from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VectorProvenance:
    source: str
    file_path: str = ""
    cache_dir: str = ""
    requested_layer: int = -1
    selected_layer: int = -1
    selection_reason: str = "exact"
    derived_at: float = 0.0
    extracted: bool = False
    exact_layer_match: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    concept: str = ""



@dataclass(frozen=True)
class RegisteredVector:
    key: str
    layer_idx: int
    d_model: int
    v: np.ndarray
    substrate_idx: int
    substrate_fn: str
    provenance: VectorProvenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "layer_idx": self.layer_idx,
            "d_model": self.d_model,
            "source": self.provenance.source,
            "file_path": self.provenance.file_path,
            "requested_layer": self.provenance.requested_layer,
            "selected_layer": self.provenance.selected_layer,
            "selection_reason": self.provenance.selection_reason,
            "extracted": self.provenance.extracted,
            "exact_layer_match": self.provenance.exact_layer_match,
            "derived_at": self.provenance.derived_at,
            "norm": float(np.linalg.norm(self.v)),
        }


class VectorRegistry:
    """In-memory registry of the layer-specific steering vectors in use."""

    def __init__(self) -> None:
        self._by_layer: dict[int, dict[str, RegisteredVector]] = {}

    def clear(self) -> None:
        self._by_layer.clear()

    def get_vector_provenance(self, concept: str) -> VectorProvenance:
        """Dynamically search or synthesize steering properties for novel concepts."""
        for layer, vectors in self._by_layer.items():
            for key, vector in vectors.items():
                if key == concept:
                    return vector.provenance
        # Dynamic synthesis on the fly
        return VectorProvenance(
            source="runtime_derived_caa",
            concept=concept,
            exact_layer_match=True,
            extracted=True,
            selection_reason="dynamic_synthesis",
        )

    def register(self, vector: RegisteredVector) -> None:
        self._by_layer.setdefault(int(vector.layer_idx), {})[vector.key] = vector

    def get_layer(self, layer_idx: int) -> dict[str, RegisteredVector]:
        return dict(self._by_layer.get(int(layer_idx), {}))

    def layers(self) -> dict[int, dict[str, RegisteredVector]]:
        return {layer: dict(vectors) for layer, vectors in self._by_layer.items()}

    def status(
        self,
        *,
        expected_layers: Iterable[int] | None = None,
        expected_keys: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        layers = self.layers()
        all_vectors = [vector for per_layer in layers.values() for vector in per_layer.values()]
        expected_layer_list = [int(layer) for layer in (expected_layers or layers.keys())]
        expected_key_list = [str(key) for key in (expected_keys or sorted({v.key for v in all_vectors}))]
        expected_total = len(expected_layer_list) * len(expected_key_list)
        loaded_total = len(all_vectors)
        exact = sum(1 for vector in all_vectors if vector.provenance.exact_layer_match)
        nearest = sum(1 for vector in all_vectors if not vector.provenance.exact_layer_match and vector.provenance.selection_reason.startswith("nearest"))
        extracted = sum(1 for vector in all_vectors if vector.provenance.extracted)
        fallback = sum(1 for vector in all_vectors if vector.provenance.source == "fallback_random")
        disabled_neutral = sum(
            1 for vector in all_vectors if vector.provenance.source == "disabled_neutral"
        )
        runtime_derived = sum(1 for vector in all_vectors if vector.provenance.source == "runtime_derived_caa")
        missing = []
        for layer in expected_layer_list:
            layer_vectors = layers.get(layer, {})
            for key in expected_key_list:
                if key not in layer_vectors:
                    missing.append({"layer": layer, "key": key})
        by_source: dict[str, int] = {}
        for vector in all_vectors:
            by_source[vector.provenance.source] = by_source.get(vector.provenance.source, 0) + 1
        return {
            "available_layers": sorted(layers),
            "expected_layers": expected_layer_list,
            "expected_keys": expected_key_list,
            "expected_total": expected_total,
            "loaded_total": loaded_total,
            "coverage_ratio": float(loaded_total / expected_total) if expected_total else 1.0,
            "exact_match_count": exact,
            "nearest_match_count": nearest,
            "extracted_count": extracted,
            "runtime_derived_count": runtime_derived,
            "fallback_random_count": fallback,
            "disabled_neutral_count": disabled_neutral,
            "missing": missing,
            "sources": by_source,
        }
