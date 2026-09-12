"""A steering vector already on disk, and whether it is the right one.

A cached vector from a different model is the wrong shape or, worse, the right
shape and the wrong meaning. So a path is only accepted when its metadata names
the model it was derived for and its width matches what this model has. Where
it does not, the answer is nothing rather than a vector that will steer
somewhere.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only; that module imports this one
    from .affective_steering import SteeringVector

import re
from pathlib import Path
from typing import Any

import numpy as np


class _ReadsACachedVector:
    """Lifted whole from SteeringVectorLibrary; see affective_steering.py."""

    def _candidate_paths_for_key(self, key: str) -> list[tuple[int, Path]]:
        candidates: list[tuple[int, Path]] = []
        for root in [self._cache_dir, *self._source_dirs]:
            for path in sorted(root.glob(f"{key}_layer*.np*")):
                match = re.match(rf"^{re.escape(key)}_layer_?(?P<layer>\d+)$", path.stem)
                if not match:
                    continue
                candidates.append((int(match.group("layer")), path))
        return candidates

    def _vector_dim_for_path(self, path: Path) -> int:
        cache_key = str(path)
        if cache_key in self._path_dim_cache:
            return self._path_dim_cache[cache_key]
        vector, _ = self._read_cached_array(path)
        dim = int(np.asarray(vector).reshape(-1).shape[0])
        self._path_dim_cache[cache_key] = dim
        return dim

    def _cached_metadata(self, path: Path) -> dict[str, Any]:
        cache_key = str(path)
        cached = self._path_meta_cache.get(cache_key)
        if cached is not None:
            return cached
        _vector, metadata = self._read_cached_array(path)
        self._path_meta_cache[cache_key] = dict(metadata)
        return metadata

    def _matches_expected_model(self, path: Path) -> bool:
        expected = self._expected_model_identity.get("descriptor_sha256")
        if not expected:
            return self._allow_unbound_artifacts
        if path.suffix != ".npz":
            return False
        try:
            observed = self._cached_metadata(path).get("model_descriptor_sha256")
        except (OSError, ValueError, RuntimeError, TypeError):
            return False
        return observed == expected

    def _resolve_cached_path(
        self,
        key: str,
        requested_layer: int,
        d_model: int,
    ) -> tuple[int, Path, bool] | None:
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .affective_steering import (
            _emit_affective_fault,
            logger,
        )

        candidates = self._candidate_paths_for_key(key)
        if not candidates:
            return None
        compatible = []
        for layer, path in candidates:
            try:
                if (
                    self._matches_expected_model(path)
                    and self._vector_dim_for_path(path) == d_model
                ):
                    compatible.append((layer, path))
            except (OSError, ValueError, RuntimeError, AttributeError, TypeError) as exc:
                _emit_affective_fault(
                    exc,
                    action="skipped unreadable cached steering vector and continued derivation",
                    severity="warning",
                    stage="resolve_cached_vector",
                    extra={"path": str(path), "key": key, "requested_layer": requested_layer},
                )
                logger.warning("Skipping unreadable steering vector %s: %s", path, exc)
        if not compatible:
            logger.debug(
                "No compatible cached CAA vector for %s at layer %d with d_model=%d; deriving.",
                key,
                requested_layer,
                d_model,
            )
            return None
        candidates = compatible
        exact = [(layer, path) for layer, path in candidates if layer == requested_layer]
        if exact:
            exact.sort(key=lambda item: (0 if item[1].suffix == ".npz" else 1, item[0]))
            layer, path = exact[0]
            return layer, path, True
        candidates.sort(
            key=lambda item: (
                abs(item[0] - requested_layer),
                0 if item[1].suffix == ".npz" else 1,
                item[0],
            )
        )
        layer, path = candidates[0]
        return layer, path, False

    def _read_cached_array(self, path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        if path.suffix == ".npy":
            return np.load(path), {}
        with np.load(path, allow_pickle=True) as data:
            vector = None
            for key in ("v", "vector", "direction", "arr_0"):
                if key in data:
                    vector = data[key]
                    break
            if vector is None:
                raise ValueError(f"no vector payload in {path}")
            meta: dict[str, Any] = {}
            for key in data.files:
                if key in {"v", "vector", "direction", "arr_0"}:
                    continue
                value = data[key]
                if getattr(value, "shape", ()) == ():
                    meta[key] = value.item()
            return vector, meta

    def _load_cached_vector(
        self,
        *,
        key: str,
        requested_layer: int,
        selected_layer: int,
        path: Path,
        d_model: int,
        dim_spec: dict[str, Any],
        exact_match: bool,
    ) -> SteeringVector:
        from .affective_steering import (
            SteeringVector,
        )

        vector, meta = self._read_cached_array(path)
        expected_identity = self._expected_model_identity.get("descriptor_sha256")
        if expected_identity and meta.get("model_descriptor_sha256") != expected_identity:
            raise ValueError(f"vector {path.name} belongs to another model basis")
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if not np.isfinite(vec).all():
            raise ValueError(f"vector {path.name} contains non-finite values")
        norm = np.linalg.norm(vec)
        if norm <= 1e-8:
            raise ValueError(f"vector {path.name} is near-zero and cannot steer safely")
        vec = vec / norm
        if vec.shape[0] != d_model:
            raise ValueError(
                f"vector {path.name} has d_model={vec.shape[0]} but runtime expects {d_model}"
            )
        source = str(meta.get("source", self._source))
        extracted = bool(meta.get("extracted", source.startswith("extracted")))
        return SteeringVector(
            key=key,
            layer_idx=requested_layer,
            d_model=d_model,
            v=vec,
            substrate_idx=dim_spec["substrate_idx"],
            substrate_fn=dim_spec["substrate_fn"],
            is_derived=True,
            derived_at=float(meta.get("derived_at", path.stat().st_mtime)),
            source=source if exact_match else f"{source}_nearest_layer",
            file_path=str(path),
            requested_layer=requested_layer,
            selected_layer=selected_layer,
            selection_reason="exact" if exact_match else f"nearest_layer:{selected_layer}",
            exact_layer_match=exact_match,
            extracted=extracted,
        )
