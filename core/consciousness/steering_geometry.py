"""Working out the shape of a model nobody described.

Hidden size and layer count are needed before a steering vector can be
attached, and the checkpoint does not always say. So: the descriptor if there
is one, the metadata if there is not, the cached vectors as a last reading, and
a refusal rather than a guess when none of them answer.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.model_layers import resolve_model_layers
from core.runtime.state_ownership import state_root


class _FindsTheModelsGeometry:
    """Lifted whole from AffectiveSteeringEngine; see affective_steering.py."""

    @staticmethod
    def _runtime_vector_cache_dir(
        *,
        n_layers: int,
        d_model: int,
        model_identity: dict[str, object] | None = None,
    ) -> Path:
        """Writable CAA cache partitioned by geometry and exact model basis."""
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .affective_steering import (
            _emit_affective_fault,
            logger,
        )

        try:
            from core.config import config as aura_config

            base = aura_config.paths.data_dir / "steering_vectors"
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _emit_affective_fault(
                exc,
                action="used user-scoped runtime steering cache after config path lookup failed",
                severity="warning",
                stage="runtime_vector_cache_dir",
                extra={"n_layers": n_layers, "d_model": d_model},
            )
            logger.debug("Runtime steering cache config unavailable, using user cache: %s", exc)
            base = state_root() / "steering_vectors"
        geometry = base / f"dmodel_{int(d_model)}_layers_{int(n_layers)}"
        digest = str((model_identity or {}).get("descriptor_sha256") or "")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("steering_model_identity_unavailable")
        return geometry / f"model_{digest[:16]}"

    @staticmethod
    def _coerce_hidden_size(candidate: Any) -> int | None:
        try:
            value = int(candidate)
        except (TypeError, ValueError, OverflowError):
            return None
        return value if value > 512 else None

    def _metadata_hidden_size(self, model: Any) -> int | None:
        """Read hidden size from common MLX/HF model metadata and embeddings."""

        roots = [model, getattr(model, "model", None), getattr(model, "args", None)]
        roots.extend(getattr(root, "config", None) for root in list(roots) if root is not None)
        for root in roots:
            if root is None:
                continue
            for attr in (
                "hidden_size",
                "d_model",
                "model_dim",
                "n_embd",
                "dim",
                "embed_dim",
            ):
                hidden = self._coerce_hidden_size(getattr(root, attr, None))
                if hidden is not None:
                    return hidden

        for root in (model, getattr(model, "model", None)):
            if root is None:
                continue
            for attr in ("embed_tokens", "tok_embeddings", "wte"):
                emb = getattr(root, attr, None)
                weight = getattr(emb, "weight", None)
                shape = getattr(weight, "shape", None)
                if shape and len(shape) >= 2:
                    hidden = self._coerce_hidden_size(shape[-1])
                    if hidden is not None:
                        return hidden
        return None

    def _cached_vector_hidden_size(
        self,
        n_layers: int,
        *,
        model_identity: dict[str, object] | None,
    ) -> int | None:
        """Infer hidden size from trusted packaged/runtime CAA vector artifacts.

        Some MLX model wrappers hide their projection weights, but Aura ships
        vetted CAA vectors for the live 32B lane. If model introspection cannot
        expose d_model, using the vector geometry is better than guessing and
        re-deriving incompatible vectors on every boot.
        """
        from .affective_steering import (
            AFFECTIVE_DIMENSIONS,
            _emit_affective_fault,
            logger,
        )


        expected_digest = str((model_identity or {}).get("descriptor_sha256") or "")
        if re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
            return None

        roots: list[Path] = []
        env_dir = os.environ.get("AURA_STEERING_DIR")
        if env_dir:
            roots.append(Path(env_dir))
        if not env_dir:
            try:
                from core.config import config as aura_config

                runtime_base = aura_config.paths.data_dir / "steering_vectors"
                roots.extend(sorted(runtime_base.glob(f"dmodel_*_layers_{int(n_layers)}")))
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                _emit_affective_fault(
                    exc,
                    action="continued model geometry discovery without runtime steering cache scan",
                    severity="warning",
                    stage="cached_vector_geometry",
                )
                logger.debug("Runtime steering cache scan unavailable: %s", exc)
            roots.append(Path(__file__).parent.parent.parent / "training" / "vectors")

        target_layers = set(self._compute_target_layers(n_layers))
        keys = {str(dim["key"]) for dim in AFFECTIVE_DIMENSIONS}
        counts: dict[int, int] = {}
        for root in roots:
            if not root.exists():
                continue
            for key in keys:
                for path in root.glob(f"{key}_layer*.np*"):
                    match = re.match(rf"^{re.escape(key)}_layer_?(?P<layer>\d+)$", path.stem)
                    if not match:
                        continue
                    try:
                        if int(match.group("layer")) not in target_layers:
                            continue
                        if path.suffix != ".npz":
                            continue
                        with np.load(path, allow_pickle=True) as data:
                            observed_digest = str(
                                data["model_descriptor_sha256"]
                                if "model_descriptor_sha256" in data
                                else ""
                            )
                            if observed_digest != expected_digest:
                                continue
                            vector = None
                            for vector_key in ("v", "vector", "direction", "arr_0"):
                                if vector_key in data:
                                    vector = data[vector_key]
                                    break
                            if vector is None:
                                continue
                        dim = int(np.asarray(vector).reshape(-1).shape[0])
                    except (OSError, ValueError, RuntimeError, AttributeError, TypeError) as exc:
                        _emit_affective_fault(
                            exc,
                            action="ignored unreadable CAA vector during model geometry inference",
                            severity="warning",
                            stage="cached_vector_geometry",
                            extra={"path": str(path)},
                        )
                        continue
                    if dim > 512:
                        counts[dim] = counts.get(dim, 0) + 1

        if not counts:
            return None
        return max(counts.items(), key=lambda item: (item[1], item[0]))[0]

    @staticmethod
    def _descriptor_geometry(
        model_identity: dict[str, object] | None,
    ) -> tuple[int, int]:
        profile = (model_identity or {}).get("artifact_profile")
        if not isinstance(profile, dict):
            return 0, 0
        try:
            n_layers = int(profile.get("num_hidden_layers") or 0)
            d_model = int(profile.get("hidden_size") or 0)
        except (TypeError, ValueError, OverflowError):
            return 0, 0
        return max(0, n_layers), max(0, d_model)

    def _discover_model_geometry(
        self,
        model,
        *,
        model_identity: dict[str, object] | None = None,
    ) -> tuple[int, int]:
        """Determine n_layers and d_model from the loaded model."""
        from .affective_steering import (
            _emit_affective_fault,
            logger,
        )

        try:
            # Pre-initialize d_model so the fallback ``return`` on line ~1107
            # never raises UnboundLocalError when no inner branch assigned it.
            d_model: int | None = None
            # Flexible layer discovery (handles model.layers and model.model.layers)
            layers = self._discover_model_layers(model)
            if not layers:
                return 0, 0
            n_layers = len(layers)
            descriptor_layers, descriptor_hidden = self._descriptor_geometry(model_identity)
            if descriptor_layers and descriptor_layers != n_layers:
                logger.error(
                    "Loaded model has %d layers but its exact descriptor records %d.",
                    n_layers,
                    descriptor_layers,
                )
                return 0, 0

            metadata_hidden = self._metadata_hidden_size(model)
            if metadata_hidden is not None:
                if descriptor_hidden and descriptor_hidden != metadata_hidden:
                    logger.error(
                        "Loaded model has d_model=%d but its exact descriptor records %d.",
                        metadata_hidden,
                        descriptor_hidden,
                    )
                    return 0, 0
                return n_layers, metadata_hidden

            if descriptor_hidden > 512:
                return n_layers, descriptor_hidden

            # d_model: find first weight with the right shape
            # Typically in attention q_proj or input_layernorm
            for layer in layers[:3]:
                for norm_name in (
                    "input_layernorm",
                    "post_attention_layernorm",
                    "ln_1",
                    "ln1",
                    "norm1",
                ):
                    norm = getattr(layer, norm_name, None)
                    weight = getattr(norm, "weight", None)
                    shape = getattr(weight, "shape", None)
                    if shape:
                        d_model = self._coerce_hidden_size(shape[0])
                        if d_model is not None:
                            return n_layers, d_model

                # Try attention layers
                for attr_name in ["self_attn", "attention", "attn"]:
                    attn = getattr(layer, attr_name, None)
                    if attn:
                        for proj_name in ["q_proj", "o_proj"]:
                            proj = getattr(attn, proj_name, None)
                            if proj and hasattr(proj, "weight"):
                                shape = proj.weight.shape
                                # q_proj: [d_model * n_heads/n_heads, d_model] or [d_model, d_model]
                                d_model = shape[-1]
                                if d_model > 512:
                                    return n_layers, d_model

                # Try feed-forward layers
                for attr_name in ["mlp", "feed_forward", "ff"]:
                    ff = getattr(layer, attr_name, None)
                    if ff:
                        # Try finding a linear projection to get d_model
                        for proj_name in ["down_proj", "w2", "gate_proj"]:
                            proj = getattr(ff, proj_name, None)
                            if proj and hasattr(proj, "weight"):
                                shape = getattr(proj.weight, "shape", None)
                                if shape:
                                    candidates = [shape[-1]]
                                    if proj_name in {"down_proj", "w2"}:
                                        candidates.insert(0, shape[0])
                                    for candidate in candidates:
                                        d_model = self._coerce_hidden_size(candidate)
                                        if d_model is not None:
                                            return n_layers, d_model

            cached_hidden = self._cached_vector_hidden_size(
                n_layers,
                model_identity=model_identity,
            )
            if cached_hidden is not None:
                logger.info(
                    "Geometry discovery using cached CAA vector d_model=%d for %d-layer model.",
                    cached_hidden,
                    n_layers,
                )
                return n_layers, cached_hidden

            logger.warning("Geometry discovery reached fallback for d_model.")
            return n_layers, 0
        except (RuntimeError, AttributeError, TypeError) as e:
            _emit_affective_fault(
                e,
                action="aborted affective steering attach because model geometry discovery failed",
                severity="degraded",
                stage="model_geometry_discovery",
            )
            logger.error("Error discovering model geometry: %s", e)
            return 0, 0

    def _discover_model_layers(self, model) -> list[Any] | None:
        """Helper to find the layers list in various MLX model structures."""
        view = resolve_model_layers(model)
        return view.layers if view is not None else None

    def _compute_target_layers(self, n_layers: int) -> list[int]:
        """
        Compute which layers to hook based on total model depth.

        Target 40-65% depth — middle layers where semantic representations
        are rich but generation hasn't been "committed" yet.

        We hook 2-3 layers in this range for multi-layer steering,
        which the literature shows is more effective than single-layer.
        (van der Weij et al., 2024: simultaneous injection at different
         layers is more effective than single-point injection.)
        """
        from .affective_steering import (
            TARGET_LAYER_RANGE,
        )

        lo = math.floor(n_layers * TARGET_LAYER_RANGE[0])
        hi = math.floor(n_layers * TARGET_LAYER_RANGE[1])
        span = hi - lo

        if span <= 2:
            return [lo]
        elif span <= 5:
            return [lo, lo + span // 2]
        else:
            # 3 evenly spaced layers in the target range
            return [lo, lo + span // 3, lo + 2 * span // 3]
