"""Residual-stream steering injection for evaluation harnesses.

Correctness notes learned the hard way:

- ``layer.__call__ = hooked`` does NOT intercept ``layer(...)``: Python
  resolves special methods on the type, not the instance. The original live
  A/B runner had exactly that bug, so its "steered" condition never injected
  anything and the reported effect was a system-prompt confound. Injection
  here uses a temporary per-instance subclass swap — the same pattern the
  production extraction script (`training/extract_steering_vectors.py`) and
  `AffectiveSteeringHook` use.
- Vectors are unit-normalized at load so ``alpha`` means the same thing
  regardless of the raw extraction norms (~40-70 for 32B layers).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.Evaluation.SteeringInjection")


def load_production_vectors(
    vectors_dir: Path,
    *,
    dimensions: tuple[str, ...] = ("valence_positive", "curiosity"),
    model_descriptor_sha256: str,
) -> dict[int, np.ndarray]:
    """Load unit-normalized production steering vectors, averaged per layer.

    Only ``extracted=True`` vectors bound to the exact evaluated model are
    eligible. Same-width or same-config directions inhabit another activation
    basis and cannot be used as production evidence. Multiple requested
    dimensions on the same layer average then renormalize (matching how
    simultaneous affect axes compose in the engine).
    """
    if re.fullmatch(r"[0-9a-f]{64}", model_descriptor_sha256) is None:
        raise ValueError("steering_model_identity_unavailable")
    per_layer: dict[int, list[np.ndarray]] = {}
    for path in sorted(Path(vectors_dir).glob("*.npz")):
        try:
            with np.load(path, allow_pickle=True) as z:
                if not bool(z.get("extracted", np.array(False)).item()):
                    continue
                observed_digest = str(
                    z["model_descriptor_sha256"].item()
                    if "model_descriptor_sha256" in z
                    else ""
                )
                if observed_digest != model_descriptor_sha256:
                    continue
                dimension = str(z["dimension"].item()) if "dimension" in z else ""
                if dimension not in dimensions:
                    continue
                layer = int(z["layer"].item()) if "layer" in z else -1
                vec = np.asarray(z["v"], dtype=np.float32).flatten()
        except (OSError, KeyError, ValueError) as exc:
            logger.debug("Skipped unreadable vector %s: %s", path.name, exc)
            continue
        norm = float(np.linalg.norm(vec))
        if layer < 0 or norm <= 1e-6:
            continue
        per_layer.setdefault(layer, []).append(vec / norm)

    combined: dict[int, np.ndarray] = {}
    for layer, vecs in per_layer.items():
        mean = np.mean(np.stack(vecs), axis=0)
        norm = float(np.linalg.norm(mean))
        if norm > 1e-6:
            combined[layer] = (mean / norm).astype(np.float32)
    return combined


def derive_control_vectors(
    vectors: dict[int, np.ndarray],
    arm: str,
    *,
    seed: int = 0,
) -> dict[int, np.ndarray]:
    """Build a specificity-control vector set from the production one.

    The arms answer the three ways a divergence result can be right about
    "something changed" and wrong about "these vectors changed it":

    ``production``
        The real thing.
    ``zero``
        Same hook, same α, same arithmetic — injecting nothing. Whatever
        effect survives here belongs to running the hook, not to steering.
        The one control that catches a harness perturbing its own decode.
    ``random``
        Norm-matched random directions on the same layers. Separates "this
        direction" from "any perturbation of this size".
    ``shuffled_layers``
        The real vectors, permuted across the same layers. Separates "this
        direction HERE" from "this direction anywhere".

    Norm-matched by construction, so α means the same thing in every arm.
    """
    if arm == "production":
        return {layer: np.array(vec, copy=True) for layer, vec in vectors.items()}
    if arm == "zero":
        return {
            layer: np.zeros_like(np.asarray(vec, dtype=np.float32))
            for layer, vec in vectors.items()
        }
    rng = np.random.default_rng(seed)
    if arm == "random":
        control: dict[int, np.ndarray] = {}
        for layer, vec in vectors.items():
            source = np.asarray(vec, dtype=np.float32)
            draw = rng.standard_normal(source.shape).astype(np.float32)
            norm = float(np.linalg.norm(draw))
            if norm <= 1e-6:
                draw = np.ones_like(source)
                norm = float(np.linalg.norm(draw))
            control[layer] = (draw / norm) * float(np.linalg.norm(source))
        return control
    if arm == "shuffled_layers":
        layers = sorted(vectors)
        if len(layers) < 2:
            raise ValueError("shuffled_layers needs at least two steered layers")
        order = list(layers)
        # A derangement: no layer may keep its own vector, or the "control"
        # is partly the treatment.
        for _ in range(64):
            rng.shuffle(order)
            if all(a != b for a, b in zip(layers, order, strict=True)):
                break
        else:  # pragma: no cover - a rotation is always a derangement
            order = layers[1:] + layers[:1]
        return {
            layer: np.array(vectors[source], copy=True)
            for layer, source in zip(layers, order, strict=True)
        }
    raise ValueError(f"unknown steering control arm: {arm!r}")


class ResidualSteeringInjector:
    """Toggleable residual-stream injection over a loaded MLX model.

    The injected vector is resolved AT CALL TIME from :attr:`arm`, so a
    campaign can run its zero-vector, random-vector and shuffled-layer
    controls through the identical hook on the identical model rather than
    reinstalling between arms — a reinstall is itself a difference between
    conditions, and the controls exist to rule differences out.
    """

    ARMS = ("production", "zero", "random", "shuffled_layers")

    def __init__(
        self,
        model: Any,
        vectors: dict[int, np.ndarray],
        *,
        alpha: float = 8.0,
        control_seed: int = 0,
    ) -> None:
        import mlx.core as mx

        self._mx = mx
        self._model = model
        self._alpha = float(alpha)
        self._source_vectors = {
            layer: np.asarray(vec, dtype=np.float32) for layer, vec in vectors.items()
        }
        self._arms: dict[str, dict[int, Any]] = {}
        for name in self.ARMS:
            try:
                derived = derive_control_vectors(
                    self._source_vectors, name, seed=control_seed
                )
            except ValueError:
                continue  # e.g. shuffled_layers with a single steered layer
            self._arms[name] = {
                layer: mx.array(vec) for layer, vec in derived.items()
            }
        self._arm = "production"
        self._installed: list[tuple[Any, type]] = []
        self.active = False
        self.injection_count = 0
        self.injections_by_arm: dict[str, int] = {name: 0 for name in self._arms}

    @property
    def available_arms(self) -> tuple[str, ...]:
        return tuple(name for name in self.ARMS if name in self._arms)

    @property
    def arm(self) -> str:
        return self._arm

    @arm.setter
    def arm(self, name: str) -> None:
        if name not in self._arms:
            raise ValueError(
                f"arm {name!r} unavailable; have {self.available_arms}"
            )
        self._arm = name

    @property
    def _vectors(self) -> dict[int, Any]:
        """The vectors of the ACTIVE arm."""
        return self._arms[self._arm]

    def _layers(self) -> Any:
        for attr_path in ("model.layers", "layers"):
            obj = self._model
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
            except AttributeError:
                continue
            if hasattr(obj, "__len__") and len(obj) > 0:
                return obj
        raise RuntimeError("cannot locate transformer layers on model")

    def install(self) -> int:
        """Subclass-swap the target layers; returns the number hooked."""
        mx = self._mx
        layers = self._layers()
        injector = self

        for layer_idx in sorted(self._source_vectors):
            if layer_idx >= len(layers):
                continue
            layer = layers[layer_idx]
            original_class = layer.__class__

            def _make_steered_class(orig_cls: type, idx: int) -> type:
                class SteeredLayer(orig_cls):  # type: ignore[misc, valid-type]
                    __module__ = orig_cls.__module__

                    def __call__(self, *args: Any, **kwargs: Any) -> Any:
                        result = super().__call__(*args, **kwargs)
                        if not injector.active:
                            return result
                        # Resolved per call, so switching arms needs no
                        # reinstall — the controls run through this exact hook.
                        vec = injector._vectors.get(idx)
                        if vec is None:
                            return result
                        hidden = result[0] if isinstance(result, tuple) else result
                        try:
                            steered = hidden + injector._alpha * vec.astype(hidden.dtype)
                            injector.injection_count += 1
                            injector.injections_by_arm[injector._arm] = (
                                injector.injections_by_arm.get(injector._arm, 0) + 1
                            )
                        except (TypeError, ValueError):
                            return result
                        if isinstance(result, tuple):
                            return (steered,) + result[1:]
                        return steered

                return SteeredLayer

            layer.__class__ = _make_steered_class(original_class, layer_idx)
            self._installed.append((layer, original_class))
        del mx
        return len(self._installed)

    def remove(self) -> None:
        for layer, original_class in self._installed:
            try:
                layer.__class__ = original_class
            except TypeError as exc:
                logger.warning("Steering layer restore failed: %s", exc)
        self._installed.clear()

    def __enter__(self) -> ResidualSteeringInjector:
        self.install()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.remove()
