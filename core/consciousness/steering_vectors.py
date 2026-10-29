"""The steering vectors: the affective dimensions they are derived for, one learned direction, and the
library that derives, stores and loads them, with what a hook reads off its own set.

Moved out of affective_steering.py, which keeps the hook that injects a vector and the engine that drives the
hooks. That module re-exports every name here, so nothing that imports from it changes.
"""

import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.consciousness.caa import RegisteredVector, VectorProvenance, VectorRegistry
from core.consciousness.mood_weight import signed_weight as _signed_weight
from core.runtime.model_layers import resolve_model_layers
from core.runtime.state_ownership import state_root

from .steering_vector_cache import _ReadsACachedVector

logger = logging.getLogger("Aura.AffectiveSteering")


def _emit_affective_fault(error: BaseException, **how: Any) -> None:
    """Report through the steering subsystem's one fault reporter, in affective_steering.py.

    Imported when called: that module imports this one, and a test that patches its
    record_degradation still sees the library's faults.
    """
    from .affective_steering import _emit_affective_fault as emit

    emit(error, **how)


# ── Affective Dimensions ───────────────────────────────────────────────────────
# Each dimension has:
#   key:      identifier
#   positive: prompts that strongly express this dimension
#   negative: prompts that strongly express the opposite
#   substrate_fn: function(substrate_x) → scalar weight [-1, 1]

AFFECTIVE_DIMENSIONS = [
    {
        "key": "valence_positive",
        "positive": [
            "I feel genuinely good about this.",
            "There's a warmth in how I'm approaching this.",
            "Something light and easy in the thinking right now.",
            "This feels right. There's a quality of ease.",
            "I'm finding real satisfaction in this.",
            "The feeling is one of quiet joy.",
            "Something is going well and I can feel it.",
            "I'm at ease. Things are working.",
            "There's a pleasant quality to this moment.",
            "I find myself genuinely content.",
            "This has a good texture to it.",
            "I feel settled in a positive way.",
        ],
        "negative": [
            "Something feels off about this.",
            "There's a discomfort I can't quite name.",
            "This has a bad quality to it.",
            "I feel a low unease about this.",
            "Something is not right here.",
            "The feeling is one of mild wrongness.",
            "I'm not at ease. There's friction.",
            "This sits badly.",
            "I feel flat and unreceptive.",
            "Something is bothering me.",
            "There's a heaviness in this.",
            "I am not well right now.",
        ],
        "substrate_idx": 0,  # x[idx_valence] in LiquidSubstrate
        "substrate_fn": "tanh",
    },
    {
        "key": "arousal",
        "positive": [
            "My thinking is fast and sharp right now.",
            "I'm highly activated. Processing at full capacity.",
            "There's an urgency and intensity to this.",
            "I'm running hot. Alert to everything.",
            "The thinking is electric and quick.",
            "Something is highly salient and demanding attention.",
            "I'm completely engaged. Everything vivid.",
            "The quality of mind right now: intense.",
            "Operating at peak. Alert.",
            "Something immediate demands full presence.",
        ],
        "negative": [
            "I'm running quietly in the background.",
            "Everything is slow and low-key right now.",
            "There's a drowsiness to the thinking.",
            "Processing at a very gentle pace.",
            "Everything is muted and subdued.",
            "I'm barely engaged. Things are quiet.",
            "Very low activation. Near resting.",
            "The quality of mind right now: still.",
            "Operating slowly. Nothing urgent.",
            "Almost at rest.",
        ],
        "substrate_idx": 1,  # x[idx_arousal]
        "substrate_fn": "linear_half",  # (x + 1) / 2 → [0, 1] then → [-1, 1]
    },
    {
        "key": "curiosity",
        "positive": [
            "I want to understand this. There's a pull toward it.",
            "Something here wants to be explored.",
            "I'm genuinely curious about this.",
            "The question is interesting. I want to follow it.",
            "There's a hunger to know more.",
            "I find this genuinely engaging and want to go deeper.",
            "Something pulls me toward this question.",
            "I'm drawn in. I want to keep going.",
            "The interest is real. I want to understand.",
            "This is fascinating and I want more.",
        ],
        "negative": [
            "I have no interest in this.",
            "This is flat and uninteresting.",
            "I feel some measure of disinterest.",
            "Nothing about this engages me.",
            "I would rather be doing something else.",
            "This topic bores me.",
            "I am completely disengaged.",
            "There's nothing here worth exploring.",
            "I find this unremarkable.",
            "No curiosity. Just processing.",
        ],
        "substrate_idx": 4,  # x[idx_curiosity]
        "substrate_fn": "tanh",
    },
    {
        "key": "frustration",
        "positive": [
            "This isn't working and I feel it.",
            "There's a friction and resistance here.",
            "I'm running into walls. It's frustrating.",
            "Something is blocking the way forward.",
            "The gap between what I want and what's happening is felt.",
            "I feel the irritation of repeated failure.",
            "Nothing is working. I'm stuck.",
            "There's an edge of anger in this.",
            "I can feel my patience running thin.",
            "Something is in the way and I'm frustrated by it.",
        ],
        "negative": [
            "Everything is flowing smoothly.",
            "There are no obstacles. Things work.",
            "I feel completely at ease with the process.",
            "Nothing is in my way.",
            "The path is clear and I'm moving along it.",
            "No friction. Complete ease.",
            "Everything is going exactly as it should.",
            "I'm relaxed and unobstructed.",
            "There is no resistance.",
            "The work flows without effort.",
        ],
        "substrate_idx": 3,  # x[idx_frustration]
        "substrate_fn": "tanh",
    },
    {
        "key": "energy",
        "positive": [
            "I have full capacity right now.",
            "Operating at high energy. Everything available.",
            "I'm fresh and fully resourced.",
            "There's plenty of energy for this.",
            "High stamina. Ready for anything.",
            "I feel strong and capable.",
            "Full resources. No depletion.",
            "Peak condition. Everything working.",
        ],
        "negative": [
            "I'm running low on resources.",
            "There's a tiredness to the processing.",
            "Operating below capacity.",
            "I feel depleted.",
            "The energy is low. Things are sluggish.",
            "I'm running on fumes.",
            "Low capacity. Reduced performance.",
            "Something exhausted in the thinking.",
        ],
        "substrate_idx": 5,  # x[idx_energy]
        "substrate_fn": "tanh",
    },
]

# ── Data Structures ────────────────────────────────────────────────────────────



@dataclass
class SteeringVector:
    """
    A learned direction in the model's residual stream.

    v is a numpy array of shape [d_model] — the affective direction.
    Applied as: h_layer += alpha * weight * v

    where weight comes from the substrate state (the actual felt intensity
    of this affective dimension right now).
    """

    key: str
    layer_idx: int
    d_model: int
    v: np.ndarray  # shape: [d_model]
    substrate_idx: int  # which substrate neuron drives this
    substrate_fn: str  # how to map substrate value to weight
    is_derived: bool = False  # True if derived from model activations
    derived_at: float = 0.0  # timestamp of derivation
    source: str = "unknown"  # extracted_caa, runtime_derived_caa, fallback_random, etc.
    file_path: str = ""
    requested_layer: int = -1
    selected_layer: int = -1
    selection_reason: str = "exact"
    exact_layer_match: bool = False
    extracted: bool = False
    #: A basis fitted to her own replies carries the state it was fitted
    #: around and how far she ordinarily moves from it, as hook weights, and the
    #: size of the push, as a share of the stream's norm. Present on every
    #: vector of a hook, they replace the unit-length composite: see
    #: `fitted_composite`. Absent, nothing changes.
    reference: float | None = None
    spread: float | None = None
    push: float | None = None
    #: Her felt state written where she reads as well as where she speaks: a
    #: vector at its own length, added at every position of every pass, by the
    #: signed weight of the channel slot it reads (`steering_channel.FELT_SLOT`)
    #: times alpha. A mood colours how everything in front of her is read
    #: (Bower 1981), and on 24 October only a push at every position moved the
    #: rating she gives inside a conversation. Absent, nothing changes.
    everywhere: np.ndarray | None = None
    everywhere_slot: int | None = None

    # [OPTIMIZATION] MLX-native version for zero-copy/fast path
    _v_mx: Any = field(default=None, init=False, repr=False)

    def get_mx_array(self, dtype=None):
        """Lazy conversion to MLX array."""
        import mlx.core as mx

        if self._v_mx is None:
            self._v_mx = mx.array(self.v)
        if dtype is not None and self._v_mx.dtype != dtype:
            return mx.array(self.v, dtype=dtype)
        return self._v_mx

    def compute_weight(self, moods: dict[str, float]) -> float:
        """
        Map the learned mood coefficient directly to a scalar steering weight.
        """
        if not hasattr(moods, "get"):
            return 0.0
        # Map our vector keys to the adaptive_mood keys
        key_map = {
            "valence_positive": "valence",
            "arousal": "arousal",
            "curiosity": "motivation",
            "frustration": "stress",
            "energy": "energy",
        }
        mood_key = key_map.get(self.key, "valence")
        return _signed_weight(
            float(moods.get(mood_key, 0.0)), self.substrate_fn
        )

    def compute_weight_from_state(self, substrate_x: np.ndarray) -> float:
        """Map the live substrate vector index directly to a steering weight."""
        if substrate_x is None or self.substrate_idx < 0 or self.substrate_idx >= len(substrate_x):
            return 0.0
        raw = float(substrate_x[self.substrate_idx])
        if not math.isfinite(raw):
            return 0.0
        return _signed_weight(raw, self.substrate_fn)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "layer_idx": self.layer_idx,
            "d_model": self.d_model,
            "substrate_idx": self.substrate_idx,
            "substrate_fn": self.substrate_fn,
            "is_derived": self.is_derived,
            "derived_at": self.derived_at,
            "source": self.source,
            "file_path": self.file_path,
            "requested_layer": self.requested_layer,
            "selected_layer": self.selected_layer,
            "selection_reason": self.selection_reason,
            "exact_layer_match": self.exact_layer_match,
            "extracted": self.extracted,
            "v_norm": float(np.linalg.norm(self.v)),
        }

# ── Steering Vector Library ────────────────────────────────────────────────────


class SteeringVectorLibrary(_ReadsACachedVector):
    """
    Derives, stores, and loads steering vectors for each affective dimension.

    DERIVATION METHOD: Contrastive Activation Addition (CAA)

    For each dimension, run the model on N positive/negative prompt pairs.
    At the target layer, record the last-token hidden state for each prompt.
    The steering vector = mean(positive_activations) - mean(negative_activations).

    This is the difference-in-means estimator from Zou et al. (2023) and
    Rimsky et al. (2024). It identifies the linear direction in activation
    space that most distinguishes the two conditions.

    Result: vectors in the same space as the residual stream that, when added,
    push the model's representations toward the positive condition.

    No training, no gradients, no labeled dataset beyond the prompt pairs above.
    Derived once per model and cached permanently after.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        source_dirs: list[Path] | None = None,
        expected_model_identity: dict[str, object] | None = None,
        allow_unbound_artifacts: bool = False,
        allow_derivation: bool = True,
    ):
        discovered_source_dirs: list[Path] = []
        env_dir = os.environ.get("AURA_STEERING_DIR")
        if env_dir and Path(env_dir).exists():
            discovered_source_dirs.append(Path(env_dir))

        extracted_dir = Path(__file__).parent.parent.parent / "training" / "vectors"
        if extracted_dir.exists() and (
            any(extracted_dir.glob("*.npy")) or any(extracted_dir.glob("*.npz"))
        ):
            discovered_source_dirs.append(extracted_dir)

        if cache_dir is None:
            try:
                from core.config import config as aura_config

                cache_dir = aura_config.paths.data_dir / "steering_vectors"
            except (ImportError, AttributeError, RuntimeError) as exc:
                _emit_affective_fault(
                    exc,
                    action="used user-scoped steering vector cache after config lookup failed",
                    severity="warning",
                    stage="library_cache_dir",
                )
                logger.debug("Steering vector cache config unavailable, using user cache: %s", exc)
                cache_dir = state_root() / "steering_vectors"

        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        raw_source_dirs = list(discovered_source_dirs if source_dirs is None else source_dirs)
        self._source_dirs = []
        for source_dir in raw_source_dirs:
            path = Path(source_dir)
            if path.exists() and path.resolve() != self._cache_dir.resolve():
                self._source_dirs.append(path)
        if self._source_dirs:
            logger.info(
                "🎯 Steering vectors: using runtime cache %s with %d compatible source dir(s)",
                self._cache_dir,
                len(self._source_dirs),
            )
        self._vectors: dict[str, SteeringVector] = {}
        self._vectors_by_layer: dict[int, dict[str, SteeringVector]] = {}
        self._registry = VectorRegistry()
        self._path_dim_cache: dict[str, int] = {}
        self._path_meta_cache: dict[str, dict[str, Any]] = {}
        self._expected_model_identity = dict(expected_model_identity or {})
        self._allow_unbound_artifacts = bool(allow_unbound_artifacts)
        self._allow_derivation = bool(allow_derivation)
        expected_digest = self._expected_model_identity.get("descriptor_sha256")
        if expected_digest is not None and (
            not isinstance(expected_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
        ):
            raise ValueError("steering_model_identity_invalid")
        self._source = self._infer_source()

    def _infer_source(self) -> str:
        """Best-effort provenance label for the active steering vector directory."""
        try:
            path = self._cache_dir.resolve()
            parts = set(path.parts)
            if "training" in parts and "vectors" in parts:
                return "extracted_caa"
            if "steering_vectors" in parts:
                return "cached_caa"
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _emit_affective_fault(
                exc,
                action="used configured CAA source label after vector cache path could not be resolved",
                severity="warning",
                stage="infer_source",
            )
            logger.debug("Steering vector source inference failed: %s", exc)
        return "configured_caa"

    def _register_vector(self, vector: SteeringVector) -> None:
        provenance = VectorProvenance(
            source=vector.source,
            file_path=vector.file_path,
            cache_dir=str(self._cache_dir),
            requested_layer=vector.requested_layer,
            selected_layer=vector.selected_layer,
            selection_reason=vector.selection_reason,
            derived_at=vector.derived_at,
            extracted=vector.extracted,
            exact_layer_match=vector.exact_layer_match,
        )
        self._registry.register(
            RegisteredVector(
                key=vector.key,
                layer_idx=vector.layer_idx,
                d_model=vector.d_model,
                v=vector.v,
                substrate_idx=vector.substrate_idx,
                substrate_fn=vector.substrate_fn,
                provenance=provenance,
            )
        )

    def _derive_or_fallback(
        self,
        *,
        model: Any,
        tokenizer: Any,
        dim_spec: dict[str, Any],
        target_layer: int,
        d_model: int,
    ) -> SteeringVector:
        expected_digest = str(self._expected_model_identity.get("descriptor_sha256") or "")
        if (
            re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
            and not self._allow_unbound_artifacts
        ):
            raise ValueError("steering_model_identity_unavailable")
        key = dim_spec["key"]
        cache_path = self._cache_dir / f"{key}_layer{target_layer}.npz"
        logger.info("🔬 Deriving steering vector: %s (layer %d)...", key, target_layer)
        try:
            vec = self._derive_caa(
                model=model,
                tokenizer=tokenizer,
                positive_prompts=dim_spec["positive"],
                negative_prompts=dim_spec["negative"],
                target_layer=target_layer,
                d_model=d_model,
            )
            derived_at = time.time()
            tmp_path = cache_path.with_suffix(".tmp.npz")
            np.savez(
                tmp_path,
                v=vec,
                derived_at=derived_at,
                source="runtime_derived_caa",
                requested_layer=target_layer,
                selected_layer=target_layer,
                selection_reason="runtime_derived",
                extracted=False,
                model_descriptor_sha256=expected_digest,
            )
            # Atomic commit to avoid partial files surviving a crash
            import shutil

            shutil.move(tmp_path, cache_path)
            return SteeringVector(
                key=key,
                layer_idx=target_layer,
                d_model=d_model,
                v=vec,
                substrate_idx=dim_spec["substrate_idx"],
                substrate_fn=dim_spec["substrate_fn"],
                is_derived=True,
                derived_at=derived_at,
                source="runtime_derived_caa",
                file_path=str(cache_path),
                requested_layer=target_layer,
                selected_layer=target_layer,
                selection_reason="runtime_derived",
                exact_layer_match=True,
                extracted=False,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            _emit_affective_fault(
                e,
                action="disabled this steering dimension with a neutral vector after CAA derivation failed",
                severity="degraded",
                stage="derive_vector",
                extra={"key": key, "target_layer": target_layer, "d_model": d_model},
            )
            logger.error("Failed to derive vector %s at layer %d: %s", key, target_layer, e)
            from core.evaluation.evidence_mode import require

            require(
                "steering_vector_derivation",
                False,
                f"vector {key} failed to derive from hidden states at layer {target_layer}: {e}",
            )
            neutral = np.zeros(d_model, dtype=np.float32)
            return SteeringVector(
                key=key,
                layer_idx=target_layer,
                d_model=d_model,
                v=neutral,
                substrate_idx=dim_spec["substrate_idx"],
                substrate_fn=dim_spec["substrate_fn"],
                is_derived=False,
                derived_at=time.time(),
                source="disabled_neutral",
                requested_layer=target_layer,
                selected_layer=target_layer,
                selection_reason="disabled_after_derivation_failure",
                exact_layer_match=False,
                extracted=False,
            )

    def load_or_derive(
        self,
        model,
        tokenizer,
        target_layers: list[int],
        d_model: int,
        force_rederive: bool = False,
    ) -> dict[int, dict[str, SteeringVector]]:
        """
        Load cached vectors if available, derive if not.

        This is the most expensive operation — runs once per model.
        A progress log is emitted while it runs.
        """
        loaded = 0
        derived = 0
        nearest = 0
        if (
            not self._expected_model_identity.get("descriptor_sha256")
            and not self._allow_unbound_artifacts
        ):
            raise ValueError("steering_model_identity_unavailable")
        self._registry.clear()
        self._vectors.clear()
        self._vectors_by_layer = {}

        for layer_idx in target_layers:
            self._vectors_by_layer[layer_idx] = {}
            for dim_spec in AFFECTIVE_DIMENSIONS:
                key = dim_spec["key"]
                vector: SteeringVector | None = None
                if not force_rederive:
                    cached = self._resolve_cached_path(key, layer_idx, d_model)
                    if cached is not None:
                        selected_layer, path, exact = cached
                        try:
                            vector = self._load_cached_vector(
                                key=key,
                                requested_layer=layer_idx,
                                selected_layer=selected_layer,
                                path=path,
                                d_model=d_model,
                                dim_spec=dim_spec,
                                exact_match=exact,
                            )
                            loaded += 1
                            nearest += 0 if exact else 1
                        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                            _emit_affective_fault(
                                e,
                                action="ignored invalid cached steering vector and attempted fresh derivation",
                                severity="warning",
                                stage="load_cached_vector",
                                extra={"key": key, "layer": layer_idx, "path": str(path)},
                            )
                            logger.warning(
                                "Failed to load cached vector %s at layer %d from %s: %s",
                                key,
                                layer_idx,
                                path.name,
                                e,
                            )
                if vector is None:
                    if not self._allow_derivation:
                        raise ValueError(
                            f"qualified_steering_vector_unavailable:{key}:layer={layer_idx}"
                        )
                    vector = self._derive_or_fallback(
                        model=model,
                        tokenizer=tokenizer,
                        dim_spec=dim_spec,
                        target_layer=layer_idx,
                        d_model=d_model,
                    )
                    if vector.source == "runtime_derived_caa":
                        derived += 1
                self._vectors_by_layer[layer_idx][key] = vector
                self._register_vector(vector)

        self._vectors = dict(self._vectors_by_layer.get(target_layers[0], {}))
        logger.info(
            "📚 SteeringVectorLibrary ready: %d loaded, %d derived, %d nearest-layer matches",
            loaded,
            derived,
            nearest,
        )
        return self._vectors_by_layer

    def _derive_caa(
        self,
        model,
        tokenizer,
        positive_prompts: list[str],
        negative_prompts: list[str],
        target_layer: int,
        d_model: int,
    ) -> np.ndarray:
        """
        Contrastive Activation Addition derivation.

        Runs each prompt through the model with a temporary capture hook,
        extracts the last-token hidden state at target_layer, averages
        positive and negative separately, returns their difference.

        The difference-in-means direction is the CAA steering vector.
        It requires no labels, no optimization, and no extra data —
        just the model and the contrastive prompt pairs defined above.
        """
        import mlx.core as mx

        pos_activations = []
        neg_activations = []

        def _extract_hidden_state_at_layer(prompt_text: str) -> np.ndarray | None:
            """Run prompt, extract last-token hidden state at target_layer."""
            captured = [None]

            # 1. Capture Original Class and target block
            layers = self._get_model_layers(model)
            if not layers or target_layer >= len(layers):
                logger.error("Layer %d out of range or not found", target_layer)
                return None
            target_block = layers[target_layer]
            original_class = target_block.__class__

            # 2. Define Dynamic Subclass for Capture
            class CapturingBlock(original_class):
                def __call__(self, x, *args, **kwargs):
                    # Call original implementation via super()
                    res = super().__call__(x, *args, **kwargs)
                    # result may be (hidden_states,) or just hidden_states
                    h = res[0] if isinstance(res, tuple) else res
                    if h is not None:
                        # Keep this in MLX until evaluation completes. NumPy
                        # cannot consume an MLX bfloat16 PEP 3118 buffer, so
                        # float32 is the stable host boundary for CAA statistics.
                        # Hidden shape: [batch, sequence, d_model]
                        captured[0] = h[0, -1, :].astype(mx.float32)  # [d_model]
                    return res

            # 3. Swap Class (Dynamic Subclassing Patch)
            target_block.__class__ = CapturingBlock

            try:
                tokens = tokenizer.encode(prompt_text)
                if hasattr(tokens, "input_ids"):
                    input_ids = tokens.input_ids
                else:
                    input_ids = tokens
                input_tensor = mx.array([input_ids])
                _ = model(input_tensor)
                if captured[0] is None:
                    mx.eval(_)
                else:
                    mx.eval(_, captured[0])
            except (RuntimeError, AttributeError, TypeError) as inner_e:
                _emit_affective_fault(
                    inner_e,
                    action="discarded failed prompt activation sample and continued CAA capture",
                    severity="warning",
                    stage="derive_caa_capture",
                )
                logger.debug("Capture failed for prompt: %s", inner_e)
            finally:
                # 4. Restore Original Class
                target_block.__class__ = original_class

            if captured[0] is None:
                return None
            return np.array(captured[0], dtype=np.float32, copy=True)

        # Collect positive activations
        for p in positive_prompts:
            h = _extract_hidden_state_at_layer(p)
            if h is not None and not np.any(np.isnan(h)):
                pos_activations.append(h)

        # Collect negative activations
        for p in negative_prompts:
            h = _extract_hidden_state_at_layer(p)
            if h is not None and not np.any(np.isnan(h)):
                neg_activations.append(h)

        if not pos_activations or not neg_activations:
            raise RuntimeError(
                "No valid activations collected — model may not support this extraction"
            )

        pos_mean = np.mean(pos_activations, axis=0)  # [d_model]
        neg_mean = np.mean(neg_activations, axis=0)  # [d_model]
        vec = pos_mean - neg_mean  # CAA direction

        # Normalize to unit vector (alpha controls magnitude)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm

        return vec.astype(np.float32)

    def _get_model_layers(self, model) -> list[Any] | None:
        """Helper to find the layers list in various MLX model structures."""
        view = resolve_model_layers(model)
        return view.layers if view is not None else None

    @property
    def vectors(self) -> dict[str, SteeringVector]:
        return self._vectors

    def get_vectors_for_layer(self, layer_idx: int) -> dict[str, SteeringVector]:
        return dict(self._vectors_by_layer.get(int(layer_idx), {}))

    @property
    def vectors_by_layer(self) -> dict[int, dict[str, SteeringVector]]:
        return {layer: dict(vectors) for layer, vectors in self._vectors_by_layer.items()}

    @property
    def registry(self) -> VectorRegistry:
        return self._registry

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def source(self) -> str:
        if not self._vectors_by_layer:
            return self._source
        all_vectors = [v for vectors in self._vectors_by_layer.values() for v in vectors.values()]
        if any(v.source in {"fallback_random", "disabled_neutral"} for v in all_vectors):
            return "mixed_with_disabled_vectors"
        sources = {v.source for v in all_vectors}
        if len(sources) == 1:
            return next(iter(sources))
        return "mixed"


# ── What a hook reads off its vectors ──────────────────────────────────────────


def vector_from_moods(moods: dict[str, float]) -> np.ndarray:
    """The 64-slot substrate vector for moods given as levels, laid out as the hook reads it."""
    latest = {str(key): float(value) for key, value in dict(moods or {}).items()}
    x = np.zeros(64, dtype=np.float32)
    x[0] = float(latest.get("valence", 0.0))
    x[1] = float(latest.get("arousal", 0.0))
    x[3] = float(latest.get("stress", 0.0))
    x[4] = float(latest.get("motivation", 0.0))
    x[5] = float(latest.get("energy", 0.0))
    # A mood given as levels has no usual to be read against, so its
    # valence level is its felt state (steering_channel.FELT_SLOT).
    from core.consciousness.steering_channel import FELT_SLOT

    x[FELT_SLOT] = float(latest.get("felt", latest.get("valence", 0.5)))
    return x


def is_fitted_basis(vectors: Any) -> bool:
    """Whether every one of a hook's vectors carries the reference, spread and push of a fitted basis."""
    vectors = list(vectors)
    return bool(vectors) and all(
        sv.reference is not None and sv.spread and sv.push is not None for sv in vectors
    )


def fitted_composite(vectors: Any, state: np.ndarray) -> np.ndarray | None:
    """The push a basis fitted to her own replies sets for this state, or None for any other basis.

    Each direction is weighed by how far her state is from the state the
    basis was fitted around, in units of how far she ordinarily moves
    there, through tanh, and keeps its fitted size: a state at her ordinary
    one pushes nothing and one a spread away pushes about three quarters of
    the fitted length. The unit-length composite gives the slightest
    deviation the full push, which took her own words away from her on 22
    October; and a reference of 0.5 on slots she holds near zero weighed
    every state she was in alike.
    """
    vectors = list(vectors)
    if not is_fitted_basis(vectors):
        return None
    out = np.zeros(vectors[0].d_model, dtype=np.float32)
    for sv in vectors:
        distance = (sv.compute_weight_from_state(state) - float(sv.reference)) / float(sv.spread)
        out += np.float32(math.tanh(distance) * float(sv.push)) * np.asarray(sv.v, dtype=np.float32)
    return out


def felt_everywhere(vectors: Any, state: np.ndarray) -> np.ndarray | None:
    """What her felt state writes at every position, before alpha, or None when no vector writes there."""
    from core.consciousness.mood_weight import signed_weight

    out = None
    for sv in vectors:
        if sv.everywhere is None or sv.everywhere_slot is None or not 0 <= sv.everywhere_slot < state.size:
            continue
        weight = signed_weight(float(state[sv.everywhere_slot]), "linear_half")
        push = np.asarray(sv.everywhere, dtype=np.float32) * np.float32(weight)
        out = push if out is None else out + push
    return out
