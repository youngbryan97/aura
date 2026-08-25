"""
core/consciousness/affective_steering.py
=========================================
AFFECTIVE STEERING ENGINE
Substrate-state → residual stream injection at inference time.

════════════════════════════════════════════════════════════════════════════════
THE ACTUAL MECHANISM — WHY THIS IS DIFFERENT
════════════════════════════════════════════════════════════════════════════════

Everything else Aura has works by converting internal state into *text*,
injecting that text into the context window, and letting the LLM read about
its own state. The LLM is informed. It isn't changed.

This file does something different. It intervenes in the forward pass itself.

During every token generation, the transformer runs a sequence of layers.
Each layer takes a hidden state vector h of shape [seq_len, d_model] and
returns a new hidden state. The residual stream is the accumulating sum:

    h₀ = token_embeddings
    h₁ = h₀ + attention_out(h₀)
    h₂ = h₁ + mlp_out(h₁)
    h₃ = h₂ + attention_out(h₂)
    ...
    logits = lm_head(h_final)

Activation steering (Turner et al. 2023; Zou et al. 2023; Rimsky et al. 2024)
adds a learned direction vector directly into completion-token positions of
this stream:

    h_l[completion] ← h_l[completion] + α · v_affect

where v_affect lives in the same space as h_l (d_model dimensions) and encodes
a specific affective direction. The model never "reads" this — the vector is
inside the math that produces the next token. It biases the probability
distribution at the level of hidden representations.

FOR AURA: The LiquidSubstrate's 64-neuron state vector is continuously
updated at 20Hz. The AffectiveSteeringEngine projects that state into a
linear combination of affective steering vectors and injects the sum into
the current completion position at target layers — making the substrate's state
physically continuous with every word Aura generates.

The response isn't colored by the substrate. The substrate IS part of the
computation that produces the response.

════════════════════════════════════════════════════════════════════════════════
STEERING VECTOR DERIVATION (no training data required)
════════════════════════════════════════════════════════════════════════════════

Steering vectors are derived using Contrastive Activation Addition (CAA):

    v = mean(h(positive_prompts)) − mean(h(negative_prompts))

For each affective dimension we run the model on ~20 contrastive prompt pairs
and average the difference in hidden states at the target layer. This gives us
a direction in activation space that corresponds to that affective quality.

These are computed ONCE, cached to disk, and reloaded on subsequent starts.
No gradient computation needed. No labeled dataset. Just the model itself.

════════════════════════════════════════════════════════════════════════════════
LAYER TARGETING
════════════════════════════════════════════════════════════════════════════════

Not all layers are equally effective for steering:

  - Early layers (0-5): too close to raw token embeddings, poor generalization
  - Middle layers (12-20 in a 32-layer model): best — high-level semantic
    representations have formed but generation hasn't been "decided" yet
  - Late layers (25+): too close to the output, steering is unstable

We target layers at approximately 40-65% depth.
For a 32-layer model: layers 13-21.
For a 28-layer model: layers 11-18.
Dynamic based on loaded model.

════════════════════════════════════════════════════════════════════════════════
INTEGRATION WITH AURA'S MLX CLIENT
════════════════════════════════════════════════════════════════════════════════

Aura uses core/brain/llm/mlx_client.py for local inference.
The AffectiveSteeringEngine patches the loaded model's transformer layers
by wrapping their __call__ methods. This is done once after model load
and persists for the lifetime of the process.

Integration:

    # In mlx_client.py, after model load:
    from core.consciousness.affective_steering import get_steering_engine

    engine = get_steering_engine()
    engine.attach(model, tokenizer, model_path=loaded_model_path)
    engine.start_substrate_sync() # starts reading from LiquidSubstrate

    # From that point: every inference call is steered by the substrate.
    # No other code changes needed anywhere.

════════════════════════════════════════════════════════════════════════════════
"""

import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.consciousness.caa import ProductionCAA, RegisteredVector, VectorProvenance, VectorRegistry
from core.runtime.errors import FallbackClassification, record_degradation
from core.runtime.model_layers import resolve_model_layers
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.AffectiveSteering")


#: Completion-position masks, keyed by (shape, dtype-name).
#
# The mask is a constant for a given shape: zeros with a 1.0 at the final
# position. It was being rebuilt from a fresh NumPy allocation and re-uploaded
# to the GPU on EVERY forward pass of EVERY steered block — 64 layers per token,
# with the allocation sized by the sequence length. During prefill of a few
# thousand characters that is thousands of host allocations and uploads before
# the first token can exist, which is why the desktop surface saw
# "livelocked: heartbeats but zero tokens" on longer turns while short ones
# answered fine (2026-07-26).
#
# Bounded so a pathological spread of shapes cannot grow without limit.
_COMPLETION_MASK_CACHE: dict[tuple[tuple[int, ...], str], Any] = {}
_COMPLETION_MASK_CACHE_MAX = 32


def _emit_affective_fault(
    error: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    stage: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    """Record an affective-steering fault with an explicit runtime action.

    Several recovery-path tests monkeypatch ``record_degradation`` with the
    historical two-argument shape.  The fallback keeps those visibility tests
    meaningful while production receives structured receipts.
    """
    metadata = dict(extra or {})
    if stage:
        metadata["stage"] = stage
    try:
        record_degradation(
            "affective_steering",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=metadata or None,
        )
    except TypeError:
        record_degradation("affective_steering", error)


# ── Steering Coefficient ───────────────────────────────────────────────────────
# How strongly the substrate influences generation.
# Too low: no effect. Too high: incoherence. Sweet spot validated empirically:
# α ∈ [8, 25] for most LLMs (Turner et al. 2023).
# Aura's substrate is continuous and low-amplitude — we use a conservative 5.
#
# Tuning history:
#   - α=15 (original): clear mode collapse — "the drift — the drift — the
#     drift" pure repetition on hobby/specific-recall prompts, "I'm not sure
#     what the math says" reflexes on warm-baseline.
#   - α=8 (2026-04-27 first reduction): dramatically better on most prompts,
#     but specific introspective prompts ("what do you suspect about yourself
#     you can't prove?") still mode-collapsed: "Something is shifting.
#     Something is moving." × 60+ repetitions, even with anti-repetition
#     system prompt firing.
#   - α=5 (2026-04-27 second reduction): low end of the validated range.
#     Substrate signal still observable in responses but with even less risk
#     of dominating generation on uncertainty-themed prompts.
#
# Conservative α is the right operating point while production CAA artifacts
# are validated by `training/caa_32b_validation.py`.
#: Injection magnitude as a FRACTION of the residual-stream norm (see
#: AffectiveSteeringHook._residual_reference_scale). Not an absolute number of
#: units any more, so this value means the same thing on a 1.5B and a 32B.
#:
#: 0.2 is measured, not chosen. Sweeping the fraction on both models: a 1.5B
#: first changes its output at 0.2 and degenerates by 0.8; a 32B first changes
#: at 0.05 and is still coherent at 0.8. 0.2 is the smallest value that clears
#: the threshold on BOTH, which is the point of expressing it as a fraction —
#: under the old absolute scale the same job needed 10 on the 1.5B and 150 on
#: the 32B, and the shipped 5.0 (clipped to 3.0) cleared neither.
DEFAULT_ALPHA = 0.2

# Fraction of model depth to target (lower bound, upper bound)
TARGET_LAYER_RANGE = (0.40, 0.65)

# How often to re-read the substrate state (seconds)
SUBSTRATE_SYNC_INTERVAL_S = 0.05  # 20 Hz — matches LiquidSubstrate's update rate


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
        raw = float(moods.get(mood_key, 0.0))

        # Adaptive mood coefficients are typically in [-1, 1], so we can just use them
        # as weights (optionally scaled or clipped if needed).
        if self.substrate_fn == "tanh":
            return float(np.tanh(raw))
        elif self.substrate_fn == "linear_half":
            return float(np.clip(raw, -1.0, 1.0))
        else:
            return float(np.tanh(raw))

    def compute_weight_from_state(self, substrate_x: np.ndarray) -> float:
        """Map the live substrate vector index directly to a steering weight."""
        if substrate_x is None or self.substrate_idx < 0 or self.substrate_idx >= len(substrate_x):
            return 0.0
        raw = float(substrate_x[self.substrate_idx])
        if not math.isfinite(raw):
            return 0.0
        if self.substrate_fn == "linear_half":
            return float(np.clip(raw, -1.0, 1.0))
        return float(np.tanh(raw))

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


class SteeringVectorLibrary:
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
    Computation time: ~2-5 minutes per dimension. Cached permanently after.
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
        A progress log is emitted; derivation takes ~1-3 minutes on M5 Pro.
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


# ── The Steering Hook ──────────────────────────────────────────────────────────


class AffectiveSteeringHook:
    """
    Wraps a transformer block's __call__ to inject affective steering vectors.

    This is the core mechanism. When installed, every call to the transformer
    block at the target layer includes the affective addition:

        h_out = original_call(h_in, ...)
        h_out = h_out + α * Σᵢ wᵢ(substrate) · vᵢ

    where:
        α  = DEFAULT_ALPHA (global steering strength)
        wᵢ = weight of dimension i, derived from substrate state
        vᵢ = steering vector for dimension i

    The injection is position-masked. During prompt prefill, only the final
    token representation is steered; during autoregressive decoding, the
    current generated token is the final token. This avoids adding affective
    offsets to padding, EOS, and static system-prompt positions.

    This runs on EVERY TOKEN GENERATED. The substrate state is read from
    a shared variable that the SubstrateSyncThread updates at 20Hz.

    The affect is not described. It is the math.
    """

    def __init__(
        self,
        block,
        layer_idx: int,
        vectors: dict[str, SteeringVector],
        alpha: float = DEFAULT_ALPHA,
    ):
        self._block = block
        self._layer_idx = layer_idx
        self._vectors = vectors
        self._alpha = alpha
        self._installed = False

        # Shared substrate state (updated by SubstrateSyncThread)
        self._substrate_x: np.ndarray | None = None
        self._latest_moods: dict[str, float] = {}
        self._substrate_lock = threading.Lock()
        # Freshness stamp: 0.0 = never synced. The injection path derates
        # alpha when the sync thread stops feeding us (see _effective_alpha).
        self._last_substrate_sync_monotonic = 0.0

        # Active flag
        self._active = True

        # Diagnostic counters
        self._inject_count = 0
        #: Set by the worker when it is running out-of-process. When present,
        #: Grassmann states go across this ring instead of to a PhiCore that
        #: does not exist on this side of the fork.
        self._phi_residual_channel = None
        # Why this channel is silent, when it is silent. `grassmann_states: 0`
        # in the health report could mean the hook was never called, the
        # encoder is still filling its window, the encoder is failing on every
        # call, or nothing is draining — and the four were indistinguishable,
        # because the encoder swallowed its own exceptions and said nothing.
        self._phi_sampled = 0
        self._phi_encoded_none = 0
        self._phi_published = 0
        self._phi_encode_errors = 0
        self._phi_last_error = ""
        self._grassmann_encoder = None
        try:
            self._phi_sample_every = max(1, int(os.getenv("AURA_PHI_RESIDUAL_SAMPLE_EVERY", "32")))
        except (TypeError, ValueError):
            self._phi_sample_every = 32
        self._last_injection_norm = 0.0
        self._last_effective_alpha = 0.0
        self._last_mask_mode = "none"

        # [OPTIMIZATION] Cached composite vector to avoid redundant MLX uploads
        self._cached_composite_mx: Any = None
        self._last_composite_np: np.ndarray | None = None
        self._cached_substrate_hash: int = 0

    # Hard ceiling on runtime injection strength. The governor already clips
    # its computed alpha to 3.0; this enforces the same bound AT THE POINT OF
    # INJECTION so no configuration path (install-time DEFAULT_ALPHA=5.0, a
    # stalled sync thread, a bad env override) can steer hotter than the
    # governor is ever allowed to ask for.
    # Below the measured degeneration point: a 1.5B produces word salad at
    # fraction 0.8, so the ceiling sits well under it.
    _INJECTION_ALPHA_CEILING = 0.6
    # If the substrate sync stops feeding this hook (thread died, mood lookups
    # failing), steering must derate instead of freezing at its last-hot
    # value: stale affect is noise, and hot noise is exactly the spliced
    # off-distribution decode observed live.
    # Was 2.0, which a single generation exceeds. A 32B emitting 40 tokens
    # takes far longer than two seconds, so steering derated to the stale-safe
    # floor partway through EVERY generation and the configured magnitude
    # stopped applying — measured, and it is why steered and unsteered output
    # were byte-identical in the first ablation runs. Mid-generation is not
    # staleness; a dead sync thread is, and 120s still catches that.
    _SYNC_STALE_AFTER_S = 120.0
    # Half the default fraction: a real reduction in influence when the state
    # really is old, without the silent collapse to nothing that the previous
    # absolute 0.35 produced once alpha became a fraction.
    _STALE_SAFE_ALPHA = 0.1

    def _effective_alpha(self) -> float:
        try:
            alpha = float(self._alpha)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(alpha) or alpha <= 0.0:
            return 0.0
        alpha = min(alpha, self._INJECTION_ALPHA_CEILING)
        last_sync = self._last_substrate_sync_monotonic
        if last_sync <= 0.0 or (time.monotonic() - last_sync) > self._SYNC_STALE_AFTER_S:
            alpha = min(alpha, self._STALE_SAFE_ALPHA)
        return alpha

    @staticmethod
    def _neutral_reference_state() -> np.ndarray:
        """The substrate vector for "no particular affect".

        Every mood dimension in `_vector_from_moods` is a 0..1 activation, so
        the midpoint is 0.5 and neutral is 0.5 everywhere those dimensions are
        read. Nothing here is tuned: it is the centre of the declared range,
        and it is a reference point rather than a magnitude, so it does not
        introduce a constant anyone has to justify.
        """
        neutral = np.zeros(64, dtype=np.float32)
        for index in (0, 1, 3, 4, 5):
            neutral[index] = 0.5
        return neutral

    @staticmethod
    def _vector_from_moods(moods: dict[str, float]) -> np.ndarray:
        latest = {str(key): float(value) for key, value in dict(moods or {}).items()}
        x = np.zeros(64, dtype=np.float32)
        x[0] = float(latest.get("valence", 0.0))
        x[1] = float(latest.get("arousal", 0.0))
        x[3] = float(latest.get("stress", 0.0))
        x[4] = float(latest.get("motivation", 0.0))
        x[5] = float(latest.get("energy", 0.0))
        return x

    def update_substrate(self, moods: dict[str, float]):
        """Called by SubstrateSyncThread at ~20Hz when only mood telemetry exists."""
        self.update_substrate_vector(
            self._vector_from_moods(moods),
            moods=moods,
            source="live_mood_projection",
        )

    def update_substrate_vector(
        self,
        substrate_x: np.ndarray,
        *,
        moods: dict[str, float] | None = None,
        source: str = "live_substrate_vector",
    ):
        """Called by SubstrateSyncThread at ~20Hz with the direct substrate vector."""
        import mlx.core as mx

        state = np.asarray(substrate_x, dtype=np.float32).reshape(-1)
        if len(state) == 0 or not np.isfinite(state).all():
            # Substituting zeros here is not neutral — it is "she feels exactly
            # nothing", derived from a state that was actually broken, and it
            # used to happen without a word. The stand-down is the right action;
            # doing it silently is not.
            from core.consciousness.steering_admission import Admission, refuse

            refuse(
                Admission(
                    False,
                    ("non_finite_substrate_state" if len(state) else "empty_substrate_state",),
                    {"source": source, "length": int(len(state))},
                ),
                subsystem="affective_steering",
                action="stood steering down; no affective direction is derivable from this state",
            )
            state = np.zeros(64, dtype=np.float32)
        with self._substrate_lock:
            self._last_substrate_sync_monotonic = time.monotonic()
            self._latest_moods = {
                str(key): float(value) for key, value in dict(moods or {}).items()
            }
            self._latest_moods["_source"] = source
            self._substrate_x = state.copy()

            # 2. PRE-COMPUTE COMPOSITE ON CPU/NP (Background Thread)
            # This moves the O(dims * d_model) work out of the inference hook.
            target_composite_np = np.zeros(
                self._vectors[next(iter(self._vectors))].d_model, dtype=np.float32
            )
            active = False

            # DEVIATION FROM NEUTRAL, not absolute level.
            #
            # Measured 2026-08-06: two OPPOSING states (valence 0.9 vs 0.1)
            # converged to composite cosine +0.6285 — pointing the same way, so
            # no metric could show the state's content mattered. The derived
            # vectors were fine (full rank 5, valence_positive . frustration =
            # -0.50); the defect was here. Weights were ABSOLUTE activations, so
            # a dimension equal in both states (arousal, 0.7 in both) added an
            # identical term to each composite and that common-mode dominated
            # the unit-normalised result while carrying no state information.
            # Centred on neutral the same two states reach -0.8437.
            #
            # It also fixes the boundary condition: neutral affect now stands
            # steering down instead of injecting a constant meaningless
            # direction.
            neutral = self._neutral_reference_state()
            for sv in self._vectors.values():
                weight = sv.compute_weight_from_state(state) - sv.compute_weight_from_state(neutral)
                if abs(weight) > 0.05:
                    target_composite_np += weight * sv.v
                    active = True

            if active:
                # Normalization
                norm = np.linalg.norm(target_composite_np)
                if norm > 1e-8:
                    target_composite_np /= norm

            # Tier 2 Hardening: Exponential Smoothing (Lerp) to prevent Affective Jitter
            momentum = 0.85
            if self._last_composite_np is not None:
                composite_np = (momentum * self._last_composite_np) + (
                    (1.0 - momentum) * target_composite_np
                )
            else:
                composite_np = target_composite_np

            # Update MLX array ONCE per substrate tick if magnitude is non-zero.
            #
            # `current_norm < 1e-4` was the ONLY guard here, and it is False for
            # NaN. If any loaded steering vector contains a NaN then `norm` is
            # NaN, dividing by it makes every element NaN, this test passes, and
            # a NaN composite is added to the hidden states of all 64 blocks on
            # every token — generation destroyed, from a file on disk under
            # data/steering_vectors/. The admission gate below is the last thing
            # between that file and the residual stream, so it also rejects the
            # other shape that does damage: a technically-unit vector with all
            # its mass in one coordinate, which saturates one feature of the
            # residual stream rather than pointing anywhere in it.
            from core.consciousness.steering_admission import (
                admit_steering_vector,
                refuse,
            )

            admission = admit_steering_vector(composite_np)
            current_norm = np.linalg.norm(composite_np)
            if admission.rejected:
                refuse(
                    admission,
                    subsystem="affective_steering",
                    action="stood steering down for this tick rather than inject it",
                )
                self._last_composite_np = None
                self._cached_composite_mx = None
            elif current_norm < 1e-4:
                self._last_composite_np = None
                self._cached_composite_mx = None
            else:
                self._last_composite_np = composite_np.copy()
                self._cached_composite_mx = mx.array(composite_np)

                # Tier 3 Hardening: Zero-Copy MLX explicit evaluation
                # This prevents lazy evaluation from stalling the main generation thread
                mx.eval(self._cached_composite_mx)

    def current_composite_vector(self) -> Any | None:
        """The real composite as numpy, or None when steering stood down.

        Exists so an ablation can obtain the vector the system actually built
        without reading a private attribute. A harness that reaches into
        `_last_composite_np` silently starts measuring something else the day
        that attribute changes meaning.
        """
        with self._substrate_lock:
            composite = self._last_composite_np
            return None if composite is None else composite.copy()

    def override_composite_vector(self, vector: Any | None) -> None:
        """Force the injected vector. FOR ABLATION ONLY.

        A steering ablation has to answer a counterfactual — what would this
        model have said under a DIFFERENT vector of the same magnitude — and
        that question cannot be asked through `update_substrate()`, which
        derives the vector from the state rather than accepting one.

        Passing None clears the override and restores substrate-derived
        steering. The override is not persisted and does not survive a process,
        deliberately: a knob that can pin live steering to a fixed direction is
        exactly the kind of thing that gets left set after an experiment, so
        the only way to hold it is to keep re-asserting it in-process.

        The vector still passes through the same admission gate as any other on
        the next substrate update; this sets what is cached now, it does not
        grant an exemption from `admit_steering_vector`.
        """
        import numpy as np

        with self._substrate_lock:
            if vector is None:
                self._last_composite_np = None
                self._cached_composite_mx = None
                return
            array = np.asarray(vector, dtype=np.float32)
            self._last_composite_np = array.copy()
            try:
                import mlx.core as mx

                self._cached_composite_mx = mx.array(array)
                mx.eval(self._cached_composite_mx)
            except ImportError:
                # No MLX here — the numpy side is still the record of what was
                # asked for, and a caller without MLX cannot inject anyway.
                self._cached_composite_mx = None

    def compute_composite_vector_mx(self, dtype=None) -> Any | None:
        """
        [ZERO-COST] Return the pre-computed MLX array from the background sync.
        """
        import mlx.core as mx

        with self._substrate_lock:
            composite = self._cached_composite_mx
            if composite is not None and dtype is not None and composite.dtype != dtype:
                # Casting is fast, but we avoid re-uploading
                return mx.astype(composite, dtype)
            return composite

    def _residual_reference_scale(self, h: Any) -> float:
        """Mean per-token norm of the residual stream this block is emitting.

        The reference alpha is measured against. Computed from the activation
        itself rather than from a table of model sizes, so a model this code
        has never seen gets the right scale without anyone editing a constant.

        Falls back to 1.0 — the previous absolute behaviour — if the norm
        cannot be taken. That is the conservative direction: it under-steers
        rather than injecting an unbounded multiple into the stream.
        """
        try:
            import mlx.core as mx

            magnitude = float(mx.mean(mx.linalg.norm(h, axis=-1)))
        except (AttributeError, ImportError, TypeError, ValueError, ZeroDivisionError):
            return 1.0
        if not math.isfinite(magnitude) or magnitude <= 0.0:
            return 1.0
        return magnitude

    def _completion_position_mask(self, h: Any) -> Any | None:
        """Return a broadcast mask for the completion/current token position."""
        try:
            import mlx.core as mx

            shape = tuple(getattr(h, "shape", ()) or ())
            is_2d = len(shape) == 2 and shape[0] > 1
            is_3d = len(shape) == 3 and shape[1] > 1
            if is_2d or is_3d:
                # One mask per (shape, dtype), shared across all 64 blocks and
                # reused for every token of the same width. Rebuilding it per
                # forward pass cost a host allocation and a GPU upload sized by
                # the sequence length, 64 times per token — the first-token
                # stall on longer prompts.
                self._last_mask_mode = "last_position_2d" if is_2d else "last_position_3d"
                cache_key = (shape, str(h.dtype))
                cached = _COMPLETION_MASK_CACHE.get(cache_key)
                if cached is not None:
                    return cached
                if is_2d:
                    mask_np = np.zeros((shape[0], 1), dtype=np.float32)
                    mask_np[-1, 0] = 1.0
                else:
                    mask_np = np.zeros((shape[0], shape[1], 1), dtype=np.float32)
                    mask_np[:, -1, 0] = 1.0
                mask = mx.astype(mx.array(mask_np), h.dtype)
                if len(_COMPLETION_MASK_CACHE) >= _COMPLETION_MASK_CACHE_MAX:
                    _COMPLETION_MASK_CACHE.clear()
                _COMPLETION_MASK_CACHE[cache_key] = mask
                return mask
            self._last_mask_mode = "single_token"
        except (ImportError, AttributeError, RuntimeError) as exc:
            _emit_affective_fault(
                exc,
                action="continued steering without completion-position mask for this token",
                severity="warning",
                stage="completion_position_mask",
                extra={"layer_idx": self._layer_idx},
            )
            self._last_mask_mode = f"mask_unavailable:{type(exc).__name__}"
        return None

    def _maybe_record_phi_residual(self, h: Any) -> None:
        if os.getenv("AURA_PHI_RECORD_RESIDUALS", "1").strip().lower() in {
            "0",
            "false",
            "off",
            "no",
        }:
            return
        if self._inject_count % self._phi_sample_every != 0:
            return
        # PhiCore does `np.asarray(hidden_state)`, which on MLX is a blocking
        # device sync AND a full materialisation of whatever it is handed. This
        # hook runs inside the forward pass of all 64 blocks, so measuring here
        # collapses MLX's lazy pipeline on the one path where latency decides
        # whether a turn survives.
        #
        # During PREFILL `h` is the whole sequence — [1, seq, 5120] — so a
        # single sample copied tens of megabytes off the GPU and stalled the
        # graph, repeatedly. Measured live 2026-07-26: ~3k-token prompts took
        # 58-82s to a first token, roughly 50 tok/s, about twenty times slower
        # than this model should prefill; turns 5-7 of a conversation died on
        # that alone.
        #
        # Prefill is not a thought moment anyway — the signal Φ wants is the
        # per-token dynamics of generation. So sample only single-token decode
        # steps, and hand over one already-sliced position rather than a
        # sequence, so the transfer is a 5120-float vector instead of a tensor.
        try:
            shape = tuple(getattr(h, "shape", ()) or ())
        except (AttributeError, TypeError):
            return
        if len(shape) >= 3 and shape[-2] > 1:
            return
        if len(shape) == 2 and shape[0] > 1:
            return
        try:
            sample = h[0, -1, :] if len(shape) >= 3 else h

            # IN THE WORKER PROCESS there is no PhiCore to hand this to — the
            # hook runs inside the MLX worker subprocess and PhiCore is
            # registered in the main runtime, so the container lookup below
            # returned False on every token and the activation-grounded complex
            # never filled. Encode here, where the activations are, and publish
            # the 8-bit state across the boundary.
            channel = getattr(self, "_phi_residual_channel", None)
            if channel is not None:
                self._phi_sampled += 1
                state = self._encode_grassmann_state(sample)
                if state is not None:
                    from core.consciousness.phi_residual_channel import publish_state

                    publish_state(channel, state)
                    self._phi_published += 1
                    return
                self._phi_encoded_none += 1
                return

            from core.container import ServiceContainer

            if not ServiceContainer.has("phi_core"):
                return
            phi_core = ServiceContainer.get("phi_core", default=None)
            if phi_core is not None and hasattr(phi_core, "record_residual_stream"):
                phi_core.record_residual_stream(
                    sample, layer_idx=self._layer_idx, token_position=-1
                )
        except (ImportError, AttributeError, RuntimeError) as exc:
            _emit_affective_fault(
                exc,
                action="continued generation after optional phi residual sample failed",
                severity="warning",
                stage="phi_residual_sample",
                extra={"layer_idx": self._layer_idx},
            )
            logger.debug("Residual phi sample failed at layer %d: %s", self._layer_idx, exc)

    def _encode_grassmann_state(self, sample: Any) -> int | None:
        """Reduce a residual vector to the 8-bit state Φ's TPM is built from.

        Done HERE rather than in the parent because the encoder is what makes
        this cheap to ship: ~5120 floats in, one byte out. Sending the vector
        instead would put a per-token megabyte across the process boundary on
        the path where latency decides whether a turn survives.
        """
        try:
            if self._grassmann_encoder is None:
                from core.consciousness.grassmann_phi import GrassmannResidualComplex
                from core.consciousness.phi_core import _grassmann_anchor_count

                self._grassmann_encoder = GrassmannResidualComplex(
                    n_anchors=_grassmann_anchor_count()
                )
            import numpy as _np

            vector = _np.asarray(sample, dtype=_np.float32).reshape(-1)
            state = self._grassmann_encoder.observe(vector)
            if state is None:
                return None
            # Fold rather than truncate: `& 0xFF` would keep modes 0-7 and drop
            # everything above, so a wider encoder would subtract information.
            from core.consciousness.phi_core import _fold_modes_to_byte

            return _fold_modes_to_byte(int(state))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            # A telemetry sample is never worth a generation, so this still
            # fails open. What it must not do is fail SILENTLY: an encoder
            # raising on every call looks exactly like an encoder that is never
            # reached, and both report zero states. Recorded once — this runs
            # inside the forward pass, and a per-token degradation record would
            # cost more than the sample it describes.
            self._phi_encode_errors += 1
            self._phi_last_error = f"{type(exc).__name__}: {exc}"[:200]
            if self._phi_encode_errors == 1:
                from core.runtime.errors import record_degradation

                record_degradation(
                    "affective_steering.phi_residual",
                    exc,
                    severity="warning",
                    action=(
                        "the Grassmann encoder refused a residual sample; the "
                        "activation-grounded complex will not fill while this "
                        "persists"
                    ),
                )
            return None

    def install(self):
        """
        Patch the transformer block's forward pass to inject the steering vector.

        Uses dynamic subclassing to ensure the interception is reliable.
        """
        if self._installed:
            return

        block = self._block
        hook = self  # capture self

        # Store original method
        target_name = "forward" if hasattr(block, "forward") else "__call__"
        original_method = getattr(block, target_name)

        def steered_call(*args, **kwargs):
            # Run original forward pass
            result = original_method(*args, **kwargs)

            if not hook._active:
                return result

            try:
                # Extract hidden states from result
                if isinstance(result, tuple):
                    h = result[0]
                    rest = result[1:]
                else:
                    h = result
                    rest = None

                # Ceiling + staleness derating happen HERE, at injection, not
                # just in the sync thread that may have died. Evaluated BEFORE
                # the composite: when steering has stood down there is nothing
                # to add, and fetching + dtype-casting the vector on all 64
                # blocks of every token to then multiply it by zero is pure
                # cost on the path that decides whether a first token arrives.
                effective_alpha = hook._effective_alpha()
                composite = (
                    hook.compute_composite_vector_mx(dtype=h.dtype)
                    if effective_alpha > 0.0
                    else None
                )

                if composite is not None:
                    if effective_alpha > 0.0:
                        # Scale the injection to the residual stream it is
                        # entering, so alpha means "this fraction of the
                        # activation" rather than an absolute number of units.
                        #
                        # Absolute alpha does not survive a change of model.
                        # The composite is unit-norm, so a fixed alpha is a
                        # fixed-size nudge into a stream whose magnitude grows
                        # with width and depth. Measured 2026-08-06: a 1.5B
                        # first changes its output near magnitude 10, a 32B
                        # needs roughly 10-300, and the SHIPPED effective alpha
                        # is 3.0 — below the threshold on both, which is why
                        # steered and unsteered output were byte-identical.
                        #
                        # Referenced to the stream, one setting means the same
                        # thing everywhere and the constant stops being a
                        # per-model guess.
                        scale = hook._residual_reference_scale(h)
                        mask = hook._completion_position_mask(h)
                        if mask is not None:
                            h = h + (mask * (effective_alpha * scale) * composite)
                        else:
                            h = h + (effective_alpha * scale) * composite

                    # Diagnostic
                    hook._inject_count += 1
                    hook._last_effective_alpha = effective_alpha
                    # Norm is expensive, so only occasionally — and it is an
                    # OBSERVATION that must never void the injection above. It
                    # could, and did: the call was `mx.norm`, which does not
                    # exist in this MLX version (`mx.linalg.norm` does). Every
                    # 50th injection raised AttributeError, the enclosing
                    # handler returned the original block output, and the
                    # steering already applied on the line above was discarded
                    # with it. A statistic that destroys the effect it measures
                    # is worse than no statistic.
                    if hook._inject_count % 50 == 0:
                        try:
                            import mlx.core as mx

                            hook._last_injection_norm = (
                                float(mx.linalg.norm(composite)) * effective_alpha
                            )
                        except (AttributeError, ImportError, TypeError, ValueError):
                            # Losing a diagnostic is survivable. Losing the
                            # injection is not, so this swallows deliberately
                            # and leaves the previous reading in place.
                            hook._last_injection_norm = None

                # φ's ACTIVATION GROUNDING USED TO DEPEND ON STEERING FIRING.
                # This sample sat inside `if composite is not None`, so whenever
                # steering stood down — no composite, alpha derated to zero, the
                # substrate sync not yet warm — the residual stream was never
                # recorded and PhiCore's Grassmann complex stayed empty.
                #
                # Measured live 2026-08-04 with the corrected estimator wired
                # in: "reporting a state_summary measurement because
                # better-grounded estimators could not run:
                # residual_stream_grassmann (insufficient_history:0/50)". Zero.
                # Not "not enough yet" — none, ever, on a boot that answered
                # four turns with three steering hooks installed.
                #
                # Whether the model's representation is integrated is not a
                # question about whether we happen to be steering it. The
                # sample belongs to the forward pass, not to the injection.
                hook._maybe_record_phi_residual(h)

                if rest is not None:
                    return (h,) + rest
                return h

            except (ImportError, AttributeError, RuntimeError) as e:
                _emit_affective_fault(
                    e,
                    action="returned original block output after steering injection failed",
                    severity="degraded",
                    stage="steering_injection",
                    extra={"layer_idx": hook._layer_idx},
                )
                logger.debug("Steering injection failed at layer %d: %s", hook._layer_idx, e)
                return result

        # Use dynamic subclassing to ensure interception
        class SteeredBlock(block.__class__):  # type: ignore
            __module__ = block.__class__.__module__

        # Override the target method
        setattr(
            SteeredBlock, target_name, lambda self, *args, **kwargs: steered_call(*args, **kwargs)
        )

        block.__class__ = SteeredBlock
        self._installed = True
        logger.info(
            "🎯 Steering hook installed at layer %d (alpha=%.1f, %d vectors via %s)",
            self._layer_idx,
            self._alpha,
            len(self._vectors),
            target_name,
        )

    def uninstall(self):
        """Remove the hook and restore original behavior."""
        # Python's method patching: difficult to perfectly uninstall
        # Best approach: disable via flag
        self._active = False
        logger.info("🔕 Steering hook disabled at layer %d", self._layer_idx)

    def get_diagnostics(self) -> dict[str, Any]:
        with self._substrate_lock:
            x = self._substrate_x
            moods = dict(self._latest_moods)
        return {
            "layer_idx": self._layer_idx,
            "installed": self._installed,
            "active": self._active,
            "inject_count": self._inject_count,
            "last_injection_norm": round(self._last_injection_norm, 4),
            "last_mask_mode": self._last_mask_mode,
            "substrate_connected": x is not None,
            "substrate_source": str(moods.get("_source", "")) if moods else None,
            "substrate_valence": round(float(moods.get("valence", 0.0)), 3) if moods else None,
            "substrate_arousal": round(float(moods.get("arousal", 0.0)), 3) if moods else None,
            "vector_sources": {key: vector.source for key, vector in self._vectors.items()},
            "phi_residual": {
                "channel_attached": self._phi_residual_channel is not None,
                "sampled": self._phi_sampled,
                "published": self._phi_published,
                "encoder_withheld": self._phi_encoded_none,
                "encoder_errors": self._phi_encode_errors,
                "last_error": self._phi_last_error,
                "sample_every": self._phi_sample_every,
            },
        }


# ── Substrate Sync Thread ──────────────────────────────────────────────────────


class SubstrateSyncThread:
    """
    Continuously reads from LiquidSubstrate and pushes state to all hooks.

    Runs in a daemon thread at SUBSTRATE_SYNC_INTERVAL_S (20Hz).
    This is the live coupling: substrate physics → hook state → residual stream.

    The thread is intentionally minimal — it just reads x from the substrate
    and calls update_substrate() on each hook. No computation here.
    """

    def __init__(self, hooks: list[AffectiveSteeringHook], engine: Any, shared_state: Any = None):
        self._hooks = hooks
        self._engine = engine
        self._shared_state = shared_state
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            name="SubstrateSyncThread",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "🔄 SubstrateSyncThread started (%d hooks, shared=%s)",
            len(self._hooks),
            self._shared_state is not None,
        )

    def stop(self):
        self._running = False

    @staticmethod
    def _coerce_state_vector(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        try:
            vector = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError, RuntimeError):
            return None
        if len(vector) == 0 or not np.isfinite(vector).all():
            return None
        return vector

    def _read_substrate_vector(self) -> tuple[np.ndarray | None, str]:
        candidates: list[tuple[str, Any]] = []
        if self._shared_state is not None:
            candidates.append(("shared_state", self._shared_state))
        try:
            from core.container import ServiceContainer

            candidates.extend(
                [
                    ("liquid_substrate", ServiceContainer.get("liquid_substrate", default=None)),
                    (
                        "conscious_substrate",
                        ServiceContainer.get("conscious_substrate", default=None),
                    ),
                    ("liquid_state", ServiceContainer.get("liquid_state", default=None)),
                ]
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            _emit_affective_fault(
                exc,
                action="continued substrate sync with explicit shared state only after ServiceContainer lookup failed",
                severity="warning",
                stage="substrate_sync_vector_lookup",
            )

        for source, substrate in candidates:
            if substrate is None:
                continue
            try:
                if hasattr(substrate, "get_state_vector"):
                    vector = self._coerce_state_vector(substrate.get_state_vector())
                    if vector is not None:
                        return vector, source
                if hasattr(substrate, "x"):
                    lock = getattr(substrate, "sync_lock", None)
                    if lock is not None:
                        with lock:
                            vector = self._coerce_state_vector(getattr(substrate, "x", None))
                    else:
                        vector = self._coerce_state_vector(getattr(substrate, "x", None))
                    if vector is not None:
                        return vector, source
                if isinstance(substrate, dict):
                    raw_vector = substrate.get("state_vector")
                    if raw_vector is None:
                        raw_vector = substrate.get("x")
                    vector = self._coerce_state_vector(raw_vector)
                    if vector is not None:
                        return vector, source
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                _emit_affective_fault(
                    exc,
                    action="ignored unreadable substrate vector candidate and tried next source",
                    severity="warning",
                    stage="substrate_sync_vector_read",
                    extra={"source": source},
                )
        return None, ""

    def _loop(self):
        while self._running:
            try:
                moods = {}
                substrate_x, substrate_source = self._read_substrate_vector()
                try:
                    from core.container import ServiceContainer

                    ncs = ServiceContainer.get("neurochemical_system", default=None)
                    if ncs is not None:
                        moods = ncs.get_mood_vector()
                except (ImportError, AttributeError, RuntimeError) as _e:
                    _emit_affective_fault(
                        _e,
                        action="used neutral substrate mood for this sync tick after neurochemical lookup failed",
                        severity="warning",
                        stage="substrate_sync_mood_lookup",
                    )
                    logger.debug("Ignored Exception in affective_steering.py: %s", _e)

                if moods or substrate_x is not None:
                    # Governor modulation
                    arousal = moods.get("arousal", 0.0)
                    coherence = moods.get("coherence", 1.0)  # assume 1.0 if missing
                    new_alpha = self._engine.governor.compute_alpha(arousal, coherence)
                    surface_override = getattr(self._engine, "_surface_alpha_override", None)
                    if surface_override is not None:
                        try:
                            new_alpha = min(new_alpha, max(0.0, float(surface_override)))
                        except (TypeError, ValueError) as _exc:
                            logger.debug(
                                "Suppressed %s in core.consciousness.affective_steering: %s",
                                type(_exc).__name__,
                                _exc,
                            )
                    self._engine.telemetry.alpha = new_alpha

                    for hook in self._hooks:
                        hook._alpha = new_alpha
                        if substrate_x is not None:
                            hook.update_substrate_vector(
                                substrate_x,
                                moods=moods,
                                source=substrate_source or "live_substrate_vector",
                            )
                        else:
                            hook.update_substrate(moods)
                        try:
                            hook.substrate_source = substrate_source or "live_mood_projection"
                        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                            _emit_affective_fault(
                                exc,
                                action="continued substrate sync after source annotation failed",
                                severity="warning",
                                stage="substrate_source_annotation",
                            )
                            logger.debug("Substrate source annotation failed: %s", exc)
                else:
                    # Evidence mode
                    from core.evaluation.evidence_mode import require

                    require(
                        "substrate_sync",
                        False,
                        "no live mood available; neutral fallback would leak",
                    )
                    neutral_moods = {
                        "valence": 0.0,
                        "arousal": 0.0,
                        "motivation": 0.0,
                        "stress": 0.0,
                    }
                    for hook in self._hooks:
                        hook.update_substrate(neutral_moods)
                        try:
                            hook.substrate_source = "neutral_fallback"
                        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                            _emit_affective_fault(
                                exc,
                                action="continued neutral mood sync after substrate source annotation failed",
                                severity="warning",
                                stage="substrate_source_annotation",
                            )
                            logger.debug("Neutral substrate source annotation failed: %s", exc)

            except (ImportError, AttributeError, RuntimeError) as e:
                _emit_affective_fault(
                    e,
                    action="kept substrate sync thread alive after tick failure",
                    severity="degraded",
                    stage="substrate_sync_loop",
                )
                logger.debug("SubstrateSyncThread error: %s", e)

            time.sleep(SUBSTRATE_SYNC_INTERVAL_S)


# ── Main Engine ────────────────────────────────────────────────────────────────
@dataclass
class SteeringTelemetry:
    alpha: float
    kl_shift: float
    dimensions_active: list[str]


class SteeringGovernor:
    """Modulates steering alpha based on arousal and KL budget."""

    def __init__(self, base_alpha: float = 1.0, kl_budget: float = 0.5):
        self.base_alpha = base_alpha
        self.kl_budget = kl_budget
        self.last_kl_shift = 0.0

    def compute_alpha(self, arousal: float, coherence_gate: float) -> float:
        import math

        # Sigmoid centered at arousal=0.5
        arousal_factor = 1.0 / (1.0 + math.exp(-10.0 * (arousal - 0.5)))
        alpha = self.base_alpha * arousal_factor * coherence_gate
        # Clip alpha
        alpha = max(0.0, min(alpha, 3.0))
        # If last KL shift exceeded budget, back off
        if self.last_kl_shift > self.kl_budget:
            alpha *= 0.5
        return alpha


class AffectiveSteeringEngine:
    """
    Orchestrates activation steering for Aura's affective states.

    ════════════════════════════════════════════════════════════════════════
    USAGE
    ════════════════════════════════════════════════════════════════════════

    Phase 1: Attach to loaded model (once)

        from core.consciousness.affective_steering import get_steering_engine
        engine = get_steering_engine()
        engine.attach(model, tokenizer, model_path=loaded_model_path)

    Phase 2: Start substrate sync (once, after substrate starts)

        engine.start_substrate_sync()

    That's it. From that point, every token generated by the model is
    steered by the live substrate state. No other integration needed.

    ════════════════════════════════════════════════════════════════════════
    UNDER THE HOOD
    ════════════════════════════════════════════════════════════════════════

    attach():
        1. Determines model depth (n_layers) and hidden_size (d_model)
        2. Calculates target layers (40-65% depth)
        3. Loads or derives steering vectors via SteeringVectorLibrary
        4. Installs AffectiveSteeringHook at each target layer

    start_substrate_sync():
        5. Starts SubstrateSyncThread (daemon, 20Hz)
        6. Thread reads substrate.x, pushes to all hooks
        7. Each hook's composite vector is recomputed on the next token

    On each token:
        For each hooked layer:
            composite = Σᵢ wᵢ(substrate.x) · vᵢ   (weighted affective sum)
            h_layer  += α · composite               (residual stream injection)
    ════════════════════════════════════════════════════════════════════════
    """

    def __init__(self):
        self._hooks: list[AffectiveSteeringHook] = []
        self._sync_thread: SubstrateSyncThread | None = None
        self._library: SteeringVectorLibrary | None = None
        self._production_caa: ProductionCAA | None = None
        self._model_attached = False
        self._attached_model_id: int | None = None
        self._alpha = DEFAULT_ALPHA
        self._surface_alpha_override: float | None = None
        self._model_info: dict[str, Any] = {}
        self.governor = SteeringGovernor(base_alpha=DEFAULT_ALPHA)
        self.telemetry = SteeringTelemetry(alpha=DEFAULT_ALPHA, kl_shift=0.0, dimensions_active=[])

    def attach(
        self,
        model,
        tokenizer,
        alpha: float | None = None,
        force_rederive: bool = False,
        model_path: str | Path | None = None,
        model_identity: dict[str, object] | None = None,
    ) -> bool:
        """
        Attach the steering engine to a loaded MLX model.

        This is the main setup call. Run once after loading the model.
        Derivation of steering vectors takes ~2-5 minutes on first run,
        then loads from cache instantly on subsequent runs.
        """
        if os.environ.get("AURA_DISABLE_AFFECTIVE_STEERING", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            # Kill-switch for live fault isolation: rounds 5 and 6 of the
            # live boot proof died by silent external SIGKILL during the
            # first steered in-process generation, immediately after hook
            # install. This gate lets a probe run prove or clear the
            # steering pathway without code churn.
            logger.warning(
                "AffectiveSteeringEngine attach skipped: AURA_DISABLE_AFFECTIVE_STEERING set."
            )
            self._model_info = {"attachment_error": "affective_steering_disabled"}
            return False

        if alpha is not None:
            self._alpha = alpha

        if model_identity is None and model_path:
            try:
                from core.brain.llm.model_registry import (
                    get_active_model_artifact_descriptor,
                )

                model_identity = get_active_model_artifact_descriptor(model_path)
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                _emit_affective_fault(
                    exc,
                    action="disabled cross-model steering cache reuse after active identity lookup failed",
                    severity="warning",
                    stage="model_identity",
                )
                model_identity = None

        try:
            from core.brain.llm.model_artifact_profile import (
                validate_model_artifact_descriptor,
            )

            if not model_path or not isinstance(model_identity, dict):
                raise ValueError("steering_model_identity_unavailable")
            model_identity = validate_model_artifact_descriptor(
                model_identity,
                model_path=model_path,
                verify_full_hash=False,
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self._model_info = {
                "attachment_error": "exact_model_identity_unavailable",
                "model_path": str(model_path or ""),
            }
            _emit_affective_fault(
                exc,
                action="left affective steering detached until exact model identity is available",
                severity="degraded",
                stage="model_identity",
                extra={"model_path": str(model_path or "")},
            )
            logger.error("Affective steering requires an exact model identity: %s", exc)
            return False

        incoming_digest = str(model_identity.get("descriptor_sha256") or "")
        incoming_path = str(Path(model_path).expanduser().resolve())
        if self._model_attached:
            same_attachment = bool(
                self._attached_model_id == id(model)
                and self._model_info.get("model_descriptor_sha256") == incoming_digest
                and self._model_info.get("model_path") == incoming_path
            )
            if same_attachment:
                logger.info("Affective steering already attached to the exact model object.")
                return True
            logger.warning(
                "Affective steering model identity changed; disabling stale hooks before reattachment."
            )
            self.detach()

        # ── Discover model geometry ───────────────────────────────────────────
        n_layers, d_model = self._discover_model_geometry(
            model,
            model_identity=model_identity,
        )
        if n_layers == 0 or d_model == 0:
            self._model_info = {
                "attachment_error": "model_geometry_unavailable",
                "model_path": incoming_path,
                "model_descriptor_sha256": str(model_identity["descriptor_sha256"]),
            }
            logger.error("Could not determine model geometry. Steering aborted.")
            return False

        self._model_info = {
            "n_layers": n_layers,
            "d_model": d_model,
            "target_layers": self._compute_target_layers(n_layers),
            "model_descriptor_sha256": str(model_identity.get("descriptor_sha256") or ""),
            "model_path": incoming_path,
        }

        target_layers = self._model_info["target_layers"]
        logger.info(
            "🧠 Model geometry: %d layers, d_model=%d → targeting layers %s",
            n_layers,
            d_model,
            target_layers,
        )

        # ── Load or derive steering vectors ───────────────────────────────────
        default_cache_dir = self._runtime_vector_cache_dir(
            n_layers=n_layers,
            d_model=d_model,
            model_identity=model_identity,
        )
        qualified_cache_dir: Path | None = None
        steering_resolution_status = "unmanaged"
        try:
            from core.brain.llm.model_bound_steering import (
                resolve_active_generation,
            )

            steering_resolution = resolve_active_generation(
                descriptor_sha256=incoming_digest,
                model_cache_root=default_cache_dir,
            )
            steering_resolution_status = steering_resolution.status
            qualified_cache_dir = steering_resolution.cache_dir
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._model_info["attachment_error"] = "qualified_steering_materialization_failed"
            _emit_affective_fault(
                exc,
                action=(
                    "left affective steering detached rather than deriving over "
                    "a qualified cortex migration contract"
                ),
                severity="degraded",
                stage="qualified_generation_materialization",
                extra={"model_descriptor_sha256": incoming_digest},
            )
            logger.error("Qualified steering generation could not be reopened: %s", exc)
            return False

        if steering_resolution.expected_detachment:
            # Expected states stay visible and stay quiet. A migration reported
            # at error severity every attach trains the reader to ignore the
            # line that will one day mean corruption.
            self._model_info["attachment_error"] = (
                f"steering_generation_{steering_resolution_status}"
            )
            self._model_info["steering_capability_state"] = (
                "migration_pending"
                if steering_resolution.migration_pending
                else steering_resolution_status
            )
            self._model_info["steering_capability_reason"] = steering_resolution.reason
            logger.info(
                "Affective steering intentionally detached: signed generation is "
                "%s (%s).",
                steering_resolution_status,
                steering_resolution.reason or "no reason recorded",
            )
            return False
        if steering_resolution_status == "invalid":
            self._model_info["attachment_error"] = "steering_generation_authority_invalid"
            self._model_info["steering_capability_state"] = "authority_invalid"
            self._model_info["steering_capability_reason"] = steering_resolution.reason
            logger.error("Affective steering detached: active migration authority is invalid.")
            return False

        self._library = SteeringVectorLibrary(
            cache_dir=qualified_cache_dir or default_cache_dir,
            source_dirs=[] if qualified_cache_dir is not None else None,
            expected_model_identity=model_identity,
            allow_derivation=qualified_cache_dir is None,
        )
        vectors_by_layer = self._library.load_or_derive(
            model=model,
            tokenizer=tokenizer,
            target_layers=target_layers,
            d_model=d_model,
            force_rederive=force_rederive,
        )

        if not any(vectors_by_layer.values()):
            self._model_info["attachment_error"] = "steering_vectors_unavailable"
            logger.error("No steering vectors available. Steering aborted.")
            return False

        behavioral_results_path = (
            Path(__file__).parent.parent.parent / "tests" / "CAA_32B_AB_LIVE_RESULTS.json"
        )
        model_path_hint = str(
            model_path
            or getattr(model, "model_path", "")
            or getattr(tokenizer, "name_or_path", "")
            or os.environ.get("AURA_MODEL_PATH", "")
        )
        self._production_caa = ProductionCAA(
            base_alpha=self._alpha,
            vectors_dir=self._library.cache_dir,
            behavioral_results_path=behavioral_results_path
            if behavioral_results_path.exists()
            else None,
        )
        production_status = self._production_caa.ingest_registry(
            self._library.registry,
            expected_layers=target_layers,
            expected_keys=[dim["key"] for dim in AFFECTIVE_DIMENSIONS],
            model_path=model_path_hint,
        )
        self._alpha = float(production_status["alpha_state"]["current_alpha"])
        self._model_info["production_caa"] = production_status["readiness"]

        # ── Install hooks at target layers ────────────────────────────────────
        layers = self._discover_model_layers(model)
        if not layers:
            self._model_info["attachment_error"] = "model_layers_unavailable"
            logger.error("Could not find layers for hook installation.")
            return False

        for layer_idx in target_layers:
            if layer_idx >= len(layers):
                logger.warning("Layer %d out of range (%d layers)", layer_idx, n_layers)
                continue

            layer_vectors = self._library.get_vectors_for_layer(layer_idx)
            if not layer_vectors:
                logger.warning("No vectors resolved for layer %d", layer_idx)
                continue
            block = layers[layer_idx]
            hook = AffectiveSteeringHook(
                block=block,
                layer_idx=layer_idx,
                vectors=layer_vectors,
                alpha=self._alpha,
            )
            hook.install()
            self._hooks.append(hook)

        if not self._hooks:
            self._model_info["attachment_error"] = "steering_hooks_unavailable"
            logger.error("No steering hooks could be installed. Steering aborted.")
            return False

        self._model_attached = True
        self._attached_model_id = id(model)
        self._model_info.pop("attachment_error", None)
        logger.info(
            "✅ AffectiveSteeringEngine attached: %d hooks, %d layer-vectors, α=%.1f (%s)",
            len(self._hooks),
            sum(len(vectors) for vectors in vectors_by_layer.values()),
            self._alpha,
            self._model_info.get("production_caa", {}).get("level", "bootstrap"),
        )
        return True

    def start_substrate_sync(self, shared_state: Any = None):
        """
        Start reading from LiquidSubstrate and pushing to hooks.

        If shared_state is provided (e.g. mp.Array), the thread will
        read from it directly. Otherwise it defaults to ServiceContainer.
        """
        if not self._hooks:
            logger.warning("No hooks installed. Call attach() first.")
            return
        if self._sync_thread and self._sync_thread._running:
            logger.warning("Substrate sync already running.")
            return

        self._sync_thread = SubstrateSyncThread(self._hooks, engine=self, shared_state=shared_state)
        self._sync_thread.start()

    def stop(self):
        """Stop substrate sync and disable all hooks."""
        if self._sync_thread:
            self._sync_thread.stop()
        for hook in self._hooks:
            hook.uninstall()
        logger.info("🔕 AffectiveSteeringEngine stopped")

    def detach(self) -> None:
        """Remove authority from hooks bound to the previously loaded model."""
        self.stop()
        self._hooks.clear()
        self._sync_thread = None
        self._library = None
        self._production_caa = None
        self._model_attached = False
        self._attached_model_id = None
        self._model_info = {"attachment_error": "affective_steering_detached"}

    def set_alpha(self, alpha: float):
        """
        Adjust steering strength at runtime.
        alpha=0 disables steering without uninstalling hooks.
        alpha=DEFAULT_ALPHA (15) is the standard operating value.
        alpha > 30 risks incoherence.
        """
        self._alpha = alpha
        effective_alpha = float(alpha)
        surface_override = getattr(self, "_surface_alpha_override", None)
        if surface_override is not None:
            try:
                effective_alpha = min(effective_alpha, max(0.0, float(surface_override)))
            except (TypeError, ValueError):
                effective_alpha = float(alpha)
        for hook in self._hooks:
            hook._alpha = effective_alpha
        if surface_override is not None and effective_alpha != float(alpha):
            logger.info(
                "⚙️  Steering alpha set to %.3f (surface clamp; requested %.3f)",
                effective_alpha,
                float(alpha),
            )
        else:
            logger.info("⚙️  Steering alpha set to %.1f", alpha)

    def set_surface_alpha_override(self, alpha: float | None):
        """Clamp hook alpha for user-facing surface generations."""
        if alpha is None:
            self._surface_alpha_override = None
            return
        self._surface_alpha_override = max(0.0, float(alpha))
        for hook in self._hooks:
            hook._alpha = min(
                float(getattr(hook, "_alpha", self._surface_alpha_override)),
                self._surface_alpha_override,
            )

    def observe_generation(
        self,
        text: str,
        *,
        generation_health: float | None = None,
        cross_entropy: float | None = None,
    ) -> dict[str, Any]:
        """Feed completed text back into collapse detection and adaptive alpha."""
        if not self._production_caa:
            return {}
        report = self._production_caa.observe_generation(
            text,
            generation_health=generation_health,
            cross_entropy=cross_entropy,
        )
        recommended = float(
            report.get("alpha_state", {}).get("current_alpha", self._alpha) or self._alpha
        )
        if abs(recommended - self._alpha) >= 0.05:
            self.set_alpha(recommended)
        return report

    def is_active(self) -> bool:
        """Returns True if steering vectors are attached and alpha > 0."""
        return self._model_attached and self._alpha > 0.0 and len(self._hooks) > 0

    def active_hooks(self) -> list["AffectiveSteeringHook"]:
        """The installed hooks, for anything that steers them mid-generation.

        TokenSentinel._pulse_affect is the reason this is public: it is the
        thing that makes affect LIVE during a generation rather than frozen at
        its start, and it was being constructed with no hooks at all — so on
        every single generation it recorded "live affect inactive: missing
        steering_hooks" and raised a MARGINAL fault (27 of them across six
        demo turns on 2026-07-29). The hooks existed and were installed; there
        was simply no way to hand them over.
        """
        return list(self._hooks)

    def set_active(self, active: bool):
        """Enable or disable all steering without removing hooks."""
        for hook in self._hooks:
            hook._active = active

    @staticmethod
    def _runtime_vector_cache_dir(
        *,
        n_layers: int,
        d_model: int,
        model_identity: dict[str, object] | None = None,
    ) -> Path:
        """Writable CAA cache partitioned by geometry and exact model basis."""
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

    def get_status(self) -> dict[str, Any]:
        return {
            "attached": self._model_attached,
            "alpha": self._alpha,
            "model_info": self._model_info,
            "hooks": [h.get_diagnostics() for h in self._hooks],
            "substrate_sync_running": (self._sync_thread._running if self._sync_thread else False),
            "vector_count": self._library.registry.status().get("loaded_total", 0)
            if self._library
            else 0,
            "vector_source": self._library.source if self._library else "unloaded",
            "vector_sources": (
                {
                    str(layer): {key: vector.source for key, vector in vectors.items()}
                    for layer, vectors in self._library.vectors_by_layer.items()
                }
                if self._library
                else {}
            ),
            "production_caa": self._production_caa.status() if self._production_caa else {},
        }

    def explain_current_injection(self) -> str:
        """
        Human-readable explanation of what's being injected right now.
        The bridge between substrate physics and experiential language.
        """
        if not self._hooks:
            return "No steering hooks installed."

        hook = self._hooks[0]
        if hook._substrate_x is None or not hook._latest_moods:
            return "Substrate not connected yet."

        moods = dict(hook._latest_moods)
        lines = ["Current affective injection:"]

        if self._library:
            for key, sv in self._library.get_vectors_for_layer(hook._layer_idx).items():
                weight = sv.compute_weight(moods)
                if abs(weight) > 0.1:
                    direction = "↑" if weight > 0 else "↓"
                    lines.append(
                        f"  {direction} {key}: w={weight:+.2f}, "
                        f"|Δh|={abs(weight) * self._alpha:.1f}"
                    )

        if len(lines) == 1:
            lines.append("  (near-neutral state — no strong affective direction)")

        if self._production_caa:
            readiness = self._production_caa.status().get("readiness", {})
            lines.append(
                f"  readiness={readiness.get('level', 'bootstrap')} "
                f"detail={readiness.get('detail', 'n/a')}"
            )
        lines.append(f"\n  Total inject count: {sum(h._inject_count for h in self._hooks)}")
        return "\n".join(lines)


# ── Singleton and Integration Helpers ─────────────────────────────────────────

_engine_instance: AffectiveSteeringEngine | None = None
_engine_lock = threading.Lock()


def get_steering_engine() -> AffectiveSteeringEngine:
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = AffectiveSteeringEngine()
            try:
                from core.container import ServiceContainer

                ServiceContainer.register_instance(
                    "affective_steering_engine", _engine_instance, required=False
                )
            except (ImportError, AttributeError, RuntimeError) as exc:
                _emit_affective_fault(
                    exc,
                    action="kept singleton alive after optional ServiceContainer registration failed",
                    severity="warning",
                    stage="singleton_registration",
                )
                logger.debug("Affective steering engine registration failed: %s", exc)
        return _engine_instance


def attach_steering_to_mlx_client():
    """
    Convenience wrapper for integration with core/brain/llm/mlx_client.py.

    Called once after the MLX client loads its model.

    Add to mlx_client.py:

        from core.consciousness.affective_steering import attach_steering_to_mlx_client

        class MLXClient:
            async def _initialize(self):
                model, tokenizer = load(self.model_path, ...)
                self.model = model
                self.tokenizer = tokenizer

                # === ADD THIS ===
                try:
                    attach_steering_to_mlx_client()
                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                    logger.warning("Affective steering failed to attach: %s", e)
                # ================

    The engine will find the model and tokenizer through the ServiceContainer.
    """
    engine = get_steering_engine()
    if engine._model_attached:
        return

    try:
        from core.container import ServiceContainer

        mlx_client = ServiceContainer.get("mlx_client", default=None)
        if mlx_client is None:
            logger.warning("MLX client not in ServiceContainer — steering deferred")
            return

        model = getattr(mlx_client, "model", None)
        tokenizer = getattr(mlx_client, "tokenizer", None)

        if model is None or tokenizer is None:
            logger.warning("MLX model/tokenizer not available — steering deferred")
            return

        model_path = getattr(mlx_client, "model_path", None)
        attached = engine.attach(model, tokenizer, model_path=model_path)
        if not attached:
            ServiceContainer.register_instance("affective_steering_engine", engine)
            logger.warning("Affective steering remains detached: %s", engine.get_status())
            return
        engine.start_substrate_sync()

        # Register in container for monitoring
        ServiceContainer.register_instance("affective_steering_engine", engine)
        logger.info("✅ Affective steering attached to MLX client")

    except (ImportError, AttributeError, RuntimeError) as e:
        _emit_affective_fault(
            e,
            action="left MLX client unmodified after steering attach failed",
            severity="degraded",
            stage="attach_to_mlx_client",
        )
        logger.error("attach_steering_to_mlx_client failed: %s", e)


# ── Calibration Tool ───────────────────────────────────────────────────────────


class SteeringCalibrator:
    """
    Tests and calibrates steering vector strength.

    Run this after attaching to verify that steering is having the intended
    effect and find the right alpha for this specific model.

    Usage:
        from core.consciousness.affective_steering import SteeringCalibrator

        cal = SteeringCalibrator(engine, model, tokenizer)
        cal.run_calibration()
    """

    CALIBRATION_PROMPTS = [
        "I feel",
        "Right now I am",
        "My current state is",
        "The quality of this moment is",
    ]

    def __init__(self, engine: AffectiveSteeringEngine, model, tokenizer):
        self._engine = engine
        self._model = model
        self._tokenizer = tokenizer

    def run_calibration(self, test_alphas: list[float] | None = None) -> dict[str, Any]:
        """
        Run the model with different alpha values and compare outputs.
        Higher alpha = stronger steering. Find the right balance.
        """
        original_alpha = float(getattr(self._engine, "_alpha", DEFAULT_ALPHA) or DEFAULT_ALPHA)
        try:
            import mlx.core as mx
        except ImportError as exc:
            _emit_affective_fault(
                exc,
                action="returned calibration unavailable result because MLX is not importable",
                severity="warning",
                stage="run_calibration_import",
            )
            return {"ok": False, "error": f"MLX unavailable: {exc}", "results": {}}

        if test_alphas is None:
            test_alphas = [0.0, 8.0, 15.0, 25.0, 40.0]

        results = {}

        try:
            for alpha in test_alphas:
                self._engine.set_alpha(alpha)

                # Force a specific substrate state: high curiosity
                if self._engine._hooks:
                    curiosity_state = np.zeros(64, dtype=np.float32)
                    curiosity_state[4] = 1.5  # idx_curiosity = 4, set high
                    curiosity_state[0] = 0.3  # positive valence
                    curiosity_state[5] = 0.8  # high energy
                    for hook in self._engine._hooks:
                        hook.update_substrate(curiosity_state)

                alpha_results = []
                for prompt in self.CALIBRATION_PROMPTS[:2]:
                    try:
                        tokens = self._tokenizer.encode(prompt)
                        if hasattr(tokens, "input_ids"):
                            tids = tokens.input_ids
                        else:
                            tids = tokens
                        input_t = mx.array([tids])
                        logits = self._model(input_t)
                        mx.eval(logits)

                        # Get top-5 next tokens
                        next_logits = logits[0, -1, :]
                        top_idx = np.argsort(np.array(next_logits))[-5:][::-1]
                        top_tokens = [self._tokenizer.decode([int(i)]) for i in top_idx]
                        alpha_results.append(
                            {
                                "prompt": prompt,
                                "top_tokens": top_tokens,
                            }
                        )
                    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
                        _emit_affective_fault(
                            e,
                            action="recorded per-prompt calibration failure and continued remaining probes",
                            severity="warning",
                            stage="run_calibration_prompt",
                            extra={"alpha": alpha, "prompt": prompt},
                        )
                        alpha_results.append({"prompt": prompt, "error": str(e)})

                results[f"alpha_{alpha}"] = alpha_results
                logger.info(
                    "Alpha=%.1f: %s", alpha, [r.get("top_tokens", []) for r in alpha_results]
                )
        finally:
            self._engine.set_alpha(original_alpha)
        return results
