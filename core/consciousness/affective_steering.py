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
    engine.attach(model)          # patches the layers
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

logger = logging.getLogger("Aura.AffectiveSteering")


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
DEFAULT_ALPHA = 5.0

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
        "substrate_idx": 0,   # x[idx_valence] in LiquidSubstrate
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
        "substrate_idx": 1,   # x[idx_arousal]
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
        "substrate_idx": 4,   # x[idx_curiosity]
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
        "substrate_idx": 3,   # x[idx_frustration]
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
        "substrate_idx": 5,   # x[idx_energy]
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
    v: np.ndarray              # shape: [d_model]
    substrate_idx: int         # which substrate neuron drives this
    substrate_fn: str          # how to map substrate value to weight
    is_derived: bool = False   # True if derived from model activations
    derived_at: float = 0.0    # timestamp of derivation
    source: str = "unknown"    # extracted_caa, runtime_derived_caa, fallback_random, etc.
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
    ):
        discovered_source_dirs: list[Path] = []
        if cache_dir is None:
            env_dir = os.environ.get("AURA_STEERING_DIR")
            if env_dir and Path(env_dir).exists():
                discovered_source_dirs.append(Path(env_dir))

            extracted_dir = Path(__file__).parent.parent.parent / "training" / "vectors"
            if extracted_dir.exists() and (any(extracted_dir.glob("*.npy")) or any(extracted_dir.glob("*.npz"))):
                discovered_source_dirs.append(extracted_dir)

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
                cache_dir = Path.home() / ".aura" / "steering_vectors"

        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        raw_source_dirs = list(source_dirs or discovered_source_dirs)
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
                if self._vector_dim_for_path(path) == d_model:
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
            tmp_path = cache_path.with_suffix(".tmp")
            np.savez(
                tmp_path,
                v=vec,
                derived_at=derived_at,
                source="runtime_derived_caa",
                requested_layer=target_layer,
                selected_layer=target_layer,
                selection_reason="runtime_derived",
                extracted=False,
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
                                key, layer_idx, path.name, e,
                            )
                if vector is None:
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
                        # Capture last token (detach via numpy copy)
                        # Hidden shape: [batch, sequence, d_model]
                        captured[0] = np.array(h[0, -1, :])  # [d_model]
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
                mx.eval(_)  # Force evaluation (MLX is lazy)
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

            return captured[0]

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
            raise RuntimeError("No valid activations collected — model may not support this extraction")

        pos_mean = np.mean(pos_activations, axis=0)  # [d_model]
        neg_mean = np.mean(neg_activations, axis=0)  # [d_model]
        vec = pos_mean - neg_mean                      # CAA direction

        # Normalize to unit vector (alpha controls magnitude)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm

        return vec.astype(np.float32)

    def _get_model_layers(self, model) -> list[Any] | None:
        """Helper to find the layers list in various MLX model structures."""
        # Standard mlx-lm structure: model.model.layers
        # But some versions (e.g. Qwen, Phi) use model.layers directly
        layers = getattr(model, "layers", None)
        if not layers and hasattr(model, "model"):
            layers = getattr(model.model, "layers", None)
        return layers

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
        
        # Active flag
        self._active = True
        
        # Diagnostic counters
        self._inject_count = 0
        try:
            self._phi_sample_every = max(1, int(os.getenv("AURA_PHI_RESIDUAL_SAMPLE_EVERY", "32")))
        except (TypeError, ValueError):
            self._phi_sample_every = 32
        self._last_injection_norm = 0.0
        self._last_mask_mode = "none"
        
        # [OPTIMIZATION] Cached composite vector to avoid redundant MLX uploads
        self._cached_composite_mx: Any = None
        self._last_composite_np: np.ndarray | None = None
        self._cached_substrate_hash: int = 0

    def update_substrate(self, moods: dict[str, float]):
        """Called by SubstrateSyncThread at ~20Hz. [OPTIMIZED]"""
        import mlx.core as mx
        with self._substrate_lock:
            # 1. Store mood state for debugging
            self._latest_moods = {str(key): float(value) for key, value in dict(moods or {}).items()}
            self._substrate_x = np.zeros(64, dtype=np.float32)
            self._substrate_x[0] = float(self._latest_moods.get("valence", 0.0))
            self._substrate_x[1] = float(self._latest_moods.get("arousal", 0.0))
            self._substrate_x[3] = float(self._latest_moods.get("stress", 0.0))
            self._substrate_x[4] = float(self._latest_moods.get("motivation", 0.0))
            self._substrate_x[5] = float(self._latest_moods.get("energy", 0.0))
            
            # 2. PRE-COMPUTE COMPOSITE ON CPU/NP (Background Thread)
            # This moves the O(dims * d_model) work out of the inference hook.
            target_composite_np = np.zeros(self._vectors[next(iter(self._vectors))].d_model, dtype=np.float32)
            active = False
            
            for sv in self._vectors.values():
                weight = sv.compute_weight(moods)
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
                composite_np = (momentum * self._last_composite_np) + ((1.0 - momentum) * target_composite_np)
            else:
                composite_np = target_composite_np
                
            # Update MLX array ONCE per substrate tick if magnitude is non-zero
            current_norm = np.linalg.norm(composite_np)
            if current_norm < 1e-4:
                self._last_composite_np = None
                self._cached_composite_mx = None
            else:
                self._last_composite_np = composite_np.copy()
                self._cached_composite_mx = mx.array(composite_np)
                
                # Tier 3 Hardening: Zero-Copy MLX explicit evaluation
                # This prevents lazy evaluation from stalling the main generation thread
                mx.eval(self._cached_composite_mx)

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

    def _completion_position_mask(self, h: Any) -> Any | None:
        """Return a broadcast mask for the completion/current token position."""
        try:
            import mlx.core as mx

            shape = tuple(getattr(h, "shape", ()) or ())
            if len(shape) == 2 and shape[0] > 1:
                mask_np = np.zeros((shape[0], 1), dtype=np.float32)
                mask_np[-1, 0] = 1.0
                self._last_mask_mode = "last_position_2d"
                return mx.astype(mx.array(mask_np), h.dtype)
            if len(shape) == 3 and shape[1] > 1:
                mask_np = np.zeros((shape[0], shape[1], 1), dtype=np.float32)
                mask_np[:, -1, 0] = 1.0
                self._last_mask_mode = "last_position_3d"
                return mx.astype(mx.array(mask_np), h.dtype)
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
        if os.getenv("AURA_PHI_RECORD_RESIDUALS", "1").strip().lower() in {"0", "false", "off", "no"}:
            return
        if self._inject_count % self._phi_sample_every != 0:
            return
        try:
            from core.container import ServiceContainer

            if not ServiceContainer.has("phi_core"):
                return
            phi_core = ServiceContainer.get("phi_core", default=None)
            if phi_core is not None and hasattr(phi_core, "record_residual_stream"):
                phi_core.record_residual_stream(h, layer_idx=self._layer_idx, token_position=-1)
        except (ImportError, AttributeError, RuntimeError) as exc:
            _emit_affective_fault(
                exc,
                action="continued generation after optional phi residual sample failed",
                severity="warning",
                stage="phi_residual_sample",
                extra={"layer_idx": self._layer_idx},
            )
            logger.debug("Residual phi sample failed at layer %d: %s", self._layer_idx, exc)

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

                # Compute the affective addition directly in MLX
                composite = hook.compute_composite_vector_mx(dtype=h.dtype)

                if composite is not None:
                    mask = hook._completion_position_mask(h)
                    if mask is not None:
                        h = h + (mask * hook._alpha * composite)
                    else:
                        h = h + hook._alpha * composite

                    # Diagnostic
                    hook._inject_count += 1
                    hook._maybe_record_phi_residual(h)
                    # Note: norm is expensive, only do it occasionally
                    if hook._inject_count % 50 == 0:
                        import mlx.core as mx
                        hook._last_injection_norm = float(mx.norm(composite)) * hook._alpha

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
        class SteeredBlock(block.__class__): # type: ignore
            __module__ = block.__class__.__module__
        
        # Override the target method
        setattr(SteeredBlock, target_name, lambda self, *args, **kwargs: steered_call(*args, **kwargs))
        
        block.__class__ = SteeredBlock
        self._installed = True
        logger.info(
            "🎯 Steering hook installed at layer %d (alpha=%.1f, %d vectors via %s)",
            self._layer_idx, self._alpha, len(self._vectors), target_name
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
            "substrate_valence": round(float(moods.get("valence", 0.0)), 3) if moods else None,
            "substrate_arousal": round(float(moods.get("arousal", 0.0)), 3) if moods else None,
            "vector_sources": {key: vector.source for key, vector in self._vectors.items()},
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
        logger.info("🔄 SubstrateSyncThread started (%d hooks, shared=%s)", 
                    len(self._hooks), self._shared_state is not None)

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                moods = {}
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
                    logger.debug('Ignored Exception in affective_steering.py: %s', _e)

                if moods:
                    # Governor modulation
                    arousal = moods.get("arousal", 0.0)
                    coherence = moods.get("coherence", 1.0) # assume 1.0 if missing
                    new_alpha = self._engine.governor.compute_alpha(arousal, coherence)
                    self._engine.telemetry.alpha = new_alpha
                    
                    for hook in self._hooks:
                        hook._alpha = new_alpha
                        hook.update_substrate(moods)
                        try:
                            hook.substrate_source = "live_mood"
                        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                            _emit_affective_fault(
                                exc,
                                action="continued live mood sync after substrate source annotation failed",
                                severity="warning",
                                stage="substrate_source_annotation",
                            )
                            logger.debug("Live mood substrate source annotation failed: %s", exc)
                else:
                    # Evidence mode
                    from core.evaluation.evidence_mode import require

                    require(
                        "substrate_sync",
                        False,
                        "no live mood available; neutral fallback would leak",
                    )
                    neutral_moods = {"valence": 0.0, "arousal": 0.0, "motivation": 0.0, "stress": 0.0}
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
        engine.attach(model, tokenizer)
    
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
        self._alpha = DEFAULT_ALPHA
        self._model_info: dict[str, Any] = {}
        self.governor = SteeringGovernor(base_alpha=DEFAULT_ALPHA)
        self.telemetry = SteeringTelemetry(alpha=DEFAULT_ALPHA, kl_shift=0.0, dimensions_active=[])

    def attach(
        self,
        model,
        tokenizer,
        alpha: float | None = None,
        force_rederive: bool = False,
    ):
        """
        Attach the steering engine to a loaded MLX model.
        
        This is the main setup call. Run once after loading the model.
        Derivation of steering vectors takes ~2-5 minutes on first run,
        then loads from cache instantly on subsequent runs.
        """
        if self._model_attached:
            logger.warning("Engine already attached. Call detach() first.")
            return

        if alpha is not None:
            self._alpha = alpha

        # ── Discover model geometry ───────────────────────────────────────────
        n_layers, d_model = self._discover_model_geometry(model)
        if n_layers == 0 or d_model == 0:
            logger.error("Could not determine model geometry. Steering aborted.")
            return

        self._model_info = {
            "n_layers": n_layers,
            "d_model": d_model,
            "target_layers": self._compute_target_layers(n_layers),
        }

        target_layers = self._model_info["target_layers"]
        logger.info(
            "🧠 Model geometry: %d layers, d_model=%d → targeting layers %s",
            n_layers, d_model, target_layers,
        )

        # ── Load or derive steering vectors ───────────────────────────────────
        self._library = SteeringVectorLibrary(
            cache_dir=self._runtime_vector_cache_dir(n_layers=n_layers, d_model=d_model)
        )
        vectors_by_layer = self._library.load_or_derive(
            model=model,
            tokenizer=tokenizer,
            target_layers=target_layers,
            d_model=d_model,
            force_rederive=force_rederive,
        )

        if not any(vectors_by_layer.values()):
            logger.error("No steering vectors available. Steering aborted.")
            return

        behavioral_results_path = Path(__file__).parent.parent.parent / "tests" / "CAA_32B_AB_LIVE_RESULTS.json"
        model_path_hint = str(
            getattr(model, "model_path", "")
            or getattr(tokenizer, "name_or_path", "")
            or os.environ.get("AURA_MODEL_PATH", "")
        )
        self._production_caa = ProductionCAA(
            base_alpha=self._alpha,
            vectors_dir=self._library.cache_dir,
            behavioral_results_path=behavioral_results_path if behavioral_results_path.exists() else None,
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
            logger.error("Could not find layers for hook installation.")
            return

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

        self._model_attached = True
        logger.info(
            "✅ AffectiveSteeringEngine attached: %d hooks, %d layer-vectors, α=%.1f (%s)",
            len(self._hooks),
            sum(len(vectors) for vectors in vectors_by_layer.values()),
            self._alpha,
            self._model_info.get("production_caa", {}).get("level", "bootstrap"),
        )

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

    def set_alpha(self, alpha: float):
        """
        Adjust steering strength at runtime.
        alpha=0 disables steering without uninstalling hooks.
        alpha=DEFAULT_ALPHA (15) is the standard operating value.
        alpha > 30 risks incoherence.
        """
        self._alpha = alpha
        for hook in self._hooks:
            hook._alpha = alpha
        logger.info("⚙️  Steering alpha set to %.1f", alpha)

    def observe_generation(self, text: str) -> dict[str, Any]:
        """Feed completed text back into collapse detection and adaptive alpha."""
        if not self._production_caa:
            return {}
        report = self._production_caa.observe_generation(text)
        recommended = float(report.get("alpha_state", {}).get("current_alpha", self._alpha) or self._alpha)
        if abs(recommended - self._alpha) >= 0.05:
            self.set_alpha(recommended)
        return report

    def is_active(self) -> bool:
        """Returns True if steering vectors are attached and alpha > 0."""
        return self._model_attached and self._alpha > 0.0 and len(self._hooks) > 0

    def set_active(self, active: bool):
        """Enable or disable all steering without removing hooks."""
        for hook in self._hooks:
            hook._active = active

    @staticmethod
    def _runtime_vector_cache_dir(*, n_layers: int, d_model: int) -> Path:
        """Writable runtime CAA cache partitioned by model geometry."""
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
            base = Path.home() / ".aura" / "steering_vectors"
        return base / f"dmodel_{int(d_model)}_layers_{int(n_layers)}"

    def _discover_model_geometry(self, model) -> tuple[int, int]:
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

            # d_model: find first weight with the right shape
            # Typically in attention q_proj or input_layernorm
            for layer in layers[:3]:
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
                                d_model = proj.weight.shape[-1]
                                if d_model > 512:
                                    return n_layers, d_model

            logger.warning("Geometry discovery reached fallback for d_model.")
            return n_layers, d_model or 4096  # Reasonable guess if discovery fails
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
        layers = getattr(model, "layers", None)
        if not layers and hasattr(model, "model"):
            layers = getattr(model.model, "layers", None)
        return layers

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
            "substrate_sync_running": (
                self._sync_thread._running if self._sync_thread else False
            ),
            "vector_count": self._library.registry.status().get("loaded_total", 0) if self._library else 0,
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
                ServiceContainer.register_instance("affective_steering_engine", _engine_instance, required=False)
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

        engine.attach(model, tokenizer)
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
                        alpha_results.append({
                            "prompt": prompt,
                            "top_tokens": top_tokens,
                        })
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
                logger.info("Alpha=%.1f: %s", alpha, [r.get("top_tokens", []) for r in alpha_results])
        finally:
            self._engine.set_alpha(original_alpha)
        return results
