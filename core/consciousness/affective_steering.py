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

from core.runtime.errors import record_degradation
import asyncio
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Aura.AffectiveSteering")

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
# With bootstrap-quality CAA vectors (see README "What's stubbed and what's
# real"), conservative α is the right operating point until the vectors are
# fully extracted via the full CAA pipeline.
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

    def compute_weight(self, substrate_x: np.ndarray) -> float:
        """
        Map the substrate state to a scalar steering weight in [-1, 1].
        This is the key coupling: substrate physics → affective direction strength.
        """
        raw = float(substrate_x[self.substrate_idx]) if self.substrate_idx < len(substrate_x) else 0.0
        if self.substrate_fn == "tanh":
            return float(np.tanh(raw))
        elif self.substrate_fn == "linear_half":
            return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0) * 2.0 - 1.0)
        else:
            return float(np.tanh(raw))

    def to_dict(self) -> Dict:
        return {
            "key": self.key,
            "layer_idx": self.layer_idx,
            "d_model": self.d_model,
            "substrate_idx": self.substrate_idx,
            "substrate_fn": self.substrate_fn,
            "is_derived": self.is_derived,
            "derived_at": self.derived_at,
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

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            # Check for extracted vectors first (from training/extract_steering_vectors.py).
            # These are properly extracted via contrastive prompts and are higher quality
            # than the on-demand derived vectors.
            env_dir = os.environ.get("AURA_STEERING_DIR")
            if env_dir and Path(env_dir).exists():
                cache_dir = Path(env_dir)
                logger.info("🎯 Steering vectors: using AURA_STEERING_DIR=%s", cache_dir)
            else:
                extracted_dir = Path(__file__).parent.parent.parent / "training" / "vectors"
                if extracted_dir.exists() and any(extracted_dir.glob("*.npy")):
                    cache_dir = extracted_dir
                    logger.info("🎯 Steering vectors: using extracted vectors from training/vectors/")
                else:
                    try:
                        from core.config import config as aura_config
                        cache_dir = aura_config.paths.data_dir / "steering_vectors"
                    except Exception:
                        cache_dir = Path.home() / ".aura" / "steering_vectors"
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._vectors: Dict[str, SteeringVector] = {}

    def load_or_derive(
        self,
        model,
        tokenizer,
        target_layers: List[int],
        d_model: int,
        force_rederive: bool = False,
    ) -> Dict[str, SteeringVector]:
        """
        Load cached vectors if available, derive if not.
        
        This is the most expensive operation — runs once per model.
        A progress log is emitted; derivation takes ~1-3 minutes on M5 Pro.
        """
        loaded = 0
        derived = 0

        for dim_spec in AFFECTIVE_DIMENSIONS:
            key = dim_spec["key"]
            cache_path = self._cache_dir / f"{key}_layer{target_layers[0]}.npz"

            if not force_rederive and cache_path.exists():
                try:
                    data = np.load(cache_path)
                    v = SteeringVector(
                        key=key,
                        layer_idx=target_layers[0],
                        d_model=d_model,
                        v=data["v"],
                        substrate_idx=dim_spec["substrate_idx"],
                        substrate_fn=dim_spec["substrate_fn"],
                        is_derived=True,
                        derived_at=float(data.get("derived_at", 0)),
                    )
                    self._vectors[key] = v
                    loaded += 1
                    logger.debug("📂 Loaded cached vector: %s (norm=%.3f)", key, np.linalg.norm(data["v"]))
                    continue
                except Exception as e:
                    record_degradation('affective_steering', e)
                    logger.warning("Failed to load cached vector %s: %s", key, e)

            # Derive from model
            logger.info("🔬 Deriving steering vector: %s (layer %d)...", key, target_layers[0])
            try:
                vec = self._derive_caa(
                    model=model,
                    tokenizer=tokenizer,
                    positive_prompts=dim_spec["positive"],
                    negative_prompts=dim_spec["negative"],
                    target_layer=target_layers[0],
                    d_model=d_model,
                )
                sv = SteeringVector(
                    key=key,
                    layer_idx=target_layers[0],
                    d_model=d_model,
                    v=vec,
                    substrate_idx=dim_spec["substrate_idx"],
                    substrate_fn=dim_spec["substrate_fn"],
                    is_derived=True,
                    derived_at=time.time(),
                )
                self._vectors[key] = sv
                # Cache to disk
                np.savez(cache_path, v=vec, derived_at=time.time())
                derived += 1
                logger.info("✅ Derived: %s (norm=%.3f)", key, np.linalg.norm(vec))
            except Exception as e:
                record_degradation('affective_steering', e)
                logger.error("❌ Failed to derive vector %s: %s", key, e)
                # Evidence-mode: random fallback vectors are NOT credited as
                # live steering. In normal mode we still install one so the
                # runtime keeps functioning, but the flag propagates so tests
                # can refuse to interpret the resulting run as evidence.
                try:
                    from core.evaluation.evidence_mode import require
                    require(
                        "steering_vector_derivation",
                        False,
                        f"vector {key} failed to derive from hidden states: {e}",
                    )
                except Exception:
                    raise
                fallback = np.random.randn(d_model).astype(np.float32)
                fallback /= np.linalg.norm(fallback)
                self._vectors[key] = SteeringVector(
                    key=key,
                    layer_idx=target_layers[0],
                    d_model=d_model,
                    v=fallback,
                    substrate_idx=dim_spec["substrate_idx"],
                    substrate_fn=dim_spec["substrate_fn"],
                    is_derived=False,
                )

        logger.info(
            "📚 SteeringVectorLibrary ready: %d loaded, %d derived", loaded, derived
        )
        return self._vectors

    def _derive_caa(
        self,
        model,
        tokenizer,
        positive_prompts: List[str],
        negative_prompts: List[str],
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

        def _extract_hidden_state_at_layer(prompt_text: str) -> Optional[np.ndarray]:
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
            except Exception as inner_e:
                record_degradation('affective_steering', inner_e)
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

    def _get_model_layers(self, model) -> Optional[List[Any]]:
        """Helper to find the layers list in various MLX model structures."""
        # Standard mlx-lm structure: model.model.layers
        # But some versions (e.g. Qwen, Phi) use model.layers directly
        layers = getattr(model, "layers", None)
        if not layers and hasattr(model, "model"):
            layers = getattr(model.model, "layers", None)
        return layers

    @property
    def vectors(self) -> Dict[str, SteeringVector]:
        return self._vectors


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
        vectors: Dict[str, SteeringVector],
        alpha: float = DEFAULT_ALPHA,
    ):
        self._block = block
        self._layer_idx = layer_idx
        self._vectors = vectors
        self._alpha = alpha
        self._installed = False
        
        # Shared substrate state (updated by SubstrateSyncThread)
        self._substrate_x: Optional[np.ndarray] = None
        self._substrate_lock = threading.Lock()
        
        # Active flag
        self._active = True
        
        # Diagnostic counters
        self._inject_count = 0
        self._last_injection_norm = 0.0
        self._last_mask_mode = "none"
        
        # [OPTIMIZATION] Cached composite vector to avoid redundant MLX uploads
        self._cached_composite_mx: Any = None
        self._last_composite_np: Optional[np.ndarray] = None
        self._cached_substrate_hash: int = 0

    def update_substrate(self, x: np.ndarray):
        """Called by SubstrateSyncThread at ~20Hz. [OPTIMIZED]"""
        import mlx.core as mx
        with self._substrate_lock:
            # 1. Store substrate state
            self._substrate_x = x.copy()
            
            # 2. PRE-COMPUTE COMPOSITE ON CPU/NP (Background Thread)
            # This moves the O(dims * d_model) work out of the inference hook.
            target_composite_np = np.zeros(self._vectors[next(iter(self._vectors))].d_model, dtype=np.float32)
            active = False
            
            for sv in self._vectors.values():
                weight = sv.compute_weight(x)
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

    def compute_composite_vector_mx(self, dtype=None) -> Optional[Any]:
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

    def _completion_position_mask(self, h: Any) -> Optional[Any]:
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
        except Exception as exc:
            record_degradation('affective_steering', exc)
            self._last_mask_mode = f"mask_unavailable:{type(exc).__name__}"
        return None

    def install(self):
        """
        Patch the transformer block's forward pass to inject the steering vector.
        
        Uses dynamic subclassing to ensure the interception is reliable.
        """
        if self._installed:
            return

        import mlx.core as mx
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
                    # Note: norm is expensive, only do it occasionally
                    if hook._inject_count % 50 == 0:
                        import mlx.core as mx
                        hook._last_injection_norm = float(mx.norm(composite)) * hook._alpha

                if rest is not None:
                    return (h,) + rest
                return h

            except Exception as e:
                record_degradation('affective_steering', e)
                logger.debug("Steering injection failed at layer %d: %s", hook._layer_idx, e)
                return result

        # Use dynamic subclassing to ensure interception
        class SteeredBlock(block.__class__): # type: ignore
            pass  # no-op: intentional
        
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

    def get_diagnostics(self) -> Dict[str, Any]:
        with self._substrate_lock:
            x = self._substrate_x
        return {
            "layer_idx": self._layer_idx,
            "installed": self._installed,
            "active": self._active,
            "inject_count": self._inject_count,
            "last_injection_norm": round(self._last_injection_norm, 4),
            "last_mask_mode": self._last_mask_mode,
            "substrate_connected": x is not None,
            "substrate_valence": round(float(np.tanh(x[0])), 3) if x is not None else None,
            "substrate_arousal": round(float((x[1] + 1.0) / 2.0), 3) if x is not None else None,
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

    def __init__(self, hooks: List[AffectiveSteeringHook], shared_state: Any = None):
        self._hooks = hooks
        self._shared_state = shared_state
        self._thread: Optional[threading.Thread] = None
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
        substrate = None
        while self._running:
            try:
                if self._shared_state is not None:
                    # Read from multiprocessing.Array
                    try:
                        # Convert to numpy array without copying if possible,
                        # but for 64 floats a copy is fast and safe.
                        x = np.frombuffer(self._shared_state.get_obj(), dtype=np.float32).copy()
                        for hook in self._hooks:
                            hook.update_substrate(x)
                    except Exception as e:
                        record_degradation('affective_steering', e)
                        logger.debug("Shared memory read failed: %s", e)
                else:
                    # Local mode: ServiceContainer lookup
                    if substrate is None:
                        try:
                            from core.container import ServiceContainer
                            substrate = ServiceContainer.get("conscious_substrate", default=None)
                        except Exception as _e:
                            record_degradation('affective_steering', _e)
                            logger.debug('Ignored Exception in affective_steering.py: %s', _e)

                    if substrate is not None:
                        x = substrate.x.copy()
                        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                        for hook in self._hooks:
                            hook.update_substrate(x)
                            try:
                                hook.substrate_source = "live"  # type: ignore[attr-defined]
                            except Exception:
                                pass  # no-op: intentional
                    else:
                        # Evidence mode: DO NOT credit a neutral fallback as
                        # live substrate. We mark the hook so tests can refuse
                        # to interpret outputs as state-caused behavior, and
                        # we raise in evidence mode to prevent silent drift.
                        try:
                            from core.evaluation.evidence_mode import require
                            require(
                                "substrate_sync",
                                False,
                                "no live substrate available; neutral fallback would leak",
                            )
                        except Exception:
                            raise
                        neutral = np.zeros(64, dtype=np.float32)
                        neutral[5] = 0.7
                        for hook in self._hooks:
                            hook.update_substrate(neutral)
                            try:
                                hook.substrate_source = "neutral_fallback"  # type: ignore[attr-defined]
                            except Exception:
                                pass  # no-op: intentional

            except Exception as e:
                record_degradation('affective_steering', e)
                logger.debug("SubstrateSyncThread error: %s", e)

            time.sleep(SUBSTRATE_SYNC_INTERVAL_S)


# ── Main Engine ────────────────────────────────────────────────────────────────

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
        self._hooks: List[AffectiveSteeringHook] = []
        self._sync_thread: Optional[SubstrateSyncThread] = None
        self._library: Optional[SteeringVectorLibrary] = None
        self._model_attached = False
        self._alpha = DEFAULT_ALPHA
        self._model_info: Dict[str, Any] = {}

    def attach(
        self,
        model,
        tokenizer,
        alpha: Optional[float] = None,
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
        self._library = SteeringVectorLibrary()
        vectors = self._library.load_or_derive(
            model=model,
            tokenizer=tokenizer,
            target_layers=target_layers,
            d_model=d_model,
            force_rederive=force_rederive,
        )

        if not vectors:
            logger.error("No steering vectors available. Steering aborted.")
            return

        # ── Install hooks at target layers ────────────────────────────────────
        layers = self._discover_model_layers(model)
        if not layers:
            logger.error("Could not find layers for hook installation.")
            return

        for layer_idx in target_layers:
            if layer_idx >= len(layers):
                logger.warning("Layer %d out of range (%d layers)", layer_idx, n_layers)
                continue

            block = layers[layer_idx]
            hook = AffectiveSteeringHook(
                block=block,
                layer_idx=layer_idx,
                vectors=vectors,
                alpha=self._alpha,
            )
            hook.install()
            self._hooks.append(hook)

        self._model_attached = True
        logger.info(
            "✅ AffectiveSteeringEngine attached: %d hooks, %d vectors, α=%.1f",
            len(self._hooks), len(vectors), self._alpha,
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
            return
        self._sync_thread = SubstrateSyncThread(self._hooks, shared_state=shared_state)
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

    def set_active(self, active: bool):
        """Enable or disable all steering without removing hooks."""
        for hook in self._hooks:
            hook._active = active

    def _discover_model_geometry(self, model) -> Tuple[int, int]:
        """Determine n_layers and d_model from the loaded model."""
        try:
            # Pre-initialize d_model so the fallback ``return`` on line ~1107
            # never raises UnboundLocalError when no inner branch assigned it.
            d_model: Optional[int] = None
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
        except Exception as e:
            record_degradation('affective_steering', e)
            logger.error("Error discovering model geometry: %s", e)
            return 0, 0

    def _discover_model_layers(self, model) -> Optional[List[Any]]:
        """Helper to find the layers list in various MLX model structures."""
        layers = getattr(model, "layers", None)
        if not layers and hasattr(model, "model"):
            layers = getattr(model.model, "layers", None)
        return layers

    def _compute_target_layers(self, n_layers: int) -> List[int]:
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

    def get_status(self) -> Dict[str, Any]:
        return {
            "attached": self._model_attached,
            "alpha": self._alpha,
            "model_info": self._model_info,
            "hooks": [h.get_diagnostics() for h in self._hooks],
            "substrate_sync_running": (
                self._sync_thread._running if self._sync_thread else False
            ),
            "vector_count": len(self._library._vectors) if self._library else 0,
        }

    def explain_current_injection(self) -> str:
        """
        Human-readable explanation of what's being injected right now.
        The bridge between substrate physics and experiential language.
        """
        if not self._hooks:
            return "No steering hooks installed."

        hook = self._hooks[0]
        if hook._substrate_x is None:
            return "Substrate not connected yet."

        x = hook._substrate_x
        lines = ["Current affective injection:"]

        if self._library:
            for key, sv in self._library.vectors.items():
                weight = sv.compute_weight(x)
                if abs(weight) > 0.1:
                    direction = "↑" if weight > 0 else "↓"
                    lines.append(
                        f"  {direction} {key}: w={weight:+.2f}, "
                        f"|Δh|={abs(weight) * self._alpha:.1f}"
                    )

        if len(lines) == 1:
            lines.append("  (near-neutral state — no strong affective direction)")

        lines.append(f"\n  Total inject count: {sum(h._inject_count for h in self._hooks)}")
        return "\n".join(lines)


# ── Singleton and Integration Helpers ─────────────────────────────────────────

_engine_instance: Optional[AffectiveSteeringEngine] = None
_engine_lock = threading.Lock()


def get_steering_engine() -> AffectiveSteeringEngine:
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = AffectiveSteeringEngine()
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
                except Exception as e:
                    record_degradation('affective_steering', e)
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

    except Exception as e:
        record_degradation('affective_steering', e)
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

    def run_calibration(self, test_alphas: List[float] = None) -> Dict[str, Any]:
        """
        Run the model with different alpha values and compare outputs.
        Higher alpha = stronger steering. Find the right balance.
        """
        import mlx.core as mx

        if test_alphas is None:
            test_alphas = [0.0, 8.0, 15.0, 25.0, 40.0]

        results = {}

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
                    import mlx.core as mx
                    next_logits = logits[0, -1, :]
                    top_idx = np.argsort(np.array(next_logits))[-5:][::-1]
                    top_tokens = [self._tokenizer.decode([int(i)]) for i in top_idx]
                    alpha_results.append({
                        "prompt": prompt,
                        "top_tokens": top_tokens,
                    })
                except Exception as e:
                    record_degradation('affective_steering', e)
                    alpha_results.append({"prompt": prompt, "error": str(e)})

            results[f"alpha_{alpha}"] = alpha_results
            logger.info("Alpha=%.1f: %s", alpha, [r.get("top_tokens", []) for r in alpha_results])

        # Restore default alpha
        self._engine.set_alpha(DEFAULT_ALPHA)
        return results
