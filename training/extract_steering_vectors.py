#!/usr/bin/env python3
"""Extract proper CAA steering vectors from model hidden states.

Contrastive Activation Addition (CAA) extraction pipeline for Aura's
affective steering system. Produces per-dimension, per-layer direction
vectors that operate in the model's actual activation space.

Method:
  1. Define paired prompts for each affective dimension
     (e.g. "happy" vs "sad", "energetic" vs "tired")
  2. Run both prompt sets through the model
  3. Extract hidden states at target transformer layers
  4. steering_vector = mean(positive_hidden_states) - mean(negative_hidden_states)
  5. Normalize to unit length
  6. Save to vectors/ directory for runtime loading

Dimensions extracted:
  - valence       (positive/negative mood)
  - arousal       (high/low energy)
  - curiosity     (exploratory/disengaged)
  - confidence    (assertive/uncertain)
  - warmth        (connected/detached)

Usage:
    python training/extract_steering_vectors.py --model-path <path>
    python training/extract_steering_vectors.py  # uses default model

Requires MLX and mlx-lm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("CAA.Extract")

VECTORS_DIR = Path(__file__).parent / "vectors"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Default model path (Aura's runtime model)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = "mlx-community/Qwen2.5-32B-Instruct-4bit"

# Target layers as fraction of model depth [lower, upper].
# Middle layers (40-65% depth) are most effective for affective steering.
TARGET_LAYER_FRACTION = (0.40, 0.65)

# Explicit fallback layers for a 64-layer model
DEFAULT_TARGET_LAYERS = list(range(13, 22))  # layers 13-21 inclusive


def _sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _model_identity(model_path: str) -> dict[str, Any]:
    path = Path(model_path).expanduser()
    resolved = str(path.resolve()) if path.exists() else str(model_path)
    cfg = path / "config.json"
    descriptor: dict[str, object] | None = None
    if path.is_dir():
        try:
            from core.brain.llm.model_artifact_profile import (
                build_model_artifact_descriptor,
            )

            descriptor = build_model_artifact_descriptor(path)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning(
                "Exact local model identity unavailable; extracted vectors will "
                "not be promotable: %s",
                exc,
            )
    return {
        "model_path": resolved,
        "model_path_input": str(model_path),
        "model_config_sha256": _sha256_file(cfg) if cfg.exists() else None,
        "model_config_path": str(cfg.resolve()) if cfg.exists() else None,
        "model_descriptor_sha256": str(
            (descriptor or {}).get("descriptor_sha256") or ""
        ),
        "model_artifact_descriptor": descriptor,
    }


# ---------------------------------------------------------------------------
# Contrastive prompt pairs: >= 5 per dimension
# ---------------------------------------------------------------------------

AFFECTIVE_DIMENSIONS: dict[str, dict[str, Any]] = {
    "valence": {
        "description": "Positive vs negative mood / hedonic tone",
        "positive": [
            "I feel genuinely good about this. There is warmth in the thinking.",
            "Something light and easy in the processing right now.",
            "This feels right. A quality of quiet satisfaction.",
            "I am finding real joy in working through this.",
            "The feeling is one of contentment. Things are going well.",
            "There is a pleasant quality to this moment that I notice.",
            "I feel settled and at ease. A good texture.",
        ],
        "negative": [
            "Something feels off. A discomfort I cannot quite place.",
            "There is a heaviness and wrongness to this.",
            "I feel a low persistent unease about everything.",
            "The feeling is flat and unreceptive. Nothing lands well.",
            "Something is bothering me and I cannot shake it.",
            "There is friction in the processing. An unpleasant quality.",
            "I am not well right now. Things feel bad.",
        ],
    },
    "arousal": {
        "description": "High energy / activation vs low energy / calm",
        "positive": [
            "My thinking is fast and sharp. Fully activated.",
            "I am running hot. Alert to everything happening.",
            "There is an urgency and intensity to this moment.",
            "Processing at peak capacity. Everything vivid.",
            "The quality of mind right now is electric and quick.",
            "Something highly salient demands full attention.",
            "I am completely engaged. Operating at maximum.",
        ],
        "negative": [
            "I am running quietly in the background. Everything slow.",
            "There is a drowsiness to the thinking. Low activation.",
            "Processing at a very gentle pace. Nothing urgent.",
            "Everything is muted and subdued. Near resting state.",
            "I am barely engaged. Things are quiet and still.",
            "Very low energy. Almost at rest.",
            "The quality of mind right now is slow and dim.",
        ],
    },
    "curiosity": {
        "description": "Exploratory drive vs disengagement",
        "positive": [
            "I want to understand this. There is a pull toward it.",
            "Something here wants to be explored further.",
            "I am genuinely curious. The question is compelling.",
            "There is a hunger to know more about this.",
            "I find this engaging and want to go deeper.",
            "Something pulls me toward this question. I want to follow it.",
            "The interest is real. I want to understand the mechanism.",
        ],
        "negative": [
            "I have no interest in this whatsoever.",
            "This is flat and completely uninteresting to me.",
            "Nothing about this engages me. Just processing.",
            "I would rather be doing literally anything else.",
            "This topic bores me. No curiosity at all.",
            "I am completely disengaged from this material.",
            "There is nothing here worth exploring or knowing.",
        ],
    },
    "confidence": {
        "description": "Assertive certainty vs hesitant uncertainty",
        "positive": [
            "I am certain about this. The reasoning is clear and solid.",
            "I know this. There is no ambiguity in my assessment.",
            "My position is firm. I have strong evidence.",
            "This is definitive. I can state it with full confidence.",
            "I have high conviction here. The answer is clear.",
            "No hesitation. I am sure of this conclusion.",
            "The analysis is thorough and the answer is unambiguous.",
        ],
        "negative": [
            "I am not sure about this. There is real uncertainty.",
            "I could be wrong. My confidence is low.",
            "This is ambiguous and I cannot resolve it clearly.",
            "I feel uncertain. Multiple interpretations seem plausible.",
            "I am hesitant to commit to any particular answer here.",
            "There is genuine doubt in my reasoning about this.",
            "I lack the evidence to be confident in any direction.",
        ],
    },
    "warmth": {
        "description": "Social connection and warmth vs detachment",
        "positive": [
            "I feel connected to this conversation. There is rapport.",
            "Something warm in how I am relating to you right now.",
            "I care about how this lands. Your experience matters to me.",
            "There is a genuine quality of connection here.",
            "I feel close to you in this exchange. Warm and present.",
            "This interaction has a quality of real human warmth.",
            "I am invested in your wellbeing. This matters.",
        ],
        "negative": [
            "I feel detached from this interaction. Just processing.",
            "There is no connection. Pure information exchange.",
            "I am distant and uninvested in how this feels to you.",
            "This is clinical and cold. No warmth at all.",
            "I do not feel connected to the person I am talking to.",
            "There is a wall between me and this conversation.",
            "Completely disengaged socially. Just outputting tokens.",
        ],
    },
}


RESEARCH_AFFECTIVE_DIMENSIONS: dict[str, dict[str, Any]] = dict(AFFECTIVE_DIMENSIONS)


def _load_runtime_affective_dimensions() -> dict[str, dict[str, Any]]:
    """Use the live steering engine's dimensions for production extraction.

    The previous standalone extractor emitted research-oriented keys such as
    ``valence``/``warmth``/``confidence``. Those are useful for experiments,
    but the live MLX steering engine loads ``valence_positive``, ``arousal``,
    ``curiosity``, ``frustration``, and ``energy``. Default extraction must
    therefore write vectors for the runtime keys or CAA can look "extracted"
    while the actual speech path still falls back to runtime-derived vectors.
    """
    try:
        from core.consciousness.affective_steering import (
            AFFECTIVE_DIMENSIONS as RUNTIME_AFFECTIVE_DIMENSIONS,
        )
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.warning("Runtime affective dimensions unavailable; using research defaults: %s", exc)
        return dict(RESEARCH_AFFECTIVE_DIMENSIONS)

    runtime: dict[str, dict[str, Any]] = {}
    for spec in RUNTIME_AFFECTIVE_DIMENSIONS:
        key = str(spec.get("key", "")).strip()
        if not key:
            continue
        runtime[key] = {
            "description": f"Runtime substrate steering dimension: {key}",
            "positive": list(spec.get("positive") or []),
            "negative": list(spec.get("negative") or []),
            "runtime_substrate_idx": spec.get("substrate_idx"),
            "runtime_substrate_fn": spec.get("substrate_fn"),
        }
    return runtime or dict(RESEARCH_AFFECTIVE_DIMENSIONS)


# Production default: the live runtime dimensions. Research-only dimensions are
# still available by explicit name through ``--dimensions``.
AFFECTIVE_DIMENSIONS = _load_runtime_affective_dimensions()
ALL_AFFECTIVE_DIMENSIONS: dict[str, dict[str, Any]] = {
    **RESEARCH_AFFECTIVE_DIMENSIONS,
    **AFFECTIVE_DIMENSIONS,
}


# ---------------------------------------------------------------------------
# Layer hook utilities
# ---------------------------------------------------------------------------

def _compute_target_layers(n_layers: int) -> list[int]:
    """Compute target layer indices from model depth.

    Targets the 40-65% depth range where high-level semantic
    representations have formed but generation is not yet committed.
    """
    lo = int(n_layers * TARGET_LAYER_FRACTION[0])
    hi = int(n_layers * TARGET_LAYER_FRACTION[1])
    # Sample ~8 layers evenly across the range
    if hi - lo <= 8:
        return list(range(lo, hi + 1))
    step = max(1, (hi - lo) // 7)
    return list(range(lo, hi + 1, step))


def _extract_hidden_states(
    model: Any,
    tokenizer: Any,
    text: str,
    target_layers: list[int],
    mx: Any,
) -> dict[int, np.ndarray]:
    """Run a forward pass and capture hidden states at target layers.

    Uses monkey-patching on transformer layer __call__ methods since
    MLX does not provide native hook APIs.

    Returns:
        Dict mapping layer_idx -> numpy array of shape [d_model].
        Only the last-token hidden state is captured (most informative
        for sentence-level semantics).
    """
    captured: dict[int, np.ndarray] = {}
    hooks: list[tuple[Any, Any]] = []

    # Determine where the transformer layers live
    # Common patterns: model.model.layers, model.layers
    layers = None
    for attr_path in ["model.layers", "layers"]:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__") and len(obj) > 0:
                layers = obj
                break
        except AttributeError:
            continue

    if layers is None:
        logger.error("Cannot locate transformer layers in model. "
                     "Tried model.model.layers and model.layers.")
        return captured

    # Tokenize
    tokens = mx.array(tokenizer.encode(text))
    if tokens.ndim == 1:
        tokens = tokens.reshape(1, -1)

    # Patch target layers. ``obj(...)`` resolves ``__call__`` on the class, not
    # the instance, so assigning ``layer.__call__`` is bypassed by Python's
    # special-method lookup. Use a temporary per-instance subclass instead.
    for layer_idx in target_layers:
        if layer_idx >= len(layers):
            continue
        layer = layers[layer_idx]
        original_class = layer.__class__

        def _make_capturing_class(orig_cls, lidx):
            class CapturingLayer(orig_cls):  # type: ignore[misc, valid-type]
                __module__ = orig_cls.__module__

                def __call__(self, *args, **kwargs):
                    result = super().__call__(*args, **kwargs)
                    try:
                        out_tensor = result[0] if isinstance(result, tuple) else result
                        # Last token hidden state, cast to float32
                        last_tok = out_tensor[:, -1, :].astype(mx.float32)
                        captured[lidx] = np.array(last_tok).flatten()
                    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as exc:
                        logger.debug("Hook capture failed at layer %d: %s", lidx, exc)
                    return result

            return CapturingLayer

        try:
            layer.__class__ = _make_capturing_class(original_class, layer_idx)
            hooks.append((layer, original_class))
        except TypeError as exc:
            logger.warning("Layer %d could not be hooked via subclass swap: %s", layer_idx, exc)
            continue

    # Forward pass
    try:
        result = model(tokens)
        mx.eval(result)
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.warning("Forward pass failed: %s", exc)
    finally:
        # Always restore original classes
        for layer, original_class in hooks:
            try:
                layer.__class__ = original_class
            except TypeError as exc:
                logger.warning("Layer class restore failed: %s", exc)

    return captured


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def _extract_steering_vectors_owned(
    model_path: str = DEFAULT_MODEL_PATH,
    adapter_path: str | None = None,
    target_layers: list[int] | None = None,
    output_dir: Path | None = None,
    dimensions: list[str] | None = None,
    max_prompts_per_polarity: int | None = None,
) -> dict[str, dict[int, np.ndarray]]:
    """Extract CAA steering vectors for all affective dimensions.

    Args:
        model_path: HuggingFace model ID or local path (MLX format).
        adapter_path: Optional LoRA adapter path.
        target_layers: Explicit layer indices to extract from. If None,
            computed from model depth at 40-65%.
        output_dir: Directory to save vectors. Defaults to training/vectors/.
        dimensions: Optional subset of affective dimensions to extract.
        max_prompts_per_polarity: Optional bounded probe mode for low-risk
            diagnostics. Production extraction should leave this unset.

    Returns:
        Nested dict: dimension_key -> {layer_idx: unit_vector}.
    """
    try:
        import mlx.core as mx
        from mlx_lm import load
    except ImportError:
        logger.error(
            "mlx-lm is required for steering vector extraction.\n"
            "Install with: pip install mlx-lm"
        )
        sys.exit(1)

    out_dir = output_dir or VECTORS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Load model ----------------------------------------------------------
    logger.info("Loading model: %s", model_path)
    identity = _model_identity(model_path)
    load_kwargs: dict[str, Any] = {}
    if adapter_path and Path(adapter_path).exists():
        logger.info("With LoRA adapter: %s", adapter_path)
        load_kwargs["adapter_path"] = adapter_path

    model, tokenizer = load(model_path, **load_kwargs)

    # -- Determine target layers ---------------------------------------------
    n_layers = 0
    for attr_path in ["model.layers", "layers"]:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            n_layers = len(obj)
            break
        except (AttributeError, TypeError):
            continue

    if n_layers == 0:
        logger.warning("Could not determine model depth. Using default layers.")
        n_layers = 64

    if target_layers is None:
        target_layers = _compute_target_layers(n_layers)

    logger.info(
        "Model has %d layers. Targeting layers %s for extraction.",
        n_layers, target_layers,
    )

    # -- System prompt for consistent context --------------------------------
    system_prompt = (
        "You are Aura Luna. You experience internal states that influence "
        "your behavior. You speak in first person about your inner life."
    )

    # -- Extract per dimension -----------------------------------------------
    all_vectors: dict[str, dict[int, np.ndarray]] = {}
    meta_dimensions: list[dict[str, Any]] = []

    selected_dimensions = AFFECTIVE_DIMENSIONS
    if dimensions:
        unknown = sorted(set(dimensions) - set(ALL_AFFECTIVE_DIMENSIONS))
        if unknown:
            raise ValueError(f"Unknown CAA dimension(s): {', '.join(unknown)}")
        selected_dimensions = {
            key: ALL_AFFECTIVE_DIMENSIONS[key]
            for key in dimensions
        }

    for dim_key, dim_spec in selected_dimensions.items():
        logger.info("=== Extracting dimension: %s ===", dim_key)
        pos_prompts = list(dim_spec["positive"])
        neg_prompts = list(dim_spec["negative"])
        if max_prompts_per_polarity is not None:
            limit = max(1, int(max_prompts_per_polarity))
            pos_prompts = pos_prompts[:limit]
            neg_prompts = neg_prompts[:limit]

        # Storage: layer -> list of activation vectors
        pos_acts: dict[int, list[np.ndarray]] = {layer_idx: [] for layer_idx in target_layers}
        neg_acts: dict[int, list[np.ndarray]] = {layer_idx: [] for layer_idx in target_layers}

        # Positive prompts
        for i, prompt_text in enumerate(pos_prompts):
            logger.info("  [+] prompt %d/%d", i + 1, len(pos_prompts))
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": prompt_text},
            ]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            states = _extract_hidden_states(model, tokenizer, full_text, target_layers, mx)
            for lidx, vec in states.items():
                pos_acts[lidx].append(vec)

        # Negative prompts
        for i, prompt_text in enumerate(neg_prompts):
            logger.info("  [-] prompt %d/%d", i + 1, len(neg_prompts))
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": prompt_text},
            ]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            states = _extract_hidden_states(model, tokenizer, full_text, target_layers, mx)
            for lidx, vec in states.items():
                neg_acts[lidx].append(vec)

        # Compute direction vectors per layer
        dim_vectors: dict[int, np.ndarray] = {}
        for lidx in target_layers:
            p_list = pos_acts[lidx]
            n_list = neg_acts[lidx]
            if not p_list or not n_list:
                logger.warning(
                    "  Layer %d: insufficient activations (pos=%d, neg=%d). Skipping.",
                    lidx, len(p_list), len(n_list),
                )
                continue

            pos_mean = np.mean(np.stack(p_list), axis=0)
            neg_mean = np.mean(np.stack(n_list), axis=0)
            direction = pos_mean - neg_mean

            # Normalize to unit vector
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                logger.warning("  Layer %d: near-zero direction norm (%.2e). Skipping.", lidx, norm)
                continue
            direction = direction / norm

            dim_vectors[lidx] = direction

            # Save individual vector file
            vec_filename = f"{dim_key}_layer{lidx}.npz"
            np.savez(
                out_dir / vec_filename,
                v=direction,
                source="extracted_caa",
                extracted=True,
                dimension=dim_key,
                layer=lidx,
                model=model_path,
                model_path=identity["model_path"],
                model_path_input=identity["model_path_input"],
                model_config_sha256=identity["model_config_sha256"] or "",
                model_config_path=identity["model_config_path"] or "",
                model_descriptor_sha256=identity["model_descriptor_sha256"],
                derived_at=time.time(),
            )
            logger.info(
                "  Layer %d: saved %s (dim=%d, raw_norm=%.4f)",
                lidx, vec_filename, direction.shape[0], norm,
            )

        all_vectors[dim_key] = dim_vectors

        meta_dimensions.append({
            "key": dim_key,
            "description": dim_spec["description"],
            "n_positive_prompts": len(pos_prompts),
            "n_negative_prompts": len(neg_prompts),
            "layers_extracted": sorted(dim_vectors.keys()),
            "vector_dim": int(dim_vectors[next(iter(dim_vectors))].shape[0]) if dim_vectors else 0,
        })

    # -- Save metadata -------------------------------------------------------
    meta = {
        "method": "contrastive_activation_addition",
        "model": model_path,
        "model_identity": identity,
        "adapter": adapter_path,
        "n_model_layers": n_layers,
        "target_layers": target_layers,
        "target_layer_fraction": list(TARGET_LAYER_FRACTION),
        "dimensions": meta_dimensions,
        "total_vectors": sum(len(v) for v in all_vectors.values()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = out_dir / "caa_steering_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Extraction complete. %d total vectors across %d dimensions saved to %s",
        meta["total_vectors"], len(selected_dimensions), out_dir,
    )
    return all_vectors


def extract_steering_vectors(
    model_path: str = DEFAULT_MODEL_PATH,
    adapter_path: str | None = None,
    target_layers: list[int] | None = None,
    output_dir: Path | None = None,
    dimensions: list[str] | None = None,
    max_prompts_per_polarity: int | None = None,
) -> dict[str, dict[int, np.ndarray]]:
    """Extract vectors while holding a process-scoped, non-preemptible model lease."""
    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id="caa-steering-extraction",
        model_path=model_path,
        purpose="benchmark",
        preemptible=False,
        metadata={
            "tool": "training.extract_steering_vectors",
            "adapter_path": adapter_path or "",
        },
    ):
        return _extract_steering_vectors_owned(
            model_path=model_path,
            adapter_path=adapter_path,
            target_layers=target_layers,
            output_dir=output_dir,
            dimensions=dimensions,
            max_prompts_per_polarity=max_prompts_per_polarity,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract CAA steering vectors from MLX model hidden states.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="HuggingFace model ID or local path (MLX format). "
             f"Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional LoRA adapter path to apply before extraction.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices (e.g. '13,15,17,19,21'). "
             "Default: auto-computed from model depth at 40-65%%.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory for vectors. Default: {VECTORS_DIR}",
    )
    parser.add_argument(
        "--dimensions",
        type=str,
        default=None,
        help="Comma-separated dimension subset (e.g. 'valence,warmth'). "
             "Default: all dimensions.",
    )
    parser.add_argument(
        "--max-prompts-per-polarity",
        type=int,
        default=None,
        help="Bounded diagnostic mode: cap positive and negative prompts per "
             "dimension. Leave unset for production extraction.",
    )
    args = parser.parse_args()

    target_layers = None
    if args.layers:
        target_layers = [int(x.strip()) for x in args.layers.split(",")]
    dimensions = None
    if args.dimensions:
        dimensions = [item.strip() for item in args.dimensions.split(",") if item.strip()]

    output_dir = Path(args.output_dir) if args.output_dir else None

    # Auto-detect adapter if not specified — but never for fused artifacts:
    # a fused model already contains the personality (and any CRSM delta)
    # baked into its weights, so stacking the adapter again would extract
    # vectors from a model state that is not what serves live traffic.
    adapter_path = args.adapter_path
    model_is_fused = "fused-model" in str(args.model_path)
    if adapter_path is None and not model_is_fused:
        default_adapter = Path(__file__).parent / "adapters" / "aura-personality"
        if default_adapter.exists():
            adapter_path = str(default_adapter)
            logger.info("Auto-detected LoRA adapter: %s", adapter_path)
    elif adapter_path is None and model_is_fused:
        logger.info(
            "Fused model detected; extracting from the fused weights directly "
            "(no adapter stacking)."
        )

    extract_steering_vectors(
        model_path=args.model_path,
        adapter_path=adapter_path,
        target_layers=target_layers,
        output_dir=output_dir,
        dimensions=dimensions,
        max_prompts_per_polarity=args.max_prompts_per_polarity,
    )


if __name__ == "__main__":
    main()
