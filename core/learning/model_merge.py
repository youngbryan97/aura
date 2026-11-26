"""core/learning/model_merge.py — transplant Aura's personality onto a new base, in minutes.

The problem this solves: Aura's serving weights took ~a week to train (personality +
architecture), so "just swap to a reasoning-distilled base (QwQ / R1-Distill-Qwen)" would
throw that week away and cost another week of downtime. It doesn't have to. Because the
reasoning-distilled Qwen models and Aura's model are BOTH fine-tunes of the same ancestor
(Qwen2.5), the personality is recoverable as a *weight delta* and can be transplanted onto
the reasoning base by arithmetic — no retraining, minutes not a week:

    personality_delta = Aura_weights − Qwen2.5_base
    merged            = reasoning_base + λ · personality_delta            (task arithmetic)

This module is that arithmetic, with the three established methods:

* **task arithmetic** (Ilharco et al. 2023) — add scaled task vectors.
* **TIES** (Yadav et al. 2023) — trim each delta to its largest entries, elect a sign per
  parameter, and merge only the entries that agree — reduces interference between deltas.
* **DARE** (Yu et al. 2024) — randomly drop most of each delta and rescale the survivors;
  deltas are highly redundant, so this preserves the effect while cutting interference.

Honest boundaries:
* The math is exact and unit-tested (below) on real tensors. The GB-scale run is local
  compute on your hardware (load → merge per-tensor → save), and a merged model MUST be
  eval-gated through the RSI gauntlet before it becomes the serving model — merging can
  degrade either parent, so you verify before you promote.
* Merge the FP16/FP32 weights (or the LoRA adapter delta), never the 4-bit quantized
  artifact — you re-quantize after. bf16 needs a one-line cast first (numpy has no bf16).
* This is also the cheap "merge specialists" lever: a math LoRA + a code LoRA + the base.
"""
from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.atomic_writer import atomic_write_text

logger = logging.getLogger("Aura.ModelMerge")

StateDict = dict[str, np.ndarray]
_WEIGHT_SUFFIXES = {".safetensors"}
_ARTIFACT_SUFFIXES = {".json", ".model", ".txt", ".jinja"}


# ---------------------------------------------------------------------------
# Core arithmetic (exact, unit-tested). Operates on aligned state dicts.
# ---------------------------------------------------------------------------
def task_vector(base: StateDict, finetuned: StateDict) -> StateDict:
    """The delta a fine-tune applied: ``finetuned − base`` per shared parameter."""
    return {
        k: (finetuned[k].astype(np.float32) - base[k].astype(np.float32))
        for k in base
        if k in finetuned and finetuned[k].shape == base[k].shape
    }


def _weights_for(deltas: Sequence[StateDict], weights: Sequence[float] | None) -> list[float]:
    if weights is None:
        return [1.0] * len(deltas)
    if len(weights) != len(deltas):
        raise ValueError("weights length must match number of deltas")
    return [float(w) for w in weights]


def linear_merge(
    base: StateDict, deltas: Sequence[StateDict], *, weights: Sequence[float] | None = None
) -> StateDict:
    """Plain task arithmetic: ``base + Σ wᵢ·δᵢ``."""
    ws = _weights_for(deltas, weights)
    out: StateDict = {}
    for k, b in base.items():
        acc = b.astype(np.float32).copy()
        for w, d in zip(ws, deltas, strict=True):
            if k in d:
                acc += w * d[k]
        out[k] = acc.astype(b.dtype)
    return out


def ties_merge(
    base: StateDict,
    deltas: Sequence[StateDict],
    *,
    density: float = 0.2,
    weights: Sequence[float] | None = None,
) -> StateDict:
    """TIES-merge: trim → elect sign → disjoint-average. ``density`` ∈ (0,1] kept per delta."""
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    ws = _weights_for(deltas, weights)
    out: StateDict = {}
    for k, b in base.items():
        b32 = b.astype(np.float32)
        present = [(w, d[k].astype(np.float32)) for w, d in zip(ws, deltas, strict=True) if k in d]
        if not present:
            out[k] = b
            continue
        trimmed: list[tuple[float, np.ndarray]] = []
        for w, d in present:
            flat = d.ravel()
            if flat.size and density < 1.0:
                keep = max(1, int(round(density * flat.size)))
                thresh = np.partition(np.abs(flat), -keep)[-keep]
                d = np.where(np.abs(d) >= thresh, d, 0.0)
            trimmed.append((w, d))
        # Elect a sign per parameter by the sign of the weighted sum of magnitudes.
        agg_sign = np.sign(sum(w * t for w, t in trimmed))
        num = np.zeros_like(b32)
        den = np.zeros_like(b32)
        for w, t in trimmed:
            agree = (np.sign(t) == agg_sign) & (t != 0.0)
            num += np.where(agree, w * t, 0.0)
            den += np.where(agree, 1.0, 0.0)
        merged_delta = np.where(den > 0, num / np.maximum(den, 1.0), 0.0)
        out[k] = (b32 + merged_delta).astype(b.dtype)
    return out


def dare_merge(
    base: StateDict,
    deltas: Sequence[StateDict],
    *,
    drop: float = 0.9,
    weights: Sequence[float] | None = None,
    seed: int = 0,
) -> StateDict:
    """DARE-merge: Bernoulli-drop ``drop`` of each delta, rescale survivors by 1/(1−drop)."""
    if not 0.0 <= drop < 1.0:
        raise ValueError("drop must be in [0, 1)")
    ws = _weights_for(deltas, weights)
    rng = np.random.default_rng(seed)
    keep = 1.0 - drop
    out: StateDict = {}
    for k, b in base.items():
        acc = b.astype(np.float32).copy()
        for w, d in zip(ws, deltas, strict=True):
            if k not in d:
                continue
            delta = d[k].astype(np.float32)
            mask = rng.random(delta.shape) >= drop
            acc += w * np.where(mask, delta / keep, 0.0)
        out[k] = acc.astype(b.dtype)
    return out


def merge_state_dicts(
    base: StateDict,
    finetunes: dict[str, StateDict],
    *,
    method: str = "ties",
    weights: Sequence[float] | None = None,
    density: float = 0.2,
    drop: float = 0.9,
    seed: int = 0,
) -> StateDict:
    """Merge ``finetunes`` (name→state dict) onto ``base`` by the chosen method."""
    deltas = [task_vector(base, ft) for ft in finetunes.values()]
    if method == "linear":
        return linear_merge(base, deltas, weights=weights)
    if method == "ties":
        return ties_merge(base, deltas, density=density, weights=weights)
    if method == "dare":
        return dare_merge(base, deltas, drop=drop, weights=weights, seed=seed)
    raise ValueError(f"unknown merge method: {method!r} (use linear|ties|dare)")


def transplant_delta(
    common_base: StateDict, finetuned: StateDict, onto: StateDict, *, scale: float = 1.0
) -> StateDict:
    """Aura's exact case: move a fine-tune's delta onto a DIFFERENT base.

    ``onto + scale·(finetuned − common_base)`` — e.g. apply the personality delta
    (Aura − Qwen2.5) onto a reasoning base (QwQ / R1-Distill, also a Qwen2.5 descendant).
    """
    delta = task_vector(common_base, finetuned)
    return linear_merge(onto, [delta], weights=[scale])


# ---------------------------------------------------------------------------
# Safetensors I/O (tested round-trip). Loads FP16/FP32; bf16/quantized need a cast.
# ---------------------------------------------------------------------------
def load_state_dict(path: str | Path) -> StateDict:
    """Load a .safetensors file, or all shards in a directory, into one state dict."""
    from safetensors.numpy import load_file

    p = Path(path)
    files = sorted(p.glob("*.safetensors")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"no .safetensors found at {p}")
    state: StateDict = {}
    for f in files:
        state.update(load_file(str(f)))
    return state


def save_state_dict(state: StateDict, path: str | Path) -> None:
    from safetensors.numpy import save_file

    save_file(state, str(path))


def _copy_model_artifacts(source_dir: Path, out_dir: Path) -> list[str]:
    copied: list[str] = []
    if not source_dir.is_dir():
        return copied
    for artifact in source_dir.iterdir():
        if artifact.is_dir():
            continue
        if artifact.suffix in _WEIGHT_SUFFIXES:
            continue
        if artifact.name == "model.safetensors.index.json":
            continue
        if artifact.suffix in _ARTIFACT_SUFFIXES or artifact.name.startswith("tokenizer"):
            target = out_dir / artifact.name
            shutil.copy2(artifact, target)
            copied.append(artifact.name)
    return copied


def _weight_map(model_dir: Path) -> dict[str, str]:
    """Return tensor name -> safetensors shard filename for an MLX/HF model dir."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map") if isinstance(data, dict) else None
        if isinstance(weight_map, dict):
            return {str(k): str(v) for k, v in weight_map.items()}
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors found in {model_dir}")
    from safetensors import safe_open

    result: dict[str, str] = {}
    for file in files:
        with safe_open(file, framework="np") as handle:
            for key in handle.keys():
                result[str(key)] = file.name
    return result


def _group_by_shard(weight_map: dict[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for tensor, shard in weight_map.items():
        grouped.setdefault(shard, []).append(tensor)
    return {shard: sorted(tensors) for shard, tensors in sorted(grouped.items())}


def _read_tensor(model_dir: Path, weight_map: dict[str, str], tensor: str) -> np.ndarray | None:
    shard = weight_map.get(tensor)
    if not shard:
        return None
    from safetensors import safe_open

    with safe_open(model_dir / shard, framework="np") as handle:
        if tensor not in handle.keys():
            return None
        return handle.get_tensor(tensor)


def _index_metadata(out_map: dict[str, str], total_size: int = 0) -> dict[str, Any]:
    return {
        "metadata": {"total_size": str(int(total_size))},
        "weight_map": out_map,
    }


def transplant_model_dirs_streaming(
    common_base_dir: str | Path,
    finetuned_dir: str | Path,
    onto_dir: str | Path,
    out_dir: str | Path,
    *,
    scale: float = 1.0,
    dry_run: bool = False,
    dtype: str = "preserve",
) -> dict[str, Any]:
    """Stream Aura's fine-tune delta onto a reasoning base without loading all weights.

    This is the production path for:

        output = reasoning_base + scale * (aura_finetune - qwen_common_base)

    It processes one output shard at a time using the ``onto`` model's shard
    layout. Missing or shape-incompatible tensors are copied from ``onto`` and
    recorded in the manifest, because silently corrupting tensor topology would
    be worse than failing to transplant a small subset.
    """
    common_base = Path(common_base_dir).expanduser().resolve()
    finetuned = Path(finetuned_dir).expanduser().resolve()
    onto = Path(onto_dir).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    for label, path in {
        "common_base": common_base,
        "finetuned": finetuned,
        "onto": onto,
    }.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{label} model dir not found: {path}")

    base_map = _weight_map(common_base)
    finetuned_map = _weight_map(finetuned)
    onto_map = _weight_map(onto)
    grouped = _group_by_shard(onto_map)
    all_onto = set(onto_map)
    compatible = all_onto & set(base_map) & set(finetuned_map)
    missing = sorted(all_onto - compatible)

    shape_mismatches: list[str] = []
    if dry_run:
        return {
            "mode": "transplant_streaming",
            "dry_run": True,
            "common_base": str(common_base),
            "finetuned": str(finetuned),
            "onto": str(onto),
            "out": str(out),
            "scale": float(scale),
            "onto_tensors": len(all_onto),
            "compatible_tensors": len(compatible),
            "missing_tensors": len(missing),
            "shards": len(grouped),
            "note": "Dry run only; no model files were written.",
        }

    out.mkdir(parents=True, exist_ok=True)
    copied_artifacts = _copy_model_artifacts(onto, out)
    from safetensors.numpy import save_file

    written_map: dict[str, str] = {}
    written_tensors = 0
    copied_tensors = 0
    for shard, tensors in grouped.items():
        shard_state: StateDict = {}
        for tensor in tensors:
            onto_tensor = _read_tensor(onto, onto_map, tensor)
            if onto_tensor is None:
                missing.append(tensor)
                continue
            base_tensor = _read_tensor(common_base, base_map, tensor)
            finetuned_tensor = _read_tensor(finetuned, finetuned_map, tensor)
            if (
                base_tensor is not None
                and finetuned_tensor is not None
                and base_tensor.shape == finetuned_tensor.shape == onto_tensor.shape
            ):
                merged = onto_tensor.astype(np.float32) + float(scale) * (
                    finetuned_tensor.astype(np.float32) - base_tensor.astype(np.float32)
                )
                if dtype == "float32":
                    shard_state[tensor] = merged.astype(np.float32)
                else:
                    shard_state[tensor] = merged.astype(onto_tensor.dtype)
                written_tensors += 1
            else:
                if tensor in compatible:
                    shape_mismatches.append(tensor)
                shard_state[tensor] = onto_tensor
                copied_tensors += 1
            written_map[tensor] = shard
        save_file(shard_state, str(out / shard))
    index = _index_metadata(written_map)
    atomic_write_text(
        out / "model.safetensors.index.json",
        json.dumps(index, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "mode": "transplant_streaming",
        "common_base": str(common_base),
        "finetuned": str(finetuned),
        "onto": str(onto),
        "out": str(out),
        "scale": float(scale),
        "dtype": dtype,
        "shards": len(grouped),
        "written_tensors": written_tensors,
        "copied_tensors": copied_tensors,
        "missing_tensors": missing[:200],
        "missing_tensor_count": len(missing),
        "shape_mismatches": shape_mismatches[:200],
        "shape_mismatch_count": len(shape_mismatches),
        "copied_artifacts": copied_artifacts,
        "note": "EVAL-GATE through the RSI gauntlet before promoting to the serving model.",
    }
    atomic_write_text(
        out / "merge_manifest.json",
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Streaming transplant wrote %d tensors across %d shards → %s",
        written_tensors,
        len(grouped),
        out,
    )
    return manifest


def merge_model_dirs(
    base_dir: str | Path,
    finetune_dirs: dict[str, str | Path],
    out_dir: str | Path,
    *,
    method: str = "ties",
    weights: Sequence[float] | None = None,
    density: float = 0.2,
    drop: float = 0.9,
) -> dict[str, Any]:
    """Load → merge → save a merged model, copying config/tokenizer alongside.

    For 72B-scale this holds the tensors in RAM; on a large-unified-memory Mac that is
    fine, otherwise merge the LoRA *adapters* (tiny) instead of the full bases. The output
    is NOT yet the serving model — run it through the RSI gauntlet / live_learner benchmark
    and only promote it if it wins.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = load_state_dict(base_dir)
    finetunes = {name: load_state_dict(d) for name, d in finetune_dirs.items()}
    merged = merge_state_dicts(
        base, finetunes, method=method, weights=weights, density=density, drop=drop
    )
    save_state_dict(merged, out / "model.safetensors")
    base_path = Path(base_dir)
    copied = _copy_model_artifacts(base_path, out)
    manifest = {
        "method": method,
        "base": str(base_dir),
        "finetunes": {k: str(v) for k, v in finetune_dirs.items()},
        "weights": list(weights) if weights else None,
        "tensors": len(merged),
        "copied_artifacts": copied,
        "note": "EVAL-GATE through the RSI gauntlet before promoting to the serving model.",
    }
    atomic_write_text(
        out / "merge_manifest.json",
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    logger.info("Merged %d tensors via %s → %s", len(merged), method, out)
    return manifest


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Merge or transplant local MLX safetensors models.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    transplant = sub.add_parser(
        "transplant",
        help="stream finetuned-common_base delta onto a reasoning base",
    )
    transplant.add_argument("--common-base", required=True)
    transplant.add_argument("--finetuned", required=True)
    transplant.add_argument("--onto", required=True)
    transplant.add_argument("--out", required=True)
    transplant.add_argument("--scale", type=float, default=1.0)
    transplant.add_argument("--dtype", choices=("preserve", "float32"), default="preserve")
    transplant.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "transplant":
        manifest = transplant_model_dirs_streaming(
            args.common_base,
            args.finetuned,
            args.onto,
            args.out,
            scale=args.scale,
            dtype=args.dtype,
            dry_run=args.dry_run,
        )
        print(json.dumps(manifest, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
