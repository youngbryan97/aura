#!/usr/bin/env python3
"""What the Qwen3.8-27B swap invalidates, decided by measurement.

The fused 27B is `qwen3_5`, and it keeps two numbers the old Qwen2.5-32B fuse
had: 64 layers and a hidden size of 5120. Everything else moved. That coincidence
is the hazard this inventory exists for -- a LoRA adapter trained on the 32B's
residual stream loads onto the 27B without a shape error and means nothing,
because equal dimensions are not equal representational geometry.

Three failure kinds are separated here, because they need different work:

  module_absent       the target does not exist. qwen3_5 gives a layer
                      ``self_attn`` only when (index + 1) % 4 == 0; the other
                      48 carry ``linear_attn``. An adapter on layer 16's
                      self_attn is addressing nothing.
  shape_break         the tensor cannot load. intermediate_size fell from
                      27648 to 17408, so every down_proj adapter is wrong.
  silent_basis_break  the tensor loads, runs, and is meaningless. This is the
                      dangerous one and it has no exception to catch it.

Nothing here rewrites evidence. Historical artifacts stay historical; the
inventory records what each one can still authorize, which for 32B-measured
evidence is never 27B serving.

    python tools/inventory_27b_migration.py --json OUT
    python tools/inventory_27b_migration.py --summary
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def installation_root() -> Path:
    """Where the checkpoints actually live.

    Models are large and untracked, so a `.claude/worktrees/` checkout has the
    source but not `training/fused-model/`. Git's common dir names the primary
    checkout, which is the one holding them.
    """
    if (ROOT / "training/fused-model/active.json").exists():
        return ROOT
    import subprocess

    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ROOT
    return Path(common).parent if common else ROOT


INSTALL = installation_root()

#: The checkpoint every 32B-era artifact was measured against. Named by the
#: CP568 activation receipt, which is the binding the runtime actually enforces.
LEGACY_MODEL_DIR = (
    INSTALL / "training/fused-model/Aura-32B-crsm-closeout-jul1-20260701-215118"
)
LEGACY_CONFIG_SHA = (
    "a4054d0840b87c2feabea6fb2014bcd5b672500b2ec4dfec6f63a31d7a6ed785"
)
LEGACY_WEIGHTS_INDEX_SHA = (
    "8ae48f398d91156dc01a99e58de936e4ddd94c0768f4d0e2e9537170aee60d0b"
)
LEGACY_PATH_TOKEN = "Aura-32B-crsm-closeout-jul1"

ACTIVE_MANIFEST = INSTALL / "training/fused-model/active.json"

EVIDENCE_ROOTS = (
    "artifacts/closeout/latent_cortex",
    "artifacts/current",
)
SOURCE_ASSET_ROOT = "core/brain/llm/latent_cortex/assets"


# ── Geometry ────────────────────────────────────────────────────────────


def _display(path: Path) -> str:
    for base in (ROOT, INSTALL):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    inner = config.get("text_config")
    return inner if isinstance(inner, dict) else config


def _geometry(config: dict[str, Any]) -> dict[str, Any]:
    text = _text_config(config)
    rope = text.get("rope_parameters")
    layer_types = text.get("layer_types")
    interval = text.get("full_attention_interval")
    layers = text.get("num_hidden_layers")
    heads = text.get("num_attention_heads")
    head_dim = text.get("head_dim")
    if head_dim is None and heads and text.get("hidden_size"):
        # Qwen2 leaves it implicit at hidden_size / heads.
        head_dim = int(text["hidden_size"]) // int(heads)
    kv_heads = text.get("num_key_value_heads")
    return {
        "model_type": config.get("model_type") or text.get("model_type"),
        "num_hidden_layers": layers,
        "hidden_size": text.get("hidden_size"),
        "intermediate_size": text.get("intermediate_size"),
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "kv_projection_width": (
            int(kv_heads) * int(head_dim) if kv_heads and head_dim else None
        ),
        "vocab_size": text.get("vocab_size"),
        "max_position_embeddings": text.get("max_position_embeddings"),
        "rope_theta": (rope or {}).get("rope_theta") or text.get("rope_theta"),
        "full_attention_interval": interval,
        "full_attention_layers": (
            layer_types.count("full_attention") if layer_types else None
        ),
        "linear_attention_layers": (
            layer_types.count("linear_attention") if layer_types else None
        ),
        "is_hybrid": bool(interval),
        "is_multimodal": bool(config.get("vision_config")),
    }


def attention_layer_indices(geometry: dict[str, Any]) -> set[int]:
    """Layers that carry ``self_attn``. Everything else carries ``linear_attn``.

    mlx_lm's DecoderLayer sets ``is_linear = (index + 1) % interval != 0``, so
    a dense checkpoint (no interval) is every layer and a hybrid one is every
    fourth.
    """
    layers = int(geometry.get("num_hidden_layers") or 0)
    interval = geometry.get("full_attention_interval")
    if not interval:
        return set(range(layers))
    return {i for i in range(layers) if (i + 1) % int(interval) == 0}


# ── Artifact classification ─────────────────────────────────────────────

VERDICTS = (
    "module_absent",
    "shape_break",
    "silent_basis_break",
    "token_id_bound",
    "identity_pin",
    "model_independent",
)


@dataclass
class Artifact:
    path: str
    kind: str
    verdict: str
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_fresh_training(self) -> bool:
        return self.verdict in {
            "module_absent",
            "shape_break",
            "silent_basis_break",
            "token_id_bound",
        }


def _lora_layer_index(key: str) -> int | None:
    parts = key.split(".")
    for index, part in enumerate(parts):
        if part == "layers" and index + 1 < len(parts):
            try:
                return int(parts[index + 1])
            except ValueError:
                return None
    return None


def classify_tensor_file(
    path: Path,
    shapes: dict[str, tuple[int, ...]],
    legacy: dict[str, Any],
    target: dict[str, Any],
) -> Artifact:
    """Decide what a trained tensor file can still be, by its own keys."""

    relative = str(_display(path))
    attention_layers = attention_layer_indices(target)
    legacy_dims = {
        int(legacy[name])
        for name in ("hidden_size", "intermediate_size", "kv_projection_width")
        if legacy.get(name)
    }
    target_dims = {
        int(target[name])
        for name in ("hidden_size", "intermediate_size", "kv_projection_width")
        if target.get(name)
    }

    absent: list[str] = []
    broken: list[dict[str, Any]] = []
    model_dim_keys: list[str] = []

    for key, shape in shapes.items():
        index = _lora_layer_index(key)
        if index is not None and ".self_attn." in key and index not in attention_layers:
            absent.append(key)
        if index is not None and ".mlp." in key:
            for dim in shape:
                if dim == legacy.get("intermediate_size") and dim not in target_dims:
                    broken.append({"key": key, "dim": dim})
        if any(dim in legacy_dims for dim in shape):
            model_dim_keys.append(key)

    # A file usually fails more than one way. The verdict is the most severe,
    # and the rest travel with it so the plan never has to re-derive them.
    also = {
        "module_absent": len(absent),
        "shape_break": len(broken),
        "silent_basis_break": len(model_dim_keys),
    }

    if absent:
        return Artifact(
            relative,
            "trained_tensors",
            "module_absent",
            "targets self_attn on layers that carry linear_attn on qwen3_5",
            {
                "example_keys": sorted(absent)[:4],
                "affected": len(absent),
                "also_detected": also,
            },
        )
    if broken:
        return Artifact(
            relative,
            "trained_tensors",
            "shape_break",
            "carries the legacy intermediate_size, which the target does not have",
            {"examples": broken[:4], "affected": len(broken), "also_detected": also},
        )
    if model_dim_keys:
        return Artifact(
            relative,
            "trained_tensors",
            "silent_basis_break",
            "dimensions still load; the basis is the 32B residual stream",
            {
                "example_keys": sorted(model_dim_keys)[:4],
                "affected": len(model_dim_keys),
                "also_detected": also,
            },
        )
    return Artifact(
        relative,
        "trained_tensors",
        "model_independent",
        "no dimension derived from either checkpoint",
        {"tensors": len(shapes)},
    )


def _read_shapes(path: Path) -> dict[str, tuple[int, ...]] | None:
    try:
        from safetensors import safe_open
    except ImportError:
        return None
    try:
        with safe_open(str(path), framework="numpy") as handle:
            return {
                key: tuple(handle.get_slice(key).get_shape())
                for key in handle.keys()
            }
    except (OSError, ValueError, RuntimeError):
        return None


def classify_json(path: Path, raw: str) -> Artifact | None:
    """Records that name the old checkpoint. None when a file is unrelated."""

    relative = str(_display(path))
    pins = LEGACY_CONFIG_SHA in raw or LEGACY_WEIGHTS_INDEX_SHA in raw
    names = LEGACY_PATH_TOKEN in raw or "Qwen2.5-32B" in raw
    if not (pins or names):
        return None
    is_activation = path.name in {"activation.json", "runtime_verification.json"}
    return Artifact(
        relative,
        "activation_package" if is_activation else "evidence_record",
        "identity_pin",
        (
            "runtime binding to the retired checkpoint; fail-closes on the 27B"
            if is_activation
            else "32B-measured evidence; valid as history, cannot authorize 27B"
        ),
        {"pins_hashes": pins, "names_path": names},
    )


# ── Tokenizer-bound source ──────────────────────────────────────────────

#: Modules that bind tokenizer ids as checkpoint identity. A vocabulary change
#: retires every id they recorded, so their contracts fail closed rather than
#: mis-decode -- correct, and it means the tissue behind them is unusable until
#: it is retrained against the new tokenizer.
TOKEN_BOUND_SOURCES = (
    "core/learning/recurrent_literal_grounding.py",
    "core/learning/recurrent_opcode_grounding.py",
    "core/learning/recurrent_answer_emission.py",
    "core/learning/recurrent_action_schema.py",
    "core/learning/recurrent_state_schema.py",
)


def modality_findings(active_path: Path, target: dict[str, Any]) -> list[Artifact]:
    """A fuse that says multimodal and carries no vision tower.

    The base checkpoint is a VLM: `Qwen3_5ForConditionalGeneration`, a 27-layer
    vision tower, image and video preprocessors. The fuse keeps the architecture
    string and the image/video token ids, drops `vision_config`, ships no
    preprocessor, and its weight index holds 1,847 tensors all under
    `language_model.`. Loading it through a vision path asks for a tower that is
    not there; the honest reading is a text-only fuse of a multimodal base.
    """
    findings: list[Artifact] = []
    try:
        config = json.loads((active_path / "config.json").read_text())
    except (OSError, ValueError):
        return findings
    architectures = config.get("architectures") or []
    declares_vision = any("ConditionalGeneration" in a for a in architectures) or (
        config.get("language_model_only") is False
    )
    try:
        index = json.loads((active_path / "model.safetensors.index.json").read_text())
        weights = index.get("weight_map") or {}
    except (OSError, ValueError):
        weights = {}
    vision_tensors = [k for k in weights if "visual" in k or "vision" in k]
    preprocessors = [
        name
        for name in ("preprocessor_config.json", "processor_config.json")
        if (active_path / name).exists()
    ]
    if declares_vision and not vision_tensors:
        findings.append(
            Artifact(
                str(active_path),
                "model_descriptor",
                "identity_pin",
                (
                    "declares a multimodal architecture and carries no vision "
                    "tensors; treat the fuse as text-only"
                ),
                {
                    "architectures": architectures,
                    "language_model_only": config.get("language_model_only"),
                    "has_vision_config": "vision_config" in config,
                    "vision_tensors": len(vision_tensors),
                    "preprocessors_present": preprocessors,
                    "tensor_count": len(weights),
                },
            )
        )
    return findings


def collect(target_geometry: dict[str, Any], legacy_geometry: dict[str, Any]):
    artifacts: list[Artifact] = []

    for root_name in EVIDENCE_ROOTS:
        # Most retained evidence is untracked, so it sits in the installation
        # rather than in a worktree checkout of the source.
        root = INSTALL / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.safetensors")):
            shapes = _read_shapes(path)
            if shapes is None:
                continue
            artifacts.append(
                classify_tensor_file(path, shapes, legacy_geometry, target_geometry)
            )
        for path in sorted(root.rglob("*.json")):
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            found = classify_json(path, raw)
            if found is not None:
                artifacts.append(found)

    asset_root = ROOT / SOURCE_ASSET_ROOT  # shipped in source, so ROOT
    for path in sorted(asset_root.rglob("*.safetensors")):
        shapes = _read_shapes(path)
        if shapes is None:
            continue
        artifacts.append(
            classify_tensor_file(path, shapes, legacy_geometry, target_geometry)
        )

    for relative in TOKEN_BOUND_SOURCES:
        path = ROOT / relative
        if not path.exists():
            continue
        artifacts.append(
            Artifact(
                relative,
                "grounding_contract",
                "token_id_bound",
                "binds tokenizer ids as checkpoint identity; the vocabulary moved",
                {
                    "legacy_vocab_size": legacy_geometry.get("vocab_size"),
                    "target_vocab_size": target_geometry.get("vocab_size"),
                },
            )
        )

    return artifacts


def build() -> dict[str, Any]:
    if not LEGACY_MODEL_DIR.exists():
        raise SystemExit(f"legacy checkpoint not found: {LEGACY_MODEL_DIR}")
    legacy = _geometry(json.loads((LEGACY_MODEL_DIR / "config.json").read_text()))

    manifest = json.loads(ACTIVE_MANIFEST.read_text())
    active_path = Path(manifest["active_model_path"])
    target = _geometry(json.loads((active_path / "config.json").read_text()))

    changed = sorted(
        name
        for name in legacy
        if legacy.get(name) != target.get(name)
    )
    unchanged_but_reused = sorted(
        name
        for name in ("num_hidden_layers", "hidden_size", "kv_projection_width")
        if legacy.get(name) == target.get(name) and legacy.get(name) is not None
    )

    artifacts = collect(target, legacy)
    artifacts.extend(modality_findings(active_path, target))
    counts: dict[str, int] = {}
    for artifact in artifacts:
        counts[artifact.verdict] = counts.get(artifact.verdict, 0) + 1

    attention = sorted(attention_layer_indices(target))
    return {
        "schema": "aura.rlc.27b_migration_inventory.v1",
        "legacy_checkpoint": {
            "path": str(LEGACY_MODEL_DIR),
            "config_sha256": LEGACY_CONFIG_SHA,
            "geometry": legacy,
        },
        "target_checkpoint": {
            "path": str(active_path),
            "geometry": target,
            "attention_layer_indices": attention,
            "linear_attention_layer_indices": sorted(
                set(range(int(target["num_hidden_layers"]))) - set(attention)
            ),
        },
        "geometry_delta": {
            "changed_fields": changed,
            "unchanged_fields_that_invite_silent_reuse": unchanged_but_reused,
        },
        "verdict_counts": counts,
        "artifacts": [asdict(a) for a in artifacts],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    inventory = build()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(inventory, indent=1, sort_keys=True))
        print(f"wrote {args.json}")
    if args.summary or not args.json:
        legacy = inventory["legacy_checkpoint"]["geometry"]
        target = inventory["target_checkpoint"]["geometry"]
        print(f"{'field':32s} {'32B fuse':>14s} {'27B fuse':>14s}")
        for name in legacy:
            mark = "" if legacy[name] == target.get(name) else "  changed"
            print(f"{name:32s} {str(legacy[name]):>14s} {str(target.get(name)):>14s}{mark}")
        print()
        for verdict, count in sorted(inventory["verdict_counts"].items()):
            print(f"  {verdict:22s} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
