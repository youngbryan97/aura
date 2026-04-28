#!/usr/bin/env python3
"""End-to-end LoRA training + fuse + auto-pickup pipeline.

What this script does, in one run:

1. Optionally rebuild the training dataset (build_dataset_v3) so the
   personality + architecture corpus is fresh.
2. Run the LoRA fine-tune (mlx_lm.lora) with the existing
   training/finetune_lora.py hyperparameters.
3. Fuse the resulting adapter into the base model with mlx_lm.fuse,
   producing a new versioned directory under training/fused-model/.
4. Write training/fused-model/active.json — a small manifest that Aura's
   model_registry reads on boot to pick up the newest fused model
   automatically. No .env edit required.
5. Verify the new model loads, then atomically swap the manifest.

After this script finishes, restarting Aura will use the new weights.
The previous fused model directory is kept (under a versioned name) so
you can roll back by editing active.json or pointing AURA_LLM__MLX_MODEL_PATH.

Usage:
    python training/train_and_fuse.py
    python training/train_and_fuse.py --skip-dataset      # reuse existing data
    python training/train_and_fuse.py --skip-train        # only fuse + publish
    python training/train_and_fuse.py --tag mythos-v1     # name this run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TRAINING_DIR = Path(__file__).parent
REPO_DIR = TRAINING_DIR.parent
DATA_DIR = TRAINING_DIR / "data"
ADAPTER_DIR = TRAINING_DIR / "adapters" / "aura-personality"
FUSED_BASE_DIR = TRAINING_DIR / "fused-model"
ACTIVE_MANIFEST = FUSED_BASE_DIR / "active.json"

DEFAULT_BASE_MODEL = REPO_DIR / "models" / "Qwen2.5-32B-Instruct-8bit"


def _run(cmd: list[str], *, timeout: float | None = None) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, timeout=timeout)
    return result.returncode


def build_dataset() -> None:
    builder = TRAINING_DIR / "build_dataset_v3.py"
    if not builder.exists():
        print(f"  Dataset builder not found at {builder}; skipping.")
        return
    rc = _run([sys.executable, str(builder)])
    if rc != 0:
        sys.exit(f"Dataset build failed (exit {rc}).")


def train_lora(*, base_model: Path) -> None:
    finetune = TRAINING_DIR / "finetune_lora.py"
    if not finetune.exists():
        sys.exit(f"finetune_lora.py not found at {finetune}.")
    # Pass base_model through env so finetune_lora's find_base_model() picks
    # the right size — same script supports 32B, 72B, 14B, 7B, etc.
    env = os.environ.copy()
    env["AURA_LORA_BASE_MODEL"] = str(base_model)
    print(f"\n$ {sys.executable} {finetune}  (AURA_LORA_BASE_MODEL={base_model})", flush=True)
    result = subprocess.run([sys.executable, str(finetune)], env=env)
    if result.returncode != 0:
        sys.exit(f"LoRA fine-tune failed (exit {result.returncode}).")


def _model_size_tag(base_model: Path) -> str:
    """Derive a short size tag from the base-model directory name ('32B',
    '72B', '14B', '7B'). Falls back to 'model' when no size token matches."""
    name = base_model.name.lower()
    for size in ("72b", "32b", "14b", "8b", "7b", "3b", "1.5b", "0.5b"):
        if size in name:
            return size.upper().replace(".", "_")
    return "model"


def fuse_adapter(*, base_model: Path, tag: str) -> Path:
    """mlx_lm fuse base_model + adapter → versioned fused-model dir."""
    if not (ADAPTER_DIR / "adapters.safetensors").exists():
        sys.exit(
            f"No adapter found at {ADAPTER_DIR}/adapters.safetensors — "
            "run training first or pass --skip-train only after a previous train."
        )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    size_tag = _model_size_tag(base_model)
    fused_name = (
        f"Aura-{size_tag}-{tag}-{timestamp}" if tag
        else f"Aura-{size_tag}-{timestamp}"
    )
    fused_path = FUSED_BASE_DIR / fused_name
    fused_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nFusing → {fused_path}")
    rc = _run(
        [
            sys.executable,
            "-m",
            "mlx_lm",
            "fuse",
            "--model",
            str(base_model),
            "--adapter-path",
            str(ADAPTER_DIR),
            "--save-path",
            str(fused_path),
        ],
        timeout=1800,
    )
    if rc != 0:
        sys.exit(f"Fuse failed (exit {rc}).")
    if not fused_path.exists() or not any(fused_path.iterdir()):
        sys.exit(f"Fuse claimed success but {fused_path} is empty.")
    return fused_path


def verify_load(fused_path: Path) -> None:
    """Smoke-test: tokenize one prompt to confirm the fused model is loadable."""
    print(f"\nVerifying fused model loads: {fused_path}")
    code = (
        "import sys\n"
        "from mlx_lm import load\n"
        f"model, tok = load({str(fused_path)!r})\n"
        "ids = tok.encode('Hello')\n"
        "print(f'OK: tokenized {len(ids)} tokens, vocab_size={tok.vocab_size}')\n"
    )
    rc = _run([sys.executable, "-c", code], timeout=600)
    if rc != 0:
        sys.exit(f"Verification load failed (exit {rc}).")


def publish_manifest(fused_path: Path, *, tag: str, base_model: Path) -> None:
    """Atomically write active.json so Aura's next boot uses the new model.

    The manifest now includes the base-model size so downstream RAM-aware
    routing (model_registry, inference_gate) can branch on it without
    re-parsing the directory name."""
    FUSED_BASE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "active_model_path": str(fused_path),
        "fused_at": int(time.time()),
        "tag": tag or "",
        "size": _model_size_tag(base_model),
        "base_model": str(base_model),
        "schema_version": 2,
    }
    tmp = ACTIVE_MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, ACTIVE_MANIFEST)
    print(f"\nWrote active manifest: {ACTIVE_MANIFEST}")
    print(json.dumps(manifest, indent=2))
    print(
        "\nNext Aura boot will use this fused model automatically. "
        "If AURA_LLM__MLX_MODEL_PATH is set in .env it still wins — "
        "remove or update that line to let the manifest drive."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--base-model",
        default=os.environ.get("AURA_LORA_BASE_MODEL", str(DEFAULT_BASE_MODEL)),
    )
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    base_model = Path(args.base_model)
    if not base_model.exists():
        sys.exit(f"Base model not found: {base_model}")

    print("=" * 60)
    print("  AURA TRAIN → FUSE → PUBLISH PIPELINE")
    print("=" * 60)
    print(f"  base_model: {base_model}")
    print(f"  adapter:    {ADAPTER_DIR}")
    print(f"  output dir: {FUSED_BASE_DIR}")
    print(f"  tag:        {args.tag or '(none)'}")
    print("=" * 60)

    if not args.skip_dataset:
        build_dataset()
    if not args.skip_train:
        train_lora(base_model=base_model)
    fused_path = fuse_adapter(base_model=base_model, tag=args.tag)
    verify_load(fused_path)
    publish_manifest(fused_path, tag=args.tag, base_model=base_model)


if __name__ == "__main__":
    main()
