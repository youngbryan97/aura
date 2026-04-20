"""LoRA finetuning with recurrent-depth enabled DURING training.

Per arXiv 2511.07384 ("Teaching Pretrained LMs to Think Deeper with
Retrofitted Recurrence"), inference-only layer looping does not unlock
the real gains — the adapter must be trained with the looped forward
pass so it learns intermediate representations that benefit from
iterative refinement.

This wrapper:
  1. Sets the recurrent-depth env (default: 2 loops).
  2. Monkey-patches `mlx_lm.lora.load` to apply `apply_for_model` to the
     freshly loaded base model BEFORE LoRA adapters are attached. This is
     a pre-training, script-level patch — not a runtime shim.
  3. Hands control to `mlx_lm.lora.main()`, which reads CLI args, loads
     the model (through our patched path), attaches LoRA, optionally
     resumes from an adapter file, and trains.

Usage mirrors `python -m mlx_lm.lora …` exactly — just run this script
with the same args instead.

Because the existing iter-3000 adapter was trained WITHOUT loops, resuming
under loops is itself a curriculum step (n=1 → n=2). Expect the first
~50–200 iters to show a loss bump while the adapter realigns to the
deeper forward pass; this is the retrofit paper's whole point.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path.home() / ".aura" / "live-source"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Default to 2 loops unless explicitly overridden.
os.environ.setdefault("AURA_RECURRENT_LOOPS", "2")

# Apply recurrent depth before training. This must happen AFTER the model
# is loaded (so model.model.layers exists) but BEFORE LoRA layer conversion
# (so the LoRA-wrapped Linears inherit the new forward pass). We achieve
# this by patching the `load` symbol inside mlx_lm.lora's namespace.
from mlx_lm.utils import load as _orig_load          # noqa: E402
from core.brain.llm.recurrent_depth import apply_for_model  # noqa: E402
import mlx_lm.lora as _lora                           # noqa: E402


def _patched_load(*args, **kwargs):
    model, tokenizer = _orig_load(*args, **kwargs)
    print("🧠 [Training] Applying recurrent depth to base model…", flush=True)
    applied = apply_for_model(model)
    loops = os.environ.get("AURA_RECURRENT_LOOPS", "<unset>")
    if applied:
        print(f"   ✅ Recurrent depth ACTIVE during training (loops={loops})", flush=True)
    else:
        print(f"   ⚠️  Recurrent depth NOT applied (loops={loops}). Training will "
              "proceed with standard forward pass.", flush=True)
    return model, tokenizer


# Patch both the canonical import site and the name inside mlx_lm.lora.
_lora.load = _patched_load


if __name__ == "__main__":
    _lora.main()
