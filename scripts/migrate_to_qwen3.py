#!/usr/bin/env python3
"""Migrate Aura's cortex from Qwen2.5 → Qwen3 (or any same-family base).

Swapping the base model is not free: the cortex is a LoRA *fuse* (Aura-32B), the CAA
steering vectors are keyed to the model's hidden dim + layer count, and the chat
template/tokenizer must stay in-family. This script makes the migration safe and
checkable rather than a pile of remembered shell commands.

Stages:
  1. PREFLIGHT  — disk/RAM headroom, base-model presence, LoRA adapter presence.
  2. FUSE       — re-fuse the Aura LoRA onto the new base (mlx_lm.fuse).  [heavy]
  3. ACTIVATE   — point training/fused-model/active.json at the new fused model.
  4. STEERING   — invalidate the CAA steering cache so vectors re-derive for the new
                  base on next boot (old vectors have the wrong hidden dim).

Dry-run by default — prints exactly what it would do. Pass --execute to perform the
ACTIVATE + STEERING steps (the safe, fast, reversible ones). The FUSE + any model
download stay manual/gated (long GPU ops); the exact commands are printed for you.

Usage:
    python scripts/migrate_to_qwen3.py --base models/Qwen3-32B-Instruct-4bit \\
        --adapter training/adapters/aura-lora --out training/fused-model/Aura-Qwen3-32B
    python scripts/migrate_to_qwen3.py ... --execute     # do ACTIVATE + STEERING
    python scripts/migrate_to_qwen3.py ... --execute --run-fuse   # also run the fuse
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime.resource_observation import (  # noqa: E402
    ResourceObserver,
    get_resource_observer,
)
from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402

ACTIVE_JSON = PROJECT_ROOT / "training" / "fused-model" / "active.json"


def _gb(n: float) -> str:
    return f"{n / (1024 ** 3):.1f} GB"


def _free_disk_gb(path: Path, *, observer: ResourceObserver | None = None) -> float:
    disk = (observer or get_resource_observer()).disk(path)
    return disk.free_bytes / (1024**3) if disk.available else 0.0


def _total_ram_gb(*, observer: ResourceObserver | None = None) -> float:
    memory = (observer or get_resource_observer()).memory()
    return memory.total_bytes / (1024**3) if memory.available else 0.0


def preflight(
    base: Path,
    adapter: Path,
    out: Path,
    *,
    observer: ResourceObserver | None = None,
) -> bool:
    print("── STAGE 1: PREFLIGHT ──────────────────────────────────────")
    ok = True
    observer = observer or get_resource_observer()
    provenance = observer.provenance
    print(
        "  Resource observation: "
        f"source={provenance.source.value} scenario={provenance.scenario_id}"
    )

    ram = _total_ram_gb(observer=observer)
    print(f"  Host RAM: {ram:.0f} GB")
    if ram and ram < 48:
        print("   <48 GB RAM: a 32B-8bit fuse is tight; prefer a 4-bit base.")

    free = _free_disk_gb(PROJECT_ROOT, observer=observer)
    print(f"  Free disk: {free:.0f} GB")
    if free < 60:
        print("   <60 GB free: a 32B fuse + base may not fit. Free space first.")
        ok = False

    if base.exists():
        print(f"  ✓ base model present: {base}")
    else:
        ok = False
        print(f"  ✗ base model MISSING: {base}")
        print("     Download e.g.:  huggingface-cli download mlx-community/Qwen3-32B-Instruct-4bit \\")
        print(f"                       --local-dir {base}")

    if adapter.exists():
        print(f"  ✓ LoRA adapter present: {adapter}")
    else:
        print(f"  ✗ LoRA adapter MISSING: {adapter}")
        print("     (If the cortex is already a full fuse with no separate adapter, you must")
        print("      re-train the LoRA on the new base — fusing requires the adapter weights.)")
        ok = False

    if out.exists():
        print(f"   output dir already exists (will be reused/overwritten): {out}")

    # Same-family tokenizer guard (best-effort): warn if base name leaves the Qwen line.
    if "qwen" not in base.name.lower():
        print("   base is not in the Qwen family — chat template / stop-sequence handling in")
        print("      mlx_worker is Qwen-tuned; expect tokenizer work for a cross-family base.")

    print(f"  Preflight: {'PASS' if ok else 'BLOCKED'}")
    return ok


def fuse_command(base: Path, adapter: Path, out: Path) -> list[str]:
    return [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", str(base),
        "--adapter-path", str(adapter),
        "--save-path", str(out),
    ]


def run_fuse(base: Path, adapter: Path, out: Path, *, execute: bool) -> bool:
    print("── STAGE 2: FUSE (heavy) ───────────────────────────────────")
    cmd = fuse_command(base, adapter, out)
    print("  command:", " ".join(cmd))
    if not execute:
        print("  (dry-run — not executed; pass --execute --run-fuse to run)")
        return True
    try:
        get_subprocess_gateway().run_model_blocking(
            cmd,
            cwd=PROJECT_ROOT,
            timeout=1800.0,
            offline_tooling=True,
            check=True,
            source="training_tooling:migrate_qwen3_fuse",
        )
        return out.exists()
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"  ✗ fuse failed: {exc}")
        return False


def activate(out: Path, base: Path, *, execute: bool) -> bool:
    print("── STAGE 3: ACTIVATE ───────────────────────────────────────")
    payload = {
        "active_model_path": str(out.resolve()),
        "base_model": str(base.resolve()),
        "fused_at": int(time.time()),
        "schema_version": 2,
        "size": "32B",
        "tag": "qwen3-migration",
    }
    print(f"  active.json -> {ACTIVE_JSON}")
    print(f"  payload: {json.dumps(payload)}")
    if not execute:
        print("  (dry-run — not written; pass --execute to apply)")
        return True
    try:
        if ACTIVE_JSON.exists():
            backup = ACTIVE_JSON.with_suffix(f".json.bak.{int(time.time())}")
            shutil.copy2(ACTIVE_JSON, backup)
            print(f"  backed up previous active.json -> {backup}")
        ACTIVE_JSON.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("  ✓ active.json updated")
        return True
    except OSError as exc:
        print(f"  ✗ failed to write active.json: {exc}")
        return False


def invalidate_steering(*, execute: bool) -> bool:
    print("── STAGE 4: STEERING re-derive ─────────────────────────────")
    try:
        from core.consciousness.affective_steering import SteeringVectorLibrary

        lib = SteeringVectorLibrary()
        cache_dir = Path(lib.cache_dir)
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        print(f"   could not locate steering cache via library ({exc}). Default likely:")
        cache_dir = Path.home() / ".aura" / "data" / "steering"
    print(f"  CAA steering cache: {cache_dir}")
    print("  Old vectors are keyed to the previous hidden dim and must be re-derived")
    print("  for the new base (the library derives lazily on next boot once cleared).")
    if not execute:
        print("  (dry-run — not cleared; pass --execute to clear so they re-derive)")
        return True
    try:
        if cache_dir.exists():
            stamp = cache_dir.with_name(cache_dir.name + f".pre_qwen3.{int(time.time())}")
            shutil.move(str(cache_dir), str(stamp))
            print(f"  ✓ moved old steering cache aside -> {stamp} (re-derives on next boot)")
        else:
            print("  (no existing steering cache to clear — will derive fresh on next boot)")
        return True
    except OSError as exc:
        print(f"  ✗ failed to clear steering cache: {exc}")
        return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Migrate Aura cortex to a new base model")
    p.add_argument("--base", default="models/Qwen3-32B-Instruct-4bit", help="New base model dir")
    p.add_argument("--adapter", default="training/adapters/aura-lora", help="Aura LoRA adapter dir")
    p.add_argument("--out", default="training/fused-model/Aura-Qwen3-32B", help="Fused output dir")
    p.add_argument("--execute", action="store_true", help="Perform ACTIVATE + STEERING (safe/reversible)")
    p.add_argument("--run-fuse", action="store_true", help="Also run the heavy mlx_lm fuse")
    args = p.parse_args(argv)

    base = (PROJECT_ROOT / args.base) if not Path(args.base).is_absolute() else Path(args.base)
    adapter = (PROJECT_ROOT / args.adapter) if not Path(args.adapter).is_absolute() else Path(args.adapter)
    out = (PROJECT_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    print(f"Qwen3 migration — base={base.name} adapter={adapter.name} out={out.name}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}\n")

    pf = preflight(base, adapter, out)
    fz = run_fuse(base, adapter, out, execute=args.execute and args.run_fuse)
    ac = activate(out, base, execute=args.execute and pf)
    st = invalidate_steering(execute=args.execute)

    print("\n── SUMMARY ─────────────────────────────────────────────────")
    print(f"  preflight={'ok' if pf else 'blocked'}  fuse={'ok' if fz else 'fail'}"
          f"  activate={'ok' if ac else 'fail'}  steering={'ok' if st else 'fail'}")
    if not args.execute:
        print("  DRY-RUN only. Re-run with --execute (and --run-fuse for the heavy step).")
    else:
        print("  Done. Re-derive a steering health check on next boot, then run the delta bench")
        print("  to confirm the new cortex ≥ the old one before trusting it.")
    return 0 if pf else 1


if __name__ == "__main__":
    raise SystemExit(main())
