#!/usr/bin/env python3
"""Unattended orchestrator for Aura's LoRA pipeline.

Drives training/train_and_fuse.py without modifying it. Adds:
  * Resume via training/resume_training.py if a partial adapter exists.
  * State persistence to training_state.json (idempotent re-spawns).
  * SIGTERM/SIGINT: writes final snapshot, then exits cleanly.
  * Dry-run short-circuit: --skip-train + --skip-dataset + --tag dryrun*
    exits 0 without touching the trainer (smoke-test).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
REPO_DIR = TRAINING_DIR.parent
ADAPTER_DIR = TRAINING_DIR / "adapters" / "aura-personality"
STATE_FILE = ADAPTER_DIR / "training_state.json"
TRAIN_AND_FUSE = TRAINING_DIR / "train_and_fuse.py"
RESUME_SCRIPT = TRAINING_DIR / "resume_training.py"
CHECKPOINT_GLOB = "*_adapters.safetensors"

_shutdown = threading.Event()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def latest_checkpoint() -> tuple[Path | None, int]:
    """Highest-numbered N_adapters.safetensors → (path, iter)."""
    if not ADAPTER_DIR.exists():
        return None, 0
    cands = [(int(c.stem.split("_", 1)[0]), c) for c in ADAPTER_DIR.glob(CHECKPOINT_GLOB)
             if c.stem.split("_", 1)[0].isdigit()]
    if not cands:
        return None, 0
    n, path = max(cands)
    return path, n


def has_partial_run() -> bool:
    return latest_checkpoint()[0] is not None


def update_state(*, started_at: str, **extra: object) -> dict:
    """Snapshot checkpoint progress + extras to STATE_FILE atomically."""
    ckpt, last_iter = latest_checkpoint()
    state: dict = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
        except Exception:
            state = {}
    state.update({
        "started_at": state.get("started_at") or started_at,
        "last_iter": last_iter,
        "last_checkpoint_path": str(ckpt) if ckpt else None,
        "last_heartbeat": _now_iso(),
    })
    state.update(extra)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(tmp, STATE_FILE)
    return state


def _spawn(cmd: list[str], *, started_at: str) -> int:
    """Run subprocess, heartbeat state, honour _shutdown."""
    print(f"[orch] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, cwd=str(REPO_DIR))
    try:
        while True:
            try:
                return proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                update_state(started_at=started_at, phase="running")
                if _shutdown.is_set():
                    print("[orch] shutdown — terminating subprocess")
                    proc.terminate()
                    try:
                        return proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        return proc.wait()
    finally:
        if proc.poll() is None:
            proc.terminate()


def run_resume(*, started_at: str) -> int:
    if not RESUME_SCRIPT.exists():
        print(f"[orch] {RESUME_SCRIPT.name} missing; skipping resume.")
        return 0
    print(f"[orch] partial run detected — resuming via {RESUME_SCRIPT.name}")
    update_state(started_at=started_at, phase="resume")
    rc = _spawn([sys.executable, str(RESUME_SCRIPT)], started_at=started_at)
    update_state(started_at=started_at, phase="resume_done", last_resume_rc=rc)
    return rc


def run_train_and_fuse(args: argparse.Namespace, *, started_at: str) -> int:
    if not TRAIN_AND_FUSE.exists():
        print(f"[orch] {TRAIN_AND_FUSE.name} missing; cannot proceed.")
        return 3
    cmd: list[str] = [sys.executable, str(TRAIN_AND_FUSE)]
    if args.skip_dataset: cmd.append("--skip-dataset")
    if args.skip_train: cmd.append("--skip-train")
    if args.base_model: cmd += ["--base-model", args.base_model]
    if args.tag: cmd += ["--tag", args.tag]
    update_state(started_at=started_at, phase="train_and_fuse")
    rc = _spawn(cmd, started_at=started_at)
    update_state(started_at=started_at, phase="train_and_fuse_done", last_pipeline_rc=rc)
    return rc


def _install_signal_handlers(started_at: str) -> None:
    def _handler(signum, _frame):  # noqa: ANN001
        print(f"[orch] signal {signum} — writing final snapshot")
        _shutdown.set()
        update_state(started_at=started_at, phase="signal_exit", last_signal=int(signum))

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="")
    p.add_argument("--base-model", default="")
    p.add_argument("--skip-dataset", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    return p.parse_args(argv)


def is_dryrun(args: argparse.Namespace) -> bool:
    return args.skip_train and args.skip_dataset and args.tag.startswith("dryrun")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started_at = _now_iso()
    _install_signal_handlers(started_at)
    print(f"[orch] started_at={started_at} args={vars(args)}")
    update_state(started_at=started_at, phase="boot", args=vars(args))

    if is_dryrun(args):
        print("[orch] dryrun mode (skip-train + skip-dataset + dryrun* tag) — clean exit.")
        update_state(started_at=started_at, phase="dryrun_done")
        return 0

    if has_partial_run() and not args.skip_train:
        rc = run_resume(started_at=started_at)
        if rc != 0:
            print(f"[orch] resume failed (rc={rc}) — wrapper will retry.")
            return rc
        if _shutdown.is_set():
            return 130

    rc = run_train_and_fuse(args, started_at=started_at)
    if rc != 0:
        print(f"[orch] pipeline failed (rc={rc}) — wrapper will retry.")
        return rc

    update_state(started_at=started_at, phase="complete")
    print("[orch] pipeline completed cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
