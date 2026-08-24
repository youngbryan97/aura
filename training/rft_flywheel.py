#!/usr/bin/env python3
"""Rejection-sampling fine-tuning flywheel — verifier-clean derivations → weights.

The compounding self-improvement loop: the reasoning amplifier's truth-engine
verifiers already tag every candidate as verified-correct or verified-wrong
(RLVR data, harvested into the verifiable-preference store). This closes the
loop by turning the accumulated verifier-CLEAN derivations into an SFT dataset
and running the proven train → fuse → publish pipeline.

Why SFT-on-chosen (rejection-sampling fine-tuning) rather than DPO: mlx_lm's
LoRA has no native DPO/ORPO mode, and for VERIFIABLE domains the verifier is
ground truth — training directly on verified-correct derivations is the
cleaner, better-founded signal (this is the RFT/STaR recipe). The rejected
half of each pair is retained in the store for a future DPO run if the
trainer gains that mode.

Operational model mirrors the CRSM delta (which closed successfully):
quiet-window gated, preflight-guarded, disruption run explicit — never
auto-launch a Cortex train beside the live instance.

    python training/rft_flywheel.py --preflight-only   # gate check + dataset stats
    python training/rft_flywheel.py --build-dataset     # write the SFT dataset
    python training/rft_flywheel.py --emit-command      # print the train/fuse command
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RFT_DATA_DIR = ROOT / "training" / "data" / "rft_flywheel"
_MIN_ROWS = 64  # below this the delta is noise; keep accumulating
_MAX_ROWS = 2000
_MIN_CONFIDENCE = 0.55

_RFT_SYSTEM = (
    "You are Aura Luna. You reason carefully and your answers are checked "
    "against ground truth. Produce the verified-correct derivation."
)


def _dedup_key(prompt: str, chosen: str) -> str:
    return hashlib.sha256(f"{prompt}␟{chosen}".encode()).hexdigest()[:16]


def gather_verified_rows(*, min_confidence: float = _MIN_CONFIDENCE, limit: int = _MAX_ROWS) -> list[dict]:
    """Collect verifier-clean (prompt, chosen) SFT rows from the preference store.

    Each stored pair carries a chosen (verified-correct) and rejected
    (verified-wrong) attempt for the same prompt. We take the chosen side,
    de-duplicated, above a confidence floor.
    """
    from core.learning.verifiable_preference_harness import (
        get_verifiable_preference_harness,
    )

    raw = get_verifiable_preference_harness().export_dpo_rows(limit=limit * 4)
    seen: set[str] = set()
    rows: list[dict] = []
    for item in raw:
        prompt = str(item.get("prompt") or "").strip()
        chosen = str(item.get("chosen") or "").strip()
        if not prompt or not chosen:
            continue
        # Confidence, when present, gates the row; absent confidence is
        # treated as "verified but unscored" and still admitted.
        try:
            conf = float(item.get("confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        if conf < min_confidence:
            continue
        key = _dedup_key(prompt, chosen)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": _RFT_SYSTEM},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen},
                ]
            }
        )
        if len(rows) >= limit:
            break
    return rows


def build_dataset(*, output_dir: Path = RFT_DATA_DIR, valid_fraction: float = 0.1) -> dict:
    """Write an MLX-compatible train/valid split; returns a provenance manifest."""
    rows = gather_verified_rows()
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    n_valid = max(1, int(n * valid_fraction)) if n >= 10 else 0
    train_rows = rows[n_valid:]
    valid_rows = rows[:n_valid] if n_valid else rows[: max(1, n // 5)]

    def _write(path: Path, data: list[dict]) -> str:
        payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in data)
        path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
        return hashlib.sha256(payload.encode()).hexdigest()

    manifest = {
        "schema": "aura.rft_flywheel.dataset.v1",
        "total_rows": n,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "train_sha256": _write(output_dir / "train.jsonl", train_rows),
        "valid_sha256": _write(output_dir / "valid.jsonl", valid_rows),
        "min_confidence": _MIN_CONFIDENCE,
        "source": "verifiable_preference_harness.chosen (verifier-clean)",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def flywheel_gate() -> dict:
    """Is a flywheel delta warranted AND safe to run right now?"""
    from core.learning.verifiable_preference_harness import (
        get_verifiable_preference_harness,
    )

    pending = int(get_verifiable_preference_harness().pending_count())
    rows = gather_verified_rows()
    enough = len(rows) >= _MIN_ROWS

    preflight: dict = {"passed": False, "reason": "not_checked"}
    if enough:
        try:
            from training.train_and_fuse import get_default_base_model, training_preflight

            preflight = training_preflight(
                base_model=get_default_base_model(), skip_train=False, crsm_delta=False
            )
        except (ImportError, RuntimeError, OSError, ValueError, SystemExit) as exc:
            preflight = {
                "passed": False,
                "reason": f"preflight_error:{type(exc).__name__}:{exc}",
            }
    else:
        preflight = {"passed": False, "reason": "insufficient_verified_rows"}

    return {
        "pending_pairs": pending,
        "verified_rows_available": len(rows),
        "min_rows_required": _MIN_ROWS,
        "enough_data": enough,
        "preflight_passed": bool(preflight.get("passed")),
        "preflight": preflight,
        "ready": bool(enough and preflight.get("passed")),
    }


def emit_train_command(tag: str = "rft-flywheel") -> str:
    """The exact command to run in a quiet window with the Cortex lane free."""
    return (
        f"{sys.executable} training/train_and_fuse.py --crsm-delta "
        f"--tag {tag} "
        f"# after: python training/rft_flywheel.py --build-dataset "
        f"and pointing AURA_CRSM_DELTA_DATA_DIR at {RFT_DATA_DIR}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--build-dataset", action="store_true")
    parser.add_argument("--emit-command", action="store_true")
    parser.add_argument("--tag", default="rft-flywheel")
    args = parser.parse_args()

    if args.build_dataset:
        manifest = build_dataset()
        print(json.dumps(manifest, indent=2))
        if manifest["total_rows"] < _MIN_ROWS:
            print(
                f"\n⚠️  Only {manifest['total_rows']} verified rows "
                f"(need {_MIN_ROWS}); keep accumulating before training.",
                file=sys.stderr,
            )
        return 0

    gate = flywheel_gate()
    print(json.dumps(gate, indent=2))
    if args.emit_command and gate["ready"]:
        print("\n" + emit_train_command(args.tag))
    return 0 if gate["ready"] or args.preflight_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
