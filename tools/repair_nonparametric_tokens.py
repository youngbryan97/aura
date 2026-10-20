#!/usr/bin/env python3
"""Give a kNN store's entries the text of the token each one predicts, from the ids it kept, and put it back.

Her recall store of 18 October (50,000 entries) was written by an encoder with no ``decode_token``, so every entry
kept its token id and "" for its text; at the next boot the worker found only 29 entries it could recall from and
quarantined the store (``*.unusable-<stamp>``). The ids were right. This decodes each with the tokenizer of the model
the store was built on (the tokenizer only; the model is not loaded), checks the store would now pass the worker's
own bar, and restores it under its active name. The quarantined meta is copied first; the keys are moved, unchanged.

    repair_nonparametric_tokens.py --meta PATH.meta.json.unusable-STAMP --model DIR            # what would change
    repair_nonparametric_tokens.py --meta PATH.meta.json.unusable-STAMP --model DIR --apply    # repair and restore
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def active_name(quarantined: Path) -> Path:
    """``x.meta.json.unusable-STAMP`` -> ``x.meta.json``."""
    name = quarantined.name
    return quarantined.with_name(name[: name.index(".unusable-")] if ".unusable-" in name else name)


def decoded(token_ids: list[int], decode) -> list[str]:
    return [str(decode([int(i)])) for i in token_ids]


def passes_the_worker(tokens: list[str]) -> tuple[bool, int]:
    """Whether the worker would let this store steer generation, by its own bar (core/brain/nonparametric_worker.py)."""
    from core.brain.nonparametric_worker import _MIN_USABLE_ENTRIES, _MIN_USABLE_ENTRY_FRACTION

    usable = sum(1 for token in tokens if str(token or "").strip())
    return usable >= _MIN_USABLE_ENTRIES and usable >= len(tokens) * _MIN_USABLE_ENTRY_FRACTION, usable


def repair(meta_path: Path, decode, *, apply: bool) -> dict:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tokens = decoded(meta["token_ids"], decode)
    ok, usable = passes_the_worker(tokens)
    before = sum(1 for token in meta.get("tokens") or [] if str(token or "").strip())
    report = {"entries": len(tokens), "usable_before": before, "usable_after": usable, "passes": ok}
    keys_quarantined = meta_path.with_name(meta_path.name.replace(".meta.json.", ".keys.npy.", 1))
    meta_active, keys_active = active_name(meta_path), active_name(keys_quarantined)
    if not apply:
        return report
    if not ok:
        raise SystemExit(f"still below the worker's bar after decoding: {report}")
    if meta_active.exists() or keys_active.exists():
        raise SystemExit(f"a store is already active at {meta_active.parent}; not overwriting it")
    shutil.copy2(meta_path, meta_path.with_name(meta_path.name + f".before-token-repair-{time.strftime('%Y%m%d-%H%M%S')}"))
    meta["tokens"] = tokens
    tmp = meta_active.with_name(meta_active.name + ".tmp")
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    keys_quarantined.rename(keys_active)
    tmp.replace(meta_active)
    return {**report, "restored": str(meta_active)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="the model the store was built on (its tokenizer)")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(args.model.expanduser())
    print(json.dumps(repair(args.meta.expanduser(), tokenizer.decode, apply=args.apply)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
