#!/usr/bin/env python3
"""Compare complete semantic feature cohorts without loading model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def compare_bundles(name: str, old: Any, new: Any) -> dict[str, Any]:
    """Compare measured content, keeping construction lineage separate."""

    def index(bundle: Any) -> dict[str, Any]:
        rows = {}
        for item in bundle.examples:
            source = item.metadata["source_text_sha256"]
            if source in rows:
                raise ValueError(f"{name} repeats a source text")
            rows[source] = item.metadata
        return rows

    old_rows, new_rows = index(old), index(new)
    shared = old_rows.keys() & new_rows.keys()
    split_changes = sorted(source for source in shared if old_rows[source]["split"] != new_rows[source]["split"])
    token_changes = sorted(source for source in shared if old_rows[source]["token_ids_sha256"] != new_rows[source]["token_ids_sha256"])
    hidden_changes = sorted(source for source in shared if old_rows[source]["hidden_states_sha256"] != new_rows[source]["hidden_states_sha256"])
    lineage_changes = sorted(source for source in shared if any(
        old_rows[source].get(field) != new_rows[source].get(field)
        for field in ("construction_id", "contrast_id")
    ))
    return {
        "old_manifest_sha256": old.manifest["manifest_sha256"],
        "new_manifest_sha256": new.manifest["manifest_sha256"],
        "old_count": len(old_rows),
        "new_count": len(new_rows),
        "removed_sources": sorted(old_rows.keys() - new_rows.keys()),
        "added_sources": sorted(new_rows.keys() - old_rows.keys()),
        "split_changes": split_changes,
        "token_changes": token_changes,
        "hidden_changes": hidden_changes,
        "lineage_changes": lineage_changes,
    }


def _cohort(value: str) -> tuple[str, Path, Path]:
    name, sep, paths = value.partition("=")
    old, comma, new = paths.partition(",")
    if not sep or not comma or not name or not old or not new or "," in new:
        raise argparse.ArgumentTypeError("cohort must be NAME=OLD_BUNDLE,NEW_BUNDLE")
    return name, Path(old), Path(new)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=_cohort, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from core.learning.semantic_program_feature_materialization import (
        load_standard_semantic_feature_bundle,
    )
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    names: set[str] = set()
    rows = {}
    for name, old_path, new_path in args.cohort:
        if name in names:
            raise ValueError(f"duplicate cohort: {name}")
        names.add(name)
        old = load_standard_semantic_feature_bundle(old_path)
        new = load_standard_semantic_feature_bundle(new_path)
        rows[name] = compare_bundles(name, old, new)
        del old, new
    equivalent = all(not any(row[field] for field in (
        "removed_sources", "added_sources", "split_changes", "token_changes", "hidden_changes"
    )) for row in rows.values())
    body = {
        "schema": "aura.semantic_feature_cohort_comparison.v1",
        "cohorts": rows,
        "measured_features_equal": equivalent,
        "representation_compatibility_claimed": False,
        "serving_authority": False,
    }
    receipt = {**body, "receipt_sha256": _sha(body)}
    payload = _canonical(receipt) + b"\n"
    if not atomic_write_bytes_if_absent(args.output, payload, mode=0o400):
        raise FileExistsError(args.output)
    print(json.dumps({"output": str(args.output), "receipt_sha256": receipt["receipt_sha256"],
                      "measured_features_equal": equivalent}, sort_keys=True), flush=True)
    return 0 if equivalent else 1


if __name__ == "__main__":
    raise SystemExit(main())
