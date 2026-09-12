#!/usr/bin/env python3
"""tools/seal_caa_generation.py — the manifest an authority is issued against.

One generation is a directory of vectors plus a statement of how they were
made, and both halves are hashed so a later reader can tell a bundle from a
bundle that has been edited. `validate_vector_generation` reopens every file,
compares size and digest, and recomputes the generation hash from the
extraction contract and the manifest, so nothing here can be asserted -- it can
only be recorded correctly or fail.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_caa_seal")

DEFAULT_PLAN = REPO / "artifacts/migration/27b/recovery/steering_plan.json"
DEFAULT_VECTORS = REPO / "training/vectors/cortex-52d313c2"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS)
    arguments = parser.parse_args(argv)

    from core.brain.llm.model_registry import get_active_cortex_spec
    from core.consciousness.affective_steering import AFFECTIVE_DIMENSIONS
    from core.evaluation.caa_causal_evaluation import (
        canonical_sha256,
        file_sha256,
        validate_vector_generation,
    )

    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    spec = get_active_cortex_spec(force_refresh=True)
    if spec is None:
        print("no active cortex", file=sys.stderr)
        return 1
    descriptor = str(spec.descriptor_sha256)

    root = arguments.vectors.expanduser().resolve(strict=True)
    vector_files = [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.glob("*_layer*.npz"))
    ]
    if not vector_files:
        print(f"no vectors under {root}", file=sys.stderr)
        return 1

    extraction = {
        "method": "contrastive_activation_addition",
        "statistic": "difference_of_means_last_token_hidden_state",
        "model_path": str(plan["model_path"]),
        "model_descriptor_sha256": descriptor,
        "hidden_size": int(plan["hidden_size"]),
        "target_layers": [
            {"index": int(row["index"]), "kind": str(row["kind"])}
            for row in plan["target_layers"]
        ],
        "dimensions": [
            {
                "key": str(dimension["key"]),
                "positive_prompts": len(dimension["positive"]),
                "negative_prompts": len(dimension["negative"]),
            }
            for dimension in AFFECTIVE_DIMENSIONS
        ],
        "captured_by": "tools/capture_27b_steering_vectors.py",
    }
    extraction["extraction_contract_sha256"] = canonical_sha256(extraction)

    metadata = {
        "schema": "aura.caa.vector_generation.v1",
        "model_identity": {"model_descriptor_sha256": descriptor},
        "extraction_contract": extraction,
        "vector_files": vector_files,
    }
    metadata["generation_sha256"] = canonical_sha256(
        {
            "extraction_contract_sha256": extraction["extraction_contract_sha256"],
            "vector_files": vector_files,
        }
    )

    out = root / "metadata.json"
    out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checked = validate_vector_generation(
        metadata, generation_dir=root, model_descriptor_sha256=descriptor
    )
    print(
        f"sealed {checked['vector_count']} vectors under {root}\n"
        f"  generation  {checked['generation_sha256'][:16]}\n"
        f"  extraction  {extraction['extraction_contract_sha256'][:16]}\n"
        f"  descriptor  {descriptor[:16]}\n"
        f"wrote {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
