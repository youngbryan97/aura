#!/usr/bin/env python3
"""Recover the exact source schemas before planning a semantic transfer run."""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.refit_semantic_argument_proposals import configure_refit_environment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transducer", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--source-manifest", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    configure_refit_environment(args.output)
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.learning.semantic_program_natural_transfer import (
        build_bound_semantic_source_inventory,
    )
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    manifests = {}
    for entry in args.source_manifest:
        name, separator, path = entry.partition("=")
        if not separator or not name or name in manifests:
            parser.error("source manifests must have unique NAME=PATH entries")
        manifests[name] = json.loads(Path(path).expanduser().read_text("ascii"))
    model = compositional_semantic_program_transducer_from_dict(
        json.loads(args.transducer.read_text("ascii"))
    )
    result = build_bound_semantic_source_inventory(
        source_manifests=manifests,
        source_campaign=json.loads(args.source_report.read_text("ascii")),
        training_receipt=model.training_receipt,
    )
    if not atomic_write_bytes_if_absent(
        args.output,
        (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii"),
        mode=0o400,
    ):
        raise FileExistsError(args.output)
    print(
        json.dumps(
            {
                "inventory_sha256": result["inventory_sha256"],
                "examples": result["example_count"],
                "schemas": len(result["schema_sha256s"]),
                "families": sorted(result["sources"]),
                "hidden_state_arrays_loaded": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
