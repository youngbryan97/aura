#!/usr/bin/env python3
"""Replay frozen compositional lesions on one named feature family."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or name in result or not raw_path:
            raise ValueError("bundles must be unique NAME=PATH values")
        result[name] = Path(raw_path).expanduser().resolve(strict=True)
    if not result:
        raise ValueError("compositional lesion verification needs feature bundles")
    return result


def _bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--family", required=True)
    parser.add_argument("--transducer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from core.learning.semantic_program_basis import (
            bind_training_examples_to_shared_representation,
            establish_semantic_training_representation_compatibility,
        )
        from core.learning.semantic_program_campaign import (
            training_examples_from_feature_bundle,
        )
        from core.learning.semantic_program_compositional_campaign import (
            diagnose_compositional_transfer_lesions,
        )
        from core.learning.semantic_program_compositional_transducer import (
            compositional_semantic_program_transducer_from_dict,
        )
        from core.learning.semantic_program_feature_materialization import (
            load_standard_semantic_feature_bundle,
        )
        from core.runtime.atomic_writer import atomic_write_bytes_if_absent

        paths = _named_paths(args.bundle)
        if args.family not in paths:
            raise ValueError("verified family must name one supplied bundle")
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError("compositional lesion output already exists")
        bundles = {
            name: load_standard_semantic_feature_bundle(path)
            for name, path in paths.items()
        }
        manifests = {name: bundle.manifest for name, bundle in bundles.items()}
        compatibility = establish_semantic_training_representation_compatibility(manifests)
        examples = bind_training_examples_to_shared_representation(
            {
                name: training_examples_from_feature_bundle(bundle)
                for name, bundle in bundles.items()
            },
            compatibility=compatibility,
        )
        selected = tuple(
            item
            for item in examples
            if item.construction_id.startswith(f"{args.family}:")
        )
        if len(selected) != len(training_examples_from_feature_bundle(bundles[args.family])):
            raise ValueError("verified family inventory changed during binding")
        model = compositional_semantic_program_transducer_from_dict(
            json.loads(args.transducer.expanduser().resolve(strict=True).read_text("ascii"))
        )
        report = diagnose_compositional_transfer_lesions(model, selected)
        if not atomic_write_bytes_if_absent(output, _bytes(report), mode=0o400):
            raise FileExistsError("compositional lesion output raced")
    except Exception as exc:  # noqa: BLE001 - terminal CLI reports exact failure
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_compositional_lesion_cli.v1",
                    "completed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(
        json.dumps(
            {
                "schema": "aura.semantic_program_compositional_lesion_cli.v1",
                "completed": True,
                "family": args.family,
                "transducer_receipt_sha256": model.receipt_sha256,
                "report_sha256": report["report_sha256"],
                "output": str(output),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
