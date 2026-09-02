#!/usr/bin/env python3
"""Measure a frozen compositional relation head on verified feature bundles."""

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
        raise ValueError("relation diagnostic needs at least one bundle")
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
            diagnose_compositional_definition_relations,
        )
        from core.learning.semantic_program_compositional_transducer import (
            compositional_semantic_program_transducer_from_dict,
        )
        from core.learning.semantic_program_feature_materialization import (
            load_standard_semantic_feature_bundle,
        )
        from core.runtime.atomic_writer import atomic_write_bytes_if_absent

        paths = _named_paths(args.bundle)
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError("relation diagnostic output already exists")
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
        model = compositional_semantic_program_transducer_from_dict(
            json.loads(args.transducer.expanduser().resolve(strict=True).read_text("ascii"))
        )
        families = {
            name: diagnose_compositional_definition_relations(
                model,
                tuple(
                    item
                    for item in examples
                    if item.construction_id.startswith(f"{name}:")
                ),
            )
            for name in sorted(bundles)
        }
        payload = {
            "schema": "aura.semantic_program_multifamily_relation_diagnostic.v1",
            "transducer_receipt_sha256": model.receipt_sha256,
            "representation_compatibility": compatibility,
            "families": families,
            "expected_answers_available": False,
            "serving_authority": False,
        }
        if not atomic_write_bytes_if_absent(output, _bytes(payload), mode=0o400):
            raise FileExistsError("relation diagnostic output raced")
    except Exception as exc:  # noqa: BLE001 - terminal CLI reports exact failure
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_relation_diagnostic_cli.v1",
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
                "schema": "aura.semantic_program_relation_diagnostic_cli.v1",
                "completed": True,
                "transducer_receipt_sha256": model.receipt_sha256,
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
