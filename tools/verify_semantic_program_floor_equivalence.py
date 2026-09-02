#!/usr/bin/env python3
"""Replay a frozen shared semantic campaign on Aura's universal floor."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_FAMILY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _bundle_arguments(values: list[str]) -> dict[str, Path]:
    bundles: dict[str, Path] = {}
    for value in values:
        family, separator, raw_path = value.partition("=")
        if not separator or not _FAMILY.fullmatch(family) or not raw_path:
            raise ValueError("bundle must be FAMILY=PATH with a stable lowercase family")
        if family in bundles:
            raise ValueError(f"bundle family repeats: {family}")
        bundles[family] = Path(raw_path).expanduser().resolve(strict=True)
    if len(bundles) < 2:
        raise ValueError("semantic floor verification needs at least two families")
    return bundles


def _canonical_bytes(value: object) -> bytes:
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
    parser.add_argument("--bundle", action="append", required=True, metavar="FAMILY=PATH")
    parser.add_argument("--model", type=Path, required=True)
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
        from core.learning.semantic_program_feature_materialization import (
            load_standard_semantic_feature_bundle,
        )
        from core.learning.semantic_program_floor_verification import (
            SEMANTIC_PROGRAM_FLOOR_VERIFICATION_SOURCES,
            verify_semantic_program_floor_equivalence,
        )
        from core.learning.semantic_program_shared_transducer import (
            shared_semantic_program_transducer_from_dict,
        )
        from core.runtime.atomic_writer import atomic_write_bytes_if_absent

        bundle_paths = _bundle_arguments(args.bundle)
        model_path = args.model.expanduser().resolve(strict=True)
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError("semantic floor verification output already exists")
        bundles = {
            family: load_standard_semantic_feature_bundle(path)
            for family, path in bundle_paths.items()
        }
        grouped = {
            family: training_examples_from_feature_bundle(bundle)
            for family, bundle in bundles.items()
        }
        manifests = {family: bundle.manifest for family, bundle in bundles.items()}
        compatibility = establish_semantic_training_representation_compatibility(
            manifests
        )
        examples = bind_training_examples_to_shared_representation(
            grouped,
            compatibility=compatibility,
        )
        model = shared_semantic_program_transducer_from_dict(
            json.loads(model_path.read_text(encoding="utf-8"))
        )
        sources = {
            relative: hashlib.sha256((_REPO_ROOT / relative).read_bytes()).hexdigest()
            for relative in SEMANTIC_PROGRAM_FLOOR_VERIFICATION_SOURCES
        }
        report = verify_semantic_program_floor_equivalence(
            model,
            examples,
            feature_manifest_sha256s={
                family: manifests[family]["manifest_sha256"] for family in sorted(manifests)
            },
            source_sha256s=sources,
        )
        if not atomic_write_bytes_if_absent(output, _canonical_bytes(report), mode=0o400):
            raise FileExistsError("semantic floor verification output raced")
    except Exception as exc:  # noqa: BLE001 - terminal CLI reports exact failure
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_floor_verification_cli.v1",
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
                "schema": "aura.semantic_program_floor_verification_cli.v1",
                "completed": True,
                "test_total": report["test_total"],
                "agreements": report["agreements"],
                "value_agreements": report["value_agreements"],
                "refusal_agreements": report["refusal_agreements"],
                "verification_sha256": report["verification_sha256"],
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
