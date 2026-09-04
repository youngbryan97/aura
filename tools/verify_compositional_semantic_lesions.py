#!/usr/bin/env python3
"""Replay frozen compositional lesions on one named feature family."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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


def _sha(value: object) -> str:
    return hashlib.sha256(_bytes(value).rstrip(b"\n")).hexdigest()


def _manifest_only(bundle_path: Path) -> dict[str, Any]:
    """Validate the immutable manifest without loading unused feature arrays."""

    from core.runtime.file_read_gateway import read_stable_bytes

    root = bundle_path.expanduser().resolve(strict=True)
    manifest_path = root / "manifest.json" if root.is_dir() else root
    payload = read_stable_bytes(manifest_path, max_bytes=16 * 1024 * 1024)
    try:
        manifest = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("training feature manifest is invalid") from exc
    if not isinstance(manifest, dict):
        raise ValueError("training feature manifest is not an object")
    manifest_sha256 = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    observed_sha256 = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    if (
        manifest.get("schema") != "aura.semantic_program_feature_manifest.v1"
        or manifest.get("complete") is not True
        or manifest_sha256 != observed_sha256
    ):
        raise ValueError("training feature manifest identity differs")
    return manifest


def _bind_compatibility_to_report(
    report: Mapping[str, Any],
    *,
    compatibility: Mapping[str, Any],
) -> dict[str, Any]:
    """Make cross-session neural-basis admission part of the evidence object."""

    body = {key: value for key, value in report.items() if key != "report_sha256"}
    body["representation_compatibility"] = dict(compatibility)
    return {**body, "report_sha256": _sha(body)}


def _bind_family_examples(
    *,
    model: Any,
    family: str,
    examples_by_family: Mapping[str, Sequence[Any]],
    manifests: Mapping[str, Mapping[str, Any]],
    training_manifest: Mapping[str, Any] | None,
) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    """Bind one evaluation family to an explicit neural representation basis."""

    from core.learning.semantic_program_basis import (
        bind_examples_to_compatible_training_session,
        bind_training_examples_to_shared_representation,
        establish_semantic_representation_compatibility,
        establish_semantic_training_representation_compatibility,
    )

    if family not in examples_by_family or set(examples_by_family) != set(manifests):
        raise ValueError("verified family inventory is incomplete")
    if training_manifest is not None:
        if set(examples_by_family) != {family}:
            raise ValueError(
                "cross-session lesion verification accepts exactly one fresh family"
            )
        compatibility = establish_semantic_representation_compatibility(
            model=model,
            training_manifest=training_manifest,
            replication_manifest=manifests[family],
        )
        selected = bind_examples_to_compatible_training_session(
            examples_by_family[family],
            compatibility=compatibility,
        )
        return selected, compatibility

    compatibility = establish_semantic_training_representation_compatibility(manifests)
    examples = bind_training_examples_to_shared_representation(
        examples_by_family,
        compatibility=compatibility,
    )
    selected = tuple(
        item for item in examples if item.construction_id.startswith(f"{family}:")
    )
    if len(selected) != len(examples_by_family[family]):
        raise ValueError("verified family inventory changed during binding")
    return selected, compatibility


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--family", required=True)
    parser.add_argument(
        "--training-bundle",
        type=Path,
        help=(
            "Feature bundle that established the frozen transducer basis. Required "
            "when the evaluated bundle was acquired in a different worker session."
        ),
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        help="Evaluate treatment plus this lesion; repeat for more lesions",
    )
    parser.add_argument("--transducer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
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
        model = compositional_semantic_program_transducer_from_dict(
            json.loads(args.transducer.expanduser().resolve(strict=True).read_text("ascii"))
        )
        manifests = {name: bundle.manifest for name, bundle in bundles.items()}
        training_manifest = (
            _manifest_only(args.training_bundle)
            if args.training_bundle is not None
            else None
        )
        examples_by_family = {
            name: training_examples_from_feature_bundle(bundle)
            for name, bundle in bundles.items()
        }
        selected, compatibility = _bind_family_examples(
            model=model,
            family=args.family,
            examples_by_family=examples_by_family,
            manifests=manifests,
            training_manifest=training_manifest,
        )
        selected_arms = (
            tuple(dict.fromkeys(("treatment", *args.arm))) if args.arm else None
        )
        report = diagnose_compositional_transfer_lesions(
            model,
            selected,
            arm_names=selected_arms,
        )
        if training_manifest is not None:
            report = _bind_compatibility_to_report(
                report,
                compatibility=compatibility,
            )
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
                "evaluated_arms": report["evaluated_arms"],
                "representation_compatibility_receipt_sha256": compatibility[
                    "receipt_sha256"
                ],
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
