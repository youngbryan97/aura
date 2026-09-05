#!/usr/bin/env python3
"""Run the preregistered causal replication of one frozen semantic path."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load(path: Path, *, max_bytes: int) -> tuple[Any, str]:
    from core.runtime.file_read_gateway import read_stable_bytes

    payload = read_stable_bytes(path.expanduser().resolve(strict=True), max_bytes=max_bytes)
    return json.loads(payload.decode("ascii")), hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--discovery-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from core.learning.semantic_program_feature_materialization import (
            load_standard_semantic_feature_bundle,
        )
        from core.learning.semantic_program_frozen_path_replication import (
            evaluate_frozen_path_replication,
        )
        from core.learning.semantic_program_path_ensemble import (
            semantic_program_path_ensemble_from_dict,
        )
        from core.runtime.atomic_writer import atomic_write_bytes_if_absent

        prereg, _ = _load(args.preregistration, max_bytes=1024 * 1024)
        prereg = _mapping(prereg, field="preregistration")
        frozen = _mapping(prereg.get("frozen_transducer"), field="frozen transducer")
        training = _mapping(prereg.get("training_evidence"), field="training evidence")
        container, container_sha = _load(
            Path(str(frozen["container_path"])), max_bytes=64 * 1024 * 1024
        )
        training_manifest, training_sha = _load(
            Path(str(training["manifest_path"])), max_bytes=16 * 1024 * 1024
        )
        discovery_manifest, _ = _load(args.discovery_manifest, max_bytes=16 * 1024 * 1024)
        if (
            container_sha != frozen.get("container_file_sha256")
            or training_sha != training.get("manifest_file_sha256")
        ):
            raise ValueError("frozen path source file identity differs")
        ensemble = semantic_program_path_ensemble_from_dict(container)
        if ensemble.receipt_sha256 != frozen.get("container_receipt_sha256"):
            raise ValueError("frozen path container receipt differs")

        def progress(index: int, total: int, counts: Mapping[str, int]) -> None:
            print(
                json.dumps(
                    {"progress": f"{index}/{total}", "answer_exact": dict(counts)},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                file=sys.stderr,
                flush=True,
            )

        report = evaluate_frozen_path_replication(
            bundle=load_standard_semantic_feature_bundle(args.bundle),
            training_manifest=_mapping(training_manifest, field="training manifest"),
            discovery_manifest=_mapping(discovery_manifest, field="discovery manifest"),
            preregistration=prereg,
            model=ensemble.challenger,
            progress=progress,
        )
        output = args.output.expanduser().resolve(strict=False)
        bundle_root = args.bundle.expanduser().resolve(strict=True)
        if output == bundle_root or output.is_relative_to(bundle_root):
            raise ValueError("replication output cannot modify the immutable feature bundle")
        if not atomic_write_bytes_if_absent(output, _bytes(report), mode=0o400):
            raise FileExistsError("frozen path replication output already exists")
    except Exception as exc:  # noqa: BLE001 - terminal CLI reports exact failure
        print(
            json.dumps(
                {"complete": False, "error": f"{type(exc).__name__}: {exc}"},
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
                "complete": True,
                "verdict": report["verdict"],
                "answer_exact": report["treatment_answer_exact"],
                "result_sha256": report["result_sha256"],
                "output": str(output),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0 if report["verdict"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
