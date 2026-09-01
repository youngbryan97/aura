#!/usr/bin/env python3
"""Independently replay one frozen semantic-program campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path, *, max_bytes: int) -> Any:
    from core.runtime.file_read_gateway import read_stable_bytes

    payload = read_stable_bytes(path.expanduser().resolve(strict=True), max_bytes=max_bytes)
    return json.loads(payload.decode("ascii"), object_pairs_hook=_strict_object)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--campaign-report", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from core.governance_context import local_internal_governed_scope
        from core.learning.semantic_program_feature_materialization import (
            load_standard_semantic_feature_bundle,
        )
        from core.learning.semantic_program_verification import (
            SEMANTIC_PROGRAM_VERIFICATION_SOURCES,
            verify_semantic_program_campaign,
        )
        from core.runtime.file_write_gateway import get_file_write_gateway

        bundle = load_standard_semantic_feature_bundle(args.bundle)
        model_payload = _load_json(args.model, max_bytes=16 * 1024 * 1024)
        campaign_report = _load_json(
            args.campaign_report,
            max_bytes=16 * 1024 * 1024,
        )
        source_sha256s = {
            relative: hashlib.sha256((_REPO_ROOT / relative).read_bytes()).hexdigest()
            for relative in SEMANTIC_PROGRAM_VERIFICATION_SOURCES
        }
        verification = verify_semantic_program_campaign(
            bundle,
            stored_model_payload=model_payload,
            stored_report=campaign_report,
            source_sha256s=source_sha256s,
        )
        with local_internal_governed_scope(
            "semantic_program_campaign.verification",
            domain="file_write",
        ):
            get_file_write_gateway().write_bytes(
                args.verification_output,
                _canonical_bytes(verification),
                source="semantic_program_campaign.independent_verification",
            )
    except Exception as exc:  # noqa: BLE001 - CLI reports exact terminal failure
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_campaign_verifier_cli.v1",
                    "verified": False,
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
                "schema": "aura.semantic_program_campaign_verifier_cli.v1",
                "verified": True,
                "verification_sha256": verification["verification_sha256"],
                "held_out_treatment_answer_exact": verification[
                    "held_out_treatment_answer_exact"
                ],
                "held_out_total": verification["held_out_total"],
                "verification_output": str(
                    args.verification_output.expanduser().resolve()
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
