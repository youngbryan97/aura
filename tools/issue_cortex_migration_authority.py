#!/usr/bin/env python3
"""Issue one producer-validated cortex migration authority."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.governance_context import local_internal_governed_scope  # noqa: E402
from core.learning.cortex_migration_authority_issuer import (  # noqa: E402
    issue_expert_retirement_authority,
    issue_persona_crsm_authority,
    issue_recurrence_authority,
    issue_steering_authority,
)
from core.learning.model_tissue_migration_inventory import (  # noqa: E402
    load_candidate_descriptor,
)
from core.runtime.file_write_gateway import get_file_write_gateway  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--custody-base", type=Path)
    subparsers = parser.add_subparsers(dest="component", required=True)

    persona = subparsers.add_parser("persona-crsm")
    persona.add_argument("--fusion-plan", type=Path, required=True)
    persona.add_argument("--fusion-receipt", type=Path, required=True)
    persona.add_argument("--journal-key", type=Path, required=True)

    steering = subparsers.add_parser("steering")
    steering.add_argument("--metadata", type=Path, required=True)
    steering.add_argument("--causal-evaluation", type=Path, required=True)
    steering.add_argument("--independent-evidence", type=Path, required=True)

    expert = subparsers.add_parser("expert-adapters")
    expert.add_argument("--inventory", type=Path, required=True)

    recurrence = subparsers.add_parser("recurrence-native")
    recurrence.add_argument("--activation", type=Path, required=True)
    return parser


def run(arguments: argparse.Namespace) -> dict:
    descriptor = load_candidate_descriptor(arguments.descriptor)
    descriptor_sha256 = str(descriptor["descriptor_sha256"])
    common = {
        "descriptor_sha256": descriptor_sha256,
        "custody_base": arguments.custody_base,
    }
    if arguments.component == "persona-crsm":
        authority = issue_persona_crsm_authority(
            fusion_plan_path=arguments.fusion_plan,
            fusion_receipt_path=arguments.fusion_receipt,
            journal_key_path=arguments.journal_key,
            **common,
        )
    elif arguments.component == "steering":
        authority = issue_steering_authority(
            metadata_path=arguments.metadata,
            causal_evaluation_path=arguments.causal_evaluation,
            independent_evidence_path=arguments.independent_evidence,
            **common,
        )
    elif arguments.component == "expert-adapters":
        authority = issue_expert_retirement_authority(
            inventory_path=arguments.inventory,
            **common,
        )
    elif arguments.component == "recurrence-native":
        authority = issue_recurrence_authority(
            activation_path=arguments.activation,
            **common,
        )
    else:  # argparse keeps this unreachable; retain a fail-closed library path.
        raise ValueError("migration_component_unknown")

    payload = json.dumps(authority, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if arguments.out is None:
        sys.stdout.write(payload)
    else:
        output = arguments.out.expanduser().absolute()
        with local_internal_governed_scope(
            "issue_cortex_migration_authority.output", domain="file_write"
        ):
            get_file_write_gateway().write_text(
                output,
                payload,
                source="issue_cortex_migration_authority.output",
            )
        print(output)
    return authority


def main(argv: list[str] | None = None) -> int:
    try:
        run(build_parser().parse_args(argv))
    except (OSError, ValueError) as exc:
        print(f"migration authority issuance failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
