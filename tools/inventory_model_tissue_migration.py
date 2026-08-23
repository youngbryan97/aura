#!/usr/bin/env python3
"""Inventory candidate-cortex tissue compatibility without loading a model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.model_tissue_migration_inventory import (  # noqa: E402
    TissueInventoryError,
    build_tissue_migration_inventory,
    default_tissue_probes,
    inventory_as_json,
    load_candidate_descriptor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--state-root", type=Path, default=Path.home() / ".aura")
    parser.add_argument("--out", type=Path)
    return parser


def run(arguments: argparse.Namespace) -> dict:
    descriptor = load_candidate_descriptor(arguments.descriptor)
    probes = default_tissue_probes(
        repo_root=arguments.repo_root,
        state_root=arguments.state_root,
    )
    inventory = build_tissue_migration_inventory(descriptor, probes)
    payload = inventory_as_json(inventory)
    if arguments.out is None:
        sys.stdout.write(payload)
    else:
        destination = arguments.out.expanduser().absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="ascii")
        print(destination)
    return inventory


def main(argv: list[str] | None = None) -> int:
    try:
        inventory = run(build_parser().parse_args(argv))
    except (OSError, ValueError, TissueInventoryError) as exc:
        print(f"tissue inventory failed: {exc}", file=sys.stderr)
        return 2
    print(
        "tissue inventory: "
        + ", ".join(
            f"{family['family']}={family['outcome']}"
            for family in inventory["families"]
        ),
        file=sys.stderr if argv is None else sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
