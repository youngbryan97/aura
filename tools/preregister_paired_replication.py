#!/usr/bin/env python3
"""Check a prospective paired design and publish it through Aura's plan store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    from core.evaluation.paired_replication import build_paired_replication_plan
    from core.governance_context import local_internal_governed_scope
    from tools.run_semantic_program_replication import _load_json

    plan = build_paired_replication_plan(_load_json(args.spec, max_bytes=32 * 1024 * 1024))
    with local_internal_governed_scope("evaluation.paired_replication", domain="file_write"):
        path = plan.write(args.store)
    print(json.dumps({"plan_hash": plan.plan_hash, "path": str(path),
                      "confirmatory_run_complete": False,
                      "external_publication_evidence_required": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
