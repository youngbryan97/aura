#!/usr/bin/env python
"""Validate proof-pack schema and preflight requirements."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.environment.benchmark import ProofPack


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("proof_pack")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    pack = ProofPack.load(args.proof_pack)
    required = {"trace_replay", "receipts", "ablations", "holdout_tasks"}
    missing = sorted(required - set(k for k, v in pack.shared_requirements.items() if v))
    ok = bool(pack.environments) and len({env.get("id", "").split(":")[0] for env in pack.environments}) >= 3 and not missing
    print(json.dumps({"proof_pack": pack.proof_pack, "environments": len(pack.environments), "ok": ok, "missing": missing}, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
