#!/usr/bin/env python3
"""Run Aura activation audit and optionally reconcile safe missing loops."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.activation_audit import get_activation_auditor


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconcile", action="store_true")
    parser.add_argument("--output", default="artifacts/activation_report.json")
    args = parser.parse_args()

    auditor = get_activation_auditor()
    report = await auditor.audit(reconcile=args.reconcile)
    auditor.write_report(report, Path(args.output))
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if not report.missing_required else 1


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
