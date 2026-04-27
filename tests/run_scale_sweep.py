from __future__ import annotations
"""Generate a bounded scale-sweep artifact for reviewers."""


import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation.hardware_reality import m1_pro_16gb_profile
from core.evaluation.scale_sweep import hardware_scale_table, run_integration_proxy_sweep


def main() -> int:
    report = {
        "generated_at": time.time(),
        "warning": (
            "Integration proxy sweep is not a consciousness or full IIT result; "
            "it is a reproducible scale-sensitivity artifact."
        ),
        "integration_proxy": [point.as_dict() for point in run_integration_proxy_sweep()],
        "hardware": hardware_scale_table(m1_pro_16gb_profile()),
    }
    out = ROOT / "tests" / "SCALE_SWEEP_RESULTS.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

