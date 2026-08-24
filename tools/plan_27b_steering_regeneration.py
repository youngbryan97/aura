#!/usr/bin/env python3
"""Write the CAA capture plan for the active checkpoint. Loads no model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.hybrid_recurrence_geometry import LayerGeometry  # noqa: E402
from core.learning.recovery_package_identity import (  # noqa: E402
    descriptor_from_manifest,
    load_manifest,
)
from core.learning.steering_regeneration import (  # noqa: E402
    authority_errors,
    build_plan,
)
from tools.export_active_descriptor import installation_root  # noqa: E402

LEGACY_BUNDLE = Path("training/vectors/caa_steering_meta.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    install = installation_root()
    manifest = load_manifest(install)
    descriptor = descriptor_from_manifest(manifest)
    config = json.loads((Path(descriptor.path) / "config.json").read_text())
    geometry = LayerGeometry.from_config(config)

    superseded: dict = {}
    legacy = install / LEGACY_BUNDLE
    if legacy.exists():
        meta = json.loads(legacy.read_text())
        superseded = {
            "path": str(LEGACY_BUNDLE),
            "model": meta.get("model"),
            "total_vectors": meta.get("total_vectors"),
            "target_layers": meta.get("target_layers"),
            "disposition": (
                "retained as 32B history; unusable here even though the widths "
                "match, because the residual basis is a different model's"
            ),
        }
        # The retained metadata records a dict per dimension; only the key
        # names carry over, and the layers it lists belong to the old model.
        dimensions = tuple(
            str(entry.get("key"))
            for entry in (meta.get("dimensions") or [])
            if isinstance(entry, dict) and entry.get("key")
        )
    else:
        dimensions = ()

    plan = build_plan(
        descriptor_fingerprint=descriptor.fingerprint(),
        model_path=descriptor.path,
        geometry=geometry,
        hidden_size=descriptor.hidden_size,
        dimensions=dimensions or ("valence", "arousal", "curiosity", "warmth", "focus"),
        superseded_bundle=superseded,
    )
    payload = plan.as_dict()
    payload["authority_errors_before_capture"] = authority_errors(payload)

    out = args.out or REPO_ROOT / "artifacts/migration/27b/recovery/steering_plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {out}")
    print(f"checkpoint        {plan.model_path}")
    print(f"target layers     {len(plan.target_layers)} "
          f"({len(plan.attention_targets())} attention, "
          f"{len(plan.linear_targets())} linear)")
    print(f"dimensions        {', '.join(plan.dimensions)}")
    print(f"vectors to capture {payload['expected_vector_count']}")
    print(f"serving authority {plan.serving_authority}")
    for error in payload["authority_errors_before_capture"]:
        print(f"   outstanding    {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
