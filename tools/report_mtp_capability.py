#!/usr/bin/env python3
"""Report whether the active checkpoint's MTP head is usable. Loads nothing."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.mtp_capability import detect  # noqa: E402
from tools.export_active_descriptor import active_manifest, installation_root  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    model = args.model or Path(active_manifest()["active_model_path"])
    models_dir = installation_root() / "models"
    candidates = (
        {entry.name: entry for entry in sorted(models_dir.iterdir()) if entry.is_dir()}
        if models_dir.exists()
        else {}
    )
    capability = detect(model, candidates)
    payload = capability.as_dict()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
        print(f"wrote {args.json}")

    print(f"checkpoint          {capability.checkpoint}")
    print(f"declares mtp layers {capability.declares_mtp_layers}")
    print(f"mtp tensors present {capability.mtp_tensor_count}")
    print(f"loader discards mtp {capability.loader_discards_mtp}")
    print(f"native MTP usable   {capability.native_supported}")
    for blocker in capability.native_blockers:
        print(f"   blocked by       {blocker}")
    print(f"draft speculation   {capability.draft_speculation_supported}")
    for draft in capability.compatible_draft_models:
        print(
            f"   candidate        {draft['name']} "
            f"({draft['num_hidden_layers']}L, same_family={draft['same_family']})"
        )
    for name in capability.unmeasured:
        print(f"   unmeasured       {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
