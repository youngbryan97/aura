#!/usr/bin/env python3
"""Write the active cortex artifact descriptor where offline tools can read it.

The manifest lives with the installation, not with a source checkout: models
are large and untracked, so `.claude/worktrees/` has the code and no
`training/fused-model/`. Every migration tool needs the same answer to "which
checkpoint is active right now", and each one growing its own path guess is how
they end up disagreeing.

    python tools/export_active_descriptor.py --out PATH
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_RELATIVE = "training/fused-model/active.json"


def installation_root() -> Path:
    """The checkout that holds the untracked model manifest."""
    if (REPO_ROOT / MANIFEST_RELATIVE).exists():
        return REPO_ROOT
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return REPO_ROOT
    return Path(common).parent if common else REPO_ROOT


def active_manifest() -> dict:
    path = installation_root() / MANIFEST_RELATIVE
    if not path.exists():
        raise SystemExit(f"no active cortex manifest at {path}")
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = active_manifest()
    descriptor = manifest.get("artifact_descriptor")
    if not isinstance(descriptor, dict):
        raise SystemExit("active manifest carries no artifact_descriptor")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(descriptor, indent=1, sort_keys=True))
    profile = descriptor.get("artifact_profile") or {}
    print(
        f"wrote {args.out} -- {profile.get('model_type')} "
        f"{profile.get('num_hidden_layers')}L "
        f"{profile.get('full_attention_layers')}attn/"
        f"{profile.get('linear_attention_layers')}linear "
        f"ctx {profile.get('native_context_window')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
