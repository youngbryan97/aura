#!/usr/bin/env python3
"""Her steering basis, bound to another checkpoint with nothing else changed.

Her steering loads no vector bound to a checkpoint other than the one in
memory, so her basis cannot steer the untuned model as it stands. The first arm
of item 18 asks whether her own directions, unchanged, carry her state to an
untuned model's rating: this copies her basis and rewrites only which
checkpoint each file is bound to, and the statement records the rebinding. Her
directions were fitted in her model, so a failure here alone could be the
directions not carrying over; the second arm fits them in the untuned model.

The identity is built from the checkpoint's own bytes, the way the certificate
tool and her worker build a model's that is not her cortex
(`build_model_artifact_descriptor`, no repository or revision), so the basis,
its certificate and the run all name the same model.

    python tools/rebind_steering_basis.py <her basis> <out dir> --model <checkpoint>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

STATEMENT: str = "basis.json"
BINDING: str = "model_descriptor_sha256"


def rebind_basis(source: Path, out: Path, descriptor: dict[str, Any], model: Path) -> dict[str, Any]:
    """Write `source` into `out`, which must not exist, bound to `descriptor`; return the statement."""
    statement = json.loads((source / STATEMENT).read_text())
    digest = str(descriptor.get("descriptor_sha256") or "")
    if len(digest) != 64:
        raise ValueError("refusing: the checkpoint's identity has no digest")
    if statement.get(BINDING) == digest:
        raise ValueError(f"refusing: {source} is already bound to {digest[:16]}")
    files = sorted(source.glob("*.npz"))
    if not files:
        raise ValueError(f"refusing: {source} holds no vectors")
    out.mkdir(parents=True, exist_ok=False)
    for path in files:
        with np.load(path, allow_pickle=False) as held:
            arrays = {key: held[key] for key in held.files}
        arrays[BINDING] = np.array(digest)
        np.savez(out / path.name, **arrays)
    for path in source.iterdir():
        # The generation's signed metadata names her checkpoint and its own
        # digest; a rebound copy is no longer that generation.
        if path.is_file() and path.suffix != ".npz" and path.name not in (STATEMENT, "metadata.json"):
            shutil.copy2(path, out / path.name)
    rebound = {
        **statement,
        BINDING: digest,
        "rebound": {
            "from": str(source.resolve()),
            "from_model_descriptor_sha256": statement.get(BINDING, ""),
            "to_model": str(model.resolve()),
            "rule": "every direction, length and weighting kept; only the checkpoint each file is bound to changed",
        },
    }
    (out / STATEMENT).write_text(json.dumps(rebound, indent=1) + "\n")
    return rebound


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path, help="a directory that does not exist yet")
    parser.add_argument("--model", type=Path, required=True, help="the checkpoint the copy is bound to")
    args = parser.parse_args(argv)
    from core.brain.llm.model_artifact_profile import build_model_artifact_descriptor

    model = args.model.expanduser().resolve(strict=True)
    made = rebind_basis(args.source, args.out, build_model_artifact_descriptor(model), model)
    print(json.dumps(made["rebound"] | {BINDING: made[BINDING]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
