#!/usr/bin/env python3
"""Run the claim-validation suite in its own process and record the verdict.

The desktop runtime installs the suite and defers the empirical run, for a
good reason: several of these tests perform bounded synthesis and hold the
interpreter for tens of seconds, which a serving process cannot afford. The
deferral named "an explicit validation process" that did not exist, so every
registered claim read "never run" forever, and the verifier reported the
runtime's own decision as a hundred structural errors.

This is that process. It runs the suite, writes what each test produced, and
stamps the result with the source it ran over. The runtime reads the verdict
back only when that stamp matches what is running — a recorded pass over
different code is a record, not evidence.

    python tools/run_model_validation.py                # the boot posture
    python tools/run_model_validation.py --expensive    # everything
    python tools/run_model_validation.py --print        # no write
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Where the runtime looks. One file, overwritten each run: the question is
#: always "does a verdict cover the code running now", never "what did it say
#: in March".
VERDICT_PATH = REPO_ROOT / "artifacts" / "validation" / "last_run.json"

SCHEMA = "aura.validation.verdict.v1"


def source_identity() -> dict[str, Any]:
    """The exact source this ran over: the commit and the working tree.

    A dirty tree is not disqualifying and it is not hidden. The digest names
    which uncommitted state, where a "dirty" flag would let two different
    edited trees compare equal, so a verdict recorded over uncommitted work
    covers exactly the tree that produced it and nothing else.
    """
    from core.runtime.launch_provenance import collect_source_identity

    identity = collect_source_identity(REPO_ROOT)
    return {
        "commit": str(identity.get("commit_sha") or ""),
        "workspace": str(identity.get("workspace_state_sha256") or ""),
        "branch": str(identity.get("branch") or ""),
        "dirty": str(identity.get("source_dirty") or "") == "True",
    }


def run(*, expensive: bool) -> dict[str, Any]:
    from core.organism.model_validation import install_runtime_validation, run_validation

    installed = install_runtime_validation()
    started = time.time()
    outcome = run_validation(include_expensive=expensive)
    return {
        "schema": SCHEMA,
        "started_at": started,
        "finished_at": time.time(),
        "duration_s": round(time.time() - started, 3),
        "include_expensive": bool(expensive),
        "source": source_identity(),
        "claims": installed.get("claims"),
        "tests": len(installed.get("tests") or []),
        "counts": {
            key: outcome.get(key)
            for key in ("passed", "failed", "errored", "not_measured", "applicable", "measured")
        },
        "results": outcome.get("results") or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expensive", action="store_true",
        help="run the experiments too, not only the instruments",
    )
    parser.add_argument(
        "--print", dest="print_only", action="store_true",
        help="report the verdict without recording it",
    )
    parser.add_argument("--out", type=Path, default=VERDICT_PATH)
    args = parser.parse_args()

    verdict = run(expensive=args.expensive)
    counts = verdict["counts"]
    print(
        f"{verdict['tests']} tests over {verdict['claims']} claims in "
        f"{verdict['duration_s']}s: {counts['passed']} passed, {counts['failed']} failed, "
        f"{counts['errored']} errored, {counts['not_measured']} not measured"
    )
    for row in verdict["results"]:
        outcome = row["score"]["outcome"]
        if outcome in {"pass", "not_applicable"}:
            continue
        print(f"  [{outcome}] {row['test']}: {row['score']['interpretation']}")

    if args.print_only:
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    print(f"recorded {args.out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
