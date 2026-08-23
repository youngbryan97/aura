#!/usr/bin/env python3
"""Issue an attempt-bound resume verdict for adaptive cortex training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.candidate_cortex_training import (  # noqa: E402
    JOURNAL_FILE,
    load_and_verify_plan,
    read_authenticated_journal,
)
from tools import run_detached_step as detached  # noqa: E402


def _env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"missing_{name.lower()}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--journal-key", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = load_and_verify_plan(args.run_root, verify_full_model=False)
    key = args.journal_key.expanduser().resolve(strict=True).read_bytes()
    events = read_authenticated_journal(
        Path(str(plan["paths"]["run_root"])) / JOURNAL_FILE,
        key=key,
    )
    plan_sha = _env("AURA_DETACHED_PLAN_SHA256")
    command_sha = _env("AURA_DETACHED_COMMAND_SHA256")
    prior_attempt = int(_env("AURA_DETACHED_PRIOR_ATTEMPT"))
    prior_head = _env("AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256")
    admitted = sum(event.get("event_type") == "stage_admitted" for event in events)
    evidence = {
        "schema": "aura.detached_step.resume_evidence.v2",
        "plan_sha256": plan_sha,
        "command_sha256": command_sha,
        "prior_attempt": prior_attempt,
        "prior_journal_head_sha256": prior_head,
        "checkpoint_sequence": admitted,
        "candidate_plan_sha256": plan["plan_sha256"],
        "journal_event_count": len(events),
        "admitted_stage_count": admitted,
    }
    evidence_sha = detached._sha256(evidence)  # noqa: SLF001
    checkpoint_identity = detached._sha256(  # noqa: SLF001
        {
            "prior_attempt": prior_attempt,
            "prior_journal_head_sha256": prior_head,
            "checkpoint_sequence": admitted,
            "evidence_sha256": evidence_sha,
        }
    )
    verdict = {
        "schema": "aura.detached_step.resume_verdict.v3",
        "plan_sha256": plan_sha,
        "command_sha256": command_sha,
        "prior_attempt": prior_attempt,
        "prior_journal_head_sha256": prior_head,
        "verdict": "safe_to_resume",
        "checkpoint_sequence": admitted,
        "checkpoint_identity": checkpoint_identity,
        "evidence_sha256": evidence_sha,
        "evidence": evidence,
    }
    detached.validate_resume_verdict(
        verdict,
        plan_sha256=plan_sha,
        command_sha256=command_sha,
        prior_attempt=prior_attempt,
        prior_journal_head_sha256=prior_head,
    )
    print(json.dumps(verdict, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
