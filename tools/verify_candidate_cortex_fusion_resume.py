#!/usr/bin/env python3
"""Issue an attempt-bound resume verdict for candidate-cortex fusion."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.candidate_cortex_fusion import (  # noqa: E402
    FUSION_PROVENANCE_FILE,
    load_and_validate_fusion_plan,
    validate_fusion_provenance,
    validate_fusion_receipt,
)
from tools import run_detached_step as detached  # noqa: E402


def _env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"missing_{name.lower()}")
    return value


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("resume_artifact_invalid")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--journal-key", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = load_and_validate_fusion_plan(
        args.plan,
        journal_key_path=args.journal_key,
        verify_full_model=False,
    )
    output = plan["output"]
    receipt_path = Path(str(output["receipt_path"]))
    model_path = Path(str(output["path"]))
    staging_path = Path(str(output["staging_path"]))
    checkpoint_sequence = 0
    state = "not_started"
    if receipt_path.exists():
        validate_fusion_receipt(
            plan,
            _json(receipt_path),
            verify_full_model=False,
        )
        checkpoint_sequence = 2
        state = "complete"
    elif model_path.exists():
        if model_path.is_symlink() or not model_path.is_dir():
            raise ValueError("fused_model_path_invalid")
        validate_fusion_provenance(
            plan,
            _json(model_path / FUSION_PROVENANCE_FILE),
        )
        checkpoint_sequence = 1
        state = "model_published"
    elif staging_path.exists():
        if staging_path.is_symlink() or not staging_path.is_dir():
            raise ValueError("fusion_staging_path_invalid")
        state = "staging_incomplete"

    plan_sha = _env("AURA_DETACHED_PLAN_SHA256")
    command_sha = _env("AURA_DETACHED_COMMAND_SHA256")
    prior_attempt = int(_env("AURA_DETACHED_PRIOR_ATTEMPT"))
    prior_head = _env("AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256")
    evidence = {
        "schema": "aura.detached_step.resume_evidence.v2",
        "plan_sha256": plan_sha,
        "command_sha256": command_sha,
        "prior_attempt": prior_attempt,
        "prior_journal_head_sha256": prior_head,
        "checkpoint_sequence": checkpoint_sequence,
        "fusion_plan_sha256": plan["fusion_plan_sha256"],
        "fusion_state": state,
    }
    evidence_sha = detached._sha256(evidence)  # noqa: SLF001
    checkpoint_identity = detached._sha256(  # noqa: SLF001
        {
            "prior_attempt": prior_attempt,
            "prior_journal_head_sha256": prior_head,
            "checkpoint_sequence": checkpoint_sequence,
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
        "checkpoint_sequence": checkpoint_sequence,
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
