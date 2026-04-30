"""CLI entrypoints for Aura's Autonomous Architecture Governor."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from core.architect.config import ASAConfig
from core.architect.governor import AutonomousArchitectureGovernor, load_plan_from_run
from core.architect.models import MutationTier
from core.architect.refactor_planner import plan_to_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m core.architect.cli")
    parser.add_argument("--repo", default=None, help="Repository root; defaults to cwd or AURA_ASA_REPO_ROOT")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("smells")
    sub.add_parser("graph")
    plan_p = sub.add_parser("plan")
    plan_p.add_argument("--target", required=True)
    shadow_p = sub.add_parser("shadow-run")
    shadow_p.add_argument("--plan", required=True)
    promote_p = sub.add_parser("promote")
    promote_p.add_argument("--run", required=True)
    monitor_p = sub.add_parser("monitor")
    monitor_p.add_argument("--run", default=None)
    rollback_p = sub.add_parser("rollback")
    rollback_p.add_argument("--run", required=True)
    auto_p = sub.add_parser("auto")
    auto_p.add_argument("--tier-max", default="T1")
    proposal_p = sub.add_parser("proposal")
    proposal_p.add_argument("--target", required=True)
    args = parser.parse_args(argv)
    config = ASAConfig.from_env(args.repo)
    governor = AutonomousArchitectureGovernor(config)

    if args.command == "audit":
        _print(governor.audit())
        return 0
    if args.command == "smells":
        graph = governor.build_graph()
        _print([asdict(smell) for smell in governor.detect_smells(graph)])
        return 0
    if args.command == "graph":
        graph = governor.build_graph()
        _print({"path": str(config.artifacts / "architecture_graph.json"), "metrics": graph.metrics})
        return 0
    if args.command == "plan":
        plan = governor.plan(args.target)
        _print(plan_to_dict(plan))
        return 0
    if args.command == "shadow-run":
        plan = governor.load_plan(args.plan)
        shadow, ghost, rollback, proof = governor.shadow_run(plan)
        _print({
            "run_id": shadow.run_id,
            "artifact_dir": shadow.artifact_dir,
            "ghost_passed": ghost.passed,
            "proof_passed": proof.passed,
            "rollback": asdict(rollback) if rollback is not None else None,
            "proof_receipt": asdict(proof),
        })
        return 0 if proof.passed else 2
    if args.command == "promote":
        plan = load_plan_from_run(config, args.run)
        shadow = governor.load_shadow_run(args.run)
        proof_path = Path(shadow.artifact_dir) / "proof_receipt.json"
        rollback = governor.rollback_manager.load_packet(args.run)
        from core.architect.models import BehaviorDelta, ProofReceipt, ProofResult
        payload = json.loads(proof_path.read_text(encoding="utf-8"))
        proof = ProofReceipt(
            run_id=payload["run_id"],
            plan_id=payload["plan_id"],
            tier=MutationTier.parse(payload["tier"]),
            results=tuple(ProofResult(**item) for item in payload.get("results", ())),
            behavior_delta=BehaviorDelta(**payload["behavior_delta"]),
            rollback_packet_hash=payload.get("rollback_packet_hash", ""),
            shadow_artifact_path=payload.get("shadow_artifact_path", ""),
            decision_hash=payload.get("decision_hash", ""),
            generated_at=float(payload.get("generated_at", 0)),
        )
        _print(asdict(governor.promote(plan, shadow, proof, rollback)))
        return 0
    if args.command == "monitor":
        result = governor.monitor_status(run_id=args.run)
        _print(asdict(result) if hasattr(result, "__dataclass_fields__") else result)
        return 0
    if args.command == "rollback":
        _print(asdict(governor.rollback(args.run)))
        return 0
    if args.command == "auto":
        _print(governor.auto(tier_max=MutationTier.parse(args.tier_max)))
        return 0
    if args.command == "proposal":
        _print(plan_to_dict(governor.proposal(args.target)))
        return 0
    return 1


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
