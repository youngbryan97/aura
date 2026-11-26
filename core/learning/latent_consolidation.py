"""core/learning/latent_consolidation.py

From temporary synapses to durable learning — with gates, not enthusiasm.

The latent cortex exports mechanically-clean episode synapses (accepted
strictly-descending fast-weight updates whose erase was PROVEN) into
``data/latent_cortex/consolidation_queue/``. This module is the consumer:

    scan → validate each candidate's evidence → aggregate by domain →
    when a domain accumulates enough independent wins, emit a
    CONSOLIDATION PROPOSAL for the compounding loop → the interference
    battery gates any activation.

Nothing here touches model weights. A proposal is evidence that a domain
repeatedly benefits from the same kind of temporary specialization — the
actual durable learning happens through the existing training/compounding
machinery with its regression gates, and `interference_battery` provides
the anti-interference verdict the spec demands ("10 learnings must not
trash prior ones").
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.LatentConsolidation")

CONSOLIDATION_PROPOSAL_SCHEMA = "aura.latent_consolidation_proposal.v1"

# A domain must accumulate this many independent mechanically-clean episode
# candidates before a proposal is emitted.
MIN_CANDIDATES_PER_DOMAIN = 3


@dataclass
class CandidateRecord:
    episode_id: str
    path: Path
    domain: str
    loss_improvement: float
    created_at: float
    valid: bool
    checkpoint_fingerprint: str = ""
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "path": str(self.path),
            "domain": self.domain,
            "loss_improvement": round(self.loss_improvement, 6),
            "created_at": self.created_at,
            "valid": self.valid,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "rejection_reasons": list(self.rejection_reasons),
        }


def _queue_dir() -> Path:
    from core.config import DATA_DIR

    return Path(DATA_DIR) / "latent_cortex" / "consolidation_queue"


def _proposal_dir() -> Path:
    from core.config import DATA_DIR

    return Path(DATA_DIR) / "latent_cortex" / "proposals"


def validate_candidate(candidate_dir: Path) -> CandidateRecord:
    """Gate one exported candidate on its own evidence — trust nothing."""
    episode_id = candidate_dir.name
    reasons: list[str] = []
    domain = "general"
    fingerprint = ""
    improvement = 0.0
    created_at = 0.0
    evidence_path = candidate_dir / "evidence.json"
    weights_path = candidate_dir / "delta_weights.npz"
    if not weights_path.is_file():
        reasons.append("delta_weights_missing")
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = None
        reasons.append("evidence_unreadable")
    if isinstance(payload, dict):
        created_at = float(payload.get("created_at") or 0.0)
        lifecycle = payload.get("lifecycle")
        evidence = payload.get("evidence")
        lifecycle = lifecycle if isinstance(lifecycle, dict) else {}
        evidence = evidence if isinstance(evidence, dict) else {}
        domain = str(evidence.get("domain") or "general")
        fingerprint = str(evidence.get("checkpoint_fingerprint") or "")
        if not fingerprint:
            # A delta without checkpoint provenance can never be evidence —
            # and mixing checkpoints inside one adapter is how the first real
            # 32B train crashed (leaked tiny-model candidates, Jul 2026).
            reasons.append("checkpoint_fingerprint_missing")
        if lifecycle.get("erase_proven") is not True:
            reasons.append("erase_unproven")
        if not lifecycle.get("optimized_steps"):
            reasons.append("no_accepted_optimization")
        trail = evidence.get("loss_trail")
        if (
            not isinstance(trail, list)
            or len(trail) < 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in trail)
            or not trail[-1] < trail[0]
        ):
            reasons.append("loss_not_descending")
        else:
            improvement = float(trail[0]) - float(trail[-1])
        flags = evidence.get("honest_flags")
        blocking = [
            flag
            for flag in (flags if isinstance(flags, list) else [])
            if not str(flag).startswith("fast_weight_candidate_exported")
        ]
        if blocking:
            reasons.append(f"honest_flags_present:{','.join(map(str, blocking))[:80]}")
    return CandidateRecord(
        episode_id=episode_id,
        path=candidate_dir,
        domain=domain,
        loss_improvement=improvement,
        created_at=created_at,
        valid=not reasons,
        checkpoint_fingerprint=fingerprint,
        rejection_reasons=reasons,
    )


def scan_queue(queue_dir: Path | str | None = None) -> list[CandidateRecord]:
    root = Path(queue_dir) if queue_dir else _queue_dir()
    if not root.is_dir():
        return []
    records = []
    for candidate_dir in sorted(root.iterdir()):
        if candidate_dir.is_dir():
            records.append(validate_candidate(candidate_dir))
    return records


def build_proposals(
    records: list[CandidateRecord],
    *,
    min_candidates: int = MIN_CANDIDATES_PER_DOMAIN,
) -> list[dict[str, Any]]:
    """Aggregate valid candidates into proposals per (domain, checkpoint).

    The checkpoint fingerprint is part of the aggregation key: one adapter
    is only ever distilled from deltas of ONE model. Cross-checkpoint mixes
    are structurally impossible here, not merely refused downstream.
    """
    by_key: dict[tuple[str, str], list[CandidateRecord]] = {}
    for record in records:
        if record.valid:
            by_key.setdefault(
                (record.domain, record.checkpoint_fingerprint), []
            ).append(record)
    proposals = []
    for (domain, fingerprint), group in sorted(by_key.items()):
        if len(group) < min_candidates:
            continue
        proposals.append(
            {
                "schema": CONSOLIDATION_PROPOSAL_SCHEMA,
                "domain": domain,
                "checkpoint_fingerprint": fingerprint,
                "candidates": [r.to_dict() for r in group],
                "candidate_count": len(group),
                "mean_loss_improvement": round(
                    sum(r.loss_improvement for r in group) / len(group), 6
                ),
                "created_at": time.time(),
                "activation_requirements": [
                    "compounding loop trains/distills via existing gates",
                    "interference battery verdict PASS before activation",
                    "held-out capability regression within tolerance",
                ],
                "status": "proposed",
            }
        )
    return proposals


def run_consolidation_cycle(
    queue_dir: Path | str | None = None,
    proposal_dir: Path | str | None = None,
    *,
    min_candidates: int = MIN_CANDIDATES_PER_DOMAIN,
) -> dict[str, Any]:
    """One governed pass: scan, validate, aggregate, persist proposals."""
    records = scan_queue(queue_dir)
    proposals = build_proposals(records, min_candidates=min_candidates)
    written: list[str] = []
    if proposals:
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            gateway = get_file_write_gateway()
            out_root = Path(proposal_dir) if proposal_dir else _proposal_dir()
            with local_internal_governed_scope("latent_consolidation_proposals"):
                gateway.ensure_directory(out_root, source="latent_consolidation")
                for proposal in proposals:
                    name = f"proposal_{proposal['domain']}_{int(proposal['created_at'])}.json"
                    gateway.write_text(
                        out_root / name,
                        json.dumps(proposal, indent=1, sort_keys=True),
                        source="latent_consolidation",
                    )
                    written.append(name)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            record_degradation(
                "latent_consolidation",
                exc,
                action="kept consolidation proposals in memory after persist failed",
            )
    receipt = {
        "schema": "aura.latent_consolidation_cycle.v1",
        "scanned": len(records),
        "valid": sum(1 for r in records if r.valid),
        "invalid": sum(1 for r in records if not r.valid),
        "rejections": {
            r.episode_id: r.rejection_reasons for r in records if not r.valid
        },
        "proposals": [p["domain"] for p in proposals],
        "written": written,
        "ran_at": time.time(),
    }
    logger.info(
        "Latent consolidation cycle: %d scanned, %d valid, %d proposals",
        receipt["scanned"],
        receipt["valid"],
        len(proposals),
    )
    return receipt


__all__ = [
    "CONSOLIDATION_PROPOSAL_SCHEMA",
    "MIN_CANDIDATES_PER_DOMAIN",
    "CandidateRecord",
    "build_proposals",
    "run_consolidation_cycle",
    "scan_queue",
    "validate_candidate",
]
