#!/usr/bin/env python3
"""What each model-dependent capability needs before the 27B can serve it.

The active cortex manifest carries a signed migration contract, and it already
answers the question for four components. This reads that contract rather than
re-deriving it, then says the part the contract does not cover: which
capabilities the runtime actually depends on have no signed authority at all.

The distinction matters because an absent component and a deferred one look
identical from a health check. A deferred component has a signed quarantine
saying "measured on another checkpoint, not usable here". An absent one has
nothing, which is not a state -- it is a gap in the contract, and it is the
only way a capability can come back without anyone deciding that it should.

One finding worth reading before the rest: ``persona_crsm`` is signed
``fused_persona_crsm``, with a fusion plan and a fusion receipt. The persona is
IN this checkpoint, not waiting to be migrated onto it. What it needs is
verification and behavioural continuity, not a recovery run.

    python tools/report_27b_migration_queue.py --json OUT
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.export_active_descriptor import active_manifest  # noqa: E402

MIGRATION_QUEUE_SCHEMA: Final = "aura.rlc.27b_migration_queue.v1"

#: Every model-dependent capability the runtime can serve, and where its load
#: path lives. A capability absent from the signed contract still appears here,
#: which is the point: the contract cannot report a component it never names.
CAPABILITIES: Final = {
    "persona_crsm": {
        "owner": "core/learning/cortex_generation_upgrade.py",
        "load_path": "fused into the checkpoint weights",
        "representation_bound": True,
    },
    "steering": {
        "owner": "core/consciousness/affective_steering.py",
        "load_path": "core/brain/llm/model_bound_steering.resolve_active_generation",
        "representation_bound": True,
    },
    "recurrence_native": {
        "owner": "core/learning/unified_intrinsic_recurrence.py",
        "load_path": "core/brain/llm/unified_recurrent_qualified_activation.py",
        "representation_bound": True,
    },
    "expert_adapters": {
        "owner": "core/learning/domain_specialists.py",
        "load_path": "adapter registry",
        "representation_bound": True,
    },
    "grounding_contracts": {
        "owner": "core/learning/recurrent_literal_grounding.py",
        "load_path": "derived from the tokenizer at use",
        "representation_bound": False,
        "tokenizer_bound": True,
        "own_gate": (
            "self-validating: every binding is re-derived from the live "
            "tokenizer and refuses when a digit does not round-trip exactly"
        ),
    },
    "episodic_plasticity": {
        "owner": "core/brain/llm/latent_cortex/semantic_plasticity.py",
        "load_path": "episode-scoped, built per episode",
        "representation_bound": True,
        "own_gate": (
            "built from the loaded model's own projections each episode and "
            "discarded with it; nothing survives a checkpoint change"
        ),
    },
    "fast_weight_surfaces": {
        "owner": "core/brain/llm/latent_cortex/fast_weights.py",
        "load_path": "attached per episode, erased at the end",
        "representation_bound": True,
        "own_gate": (
            "identity at attach and a proven erase bound the lifetime to one "
            "episode, so no surface crosses a checkpoint boundary"
        ),
    },
    "qualified_rlc_serving": {
        "owner": "core/brain/llm/semantic_neural_serving.py",
        "load_path": "content-addressed activation package",
        "representation_bound": True,
        "own_gate": (
            "the activation record pins the model descriptor and reports "
            "semantic_neural_model_basis_migration_deferred on a mismatch"
        ),
    },
}

#: How a signed authority kind maps onto what has to happen next.
DISPOSITION_BY_KIND: Final = {
    "fused_persona_crsm": (
        "portable",
        "already inside the checkpoint; verify and test continuity",
    ),
    "caa_model_bound": ("rebindable", "bound to this checkpoint already"),
    "qualified_recurrent_activation": ("rebindable", "qualified for this checkpoint"),
    "model_basis_quarantine": (
        "retrain_required",
        "measured on another checkpoint and quarantined here",
    ),
    "retirement_inventory": ("retired", "withdrawn from measured evidence"),
}


def _components(manifest: dict[str, Any]) -> dict[str, Any]:
    contract = manifest.get("migration_contract")
    components = contract.get("components") if isinstance(contract, dict) else None
    return components if isinstance(components, dict) else {}


def build() -> dict[str, Any]:
    try:
        manifest = active_manifest()
    except SystemExit:
        return {
            "schema": MIGRATION_QUEUE_SCHEMA,
            "blocked": "no active cortex manifest; nothing can be classified",
        }
    signed = _components(manifest)
    rows: list[dict[str, Any]] = []
    for name, facts in sorted(CAPABILITIES.items()):
        entry = signed.get(name)
        row: dict[str, Any] = {
            "capability": name,
            "code_owner": facts["owner"],
            "active_load_path": facts["load_path"],
            "representation_bound": facts["representation_bound"],
            "tokenizer_bound": facts.get("tokenizer_bound", False),
        }
        if isinstance(entry, dict):
            kind = str(entry.get("authority_kind") or "")
            disposition, why = DISPOSITION_BY_KIND.get(
                kind, ("unclassified", "authority kind not recognised")
            )
            row.update(
                {
                    "signed_authority": True,
                    "authority_kind": kind,
                    "disposition": disposition,
                    "why": why,
                    "claims": sorted((entry.get("claims") or {}).keys()),
                    "admission_gate": (
                        "core/learning/cortex_migration_authority.py"
                        " validates this component's claims"
                    ),
                }
            )
        else:
            own_gate = facts.get("own_gate")
            # A capability outside the contract is not automatically a gap. It
            # is a gap only when nothing else decides whether it may serve, so
            # reporting one that has its own gate would be as wrong as missing
            # one that does not.
            row.update(
                {
                    "signed_authority": False,
                    "authority_kind": None,
                    "disposition": (
                        "gated_outside_the_contract" if own_gate else "uncovered"
                    ),
                    "why": own_gate
                    or (
                        "no component in the signed migration contract names "
                        "this capability and it has no gate of its own, so "
                        "nothing decides whether it may serve on this checkpoint"
                    ),
                    "claims": [],
                    "admission_gate": own_gate and facts["owner"],
                }
            )
        rows.append(row)

    by_disposition: dict[str, list[str]] = {}
    for row in rows:
        by_disposition.setdefault(row["disposition"], []).append(row["capability"])

    uncovered = by_disposition.get("uncovered", [])
    return {
        "schema": MIGRATION_QUEUE_SCHEMA,
        "model_tag": manifest.get("tag"),
        "base_model": manifest.get("base_model"),
        "signed_components": sorted(signed),
        "rows": rows,
        "by_disposition": {k: sorted(v) for k, v in sorted(by_disposition.items())},
        "uncovered_capabilities": sorted(uncovered),
        "contract_covers": f"{len(signed)} of {len(CAPABILITIES)} capabilities",
        "persona_finding": (
            "persona_crsm is signed fused_persona_crsm with a fusion plan and "
            "receipt: the persona is in this checkpoint, so it needs "
            "verification and behavioural continuity rather than a recovery run"
        )
        if isinstance(signed.get("persona_crsm"), dict)
        else "persona_crsm has no signed fusion authority on this checkpoint",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    queue = build()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(queue, indent=1, sort_keys=True))
        print(f"wrote {args.json}\n")

    if queue.get("blocked"):
        print(f"blocked: {queue['blocked']}")
        return 1

    print(f"checkpoint  {queue['model_tag']}")
    print(f"contract    {queue['contract_covers']}\n")
    for row in queue["rows"]:
        mark = (
            "signed   "
            if row["signed_authority"]
            else ("own gate " if row["disposition"] != "uncovered" else "UNCOVERED")
        )
        print(f"  {mark} {row['capability']:22s} {row['disposition']}")
        print(f"            {row['why']}")
    if queue["uncovered_capabilities"]:
        print(
            f"\n{len(queue['uncovered_capabilities'])} capability(ies) have no "
            "signed authority; an absent component is not a state."
        )
    print(f"\n{queue['persona_finding']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
