#!/usr/bin/env python3
"""Freeze one resumable 27B recurrence campaign before any model loads.

The bundle this writes is the whole launch decision, made on CPU. It commits
the checkpoint's identity and geometry, the source it will run, the datasets it
will read, the stage graph, and the point at which each stage is allowed to
give up. After it is written, launching is a matter of executing stages in
order; nothing about what the campaign IS can drift between preparation and
run, because every input carries a digest the runner re-checks.

Three properties are the reason it exists rather than a shell script.

**One model load.** The 27B is 15 GB and this host has 64. Loading it once per
stage is the difference between a campaign that fits overnight and one that
does not, so calibration, training, canary, lesion arms, the equal-compute
baseline, and artifact export all run inside a single residency, in that order.
Verification and adjudication run after the unload, against files, by a
different process — independence comes from re-reading the artifacts, not from
reloading the weights.

**Futility gates that cannot be tuned afterwards.** Each one names its measure,
its threshold, and the stage it stops, and all of them are committed in the
frozen bundle. A gate added after seeing the numbers is not a gate.

**No inherited evidence.** CP566's bounded WOW was measured on the Qwen2.5-32B
fuse. The 27B is a different model that happens to share a layer count and a
hidden size, so the bundle carries the old result as context and grants it no
authority: the recovery experiment has to earn the claim again, and the
activation stage refuses to seal a package whose evidence names another model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.learning.hybrid_recurrence_geometry import (  # noqa: E402
    LayerGeometry,
    geometry_receipt,
)

CAMPAIGN_SCHEMA: Final = "aura.rlc.27b_recurrence_campaign.v1"

#: The 32B result this campaign has to re-earn. Named so the bundle can refuse
#: to be mistaken for it, never to borrow from it.
LEGACY_CLAIM: Final = {
    "checkpoint": "Aura-32B-crsm-closeout-jul1-20260701-215118",
    "adjudication": (
        "artifacts/closeout/latent_cortex/"
        "cp566_resident_mixed_multidomain_replication/adjudication.json"
    ),
    "verdict": "BOUNDED_WOW_SIGNAL",
    "authority_over_this_campaign": "none",
    "why": (
        "measured on a different checkpoint; equal layer count and hidden size "
        "are not equal representational geometry"
    ),
}

#: Source the campaign executes. A digest of each is frozen into the bundle, so
#: an edit between preparation and launch is a refusal rather than a surprise.
SOURCE_CLOSURE: Final = (
    "core/learning/hybrid_recurrence_geometry.py",
    "core/learning/intrinsic_recurrence.py",
    "core/learning/unified_intrinsic_recurrence.py",
    "core/learning/unified_intrinsic_objective.py",
    "core/learning/recurrent_action_schema.py",
    "core/learning/recurrent_state_schema.py",
    "core/learning/recurrent_literal_grounding.py",
    "core/learning/recurrent_opcode_grounding.py",
    "core/learning/recurrent_answer_emission.py",
    "core/learning/recurrence_checkpoint_migration.py",
    "core/learning/semantic_neural_machine.py",
    "core/learning/semantic_neural_controls.py",
    "core/brain/llm/latent_cortex/semantic_surface_adapter.py",
    "core/brain/llm/latent_cortex/systematic_neural_alu.py",
    "core/brain/llm/latent_cortex/frontier_tasks.py",
    "core/brain/llm/unified_recurrent_transfer_decode.py",
    "tools/train_unified_intrinsic_recurrence.py",
    "tools/run_unified_recurrent_broad_canary.py",
    "tools/run_semantic_neural_decode_canary.py",
    "tools/prepare_27b_recurrence_campaign.py",
)

#: Tissue that carries no dimension from either checkpoint, so it crosses the
#: migration intact. The inventory decides this by reading tensor shapes; the
#: bundle records the decision and re-checks the digest.
PORTABLE_TISSUE: Final = (
    "core/brain/llm/latent_cortex/assets/systematic_neural_alu_v1/weights.safetensors",
    "core/brain/llm/latent_cortex/assets/neural_transition_tissue_v1/weights.safetensors",
    "core/brain/llm/latent_cortex/assets/mathematics_memory_tissue_v1/weights.safetensors",
)


def _sha_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# ── Stages ──────────────────────────────────────────────────────────────


def _stages(geometry: dict[str, Any]) -> list[dict[str, Any]]:
    """The order that keeps the weights resident exactly once.

    ``model_active`` marks a stage that needs the checkpoint in memory. They
    are contiguous on purpose: the load happens before the first and the unload
    after the last, and every CPU stage sits outside that span so it cannot
    hold the lane.
    """
    window = geometry["window"]
    return [
        {
            "name": "preflight",
            "model_active": False,
            "does": (
                "verify source digests, checkpoint identity, geometry, and "
                "window alignment; refuse on any drift"
            ),
            "produces": ["preflight.json"],
            "resumable": True,
        },
        {
            "name": "regrounding",
            "model_active": False,
            "does": (
                "regenerate tokenizer-bound grounding contracts against the "
                "27B vocabulary; the 32B digit and opcode ids are retired"
            ),
            "produces": ["grounding.json"],
            "resumable": True,
            "note": (
                "vocabulary moved 152,064 -> 248,320, so every recorded token "
                "id is stale and the contracts fail closed until rebuilt"
            ),
        },
        {
            "name": "calibration",
            "model_active": True,
            "does": (
                "measure ordinary decode on the frozen cohort and record the "
                "equal-compute baseline before anything is trained"
            ),
            "produces": ["baseline.json"],
            "resumable": True,
        },
        {
            "name": "training",
            "model_active": True,
            "does": f"train the controller over window {window} on typed traces",
            "produces": ["adapter/", "resume_journal.jsonl"],
            "resumable": True,
        },
        {
            "name": "canary",
            "model_active": True,
            "does": "the frozen bounded behavioral canary",
            "produces": ["canary.json"],
            "resumable": True,
        },
        {
            "name": "lesion_arms",
            "model_active": True,
            "does": (
                "coefficient lesion, syntax-matched wire base, and same-family "
                "wrong-state control on the identical cohort"
            ),
            "produces": ["lesions.json"],
            "resumable": True,
        },
        {
            "name": "export",
            "model_active": True,
            "does": "write every receipt and tensor while the weights are still up",
            "produces": ["export/"],
            "resumable": False,
        },
        {
            "name": "unload",
            "model_active": False,
            "does": "release the model lane; nothing after this may reload it",
            "produces": [],
            "resumable": False,
        },
        {
            "name": "independent_verification",
            "model_active": False,
            "does": (
                "a separate process regrades every raw response from disk and "
                "recomputes each receipt; it never sees the weights"
            ),
            "produces": ["verification.json"],
            "resumable": True,
        },
        {
            "name": "adjudication",
            "model_active": False,
            "does": "the fixed adjudicator renders a verdict and its limitations",
            "produces": ["adjudication.json"],
            "resumable": True,
        },
        {
            "name": "activation_materialization",
            "model_active": False,
            "does": (
                "seal a content-addressed package bound to THIS checkpoint; "
                "refuse if the evidence names another model"
            ),
            "produces": ["activation.json"],
            "resumable": True,
        },
    ]


#: Each names a measure, a threshold, and what it stops. Committed before the
#: run, because a threshold chosen after seeing the numbers measures nothing.
FUTILITY_GATES: Final = (
    {
        "after": "calibration",
        "measure": "ordinary_base_parsed_fraction",
        "stops_when": "< 0.2",
        "reason": (
            "if ordinary decode cannot even produce parseable answers on the "
            "cohort, a gain over it is not a reasoning result"
        ),
    },
    {
        "after": "calibration",
        "measure": "grounding_contract_status",
        "stops_when": "!= complete",
        "reason": "an ungrounded tokenizer makes every typed comparison meaningless",
    },
    {
        "after": "training",
        "measure": "heldout_state_accuracy_delta_vs_initialization",
        "stops_when": "<= 0",
        "reason": (
            "trained parameters that do not beat their own initialization "
            "control have nothing to serve"
        ),
    },
    {
        "after": "canary",
        "measure": "treatment_exact",
        "stops_when": "<= ordinary_base_exact",
        "reason": "no gain to adjudicate; stop before spending the lesion arms",
    },
    {
        "after": "lesion_arms",
        "measure": "families_separating_under_lesion",
        "stops_when": "< all",
        "reason": (
            "a gain that survives its own lesion is a decode-budget effect, "
            "not a claim about trained coefficients"
        ),
    },
)


def _experiments() -> dict[str, Any]:
    """Recovery first, then generalization. Never both in one verdict."""
    return {
        "recovery": {
            "goal": (
                "re-earn the CP566 bounded claim on the 27B, on the same "
                "frozen four-domain cohort and the same arm structure"
            ),
            "cohort": "frozen four-domain semantic cohort, 15 tasks per family",
            "task_count": 60,
            "arms": [
                "treatment",
                "ordinary_base",
                "matched_wire_base",
                "coefficient_lesion",
                "matched_wrong_state",
            ],
            "decodes": 300,
            "success": (
                "treatment beats ordinary_base with zero regressions, every "
                "family separates under lesion, paired one-sided exact p < 0.01"
            ),
            "authorizes": (
                "a bounded 27B claim on these four executable families and "
                "nothing else"
            ),
        },
        "generalization": {
            "goal": "whether the recovered gain reaches families it was not trained on",
            "precondition": "recovery adjudicated positive",
            "cohort": "held-out families, frozen before the recovery run is read",
            "authorizes": (
                "a wider bounded claim if it passes; never ordinary chat and "
                "never global serving"
            ),
        },
        "never_authorized_by_training_completion": [
            "ordinary_chat_authorized",
            "arbitrary_reasoning_authorized",
            "global runtime promotion",
            "static weight fusion",
            "frontier performance",
        ],
    }


def build(model_path: Path, install: Path) -> dict[str, Any]:
    config = json.loads((model_path / "config.json").read_text())
    geometry = LayerGeometry.from_config(config)
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config

    prelude_end = int(geometry.num_hidden_layers * 0.25)
    coda_start = geometry.num_hidden_layers - prelude_end
    receipt = geometry_receipt(
        geometry, prelude_end, coda_start, ("o_proj", "v_proj", "down_proj")
    )
    if receipt["alignment_errors"]:
        raise SystemExit(
            "window is misaligned for this checkpoint: "
            + "; ".join(receipt["alignment_errors"])
        )

    weight_files = sorted(model_path.glob("*.safetensors"))
    descriptor = {
        "path": str(model_path),
        "config_sha256": _sha_file(model_path / "config.json"),
        "weights_index_sha256": (
            _sha_file(model_path / "model.safetensors.index.json")
            if (model_path / "model.safetensors.index.json").exists()
            else None
        ),
        "tokenizer_sha256": (
            _sha_file(model_path / "tokenizer.json")
            if (model_path / "tokenizer.json").exists()
            else None
        ),
        "fusion_provenance_sha256": (
            _sha_file(model_path / "aura_fusion_provenance.json")
            if (model_path / "aura_fusion_provenance.json").exists()
            else None
        ),
        "weight_file_count": len(weight_files),
        "weight_bytes": sum(path.stat().st_size for path in weight_files),
        "model_type": config.get("model_type") or text.get("model_type"),
        "architectures": config.get("architectures"),
        "vocab_size": text.get("vocab_size"),
        "max_position_embeddings": text.get("max_position_embeddings"),
    }

    source_freeze = {}
    for relative in SOURCE_CLOSURE:
        path = REPO_ROOT / relative
        if not path.exists():
            raise SystemExit(f"source closure names a missing file: {relative}")
        source_freeze[relative] = _sha_file(path)

    portable = {}
    for relative in PORTABLE_TISSUE:
        path = REPO_ROOT / relative
        if not path.exists():
            raise SystemExit(f"portable tissue is missing: {relative}")
        portable[relative] = _sha_file(path)

    stages = _stages(receipt)
    active = [stage["name"] for stage in stages if stage["model_active"]]
    first = next(i for i, s in enumerate(stages) if s["model_active"])
    last = max(i for i, s in enumerate(stages) if s["model_active"])
    if any(not stages[i]["model_active"] for i in range(first, last + 1)):
        raise SystemExit("model-active stages are not contiguous; that is two loads")

    body = {
        "schema": CAMPAIGN_SCHEMA,
        "target_checkpoint": descriptor,
        "geometry": receipt,
        "recurrence_layer_mapping": {
            "attention_layer_indices": list(geometry.attention_layers()),
            "linear_attention_layer_indices": list(geometry.linear_layers()),
            "full_attention_interval": geometry.full_attention_interval,
            "window": [prelude_end, coda_start],
            "why_aligned": (
                "each group of four ends in the attention layer that mixes "
                "across tokens, so an aligned window ends every iteration on "
                "one rather than on a positionwise update"
            ),
        },
        "source_freeze": source_freeze,
        "portable_tissue": portable,
        "stages": stages,
        "model_active_stages": active,
        "single_residency": {
            "load_before": stages[first]["name"],
            "unload_after": stages[last]["name"],
            "independence_comes_from": (
                "re-reading exported artifacts in a separate process, not from "
                "reloading the weights"
            ),
        },
        "futility_gates": list(FUTILITY_GATES),
        "experiments": _experiments(),
        "legacy_claim": LEGACY_CLAIM,
        "lifecycle_owner": {
            "runner": "tools/run_unified_intrinsic_resident_campaign.py",
            "provides": [
                "authenticated heartbeat",
                "independent watchdog",
                "caffeinate custody",
                "durable resume journal",
                "append-only attempt ledger",
                "campaign and model-lane locks",
            ],
            "note": (
                "this bundle is the campaign's definition; the runner above is "
                "its lifecycle. Neither reimplements the other."
            ),
        },
    }
    return {**body, "campaign_sha256": _sha(body)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)

    install = REPO_ROOT
    if not (install / "training/fused-model/active.json").exists():
        import subprocess

        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        ).stdout.strip()
        if common:
            install = Path(common).parent

    model_path = args.model
    if model_path is None:
        manifest = json.loads(
            (install / "training/fused-model/active.json").read_text()
        )
        model_path = Path(manifest["active_model_path"])
    model_path = model_path.expanduser().resolve(strict=True)

    bundle = build(model_path, install)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(bundle, indent=1, sort_keys=True))
        print(f"wrote {args.out}")
    if args.summary or not args.out:
        print(f"campaign_sha256 {bundle['campaign_sha256']}")
        print(f"checkpoint      {bundle['target_checkpoint']['path']}")
        print(
            f"window          {bundle['geometry']['window']} "
            f"({bundle['geometry']['expected_adapter_site_count']} adapter sites)"
        )
        print(f"model-active    {', '.join(bundle['model_active_stages'])}")
        print(f"futility gates  {len(bundle['futility_gates'])}")
        print(f"source frozen   {len(bundle['source_freeze'])} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
