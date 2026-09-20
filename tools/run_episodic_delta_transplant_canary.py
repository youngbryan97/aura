#!/usr/bin/env python3
"""Prove an accepted episodic delta survives an exact persistent transplant.

This is a mechanism discriminator, not a reasoning benchmark.  It first runs
one source-bound 1.5B RLC episode with candidate export enabled, then reloads
the frozen checkpoint and compares that real episodic operator against
same-site decode-scoped LoRA tissue on identical model inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.brain.llm.latent_cortex.incumbent_artifact import build_incumbent_artifact
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
    full_weight_checkpoint_identity,
)
from core.brain.llm.latent_cortex.task_verifiers import EpisodeTaskVerifier
from core.brain.llm.latent_cortex.verified_best import tensor_sha256
from core.learning.recurrence_curriculum import modular_chain
from core.learning.recurrent_behavioral_probe import (
    _full_engine_config,
    _ordinary_decode_once,
    free_generation_sampling_config,
    paired_generation_seed,
    tokenize_task,
)
from core.learning.recurrent_checkpoint_admission import build_recurrence_task_manifest
from core.learning.recurrent_grpo import (
    attach_coda_policy_adapters_at_sites,
    attach_recurrent_policy_adapters,
)
from core.learning.recurrent_sft_execution import (
    adapter_tensor_dict,
    adapter_tensor_fingerprint,
)
from core.learning.verified_trajectory_distillation import (
    compile_episodic_delta_inventory,
    install_verified_trajectory_inventory,
    load_verified_trajectory_artifact,
    publish_verified_trajectory_artifact,
)
from core.runtime.atomic_writer import atomic_write_bytes
from core.runtime.mlx_memory_guard import mlx_memory_envelope
from core.runtime.model_lane_control import standalone_model_lane

SCHEMA = "aura.rlc.episodic_delta_transplant_canary.v1"
CAMPAIGN_SEED = 20_260_810_156
TASK_SEED = 2_394_370_837_916_956_658
EPISODE_ID = "episodic-transplant-modular-d3-v1"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.fsmonitor=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _emit(path: Path, event: str, **details: Any) -> None:
    row = {"at": time.time(), "event": event, **details}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json_bytes(row).decode("ascii") + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def _write_private_json(path: Path, value: Any) -> None:
    atomic_write_bytes(
        path,
        json.dumps(value, indent=2, sort_keys=True).encode("ascii") + b"\n",
        mode=0o600,
    )


def _producer_export_diagnostic(receipt: dict[str, Any]) -> dict[str, Any]:
    flags = tuple(str(flag) for flag in receipt.get("honest_flags", ()))
    loss_trail = tuple(float(value) for value in receipt.get("fast_weight_loss_trail", ()))
    accepted_steps = int(receipt.get("fast_weight_optimized_steps", 0))
    loss_improved = len(loss_trail) >= 2 and loss_trail[-1] < loss_trail[0]
    verifier_decision = str(
        (receipt.get("fast_weight_verifier") or {}).get("decision", "")
    )
    learning_disposition = str(
        (receipt.get("fast_weight_learning") or {}).get("disposition", "")
    )
    adaptation_retained = (
        verifier_decision == "accepted_causal_improvement"
        and learning_disposition
        in {
            "accepted_causal_improvement",
            "accepted_probe_not_output_under_incumbent_policy",
        }
    )
    prerequisites = {
        "checkpoint_bound": bool(receipt.get("checkpoint_fingerprint")),
        "erase_proven": receipt.get("fast_weights_erased") is True,
        "adaptation_retained": adaptation_retained,
        "accepted_step": accepted_steps > 0,
        "loss_improved": loss_improved,
    }
    exported = "fast_weight_candidate_exported" in flags
    return {
        "schema": "aura.rlc.episodic_delta_export_diagnostic.v1",
        "exported": exported,
        "eligible_by_receipt": all(prerequisites.values()),
        "prerequisites": prerequisites,
        "accepted_steps": accepted_steps,
        "rejected_steps": int(receipt.get("fast_weight_rejected_steps", 0)),
        "verifier_decision": verifier_decision,
        "learning_disposition": learning_disposition,
        "loss_trail": list(loss_trail),
        "honest_flags": list(flags),
        "reason": (
            "exported"
            if exported
            else "export_boundary_failed_or_refused"
            if all(prerequisites.values())
            else "producer_not_export_eligible"
        ),
    }


def _load_candidate(
    candidate_dir: Path,
    *,
    episode_id: str,
    expected_layers: tuple[int, ...],
    expected_rank: int,
    scale: float,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    evidence_path = candidate_dir / "evidence.json"
    delta_path = candidate_dir / "delta_weights.npz"
    evidence_bytes = evidence_path.read_bytes()
    evidence = json.loads(evidence_bytes)
    binding = (evidence.get("artifacts") or {}).get("delta_weights.npz") or {}
    delta_bytes = delta_path.read_bytes()
    if (
        evidence.get("schema") != "aura.latent_cortex.fast_weight_candidate.v1"
        or evidence.get("episode_id") != episode_id
        or tuple(evidence.get("layers") or ()) != expected_layers
        or evidence.get("rank") != expected_rank
        or evidence.get("target") != "o_proj"
        or binding.get("sha256") != _sha256_bytes(delta_bytes)
        or binding.get("size_bytes") != len(delta_bytes)
    ):
        raise RuntimeError("exported episodic candidate identity differs")
    expected_keys = {
        f"layer{layer}_{factor}"
        for layer in expected_layers
        for factor in ("U", "V")
    }
    with np.load(delta_path, allow_pickle=False) as arrays:
        if set(arrays.files) != expected_keys:
            raise RuntimeError("exported episodic tensor inventory differs")
        snapshots = tuple(
            {
                "layer": layer,
                "scale": scale,
                "U": np.asarray(arrays[f"layer{layer}_U"]).copy(),
                "V": np.asarray(arrays[f"layer{layer}_V"]).copy(),
            }
            for layer in expected_layers
        )
    return snapshots, {
        "candidate_dir": str(candidate_dir),
        "evidence_sha256": _sha256_bytes(evidence_bytes),
        "delta_sha256": _sha256_bytes(delta_bytes),
        "delta_size_bytes": len(delta_bytes),
    }


def _model_logits(model: Any, tokens: list[int]):
    import mlx.core as mx

    output = model(mx.array([tokens]))
    logits = getattr(output, "logits", output)
    mx.eval(logits)
    return logits


def run(
    *,
    repo: Path,
    model_path: Path,
    out_dir: Path,
    expected_source: str,
) -> dict[str, Any]:
    source = _git(repo, "rev-parse", "HEAD")
    if source != expected_source or source != _git(repo, "rev-parse", "origin/main"):
        raise RuntimeError("source identity differs from expected origin/main")
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("source worktree is not clean")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError("output directory is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"
    private_data = out_dir / "private-data"
    private_data.mkdir()

    task = modular_chain(3, TASK_SEED)
    tasks = [task]
    task_manifest, task_manifest_sha256 = build_recurrence_task_manifest(tasks)
    spec = RLCExecutionSpec(
        n_slots=8,
        branch_roles=("constructive_solution", "critical_audit"),
        recurrent_steps=1,
        exchange_interval=1,
    )
    checkpoint = full_weight_checkpoint_identity(model_path)
    started = time.monotonic()
    _emit(
        progress_path,
        "campaign_started",
        source_commit=source,
        task_id=task.task_id,
        execution_spec_sha256=spec.sha256,
    )

    import mlx.core as mx
    from mlx_lm import load

    import core.config as aura_config
    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.fast_weights import EpisodicFastWeights
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        coda_adapter_disabled,
        coda_adapter_scope,
        recurrence_adapter_activation_collector,
    )
    from core.brain.llm.latent_cortex.types import ComputeBudget, FastWeightsConfig

    previous_data_dir = aura_config.DATA_DIR
    try:
        aura_config.DATA_DIR = private_data
        with (
            standalone_model_lane(
                owner_id=f"episodic-transplant:{source[:12]}",
                model_path=str(model_path),
                purpose="evaluation",
                priority=10,
                preemptible=True,
                metadata={"checkpoint": "CP157", "source_commit": source},
            ),
            mlx_memory_envelope(fraction=0.30, restore_limits_on_exit=True),
        ):
            model, tokenizer = load(str(model_path))
            recurrent_sites = attach_recurrent_policy_adapters(
                model,
                spec,
                lora_rank=8,
                lora_layers=8,
                lora_targets=("o_proj",),
                initialization_seed=CAMPAIGN_SEED & 0xFFFFFFFF,
                lora_scale=1.0,
                lora_layer_placement="early",
            )
            adapter_sha256 = adapter_tensor_fingerprint(adapter_tensor_dict(model))
            prompt_tokens, _answer_tokens = tokenize_task(
                tokenizer,
                task.prompt,
                task.answer,
            )
            max_tokens = free_generation_sampling_config().max_tokens
            generation_seed = paired_generation_seed(
                CAMPAIGN_SEED,
                0,
                task.task_id,
                1,
            )
            mx.random.seed(generation_seed)
            ordinary_text, ordinary_tokens, ordinary_termination = _ordinary_decode_once(
                model,
                tokenizer,
                prompt_tokens,
                max_tokens=max_tokens,
            )
            incumbent = build_incumbent_artifact(
                input_tokens=prompt_tokens,
                output_tokens=ordinary_tokens,
                output_text=ordinary_text,
                checkpoint_fingerprint=checkpoint["fingerprint"],
                checkpoint_fingerprint_method=checkpoint["method"],
                max_tokens=max_tokens,
                n_layers=len(model.model.layers),
                termination=ordinary_termination,
            )
            config = _full_engine_config(
                spec,
                objective_program_enabled=False,
                verified_objective_teacher_enabled=True,
            )
            config.fast_weights.canary_enabled = False
            config.fast_weights.export_candidates = True
            problems = config.validate()
            if problems:
                raise RuntimeError(f"transplant producer config rejected: {problems}")
            # Source is independently committed in the campaign receipt. It
            # must not seed the scientific arm: doing so changed adapter
            # initialization and the deterministic sham after every code-only
            # checkpoint, making consecutive reruns incomparable.
            episode_id = EPISODE_ID
            engine = LatentCortexEngine(
                model,
                tokenizer=tokenizer,
                config=config,
                model_path=str(model_path),
                schedule_library=None,
            )
            mx.random.seed(generation_seed)
            result = engine.reason(
                messages=[{"role": "user", "content": task.prompt}],
                verifier=EpisodeTaskVerifier(
                    task.prompt,
                    response_contract=task.response_contract,
                ),
                budget=ComputeBudget(wall_clock_s=600.0),
                domain=task.domain,
                decode_max_tokens=max_tokens,
                decode_sentence_grace_tokens=0,
                nonparametric_memory_enabled=False,
                sample_seed=generation_seed,
                incumbent_artifact=incumbent,
                episode_id=episode_id,
            )
            producer_receipt = result.receipt.to_dict()
            _write_private_json(
                out_dir / "producer_receipt.private.json",
                producer_receipt,
            )
            export_diagnostic = _producer_export_diagnostic(producer_receipt)
            _write_private_json(
                out_dir / "producer_export_diagnostic.json",
                export_diagnostic,
            )
            _emit(
                progress_path,
                "producer_completed",
                episode_ok=result.ok,
                exported=export_diagnostic["exported"],
                export_reason=export_diagnostic["reason"],
                accepted_steps=export_diagnostic["accepted_steps"],
            )
            locality = producer_receipt.get("fast_weight_locality") or {}
            if locality.get("schema") != "aura.rlc.fast_weight_phase_locality.v1":
                raise RuntimeError("episodic producer did not measure phase locality")
            if "fast_weight_candidate_exported" not in producer_receipt.get(
                "honest_flags", []
            ):
                raise RuntimeError(
                    "episodic producer did not export its candidate: "
                    + json.dumps(export_diagnostic, sort_keys=True)
                )
            candidate_dir = (
                private_data
                / "latent_cortex"
                / "consolidation_queue"
                / episode_id
            )
            expected_layers = tuple(int(site.split(".")[2]) for site in recurrent_sites)
            snapshots, candidate_binding = _load_candidate(
                candidate_dir,
                episode_id=episode_id,
                expected_layers=expected_layers,
                expected_rank=config.fast_weights.rank,
                scale=config.fast_weights.scale,
            )
            episodic_delta_sha256 = locality["delta_sha256"]
            producer_grade = dict(task.grade(result.text if result.ok else ""))
            producer_ok = bool(result.ok)
            producer_response_sha256 = _sha256_bytes(result.text.encode("utf-8"))
            _emit(
                progress_path,
                "candidate_exported",
                episode_ok=result.ok,
                episode_correct=bool(producer_grade.get("correct")),
                delta_sha256=episodic_delta_sha256,
            )

            del engine, result, model
            mx.synchronize()
            mx.clear_cache()

            model, _transplant_tokenizer = load(str(model_path))
            rank = int(snapshots[0]["U"].shape[1])
            fast_weights = EpisodicFastWeights(
                FastWeightsConfig(
                    enabled=True,
                    rank=rank,
                    scale=float(snapshots[0]["scale"]),
                    target="o_proj",
                    max_wrapped_layers=len(snapshots),
                    layer_placement="early",
                )
            )
            prelude_end = max(1, int(len(model.model.layers) * spec.prelude_frac))
            coda_start = min(
                len(model.model.layers) - 1,
                len(model.model.layers)
                - max(1, int(len(model.model.layers) * spec.coda_frac)),
            )
            fast_weights.attach(
                model.model,
                (prelude_end, coda_start),
                seed_stat=1.0,
                episode_id=f"{episode_id}-replay",
            )
            replay_layers = tuple(handle.layer_index for handle in fast_weights.handles)
            if replay_layers != expected_layers:
                raise RuntimeError("episodic replay sites differ from exported candidate")
            fast_weights.restore_delta(
                snapshots,
                reason="episodic_transplant_replay",
            )
            fast_weights.activate_adaptation_path()
            fast_weights.set_activation_policy("decode_only")

            probe_tokens = prompt_tokens[-min(64, len(prompt_tokens)) :]
            base_logits = _model_logits(model, probe_tokens)
            with coda_adapter_scope():
                episodic_logits = _model_logits(model, probe_tokens)
            episodic_locality = fast_weights.activation_locality_receipt()
            episodic_logits_sha256 = tensor_sha256(episodic_logits)
            episodic_tokens = mx.argmax(episodic_logits, axis=-1)
            mx.eval(episodic_tokens)
            fast_weights.detach()
            detached_logits = _model_logits(model, probe_tokens)
            if not bool(mx.array_equal(base_logits, detached_logits)):
                raise RuntimeError("episodic wrapper did not erase exactly")

            inventory = compile_episodic_delta_inventory(
                snapshots,
                target="o_proj",
            )
            artifact_publication = publish_verified_trajectory_artifact(
                out_dir / "trajectory_artifact",
                inventory,
                checkpoint_fingerprint=checkpoint["fingerprint"],
                source_evidence_sha256=episodic_delta_sha256,
            )
            del inventory
            inventory, artifact_manifest = load_verified_trajectory_artifact(
                artifact_publication["artifact_dir"],
                expected_checkpoint_fingerprint=checkpoint["fingerprint"],
                expected_source_evidence_sha256=episodic_delta_sha256,
            )
            sites = attach_coda_policy_adapters_at_sites(
                model,
                tuple(inventory),
                lora_rank=rank,
                initialization_seed=CAMPAIGN_SEED & 0xFFFFFFFF,
                lora_scale=float(snapshots[0]["scale"]),
            )
            install_receipt = install_verified_trajectory_inventory(
                model,
                inventory,
                expected_sites=sites,
            )
            with recurrence_adapter_activation_collector() as ordinary_activation:
                persistent_ordinary_logits = _model_logits(model, probe_tokens)
            with recurrence_adapter_activation_collector() as coda_activation:
                with coda_adapter_scope():
                    persistent_logits = _model_logits(model, probe_tokens)
            with coda_adapter_disabled(), coda_adapter_scope():
                lesioned_logits = _model_logits(model, probe_tokens)
            persistent_tokens = mx.argmax(persistent_logits, axis=-1)
            mx.eval(persistent_tokens)

            parity = {
                "base_outside_scope_equal": bool(
                    mx.array_equal(base_logits, persistent_ordinary_logits)
                ),
                "episodic_persistent_logits_equal": bool(
                    mx.array_equal(episodic_logits, persistent_logits)
                ),
                "episodic_persistent_tokens_equal": bool(
                    mx.array_equal(episodic_tokens, persistent_tokens)
                ),
                "coda_lesion_restores_base": bool(
                    mx.array_equal(base_logits, lesioned_logits)
                ),
                "ordinary_applied_sites": dict(ordinary_activation.applied_sites),
                "coda_applied_sites": dict(coda_activation.applied_sites),
                "coda_unfired_sites": coda_activation.unfired_sites(sites),
                "episodic_logits_sha256": episodic_logits_sha256,
                "persistent_logits_sha256": tensor_sha256(persistent_logits),
                "base_logits_sha256": tensor_sha256(base_logits),
                "lesioned_logits_sha256": tensor_sha256(lesioned_logits),
                "episodic_tokens_sha256": tensor_sha256(episodic_tokens),
                "persistent_tokens_sha256": tensor_sha256(persistent_tokens),
            }
            _write_private_json(
                out_dir / "parity_diagnostic.json",
                parity,
            )
            if not all(
                parity[key]
                for key in (
                    "base_outside_scope_equal",
                    "episodic_persistent_logits_equal",
                    "episodic_persistent_tokens_equal",
                    "coda_lesion_restores_base",
                )
            ):
                raise RuntimeError("episodic-to-persistent model parity failed")
            if parity["ordinary_applied_sites"] or parity["coda_unfired_sites"]:
                raise RuntimeError("persistent coda activation inventory differs")
            if set(parity["coda_applied_sites"]) != set(sites):
                raise RuntimeError("persistent coda activation missed an exact site")

            receipt = {
                "schema": SCHEMA,
                "source_commit": source,
                "model_path": str(model_path),
                "checkpoint": checkpoint,
                "task_manifest": task_manifest,
                "task_manifest_sha256": task_manifest_sha256,
                "execution_spec": spec.to_dict(),
                "execution_spec_sha256": spec.sha256,
                "campaign_seed": CAMPAIGN_SEED,
                "adapter_sha256": adapter_sha256,
                "episodic_delta_sha256": episodic_delta_sha256,
                "candidate": candidate_binding,
                "candidate_sites": list(sites),
                "candidate_factor_receipts": {
                    site: inventory[site].receipt for site in sites
                },
                "trajectory_artifact": {
                    "artifact_dir": artifact_publication["artifact_dir"],
                    "manifest_receipt_sha256": artifact_manifest["receipt_sha256"],
                    "tensor_sha256": artifact_manifest["tensor_artifact"]["sha256"],
                    "publication": artifact_publication["publication"],
                },
                "install_receipt": install_receipt,
                "episodic_activation": episodic_locality,
                "parity": parity,
                "producer": {
                    "episode_ok": producer_ok,
                    "correct": bool(producer_grade.get("correct")),
                    "response_sha256": producer_response_sha256,
                    "receipt_sha256": _sha256_bytes(
                        canonical_json_bytes(producer_receipt)
                    ),
                    "phase_locality": locality,
                },
                "elapsed_s": round(time.monotonic() - started, 3),
                "claim_boundary": (
                    "exact_real_checkpoint_mechanism_transplant_not_reasoning_gain"
                ),
            }
            receipt["receipt_sha256"] = _sha256_bytes(canonical_json_bytes(receipt))
            (out_dir / "receipt.json").write_bytes(
                json.dumps(receipt, indent=2, sort_keys=True).encode("ascii") + b"\n"
            )
            _emit(
                progress_path,
                "campaign_completed",
                receipt_sha256=receipt["receipt_sha256"],
                elapsed_s=receipt["elapsed_s"],
            )
            return receipt
    finally:
        aura_config.DATA_DIR = previous_data_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-source", required=True)
    args = parser.parse_args()
    run(
        repo=args.repo.resolve(),
        model_path=args.model.resolve(),
        out_dir=args.out.resolve(),
        expected_source=args.expected_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
