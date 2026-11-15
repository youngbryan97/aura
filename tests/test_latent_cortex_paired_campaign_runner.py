from __future__ import annotations

import base64
import hashlib
import json
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_journal import (
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    TASK_ISSUER,
    build_role_attestation,
    validate_campaign_trust_policy,
)
from core.brain.llm.latent_cortex.detached_campaign_evidence import (
    VerifiedDetachedBrokerEvidence,
    VerifiedDetachedTerminal,
)
from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.brain.llm.latent_cortex.frontier_tasks import generate_task_battery
from core.brain.llm.latent_cortex.paired_campaign import build_campaign_plan
from core.brain.llm.latent_cortex.resident_adapter_loader import (
    ResidentAdapterLoadError,
    load_resident_adapter,
)
from core.brain.llm.latent_cortex.resource_accounting import (
    ModelComputeProfile,
    ResourceLedger,
    build_information_receipt,
)
from core.learning.recurrent_grpo import attach_recurrent_policy_adapters
from tests.clock_patch import patch_module_clock
from tools import run_latent_cortex_paired_campaign as runner


def test_resident_sft_absent_personality_identity_is_semantically_bound() -> None:
    plain_absence = runner.personality_bundle_identity(None)
    semantic_absence = runner.absent_personality_identity()

    assert plain_absence == {
        "present": False,
        "bundle_sha256": "",
        "file_count": 0,
        "files": [],
    }
    assert semantic_absence == {
        **plain_absence,
        "identity_sha256": runner.resident_recurrent_sft_adapter_identity.sha256_json(
            plain_absence
        ),
    }


def test_resident_adapter_loader_rejects_symlinked_package_root(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    alias = tmp_path / "package-alias"
    alias.symlink_to(package, target_is_directory=True)

    with pytest.raises(
        ResidentAdapterLoadError,
        match="resident_adapter_package_symlink_forbidden",
    ):
        load_resident_adapter(object(), alias, {})


def test_depth_conditioned_adapter_load_reconstructs_and_reads_back_bank(
    tmp_path: Path,
) -> None:
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen2 import Model, ModelArgs

    def model() -> Model:
        return Model(
            ModelArgs(
                model_type="qwen2",
                hidden_size=32,
                num_hidden_layers=4,
                intermediate_size=64,
                num_attention_heads=4,
                rms_norm_eps=1e-6,
                vocab_size=64,
                num_key_value_heads=2,
                max_position_embeddings=128,
                rope_theta=10000.0,
            )
        )

    spec = RLCExecutionSpec(
        n_slots=2,
        branch_roles=("constructive_solution",),
        recurrent_steps=2,
        prelude_frac=0.25,
        coda_frac=0.25,
    )
    source = model()
    attach_recurrent_policy_adapters(
        source,
        spec,
        lora_rank=2,
        lora_layers=1,
        lora_targets=("q_proj", "o_proj"),
        initialization_seed=41,
        depth_conditioned_steps=2,
    )
    source.model.layers[2].self_attn.q_proj.depth_a[1] = mx.ones(
        source.model.layers[2].self_attn.q_proj.depth_a[1].shape
    )
    tensors = dict(tree_flatten(source.trainable_parameters()))
    mx.eval(tensors)
    adapter_path = tmp_path / "adapter.safetensors"
    mx.save_safetensors(str(adapter_path), tensors)
    payload = adapter_path.read_bytes()
    projections = [
        "model.layers.2.self_attn.o_proj",
        "model.layers.2.self_attn.q_proj",
    ]
    manifest = {
        "schema": runner.resident_recurrent_sft_adapter_identity.MANIFEST_SCHEMA,
        "lora": {
            "rank": 2,
            "scale": 20.0,
            "dropout": 0.0,
            "targets": ["q_proj", "o_proj"],
            "wrapped_projections": 2,
            "projection_paths": projections,
            "conditioning_schema": "aura.depth_conditioned_lora.v1",
            "depth_bank_size": 2,
        },
        "tensors": [
            {"key": key, "shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in sorted(tensors.items())
        ],
        "bindings": {
            "adapter": {
                "path": adapter_path.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        },
    }
    target = model()
    original_q_proj = target.model.layers[2].self_attn.q_proj
    attacked = json.loads(json.dumps(manifest))
    attacked["bindings"]["adapter"]["sha256"] = "0" * 64

    with pytest.raises(
        runner.CampaignProducerError,
        match="resident_adapter_weights_identity_mismatch",
    ):
        runner._load_adapter(target, tmp_path, attacked)
    assert target.model.layers[2].self_attn.q_proj is original_q_proj
    assert not hasattr(original_q_proj, "depth_bank")

    loaded_count = runner._load_adapter(target, tmp_path, manifest)
    loaded = dict(tree_flatten(target.parameters()))
    mx.eval(*(loaded[key] for key in sorted(tensors)))

    assert loaded_count == 2
    assert set(tensors).issubset(loaded)
    assert all(bool(mx.array_equal(loaded[key], value)) for key, value in tensors.items())
    assert bool(mx.all(target.model.layers[2].self_attn.q_proj.depth_a[1] == 1.0))


def test_role_conditioned_adapter_load_reconstructs_and_reads_back_bank(
    tmp_path: Path,
) -> None:
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen2 import Model, ModelArgs

    def model() -> Model:
        return Model(
            ModelArgs(
                model_type="qwen2",
                hidden_size=32,
                num_hidden_layers=4,
                intermediate_size=64,
                num_attention_heads=4,
                rms_norm_eps=1e-6,
                vocab_size=64,
                num_key_value_heads=2,
                max_position_embeddings=128,
                rope_theta=10000.0,
            )
        )

    spec = RLCExecutionSpec(
        n_slots=2,
        branch_roles=("constructive_solution", "adversarial_verifier"),
        recurrent_steps=2,
        prelude_frac=0.25,
        coda_frac=0.25,
    )
    source = model()
    attach_recurrent_policy_adapters(
        source,
        spec,
        lora_rank=2,
        lora_layers=1,
        lora_targets=("q_proj", "o_proj"),
        initialization_seed=41,
        depth_conditioned_steps=2,
        role_conditioned_branches=2,
    )
    source.model.layers[2].self_attn.q_proj.role_a[1] = mx.ones(
        source.model.layers[2].self_attn.q_proj.role_a[1].shape
    )
    source.model.layers[2].self_attn.q_proj.role_b[1] = mx.ones(
        source.model.layers[2].self_attn.q_proj.role_b[1].shape
    )
    tensors = dict(tree_flatten(source.trainable_parameters()))
    mx.eval(tensors)
    adapter_path = tmp_path / "adapter.safetensors"
    mx.save_safetensors(str(adapter_path), tensors)
    payload = adapter_path.read_bytes()
    projections = [
        "model.layers.2.self_attn.o_proj",
        "model.layers.2.self_attn.q_proj",
    ]
    manifest = {
        "schema": (runner.resident_recurrent_sft_adapter_identity.ROLE_CONDITIONED_MANIFEST_SCHEMA),
        "lora": {
            "rank": 2,
            "scale": 20.0,
            "dropout": 0.0,
            "targets": ["q_proj", "o_proj"],
            "wrapped_projections": 2,
            "projection_paths": projections,
            "conditioning_schema": "aura.depth_conditioned_lora.v1",
            "depth_bank_size": 2,
            "role_conditioning_schema": "aura.role_conditioned_lora.v1",
            "role_bank_size": 2,
        },
        "tensors": [
            {"key": key, "shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in sorted(tensors.items())
        ],
        "bindings": {
            "adapter": {
                "path": adapter_path.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        },
    }
    target = model()

    loaded_count = runner._load_adapter(target, tmp_path, manifest)
    loaded = dict(tree_flatten(target.parameters()))
    mx.eval(*(loaded[key] for key in sorted(tensors)))

    assert loaded_count == 2
    assert set(tensors).issubset(loaded)
    assert all(bool(mx.array_equal(loaded[key], value)) for key, value in tensors.items())
    assert bool(mx.all(target.model.layers[2].self_attn.q_proj.role_a[1] == 1.0))
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )
    from core.learning.depth_conditioned_lora import recurrent_depth_index
    from core.learning.role_conditioned_lora import recurrent_branch_index

    projection = target.model.layers[2].self_attn.q_proj
    x = mx.ones((1, 3, 32))
    with (
        recurrence_adapter_scope(start=0, stop=3),
        recurrent_depth_index(0),
        recurrent_branch_index(0),
    ):
        branch_zero = projection(x)
    with (
        recurrence_adapter_scope(start=0, stop=3),
        recurrent_depth_index(0),
        recurrent_branch_index(1),
    ):
        branch_one = projection(x)
    mx.eval(branch_zero, branch_one)

    assert not bool(mx.array_equal(branch_zero, branch_one))


def _synthetic_claim_plan_for_nonstatistical_contract(
    unsigned: CampaignPlan,
    *,
    campaign_trust: dict,
) -> CampaignPlan:
    """Elevate a small fixture only for worker/trust state-machine tests."""

    document = unsigned.to_dict()
    metadata = document["metadata"]
    metadata["claim_eligible"] = True
    metadata["claim_scope"] = "resident same-checkpoint causal attribution"
    metadata["campaign_trust"] = campaign_trust
    return CampaignPlan.build(
        document["campaign_name"],
        [unsigned.cell_definition(cell_id) for cell_id in unsigned.cell_ids],
        metadata=metadata,
    )


def _external_policy_fixture(campaign_name: str, now: int):
    root = Ed25519PrivateKey.generate()
    role_keys = {role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES}
    roles = {}
    for role, key in role_keys.items():
        raw = key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        roles[role] = {
            "signer_id": f"{role}-signer",
            "organization_id": f"{role}-organization",
            "public_key_b64": base64.b64encode(raw).decode("ascii"),
            "key_id": hashlib.sha256(raw).hexdigest(),
            "implementation_sha256": hashlib.sha256(f"{role}:impl".encode()).hexdigest(),
            "release_sha256": hashlib.sha256(f"{role}:release".encode()).hexdigest(),
            "custody_class": "remote_hsm",
            "custody_evidence_sha256": hashlib.sha256(f"{role}:custody".encode()).hexdigest(),
        }
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": f"{campaign_name}-policy",
        "policy_revision": 1,
        "campaign_name": campaign_name,
        "protocol_sha256": runner._campaign_protocol_sha256(),
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": now - 10,
        "not_before_unix": now - 5,
        "expires_at_unix": now + 3600,
        "roles": roles,
    }
    root_raw = root.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    signed = canonical_json_bytes(body)
    document = {
        **body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(root.sign(signed)).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        },
    }
    root_pem = root.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return (
        validate_campaign_trust_policy(
            document,
            trusted_root_public_key_pem=root_pem,
            now_unix=now,
        ),
        role_keys,
        root_pem,
    )


def test_majority_output_uses_parsed_answer_without_gold_access():
    first = 'reasoning\nFINAL_ANSWER: {"count":2,"witness":[1,5]}'
    second = 'different\nFINAL_ANSWER: {"witness":[1,5],"count":2}'
    wrong = 'FINAL_ANSWER: {"count":9,"witness":[]}'

    assert runner._majority_output([first, wrong, second]) == first


def test_worker_reconstructs_public_tasks_without_answer_payloads():
    tasks = generate_task_battery([7], domains=("mathematics",), difficulty=1)
    plan = build_campaign_plan(
        "public-worker-test",
        tasks,
        model_identity={"model": "sealed"},
        adapter_identity={"adapter": "sealed"},
        execution_config={"difficulty": 1},
    )

    public_tasks = runner._public_tasks_from_plan(plan)
    rebuilt = build_campaign_plan(
        "public-worker-test",
        public_tasks,
        model_identity={"model": "sealed"},
        adapter_identity={"adapter": "sealed"},
        execution_config={"difficulty": 1},
    )

    assert rebuilt.to_dict() == plan.to_dict()
    assert all(not hasattr(task, "blinded_answer") for task in public_tasks)
    assert all(not hasattr(task, "score") for task in public_tasks)


def test_expected_worker_plan_never_invokes_answer_bearing_generator(
    monkeypatch: pytest.MonkeyPatch,
):
    tasks = generate_task_battery([7], domains=("mathematics",), difficulty=1)
    model_identity = {"model": "sealed"}
    adapter_identity = {"adapter": "sealed"}
    execution_config = {"difficulty": 1}
    plan = build_campaign_plan(
        "generator-isolation-test",
        tasks,
        model_identity=model_identity,
        adapter_identity=adapter_identity,
        execution_config=execution_config,
    )
    args = SimpleNamespace(campaign_name="generator-isolation-test")
    monkeypatch.setattr(
        runner,
        "generate_task_battery",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("answer-bearing generator entered worker path")
        ),
    )
    monkeypatch.setattr(
        runner, "_identity_material", lambda _args: (model_identity, adapter_identity)
    )
    monkeypatch.setattr(runner, "_contamination_audit", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "_execution_config", lambda *_args: execution_config)
    monkeypatch.setattr(runner, "_verified_campaign_trust", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_claim_eligible", lambda *_args: False)
    monkeypatch.setattr(runner, "_arms", lambda _args: runner.FULL_ARMS)

    rebuilt, public_tasks = runner._expected_worker_plan(args, plan)

    assert rebuilt.to_dict() == plan.to_dict()
    assert all(not hasattr(task, "blinded_answer") for task in public_tasks)


def test_outputs_are_sealed_before_answer_reveal_and_scoring(tmp_path: Path):
    tasks = generate_task_battery([7], domains=("mathematics",), difficulty=1)
    plan = build_campaign_plan(
        "two-phase-preflight",
        tasks,
        model_identity={"model": "sealed"},
        adapter_identity={"adapter": "sealed"},
        execution_config={
            "difficulty": 1,
            "answer_reveal_protocol": "sealed_outputs_then_issuer_reveal_v1",
        },
    )
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    runner._persist_plan(campaign_dir, plan)
    task_by_id = {task.task_id: task for task in tasks}
    with runner.CampaignJournal(campaign_dir / runner.JOURNAL_FILE, plan) as journal:
        for cell_id in plan.cell_ids:
            definition = plan.cell_definition(cell_id)
            task = task_by_id[definition["task_id"]]
            text = "FINAL_ANSWER: " + json.dumps(
                task.reveal_for_verifier()["expected"],
                sort_keys=True,
                separators=(",", ":"),
            )
            attempt_id = journal.start_cell(cell_id)
            journal.record_arm_result(
                cell_id,
                attempt_id,
                {
                    "arm": definition["arm"],
                    "text": text,
                    "layer_apps": 1,
                },
            )
        assert journal.resume().committed_cell_ids == ()
        assert journal.resume().sealed_cell_ids == plan.cell_ids

    assert all(runner._arm_outputs_sealed(campaign_dir, plan, arm) for arm in runner.FULL_ARMS)
    sealed = runner._seal_output_manifest(campaign_dir, plan)
    reveal = runner._admit_answer_reveal(
        SimpleNamespace(
            campaign_dir=str(campaign_dir),
            answer_reveal_attestation="",
        ),
        plan,
        tasks,
        sealed,
    )
    assert reveal is not None
    assert (campaign_dir / runner.SEALED_OUTPUT_MANIFEST_FILE).exists()
    assert (campaign_dir / runner.ANSWER_REVEAL_FILE).exists()
    assert not (campaign_dir / runner.ANSWER_REVEAL_REQUEST_FILE).exists()
    with runner.CampaignJournal(campaign_dir / runner.JOURNAL_FILE, plan) as journal:
        assert journal.resume().committed_cell_ids == ()

    runner._score_sealed_outputs(campaign_dir, plan, tasks)
    with runner.CampaignJournal(campaign_dir / runner.JOURNAL_FILE, plan) as journal:
        assert journal.resume().committed_cell_ids == plan.cell_ids
        assert all(
            record["verification"]["correct"] is True for record in journal.committed_records()
        )


def test_claim_reveal_pauses_for_exact_external_issuer_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    now = int(time.time())
    campaign_name = "signed-reveal-test"
    policy, role_keys, _root_pem = _external_policy_fixture(campaign_name, now)
    tasks = generate_task_battery([9], domains=("mathematics",), difficulty=1)
    manifest = runner.build_task_manifest(tasks)
    auditor = Ed25519PrivateKey.generate()
    auditor_der = auditor.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    audit_body = {
        "schema": runner.CONTAMINATION_AUDIT_SCHEMA,
        "training_dataset_identity_sha256": "d" * 64,
        "task_manifest_sha256": manifest.manifest_sha256,
        "status": "passed_zero_overlap",
        "overlap_count": 0,
        "auditor_independence": "external",
        "corpora": [{"name": "training", "snapshot_sha256": "d" * 64}],
        "methods": ["exact_prompt", "normalized_prompt", "token_fivegram"],
    }
    audit_bytes = canonical_json_bytes(audit_body)
    audit_sha = hashlib.sha256(auditor_der).hexdigest()
    audit = {
        **audit_body,
        "signature": {
            "algorithm": "ed25519",
            "key_id": audit_sha,
            "signature_b64": base64.b64encode(auditor.sign(audit_bytes)).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(audit_bytes).hexdigest(),
            "public_key_der_b64": base64.b64encode(auditor_der).decode("ascii"),
            "trust_root_sha256": audit_sha,
            "verified": True,
        },
    }
    model_identity = {"model": "sealed"}
    adapter_identity = {"adapter": "sealed"}
    execution_config = {
        "difficulty": 1,
        "worker_task_material": "public_manifest_only",
        "answer_reveal_protocol": "sealed_outputs_then_issuer_reveal_v1",
        "worker_origin_protocol": "detached_supervisor_staged_arm_import_v3",
        "worker_origin_attempt_slots": 3,
        "generation_seed_count": 1,
        "generation_seed_min_entropy_bits": 60,
        "generation_seed_policy": "external_issuer_uniform_63bit",
        "generation_seed_disclosure": "post_seal_answer_reveal",
    }
    unsigned = build_campaign_plan(
        campaign_name,
        tasks,
        model_identity=model_identity,
        adapter_identity=adapter_identity,
        execution_config=execution_config,
        contamination_audit=audit,
        claim_eligible=False,
    )
    trust = {
        "prelaunch_verified": True,
        "externally_custodied": True,
        "policy_sha256": policy.policy_sha256,
        "unsigned_plan_sha256": unsigned.plan_sha256,
    }
    plan = _synthetic_claim_plan_for_nonstatistical_contract(
        unsigned,
        campaign_trust=trust,
    )
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    runner._persist_plan(campaign_dir, plan)
    with runner.CampaignJournal(campaign_dir / runner.JOURNAL_FILE, plan) as journal:
        for cell_id in plan.cell_ids:
            definition = plan.cell_definition(cell_id)
            attempt = journal.start_cell(cell_id)
            journal.record_arm_result(
                cell_id,
                attempt,
                {"arm": definition["arm"], "text": "candidate", "layer_apps": 1},
            )
    worker_execution = {
        "manifest_sha256": "a" * 64,
        "detached_plan_sha256": "b" * 64,
        "detached_classification_head_sha256": "c" * 64,
        "detached_classifications_sha256": "d" * 64,
        "imports_sha256": "e" * 64,
        "excluded_attempts_sha256": "f" * 64,
    }
    (campaign_dir / runner.WORKER_EXECUTION_MANIFEST_FILE).write_bytes(
        canonical_json_bytes(worker_execution) + b"\n"
    )
    sealed = runner._seal_output_manifest(
        campaign_dir,
        plan,
        worker_execution=worker_execution,
    )
    monkeypatch.setattr(runner, "_load_campaign_trust_policy", lambda *_args, **_kwargs: policy)
    args = SimpleNamespace(
        campaign_dir=str(campaign_dir),
        answer_reveal_attestation="",
    )

    assert runner._admit_answer_reveal(args, plan, tasks, sealed) is None
    request = json.loads((campaign_dir / runner.ANSWER_REVEAL_REQUEST_FILE).read_bytes())
    attestation = build_role_attestation(
        policy,
        role=TASK_ISSUER,
        payload=request["signed_payload"]["payload"],
        signed_at_unix=request["signed_payload"]["signed_at_unix"],
        private_key=role_keys[TASK_ISSUER],
        operation="answer_reveal",
        purpose=f"{campaign_name}:answer-reveal",
    )
    attestation_path = tmp_path / "answer-reveal-attestation.json"
    attestation_path.write_bytes(canonical_json_bytes(attestation) + b"\n")
    args.answer_reveal_attestation = str(attestation_path)

    reveal = runner._admit_answer_reveal(args, plan, tasks, sealed)
    assert reveal is not None
    assert reveal["request_sha256"] == request["request_sha256"]
    assert reveal["task_issuer_attestation"] == attestation

    runner._score_sealed_outputs(campaign_dir, plan, tasks)
    with runner.CampaignJournal(campaign_dir / runner.JOURNAL_FILE, plan) as journal:
        campaign_manifest = journal.finalize(campaign_dir / runner.MANIFEST_FILE)
    args.final_run_attestation = ""
    grade = {"grade_sha256": "1" * 64}
    assert (
        runner._admit_final_run_envelope(
            args,
            plan,
            sealed_outputs=sealed,
            answer_reveal=reveal,
            campaign_manifest=campaign_manifest,
            grade=grade,
            worker_execution=worker_execution,
        )
        is None
    )
    final_request = json.loads((campaign_dir / runner.FINAL_RUN_REQUEST_FILE).read_bytes())
    assert (
        final_request["signed_payload"]["payload"]["worker_execution_manifest_sha256"]
        == worker_execution["manifest_sha256"]
    )
    final_attestation = build_role_attestation(
        policy,
        role=CAMPAIGN_RUNNER,
        payload=final_request["signed_payload"]["payload"],
        signed_at_unix=final_request["signed_payload"]["signed_at_unix"],
        private_key=role_keys[CAMPAIGN_RUNNER],
        operation="final_run",
        purpose=f"{campaign_name}:final-run",
    )
    final_attestation_path = tmp_path / "final-run-attestation.json"
    final_attestation_path.write_bytes(canonical_json_bytes(final_attestation) + b"\n")
    args.final_run_attestation = str(final_attestation_path)

    final_envelope = runner._admit_final_run_envelope(
        args,
        plan,
        sealed_outputs=sealed,
        answer_reveal=reveal,
        campaign_manifest=campaign_manifest,
        grade=grade,
        worker_execution=worker_execution,
    )
    assert final_envelope is not None
    assert final_envelope["request_sha256"] == final_request["request_sha256"]
    assert final_envelope["campaign_runner_attestation"] == final_attestation
    assert (
        final_envelope["payload"]["detached_classifications_sha256"]
        == worker_execution["detached_classifications_sha256"]
    )
    with pytest.raises(
        runner.CampaignProducerError,
        match="campaign_runner request differs from the current payload",
    ):
        runner._admit_final_run_envelope(
            args,
            plan,
            sealed_outputs=sealed,
            answer_reveal=reveal,
            campaign_manifest=campaign_manifest,
            grade={"grade_sha256": "2" * 64},
            worker_execution=worker_execution,
        )


def test_atomic_plan_artifact_is_create_or_exact_verify(tmp_path: Path):
    path = tmp_path / "plan.json"
    payload = b'{"plan":1}\n'

    runner._atomic_create_or_verify(path, payload)
    runner._atomic_create_or_verify(path, payload)

    assert path.read_bytes() == payload
    with pytest.raises(runner.CampaignProducerError, match="existing artifact differs"):
        runner._atomic_create_or_verify(path, b'{"plan":2}\n')


def test_atomic_artifact_rejects_concurrent_symlink_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "reveal.json"
    victim = tmp_path / "victim.json"
    victim.write_bytes(b'{"trusted":true}\n')

    def racing_link(
        _source: os.PathLike[str] | str,
        target: os.PathLike[str] | str,
        *,
        follow_symlinks: bool,
    ) -> None:
        assert follow_symlinks is False
        Path(target).symlink_to(victim)
        raise FileExistsError

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(runner.CampaignProducerError, match="symlink artifact rejected"):
        runner._atomic_create_or_verify(destination, victim.read_bytes())

    assert victim.read_bytes() == b'{"trusted":true}\n'
    assert destination.is_symlink()
    assert not list(tmp_path.glob(".*.tmp"))


def test_implementation_identity_covers_complete_latent_cortex_source():
    observed = runner._implementation_sha256()
    latent_root = runner.REPO_ROOT / "core/brain/llm/latent_cortex"
    expected = {str(path.relative_to(runner.REPO_ROOT)) for path in latent_root.glob("*.py")}

    assert expected.issubset(observed)
    assert "core/brain/llm/latent_cortex/fast_weights.py" in observed
    assert "tools/run_latent_cortex_paired_campaign.py" in observed
    assert all(len(digest) == 64 for digest in observed.values())


def test_fresh_checkpoint_identity_detects_same_size_weight_replacement(
    tmp_path: Path,
):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"original-weight-bytes")
    first = runner._fresh_checkpoint_file_fingerprint(tmp_path)

    shard.write_bytes(b"replaced-weight-bytes")
    second = runner._fresh_checkpoint_file_fingerprint(tmp_path)

    assert first["method"] == second["method"] == "sha256"
    assert first["files"] == second["files"] == 1
    assert first["fingerprint"] != second["fingerprint"]


def test_worker_command_resolves_relative_campaign_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(
        campaign_dir="relative-campaign",
        campaign_name="test",
        model="model",
        adapter="adapter",
        adapter_id="adapter-id",
        seeds="1",
        domains="mathematics",
        difficulty=2,
        profile="full",
        n_slots=4,
        branches=2,
        rlc_steps=2,
        rlc_profile="resident_full_stack",
        decode_max_tokens=64,
        episode_timeout=10.0,
        load_timeout=10.0,
        warmup_timeout=10.0,
        arm_timeout=20.0,
        campaign_timeout=30.0,
        equal_compute_max_samples=2,
        max_infra_attempts=1,
        confirmatory=False,
        contamination_audit="",
        contamination_trust_root="",
    )

    command = runner._worker_args(args, runner.BASE_RLC)

    campaign_index = command.index("--campaign-dir") + 1
    assert command[campaign_index] == str((tmp_path / "relative-campaign").resolve())
    assert "--seeds" not in command
    assert command[command.index("--seed-count") + 1] == "1"

    policy = runner._detached_broker_policy(args)
    assert len(policy) == len(runner.FULL_ARMS)
    assert {entry["command"][-1] for entry in policy} == set(runner.FULL_ARMS)
    assert all(entry["cwd"] == str(runner.REPO_ROOT) for entry in policy)
    assert all(entry["max_invocations"] == 1 for entry in policy)
    assert all(all(argument for argument in entry["command"]) for entry in policy)
    assert all("--sequential-look-observations" not in entry["command"] for entry in policy)

    args.sequential_look_observations = "1"
    args.sequential_alpha_weights = "1/1"
    sequential = runner._worker_args(args, runner.BASE_RLC, worker_look=1)
    assert sequential[sequential.index("--sequential-look-observations") + 1] == "1"
    assert sequential[sequential.index("--sequential-alpha-weights") + 1] == "1/1"
    args.sequential_alpha_weights = ""
    with pytest.raises(runner.CampaignProducerError, match="sequential contract is incomplete"):
        runner._worker_args(args, runner.BASE_RLC, worker_look=1)


def _sequential_runner_plan():
    tasks = generate_task_battery(
        [11, 22, 33, 44],
        domains=("mathematics", "coding"),
        difficulty=1,
    )
    power = runner.exact_group_sequential_power_plan(
        domain_count=2,
        comparison_count=4,
        arm_count=len(runner.PRIMARY_ARMS),
        look_observations_per_domain=(2, 4),
        alpha_weights=(runner.Rational(1, 10), runner.Rational(9, 10)),
    )
    return build_campaign_plan(
        "sequential-runner-contract",
        tasks,
        model_identity={"model": "test"},
        adapter_identity={"adapter": "test"},
        execution_config={
            "sequential_look_observations_per_domain": [2, 4],
            "sequential_alpha_weights": [
                {"numerator": 1, "denominator": 10},
                {"numerator": 9, "denominator": 10},
            ],
            "exact_statistical_power": power,
        },
        arms=runner.PRIMARY_ARMS,
    )


def test_sequential_worker_batches_are_balanced_disjoint_and_cumulative():
    plan = _sequential_runner_plan()
    assignments = runner._task_look_assignments(plan)
    task_domains = {
        task["task_id"]: task["domain"]
        for task in plan.to_dict()["metadata"]["task_manifest"]["tasks"]
    }

    for worker_look in (1, 2):
        assigned = [task_id for task_id, look in assignments.items() if look == worker_look]
        assert len(assigned) == 4
        assert {
            domain: sum(task_domains[item] == domain for item in assigned)
            for domain in task_domains.values()
        } == {
            "coding": 2,
            "mathematics": 2,
        }
    first = runner._arm_cell_ids_for_look(
        plan,
        runner.BASE_RLC,
        1,
        cumulative=False,
    )
    second = runner._arm_cell_ids_for_look(
        plan,
        runner.BASE_RLC,
        2,
        cumulative=False,
    )
    cumulative = runner._arm_cell_ids_for_look(
        plan,
        runner.BASE_RLC,
        2,
        cumulative=True,
    )
    assert len(first) == len(second) == 4
    assert first.isdisjoint(second)
    assert cumulative == first | second


def test_sequential_resume_excludes_imported_and_future_cells():
    plan = _sequential_runner_plan()
    first = runner._arm_cell_ids_for_look(
        plan,
        runner.BASE_RLC,
        1,
        cumulative=False,
    )
    second = runner._arm_cell_ids_for_look(
        plan,
        runner.BASE_RLC,
        2,
        cumulative=False,
    )
    already_imported = {next(iter(first))}
    pending = runner._pending_worker_cell_ids(
        plan,
        arm=runner.BASE_RLC,
        worker_look=1,
        runnable_cell_ids=plan.cell_ids,
        stage_sealed_cell_ids=set(),
        canonical_sealed_cell_ids=already_imported,
    )

    assert set(pending) == first - already_imported
    assert set(pending).isdisjoint(second)
    assert pending == sorted(
        pending,
        key=lambda cell_id: plan.cell_definition(cell_id)["execution_ordinal_within_arm"],
    )


def test_sequential_attempt_slots_are_disjoint_per_look():
    args = SimpleNamespace(max_infra_attempts=3)
    assert tuple(runner._worker_attempt_slot_range(args, 1)) == (1, 2, 3)
    assert tuple(runner._worker_attempt_slot_range(args, 2)) == (4, 5, 6)


def test_sequential_look_evaluation_publishes_independent_chained_certificate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan = _sequential_runner_plan()
    task_ids = runner.cumulative_task_ids(plan, 1)
    records = []
    for cell_id in plan.cell_ids:
        definition = plan.cell_definition(cell_id)
        if definition["task_id"] in task_ids:
            records.append(
                {
                    "cell_id": cell_id,
                    "definition": definition,
                    "commit": {
                        "result_sha256": "a" * 64,
                        "verification_sha256": "b" * 64,
                    },
                }
            )
    grade = {
        "plan_sha256": plan.plan_sha256,
        "expected_task_count": len(task_ids),
        "expected_cell_count": len(records),
        "statistical_policy": {"alpha": {"numerator": 1, "denominator": 200}},
        "verdict": "inconclusive",
    }

    class FakeJournal:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def committed_records(self):
            return tuple(records)

    monkeypatch.setattr(runner, "CampaignJournal", FakeJournal)
    monkeypatch.setattr(runner, "_score_sealed_outputs", lambda *_args: None)
    monkeypatch.setattr(runner, "grade_campaign", lambda *_args, **_kwargs: grade)
    monkeypatch.setattr(
        runner,
        "independent_grade_campaign",
        lambda *_args, **_kwargs: {"semantic_grade": grade},
    )
    args = SimpleNamespace(
        campaign_dir=str(tmp_path),
        contamination_audit="",
        contamination_trust_root="",
    )

    certificate = runner._evaluate_sequential_look(
        args,
        plan,
        (),
        worker_look=1,
    )

    assert certificate is not None
    assert certificate["decision"] == "continue"
    path = tmp_path / runner.SEQUENTIAL_LOOK_DIR / "look-001.json"
    assert json.loads(path.read_text()) == certificate


def _write_sequential_final_chain(campaign_dir: Path, plan) -> list[dict]:
    look_dir = campaign_dir / runner.SEQUENTIAL_LOOK_DIR
    look_dir.mkdir()
    certificates = []
    previous_sha256 = None
    power_looks = plan.to_dict()["metadata"]["execution_config"]["exact_statistical_power"]["looks"]
    for look, power in enumerate(power_looks, 1):
        task_ids = runner.cumulative_task_ids(plan, look)
        records = []
        for cell_id in plan.cell_ids:
            definition = plan.cell_definition(cell_id)
            if definition["task_id"] in task_ids:
                records.append(
                    {
                        "cell_id": cell_id,
                        "definition": definition,
                        "commit": {
                            "result_sha256": "a" * 64,
                            "verification_sha256": "b" * 64,
                        },
                    }
                )
        grade = {
            "plan_sha256": plan.plan_sha256,
            "expected_task_count": len(task_ids),
            "expected_cell_count": len(records),
            "statistical_policy": {"alpha": power["family_alpha"]},
            "verdict": "inconclusive",
        }
        certificate = runner.build_sequential_look_certificate(
            plan=plan,
            look=look,
            committed_records=records,
            production_grade=grade,
            independent_grade=grade,
            previous_certificate_sha256=previous_sha256,
        )
        (look_dir / f"look-{look:03d}.json").write_bytes(canonical_json_bytes(certificate) + b"\n")
        certificates.append(certificate)
        previous_sha256 = certificate["certificate_sha256"]
    return certificates


def test_sequential_final_binding_commits_complete_chain(tmp_path: Path):
    plan = _sequential_runner_plan()
    certificates = _write_sequential_final_chain(tmp_path, plan)

    binding = runner._sequential_final_evidence_binding(tmp_path, plan)

    assert binding == {
        "sequential_look_count": 2,
        "sequential_certificate_head_sha256": certificates[-1]["certificate_sha256"],
        "sequential_certificate_chain_sha256": hashlib.sha256(
            canonical_json_bytes(
                [certificate["certificate_sha256"] for certificate in certificates]
            )
        ).hexdigest(),
        "sequential_first_boundary_look": None,
        "sequential_first_boundary_decision": None,
        "sequential_terminal_decision": "terminal_inconclusive",
    }


def test_sequential_final_binding_rejects_rehashed_broken_link(tmp_path: Path):
    plan = _sequential_runner_plan()
    _write_sequential_final_chain(tmp_path, plan)
    path = tmp_path / runner.SEQUENTIAL_LOOK_DIR / "look-002.json"
    certificate = json.loads(path.read_bytes())
    material = dict(certificate)
    material.pop("certificate_sha256")
    material["previous_certificate_sha256"] = "0" * 64
    certificate = {
        **material,
        "certificate_sha256": hashlib.sha256(canonical_json_bytes(material)).hexdigest(),
    }
    path.write_bytes(canonical_json_bytes(certificate) + b"\n")

    with pytest.raises(
        runner.CampaignProducerError,
        match="cannot bind the final run",
    ):
        runner._sequential_final_evidence_binding(tmp_path, plan)


def test_claim_broker_policy_covers_every_exact_worker_attempt_command(
    tmp_path: Path,
):
    args, _plan, _policy, _role_keys = _worker_origin_claim_fixture(tmp_path)
    policy = runner._detached_broker_policy(args, _plan)

    expected = {
        tuple(runner._worker_args(args, arm, worker_attempt_slot=attempt_slot))
        for arm in runner.PRIMARY_ARMS
        for attempt_slot in range(1, args.max_infra_attempts + 1)
    }
    observed = {tuple(entry["command"]) for entry in policy}
    assert observed == expected
    assert len(policy) == len(expected)
    assert all(entry["max_invocations"] == 1 for entry in policy)
    assert all(
        forbidden not in entry["command"]
        for entry in policy
        for forbidden in (
            "--worker-private-key",
            "--worker-authorization",
            "--worker-boot-id",
        )
    )
    for entry in policy:
        contract = entry["worker_origin"]
        arm = entry["command"][entry["command"].index("--worker-arm") + 1]
        slot = int(entry["command"][entry["command"].index("--worker-attempt-slot") + 1])
        assert contract["arm"] == arm
        assert contract["worker_attempt_slot"] == slot
        assert contract["allowed_cells"] == [
            {
                "cell_id": cell_id,
                "cell_type": runner.PAIRED_CAMPAIGN_CELL_TYPE,
            }
            for cell_id in _plan.cell_ids
            if _plan.cell_definition(cell_id)["arm"] == arm
        ]


def test_run_child_uses_detached_broker_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    args = SimpleNamespace(
        campaign_dir=str(tmp_path),
        campaign_name="test",
        model="model",
        adapter="adapter",
        adapter_id="adapter-id",
        seeds="1",
        domains="mathematics",
        difficulty=2,
        profile="primary",
        n_slots=4,
        branches=2,
        rlc_steps=2,
        rlc_profile="resident_full_stack",
        decode_max_tokens=64,
        episode_timeout=10.0,
        load_timeout=10.0,
        warmup_timeout=10.0,
        arm_timeout=20.0,
        campaign_timeout=30.0,
        equal_compute_max_samples=2,
        max_infra_attempts=1,
        confirmatory=False,
        contamination_audit="",
        contamination_trust_root="",
    )
    observed: dict[str, object] = {}

    def fake_broker(command, *, cwd, stdout_path, timeout_s):
        observed.update(
            command=command,
            cwd=cwd,
            stdout_path=stdout_path,
            timeout_s=timeout_s,
        )
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(runner, "broker_available", lambda: True)
    monkeypatch.setattr(runner, "run_brokered_process", fake_broker)

    outcome = runner._run_child(args, runner.BASE_RLC, 12.5)
    assert outcome.returncode == 17
    assert observed["command"] == runner._worker_args(args, runner.BASE_RLC)
    assert observed["cwd"] == runner.REPO_ROOT
    assert observed["stdout_path"] == tmp_path / runner.LOG_FILE
    assert observed["timeout_s"] == pytest.approx(12.5, abs=0.01)


def _worker_origin_claim_fixture(tmp_path: Path):
    now = int(time.time())
    campaign_name = "worker-origin-claim-test"
    policy, role_keys, root_pem = _external_policy_fixture(campaign_name, now)
    tasks = generate_task_battery([7], domains=("mathematics",), difficulty=1)
    task_manifest = runner.build_task_manifest(tasks)
    auditor = Ed25519PrivateKey.generate()
    auditor_der = auditor.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    audit_body = {
        "schema": runner.CONTAMINATION_AUDIT_SCHEMA,
        "training_dataset_identity_sha256": "d" * 64,
        "task_manifest_sha256": task_manifest.manifest_sha256,
        "status": "passed_zero_overlap",
        "overlap_count": 0,
        "auditor_independence": "external",
        "corpora": [{"name": "training", "snapshot_sha256": "d" * 64}],
        "methods": ["exact_prompt", "normalized_prompt", "token_fivegram"],
    }
    audit_bytes = canonical_json_bytes(audit_body)
    audit = {
        **audit_body,
        "signature": {
            "algorithm": "ed25519",
            "key_id": hashlib.sha256(auditor_der).hexdigest(),
            "signature_b64": base64.b64encode(auditor.sign(audit_bytes)).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(audit_bytes).hexdigest(),
            "public_key_der_b64": base64.b64encode(auditor_der).decode("ascii"),
            "trust_root_sha256": hashlib.sha256(auditor_der).hexdigest(),
            "verified": True,
        },
    }
    execution_config = {
        "difficulty": 1,
        "task_registry_version": runner.REGISTRY_VERSION,
        "generation_seed_count": 1,
        "generation_seed_min_entropy_bits": 61,
        "generation_seed_policy": "external_issuer_uniform_63bit",
        "generation_seed_disclosure": "post_seal_answer_reveal",
        "domains": ["mathematics"],
        "worker_task_material": "public_manifest_only",
        "answer_reveal_protocol": "sealed_outputs_then_issuer_reveal_v1",
        "worker_origin_protocol": "detached_supervisor_staged_arm_import_v3",
        "worker_origin_attempt_slots": 2,
        "implementation_sha256": runner._implementation_sha256(),
    }
    unsigned = build_campaign_plan(
        campaign_name,
        tasks,
        model_identity={"model": "sealed"},
        adapter_identity={"adapter": "sealed"},
        execution_config=execution_config,
        contamination_audit=audit,
        claim_eligible=False,
        arms=runner.PRIMARY_ARMS,
    )
    plan = _synthetic_claim_plan_for_nonstatistical_contract(
        unsigned,
        campaign_trust={
            "prelaunch_verified": True,
            "externally_custodied": True,
            "policy_sha256": policy.policy_sha256,
            "unsigned_plan_sha256": unsigned.plan_sha256,
        },
    )
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    runner._persist_plan(campaign_dir, plan)
    args = SimpleNamespace(
        campaign_dir=str(campaign_dir),
        campaign_name=campaign_name,
        model="/sealed/model",
        adapter="/sealed/adapter",
        adapter_id="adapter-id",
        personality_adapter="none",
        seeds=str(1 << 60),
        seed_values=(1 << 60,),
        seed_count=0,
        seed_entropy_bits=0,
        domains="mathematics",
        domain_values=("mathematics",),
        difficulty=1,
        task_registry_version=runner.REGISTRY_VERSION,
        profile="primary",
        n_slots=4,
        branches=2,
        rlc_steps=2,
        rlc_profile="resident_full_stack",
        decode_max_tokens=64,
        episode_timeout=10.0,
        load_timeout=10.0,
        warmup_timeout=10.0,
        arm_timeout=20.0,
        campaign_timeout=30.0,
        equal_compute_max_samples=2,
        max_infra_attempts=2,
        confirmatory=True,
        contamination_audit="",
        contamination_trust_root="",
        campaign_trust_policy=str(tmp_path / "policy.json"),
        campaign_trust_root=str(tmp_path / "root.pem"),
        task_issuer_attestation="/external/issuer.json",
        runner_attestation="/external/runner.json",
        worker_arm="",
    )
    Path(args.campaign_trust_policy).write_bytes(canonical_json_bytes(policy.document) + b"\n")
    Path(args.campaign_trust_root).write_bytes(root_pem)
    Path(args.campaign_trust_policy).chmod(0o600)
    Path(args.campaign_trust_root).chmod(0o600)
    return args, plan, policy, role_keys


def test_claim_worker_accepts_only_deterministic_stage_and_inherited_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, plan, _policy, _role_keys = _worker_origin_claim_fixture(tmp_path)
    arm = runner.PRIMARY_ARMS[0]
    paths = runner._worker_attempt_paths(Path(args.campaign_dir), arm, 1)
    client = SimpleNamespace(session_id="1" * 32, close=lambda: None)
    monkeypatch.setattr(
        runner.DetachedWorkerOriginChannelClient,
        "from_environment",
        lambda: client,
    )
    args.worker_arm = arm
    args.worker_attempt_slot = 1
    args.worker_stage_journal = str(paths["stage"])

    context = runner._worker_origin_context(args, plan)

    assert context == {"client": client, "paths": paths}
    assert not any("private" in key or "authorization" in key for key in paths)

    args.worker_stage_journal = str(tmp_path / "substituted.jsonl")
    with pytest.raises(runner.CampaignProducerError, match="path substitution"):
        runner._worker_origin_context(args, plan)

    args.worker_stage_journal = str(paths["stage"])
    paths["stage"].write_text("consumed\n", encoding="utf-8")
    with pytest.raises(runner.CampaignProducerError, match="already consumed"):
        runner._worker_origin_context(args, plan)


def test_claim_worker_waits_for_external_authorization_without_consuming_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _plan, _policy, _role_keys = _worker_origin_claim_fixture(tmp_path)
    attempts = 0
    accepted = SimpleNamespace(returncode=0)

    def broker(command, **_kwargs):
        nonlocal attempts
        attempts += 1
        assert "--worker-private-key" not in command
        if attempts < 3:
            raise runner.DetachedBrokerError(
                "worker-origin external authorization required at /tmp/attestation.json"
            )
        return accepted

    monkeypatch.setattr(runner, "broker_available", lambda: True)
    monkeypatch.setattr(runner, "run_brokered_process", broker)
    patch_module_clock(monkeypatch, runner, sleep=lambda _seconds: None)

    outcome = runner._run_child(
        args,
        runner.PRIMARY_ARMS[0],
        2.0,
        worker_attempt_slot=1,
    )

    assert outcome is accepted
    assert attempts == 3
    assert (
        runner._next_worker_attempt_slot(
            Path(args.campaign_dir),
            runner.PRIMARY_ARMS[0],
            maximum=args.max_infra_attempts,
        )
        == 1
    )


def test_worker_attempt_slot_reuses_pending_authorization_and_skips_terminal_work(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    arm = runner.PRIMARY_ARMS[0]
    first = runner._worker_attempt_paths(campaign_dir, arm, 1)
    first["origin_dir"].mkdir()
    (first["origin_dir"] / "pending.request.json").write_text("{}\n", encoding="utf-8")

    assert runner._next_worker_attempt_slot(campaign_dir, arm, maximum=2) == 1

    first["broker_result"].write_text("{}\n", encoding="utf-8")
    assert runner._next_worker_attempt_slot(campaign_dir, arm, maximum=2) == 2

    second = runner._worker_attempt_paths(campaign_dir, arm, 2)
    second["stage"].write_text("partial\n", encoding="utf-8")
    assert runner._next_worker_attempt_slot(campaign_dir, arm, maximum=2) is None


def _broker_result_fixture(
    *,
    index: int,
    status: str = "passed",
) -> runner.BrokeredProcessResult:
    passed = status == "passed"

    def digest(suffix: str) -> str:
        return hashlib.sha256(f"{index}:{suffix}".encode()).hexdigest()

    return runner.BrokeredProcessResult(
        returncode=0 if passed else 17,
        request_id=digest("request")[:32],
        policy_sha256=digest("policy"),
        worker_pid=3000 + index,
        worker_process_group_id=3000 + index,
        worker_start_token=f"worker-{index}",
        started_at=float(index),
        finished_at=float(index + 1),
        duration_s=1.0,
        timed_out=False,
        containment_verified=True,
        status=status,
        error=None if passed else "worker exited 17",
        worker_origin_lifecycle={
            "artifact_path": f"/detached/lifecycle-{index}.json",
            "artifact_sha256": digest("lifecycle-artifact"),
            "event_type": "terminal" if passed else "abandoned",
            "event_sha256": digest("lifecycle-event"),
            "result_count": 1 if passed else 0,
            "session_id": digest("session")[:32],
        },
        receipt_sha256=digest("receipt"),
        response_hmac_sha256=digest("response-hmac"),
    )


def test_brokered_worker_import_requires_passed_terminal_detached_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, plan, policy, _role_keys = _worker_origin_claim_fixture(tmp_path)
    campaign_dir = Path(args.campaign_dir)
    failed_arm, passed_arm = runner.PRIMARY_ARMS[:2]
    failed = _broker_result_fixture(index=1, status="failed")

    assert (
        runner._import_brokered_worker_attempt(
            args,
            plan,
            arm=failed_arm,
            attempt_slot=1,
            result=failed,
        )
        is None
    )
    failed_paths = runner._worker_attempt_paths(campaign_dir, failed_arm, 1)
    assert failed_paths["broker_result"].is_file()
    assert not failed_paths["verified_stage"].exists()

    passed_paths = runner._worker_attempt_paths(campaign_dir, passed_arm, 1)
    passed_paths["origin_dir"].mkdir()
    lifecycle_path = passed_paths["origin_dir"] / "terminal.lifecycle.json"
    lifecycle = {"authorization_payload": {"detached_plan_sha256": "a" * 64}}
    lifecycle_path.write_bytes(canonical_json_bytes(lifecycle) + b"\n")
    base_passed = _broker_result_fixture(index=2)
    passed = runner.BrokeredProcessResult(
        **{
            **base_passed.__dict__,
            "worker_origin_lifecycle": {
                **base_passed.worker_origin_lifecycle,
                "artifact_path": str(lifecycle_path),
            },
        }
    )
    detached_evidence = VerifiedDetachedBrokerEvidence(
        plan={"plan_sha256": "a" * 64},
        journal_head_sha256="b" * 64,
        classification_head_sha256="c" * 64,
        attempt=1,
        terminal_event={},
        policy={},
        request={},
        terminal_summaries=(),
        quarantine_summaries=(),
    )
    verified_stage = SimpleNamespace(manifest={"manifest_sha256": "d" * 64})
    import_receipt = {"receipt_sha256": "e" * 64, "imported": []}
    monkeypatch.setattr(
        runner,
        "_verify_detached_worker_broker_result",
        lambda *_args, **_kwargs: detached_evidence,
    )
    monkeypatch.setattr(
        runner,
        "_load_campaign_trust_policy",
        lambda *_args, **_kwargs: policy,
    )
    monkeypatch.setattr(
        runner,
        "verify_terminal_worker_stage",
        lambda **_kwargs: verified_stage,
    )
    monkeypatch.setattr(
        runner,
        "import_verified_worker_stage",
        lambda **_kwargs: import_receipt,
    )

    observed = runner._import_brokered_worker_attempt(
        args,
        plan,
        arm=passed_arm,
        attempt_slot=1,
        result=passed,
    )

    assert observed == import_receipt
    assert passed_paths["verified_stage"].is_file()


def test_worker_execution_manifest_binds_imports_exclusions_and_detached_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, plan, policy, _role_keys = _worker_origin_claim_fixture(tmp_path)
    campaign_dir = Path(args.campaign_dir)
    detached_plan_sha256 = "a" * 64
    results: list[runner.BrokeredProcessResult] = []
    terminals: list[VerifiedDetachedTerminal] = []
    imported_by_position: dict[tuple[str, int], dict] = {}

    with runner.CampaignJournal(campaign_dir / runner.JOURNAL_FILE, plan) as journal:
        for arm_index, arm in enumerate(runner.PRIMARY_ARMS, start=1):
            cell_ids = [
                cell_id for cell_id in plan.cell_ids if plan.cell_definition(cell_id)["arm"] == arm
            ]
            assert len(cell_ids) == 1
            cell_id = cell_ids[0]
            origin_sha256 = hashlib.sha256(f"origin:{arm}".encode()).hexdigest()
            attempt_id = journal.start_cell(cell_id)
            journal.record_arm_result(
                cell_id,
                attempt_id,
                {
                    "arm": arm,
                    "text": "candidate",
                    "worker_origin": {"origin_sha256": origin_sha256},
                },
            )
            slots = (1, 2) if arm_index == 1 else (1,)
            for slot in slots:
                status = "failed" if arm_index == 1 and slot == 1 else "passed"
                result_index = len(results) + 1
                result = _broker_result_fixture(index=result_index, status=status)
                results.append(result)
                terminals.append(
                    VerifiedDetachedTerminal(
                        attempt=1,
                        request_id=result.request_id,
                        policy_sha256=result.policy_sha256,
                        session_id=result.worker_origin_lifecycle["session_id"],
                        event_sha256=hashlib.sha256(
                            f"terminal:{result_index}".encode()
                        ).hexdigest(),
                        receipt_sha256=result.receipt_sha256,
                        response_hmac_sha256=result.response_hmac_sha256,
                        status=status,
                        returncode=result.returncode,
                        claim_eligible=status == "passed",
                    )
                )
                paths = runner._worker_attempt_paths(campaign_dir, arm, slot)
                runner._persist_brokered_worker_result(paths, result)
                if status != "passed":
                    continue
                verified_stage = {
                    "manifest_sha256": hashlib.sha256(
                        f"manifest:{result_index}".encode()
                    ).hexdigest(),
                    "stage_sha256": hashlib.sha256(f"stage:{result_index}".encode()).hexdigest(),
                    "stage_journal_head_sha256": hashlib.sha256(
                        f"journal:{result_index}".encode()
                    ).hexdigest(),
                    "result_chain_head_sha256": hashlib.sha256(
                        f"chain:{result_index}".encode()
                    ).hexdigest(),
                    "cell_ids": cell_ids,
                    "detached_plan_sha256": detached_plan_sha256,
                }
                import_intent = {
                    "intent_sha256": hashlib.sha256(f"intent:{result_index}".encode()).hexdigest()
                }
                import_receipt = {
                    "receipt_sha256": hashlib.sha256(f"import:{result_index}".encode()).hexdigest(),
                    "imported": [{"result_origin_sha256": origin_sha256}],
                }
                for path, document in (
                    (paths["verified_stage"], verified_stage),
                    (paths["import_intent"], import_intent),
                    (paths["import_receipt"], import_receipt),
                ):
                    path.write_bytes(canonical_json_bytes(document) + b"\n")
                imported_by_position[(arm, slot)] = import_receipt

    evidence = VerifiedDetachedBrokerEvidence(
        plan={"plan_sha256": detached_plan_sha256},
        journal_head_sha256="b" * 64,
        classification_head_sha256="c" * 64,
        attempt=1,
        terminal_event={},
        policy={},
        request={},
        terminal_summaries=tuple(terminals),
        quarantine_summaries=(),
    )
    detached_run_dir = tmp_path / "detached-run"
    detached_run_dir.mkdir(mode=0o700)
    detached_plan_path = detached_run_dir / "detached_plan.json"
    detached_attempts_path = detached_run_dir / "detached_attempts.jsonl"
    detached_plan_path.write_bytes(canonical_json_bytes(evidence.plan) + b"\n")
    detached_attempts_path.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(
        runner,
        "_detached_evidence_environment",
        lambda: (
            detached_run_dir,
            detached_plan_path,
            detached_attempts_path,
            detached_plan_sha256,
            1,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_verify_detached_worker_broker_result",
        lambda *_args, **_kwargs: evidence,
    )
    monkeypatch.setattr(
        runner,
        "_import_brokered_worker_attempt",
        lambda _args, _plan, *, arm, attempt_slot, result: imported_by_position[
            (arm, attempt_slot)
        ],
    )
    monkeypatch.setattr(
        runner,
        "_load_campaign_trust_policy",
        lambda *_args, **_kwargs: policy,
    )

    manifest = runner._build_worker_execution_manifest(args, plan)

    assert manifest is not None
    assert manifest["import_count"] == len(runner.PRIMARY_ARMS)
    assert manifest["excluded_count"] == 1
    assert manifest["detached_plan_sha256"] == detached_plan_sha256
    assert manifest["detached_run_dir"] == str(detached_run_dir)
    assert manifest["detached_plan_path"] == str(detached_plan_path)
    assert manifest["detached_attempts_path"] == str(detached_attempts_path)
    assert (
        manifest["detached_plan_artifact_sha256"]
        == hashlib.sha256(detached_plan_path.read_bytes()).hexdigest()
    )
    assert manifest["detached_classification_head_sha256"] == "c" * 64
    assert manifest["detached_classifications"]["terminal_count"] == len(results)
    assert manifest["excluded_attempts"][0]["classification"] == "terminal_excluded"


def test_claim_eligibility_requires_full_powered_seven_domain_protocol():
    args = SimpleNamespace(
        confirmatory=True,
        seed_values=tuple((1 << 60) + value for value in range(20)),
        profile="full",
        domain_values=runner.FRONTIER_DOMAINS,
    )
    minimum_observations = runner._statistical_power_plan(args)["minimum_observations"]
    assert minimum_observations == 411
    args.seed_values = tuple((1 << 60) + value for value in range(minimum_observations))
    model_identity = {
        "runtime_bundle": {
            "model_type": "qwen2",
            "logical_parameter_count": 32_000_000_000,
            "logical_parameter_count_basis": "architecture_config_logical",
        }
    }
    adapter_identity = {"manifest": {"dataset_manifest": {"sha256": "d" * 64}}}
    audit = {"status": "passed_zero_overlap", "signature": {"verified": True}}
    trust = {"prelaunch_verified": True, "externally_custodied": True}
    assert runner._claim_eligible(args, model_identity, adapter_identity, audit, trust) is True

    args.domain_values = ("mathematics", "coding")
    assert runner._claim_eligible(args, model_identity, adapter_identity, audit, trust) is False
    args.domain_values = runner.FRONTIER_DOMAINS
    args.seed_values = tuple((1 << 60) + value for value in range(minimum_observations - 1))
    assert runner._claim_eligible(args, model_identity, adapter_identity, audit, trust) is False
    args.seed_values = tuple((1 << 60) + value for value in range(minimum_observations))
    assert runner._claim_eligible(args, model_identity, adapter_identity, {}, trust) is False
    assert runner._claim_eligible(args, model_identity, adapter_identity, audit, None) is False
    args.seed_values = tuple(range(minimum_observations))
    assert runner._claim_eligible(args, model_identity, adapter_identity, audit, trust) is False


def test_claim_eligibility_accepts_only_a_terminal_powered_sequential_plan():
    args = SimpleNamespace(
        confirmatory=True,
        seed_values=tuple((1 << 60) + value for value in range(640)),
        profile="full",
        domain_values=runner.FRONTIER_DOMAINS,
        sequential_look_values=(160, 320, 480, 640),
        sequential_alpha_weight_values=(
            runner.Rational(1, 100),
            runner.Rational(4, 100),
            runner.Rational(15, 100),
            runner.Rational(80, 100),
        ),
    )
    model_identity = {
        "runtime_bundle": {
            "model_type": "qwen2",
            "logical_parameter_count": 32_000_000_000,
            "logical_parameter_count_basis": "architecture_config_logical",
        }
    }
    adapter_identity = {"manifest": {"dataset_manifest": {"sha256": "d" * 64}}}
    audit = {"status": "passed_zero_overlap", "signature": {"verified": True}}
    trust = {"prelaunch_verified": True, "externally_custodied": True}

    power = runner._statistical_power_plan(args)
    assert power["terminal_look_powered_for_zero_loss_noninferiority"] is True
    assert power["looks"][0]["powered_for_zero_loss_noninferiority"] is False
    assert runner._claim_eligible(args, model_identity, adapter_identity, audit, trust) is True

    args.seed_values = args.seed_values[:400]
    args.sequential_look_values = (160, 320, 400)
    args.sequential_alpha_weight_values = (
        runner.Rational(1, 100),
        runner.Rational(4, 100),
        runner.Rational(95, 100),
    )
    assert runner._claim_eligible(args, model_identity, adapter_identity, audit, trust) is False


def test_contamination_audit_verifies_ed25519_external_trust_root(tmp_path: Path):
    tasks = generate_task_battery([7], domains=("mathematics",), difficulty=2)
    manifest = runner.build_task_manifest(tasks)
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    public_key = private_key.public_key()
    public_der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    key_id = hashlib.sha256(public_der).hexdigest()
    body = {
        "schema": runner.CONTAMINATION_AUDIT_SCHEMA,
        "training_dataset_identity_sha256": "f" * 64,
        "task_manifest_sha256": manifest.manifest_sha256,
        "status": "passed_zero_overlap",
        "overlap_count": 0,
        "auditor_independence": "external",
        "corpora": [{"name": "external-corpus", "snapshot_sha256": "f" * 64}],
        "methods": ["exact_prompt", "normalized_prompt", "token_fivegram"],
    }
    audit = {
        **body,
        "signature": {
            "algorithm": "ed25519",
            "key_id": key_id,
            "signature_b64": base64.b64encode(private_key.sign(canonical_json_bytes(body))).decode(
                "ascii"
            ),
        },
    }
    audit_path = tmp_path / "audit.json"
    trust_path = tmp_path / "auditor.pem"
    audit_path.write_text(json.dumps(audit))
    trust_path.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    args = SimpleNamespace(
        contamination_audit=str(audit_path),
        contamination_trust_root=str(trust_path),
    )

    verified = runner._contamination_audit(args, tasks)

    assert verified["signature"]["verified"] is True
    assert verified["signature"]["trust_root_sha256"] == key_id
    assert (
        verified["signature"]["signed_payload_sha256"]
        == hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    )

    audit["overlap_count"] = 1
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(runner.CampaignProducerError):
        runner._contamination_audit(args, tasks)


def test_contamination_audit_must_cover_bound_adapter_training_corpus(
    tmp_path: Path,
):
    tasks = generate_task_battery([7], domains=("mathematics",), difficulty=2)
    manifest = runner.build_task_manifest(tasks)
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    public_key = private_key.public_key()
    public_der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    body = {
        "schema": runner.CONTAMINATION_AUDIT_SCHEMA,
        "training_dataset_identity_sha256": "f" * 64,
        "task_manifest_sha256": manifest.manifest_sha256,
        "status": "passed_zero_overlap",
        "overlap_count": 0,
        "auditor_independence": "external",
        "corpora": [{"name": "wrong-corpus", "snapshot_sha256": "f" * 64}],
        "methods": ["exact_prompt", "normalized_prompt", "token_fivegram"],
    }
    audit = {
        **body,
        "signature": {
            "algorithm": "ed25519",
            "key_id": hashlib.sha256(public_der).hexdigest(),
            "signature_b64": base64.b64encode(private_key.sign(canonical_json_bytes(body))).decode(
                "ascii"
            ),
        },
    }
    audit_path = tmp_path / "audit.json"
    trust_path = tmp_path / "auditor.pem"
    audit_path.write_text(json.dumps(audit))
    trust_path.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    args = SimpleNamespace(
        contamination_audit=str(audit_path),
        contamination_trust_root=str(trust_path),
    )

    with pytest.raises(
        runner.CampaignProducerError,
        match="does not cover the adapter training corpus",
    ):
        runner._contamination_audit(
            args,
            tasks,
            expected_training_corpus_sha256="e" * 64,
        )


def test_prelaunch_trust_verifies_all_pinned_roles_before_inference(
    tmp_path: Path,
):
    now = int(time.time())
    root_key = Ed25519PrivateKey.generate()
    role_keys = {role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES}

    def role_pin(role: str) -> dict[str, str]:
        raw = (
            role_keys[role]
            .public_key()
            .public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        )
        implementation_sha256 = (
            runner._prelaunch_role_implementation_sha256(role)
            if role in {TASK_ISSUER, CAMPAIGN_RUNNER, "contamination_auditor"}
            else hashlib.sha256(f"{role}:implementation".encode()).hexdigest()
        )
        return {
            "signer_id": f"{role}-signer",
            "organization_id": f"{role}-organization",
            "public_key_b64": base64.b64encode(raw).decode("ascii"),
            "key_id": hashlib.sha256(raw).hexdigest(),
            "implementation_sha256": implementation_sha256,
            "release_sha256": hashlib.sha256(f"{role}:release".encode()).hexdigest(),
            "custody_class": "remote_hsm",
            "custody_evidence_sha256": hashlib.sha256(f"{role}:custody".encode()).hexdigest(),
        }

    policy_body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "resident-prelaunch-test",
        "policy_revision": 1,
        "campaign_name": "resident-prelaunch-test",
        "protocol_sha256": runner._campaign_protocol_sha256(),
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": now - 10,
        "not_before_unix": now - 5,
        "expires_at_unix": now + 3600,
        "roles": {role: role_pin(role) for role in CAMPAIGN_TRUST_ROLES},
    }
    root_payload = canonical_json_bytes(policy_body)
    root_raw = root_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    policy_document = {
        **policy_body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(root_key.sign(root_payload)).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(root_payload).hexdigest(),
        },
    }
    root_pem = root_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    verified_policy = validate_campaign_trust_policy(
        policy_document,
        trusted_root_public_key_pem=root_pem,
        expected_campaign_name="resident-prelaunch-test",
        expected_protocol_sha256=runner._campaign_protocol_sha256(),
        now_unix=now,
    )

    tasks = generate_task_battery([7], domains=("mathematics",), difficulty=1)
    task_manifest = runner.build_task_manifest(tasks)
    auditor_key = role_keys["contamination_auditor"]
    auditor_der = auditor_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    audit_body = {
        "schema": runner.CONTAMINATION_AUDIT_SCHEMA,
        "training_dataset_identity_sha256": "d" * 64,
        "task_manifest_sha256": task_manifest.manifest_sha256,
        "status": "passed_zero_overlap",
        "overlap_count": 0,
        "auditor_independence": "external",
        "corpora": [{"name": "training-corpus", "snapshot_sha256": "d" * 64}],
        "methods": ["exact_prompt", "normalized_prompt", "token_fivegram"],
    }
    audit_payload = canonical_json_bytes(audit_body)
    audit = {
        **audit_body,
        "signature": {
            "algorithm": "ed25519",
            "key_id": hashlib.sha256(auditor_der).hexdigest(),
            "signature_b64": base64.b64encode(auditor_key.sign(audit_payload)).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(audit_payload).hexdigest(),
            "public_key_der_b64": base64.b64encode(auditor_der).decode("ascii"),
            "trust_root_sha256": hashlib.sha256(auditor_der).hexdigest(),
            "verified": True,
        },
    }
    execution_config = {
        "difficulty": 1,
        "task_registry_version": runner.REGISTRY_VERSION,
        "generation_seeds": [7],
        "domains": ["mathematics"],
    }
    unsigned_plan = build_campaign_plan(
        "resident-prelaunch-test",
        tasks,
        model_identity={"model": "sealed"},
        adapter_identity={"adapter": "sealed"},
        execution_config=execution_config,
        contamination_audit=audit,
        claim_eligible=False,
    )
    args = SimpleNamespace(campaign_name="resident-prelaunch-test")
    payloads = runner._prelaunch_payloads(
        args,
        unsigned_plan=unsigned_plan,
        policy=verified_policy,
    )
    issuer_attestation = build_role_attestation(
        verified_policy,
        role=TASK_ISSUER,
        payload=payloads[TASK_ISSUER],
        signed_at_unix=now,
        private_key=role_keys[TASK_ISSUER],
    )
    runner_attestation = build_role_attestation(
        verified_policy,
        role=CAMPAIGN_RUNNER,
        payload=payloads[CAMPAIGN_RUNNER],
        signed_at_unix=now,
        private_key=role_keys[CAMPAIGN_RUNNER],
    )
    policy_path = tmp_path / "policy.json"
    root_path = tmp_path / "root.pem"
    issuer_path = tmp_path / "issuer.json"
    runner_path = tmp_path / "runner.json"
    policy_path.write_text(json.dumps(policy_document))
    root_path.write_bytes(root_pem)
    issuer_path.write_text(json.dumps(issuer_attestation))
    runner_path.write_text(json.dumps(runner_attestation))
    args.campaign_trust_policy = str(policy_path)
    args.campaign_trust_root = str(root_path)
    args.task_issuer_attestation = str(issuer_path)
    args.runner_attestation = str(runner_path)

    trust = runner._verified_campaign_trust(
        args,
        unsigned_plan=unsigned_plan,
        contamination_audit=audit,
    )

    assert trust is not None
    assert trust["prelaunch_verified"] is True
    assert trust["externally_custodied"] is True
    assert trust["policy_sha256"] == verified_policy.policy_sha256
    assert trust["unsigned_plan_sha256"] == unsigned_plan.plan_sha256


def test_effective_full_stack_shape_is_frozen_not_cli_placeholder():
    args = SimpleNamespace(
        rlc_profile="resident_full_stack",
        n_slots=16,
        branches=7,
        rlc_steps=8,
        decode_max_tokens=512,
    )

    effective = runner._build_rlc_config(args)

    assert effective.workspace.n_slots == 4
    assert effective.branches.n_branches == 2
    assert effective.recurrence.max_steps == 2
    assert effective.latent_opt.enabled is True
    assert effective.fast_weights.enabled is True
    assert effective.decode_contract == "final_answer_v1"
    assert effective.decode_contract_grace_tokens == 512
    assert effective.verifier_probe_max_tokens == 192
    assert effective.answer_replacement_enabled is False


@pytest.mark.parametrize(
    ("profile", "latent_opt", "fast_weights", "exchange_gamma"),
    [
        ("resident_full_stack", True, True, 0.35),
        ("resident_full_stack_no_latent_opt", False, True, 0.35),
        ("resident_full_stack_no_fast_weights", True, False, 0.35),
        ("resident_full_stack_no_branch_exchange", True, True, 0.0),
    ],
)
def test_full_stack_mechanism_profiles_toggle_one_mechanism(
    profile: str,
    latent_opt: bool,
    fast_weights: bool,
    exchange_gamma: float,
):
    args = SimpleNamespace(
        rlc_profile=profile,
        n_slots=16,
        branches=7,
        rlc_steps=8,
        decode_max_tokens=512,
    )

    effective = runner._build_rlc_config(args)

    assert effective.workspace.n_slots == 4
    assert effective.branches.n_branches == 2
    assert effective.latent_opt.enabled is latent_opt
    assert effective.fast_weights.enabled is fast_weights
    assert effective.branches.exchange_gamma == pytest.approx(exchange_gamma)
    assert effective.answer_replacement_enabled is False
    if profile == "resident_full_stack_no_branch_exchange":
        assert effective.branches.exchange_interval > effective.recurrence.max_steps


def test_v2_execution_spec_overrides_cli_and_preserves_training_graph():
    args = SimpleNamespace(
        rlc_profile="resident_full_stack",
        n_slots=99,
        branches=7,
        rlc_steps=19,
        decode_max_tokens=128,
    )
    spec = RLCExecutionSpec(
        n_slots=6,
        branch_roles=("constructive_solution", "counterexample_search"),
        recurrent_steps=4,
        exchange_interval=1,
        alpha=0.4,
    )

    effective = runner._build_rlc_config(args, spec.to_dict())

    assert effective.workspace.n_slots == 6
    assert effective.branches.roles == spec.branch_roles
    assert effective.recurrence.max_steps == 4
    assert effective.recurrence.min_steps == 4
    assert effective.recurrence.fixed_depth is True
    assert effective.latent_opt.enabled is False
    assert effective.fast_weights.enabled is False
    assert effective.escape == {"enabled": False}
    assert effective.decode_contract == "final_answer_v1"
    assert effective.decode_contract_grace_tokens == 128
    assert effective.verifier_probe_max_tokens == 128
    assert effective.answer_replacement_enabled is False


def test_paired_campaign_contract_discloses_raw_output_attribution_policy():
    args = SimpleNamespace(
        rlc_profile="recurrence_attribution",
        profile="primary",
        n_slots=4,
        branches=2,
        rlc_steps=2,
        decode_max_tokens=128,
        difficulty=2,
        task_registry_version="2026.08.06.1",
        seed_count=4,
        seed_values=(1, 2, 3, 4),
        domain_values=("mathematics",),
        episode_timeout=600.0,
        load_timeout=1200.0,
        warmup_timeout=600.0,
        arm_timeout=14400.0,
        campaign_timeout=43200.0,
        equal_compute_max_samples=8,
        max_infra_attempts=3,
        sequential_look_values=(),
        sequential_alpha_weight_values=(),
    )
    policy = runner._execution_config(args, {})["response_contract_policy"]

    assert policy == {
        "schema": "public_response_contract_v2",
        "termination": "final_answer_v1",
        "contract_grace_tokens": 128,
        "verifier_probe_max_tokens": 128,
        "applies_identically_to_all_decode_arms": True,
        "output_editing": False,
        "rlc_answer_replacement_enabled": False,
        "causal_attribution_rule": "raw_terminal_decode_all_arms",
        "observed_decode_receipt_required": True,
    }


def test_vanilla_decode_uses_same_contract_stop_and_bounded_grace(monkeypatch):
    task = generate_task_battery([7], domains=("mathematics",), difficulty=1)[0].public
    generated = 'work\nFINAL_ANSWER: {"answer":7}TRAILING'
    consumed: list[str] = []
    observed: dict = {}

    def stream_generate(_model, _tokenizer, *, prompt, max_tokens, **kwargs):
        observed["prompt"] = prompt
        observed["max_tokens"] = max_tokens
        for index, character in enumerate(generated, start=1):
            consumed.append(character)
            yield SimpleNamespace(text=character, generation_tokens=index)

    import mlx_lm

    monkeypatch.setattr(mlx_lm, "stream_generate", stream_generate)

    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return "rendered"

        def encode(self, text):
            return list(str(text))

    class AccountingEngine:
        def _encode(self, *_args):
            return list("rendered")

        def _information_receipt(
            self,
            *,
            encoded_tokens,
            token_count,
            context_items,
            policy_evidence,
            verifier,
        ):
            del context_items, policy_evidence, verifier
            return build_information_receipt(
                sources=[
                    {
                        "source_id": "rendered_model_input",
                        "kind": "model_input_tokens",
                        "content_sha256": hashlib.sha256(encoded_tokens).hexdigest(),
                        "byte_count": len(encoded_tokens),
                        "token_count": token_count,
                    }
                ],
                policies={"fixture": "a" * 64},
            )

    args = SimpleNamespace(
        model_type="qwen2",
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        head_dim=16,
    )
    model = SimpleNamespace(
        args=args,
        model=SimpleNamespace(args=args, layers=[object(), object()]),
    )
    text, layer_apps, *_prefix, decode_receipt = runner._vanilla_once(
        model,
        Tokenizer(),
        task,
        max_tokens=64,
        accounting_engine=AccountingEngine(),
    )

    expected = 'work\nFINAL_ANSWER: {"answer":7}'
    assert text == expected
    assert "TRAILING" not in text
    assert "".join(consumed) == expected
    assert observed["max_tokens"] == 128
    assert observed["prompt"] == list("rendered")
    assert layer_apps == (len("rendered") + len(expected) - 1) * 2
    assert decode_receipt == {
        "schema": "aura.latent_cortex.arm_decode_receipt.v1",
        "decode_contract": "final_answer_v1",
        "decode_termination": "contract_complete",
        "decode_generated_tokens": len(expected),
        "decode_max_tokens": 64,
        "decode_contract_grace_tokens": 64,
        "raw_output_sha256": hashlib.sha256(expected.encode()).hexdigest(),
    }


def test_rlc_campaign_verifier_receives_public_response_contract():
    from core.brain.llm.latent_cortex.types import (
        EpisodeReceipt,
        LatentReasoningResult,
    )

    task = generate_task_battery([7], domains=("mathematics",), difficulty=1)[0].public
    captured: dict = {}
    profile = ModelComputeProfile(
        model_type="runner-fixture",
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=64,
        head_dim=4,
    )
    ledger = ResourceLedger(profile)
    ledger.charge("fixture_episode", transformer_layer_apps=1)
    information = build_information_receipt(
        sources=[
            {
                "source_id": "task",
                "kind": "task_prompt",
                "content_sha256": task.task_payload_sha256,
                "byte_count": len(task.prompt.encode()),
                "token_count": 1,
            }
        ],
        policies={"fixture": "a" * 64},
    )

    class Engine:
        def reason(self, **kwargs):
            captured.update(kwargs)
            verifier = kwargs["verifier"]
            verifier('FINAL_ANSWER: {"count":2,"witness":[1,2]}')
            return LatentReasoningResult(
                ok=True,
                text='FINAL_ANSWER: {"count":2,"witness":[1,2]}',
                receipt=EpisodeReceipt(
                    budget={
                        "resource_accounting": ledger.to_receipt(),
                        "information_accounting": information,
                    }
                ),
            )

    text, _cost, receipt, resource, observed_information = runner._run_rlc(
        Engine(),
        task,
        SimpleNamespace(episode_timeout=10.0, decode_max_tokens=128),
    )

    assert text.startswith("FINAL_ANSWER:")
    assert captured["verifier"].response_contract == task.response_contract
    assert receipt["verifier_guidance"]["response_contract_required"] is True
    assert receipt["verifier_guidance"]["response_contract_satisfied"] is True
    assert resource["accounting_complete"] is True
    assert observed_information == information


@pytest.mark.parametrize(
    ("reason", "text"),
    [
        ("answer_replacement_abstained", ""),
        ("decode_incomplete:token_budget", "partial answer"),
    ],
)
def test_rlc_campaign_scores_bounded_policy_failures_as_negative_evidence(reason, text):
    from core.brain.llm.latent_cortex.types import EpisodeReceipt, LatentReasoningResult

    task = generate_task_battery([7], domains=("mathematics",), difficulty=1)[0].public
    profile = ModelComputeProfile(
        model_type="runner-fixture",
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=64,
        head_dim=4,
    )
    ledger = ResourceLedger(profile)
    ledger.charge("fixture_episode", transformer_layer_apps=1)
    information = build_information_receipt(
        sources=[
            {
                "source_id": "task",
                "kind": "task_prompt",
                "content_sha256": task.task_payload_sha256,
                "byte_count": len(task.prompt.encode()),
                "token_count": 1,
            }
        ],
        policies={"fixture": "a" * 64},
    )

    class Engine:
        @staticmethod
        def reason(**_kwargs):
            return LatentReasoningResult(
                ok=False,
                text=text,
                reason=reason,
                receipt=EpisodeReceipt(
                    budget={
                        "resource_accounting": ledger.to_receipt(),
                        "information_accounting": information,
                    }
                ),
            )

    observed_text, _cost, _receipt, resource, observed_information = runner._run_rlc(
        Engine(),
        task,
        SimpleNamespace(episode_timeout=10.0, decode_max_tokens=128),
    )

    assert observed_text == text
    assert resource["accounting_complete"] is True
    assert observed_information == information


def test_rlc_campaign_still_rejects_non_policy_episode_failure():
    from core.brain.llm.latent_cortex.types import EpisodeReceipt, LatentReasoningResult

    task = generate_task_battery([7], domains=("mathematics",), difficulty=1)[0].public

    class Engine:
        @staticmethod
        def reason(**_kwargs):
            return LatentReasoningResult(
                ok=False,
                text="must not escape",
                reason="checkpoint_invariant_violated",
                receipt=EpisodeReceipt(),
            )

    with pytest.raises(
        runner.CampaignProducerError,
        match="checkpoint_invariant_violated",
    ):
        runner._run_rlc(
            Engine(),
            task,
            SimpleNamespace(episode_timeout=10.0, decode_max_tokens=128),
        )


def test_projection_resolution_is_exact_and_rejects_missing_owner():
    projection = object()
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(o_proj=projection),
                )
            ]
        )
    )

    parent, leaf, observed = runner._resolve_projection(model, "model.layers.0.self_attn.o_proj")
    assert parent is model.model.layers[0].self_attn
    assert leaf == "o_proj"
    assert observed is projection
    with pytest.raises(runner.CampaignProducerError, match="owner is missing"):
        runner._resolve_projection(model, "model.layers.0.mlp.o_proj")


def test_deadline_alarm_interrupts_and_restores_process_timer():
    # Under pytest-timeout's signal method the process already carries its timer, and the test failed only when run
    # with --timeout (15 November 2026). That timer is set aside here and put back after.
    ambient = signal.setitimer(signal.ITIMER_REAL, 0.0)
    try:
        before = signal.getitimer(signal.ITIMER_REAL)
        with pytest.raises(TimeoutError, match="test_stage exceeded"):
            with runner._deadline_alarm(0.02, "test_stage"):
                time.sleep(0.2)

        after = signal.getitimer(signal.ITIMER_REAL)
        assert before == (0.0, 0.0)
        assert after == (0.0, 0.0)
    finally:
        if ambient[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, *ambient)
