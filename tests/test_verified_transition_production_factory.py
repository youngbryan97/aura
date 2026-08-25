"""Adversarial tests for post-load verified-provider construction."""

from __future__ import annotations

import base64
import hashlib
import stat
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    EVIDENCE_VERIFIER,
    TASK_ISSUER,
    build_role_attestation,
    policy_signed_payload,
    validate_campaign_trust_policy,
    verify_role_attestation,
)
from core.learning import verified_transition_launch_bundle as launch_bundle
from core.learning import verified_transition_production_factory as production_factory
from core.learning.recurrence_curriculum import khop_reachability
from core.learning.recurrent_grpo import RecurrentSamplingConfig
from core.learning.verified_token_trace import (
    build_tokenizer_bundle_identity,
    tokenizer_file_bindings_from_bytes,
)
from core.learning.verified_training_task import build_verified_training_task
from core.learning.verified_transition_causal_campaign import (
    CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA,
    CausalCampaignScheduleEntry,
    VerifiedTransitionCausalCampaignLedger,
    build_causal_campaign_manifest,
)
from core.learning.verified_transition_episode import canonical_json_bytes
from core.learning.verified_transition_group_admission import (
    sampling_config_document_sha256,
    validate_transition_group_manifest,
)
from core.learning.verified_transition_launch_bundle import (
    VERIFIED_TRANSITION_LAUNCH_BUNDLE_SCHEMA,
    VerifiedTransitionLaunchBundleError,
    VerifiedTransitionRuntimeComponents,
)
from core.learning.verified_transition_production_factory import (
    COMMAND_EVIDENCE_VERIFIER_RESPONSE_SCHEMA,
    COMMAND_SIGNER_RESPONSE_SCHEMA,
    JIT_PROVIDER_CONFIG_SCHEMA,
    CommandRoleSignerBroker,
    JITAdmittingVerifiedTransitionGroupProvider,
    JITVerifiedTransitionPlanStore,
    ProductionVerifiedTransitionProviderFactory,
    ProviderBoundTrainingTask,
    VerifiedTransitionProductionFactoryError,
    sampling_config_contract_document,
)
from core.learning.verified_transition_provider import (
    TASK_COMMITMENT_SCHEMA,
    VerifiedTransitionProviderError,
    build_verified_transition_provider_contract,
    callable_source_sha256,
)
from core.learning.verified_transition_reward import TransitionRewardConfig
from core.learning.verified_transition_trainer import (
    VerifiedTransitionProviderRuntime,
    VerifiedTransitionSamplingEntry,
    VerifiedTransitionSamplingPlan,
    VerifiedTransitionTrainingScheduleEntry,
)

BASE_SECOND = 1_800_000_000

_TOKENIZER_BUNDLE = build_tokenizer_bundle_identity(
    tokenizer_class="test.IntegerTokenizer",
    tokenizer_files=tokenizer_file_bindings_from_bytes(
        {
            "tokenizer.json": b'{"kind":"integer"}',
            "tokenizer_config.json": b'{"separator":" "}',
        }
    ),
    chat_template=None,
    special_token_map={},
    encode_options={},
    decode_options={},
    implementation_source_sha256="b" * 64,
)


class _TokenizerAdapter:
    bundle_identity = _TOKENIZER_BUNDLE

    @staticmethod
    def encode_prompt(text: str) -> tuple[int, ...]:
        return tuple(int(value) for value in text.split())

    @staticmethod
    def decode_output(tokens: Any) -> str:
        return " ".join(str(token) for token in tokens)

    @classmethod
    def stream_decode_deltas(cls, tokens: Any) -> tuple[str, ...]:
        values = tuple(tokens)
        rendered = [cls.decode_output(values[: index + 1]) for index in range(len(values))]
        return tuple(
            value if index == 0 else value[len(rendered[index - 1]) :]
            for index, value in enumerate(rendered)
        )


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _producer(_request: Any) -> Any:
    raise AssertionError("not called by construction test")


def _loader(_request: Any) -> tuple[Any, ...]:
    return ()


def _finalizer(_request: Any) -> Any:
    raise AssertionError("not called by construction test")


def _scorer(_task: Any, _response: Any) -> dict[str, Any]:
    return {"correct": True}


def _encode(value: bytes) -> tuple[int, ...]:
    return tuple(value)


def _decode(value: Any) -> bytes:
    return bytes(value)


def _public_raw(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _public_pem(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _pin(
    role: str,
    key: Ed25519PrivateKey,
    *,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    public = _public_raw(key)
    pin = {
        "signer_id": f"{role}-signer",
        "organization_id": f"{role}-external-organization",
        "public_key_b64": base64.b64encode(public).decode("ascii"),
        "key_id": hashlib.sha256(public).hexdigest(),
        "implementation_sha256": _sha(f"{role}-implementation"),
        "release_sha256": _sha(f"{role}-release"),
        "custody_class": "external_service",
        "custody_evidence_sha256": _sha(f"{role}-custody"),
    }
    pin.update(overrides or {})
    return pin


def _trust_material(
    *,
    role_keys: dict[str, Ed25519PrivateKey] | None = None,
    task_issuer_pin_overrides: dict[str, str] | None = None,
    role_pin_overrides: dict[str, dict[str, str]] | None = None,
) -> tuple[Any, dict[str, Ed25519PrivateKey]]:
    root = Ed25519PrivateKey.generate()
    role_keys = role_keys or {role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES}
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "jit-provider-test-policy",
        "policy_revision": 1,
        "campaign_name": "jit-provider-test",
        "protocol_sha256": _sha("protocol"),
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": BASE_SECOND,
        "not_before_unix": BASE_SECOND + 100,
        "expires_at_unix": BASE_SECOND + 10_000,
        "roles": {
            role: _pin(
                role,
                role_keys[role],
                overrides=(
                    (role_pin_overrides or {}).get(role)
                    or (task_issuer_pin_overrides if role == TASK_ISSUER else None)
                ),
            )
            for role in CAMPAIGN_TRUST_ROLES
        },
    }
    signed = canonical_json_bytes(body)
    root_raw = _public_raw(root)
    document = {
        **body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(root.sign(signed)).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        },
    }
    assert policy_signed_payload(document) == body
    policy = validate_campaign_trust_policy(
        document,
        trusted_root_public_key_pem=_public_pem(root),
        expected_campaign_name="jit-provider-test",
        expected_protocol_sha256=_sha("protocol"),
        now_unix=BASE_SECOND + 120,
    )
    return policy, role_keys


def _external_signer(
    tmp_path: Path,
    *,
    role: str,
    key: Ed25519PrivateKey,
) -> tuple[CommandRoleSignerBroker, Path, dict[str, str]]:
    private_raw = key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    script = tmp_path / f"{role}-external-signer.py"
    verifier_branch = ""
    if role == EVIDENCE_VERIFIER:
        verifier_branch = (
            "if envelope.get('schema') == "
            "'aura.external_evidence_verifier.request.v2':\n"
            " from pathlib import Path\n"
            " sys.path.insert(0,str(Path.cwd()))\n"
            " from core.learning.verified_recurrent_transition_repository "
            "import campaign_trust_policy_from_verifier_material,"
            "verify_recurrent_evidence_manifest_artifacts\n"
            " policy=campaign_trust_policy_from_verifier_material("
            "envelope['campaign_trust_policy'])\n"
            " receipt=verify_recurrent_evidence_manifest_artifacts("
            "envelope['evidence_manifest'],"
            "campaign_trust_policy=policy,"
            "verifier_identity=envelope['verifier_identity'],"
            "verified_at_unix=envelope['verified_at_unix'])\n"
            f" response={{'schema':'{COMMAND_EVIDENCE_VERIFIER_RESPONSE_SCHEMA}',"
            "'request_sha256':envelope['request_sha256'],"
            "'verification_receipt':receipt}\n"
            " sys.stdout.write(json.dumps(response,sort_keys=True,"
            "separators=(',',':'))+'\\n')\n"
            " raise SystemExit(0)\n"
        )
    script.write_text(
        f"#!{sys.executable}\n"
        "import base64,json,sys\n"
        "from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey\n"
        f"key=Ed25519PrivateKey.from_private_bytes(bytes.fromhex('{private_raw.hex()}'))\n"
        "envelope=json.loads(sys.stdin.buffer.read())\n"
        f"{verifier_branch}"
        "request=envelope['signature_request']\n"
        "signature=key.sign(base64.b64decode(request['signed_payload_b64']))\n"
        f"response={{'schema':'{COMMAND_SIGNER_RESPONSE_SCHEMA}',"
        "'request_sha256':request['request_sha256'],"
        "'signature_b64':base64.b64encode(signature).decode('ascii')}\n"
        "sys.stdout.write(json.dumps(response,sort_keys=True,separators=(',',':'))+'\\n')\n",
        encoding="ascii",
    )
    script.chmod(0o700)
    release = tmp_path / f"{role}-external-signer-release.json"
    custody = tmp_path / f"{role}-external-signer-custody.json"
    release.write_bytes(canonical_json_bytes({"release": f"{role}-test-v1"}))
    custody.write_bytes(canonical_json_bytes({"custody": f"{role}-external-test"}))
    release.chmod(0o600)
    custody.chmod(0o600)
    executable_sha = hashlib.sha256(script.read_bytes()).hexdigest()
    overrides = {
        "implementation_sha256": executable_sha,
        "release_sha256": hashlib.sha256(release.read_bytes()).hexdigest(),
        "custody_evidence_sha256": hashlib.sha256(custody.read_bytes()).hexdigest(),
    }
    broker = CommandRoleSignerBroker(
        identity=f"{role}-external-command-test",
        executable=script,
        executable_sha256=executable_sha,
        release_manifest=release,
        custody_evidence=custody,
        timeout_seconds=30.0 if role == EVIDENCE_VERIFIER else 5.0,
        inherited_environment_names=(),
    )
    return broker, script, overrides


def _command_signer_material(
    tmp_path: Path,
) -> tuple[Any, dict[str, Ed25519PrivateKey], CommandRoleSignerBroker, Path]:
    role_keys = {role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES}
    broker, script, overrides = _external_signer(
        tmp_path,
        role=TASK_ISSUER,
        key=role_keys[TASK_ISSUER],
    )
    policy, keys = _trust_material(
        role_keys=role_keys,
        task_issuer_pin_overrides=overrides,
    )
    return policy, keys, broker, script


@dataclass
class _Task:
    task_id: str
    prompt: str = "problem"
    domain: str = "logic"
    depth: int = 1

    def grade(self, _response: str) -> dict[str, Any]:
        return {"correct": True}


class _SigningBroker:
    identity = "external-test-broker"
    source_sha256 = _sha("external-test-broker-source")

    @property
    def implementation_sha256(self) -> str:
        return _sha(f"{TASK_ISSUER}-implementation")

    @property
    def release_sha256(self) -> str:
        return _sha(f"{TASK_ISSUER}-release")

    @property
    def custody_evidence_sha256(self) -> str:
        return _sha(f"{TASK_ISSUER}-custody")

    def __init__(self, key: Ed25519PrivateKey) -> None:
        self.key = key
        self.calls: list[tuple[str, str]] = []

    def attest(self, policy: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((kwargs["role"], kwargs["purpose"]))
        return build_role_attestation(
            policy,
            role=kwargs["role"],
            payload=kwargs["payload"],
            signed_at_unix=kwargs["signed_at_unix"],
            private_key=self.key,
        )


class _NoSigningBroker(_SigningBroker):
    def attest(self, policy: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("durable JIT plan should avoid a second signing call")


class _Provider:
    def __init__(self, policy: Any, commitment: dict[str, Any]) -> None:
        self.contract_sha256 = _sha("contract")
        self.campaign_id = "campaign"
        self.campaign_schedule_root_sha256 = _sha("schedule")
        self.expected_policy_sha256 = _sha("policy")
        self.policy = policy
        self.commitment = commitment
        self.manifest: dict[str, Any] | None = None

    def task_commitment(self, *, sequence: int) -> dict[str, Any]:
        assert sequence == 0
        return dict(self.commitment)

    def training_schedule_entry(self, *, sequence: int) -> VerifiedTransitionTrainingScheduleEntry:
        assert sequence == 0
        return VerifiedTransitionTrainingScheduleEntry(
            campaign_sequence=0,
            task_id=self.commitment["task_id"],
            trainer_sample_seed=self.commitment["trainer_sample_seed"],
        )

    def lineage_plan_for_manifest(self, **kwargs: Any) -> dict[str, Any]:
        manifest = validate_transition_group_manifest(kwargs["group_manifest"])
        return {
            "schema": "aura.verified_transition.lineage_plan.v1",
            "contract_sha256": self.contract_sha256,
            "campaign_id": self.campaign_id,
            "campaign_schedule_root_sha256": self.campaign_schedule_root_sha256,
            "sequence": kwargs["sequence"],
            "task_commitment_sha256": _digest(self.commitment),
            "policy_before_sha256": kwargs["policy_before_sha256"],
            "group_manifest_sha256": manifest["manifest_sha256"],
        }

    def admit_group_plan(self, **kwargs: Any) -> dict[str, Any]:
        manifest = validate_transition_group_manifest(kwargs["group_manifest"])
        lineage = self.lineage_plan_for_manifest(
            sequence=kwargs["sequence"],
            policy_before_sha256=kwargs["policy_before_sha256"],
            group_manifest=manifest,
        )
        verify_role_attestation(
            self.policy,
            kwargs["group_manifest_attestation"],
            role=TASK_ISSUER,
            expected_payload=manifest,
        )
        verify_role_attestation(
            self.policy,
            kwargs["lineage_attestation"],
            role=TASK_ISSUER,
            expected_payload=lineage,
        )
        self.manifest = manifest
        return {"sequence": 0}

    def sampling_plan(self, **kwargs: Any) -> VerifiedTransitionSamplingPlan:
        if self.manifest is None:
            raise VerifiedTransitionProviderError("provider_sampling_plan_missing")
        entries = tuple(
            VerifiedTransitionSamplingEntry(
                episode_id=entry["episode_id"],
                rng_root_sha256=entry["rng_root_sha256"],
                producing_branch_index=entry["producing_branch_index"],
                sample_seed=entry["sample_seed"],
                sampling_config_sha256=entry["sampling_config_sha256"],
            )
            for entry in self.manifest["entries"]
        )
        return VerifiedTransitionSamplingPlan(
            campaign_sequence=kwargs["sequence"],
            group_manifest_sha256=self.manifest["manifest_sha256"],
            task_id=self.manifest["task_id"],
            policy_sha256=kwargs["policy_sha256"],
            prompt_tokens_sha256=_digest(list(kwargs["prompt_tokens"])),
            execution_spec_sha256=self.commitment["recurrent_execution_spec_sha256"],
            entries=entries,
            sampling_config={},
        )


def _jit_material(tmp_path: Path) -> dict[str, Any]:
    policy, keys = _trust_material()
    prompt = (11, 12)
    commitment = {
        "schema": "aura.verified_transition.task_commitment.v2",
        "sequence": 0,
        "task_id": "task-0",
        "trainer_sample_seed": 99,
        "immutable_task_sha256": _sha("task"),
        "prompt_tokens_sha256": _digest(list(prompt)),
        "recurrent_execution_spec_sha256": _sha("execution"),
        "sample_seeds": [101, 102],
    }
    return {
        "policy": policy,
        "keys": keys,
        "prompt": prompt,
        "task": _Task("task-0"),
        "commitment": commitment,
        "store": JITVerifiedTransitionPlanStore(
            tmp_path / "plans", contract_sha256=_sha("contract")
        ),
        "sampling": RecurrentSamplingConfig(max_tokens=2),
    }


def _factory_material(
    tmp_path: Path,
    *,
    use_in_process_signer: bool = False,
    shared_external_signer: bool = False,
    shared_custody_only: bool = False,
) -> dict[str, Any]:
    role_keys = {role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES}
    broker, _task_script, task_overrides = _external_signer(
        tmp_path,
        role=TASK_ISSUER,
        key=role_keys[TASK_ISSUER],
    )
    verifier_broker, _verifier_script, verifier_overrides = _external_signer(
        tmp_path,
        role=EVIDENCE_VERIFIER,
        key=role_keys[EVIDENCE_VERIFIER],
    )
    if shared_external_signer:
        verifier_broker = broker
        verifier_overrides = task_overrides
    elif shared_custody_only:
        verifier_broker = CommandRoleSignerBroker(
            identity=verifier_broker.identity,
            executable=verifier_broker._executable,
            executable_sha256=verifier_broker.implementation_sha256,
            release_manifest=verifier_broker._release_manifest,
            custody_evidence=broker._custody_evidence,
            arguments=(),
            timeout_seconds=5.0,
            inherited_environment_names=(),
        )
        verifier_overrides = {
            **verifier_overrides,
            "custody_evidence_sha256": broker.custody_evidence_sha256,
        }
    policy, keys = _trust_material(
        role_keys=role_keys,
        role_pin_overrides={
            TASK_ISSUER: task_overrides,
            EVIDENCE_VERIFIER: verifier_overrides,
        },
    )
    task_factory_broker: Any = (
        _SigningBroker(keys[TASK_ISSUER]) if use_in_process_signer else broker
    )
    verifier_factory_broker: Any = (
        _SigningBroker(keys[EVIDENCE_VERIFIER]) if use_in_process_signer else verifier_broker
    )
    roots = {
        name: str((tmp_path / name).resolve())
        for name in (
            "campaign",
            "transition_artifacts",
            "updates",
            "replay_artifacts",
        )
    }
    for root in roots.values():
        Path(root).mkdir(mode=0o700, parents=True)
    sampling = RecurrentSamplingConfig(max_tokens=2)
    output_root = (tmp_path / "output").resolve()
    transaction_root = output_root / "verified-transition-transactions"
    provider_config = {
        "evidence_timeout_ms": 30_000,
        "training_argv": [
            "tools/train_grpo.py",
            "--model",
            "/models/resident",
        ],
        "training_argv_sha256": _digest(["tools/train_grpo.py", "--model", "/models/resident"]),
        "jit_plan": {
            "schema": JIT_PROVIDER_CONFIG_SCHEMA,
            "reward_config_sha256": _digest(TransitionRewardConfig().to_dict()),
            "sampling_config": sampling_config_contract_document(sampling),
            "branch_count": 2,
            "signer_broker_identity": broker.identity,
            "signer_broker_source_sha256": broker.source_sha256,
            "plan_store_root": str((Path(roots["replay_artifacts"]) / "jit-plans").resolve()),
            "trainer_output_root": str(output_root),
            "transaction_root": str(transaction_root),
        },
    }
    task = khop_reachability(1, 123)
    answer_nonce = b"factory-test-answer-nonce-material" * 2
    public_task, _sealed_answer = build_verified_training_task(task, answer_nonce=answer_nonce)
    task_document = public_task.to_dict()
    schedule = [
        {
            "schema": TASK_COMMITMENT_SCHEMA,
            "sequence": 0,
            "task_id": task.task_id,
            "trainer_sample_seed": 99,
            "immutable_task_sha256": _digest(task_document),
            "prompt_tokens_sha256": _sha("prompt"),
            "recurrent_execution_spec_sha256": _sha("execution"),
            "sample_seeds": [101, 102],
        }
    ]
    contract = build_verified_transition_provider_contract(
        provider_config=provider_config,
        evidence_producer_identity="evidence-producer",
        evidence_producer_source_sha256=callable_source_sha256(_producer),
        durable_artifact_loader_identity="replay-loader",
        durable_artifact_loader_source_sha256=callable_source_sha256(_loader),
        campaign_finalizer_identity="campaign-finalizer",
        campaign_finalizer_source_sha256=callable_source_sha256(_finalizer),
        trust_policy_sha256=policy.policy_sha256,
        trust_root_key_id=policy.root_key_id,
        campaign_id="jit-provider-test",
        initial_policy_sha256=_sha("initial-policy"),
        scorer_identity="independent-scorer",
        scorer_source_sha256=callable_source_sha256(_scorer),
        token_codec_identity="byte-codec",
        token_encoder_source_sha256=callable_source_sha256(_encode),
        token_decoder_source_sha256=callable_source_sha256(_decode),
        tokenizer_bundle=_TOKENIZER_BUNDLE,
        dataset_sha256=_sha("dataset"),
        task_schedule=schedule,
        ledger_roots=roots,
        frozen_at_unix_ns=(BASE_SECOND + 150) * 1_000_000_000,
    )
    ledger_manifest = {
        "campaign_id": contract["campaign_id"],
        "provider_contract_sha256": contract["contract_sha256"],
        "campaign_schedule_root_sha256": contract["campaign_schedule_root_sha256"],
        "trust_policy_sha256": contract["trust_policy_sha256"],
        "initial_policy_sha256": contract["initial_policy_sha256"],
    }
    ledger = SimpleNamespace(
        root=Path(roots["campaign"]),
        campaign_manifest=lambda: dict(ledger_manifest),
    )
    factory = ProductionVerifiedTransitionProviderFactory(
        contract=contract,
        provider_config=provider_config,
        campaign_ledger=ledger,
        campaign_trust_policy=policy,
        evidence_producer=_producer,
        evidence_producer_identity="evidence-producer",
        durable_artifact_loader=_loader,
        durable_artifact_loader_identity="replay-loader",
        campaign_finalizer=_finalizer,
        campaign_finalizer_identity="campaign-finalizer",
        independent_scorer=_scorer,
        scorer_identity="independent-scorer",
        token_encoder=_encode,
        token_decoder=_decode,
        token_codec_identity="byte-codec",
        task_issuer_signer_broker=task_factory_broker,
        evidence_verifier_signer_broker=verifier_factory_broker,
        task_commitments={task.task_id: task_document},
        task_answer_nonces={task.task_id: answer_nonce},
    )
    return {
        "factory": factory,
        "policy": policy,
        "broker": broker,
        "verifier_broker": verifier_broker,
        "contract": contract,
        "provider_config": provider_config,
        "ledger": ledger,
        "task_document": task_document,
        "task": task,
        "output_root": output_root,
        "transaction_root": transaction_root,
        "role_keys": keys,
    }


def test_jit_provider_signs_and_persists_exact_plan_before_sampling(
    tmp_path: Path,
) -> None:
    material = _jit_material(tmp_path)
    provider = _Provider(material["policy"], material["commitment"])
    broker = _SigningBroker(material["keys"][TASK_ISSUER])
    wrapped = JITAdmittingVerifiedTransitionGroupProvider(
        provider=provider,
        policy=material["policy"],
        signer_broker=broker,
        plan_store=material["store"],
        sampling_config=material["sampling"],
        branch_count=2,
        reward_config_sha256=_sha("reward-config"),
        now_unix_ns=lambda: (BASE_SECOND + 200) * 1_000_000_000,
    )

    plan = wrapped.sampling_plan(
        sequence=0,
        task=material["task"],
        prompt_tokens=material["prompt"],
        policy_sha256=_sha("policy"),
    )

    assert len(plan.entries) == 2
    assert plan.sampling_config == material["sampling"].to_dict()
    assert [entry.sample_seed for entry in plan.entries] == [101, 102]
    assert [entry.producing_branch_index for entry in plan.entries] == [0, 1]
    assert all(
        entry.sampling_config_sha256 == sampling_config_document_sha256(material["sampling"])
        for entry in plan.entries
    )
    assert [purpose.rsplit(":", 1)[-1] for _role, purpose in broker.calls] == [
        "manifest",
        "lineage",
    ]
    assert material["store"].load(sequence=0) is not None


def test_jit_provider_reuses_crash_durable_plan_without_resigning(
    tmp_path: Path,
) -> None:
    material = _jit_material(tmp_path)
    first = JITAdmittingVerifiedTransitionGroupProvider(
        provider=_Provider(material["policy"], material["commitment"]),
        policy=material["policy"],
        signer_broker=_SigningBroker(material["keys"][TASK_ISSUER]),
        plan_store=material["store"],
        sampling_config=material["sampling"],
        branch_count=2,
        reward_config_sha256=_sha("reward-config"),
        now_unix_ns=lambda: (BASE_SECOND + 200) * 1_000_000_000,
    )
    first.sampling_plan(
        sequence=0,
        task=material["task"],
        prompt_tokens=material["prompt"],
        policy_sha256=_sha("policy"),
    )

    restarted = JITAdmittingVerifiedTransitionGroupProvider(
        provider=_Provider(material["policy"], material["commitment"]),
        policy=material["policy"],
        signer_broker=_NoSigningBroker(material["keys"][TASK_ISSUER]),
        plan_store=material["store"],
        sampling_config=material["sampling"],
        branch_count=2,
        reward_config_sha256=_sha("reward-config"),
    )
    replayed = restarted.sampling_plan(
        sequence=0,
        task=material["task"],
        prompt_tokens=material["prompt"],
        policy_sha256=_sha("policy"),
    )
    assert [entry.episode_id for entry in replayed.entries] == [
        "campaign:s0:e0",
        "campaign:s0:e1",
    ]


@pytest.mark.parametrize("drift", ["prompt", "policy", "sampling"])
def test_durable_plan_rejects_runtime_drift(tmp_path: Path, drift: str) -> None:
    material = _jit_material(tmp_path)
    first = JITAdmittingVerifiedTransitionGroupProvider(
        provider=_Provider(material["policy"], material["commitment"]),
        policy=material["policy"],
        signer_broker=_SigningBroker(material["keys"][TASK_ISSUER]),
        plan_store=material["store"],
        sampling_config=material["sampling"],
        branch_count=2,
        reward_config_sha256=_sha("reward-config"),
        now_unix_ns=lambda: (BASE_SECOND + 200) * 1_000_000_000,
    )
    first.sampling_plan(
        sequence=0,
        task=material["task"],
        prompt_tokens=material["prompt"],
        policy_sha256=_sha("policy"),
    )
    prompt = (11, 13) if drift == "prompt" else material["prompt"]
    policy_sha = _sha("other-policy") if drift == "policy" else _sha("policy")
    sampling = (
        RecurrentSamplingConfig(max_tokens=3) if drift == "sampling" else material["sampling"]
    )
    provider = _Provider(material["policy"], material["commitment"])
    if drift == "policy":
        provider.expected_policy_sha256 = policy_sha
    restarted = JITAdmittingVerifiedTransitionGroupProvider(
        provider=provider,
        policy=material["policy"],
        signer_broker=_NoSigningBroker(material["keys"][TASK_ISSUER]),
        plan_store=material["store"],
        sampling_config=sampling,
        branch_count=2,
        reward_config_sha256=_sha("reward-config"),
    )
    with pytest.raises(
        VerifiedTransitionProductionFactoryError,
        match="persisted_plan_runtime_mismatch",
    ):
        restarted.sampling_plan(
            sequence=0,
            task=material["task"],
            prompt_tokens=prompt,
            policy_sha256=policy_sha,
        )


def test_bound_task_delegates_behavior_but_keeps_commitment_immutable() -> None:
    source = _Task("task-0")
    commitment = {"schema": "test", "task_id": "task-0"}
    bound = ProviderBoundTrainingTask(source, commitment)
    observed = bound.verified_transition_task_commitment()
    observed["task_id"] = "tampered"
    assert bound.task_id == "task-0"
    assert bound.grade("anything") == {"correct": True}
    assert bound.verified_transition_task_commitment() == commitment


def test_factory_binds_schedule_and_constructs_only_after_live_policy_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    material = _factory_material(tmp_path)
    factory = material["factory"]
    task = material["task"]
    bound = tuple(factory.bind_training_tasks((task,)))
    assert isinstance(bound[0], ProviderBoundTrainingTask)
    assert bound[0].verified_transition_task_commitment() == material["task_document"]

    class LowLevelProvider:
        def __init__(self, **kwargs: Any) -> None:
            self.contract_sha256 = kwargs["contract"]["contract_sha256"]
            self.expected_policy_sha256 = kwargs["contract"]["initial_policy_sha256"]
            self.campaign_id = kwargs["contract"]["campaign_id"]
            self.campaign_schedule_root_sha256 = kwargs["contract"]["campaign_schedule_root_sha256"]

    monkeypatch.setattr(
        "core.learning.verified_transition_production_factory."
        "ProductionVerifiedTransitionGroupProvider",
        LowLevelProvider,
    )
    monkeypatch.setattr(
        "core.learning.verified_transition_production_factory.recurrent_policy_sha256",
        lambda _model, _spec: _sha("initial-policy"),
    )
    provider = factory.create(
        VerifiedTransitionProviderRuntime(
            model=object(),
            tokenizer=object(),
            tokenizer_trace_adapter=_TokenizerAdapter(),
            execution_spec=SimpleNamespace(
                sha256=_sha("execution"),
                branch_roles=("constructive_solution", "counterexample_search"),
            ),
            training_tasks=bound,
            output_directory=material["output_root"],
            transaction_root=material["transaction_root"],
            dataset_sha256=_sha("dataset"),
            group_size=2,
            sampling_max_tokens=2,
        )
    )
    assert provider.contract_sha256 == material["contract"]["contract_sha256"]


def test_factory_rejects_live_initial_policy_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    material = _factory_material(tmp_path)
    bound = tuple(material["factory"].bind_training_tasks((material["task"],)))
    monkeypatch.setattr(
        "core.learning.verified_transition_production_factory.recurrent_policy_sha256",
        lambda _model, _spec: _sha("substituted-policy"),
    )
    with pytest.raises(
        VerifiedTransitionProductionFactoryError,
        match="initial_policy_mismatch",
    ):
        material["factory"].create(
            VerifiedTransitionProviderRuntime(
                model=object(),
                tokenizer=object(),
                tokenizer_trace_adapter=_TokenizerAdapter(),
                execution_spec=SimpleNamespace(
                    sha256=_sha("execution"),
                    branch_roles=("constructive_solution", "counterexample_search"),
                ),
                training_tasks=bound,
                output_directory=material["output_root"],
                transaction_root=material["transaction_root"],
                dataset_sha256=_sha("dataset"),
                group_size=2,
                sampling_max_tokens=2,
            )
        )


def test_factory_rejects_in_process_self_attested_signer(tmp_path: Path) -> None:
    with pytest.raises(
        VerifiedTransitionProductionFactoryError,
        match="external_command_signers_required",
    ):
        _factory_material(tmp_path, use_in_process_signer=True)


def test_factory_rejects_shared_signer_identity_and_custody(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        VerifiedTransitionProductionFactoryError,
        match="signer_role_separation_required",
    ):
        _factory_material(tmp_path, shared_external_signer=True)


def test_factory_rejects_distinct_signers_with_shared_custody(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        VerifiedTransitionProductionFactoryError,
        match="signer_role_separation_required",
    ):
        _factory_material(tmp_path, shared_custody_only=True)


def test_factory_rejects_task_behavior_drift_with_same_task_id(
    tmp_path: Path,
) -> None:
    material = _factory_material(tmp_path)
    original = material["task"]
    drifted_answer = 'FINAL_ANSWER: {"node":999999}'
    # A task validates that its solution ends with its answer, so moving only
    # the answer raises inside the dataclass and the factory is never asked
    # anything. The drift under test is behavioural: same task_id, different
    # answer, and a solution that still agrees with it.
    drifted = replace(
        original,
        answer=drifted_answer,
        solution=original.solution[: -len(original.answer)] + drifted_answer,
    )
    assert drifted.solution.endswith(drifted.answer)
    assert drifted.task_id == original.task_id
    with pytest.raises(
        VerifiedTransitionProductionFactoryError,
        match="runtime_task_commitment_mismatch",
    ):
        material["factory"].bind_training_tasks((drifted,))


@pytest.mark.parametrize(
    ("drift", "expected"),
    [
        ("dataset", "runtime_graph_mismatch"),
        ("group_size", "runtime_graph_mismatch"),
        ("output", "runtime_graph_mismatch"),
        ("transaction", "runtime_graph_mismatch"),
    ],
)
def test_factory_rejects_live_runtime_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    expected: str,
) -> None:
    material = _factory_material(tmp_path)
    bound = tuple(material["factory"].bind_training_tasks((material["task"],)))
    monkeypatch.setattr(
        "core.learning.verified_transition_production_factory.recurrent_policy_sha256",
        lambda _model, _spec: _sha("initial-policy"),
    )
    runtime = VerifiedTransitionProviderRuntime(
        model=object(),
        tokenizer=object(),
        tokenizer_trace_adapter=_TokenizerAdapter(),
        execution_spec=SimpleNamespace(
            sha256=_sha("execution"),
            branch_roles=("constructive_solution", "counterexample_search"),
        ),
        training_tasks=bound,
        output_directory=(
            tmp_path / "other-output" if drift == "output" else material["output_root"]
        ),
        transaction_root=(
            tmp_path / "other-transactions"
            if drift == "transaction"
            else material["transaction_root"]
        ),
        dataset_sha256=(_sha("other-dataset") if drift == "dataset" else _sha("dataset")),
        group_size=3 if drift == "group_size" else 2,
        sampling_max_tokens=2,
    )
    with pytest.raises(VerifiedTransitionProductionFactoryError, match=expected):
        material["factory"].create(runtime)


def test_command_signer_broker_accepts_only_pinned_canonical_response(
    tmp_path: Path,
) -> None:
    policy, _keys, broker, script = _command_signer_material(tmp_path)
    payload = {"schema": "test.payload.v1", "value": 7}
    attestation = broker.attest(
        policy,
        role=TASK_ISSUER,
        payload=payload,
        signed_at_unix=BASE_SECOND + 200,
        purpose="test:manifest",
    )
    assert (
        verify_role_attestation(
            policy,
            attestation,
            role=TASK_ISSUER,
            expected_payload=payload,
        )["payload"]
        == payload
    )

    script.chmod(stat.S_IRWXU)
    script.write_text(script.read_text(encoding="ascii") + "# drift\n", encoding="ascii")
    with pytest.raises(
        VerifiedTransitionProductionFactoryError,
        match="executable_identity_mismatch",
    ):
        broker.attest(
            policy,
            role=TASK_ISSUER,
            payload=payload,
            signed_at_unix=BASE_SECOND + 200,
            purpose="test:manifest",
        )


def test_command_signer_uses_authenticated_detached_broker_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, _keys, broker, script = _command_signer_material(tmp_path)
    observed: dict[str, Any] = {}

    def run_brokered(command, *, cwd, stdout_path, timeout_s):
        request_index = command.index("--request-file") + 1
        request_path = Path(command[request_index])
        observed.update(
            {
                "command": command,
                "cwd": cwd,
                "request_path": request_path,
                "timeout_s": timeout_s,
            }
        )
        import subprocess

        completed = subprocess.run(
            command[:request_index - 1],
            input=request_path.read_bytes(),
            capture_output=True,
            check=False,
        )
        stdout_path.write_bytes(completed.stdout)
        return SimpleNamespace(returncode=completed.returncode)

    monkeypatch.setattr(production_factory, "broker_available", lambda: True)
    monkeypatch.setattr(
        production_factory,
        "run_brokered_process",
        run_brokered,
    )
    payload = {"schema": "test.payload.v1", "value": 9}
    attestation = broker.attest(
        policy,
        role=TASK_ISSUER,
        payload=payload,
        signed_at_unix=BASE_SECOND + 200,
        purpose="test:brokered-manifest",
    )

    assert verify_role_attestation(
        policy,
        attestation,
        role=TASK_ISSUER,
        expected_payload=payload,
    )
    assert observed["command"][0] == str(script)
    assert "--request-file" in observed["command"]
    assert not observed["request_path"].exists()
    assert observed["cwd"] == Path.cwd().resolve()


def test_external_verifier_broker_returns_replay_receipt(
    tmp_path: Path,
) -> None:
    material = _factory_material(tmp_path)
    contract = material["contract"]
    campaign_manifest = build_causal_campaign_manifest(
        campaign_id=contract["campaign_id"],
        provider_contract_sha256=contract["contract_sha256"],
        campaign_schedule_root_sha256=contract["campaign_schedule_root_sha256"],
        trust_policy_sha256=material["policy"].policy_sha256,
        initial_policy_sha256=contract["initial_policy_sha256"],
        schedule=tuple(
            CausalCampaignScheduleEntry(
                sequence=row["sequence"],
                task_id=row["task_id"],
                task_commitment_sha256=_digest(row),
            )
            for row in contract["task_schedule"]
        ),
        planned_at_unix_ns=(BASE_SECOND + 150) * 1_000_000_000,
    )
    campaign_attestation = build_role_attestation(
        material["policy"],
        role=TASK_ISSUER,
        payload=campaign_manifest,
        signed_at_unix=BASE_SECOND + 150,
        private_key=material["role_keys"][TASK_ISSUER],
    )
    VerifiedTransitionCausalCampaignLedger.create(
        contract["ledger_roots"]["campaign"],
        campaign_manifest=campaign_manifest,
        campaign_manifest_attestation=campaign_attestation,
        policy=material["policy"],
    )
    material["transaction_root"].mkdir(mode=0o700, parents=True)
    body = {
        "schema": CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA,
        "contract_sha256": material["contract"]["contract_sha256"],
        "campaign_schedule_root_sha256": material["contract"]["campaign_schedule_root_sha256"],
        "trust_policy_sha256": material["policy"].policy_sha256,
        "campaign_ledger_root": material["contract"]["ledger_roots"]["campaign"],
        "transition_artifact_root": material["contract"]["ledger_roots"]["transition_artifacts"],
        "update_journal_root": material["contract"]["ledger_roots"]["updates"],
        "transaction_root": str(material["transaction_root"]),
        "completed_groups": 0,
        "halt_reason": "preflight",
        "group_packages": [],
        "updated_replay_sequences": [],
        "created_at_unix_ns": (BASE_SECOND + 200) * 1_000_000_000,
    }
    evidence = {**body, "manifest_sha256": _digest(body)}
    receipt = material["verifier_broker"].verify_evidence_manifest(
        material["policy"],
        evidence_manifest=evidence,
        verified_at_unix=BASE_SECOND + 200,
        purpose="fixture:evidence-replay",
    )
    assert receipt["evidence_manifest_sha256"] == evidence["manifest_sha256"]
    assert receipt["verifier_identity"] == material["verifier_broker"].identity


def test_command_broker_requires_durable_policy_replay_job(
    tmp_path: Path,
) -> None:
    broker, _script, _overrides = _external_signer(
        tmp_path,
        role=EVIDENCE_VERIFIER,
        key=Ed25519PrivateKey.generate(),
    )
    request = {
        "schema": "test.durable.request.v1",
        "purpose": "test:durable-interface",
        "request_sha256": _sha("durable-interface-request"),
    }

    with pytest.raises(
        VerifiedTransitionProductionFactoryError,
        match="durable_policy_state_replay_job_unavailable",
    ):
        broker.replay_policy_states(
            request=request,
            timeout_seconds=93_600.0,
        )

    with pytest.raises(
        VerifiedTransitionProductionFactoryError,
        match="durable_evidence_verifier_purpose_invalid",
    ):
        broker.replay_policy_states(
            request={**request, "purpose": ""},
            timeout_seconds=93_600.0,
        )


def test_command_broker_delegates_only_to_frozen_policy_replay_job(
    tmp_path: Path,
) -> None:
    broker, _script, _overrides = _external_signer(
        tmp_path,
        role=EVIDENCE_VERIFIER,
        key=Ed25519PrivateKey.generate(),
    )
    request = {
        "schema": "test.durable.request.v1",
        "purpose": "test:durable-interface",
        "request_sha256": _sha("durable-interface-request"),
    }

    class Job:
        target_command = (sys.executable, "target.py", "run")
        timeout_seconds = 93_600.0
        calls: list[tuple[Any, ...]] = []

        @classmethod
        def run_file_protocol(cls, *args):
            cls.calls.append(args)
            return {
                "request_sha256": request["request_sha256"],
                "accepted": True,
            }

    broker._durable_policy_state_replay_job = Job()
    response = broker.replay_policy_states(
        request=request,
        timeout_seconds=93_600.0,
    )

    assert response["accepted"] is True
    assert Job.calls == [
        (
            request,
            Job.target_command,
            Job.timeout_seconds,
            request["purpose"],
        )
    ]


@pytest.mark.parametrize("artifact", ["release", "custody"])
def test_command_signer_rejects_root_pinned_artifact_drift(tmp_path: Path, artifact: str) -> None:
    policy, _keys, broker, _script = _command_signer_material(tmp_path)
    path = broker._release_manifest if artifact == "release" else broker._custody_evidence
    path.write_bytes(canonical_json_bytes({artifact: "substituted"}))
    with pytest.raises(
        VerifiedTransitionProductionFactoryError,
        match="artifact_identity_mismatch",
    ):
        broker.attest(
            policy,
            role=TASK_ISSUER,
            payload={"schema": "test.payload.v1", "value": 7},
            signed_at_unix=BASE_SECOND + 200,
            purpose="test:artifact-drift",
        )


def test_plan_store_rejects_writable_peer_or_digest_tampering(tmp_path: Path) -> None:
    material = _jit_material(tmp_path)
    path = material["store"]._path(0)
    path.write_bytes(b"{}")
    path.chmod(0o644)
    with pytest.raises(
        VerifiedTransitionProductionFactoryError,
        match="not_private_owned_file",
    ):
        material["store"].load(sequence=0)


def test_root_bound_launch_bundle_constructs_only_pinned_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material = _factory_material(tmp_path)

    def binding(name: str, value: Any) -> dict[str, Any]:
        payload = canonical_json_bytes(value) + b"\n"
        path = tmp_path / name
        path.write_bytes(payload)
        path.chmod(0o600)
        return {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    root = tmp_path / "campaign-root.pem"
    root.write_bytes(b"externally supplied root placeholder\n")
    root.chmod(0o600)
    root_binding = {
        "path": str(root),
        "sha256": hashlib.sha256(root.read_bytes()).hexdigest(),
        "size_bytes": root.stat().st_size,
    }
    answer_nonce = b"factory-test-answer-nonce-material" * 2
    preregistration_body = {
        "schema": "test.preregistration.v1",
        "campaign_id": "jit-provider-test",
    }
    preregistration_sha256 = hashlib.sha256(canonical_json_bytes(preregistration_body)).hexdigest()
    unsigned = {
        "schema": VERIFIED_TRANSITION_LAUNCH_BUNDLE_SCHEMA,
        "campaign_name": "jit-provider-test",
        "preregistration_contract": binding(
            "preregistration-contract.json",
            {
                **preregistration_body,
                "contract_sha256": preregistration_sha256,
            },
        ),
        "provider_contract": binding("provider-contract.json", material["contract"]),
        "provider_config": binding("provider-config.json", material["provider_config"]),
        "trust_policy": binding("trust-policy.json", material["policy"].document),
        "trust_root": root_binding,
        "campaign_ledger_root": str(material["ledger"].root),
        "signers": {
            role: {
                "identity": role_broker.identity,
                "executable": str(role_broker._executable),
                "executable_sha256": role_broker.implementation_sha256,
                "release_manifest": str(role_broker._release_manifest),
                "release_sha256": role_broker.release_sha256,
                "custody_evidence": str(role_broker._custody_evidence),
                "custody_evidence_sha256": (role_broker.custody_evidence_sha256),
                "arguments": [],
                "timeout_millis": 5_000,
                "inherited_environment_names": [],
            }
            for role, role_broker in {
                "task_issuer": material["broker"],
                "evidence_verifier": material["verifier_broker"],
            }.items()
        },
        "task_commitments": binding(
            "task-commitments.json",
            {
                "schema": "aura.verified_transition.task_commitments.v1",
                "tasks": {material["task"].task_id: material["task_document"]},
            },
        ),
        "task_answer_nonces": binding(
            "task-answer-nonces.json",
            {
                "schema": "aura.verified_transition.task_answer_nonces.v1",
                "nonces_b64": {
                    material["task"].task_id: base64.b64encode(answer_nonce).decode("ascii")
                },
            },
        ),
    }
    bundle = {
        **unsigned,
        "bundle_sha256": hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
    }
    bundle_path = tmp_path / "launch-bundle.json"
    bundle_raw = canonical_json_bytes(bundle) + b"\n"
    bundle_path.write_bytes(bundle_raw)
    bundle_path.chmod(0o600)
    monkeypatch.setattr(
        launch_bundle,
        "validate_campaign_trust_policy",
        lambda *_args, **_kwargs: material["policy"],
    )
    monkeypatch.setattr(
        launch_bundle.VerifiedTransitionCausalCampaignLedger,
        "open",
        lambda *_args, **_kwargs: material["ledger"],
    )
    components = VerifiedTransitionRuntimeComponents(
        evidence_producer=_producer,
        evidence_producer_identity="evidence-producer",
        durable_artifact_loader=_loader,
        durable_artifact_loader_identity="replay-loader",
        campaign_finalizer=_finalizer,
        campaign_finalizer_identity="campaign-finalizer",
        independent_scorer=_scorer,
        scorer_identity="independent-scorer",
        token_encoder=_encode,
        token_decoder=_decode,
        token_codec_identity="byte-codec",
    )

    factory = launch_bundle.load_verified_transition_provider_factory(
        bundle_path,
        expected_bundle_sha256=hashlib.sha256(bundle_raw).hexdigest(),
        expected_preregistration_sha256=preregistration_sha256,
        components=components,
        now_unix=BASE_SECOND + 200,
    )

    assert factory.contract_sha256 == material["contract"]["contract_sha256"]
    assert factory.ledger_roots == material["contract"]["ledger_roots"]
    original_ledger_manifest = material["ledger"].campaign_manifest()
    for field in (
        "campaign_id",
        "provider_contract_sha256",
        "campaign_schedule_root_sha256",
        "trust_policy_sha256",
        "initial_policy_sha256",
    ):
        drifted_manifest = dict(original_ledger_manifest)
        drifted_manifest[field] = (
            "different-campaign" if field == "campaign_id" else _sha(f"drifted-{field}")
        )
        monkeypatch.setattr(
            material["ledger"],
            "campaign_manifest",
            lambda manifest=drifted_manifest: dict(manifest),
        )
        with pytest.raises(
            VerifiedTransitionLaunchBundleError,
            match="launch_causal_campaign_identity_mismatch",
        ):
            launch_bundle.load_verified_transition_provider_factory(
                bundle_path,
                expected_bundle_sha256=hashlib.sha256(bundle_raw).hexdigest(),
                expected_preregistration_sha256=preregistration_sha256,
                components=components,
                now_unix=BASE_SECOND + 200,
            )
    monkeypatch.setattr(
        material["ledger"],
        "campaign_manifest",
        lambda: dict(original_ledger_manifest),
    )
    with pytest.raises(
        VerifiedTransitionLaunchBundleError,
        match="launch_scorer_identity_mismatch",
    ):
        launch_bundle.load_verified_transition_provider_factory(
            bundle_path,
            expected_bundle_sha256=hashlib.sha256(bundle_raw).hexdigest(),
            expected_preregistration_sha256=preregistration_sha256,
            components=replace(components, scorer_identity="substituted-scorer"),
            now_unix=BASE_SECOND + 200,
        )

    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(
        VerifiedTransitionLaunchBundleError,
        match="path_symlink_rejected",
    ):
        launch_bundle.load_verified_transition_provider_factory(
            linked_parent / bundle_path.name,
            expected_bundle_sha256=hashlib.sha256(bundle_raw).hexdigest(),
            expected_preregistration_sha256=preregistration_sha256,
            components=components,
            now_unix=BASE_SECOND + 200,
        )

    with pytest.raises(
        VerifiedTransitionLaunchBundleError,
        match="external_digest_mismatch",
    ):
        launch_bundle.load_verified_transition_provider_factory(
            bundle_path,
            expected_bundle_sha256="0" * 64,
            expected_preregistration_sha256=preregistration_sha256,
            components=components,
            now_unix=BASE_SECOND + 200,
        )

    drifted_preregistration_body = {
        **preregistration_body,
        "campaign_id": "different-campaign",
    }
    drifted_preregistration_sha256 = hashlib.sha256(
        canonical_json_bytes(drifted_preregistration_body)
    ).hexdigest()
    drifted_unsigned = {
        **unsigned,
        "preregistration_contract": binding(
            "drifted-preregistration-contract.json",
            {
                **drifted_preregistration_body,
                "contract_sha256": drifted_preregistration_sha256,
            },
        ),
    }
    drifted_bundle = {
        **drifted_unsigned,
        "bundle_sha256": hashlib.sha256(canonical_json_bytes(drifted_unsigned)).hexdigest(),
    }
    drifted_bundle_path = tmp_path / "drifted-launch-bundle.json"
    drifted_bundle_raw = canonical_json_bytes(drifted_bundle) + b"\n"
    drifted_bundle_path.write_bytes(drifted_bundle_raw)
    drifted_bundle_path.chmod(0o600)
    with pytest.raises(
        VerifiedTransitionLaunchBundleError,
        match="launch_preregistration_campaign_mismatch",
    ):
        launch_bundle.load_verified_transition_provider_factory(
            drifted_bundle_path,
            expected_bundle_sha256=hashlib.sha256(drifted_bundle_raw).hexdigest(),
            expected_preregistration_sha256=drifted_preregistration_sha256,
            components=components,
            now_unix=BASE_SECOND + 200,
        )
