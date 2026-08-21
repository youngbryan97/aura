"""Contracts for the resident worker's boot-scoped capture signer."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.worker_capture_identity import (
    WorkerCaptureIdentityError,
    build_worker_capture_identity,
    build_worker_capture_launch_authority,
    build_worker_capture_origin_binding,
    validate_worker_capture_identity,
    validate_worker_capture_origin_binding,
)

NOW = 10_000


def _private(label: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(label.encode()).digest())


def _launch_case(*, label: str = "honest"):
    authority = build_worker_capture_launch_authority(
        issued_at_unix=NOW,
        lifetime_s=120,
        private_key=_private(f"supervisor:{label}"),
        challenge_nonce=hashlib.sha256(f"nonce:{label}".encode()).digest(),
        challenge_id=hashlib.sha256(f"challenge:{label}".encode()).hexdigest()[:32],
    )
    identity = build_worker_capture_identity(
        worker_boot_id="a" * 32,
        worker_pid=4242,
        private_key=_private(f"worker:{label}"),
        launch_challenge=authority.challenge,
        now_unix=NOW + 1,
    )
    binding = build_worker_capture_origin_binding(
        authority,
        identity.public_identity,
        attested_at_unix=NOW + 2,
        expected_worker_pid=4242,
    )
    return authority, identity, binding


def test_worker_capture_identity_round_trip_is_boot_scoped():
    identity = build_worker_capture_identity(
        worker_boot_id="a" * 32,
        worker_pid=4242,
    )

    assert validate_worker_capture_identity(identity.public_identity) == (identity.public_identity)
    assert identity.public_identity["worker_boot_id"] == "a" * 32
    assert identity.public_identity["worker_pid"] == 4242


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    (
        ("worker_boot_id", "b" * 32, "hash_mismatch"),
        ("worker_pid", 4243, "hash_mismatch"),
        ("public_key_b64", "A" * 44, "hash_mismatch"),
        ("signature_b64", "A" * 88, "hash_mismatch"),
    ),
)
def test_worker_capture_identity_rejects_public_tampering(
    field: str,
    replacement,
    error: str,
):
    identity = build_worker_capture_identity(
        worker_boot_id="a" * 32,
        worker_pid=4242,
    )
    attacked = deepcopy(identity.public_identity)
    attacked[field] = replacement

    with pytest.raises(WorkerCaptureIdentityError, match=error):
        validate_worker_capture_identity(attacked)


def test_worker_capture_identity_rejects_extra_fields():
    identity = build_worker_capture_identity(
        worker_boot_id="a" * 32,
        worker_pid=4242,
    )
    attacked = {**identity.public_identity, "private_key": "leak"}

    with pytest.raises(WorkerCaptureIdentityError, match="identity_fields"):
        validate_worker_capture_identity(attacked)


def test_worker_capture_launch_binding_round_trip_uses_expected_parent_key():
    authority, identity, binding = _launch_case()

    assert validate_worker_capture_origin_binding(
        binding,
        expected_supervisor_public_key=authority.private_key.public_key(),
        now_unix=NOW + 3,
    ) == binding
    assert binding["worker_identity"] == identity.public_identity
    assert binding["launch_attestation"]["worker_pid"] == 4242


def test_worker_capture_launch_binding_rejects_self_rooted_rogue_supervisor():
    honest, _, _ = _launch_case(label="honest-root")
    _, _, rogue_binding = _launch_case(label="rogue-root")

    with pytest.raises(WorkerCaptureIdentityError, match="public_key_mismatch"):
        validate_worker_capture_origin_binding(
            rogue_binding,
            expected_supervisor_public_key=honest.private_key.public_key(),
            now_unix=NOW + 3,
        )


def test_worker_capture_launch_binding_rejects_cross_worker_substitution():
    authority, _, binding = _launch_case()
    other_identity = build_worker_capture_identity(
        worker_boot_id="b" * 32,
        worker_pid=4243,
        private_key=_private("worker:other"),
        launch_challenge=authority.challenge,
        now_unix=NOW + 1,
    )
    attacked = deepcopy(binding)
    attacked["worker_identity"] = other_identity.public_identity
    body = {name: value for name, value in attacked.items() if name != "binding_sha256"}
    attacked["binding_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()

    with pytest.raises(WorkerCaptureIdentityError, match="attestation_invalid"):
        validate_worker_capture_origin_binding(
            attacked,
            expected_supervisor_public_key=authority.private_key.public_key(),
            now_unix=NOW + 3,
        )


@pytest.mark.parametrize(
    ("section", "field", "replacement", "error"),
    (
        ("launch_challenge", "challenge_id", "f" * 32, "challenge_hash_mismatch"),
        ("launch_attestation", "worker_pid", 9001, "attestation_invalid"),
        ("launch_attestation", "signature_b64", "A" * 88, "attestation_hash_mismatch"),
    ),
)
def test_worker_capture_launch_binding_rejects_rehashed_inner_tampering(
    section: str,
    field: str,
    replacement,
    error: str,
):
    authority, _, binding = _launch_case()
    attacked = deepcopy(binding)
    attacked[section][field] = replacement
    body = {name: value for name, value in attacked.items() if name != "binding_sha256"}
    attacked["binding_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()

    with pytest.raises(WorkerCaptureIdentityError, match=error):
        validate_worker_capture_origin_binding(
            attacked,
            expected_supervisor_public_key=authority.private_key.public_key(),
            now_unix=NOW + 3,
        )


def test_worker_capture_launch_current_admission_expires_but_historical_replay_survives():
    authority, _, binding = _launch_case()

    with pytest.raises(WorkerCaptureIdentityError, match="challenge_not_current"):
        validate_worker_capture_origin_binding(
            binding,
            expected_supervisor_public_key=authority.private_key.public_key(),
            now_unix=NOW + 121,
        )
    assert validate_worker_capture_origin_binding(
        binding,
        expected_supervisor_public_key=authority.private_key.public_key(),
    ) == binding


def test_worker_capture_parent_refuses_wrong_pid_and_wrong_challenge():
    authority, identity, _ = _launch_case()
    with pytest.raises(WorkerCaptureIdentityError, match="identity_mismatch"):
        build_worker_capture_origin_binding(
            authority,
            identity.public_identity,
            attested_at_unix=NOW + 2,
            expected_worker_pid=4243,
        )

    other_authority, other_identity, _ = _launch_case(label="other-challenge")
    with pytest.raises(WorkerCaptureIdentityError, match="identity_mismatch"):
        build_worker_capture_origin_binding(
            authority,
            other_identity.public_identity,
            attested_at_unix=NOW + 2,
            expected_worker_pid=4242,
        )
    assert other_authority.challenge != authority.challenge


def test_mlx_parent_attests_spawned_worker_before_exposing_identity():
    from core.brain.llm.mlx_client import MLXLocalClient

    authority, identity, _ = _launch_case()
    client = object.__new__(MLXLocalClient)
    client._worker_capture_launch_authority = authority
    client._process = SimpleNamespace(pid=4242)
    client._worker_identity = {}

    bound = client._attest_worker_capture_origin(
        {"worker_action_capture_identity": identity.public_identity},
        attested_at_unix=NOW + 2,
    )
    client._worker_identity = bound

    assert validate_worker_capture_origin_binding(
        bound["worker_action_capture_origin_binding"],
        expected_supervisor_public_key=authority.private_key.public_key(),
        now_unix=NOW + 3,
    )
    supervisor_raw = client.get_worker_capture_supervisor_public_key()
    assert len(supervisor_raw) == 32
    assert hashlib.sha256(supervisor_raw).hexdigest() == authority.challenge[
        "supervisor_key_id"
    ]


def test_mlx_parent_reuses_early_binding_after_launch_challenge_expires():
    from core.brain.llm.mlx_client import MLXLocalClient

    authority, identity, _ = _launch_case()
    client = object.__new__(MLXLocalClient)
    client._worker_capture_launch_authority = authority
    client._worker_capture_origin_binding = {}
    client._process = SimpleNamespace(pid=4242)

    bootstrap = client._accept_worker_capture_bootstrap(
        identity.public_identity,
        attested_at_unix=NOW + 2,
    )
    bound = client._attest_worker_capture_origin(
        {"worker_action_capture_identity": identity.public_identity},
        # A fresh attestation would now be refused. The early binding remains
        # valid historical evidence for the same process and capture key.
        attested_at_unix=NOW + 121,
    )

    assert bound["worker_action_capture_origin_binding"] == bootstrap
    assert validate_worker_capture_origin_binding(
        bound["worker_action_capture_origin_binding"],
        expected_supervisor_public_key=authority.private_key.public_key(),
    ) == bootstrap


def test_mlx_parent_refuses_ready_identity_that_differs_from_bootstrap():
    from core.brain.llm.mlx_client import MLXLocalClient

    authority, identity, _ = _launch_case()
    substituted = build_worker_capture_identity(
        worker_boot_id="b" * 32,
        worker_pid=4242,
        private_key=_private("worker:substituted"),
        launch_challenge=authority.challenge,
        now_unix=NOW + 1,
    )
    client = object.__new__(MLXLocalClient)
    client._worker_capture_launch_authority = authority
    client._worker_capture_origin_binding = {}
    client._process = SimpleNamespace(pid=4242)
    client._accept_worker_capture_bootstrap(
        identity.public_identity,
        attested_at_unix=NOW + 2,
    )

    with pytest.raises(ValueError, match="bootstrap_ready_identity_mismatch"):
        client._attest_worker_capture_origin(
            {"worker_action_capture_identity": substituted.public_identity},
            attested_at_unix=NOW + 3,
        )


def test_mlx_parent_refuses_identity_from_a_different_child_pid():
    from core.brain.llm.mlx_client import MLXLocalClient

    authority, identity, _ = _launch_case()
    client = object.__new__(MLXLocalClient)
    client._worker_capture_launch_authority = authority
    client._process = SimpleNamespace(pid=4243)

    with pytest.raises(WorkerCaptureIdentityError, match="identity_mismatch"):
        client._attest_worker_capture_origin(
            {"worker_action_capture_identity": identity.public_identity},
            attested_at_unix=NOW + 2,
        )
