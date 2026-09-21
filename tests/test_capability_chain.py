"""Adversarial tests for the Will→sink capability chain.

The mandate this closes: "Fabricated, altered, refused, expired, replayed, or
domain-mismatched capabilities must fail at the moment of execution." There is
one test per named failure mode, plus the structural property that makes the
chain worth having (a sink cannot mint).

These tests are deliberately adversarial rather than confirmatory: each one
tries to get a sink to execute without legitimate authority.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from core.governance.capability_chain import (
    APPROVING_OUTCOMES,
    CapabilityDenial,
    CapabilityViolation,
    SignedCapability,
    attach_capability,
    capability_chain_status,
    compute_action_digest,
    enforce_capability,
    get_capability_issuer,
    get_capability_verifier,
    get_nonce_ledger,
    issuer_is_asymmetric,
    reset_capability_chain,
)


@pytest.fixture(autouse=True)
def _isolated_keys(tmp_path, monkeypatch):
    """Never touch the live key material or nonce ledger."""
    monkeypatch.setenv("AURA_CAPABILITY_KEY_DIR", str(tmp_path / "keys"))
    reset_capability_chain()
    yield
    reset_capability_chain()


class _Decision:
    """Stand-in for a WillDecision (duck-typed, as the issuer expects)."""

    def __init__(self, outcome="proceed", domain="tool_execution", receipt_id="r-1",
                 constraints=()):
        self.outcome = outcome
        self.domain = domain
        self.receipt_id = receipt_id
        self.constraints = list(constraints)


def _issue(action="shell_command", payload=None, **kw):
    return get_capability_issuer().issue_from_decision(
        _Decision(**kw), action=action, payload=payload
    )


# ---------------------------------------------------------------------------
# The happy path must actually work, or the failure tests prove nothing.
# ---------------------------------------------------------------------------


def test_legitimate_capability_authorizes_its_own_action():
    payload = {"cmd": "ls"}
    cap = _issue(action="shell_command", payload=payload)
    ctx = attach_capability({}, cap)

    verified = enforce_capability(
        ctx, sink="test", domain="tool_execution", action="shell_command", payload=payload
    )
    assert verified.capability_id == cap.capability_id
    assert verified.receipt_id == "r-1"


# ---------------------------------------------------------------------------
# 1. Fabricated
# ---------------------------------------------------------------------------


def test_fabricated_capability_fails():
    """A hand-built capability with no real signature must not execute."""
    now = time.time()
    forged = SignedCapability(
        capability_id="cap-forged",
        schema_version=1,
        outcome="proceed",
        domain="tool_execution",
        action_digest=compute_action_digest("shell_command", {"cmd": "rm -rf /"}),
        issuer="UnifiedWill",
        key_id="ed25519-deadbeefdeadbeef",
        nonce="f" * 32,
        receipt_id="r-fake",
        scope="",
        constraints=(),
        issued_at=now,
        expires_at=now + 300,
        signature="00" * 64,
    )
    ctx = attach_capability({}, forged)
    with pytest.raises(CapabilityViolation) as exc:
        enforce_capability(
            ctx, sink="test", domain="tool_execution",
            action="shell_command", payload={"cmd": "rm -rf /"},
        )
    assert exc.value.denial is CapabilityDenial.BAD_SIGNATURE


def test_fabricated_governance_context_fails():
    """The exact old bypass: asserting verification in the context dict.

    ``ctx["_capability_token_verified"] = True`` used to be sufficient. It must
    now be inert — the sink authenticates a signature, not a claim.
    """
    ctx = {
        "capability_token_id": "anything",
        "_capability_token_verified": True,
        "governance_approved": True,
        "will_receipt_id": "r-i-made-this-up",
    }
    with pytest.raises(CapabilityViolation) as exc:
        enforce_capability(ctx, sink="test", domain="tool_execution", action="shell_command")
    assert exc.value.denial is CapabilityDenial.MISSING


def test_capability_signed_by_a_foreign_key_fails():
    """A capability minted under a key this host does not hold is rejected."""
    cap = _issue()
    impostor = replace(cap, key_id="ed25519-0000000000000000")
    ctx = attach_capability({}, impostor)
    with pytest.raises(CapabilityViolation) as exc:
        enforce_capability(ctx, sink="test", domain="tool_execution", action="shell_command")
    assert exc.value.denial is CapabilityDenial.BAD_SIGNATURE


# ---------------------------------------------------------------------------
# 2. Altered
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("outcome", "proceed"),
        ("domain", "file_write"),
        ("action_digest", compute_action_digest("other", None)),
        ("issuer", "NotTheWill"),
        ("receipt_id", "r-tampered"),
        ("scope", "escalated"),
        ("expires_at", time.time() + 99999),
        ("nonce", "a" * 32),
        ("capability_id", "cap-swapped"),
    ],
    # Named by the field. The expiry is read off the clock, and an id built
    # from the value differed between xdist workers, which then refused to
    # run anything because they had not collected the same tests.
    ids=lambda item: item if isinstance(item, str) and item.isidentifier() else "value",
)
def test_altering_any_signed_field_fails(field, value):
    """Every field in the payload is covered by the signature."""
    cap = _issue(action="shell_command", payload={"cmd": "ls"}, outcome="constrain")
    tampered = replace(cap, **{field: value})
    ctx = attach_capability({}, tampered)

    result = get_capability_verifier().verify(ctx["signed_capability"])
    assert not result.ok
    assert result.denial is CapabilityDenial.BAD_SIGNATURE, (
        f"altering {field!r} did not invalidate the signature"
    )


def test_constraints_are_covered_by_the_signature():
    """Constraints are authority-bearing; stripping them must break the seal."""
    cap = _issue(constraints=["no_network", "read_only"])
    stripped = replace(cap, constraints=())
    result = get_capability_verifier().verify(stripped)
    assert not result.ok
    assert result.denial is CapabilityDenial.BAD_SIGNATURE


# ---------------------------------------------------------------------------
# 3. Refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["refuse", "defer"])
def test_refused_decision_never_mints_a_capability(outcome):
    """A refusal cannot become authority. The issuer refuses at the source."""
    with pytest.raises(CapabilityViolation) as exc:
        get_capability_issuer().issue_from_decision(
            _Decision(outcome=outcome), action="shell_command"
        )
    assert exc.value.denial is CapabilityDenial.NOT_APPROVED


def test_refused_outcome_is_rejected_at_the_sink_even_if_signed():
    """Defence in depth: even a *validly signed* refusal is not authority.

    Signed with the real key via the internal path, so this tests the sink's
    own outcome check rather than the issuer's.
    """
    from core.governance.capability_chain import _sign

    cap = _issue()
    refused = replace(cap, outcome="refuse", signature="")
    refused = replace(refused, signature=_sign(refused.signing_payload()))

    # The signature is genuine...
    from core.governance.capability_chain import _verify_signature

    assert _verify_signature(refused.signing_payload(), refused.signature, refused.key_id)

    # ...and it still must not authorize anything.
    result = get_capability_verifier().verify(refused)
    assert not result.ok
    assert result.denial is CapabilityDenial.NOT_APPROVED


def test_only_approving_outcomes_are_accepted():
    assert APPROVING_OUTCOMES == {"proceed", "constrain", "critical"}
    for outcome in ("refuse", "defer"):
        assert outcome not in APPROVING_OUTCOMES


# ---------------------------------------------------------------------------
# 4. Expired
# ---------------------------------------------------------------------------


def test_expired_capability_fails():
    cap = _issue()
    result = get_capability_verifier().verify(cap, now=cap.expires_at + 0.001)
    assert not result.ok
    assert result.denial is CapabilityDenial.EXPIRED


def test_expiry_is_enforced_at_execution_not_issue(monkeypatch):
    """A capability valid at plan time must still fail if it ages out.

    Authority is checked at the sink, at the moment of execution — so a grant
    that was legitimate when planned is worthless once it expires in transit.
    """
    cap = get_capability_issuer().issue_from_decision(
        _Decision(), action="shell_command", ttl_s=1.0
    )
    ctx = attach_capability({}, cap)

    # Valid at plan time.
    assert get_capability_verifier().verify(cap, consume=False).ok

    # The clock moves past expiry before the sink runs.
    import core.governance.capability_chain as chain

    monkeypatch.setattr(chain.time, "time", lambda: cap.expires_at + 1.0)
    with pytest.raises(CapabilityViolation) as exc:
        enforce_capability(
            ctx, sink="test", domain="tool_execution", action="shell_command"
        )
    assert exc.value.denial is CapabilityDenial.EXPIRED


def test_ttl_is_capped():
    """A caller cannot request an effectively immortal capability."""
    from core.governance.capability_chain import MAX_TTL_S

    cap = get_capability_issuer().issue_from_decision(
        _Decision(), action="x", ttl_s=10 ** 9
    )
    assert cap.expires_at - cap.issued_at <= MAX_TTL_S + 0.001


def test_capability_from_the_future_fails():
    cap = _issue()
    result = get_capability_verifier().verify(cap, now=cap.issued_at - 60)
    assert not result.ok
    assert result.denial is CapabilityDenial.NOT_YET_VALID


# ---------------------------------------------------------------------------
# 5. Replayed
# ---------------------------------------------------------------------------


def test_replayed_capability_fails():
    """Single use. The second presentation of the same grant is a replay."""
    payload = {"cmd": "ls"}
    cap = _issue(payload=payload)
    ctx = attach_capability({}, cap)

    enforce_capability(
        ctx, sink="test", domain="tool_execution", action="shell_command", payload=payload
    )
    with pytest.raises(CapabilityViolation) as exc:
        enforce_capability(
            ctx, sink="test", domain="tool_execution", action="shell_command", payload=payload
        )
    assert exc.value.denial is CapabilityDenial.REPLAYED


def test_a_failed_check_does_not_burn_the_nonce():
    """A domain typo must not consume real authority.

    Nonce consumption happens last precisely so that a capability rejected for
    some other reason remains usable for its legitimate call.
    """
    payload = {"cmd": "ls"}
    cap = _issue(payload=payload)
    ctx = attach_capability({}, cap)

    with pytest.raises(CapabilityViolation):
        enforce_capability(
            ctx, sink="test", domain="file_write", action="shell_command", payload=payload
        )
    assert not get_nonce_ledger().seen(cap.nonce)

    # The legitimate call still works.
    enforce_capability(
        ctx, sink="test", domain="tool_execution", action="shell_command", payload=payload
    )


def test_nonce_ledger_survives_restart(tmp_path):
    """Replay must not become possible by restarting the process."""
    from core.governance.capability_chain import NonceLedger

    path = tmp_path / "nonces.json"
    ledger = NonceLedger(path=path)
    assert ledger.consume("nonce-abc", time.time() + 300)

    reborn = NonceLedger(path=path)
    assert reborn.seen("nonce-abc")
    assert not reborn.consume("nonce-abc", time.time() + 300)


def test_nonce_ledger_forgets_expired_nonces(tmp_path):
    """Expired nonces buy nothing — the expiry check already rejects them."""
    from core.governance.capability_chain import NonceLedger

    path = tmp_path / "nonces.json"
    ledger = NonceLedger(path=path)
    ledger.consume("old", time.time() - 1)
    ledger.flush()

    reborn = NonceLedger(path=path)
    assert not reborn.seen("old")


def test_two_ledger_instances_cannot_both_consume_one_nonce(tmp_path):
    """The process-local mutex and file lock close the cross-instance race."""
    from core.governance.capability_chain import NonceLedger

    path = tmp_path / "nonces.json"
    ledgers = (NonceLedger(path=path), NonceLedger(path=path))
    expires_at = time.time() + 300
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda ledger: ledger.consume("one-use", expires_at), ledgers))

    assert sorted(results) == [False, True]
    assert NonceLedger(path=path).seen("one-use")


def test_corrupt_nonce_ledger_fails_closed_without_overwrite(tmp_path):
    from core.governance.capability_chain import NonceLedger

    path = tmp_path / "nonces.json"
    path.write_bytes(b'{"nonces":{"prior":')
    before = path.read_bytes()

    ledger = NonceLedger(path=path)
    accepted, reason = ledger.consume_with_reason("new", time.time() + 300)

    assert accepted is False
    assert reason
    assert ledger.status()["healthy"] is False
    assert path.read_bytes() == before


def test_nonce_ledger_rejects_duplicate_json_keys(tmp_path):
    from core.governance.capability_chain import NonceLedger

    path = tmp_path / "nonces.json"
    path.write_text(
        '{"nonces":{"same":19999999999},"nonces":{"other":19999999999}}',
        encoding="utf-8",
    )
    accepted, reason = NonceLedger(path=path).consume_with_reason(
        "fresh", time.time() + 300
    )
    assert accepted is False
    assert "duplicate JSON key" in str(reason)


def test_nonce_persistence_failure_is_not_mislabeled_as_replay(monkeypatch):
    import core.governance.capability_chain as chain

    cap = _issue()
    monkeypatch.setattr(
        chain,
        "atomic_write_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk unavailable")),
    )
    result = get_capability_verifier().verify(cap)
    assert result.ok is False
    assert result.denial is CapabilityDenial.LEDGER_UNAVAILABLE
    assert "disk unavailable" in result.detail


# ---------------------------------------------------------------------------
# 6. Domain mismatch
# ---------------------------------------------------------------------------


def test_domain_mismatched_capability_fails():
    """A grant for one domain cannot be spent in another."""
    cap = _issue(domain="reflection")
    ctx = attach_capability({}, cap)
    with pytest.raises(CapabilityViolation) as exc:
        enforce_capability(ctx, sink="test", domain="file_write", action="shell_command")
    assert exc.value.denial is CapabilityDenial.DOMAIN_MISMATCH


def test_domain_comparison_accepts_enum_or_str():
    from core.will import ActionDomain

    cap = _issue(domain="tool_execution")
    assert get_capability_verifier().verify(
        cap, expected_domain=ActionDomain.TOOL_EXECUTION, consume=False
    ).ok


# ---------------------------------------------------------------------------
# 7. Action mismatch — the confused-deputy case
# ---------------------------------------------------------------------------


def test_capability_for_one_action_cannot_run_another():
    """The named case: a read_file grant must not execute a shell command."""
    cap = _issue(action="read_file", payload={"path": "/etc/hosts"})
    ctx = attach_capability({}, cap)
    with pytest.raises(CapabilityViolation) as exc:
        enforce_capability(
            ctx, sink="test", domain="tool_execution",
            action="shell_command", payload={"cmd": "rm -rf /"},
        )
    assert exc.value.denial is CapabilityDenial.ACTION_MISMATCH


def test_capability_is_bound_to_its_parameters():
    """Same action, different arguments, is a different action."""
    cap = _issue(action="shell_command", payload={"cmd": "ls"})
    ctx = attach_capability({}, cap)
    with pytest.raises(CapabilityViolation) as exc:
        enforce_capability(
            ctx, sink="test", domain="tool_execution",
            action="shell_command", payload={"cmd": "curl evil.example"},
        )
    assert exc.value.denial is CapabilityDenial.ACTION_MISMATCH


def test_action_digest_is_stable_and_order_independent():
    a = compute_action_digest("t", {"b": 1, "a": [1, 2], "c": {"x": None}})
    b = compute_action_digest("t", {"c": {"x": None}, "a": [1, 2], "b": 1})
    assert a == b
    assert a != compute_action_digest("t", {"b": 2, "a": [1, 2], "c": {"x": None}})
    # List order is meaningful and must not be normalized away.
    assert compute_action_digest("t", [1, 2]) != compute_action_digest("t", [2, 1])


def test_action_digest_does_not_depend_on_memory_addresses():
    """Unserializable params must digest stably across instances."""

    class Opaque:
        pass

    assert compute_action_digest("t", Opaque()) == compute_action_digest("t", Opaque())


# ---------------------------------------------------------------------------
# 8. Revocation
# ---------------------------------------------------------------------------


def test_revoked_capability_fails():
    cap = _issue()
    get_capability_issuer().revoke(cap.capability_id)
    result = get_capability_verifier().verify(cap)
    assert not result.ok
    assert result.denial is CapabilityDenial.REVOKED


# ---------------------------------------------------------------------------
# Malformed / transport
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"capability_id": "x"},
        {"not": "a capability"},
        "a string",
        42,
        [],
    ],
)
def test_malformed_capabilities_fail(bad):
    result = get_capability_verifier().verify(bad)
    assert not result.ok
    assert result.denial in {CapabilityDenial.MALFORMED, CapabilityDenial.MISSING}


def test_capability_survives_dict_roundtrip():
    """Capabilities cross process/serialization boundaries intact."""
    cap = _issue(payload={"a": 1})
    restored = SignedCapability.from_dict(cap.to_dict())
    assert restored == cap
    assert get_capability_verifier().verify(restored, consume=False).ok


def test_schema_version_mismatch_fails():
    cap = _issue()
    result = get_capability_verifier().verify(replace(cap, schema_version=99))
    assert not result.ok
    assert result.denial is CapabilityDenial.SCHEMA_MISMATCH


# ---------------------------------------------------------------------------
# The structural property
# ---------------------------------------------------------------------------


def test_sink_verifier_cannot_mint():
    """The invariant the old token system could not offer.

    Under Ed25519 the verifier holds only a public key. It exposes no minting
    surface at all — not a private method, not a helper.
    """
    verifier = get_capability_verifier()
    for name in dir(verifier):
        assert not any(
            mint in name.lower() for mint in ("issue", "mint", "sign", "generate")
        ), f"CapabilityVerifier exposes a minting-shaped attribute: {name}"


def test_default_key_is_asymmetric():
    """If this fails, the chain is in its degraded symmetric mode."""
    assert issuer_is_asymmetric(), (
        "capability chain fell back to HMAC — sinks can mint; "
        f"status={capability_chain_status()}"
    )


def test_concurrent_key_initializers_converge_on_one_identity(tmp_path):
    from cryptography.hazmat.primitives import serialization

    from core.governance.capability_chain import _KeyMaterial

    with ThreadPoolExecutor(max_workers=4) as pool:
        materials = list(pool.map(lambda _index: _KeyMaterial._load_or_create_ed25519(True), range(4)))

    public_keys = {
        private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        for private, persisted in materials
        if persisted
    }
    assert len(public_keys) == 1
    assert os.stat(tmp_path / "keys" / "will_capability_ed25519_priv.pem").st_mode & 0o077 == 0


def test_malformed_private_key_is_preserved_and_reported_as_ephemeral(tmp_path):
    key_dir = tmp_path / "keys"
    key_dir.mkdir(mode=0o700)
    key_path = key_dir / "will_capability_ed25519_priv.pem"
    key_path.write_bytes(b"not a private key")
    key_path.chmod(0o600)
    before = key_path.read_bytes()

    status = capability_chain_status()

    assert status["asymmetric"] is True
    assert status["keys_persisted"] is False
    assert status["authority_durable"] is False
    assert status["degraded"] is True
    assert "ephemeral" in status["note"].lower()
    assert key_path.read_bytes() == before


def test_status_reports_degradation_honestly(monkeypatch):
    monkeypatch.setenv("AURA_CAPABILITY_FORCE_HMAC", "1")
    reset_capability_chain()
    status = capability_chain_status()
    assert status["degraded"] is True
    assert status["asymmetric"] is False
    assert "DEGRADED" in status["note"]


def test_hmac_fallback_still_defeats_fabrication(monkeypatch):
    """The degraded mode is weaker, not broken."""
    monkeypatch.setenv("AURA_CAPABILITY_FORCE_HMAC", "1")
    reset_capability_chain()
    cap = _issue()
    assert cap.key_id.startswith("hmac-")
    assert get_capability_verifier().verify(cap, consume=False).ok
    assert not get_capability_verifier().verify(replace(cap, outcome="proceed", scope="x")).ok


def test_signatures_are_domain_separated():
    """A signature over the same bytes in another context must not verify here."""
    from core.governance.capability_chain import _SIGNING_TAG, _KeyMaterial

    cap = _issue()
    keys = _KeyMaterial.load()
    naked = cap.signing_payload()[len(_SIGNING_TAG):]
    if keys["asymmetric"]:
        foreign_sig = keys["private"].sign(naked).hex()
    else:
        import hashlib as _h
        import hmac as _hm

        foreign_sig = _hm.new(keys["private"], naked, _h.sha256).hexdigest()

    result = get_capability_verifier().verify(replace(cap, signature=foreign_sig))
    assert not result.ok
    assert result.denial is CapabilityDenial.BAD_SIGNATURE


def test_concurrent_initializers_survive_publish_once_cleanup(tmp_path, monkeypatch):
    """Concurrent first boots converge on the one published signing identity."""
    from cryptography.hazmat.primitives import serialization

    from core.governance.capability_chain import _KeyMaterial

    workers = 16
    monkeypatch.setenv("AURA_CAPABILITY_KEY_DIR", str(tmp_path / "keys"))
    reset_capability_chain()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        materials = list(
            pool.map(lambda _i: _KeyMaterial._load_or_create_ed25519(True), range(workers))
        )

    assert all(persisted for _private, persisted in materials)
    identities = {
        private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        for private, persisted in materials
        if persisted
    }
    assert len(identities) == 1, (
        f"{workers} concurrent initializers produced {len(identities)} distinct "
        "signing identities — capabilities minted under one would not verify "
        "under another"
    )


def test_private_material_accepts_only_publish_link_cleanup(tmp_path, monkeypatch):
    """Unlinking the publisher's second hard link cannot invalidate safe bytes."""
    import core.governance.capability_chain as chain

    payload = b"the winner's key material"
    staged = tmp_path / "publisher-temp"
    path = tmp_path / "published.pem"
    staged.write_bytes(payload)
    os.chmod(staged, 0o600)
    os.link(staged, path)

    real_read = chain.os.read
    cleaned = False

    def _unlink_staging_during_read(fd, size):
        nonlocal cleaned
        if not cleaned:
            staged.unlink()
            cleaned = True
        return real_read(fd, size)

    monkeypatch.setattr(chain.os, "read", _unlink_staging_during_read)

    assert chain._read_private_material(path) == payload
    assert path.stat().st_nlink == 1


def test_private_material_refuses_permission_change_during_read(tmp_path, monkeypatch):
    import core.governance.capability_chain as chain

    path = tmp_path / "changing-mode.pem"
    path.write_bytes(b"private material")
    os.chmod(path, 0o600)
    real_read = chain.os.read
    changed = False

    def _change_mode_during_read(fd, size):
        nonlocal changed
        if not changed:
            os.chmod(path, 0o640)
            changed = True
        return real_read(fd, size)

    monkeypatch.setattr(chain.os, "read", _change_mode_during_read)
    try:
        with pytest.raises(RuntimeError, match="changed while reading"):
            chain._read_private_material(path)
    finally:
        os.chmod(path, 0o600)


def test_private_material_refuses_path_replacement_during_read(tmp_path, monkeypatch):
    import core.governance.capability_chain as chain

    path = tmp_path / "identity.pem"
    replacement = tmp_path / "replacement.pem"
    path.write_bytes(b"original private material")
    replacement.write_bytes(b"replacement private data")
    os.chmod(path, 0o600)
    os.chmod(replacement, 0o600)
    real_read = chain.os.read
    replaced = False

    def _replace_path_during_read(fd, size):
        nonlocal replaced
        if not replaced:
            os.replace(replacement, path)
            replaced = True
        return real_read(fd, size)

    monkeypatch.setattr(chain.os, "read", _replace_path_during_read)
    with pytest.raises(RuntimeError, match="changed while reading"):
        chain._read_private_material(path)


def test_the_public_key_is_published_once_and_never_from_the_loop(monkeypatch):
    """LIVE 2026-09-21: every load rewrote the public key, fsync and all, from
    a boot coroutine. A key already on disk unchanged is not written again."""
    import asyncio
    import threading

    import core.governance.capability_chain as chain

    if not chain._ED25519_AVAILABLE:
        pytest.skip("Ed25519 is not available here")

    writes: list[str] = []
    real = chain.atomic_write_bytes

    def counting(path, data, **kwargs):
        writes.append(threading.current_thread().name)
        return real(path, data, **kwargs)

    monkeypatch.setattr(chain, "atomic_write_bytes", counting)

    async def first_load():
        chain._KeyMaterial.load()
        # Behind the loop: give the lane a moment to land it.
        for _ in range(200):
            if writes:
                break
            await asyncio.sleep(0.01)
        return threading.current_thread().name

    loop_thread = asyncio.run(first_load())
    assert len(writes) == 1, "a new key is published exactly once"
    assert writes[0] != loop_thread, "the publish ran on the loop thread"

    reset_capability_chain()
    chain._KeyMaterial.load()
    assert len(writes) == 1, "an unchanged key on disk was written again"
