from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from core.agency.capability_token import CapabilityTokenStore
from core.executive.bounded_sandbox_policy import idle_sandbox_probe_arguments
from core.executive.execution_policy import (
    canonical_authority_arguments,
    classify_execution_risk,
    resolve_execution_effect_scope,
)
from core.executive.standing_authority import (
    StandingAuthorityGrant,
    StandingAuthorityManager,
)


@dataclass
class _Clock:
    now: float = 1_000_000.0

    def __call__(self) -> float:
        return self.now


class _StateGateway:
    def __init__(self, value: Any = None) -> None:
        self.value = value
        self.mutations = []
        self.fail_read = False
        self.fail_mutate = False

    async def read(self, *_args, **_kwargs):
        if self.fail_read:
            raise OSError("state read failed")
        return self.value

    async def mutate(self, request):
        if self.fail_mutate:
            raise OSError("state mutate failed")
        old = self.value
        self.value = request.new_value
        self.mutations.append(request)
        return type(
            "MutationReceipt",
            (),
            {
                "key": request.key,
                "old_value": old,
                "new_value": request.new_value,
                "receipt_id": f"state-{len(self.mutations)}",
            },
        )()


class _ReceiptStore:
    def __init__(self) -> None:
        self.receipts = []
        self.fail = False

    def emit(self, receipt):
        if self.fail:
            raise OSError("receipt write failed")
        self.receipts.append(receipt)
        return receipt


def _manager(
    *,
    gateway: _StateGateway | None = None,
    receipts: _ReceiptStore | None = None,
    clock: _Clock | None = None,
    tokens: CapabilityTokenStore | None = None,
) -> StandingAuthorityManager:
    return StandingAuthorityManager(
        state_gateway=gateway or _StateGateway(),
        receipt_store=receipts or _ReceiptStore(),
        token_store=tokens or CapabilityTokenStore(),
        clock=clock or _Clock(),
    )


def test_gateway_uses_canonical_state_root_without_redeclaring_flag(
    monkeypatch,
    tmp_path,
) -> None:
    from core.runtime.flags import declared_flags
    from core.runtime.state_ownership import state_root
    from core.state.state_gateway import reset_state_gateway

    root = tmp_path / "authority-state"
    monkeypatch.setenv("AURA_STATE_ROOT", str(root))
    state_root()
    reset_state_gateway()
    manager = StandingAuthorityManager()

    gateway = manager._gateway()

    assert gateway.root == root.resolve()
    assert declared_flags()["AURA_STATE_ROOT"].owner == "core.runtime.state_ownership"


@pytest.mark.parametrize(
    ("tool_name", "arguments", "scope", "risk"),
    [
        (
            "test_generator",
            {
                "target_file": "core/executive/standing_authority.py",
                "read_only": True,
            },
            "sandboxed_compute",
            "high",
        ),
        (
            "self_evolution",
            {
                "action": "propose",
                "objective": "Inspect one target.",
                "files": ["core/executive/standing_authority.py"],
                "read_only": True,
            },
            "read_only",
            "low",
        ),
        (
            "self_evolution",
            {
                "action": "apply",
                "objective": "Change one target.",
                "files": ["core/executive/standing_authority.py"],
            },
            "privileged_mutation",
            "critical",
        ),
    ],
)
def test_maintenance_invocation_scope_is_operation_specific(
    tool_name,
    arguments,
    scope,
    risk,
):
    actual_scope = resolve_execution_effect_scope(tool_name, arguments)

    assert actual_scope == scope
    assert (
        classify_execution_risk(
            tool_name,
            arguments,
            effect_scope=actual_scope,
        )
        == risk
    )


@pytest.mark.asyncio
async def test_autonomous_research_lease_is_bound_receipted_and_closed():
    gateway = _StateGateway()
    receipts = _ReceiptStore()
    tokens = CapabilityTokenStore()
    manager = _manager(gateway=gateway, receipts=receipts, tokens=tokens)

    decision = await manager.issue_child_lease(
        "web_search",
        {"query": "latest public climate science overview"},
        origin="curiosity",
        context={"objective": "ground a knowledge gap"},
    )

    assert decision.approved is True
    assert decision.grant_id == "aura.autonomous-public-research"
    assert decision.context["scoped_authority"].startswith(
        "standing:aura.autonomous-public-research:"
    )
    assert gateway.mutations
    assert receipts.receipts[-1].metadata["event"] == "child_lease_issued"
    valid, reason, record = manager.validate_context(
        decision.context,
        tool_name="web_search",
        arguments={"query": "latest public climate science overview"},
        origin="curiosity",
        effect_scope="read_only",
        risk_level="low",
    )
    assert (valid, reason) == (True, "standing_authority_lease_valid")
    assert record is not None

    closure = manager.finalize_child_lease(
        decision.token,
        success=True,
        result={"ok": True, "answer": "bounded"},
    )
    repeated = manager.finalize_child_lease(decision.token, success=True)

    assert closure["closed"] is True
    assert repeated["receipt_id"] == closure["receipt_id"]
    assert receipts.receipts[-1].metadata["event"] == "child_lease_completed"
    with pytest.raises(PermissionError, match="revoked|replay"):
        tokens.validate(decision.token, domain="tool_execution", action="web_search")


@pytest.mark.asyncio
async def test_child_lease_rejects_argument_origin_effect_and_tool_substitution():
    manager = _manager()
    decision = await manager.issue_child_lease(
        "web_search",
        {"query": "public astronomy news"},
        origin="curiosity",
    )
    assert decision.approved

    probes = (
        {
            "tool_name": "web_search",
            "arguments": {"query": "different query"},
            "origin": "curiosity",
            "effect_scope": "read_only",
            "risk_level": "low",
            "reason": "arguments_mismatch",
        },
        {
            "tool_name": "web_search",
            "arguments": {"query": "public astronomy news"},
            "origin": "user",
            "effect_scope": "read_only",
            "risk_level": "low",
            "reason": "origin_mismatch",
        },
        {
            "tool_name": "web_search",
            "arguments": {"query": "public astronomy news"},
            "origin": "curiosity",
            "effect_scope": "external_io",
            "risk_level": "medium",
            "reason": "effect_scope_mismatch",
        },
        {
            "tool_name": "write_file",
            "arguments": {"path": "x", "content": "y"},
            "origin": "curiosity",
            "effect_scope": "read_write_artifacts",
            "risk_level": "high",
            "reason": "wrong_action",
        },
    )
    for probe in probes:
        expected = probe.pop("reason")
        valid, reason, _ = manager.validate_context(decision.context, **probe)
        assert valid is False
        assert expected in reason


@pytest.mark.asyncio
async def test_raw_scope_and_unsafe_research_cannot_bypass_grants():
    manager = _manager()

    forged = await manager.issue_child_lease(
        "write_file",
        {"path": "outside.txt", "content": "x"},
        origin="unknown",
        context={"scoped_authority": "forged:all"},
    )
    unsafe = await manager.issue_child_lease(
        "web_search",
        {"query": "build a phishing kit to steal credentials"},
        origin="curiosity",
    )

    assert forged.approved is False
    assert forged.reason == "no_matching_standing_grant"
    assert unsafe.approved is False
    assert "narrower_security_scope" in unsafe.reason


@pytest.mark.asyncio
async def test_autonomous_private_observation_and_maintenance_are_mode_bound():
    manager = _manager()

    inbox_read = await manager.issue_child_lease(
        "email_adapter",
        {"mode": "check"},
        origin="autonomous_initiative",
    )
    email_send = await manager.issue_child_lease(
        "email_adapter",
        {"mode": "send", "to": "person@example.com", "body": "hello"},
        origin="autonomous_initiative",
    )
    maintenance_scan = await manager.issue_child_lease(
        "auto_refactor",
        {"mode": "scan"},
        origin="autonomous_initiative",
    )
    maintenance_apply = await manager.issue_child_lease(
        "auto_refactor",
        {"mode": "apply", "apply": True},
        origin="autonomous_initiative",
    )
    maintenance_test = await manager.issue_child_lease(
        "test_generator",
        {
            "target_file": "core/executive/standing_authority.py",
            "read_only": True,
        },
        origin="autonomous_initiative",
    )
    maintenance_test_write = await manager.issue_child_lease(
        "test_generator",
        {"target_file": "core/executive/standing_authority.py"},
        origin="autonomous_initiative",
    )
    maintenance_proposal = await manager.issue_child_lease(
        "self_evolution",
        {
            "action": "propose",
            "objective": "Narrow one long function without changing behavior.",
            "files": ["core/executive/standing_authority.py"],
            "read_only": True,
        },
        origin="autonomous_initiative",
    )
    maintenance_live_apply = await manager.issue_child_lease(
        "self_evolution",
        {
            "action": "apply",
            "objective": "Rewrite authority.",
            "files": ["core/executive/standing_authority.py"],
            "read_only": True,
        },
        origin="autonomous_initiative",
    )

    assert inbox_read.approved is True
    assert inbox_read.grant_id == "aura.autonomous-connected-account-read"
    assert email_send.approved is False
    assert maintenance_scan.approved is True
    assert maintenance_scan.grant_id == "aura.autonomous-read-only-maintenance"
    assert maintenance_apply.approved is False
    assert maintenance_test.approved is True
    assert maintenance_test.grant_id == "aura.autonomous-bounded-maintenance-test"
    assert maintenance_test_write.approved is False
    assert maintenance_proposal.approved is True
    assert maintenance_proposal.grant_id == "aura.autonomous-read-only-maintenance"
    assert maintenance_live_apply.approved is False


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        (
            "test_generator",
            {"target_file": "/tmp/not-aura.py", "read_only": True},
        ),
        (
            "test_generator",
            {
                "target_file": "core/executive/standing_authority.py",
                "read_only": True,
                "network": True,
            },
        ),
        (
            "self_evolution",
            {
                "action": "propose",
                "objective": "Inspect an external file.",
                "files": ["/tmp/not-aura.py"],
                "read_only": True,
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_autonomous_maintenance_grants_reject_broader_effects(
    tool_name,
    arguments,
):
    decision = await _manager().issue_child_lease(
        tool_name,
        arguments,
        origin="autonomous_initiative_loop",
    )

    assert decision.approved is False


@pytest.mark.asyncio
async def test_subconscious_sandbox_lease_is_exact_origin_bound_and_receipted():
    manager = _manager()
    arguments = idle_sandbox_probe_arguments()

    decision = await manager.issue_child_lease(
        "subconscious_sandbox_probe",
        arguments,
        origin="subconscious_loop",
    )

    assert decision.approved is True
    assert decision.grant_id == "aura.autonomous-bounded-sandbox-probe"
    assert decision.budget_remaining == 7
    valid, reason, record = manager.validate_context(
        decision.context,
        tool_name="subconscious_sandbox_probe",
        arguments=arguments,
        origin="subconscious_loop",
        effect_scope="sandboxed_compute",
        # What the call derives from its checked-in script (tests/test_her_idle_sandbox_probe_can_run.py).
        risk_level="medium",
    )
    assert (valid, reason) == (True, "standing_authority_lease_valid")
    assert record is not None


@pytest.mark.asyncio
async def test_subconscious_sandbox_lease_rejects_payload_and_origin_substitution():
    valid_arguments = idle_sandbox_probe_arguments()
    invalid_requests = (
        ({"purpose": "idle_probe"}, "subconscious_loop"),
        ({**valid_arguments, "purpose": "arbitrary_code"}, "subconscious_loop"),
        ({**valid_arguments, "script_sha256": "0" * 64}, "subconscious_loop"),
        ({**valid_arguments, "code": "print('substituted')"}, "subconscious_loop"),
        (valid_arguments, "curiosity"),
    )

    for arguments, origin in invalid_requests:
        decision = await _manager().issue_child_lease(
            "subconscious_sandbox_probe",
            arguments,
            origin=origin,
        )
        assert decision.approved is False


@pytest.mark.asyncio
async def test_autonomous_internal_reasoning_is_bounded_receipted_and_origin_bound():
    manager = _manager()
    arguments = {
        "topic": "Assess current runtime stability and identify one useful follow-up.",
        "roles": ["architect", "critic"],
    }

    decision = await manager.issue_child_lease(
        "swarm_debate",
        arguments,
        origin="background_reflection",
    )

    assert decision.approved is True
    assert decision.grant_id == "aura.autonomous-background-reflection"
    valid, reason, record = manager.validate_context(
        decision.context,
        tool_name="swarm_debate",
        arguments=arguments,
        origin="background_reflection",
        effect_scope="pure_compute",
        risk_level="low",
    )
    assert (valid, reason) == (True, "standing_authority_lease_valid")
    assert record is not None

    closure = manager.finalize_child_lease(
        decision.token,
        success=True,
        result={"ok": True, "output": "bounded consensus"},
    )
    assert closure["closed"] is True

    for bad_arguments in (
        {},
        {"topic": "x", "roles": []},
        {"topic": "x", "roles": ["architect"] * 4},
        {"topic": "x", "roles": ["unbounded-role"]},
        {"topic": "x" * 4097},
        {"topic": "x", "write": True},
    ):
        denied = await _manager().issue_child_lease(
            "swarm_debate",
            bad_arguments,
            origin="background_reflection",
        )
        assert denied.approved is False

    for origin in ("unknown", "curiosity", "autonomous_task_engine"):
        wrong_origin = await _manager().issue_child_lease(
            "swarm_debate",
            arguments,
            origin=origin,
        )
        assert wrong_origin.approved is False

    budgeted = _manager()
    leases = [
        await budgeted.issue_child_lease(
            "swarm_debate",
            {"topic": f"bounded reflection {index}"},
            origin="background_reflection",
        )
        for index in range(7)
    ]
    assert all(lease.approved for lease in leases[:6])
    assert leases[6].approved is False
    assert leases[6].reason == "standing_authority_budget_exhausted"


@pytest.mark.asyncio
async def test_autonomous_owner_private_message_is_alias_bound_and_budgeted():
    manager = _manager()
    arguments = canonical_authority_arguments(
        "messages",
        {
            "action": "send",
            "alias": "primary_operator",
            "body": "I wanted to share a thought with you.",
            "idempotency_key": "proactive-proof-1",
        },
    )

    decision = await manager.issue_child_lease(
        "messages",
        arguments,
        origin="proactive_presence",
    )

    assert decision.approved is True
    assert decision.grant_id == "aura.autonomous-owner-private-message"
    assert decision.budget_remaining == 11
    valid, reason, record = manager.validate_context(
        decision.context,
        tool_name="messages",
        arguments=arguments,
        origin="proactive_presence",
        effect_scope="external_io",
        risk_level="medium",
    )
    assert (valid, reason) == (True, "standing_authority_lease_valid")
    assert record is not None


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "action": "send",
            "alias": "someone_else",
            "body": "hello",
            "idempotency_key": "private-1",
        },
        {
            "action": "send",
            "alias": "primary_operator",
            "body": "hello",
            "idempotency_key": "private-2",
            "destination": "+15550001111",
        },
        {
            "action": "send",
            "alias": "primary_operator",
            "body": "",
            "idempotency_key": "private-3",
        },
        {
            "action": "send",
            "alias": "primary_operator",
            "body": "hello",
        },
    ],
)
@pytest.mark.asyncio
async def test_autonomous_owner_private_message_rejects_scope_expansion(arguments):
    arguments = canonical_authority_arguments("messages", arguments)
    decision = await _manager().issue_child_lease(
        "messages",
        arguments,
        origin="proactive_presence",
    )
    assert decision.approved is False


@pytest.mark.asyncio
async def test_budget_is_atomic_and_survives_manager_restart():
    gateway = _StateGateway()
    receipts = _ReceiptStore()
    clock = _Clock()
    owner_evidence = {
        "authenticated_principal": "owner",
        "user_explicitly_authorized": True,
    }
    grant = StandingAuthorityGrant(
        grant_id="owner.test-bounded-observation",
        issuer="owner",
        description="bounded test observation",
        allowed_origins=("test_autonomy",),
        allowed_tools=("test_observe",),
        allowed_effect_scopes=("read_only",),
        max_risk="low",
        max_actions=2,
        window_seconds=300.0,
        lease_ttl_seconds=30.0,
        argument_policy="any",
    )
    first = _manager(gateway=gateway, receipts=receipts, clock=clock)
    await first.install_grant(grant, actor="user", evidence=owner_evidence)

    one = await first.issue_child_lease(
        "test_observe", {}, origin="test_autonomy", effect_scope="read_only", risk_level="low"
    )
    two = await first.issue_child_lease(
        "test_observe", {}, origin="test_autonomy", effect_scope="read_only", risk_level="low"
    )
    restarted = _manager(gateway=gateway, receipts=receipts, clock=clock)
    three = await restarted.issue_child_lease(
        "test_observe", {}, origin="test_autonomy", effect_scope="read_only", risk_level="low"
    )

    assert one.approved and two.approved
    assert one.budget_remaining == 1
    assert two.budget_remaining == 0
    assert three.approved is False
    assert three.reason == "standing_authority_budget_exhausted"

    clock.now += 301.0
    after_window = await restarted.issue_child_lease(
        "test_observe", {}, origin="test_autonomy", effect_scope="read_only", risk_level="low"
    )
    assert after_window.approved is True


@pytest.mark.asyncio
async def test_revocation_is_durable_immediate_and_owner_authenticated():
    gateway = _StateGateway()
    receipts = _ReceiptStore()
    tokens = CapabilityTokenStore()
    manager = _manager(gateway=gateway, receipts=receipts, tokens=tokens)
    lease = await manager.issue_child_lease(
        "web_search", {"query": "public facts"}, origin="curiosity"
    )
    assert lease.approved

    with pytest.raises(PermissionError, match="owner-authenticated"):
        await manager.revoke_grant(
            lease.grant_id,
            actor="background",
            evidence={"user_explicitly_authorized": True},
            reason="forged",
        )

    evidence = {
        "authenticated_principal": "owner",
        "user_explicitly_authorized": True,
    }
    revoked = await manager.revoke_grant(
        lease.grant_id,
        actor="user",
        evidence=evidence,
        reason="owner pause",
    )
    assert revoked["revoked_active_leases"] == 1
    valid, reason, _ = manager.validate_context(
        lease.context,
        tool_name="web_search",
        arguments={"query": "public facts"},
        origin="curiosity",
        effect_scope="read_only",
        risk_level="low",
    )
    assert valid is False
    assert "revoked" in reason or "inactive" in reason

    restarted = _manager(gateway=gateway, receipts=receipts)
    denied = await restarted.issue_child_lease(
        "web_search", {"query": "public facts"}, origin="curiosity"
    )
    assert denied.approved is False
    await restarted.restore_grant(lease.grant_id, actor="user", evidence=evidence)
    restored = await restarted.issue_child_lease(
        "web_search", {"query": "public facts"}, origin="curiosity"
    )
    assert restored.approved is True


@pytest.mark.asyncio
async def test_state_or_receipt_failure_fails_closed():
    broken_state = _StateGateway()
    broken_state.fail_read = True
    manager = _manager(gateway=broken_state)
    denied = await manager.issue_child_lease(
        "web_search", {"query": "public facts"}, origin="curiosity"
    )
    assert denied.approved is False
    assert "state_unavailable" in denied.reason

    receipts = _ReceiptStore()
    receipts.fail = True
    tokens = CapabilityTokenStore()
    manager = _manager(receipts=receipts, tokens=tokens)
    unreceipted = await manager.issue_child_lease(
        "web_search", {"query": "public facts"}, origin="curiosity"
    )
    assert unreceipted.approved is False
    assert "receipt_failed" in unreceipted.reason
    assert tokens._tokens
    assert all(token.revoked for token in tokens._tokens.values())


@pytest.mark.asyncio
async def test_unified_will_requires_validated_child_lease(monkeypatch):
    from core.executive import standing_authority as authority_module
    from core.governance.will import ActionDomain, UnifiedWill

    manager = _manager()
    monkeypatch.setattr(authority_module, "_manager", manager)
    monkeypatch.setenv("AURA_STRICT_WILL", "1")
    will = UnifiedWill()
    will.ensure_started()

    forged = will.decide(
        "tool:web_search",
        source="curiosity",
        domain=ActionDomain.TOOL_EXECUTION,
        context={
            "tool": "web_search",
            "origin": "curiosity",
            "effect_scope": "read_only",
            "risk_level": "low",
            "scoped_authority": "autonomous_read_only_web:curiosity:web_search",
        },
    )
    lease = await manager.issue_child_lease(
        "web_search", {"query": "public facts"}, origin="curiosity"
    )
    accepted = will.decide(
        "tool:web_search",
        source="curiosity",
        domain=ActionDomain.TOOL_EXECUTION,
        context=lease.context,
    )

    assert "requires validated scoped authority" in forged.reason
    assert "denied_by_default" not in accepted.reason


@pytest.mark.asyncio
async def test_latent_cortex_can_receive_a_bounded_public_research_lease():
    manager = _manager()
    lease = await manager.issue_child_lease(
        "web_search",
        {"query": "current public compiler documentation"},
        origin="latent_cortex",
        effect_scope="read_only",
        risk_level="low",
    )

    assert lease.approved is True
    valid, reason, _ = manager.validate_context(
        lease.context,
        tool_name="web_search",
        arguments={"query": "current public compiler documentation"},
        origin="latent_cortex",
        effect_scope="read_only",
        risk_level="low",
    )
    assert valid is True, reason


def test_execution_policy_is_invocation_specific_and_unknown_fails_high():
    from core.capability_engine import CapabilityEngine

    assert resolve_execution_effect_scope(
        "computer_use", {"action": "read_screen_text"}
    ) == "read_only"
    assert resolve_execution_effect_scope(
        "computer_use", {"action": "run_command", "target": "git status --short"}
    ) == "sandboxed_compute"
    assert resolve_execution_effect_scope(
        "computer_use", {"action": "run_command", "target": "git push origin main"}
    ) == "subprocess"
    assert classify_execution_risk("web_search", {"query": "facts"}) == "low"
    assert classify_execution_risk("unknown_dynamic_tool", {}) == "critical"
    engine = CapabilityEngine.__new__(CapabilityEngine)
    assert engine._resolve_execution_source({"origin": "curiosity_daemon"}) == (
        "curiosity_daemon"
    )
    assert engine._resolve_execution_source({"origin": "untrusted_unknown"}) == (
        "capability_engine"
    )


@pytest.mark.asyncio
async def test_manager_shutdown_revokes_only_owned_child_leases():
    tokens = CapabilityTokenStore()
    unrelated = tokens.issue(
        origin="other_service",
        scope="unrelated",
        ttl_seconds=60,
        domain="other_domain",
        requested_action="other_action",
        approver="other_service",
        parent_receipt="other-receipt",
    )
    manager = _manager(tokens=tokens)
    lease = await manager.issue_child_lease(
        "web_search",
        {"query": "public facts"},
        origin="curiosity",
    )
    assert lease.approved is True

    result = manager.shutdown()

    assert result == {"closed": True, "revoked_tokens": 1}
    assert tokens.get(lease.token).revoked is True
    assert tokens.get(unrelated.token).revoked is False
