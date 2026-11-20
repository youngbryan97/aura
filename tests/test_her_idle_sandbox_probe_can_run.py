"""Her idle sandbox probe is rated by what its code can reach, so the grant that allows it is not overruled.

From 10 November 2026 her will deferred the probe 313 times and it never ran: its checked-in script read ``platform``,
which can start a process, so it was rated high, and a high call from her subconscious asks for a confirmation nobody is
there to give. The standing grant written for exactly this probe (aura.autonomous-bounded-sandbox-probe) could not
help. The probe now only computes and prints, and is rated as any such snippet is.
"""
from __future__ import annotations

import pytest

from core.executive.bounded_sandbox_policy import IDLE_SANDBOX_PROBE_SCRIPT, idle_sandbox_probe_arguments
from core.executive.execution_policy import classify_execution_risk, resolve_execution_effect_scope

pytestmark = pytest.mark.unit

TOOL = "subconscious_sandbox_probe"


def test_the_probe_only_computes():
    from core.skills.snippet_reach import reach_of

    assert reach_of(IDLE_SANDBOX_PROBE_SCRIPT).only_computes


def test_the_probe_is_rated_as_a_snippet_that_computes():
    arguments = idle_sandbox_probe_arguments()
    assert resolve_execution_effect_scope(TOOL, arguments) == "sandboxed_compute"
    assert classify_execution_risk(TOOL, arguments) == "medium"


@pytest.mark.parametrize("arguments", [
    {"purpose": "idle_probe", "script_sha256": "0" * 64},
    {"purpose": "idle_probe", "script_sha256": idle_sandbox_probe_arguments()["script_sha256"], "code": "print(1)"},
    {},
])
def test_any_other_script_keeps_the_high_rating(arguments):
    assert classify_execution_risk(TOOL, arguments) == "high"


def test_her_permission_model_lets_it_run_with_nobody_there():
    from core.capabilities.permission_model import PermissionRiskModel

    arguments = idle_sandbox_probe_arguments()
    decision = PermissionRiskModel().check_permission(
        "tool_execution", f"tool:{TOOL}", {"origin": "subconscious_loop", "tool": TOOL},
        effect_scope="sandboxed_compute", execution_risk=classify_execution_risk(TOOL, arguments))
    assert decision.approved is True and decision.requires_confirmation is False


def _her_will_on(context, monkeypatch):
    import core.governance.will as will_module
    from core.capabilities.permission_model import PermissionRiskModel
    from core.governance.will import ActionDomain, UnifiedWill, WillOutcome

    model = PermissionRiskModel()
    monkeypatch.setattr(will_module.ServiceContainer, "get",
                        lambda name, default=None: model if name == "permission_model" else default)
    will = UnifiedWill.__new__(UnifiedWill)
    return will._decide_inject_scar_constraints(None, [], f"tool:{TOOL}", context, ActionDomain.TOOL_EXECUTION,
                                                WillOutcome.PROCEED, "ok", [])


def test_her_will_rates_the_probe_by_its_arguments_and_not_by_the_context_around_them(monkeypatch):
    """LIVE 20 November 2026, after the probe was rated medium: its lease said medium, and her will still deferred it as
    "risk=high", rating the gateway's whole context (the arguments, the lease, who is present) as if it were the call."""
    from core.governance.will import WillOutcome

    arguments = idle_sandbox_probe_arguments()
    around = {**arguments, "standing_authority_grant_id": "aura.autonomous-bounded-sandbox-probe",
              "standing_authority_lease_id": "lease-1", "tool": TOOL, "skill": TOOL,
              "authority_origin": "subconscious_loop", "effect_scope": "sandboxed_compute", "risk_level": "medium"}
    outcome, reason = _her_will_on({**around, "authority_arguments": dict(arguments)}, monkeypatch)
    assert outcome == WillOutcome.PROCEED, reason
    outcome, reason = _her_will_on(around, monkeypatch)
    assert outcome == WillOutcome.DEFER and "risk=high" in reason, "without the arguments it is the context rated"


def test_the_gateway_hands_her_will_the_arguments():
    import inspect

    from core.executive.authority_gateway import AuthorityGateway

    assert '"authority_arguments": dict(args or {}),' in inspect.getsource(AuthorityGateway.authorize_tool_execution)
