"""RENDER THIS must traverse the governed tool spine as an OWNER action.

Live failure (July 2026 screenshot): clicking RENDER THIS on the Imagine
panel surfaced "Executive veto: denied_by_default: tool_execution requires
validated scoped authority (signed_standing_authority_lease_missing)" —
and earlier "aura_now_defer: present-state policy requires stabilization".
The route executed image_gen with NO context, so standing authority saw an
unknown origin (no user-facing grant matched) and the AuraNow policy treated
an explicit owner click as an ungrounded autonomous act.

These tests pin the fix: the route ships the full desktop execution
contract, that contract satisfies standing authority's user-authority check,
and it satisfies the exact flag quintet the being-runtime policy requires to
lift soft defers for explicit foreground desktop tools.
"""
from __future__ import annotations

import pytest


def _extract_visualize_context() -> dict:
    """Parse the context literal the route passes to execute_tool.

    Static extraction keeps this test independent of the full FastAPI/runtime
    stack while still failing if the contract keys drift or are dropped.
    """
    import inspect

    from interface.routes import system as system_routes

    source = inspect.getsource(system_routes.api_imagination_visualize)
    assert "execute_tool(" in source
    assert "context=" in source, "visualize no longer passes an execution context"
    return source


@pytest.fixture()
def route_source() -> str:
    return _extract_visualize_context()


REQUIRED_CONTRACT_KEYS = (
    # The quintet core/being/runtime.py:action_policy requires before an
    # explicit foreground desktop tool may lift soft AuraNow defers.
    "desktop_execution_contract",
    "foreground_request",
    "user_explicitly_authorized",
    "user_visible_desktop_action",
    "verification_required",
    # Standing-authority user-authority evidence.
    "user_requested_action",
    "user_explicit_action_request",
)


def test_route_ships_full_desktop_execution_contract(route_source):
    for key in REQUIRED_CONTRACT_KEYS:
        assert f'"{key}": True' in route_source, f"missing contract key: {key}"
    assert '"origin": "desktop_ui"' in route_source
    assert '"source": "desktop_ui"' in route_source


def test_contract_context_satisfies_standing_authority_user_check():
    from core.executive.standing_authority import context_has_user_authority

    context = {
        "foreground_request": True,
        "user_requested_action": True,
        "user_explicit_action_request": True,
        "user_explicitly_authorized": True,
    }
    assert context_has_user_authority("desktop_ui", context)
    # The pre-fix call: no context, unknown origin — must remain unauthorized.
    assert not context_has_user_authority("unknown", {})
    assert not context_has_user_authority("capability_engine", {})


def test_contract_context_matches_being_runtime_exemption_quintet():
    """The exact flags action_policy checks must all be shipped by the route."""
    from core.being import runtime as being_runtime
    from tests.source_contract import function_with_its_helpers

    # The exemption moved into `_action_policy_internal_runtime_maintenance`,
    # which `action_policy` calls, so reading the method alone found none of
    # the flags it still consults.
    policy_source = function_with_its_helpers(
        being_runtime, "BeingRuntime.action_policy", depth=2
    )
    quintet = [
        "desktop_execution_contract",
        "foreground_request",
        "user_explicitly_authorized",
        "user_visible_desktop_action",
        "verification_required",
    ]
    for key in quintet:
        # CP126 310a67ee moved these from a bare context.get() to
        # attested_context_flag(), which additionally requires a capability
        # token bound to the domain+action. Either spelling means the
        # exemption still reads the flag; what matters to this contract is
        # that the route ships every key the policy consults.
        assert (
            f'context.get("{key}")' in policy_source
            or f'"{key}"' in policy_source
        ), (
            f"being-runtime exemption no longer reads {key}; update the "
            "visualize contract test and route together"
        )
    route = _extract_visualize_context()
    for key in quintet:
        assert f'"{key}": True' in route


def test_user_facing_origin_resolves_through_capability_engine():
    from core.capability_engine import CapabilityEngine

    engine = CapabilityEngine.__new__(CapabilityEngine)
    resolved = CapabilityEngine._resolve_execution_source(
        engine, {"origin": "desktop_ui", "foreground_request": True}
    )
    assert resolved == "desktop_ui"
