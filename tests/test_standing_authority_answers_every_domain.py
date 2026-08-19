"""A permission that could be held and never used.

`validate_standing_authority_context` takes a tool, an origin, an effect scope
and a risk level — and no domain. It is domain-agnostic by construction,
because the grants are written in those terms. `will.py` consulted it for
`tool_execution` only, so the other four default-deny domains fell through to a
raw `context.get("scoped_authority")` check that no legitimate issuer ever
populates — and that key sits on the agency orchestrator's forgeable list
precisely so nothing may set it from a payload.

MEASURED live 2026-08-18. An authenticated owner foreground request, which
holds `owner.foreground-request` (allowed_tools "*", allowed_effect_scopes "*",
max_risk critical), asked to work through a web page:

    WILL REFUSED: desktop_ui/network_call -- denied_by_default: network_call
    requires validated scoped authority

The authority existed, was correct, and was unreachable.

Widening it must not widen anything else, which is what the second half of this
file is for: autonomous origins keep their narrow grants, and a domain nobody
granted stays refused.
"""

from __future__ import annotations

import inspect

import pytest

from core.governance import will as will_module


def _authority_source() -> str:
    return inspect.getsource(will_module)


class TestTheMechanismIsConsultedEverywhereItApplies:
    @pytest.mark.parametrize(
        "domain",
        ["TOOL_EXECUTION", "NETWORK_CALL", "FILE_WRITE", "CLOUD_CALL", "CI_CD"],
    )
    def test_every_default_deny_domain_consults_standing_authority(self, domain):
        source = _authority_source()
        block_start = source.index("authority_validation_reason = \"\"")
        block = source[block_start : block_start + 2000]
        assert f"ActionDomain.{domain}" in block, (
            f"{domain} defaults to deny but never asks the grant system"
        )

    def test_the_refusal_now_says_why(self):
        """The reason was suppressed for every domain but one."""
        source = _authority_source()
        assert 'detail = f" ({authority_validation_reason})" if authority_validation_reason else ""' in source


class TestNothingIsWidenedThatWasNotGranted:
    def test_validation_is_domain_agnostic_by_signature(self):
        """The grants speak in tools and scopes, not domains."""
        from core.executive.standing_authority import validate_standing_authority_context

        parameters = set(inspect.signature(validate_standing_authority_context).parameters)
        assert "domain" not in parameters
        assert {"tool_name", "origin", "effect_scope", "risk_level"} <= parameters

    def test_an_unauthorised_context_is_still_refused(self):
        from core.executive.standing_authority import validate_standing_authority_context

        allowed, reason = validate_standing_authority_context(
            {},
            tool_name="sovereign_browser",
            origin="unknown_origin",
            effect_scope="state_changing",
            risk_level="critical",
        )
        assert allowed is False
        assert reason

    def test_autonomous_research_stays_read_only(self):
        """The narrow grants are what keep autonomy bounded."""
        from core.executive.standing_authority import _builtin_grants

        research = next(
            grant for grant in _builtin_grants() if grant.grant_id == "aura.autonomous-public-research"
        )
        assert research.allowed_effect_scopes == ("read_only",)
        assert research.max_risk == "low"
        assert "*" not in research.allowed_tools

    def test_the_owner_grant_is_the_one_that_carries_breadth(self):
        from core.executive.standing_authority import _builtin_grants

        owner = next(
            grant for grant in _builtin_grants() if grant.grant_id == "owner.foreground-request"
        )
        assert owner.allowed_tools == ("*",)
        assert owner.allowed_effect_scopes == ("*",)
        assert owner.max_risk == "critical"
