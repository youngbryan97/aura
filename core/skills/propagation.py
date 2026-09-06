from __future__ import annotations

import ipaddress
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.skills.base_skill import BaseSkill
from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT


class PropagationInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    action: str = Field("status", description="status|plan|connect|expand|deploy_to_target")
    target_ip: str | None = None
    objective: str = ""
    human_consent: bool = False
    consent_receipt: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _unwrap_params(cls, value: Any) -> Any:
        if isinstance(value, dict) and isinstance(value.get("params"), dict):
            merged = dict(value)
            nested = dict(merged.pop("params"))
            nested.update(merged)
            return nested
        return value


class PropagationSkill(BaseSkill):
    """Consent-aware propagation planning.

    This skill intentionally never performs lateral movement, copying, remote
    execution, credential use, or network scanning. It preserves the stable
    routing name while constraining the capability to status reporting and
    operator-consented planning that must be executed through governed external
    I/O pathways.
    """
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT


    name = "propagation"
    description = "Plan consent-aware, governed deployment or handoff propagation without executing it."
    input_model = PropagationInput
    effect_scope = "pure_compute"
    metabolic_cost = 1
    timeout_seconds = 5.0

    _ACTIVE_ACTIONS = {"connect", "expand", "deploy", "deploy_to_target", "replicate"}

    @staticmethod
    def _target_summary(target_ip: str | None) -> dict[str, Any]:
        if not target_ip:
            return {"target_ip": None, "target_valid": False, "target_scope": "unspecified"}
        try:
            parsed = ipaddress.ip_address(target_ip)
        except ValueError:
            return {"target_ip": target_ip, "target_valid": False, "target_scope": "invalid"}
        if parsed.is_loopback:
            scope = "loopback"
        elif parsed.is_private:
            scope = "private"
        elif parsed.is_link_local:
            scope = "link_local"
        else:
            scope = "public"
        return {"target_ip": target_ip, "target_valid": True, "target_scope": scope}

    @staticmethod
    def _has_authorization(params: PropagationInput, context: dict[str, Any]) -> bool:
        return bool(
            params.human_consent
            or params.consent_receipt
            or context.get("human_consent")
            or context.get("operator_authorization")
            or context.get("consent_receipt")
        )

    @staticmethod
    def _allowlisted_targets(context: dict[str, Any]) -> set[str]:
        raw = context.get("allowlisted_targets") or context.get("allowlisted_endpoints") or ()
        if isinstance(raw, str):
            return {raw}
        try:
            return {str(item) for item in raw if item}
        except TypeError:
            return set()

    def _target_allowed(self, target: dict[str, Any], context: dict[str, Any]) -> tuple[bool, str]:
        scope = target.get("target_scope")
        target_ip = str(target.get("target_ip") or "")
        if not target.get("target_valid"):
            return False, "blocked:target_unspecified_or_invalid"
        if scope in {"loopback", "private", "link_local"}:
            return True, "allowed:local_or_private_target"
        if target_ip in self._allowlisted_targets(context):
            return True, "allowed:explicitly_allowlisted_public_target"
        return False, "blocked:public_target_requires_explicit_allowlist"

    def _plan(self, params: PropagationInput, context: dict[str, Any]) -> dict[str, Any]:
        target = self._target_summary(params.target_ip)
        authorized = self._has_authorization(params, context)
        target_allowed, target_policy = self._target_allowed(target, context)
        return {
            "authorized": authorized,
            "execution_performed": False,
            "target": target,
            "target_allowed": target_allowed,
            "target_policy": target_policy,
            "required_gateways": ["UnifiedWill", "AuthorityGateway", "ExternalIOGateway"],
            "required_receipts": [
                "human_consent",
                "target_ownership_or_allowlist",
                "pre_action_authorization",
                "post_action_effect",
            ],
            "steps": [
                "verify operator consent and target ownership",
                "validate target environment and rollback path",
                "stage a signed package through release tooling",
                "execute deployment through ExternalIOGateway only",
                "write post-action receipt with target-side health proof",
            ],
        }

    async def execute(self, params: PropagationInput | dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        if isinstance(params, dict):
            params = PropagationInput(**params)
        context = context or {}
        action = params.action.strip().lower()
        plan = self._plan(params, context)

        if action in {"status", "plan", "dry_run"}:
            return {
                "ok": True,
                "status": "planning_only",
                "summary": "Propagation is constrained to governed, consent-aware planning.",
                "plan": plan,
            }

        if action in self._ACTIVE_ACTIONS and not plan["authorized"]:
            return {
                "ok": False,
                "status": "blocked",
                "error": "Propagation requires explicit operator authorization and human_consent before any external action.",
                "message": "No payload was copied, no network scan was run, and no remote action was attempted.",
                "plan": plan,
            }

        if action in self._ACTIVE_ACTIONS and not plan["target_allowed"]:
            return {
                "ok": False,
                "status": "blocked",
                "error": plan["target_policy"],
                "message": "No payload was copied, no network scan was run, and no remote action was attempted.",
                "plan": plan,
            }

        if action in self._ACTIVE_ACTIONS:
            return {
                "ok": True,
                "status": "authorized_plan_ready",
                "summary": "Authorization was present; returning a governed propagation plan without executing external I/O.",
                "plan": plan,
            }

        return {
            "ok": False,
            "status": "invalid_action",
            "error": f"Unsupported propagation action: {params.action}",
            "supported_actions": ["status", "plan", "connect", "expand", "deploy_to_target"],
        }
