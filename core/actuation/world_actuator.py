"""core/actuation/world_actuator.py — Sovereign External Actuation Coordinator.

Routes all actuation requests through the canonical ActionExecutor and enforces
safety approval gates for high-risk actions.

Hardening (CP126): unknown categories are refused terminally (never defaulted
to a broad domain); risk is parameter-aware and the high-risk verdict is passed
to the executor; every request carries an operation id + deadline; the audit
trail is lock-synchronized, bounded, and stores redacted parameter/result
digests rather than raw secrets; and executor faults always reconcile the
pending record.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from typing import Any

from core.runtime.action_executor import ActionExecutor
from core.runtime.lockdep import LockRank, checked_lock
from core.will import ActionDomain

logger = logging.getLogger("Aura.WorldActuator")

# Allowed/safe categories
ALLOWED_CATEGORIES: set[str] = {
    "local_files",
    "code_repos",
    "browser",
    "desktop",
    "email_drafts",
    "calendar_drafts",
    "issue_drafts",
    "pr_drafts",
    "cloud_resources_owned",
    "databases_owned",
    "documents_owned",
}

# High-risk actions requiring strict checks
HIGH_RISK_ACTIONS: set[str] = {
    "send_message",
    "post_publicly",
    "delete_file",
    "spend_money",
    "change_cloud_infra",
    "publish_code",
    "contact_third_party",
    "run_downloaded_code",
    "modify_credentials",
    "modify_secrets",
}

# Parameter shapes that make an otherwise ordinary action high-risk regardless
# of its name (5773a761: the same action can be harmless or destructive).
_HIGH_RISK_PARAM_KEYS = {
    "recipient", "recipients", "to", "amount", "price", "sql", "query",
    "credential", "credentials", "secret", "password", "token", "api_key",
}
_HIGH_RISK_VALUE_MARKERS = (
    "drop table", "delete from", "truncate", "rm -rf", "; delete", "--force",
    "transfer", "withdraw", "wire ", "sudo ",
)
_SENSITIVE_KEY_MARKERS = ("secret", "password", "passwd", "token", "key", "credential", "auth")

_MAX_AUDIT_RECORDS = 500
_MAX_PARAM_BYTES = 256 * 1024
_DEFAULT_DEADLINE_S = 60.0
# Hard coordinator backstop over the executor's own soft timeout.
_COORDINATOR_GRACE_S = 5.0

_CATEGORY_DOMAIN = {
    "local_files": ActionDomain.FILE_WRITE,
    "code_repos": ActionDomain.FILE_WRITE,
    "documents_owned": ActionDomain.FILE_WRITE,
    "browser": ActionDomain.NETWORK_CALL,
    "desktop": ActionDomain.ENVIRONMENT_ACTION,
    "cloud_resources_owned": ActionDomain.CLOUD_CALL,
    "databases_owned": ActionDomain.CLOUD_CALL,
    # email/calendar/issue/pr drafts remain EXTERNAL_ACTION (no finer domain in
    # the current ActionDomain enum — tracked as 80efe4ad).
}


def _redact(value: Any, _depth: int = 0) -> Any:
    """Redact secret-bearing keys and truncate large blobs for the audit trail."""
    if _depth > 4:
        return "…"
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(k, str) and any(m in k.lower() for m in _SENSITIVE_KEY_MARKERS):
                out[k] = "***redacted***"
            else:
                out[k] = _redact(v, _depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [_redact(v, _depth + 1) for v in value[:20]]
    if isinstance(value, str):
        return value if len(value) <= 256 else value[:256] + "…"
    return value


def _digest(value: Any) -> str:
    try:
        return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
    except (TypeError, ValueError):
        return hashlib.sha256(repr(value).encode()).hexdigest()


class WorldActuator:
    """Coordinates and governs all external digital and physical actuation."""

    def __init__(self) -> None:
        self.last_actuations: list[dict[str, Any]] = []
        self._audit_lock = checked_lock(
            f"world_actuator.audit.{id(self):x}", rank=LockRank.LEAF
        )

    @staticmethod
    def _param_aware_high_risk(action_name: str, params: dict[str, Any]) -> bool:
        if action_name in HIGH_RISK_ACTIONS:
            return True
        if not isinstance(params, dict):
            return False
        if any(k in _HIGH_RISK_PARAM_KEYS for k in params if isinstance(k, str)):
            return True
        try:
            blob = json.dumps(params, default=str).lower()
        except (TypeError, ValueError):
            blob = str(params).lower()
        return any(marker in blob for marker in _HIGH_RISK_VALUE_MARKERS)

    def _record(self, entry: dict[str, Any]) -> dict[str, Any]:
        with self._audit_lock:
            self.last_actuations.append(entry)
            if len(self.last_actuations) > _MAX_AUDIT_RECORDS:
                del self.last_actuations[: len(self.last_actuations) - _MAX_AUDIT_RECORDS]
        return entry

    def _update(self, entry: dict[str, Any], **fields: Any) -> None:
        with self._audit_lock:
            entry.update(fields)

    async def actuate(
        self,
        category: str,
        action_name: str,
        params: dict[str, Any],
        source: str = "world_actuator",
        high_risk_flag: bool | None = None,
        deadline_s: float = _DEFAULT_DEADLINE_S,
    ) -> dict[str, Any]:
        """Main entry point to perform any external actuation.

        Routes the request to the correct ActionDomain, verifying risk first.
        Unknown categories are refused terminally instead of silently widening
        to EXTERNAL_ACTION.
        """
        operation_id = uuid.uuid4().hex
        if category == "robotics_devices":
            logger.warning(
                "Refused legacy physical dispatch outside Reality Reach (op=%s)",
                operation_id,
            )
            return {
                "ok": False,
                "error": "physical_category_requires_reality_reach",
                "category": category,
                "operation_id": operation_id,
            }
        if not isinstance(category, str) or category not in ALLOWED_CATEGORIES:
            logger.warning("Refused actuation: unknown category %r (op=%s)", category, operation_id)
            return {"ok": False, "error": "unknown_category", "category": str(category), "operation_id": operation_id}
        if not isinstance(action_name, str) or not action_name.strip():
            return {"ok": False, "error": "invalid_action_name", "operation_id": operation_id}
        if not isinstance(params, dict):
            return {"ok": False, "error": "params_must_be_a_mapping", "operation_id": operation_id}
        try:
            if len(json.dumps(params, default=str).encode()) > _MAX_PARAM_BYTES:
                return {"ok": False, "error": "params_too_large", "operation_id": operation_id}
        except (TypeError, ValueError):
            return {"ok": False, "error": "params_not_serializable", "operation_id": operation_id}
        source = (str(source) or "world_actuator").strip()[:128]

        is_high_risk = self._param_aware_high_risk(action_name, params) or high_risk_flag is True

        logger.info(
            "Actuation request op=%s category=%s action=%s (high_risk=%s) source=%s",
            operation_id, category, action_name, is_high_risk, source,
        )

        # Audit stores REDACTED digests, never the raw secret-bearing payload.
        record = self._record({
            "operation_id": operation_id,
            "category": category,
            "action": action_name,
            "params_digest": _digest(params),
            "params_redacted": _redact(params),
            "source": source,
            "is_high_risk": is_high_risk,
            "status": "pending",
            "started_at": time.time(),
        })

        if is_high_risk:
            logger.warning("⚠️ High-risk action op=%s: %s — requiring Will validation.", operation_id, action_name)

        domain = _CATEGORY_DOMAIN.get(category, ActionDomain.EXTERNAL_ACTION)

        try:
            # The high-risk verdict and operation identity travel WITH the call so
            # the executor/Will act on them (09dcc0bc), and both a coordinator
            # deadline and the executor's own timeout bound the effect (9a68d683).
            result = await asyncio.wait_for(
                ActionExecutor.execute(
                    domain=domain,
                    action_name=f"actuation.{category}.{action_name}",
                    params={**params, "_is_high_risk": is_high_risk, "_operation_id": operation_id},
                    source=source,
                    execution_timeout_s=deadline_s,
                ),
                timeout=deadline_s + _COORDINATOR_GRACE_S,
            )
        except TimeoutError:
            self._update(record, status="uncertain_timeout", error="coordinator_deadline_exceeded",
                         result_digest="", ended_at=time.time())
            return {"ok": False, "error": "coordinator_deadline_exceeded", "operation_id": operation_id, "outcome": "uncertain"}
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
            # An executor fault must never leave a permanently-pending record.
            self._update(record, status="error", error=str(exc)[:300], ended_at=time.time())
            logger.error("Actuation op=%s failed in executor: %s", operation_id, exc)
            return {"ok": False, "error": f"executor_error:{exc}", "operation_id": operation_id, "outcome": "uncertain"}

        if not isinstance(result, dict):
            self._update(record, status="malformed_result", ended_at=time.time())
            return {"ok": False, "error": "executor_returned_non_dict", "operation_id": operation_id, "outcome": "uncertain"}

        self._update(record, status="success" if result.get("ok") else "failed",
                     result_digest=_digest(result), ended_at=time.time())
        result.setdefault("operation_id", operation_id)
        return result


_actuator_instance: WorldActuator | None = None
_instance_lock = checked_lock("world_actuator.singleton", rank=LockRank.REGISTRY)


def get_world_actuator() -> WorldActuator:
    global _actuator_instance
    if _actuator_instance is None:
        with _instance_lock:
            if _actuator_instance is None:
                _actuator_instance = WorldActuator()
                try:
                    from core.container import ServiceContainer
                    ServiceContainer.register_instance("world_actuator", _actuator_instance, required=False)
                except (ImportError, RuntimeError, AttributeError, TypeError):
                    pass  # registration is best-effort; the singleton still stands
    return _actuator_instance
