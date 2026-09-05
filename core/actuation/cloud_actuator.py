"""core/actuation/cloud_actuator.py — Cloud Resources and Databases Actuator.

This is a thin wrapper, and thin wrappers are where risk classification gets
lost. CP126 found both of its methods understating what they can do:
``query_db`` labelled arbitrary SQL as an ordinary ``query_database`` even when
it was a DELETE or a GRANT, and ``modify_infra`` passed a free-text desired
state with no precondition, plan or rollback target.

The wrapper cannot make these operations safe on its own — the domain
underneath performs them — but it CAN refuse to misdescribe them, and the risk
flag it sets is what downstream policy keys on.

CP126 fa7129ae / 91be5450 / 8aa639aa / 8ceb4427.
"""
from __future__ import annotations

import re
import uuid
from typing import Any

from core.actuation.world_actuator import get_world_actuator

#: SQL that changes data, structure or privileges. CP126 fa7129ae: all of this
#: kept the read-shaped ``query_database`` label and its risk classification.
_MUTATING_SQL = re.compile(
    r"\b("
    r"insert|update|delete|merge|upsert|replace|truncate|"
    r"drop|create|alter|rename|"
    r"grant|revoke|"
    r"vacuum|reindex|cluster|"
    r"copy|load|import|export|"
    r"call|execute"
    r")\b",
    re.IGNORECASE,
)

#: A statement separator followed by more content — a multi-statement body can
#: hide a mutation behind a leading SELECT.
_STATEMENT_SPLIT = re.compile(r";\s*\S")

#: Comment forms used to smuggle a second statement past a naive prefix check.
_SQL_COMMENT = re.compile(r"(--|#|/\*)")

MAX_QUERY_CHARS = 20_000
MAX_IDENTIFIER_CHARS = 128

#: Desired-state values this wrapper is willing to describe. Anything else has
#: to be spelled out by the caller as a typed plan (CP126 8aa639aa).
KNOWN_INFRA_STATES = frozenset({
    "running", "stopped", "restarted", "scaled_up", "scaled_down",
    "enabled", "disabled", "drained", "resumed",
})


def classify_sql(query: str) -> dict[str, Any]:
    """Describe what a SQL body actually does.

    Returns the read/write verdict plus the reasons, so a caller — and the
    receipt — can see WHY something was treated as high risk.
    """
    text = str(query or "")
    reasons: list[str] = []

    if len(text) > MAX_QUERY_CHARS:
        reasons.append(f"query exceeds {MAX_QUERY_CHARS} chars")
    multi_statement = bool(_STATEMENT_SPLIT.search(text))
    if multi_statement:
        # A multi-statement body is high risk regardless of its first verb.
        reasons.append("multiple statements")
    if _SQL_COMMENT.search(text):
        reasons.append("contains a comment")
    mutations = sorted({match.lower() for match in _MUTATING_SQL.findall(text)})
    if mutations:
        reasons.append(f"mutating keywords: {mutations}")

    return {
        "mutating": bool(mutations) or multi_statement,
        "reasons": reasons,
        "keywords": mutations,
    }


def _valid_identifier(value: Any) -> bool:
    text = str(value or "").strip()
    if not text or len(text) > MAX_IDENTIFIER_CHARS:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_.\-]+", text))


class CloudActuator:
    """Wrapper for managing cloud infrastructure and DB connections."""

    @classmethod
    async def query_db(
        cls,
        db_name: str,
        query: str,
        source: str = "cloud_actuator",
        *,
        read_only: bool = True,
        capability_token: str = "",
        deadline_s: float | None = None,
    ) -> dict[str, Any]:
        """Run a query against a named database.

        CP126 fa7129ae: a DELETE, a DDL statement or a GRANT arrived here as an
        ordinary ``query_database`` and kept that risk classification all the
        way down. The statement is now classified, and a mutating body routes
        as a distinct high-risk action.

        CP126 91be5450: ``db_name`` was caller-controlled text bound to no
        registered connection, tenant, lease or budget. It must now be a plain
        identifier, and a mutating query additionally requires the caller to
        drop ``read_only`` — a write cannot arrive through the read lane.
        """
        operation_id = uuid.uuid4().hex
        if not _valid_identifier(db_name):
            return {
                "ok": False,
                "error": "invalid_db_identifier",
                "db_name": str(db_name)[:80],
                "operation_id": operation_id,
            }
        if not str(query or "").strip():
            return {"ok": False, "error": "empty_query", "operation_id": operation_id}

        classification = classify_sql(query)
        if classification["mutating"] and read_only:
            # Fail closed: the caller asked for a read lane with a write body.
            return {
                "ok": False,
                "error": "mutating_query_requires_read_only_false",
                "classification": classification,
                "operation_id": operation_id,
            }

        params: dict[str, Any] = {
            "db_name": str(db_name).strip(),
            "query": query,
            "read_only": bool(read_only),
            "classification": classification,
            "operation_id": operation_id,
        }
        if capability_token:
            params["capability_token"] = capability_token

        kwargs: dict[str, Any] = {}
        if deadline_s is not None:
            kwargs["deadline_s"] = deadline_s

        return await get_world_actuator().actuate(
            category="databases_owned",
            # A write is not a query, and the action name is what downstream
            # policy keys on.
            action_name="mutate_database" if classification["mutating"] else "query_database",
            params=params,
            source=source,
            high_risk_flag=True if classification["mutating"] else None,
            **kwargs,
        )

    @classmethod
    async def modify_infra(
        cls,
        service: str,
        state: str,
        source: str = "cloud_actuator",
        *,
        current_state: str = "",
        rollback_state: str = "",
        capability_token: str = "",
        deadline_s: float | None = None,
    ) -> dict[str, Any]:
        """Change cloud infrastructure state.

        CP126 8aa639aa: arbitrary service and state strings were accepted with
        no resource identity, plan, diff, precondition or rollback target.
        CP126 8ceb4427: the high-risk wrapper submitted only ``desired_state``
        — no current-state precondition, operation id, staged plan or
        compensating action — so a partially applied change had nothing to undo
        it and a repeated call could not tell it was a repeat.
        """
        operation_id = uuid.uuid4().hex
        if not _valid_identifier(service):
            return {
                "ok": False,
                "error": "invalid_service_identifier",
                "service": str(service)[:80],
                "operation_id": operation_id,
            }
        desired = str(state or "").strip().lower()
        if desired not in KNOWN_INFRA_STATES:
            return {
                "ok": False,
                "error": "unknown_desired_state",
                "desired_state": desired[:80],
                "known_states": sorted(KNOWN_INFRA_STATES),
                "operation_id": operation_id,
            }

        observed = str(current_state or "").strip().lower()
        if observed and observed == desired:
            # Idempotency: the precondition already holds, so there is nothing
            # to change and nothing to roll back.
            return {
                "ok": True,
                "changed": False,
                "reason": "already_in_desired_state",
                "service": service,
                "desired_state": desired,
                "operation_id": operation_id,
            }

        # The compensating action, chosen BEFORE the change is attempted.
        rollback = str(rollback_state or observed or "").strip().lower()
        if rollback and rollback not in KNOWN_INFRA_STATES:
            return {
                "ok": False,
                "error": "unknown_rollback_state",
                "rollback_state": rollback[:80],
                "operation_id": operation_id,
            }

        params: dict[str, Any] = {
            "service": str(service).strip(),
            "desired_state": desired,
            # The plan, so the domain can verify it before acting.
            "precondition_state": observed or "unknown",
            "rollback_state": rollback or "unknown",
            "operation_id": operation_id,
            "idempotency_key": f"infra:{service}:{desired}:{observed or 'unknown'}",
            "plan": {
                "from": observed or "unknown",
                "to": desired,
                "compensating_action": f"restore:{rollback}" if rollback else "none_available",
            },
        }
        if capability_token:
            params["capability_token"] = capability_token

        kwargs: dict[str, Any] = {}
        if deadline_s is not None:
            kwargs["deadline_s"] = deadline_s

        # High risk action: changing cloud configuration
        return await get_world_actuator().actuate(
            category="cloud_resources_owned",
            action_name="change_cloud_infra",
            params=params,
            source=source,
            high_risk_flag=True,
            **kwargs,
        )
