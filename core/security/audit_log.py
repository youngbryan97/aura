"""core/security/audit_log.py
Durable append-only audit logger for tracking agent executions.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any

from core.config import get_config
from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger("Security.AuditLogger")

_AUDIT_LOG_ERRORS = (OSError, RuntimeError, TypeError, ValueError)


class SecurityAuditLogger:
    """Writes security events to an append-only file in the logs directory."""

    def __init__(self):
        self.config = get_config()
        self.log_path = Path(self.config.paths.log_dir) / "security_audit.jsonl"

    def log_event(self, action: str, details: dict[str, Any]) -> None:
        event = {
            "timestamp": time.time(),
            "action": action,
            "details": details
        }
        try:
            with local_internal_governed_scope(
                "security.audit_log",
                domain="file_write",
                receipt_prefix="security-audit-log",
            ):
                get_file_write_gateway().append_text(
                    self.log_path,
                    json.dumps(event, sort_keys=True) + "\n",
                    source="security.audit_log",
                )
        except _AUDIT_LOG_ERRORS as e:
            record_degradation("security.audit_log", e)
            logger.error("Failed to append security audit event: %s", e)
