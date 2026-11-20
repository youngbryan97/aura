"""Exact policy contract for Aura's autonomous idle sandbox probe."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping
from typing import Any

IDLE_SANDBOX_PROBE_PURPOSE = "idle_probe"
#: A canary: her code runner runs, and says on what. It computes and prints only, so it is rated as a snippet that
#: computes (core/executive/execution_policy.classify_execution_risk). It read ``platform`` until 20 November 2026,
#: which can start a process, so it was rated high, needed a confirmation nobody is there to give, and was deferred by
#: her will 313 times from 10 November without running once.
IDLE_SANDBOX_PROBE_SCRIPT = """\
# Autonomous Subconscious Check: the code runner runs, and says on what.
import sys
print(f"Subconscious ping: Python {sys.version_info.major}.{sys.version_info.minor} on {sys.platform}")
"""
IDLE_SANDBOX_PROBE_SCRIPT_SHA256 = hashlib.sha256(
    IDLE_SANDBOX_PROBE_SCRIPT.encode("utf-8")
).hexdigest()

_IDLE_SANDBOX_ARGUMENT_KEYS = frozenset({"purpose", "script_sha256"})


def idle_sandbox_probe_arguments() -> dict[str, str]:
    """Return the complete argument envelope bound into the child lease."""

    return {
        "purpose": IDLE_SANDBOX_PROBE_PURPOSE,
        "script_sha256": IDLE_SANDBOX_PROBE_SCRIPT_SHA256,
    }


def validate_idle_sandbox_probe_arguments(
    arguments: Mapping[str, Any],
) -> tuple[bool, str]:
    """Accept only the checked-in probe and reject payload substitution."""

    if frozenset(str(key) for key in arguments) != _IDLE_SANDBOX_ARGUMENT_KEYS:
        return False, "bounded_sandbox_probe_argument_shape_mismatch"
    if str(arguments.get("purpose") or "") != IDLE_SANDBOX_PROBE_PURPOSE:
        return False, "bounded_sandbox_probe_purpose_mismatch"
    supplied_digest = str(arguments.get("script_sha256") or "").strip().lower()
    if not hmac.compare_digest(supplied_digest, IDLE_SANDBOX_PROBE_SCRIPT_SHA256):
        return False, "bounded_sandbox_probe_script_mismatch"
    return True, "bounded_sandbox_probe"
