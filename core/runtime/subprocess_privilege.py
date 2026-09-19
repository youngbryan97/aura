"""What a subprocess is allowed to be, checked before it exists.

Lifted whole out of `subprocess_gateway`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence


def _validate_read_only_source(source: str) -> None:
    if not isinstance(source, str) or source.strip() in {"", "unknown"}:
        raise ValueError("read-only subprocess probes require a specific source label")
    if "\n" in source or "\r" in source:
        raise ValueError("subprocess source label must be single-line")


def _truthy_env_value(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _effective_env_value(env: Mapping[str, str] | None, key: str) -> str | None:
    if env is not None:
        value = env.get(key)
        return str(value) if value is not None else None
    return os.getenv(key)


def _desktop_safe_mode_requested(env: Mapping[str, str] | None) -> bool:
    from .subprocess_gateway import (
        _effective_env_value,
        _truthy_env_value,
    )

    return _truthy_env_value(_effective_env_value(env, "AURA_DESKTOP_RESOURCE_GUARD")) or _truthy_env_value(
        _effective_env_value(env, "AURA_SAFE_BOOT_DESKTOP")
    ) or _truthy_env_value(
        _effective_env_value(env, "AURA_LAUNCHED_FROM_APP")
    )


def _desktop_longrun_override(env: Mapping[str, str] | None) -> bool:
    from .subprocess_gateway import (
        _effective_env_value,
        _truthy_env_value,
    )

    return _truthy_env_value(_effective_env_value(env, "AURA_ALLOW_DESKTOP_LONGRUNS")) or _truthy_env_value(
        _effective_env_value(env, "AURA_ALLOW_DESKTOP_NETHACK")
    )


def _validate_delegated_governance_environment(
    env: Mapping[str, str] | None,
    *,
    source: str,
) -> None:
    """Bind delegated child provenance to the active in-process receipt."""
    from .subprocess_gateway import (
        _DELEGATED_GOVERNANCE_ENV_KEYS,
        GovernanceViolation,
        _effective_env_value,
        _governance_context,
    )

    mode = str(_effective_env_value(env, "AURA_GOVERNANCE_MODE") or "").strip()
    values = {
        key: str(_effective_env_value(env, key) or "").strip()
        for key in _DELEGATED_GOVERNANCE_ENV_KEYS
    }
    if mode != "delegated_subprocess" and not any(values.values()):
        return
    if mode != "delegated_subprocess" or not all(values.values()):
        raise GovernanceViolation("incomplete delegated governance environment")
    if values["AURA_DELEGATED_GOVERNANCE_PARENT_PID"] != str(os.getpid()):
        raise GovernanceViolation("delegated governance parent PID mismatch")

    token = _governance_context.get_active_governance()
    constraints = dict(getattr(token, "constraints", ()) or ()) if token else {}
    expected_source = values["AURA_DELEGATED_GOVERNANCE_SOURCE"]
    if (
        token is None
        or token.receipt_id != values["AURA_DELEGATED_GOVERNANCE_RECEIPT_ID"]
        or token.domain != values["AURA_DELEGATED_GOVERNANCE_DOMAIN"]
        or token.source != expected_source
        or constraints.get("executive_intent_id")
        != values["AURA_DELEGATED_AUTHORITY_INTENT_ID"]
        or not (source == expected_source or source.startswith(f"{expected_source}:"))
    ):
        raise GovernanceViolation("delegated governance receipt does not match active scope")


def _validate_desktop_safe_subprocess(
    command: Sequence[str] | str,
    *,
    env: Mapping[str, str] | None,
    source: str,
    operation: str,
) -> None:
    """Prevent desktop boot/chat sessions from launching proof-scale child jobs.

    Long environment batteries are valid proof tooling, but they are not part of
    the live user desktop lane. They can exceed desktop memory budgets when they
    are started by a stale shell, launch agent, or task handoff. An explicit
    operator opt-in keeps proof work possible while making false "normal desktop"
    launches fail closed.
    """
    from .subprocess_gateway import (
        _DESKTOP_LONGRUN_COMMAND_MARKERS,
        GovernanceViolation,
        _desktop_longrun_override,
        _desktop_safe_mode_requested,
    )

    if not _desktop_safe_mode_requested(env) or _desktop_longrun_override(env):
        return
    if isinstance(command, str):
        normalized = command
    else:
        normalized = " ".join(str(part) for part in command)
    lowered = normalized.lower()
    if any(marker in lowered for marker in _DESKTOP_LONGRUN_COMMAND_MARKERS):
        raise GovernanceViolation(
            f"{operation}:{source} denied desktop-safe long-run subprocess; "
            "set AURA_ALLOW_DESKTOP_LONGRUNS=1 for an intentional proof run"
        )


def _enforce_process_privilege(
    *,
    env: Mapping[str, str] | None,
    source: str,
    operation: str,
) -> None:
    """Refuse to hand credentials to a process whose role may not hold them.

    Chromium's rule, applied to Aura's own children: the component that parses
    hostile input does not get the parent's secrets. A PDF decoder, a browser
    worker, or a run of generated code has no business inheriting an API key —
    if any of them is compromised, that key is the blast radius.

    Only what is mechanically checkable is checked. An explicitly-built env is
    inspectable, so a low-trust role receiving a secret in it is REFUSED.
    ``env=None`` means the child inherits Aura's entire environment, which is
    the larger exposure but not something this function can narrow without
    breaking every spawn that legitimately relies on inheritance; it is
    recorded as a degradation so the inheriting call sites become visible and
    can be given explicit envs, rather than being refused blind.
    """
    from core.runtime.process_privilege import Privilege, ProcessRole, check_spawn, role_for_source

    from .subprocess_gateway import (
        GovernanceViolation,
        _record_privilege_degradation,
        logger,
    )

    role = role_for_source(source)
    # Only the roles that exist to handle untrusted input are constrained here.
    # Constraining the coordinator would refuse the process that legitimately
    # holds everything, and constraining unknown roles would refuse traffic the
    # matrix has not learned yet.
    if role is None or role > ProcessRole.UNTRUSTED_CODE:
        return

    if env is None:
        _record_privilege_degradation(
            source=source,
            operation=operation,
            detail=(
                f"{role.name.lower()} spawn inherits the full parent environment; "
                "pass an explicit env so credentials are not handed to it"
            ),
        )
        return

    try:
        from core.security.structural_redaction import is_sensitive_key
    except Exception as exc:  # noqa: BLE001 — security dependency must fail closed
        logger.error(
            "Refusing low-trust subprocess because secret-key classification failed: %s",
            exc,
        )
        raise GovernanceViolation(
            f"{operation}:{source} denied: secret-key classification unavailable"
        ) from exc

    leaked = sorted({str(key) for key in env if is_sensitive_key(key)})
    if not leaked:
        return

    decision = check_spawn(source, {Privilege.SECRETS}, role=role)
    if decision.allowed:
        return
    raise GovernanceViolation(
        f"{operation}:{source} denied: {decision.reason}; "
        f"environment carries {', '.join(leaked)}"
    )


def _record_privilege_degradation(*, source: str, operation: str, detail: str) -> None:
    """Report an inherited-environment spawn without ever blocking on the report."""
    from .subprocess_gateway import (
        logger,
    )

    try:
        from core.runtime.errors import record_degradation

        record_degradation(
            "subprocess_gateway",
            RuntimeError(detail),
            action=f"{operation}:{source} privilege_inheritance_unnarrowed",
        )
    except Exception:  # noqa: BLE001 — observability must not break spawning
        logger.debug("privilege degradation record failed for %s", source, exc_info=True)


def _validate_offline_tooling_bypass(
    *,
    offline_tooling: bool,
    source: str,
    command: Sequence[str],
    env: Mapping[str, str] | None = None,
) -> bool:
    """Allow named repo tooling to launch child processes outside live Aura.

    This is intentionally not a general governance bypass. It exists for CLI
    proof, certification, benchmark, maintenance, and training wrappers that
    orchestrate Aura from outside her live runtime. If live/strict governance is
    active, the bypass fails closed except for proof/certification harnesses
    running under AURA_TEST_MODE.
    """
    from .subprocess_gateway import (
        _OFFLINE_TOOLING_SOURCE_PREFIXES,
        _TEST_MODE_GOVERNANCE_BYPASS_PREFIXES,
        GovernanceViolation,
        governance_runtime_active,
        logger,
    )

    if not offline_tooling:
        return False
    if not any(source.startswith(prefix) for prefix in _OFFLINE_TOOLING_SOURCE_PREFIXES):
        raise ValueError(
            "offline subprocess tooling requires a source prefix of "
            f"{', '.join(_OFFLINE_TOOLING_SOURCE_PREFIXES)}"
        )
    if governance_runtime_active():
        is_certification_harness = any(
            source.startswith(prefix) for prefix in _TEST_MODE_GOVERNANCE_BYPASS_PREFIXES
        )
        explicit_test_mode = env is not None and str(env.get("AURA_TEST_MODE", "")) == "1"
        process_test_mode = os.getenv("AURA_TEST_MODE", "") == "1"
        if is_certification_harness and (process_test_mode or explicit_test_mode):
            logger.info(
                "offline subprocess tooling bypass (test-mode) source=%s argv0=%s argc=%s",
                source,
                command[0] if command else "",
                len(command),
            )
            return True
        raise GovernanceViolation(
            f"offline subprocess tooling bypass denied while live governance is active: {source}"
        )
    logger.info(
        "offline subprocess tooling bypass source=%s argv0=%s argc=%s",
        source,
        command[0] if command else "",
        len(command),
    )
    return True


def _require_effect_governance(operation: str) -> None:
    from .subprocess_gateway import (
        _EFFECT_DOMAINS,
        GovernanceViolation,
        governance_runtime_active,
        require_governance,
    )

    should_fail_closed = governance_runtime_active()
    token = require_governance(
        operation,
        strict=True,
        allowed_domains=_EFFECT_DOMAINS,
    )
    # `token.authorizes` rather than a hand-rolled domain string. This
    # checked only "degraded" and so missed the OTHER non-authority token —
    # domain "ungoverned", receipt "VIOLATION", handed back when a call is
    # made outside a governed context. Both record the ABSENCE of the
    # boundary; only one was being caught here.
    if should_fail_closed and (token is None or not getattr(token, "authorizes", False)):
        raise GovernanceViolation(f"{operation} called outside governed context")


