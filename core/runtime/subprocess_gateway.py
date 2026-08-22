"""Canonical subprocess gateway.

All live runtime subprocess creation should flow through this module. Effectful
calls require an active governance context; explicitly read-only probes may opt
out while still receiving consistent validation and logging behavior.
"""
from __future__ import annotations

import ast
import asyncio
import importlib.util
import logging
import multiprocessing as mp
import os
import shlex
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import IO, Any, Protocol, cast

from core import governance_context as _governance_context
from core.runtime.process_privilege import Privilege, ProcessRole, check_spawn
from core.runtime.shutdown_coordinator import (
    is_shutdown_requested,
    record_shutdown_admission_event,
)
from core.runtime.shutdown_execution import run_sync_shutdown_callable
from core.utils.task_tracker import (
    begin_shutdown_resource_creation_scope,
    end_shutdown_resource_creation_scope,
)

GovernanceViolation = _governance_context.GovernanceViolation


class _PythonProcessFactory(Protocol):
    def __call__(
        self,
        *,
        target: Callable[..., Any],
        args: tuple[Any, ...],
        name: str,
        daemon: bool,
    ) -> Any: ...


class _PythonProcessContext(Protocol):
    Process: _PythonProcessFactory

    def get_start_method(self, allow_none: bool = False) -> str | None: ...

_EFFECT_DOMAINS = (
    "environment_action",
    "external_action",
    "tool_execution",
    "state_mutation",
    "file_write",
    "semantic_weight_update",
    "self_modification",
)
_OFFLINE_TOOLING_SOURCE_PREFIXES = (
    "benchmark_tooling:",
    "certification_tooling:",
    "maintenance_tooling:",
    "proof_tooling:",
    "training_tooling:",
)
_TEST_MODE_GOVERNANCE_BYPASS_PREFIXES = (
    "certification_tooling:",
    "proof_tooling:",
)
_DESKTOP_LONGRUN_COMMAND_MARKERS = (
    "challenges/nethack_challenge.py",
    "nethack_challenge.py",
    "run_dnu_agi_proof_battery.py",
    "run_longevity_soak.py",
    "aletheia_tier5",
    "run_aletheia",
)
_DELEGATED_GOVERNANCE_ENV_KEYS = (
    "AURA_DELEGATED_GOVERNANCE_RECEIPT_ID",
    "AURA_DELEGATED_GOVERNANCE_DOMAIN",
    "AURA_DELEGATED_GOVERNANCE_SOURCE",
    "AURA_DELEGATED_AUTHORITY_INTENT_ID",
    "AURA_DELEGATED_GOVERNANCE_PARENT_PID",
)
_INHERITED_MODEL_LANE_ENV_KEYS = (
    "AURA_MODEL_LANE_INHERITED_OWNER_ID",
    "AURA_MODEL_LANE_INHERITED_REQUEST_ID",
    "AURA_MODEL_LANE_INHERITED_MODEL_PATH",
    "AURA_MODEL_LANE_INHERITED_PURPOSE",
    "AURA_MODEL_LANE_DELEGATION_TOKEN",
)
logger = logging.getLogger("Aura.SubprocessGateway")
_PROCESS_CONTRACT_ATTRIBUTE = "_aura_python_process_contract"


class AcceleratorCapability(StrEnum):
    """Caller-owned declaration of a child process's accelerator behavior."""

    NONE = "none"
    MODEL = "model"
    AUTO = "auto"


@dataclass(frozen=True)
class PythonProcessSpec:
    """Complete admission contract for one Python multiprocessing child."""

    target: Callable[..., Any]
    source: str
    name: str
    role: ProcessRole
    accelerator_capability: AcceleratorCapability | str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    requested_privileges: frozenset[Privilege] = field(default_factory=frozenset)
    daemon: bool = False
    start_method: str = "spawn"
    environment_overrides: Mapping[str, str] = field(default_factory=dict)


class PythonProcessOwnershipError(RuntimeError):
    """A Python child could not satisfy the gateway ownership contract."""


def python_process_contract(process: Any) -> dict[str, Any] | None:
    """Return the gateway-authored contract attached to a process handle.

    Command lines cannot distinguish multiprocessing children: model workers,
    coordinators and state organs all execute the same ``spawn_main`` bootstrap.
    The parent-owned handle is the authoritative identity surface because the
    gateway attaches this contract before ``start()`` and retains the handle for
    lifecycle operations.
    """

    raw = getattr(process, _PROCESS_CONTRACT_ATTRIBUTE, None)
    if not isinstance(raw, Mapping):
        return None
    required = {
        "source",
        "name",
        "role",
        "requested_privileges",
        "accelerator_capability",
        "start_method",
    }
    if not required.issubset(raw):
        return None
    return {str(key): value for key, value in raw.items()}


def python_process_role(process: Any) -> ProcessRole | None:
    """Resolve the declared role of one gateway-owned process handle."""

    contract = python_process_contract(process)
    if contract is None:
        return None
    role = str(contract.get("role") or "").strip().upper()
    try:
        return ProcessRole[role]
    except KeyError:
        return None


_ACCELERATOR_IMPORT_ROOTS = frozenset(
    {
        "jax",
        "llama_cpp",
        "mlx",
        "mlx_lm",
        "tensorflow",
        "torch",
        "transformers",
        "vllm",
    }
)


def _inferred_model_lane_claim(
    command: Sequence[str],
    *,
    source: str,
    timeout_s: float,
) -> Any | None:
    from core.runtime.model_lane_control import infer_model_process_claim

    return infer_model_process_claim(
        command,
        source=source,
        timeout_s=timeout_s,
    )


def _declared_model_lane_claim(
    command: Sequence[str],
    *,
    source: str,
    timeout_s: float,
) -> Any:
    from core.runtime.model_lane_control import declared_model_process_claim

    return declared_model_process_claim(
        command,
        source=source,
        timeout_s=timeout_s,
    )


def _coerce_accelerator_capability(
    value: AcceleratorCapability | str | None,
    *,
    source: str,
) -> AcceleratorCapability:
    if value is None:
        raise GovernanceViolation(
            f"subprocess_accelerator_capability_undeclared:{source}"
        )
    try:
        return AcceleratorCapability(str(value).strip().lower())
    except ValueError as exc:
        raise GovernanceViolation(
            f"subprocess_accelerator_capability_invalid:{source}"
        ) from exc


def _python_source_for_command(command: Sequence[str]) -> str | None:
    """Return inspectable Python source without importing the target module."""

    if not command:
        return None
    executable = Path(str(command[0])).name.lower()
    if not executable.startswith("python") and executable != Path(sys.executable).name.lower():
        return None
    argv = [str(part) for part in command[1:]]
    if "-c" in argv:
        index = argv.index("-c")
        return argv[index + 1] if index + 1 < len(argv) else None
    if "-m" in argv:
        index = argv.index("-m")
        if index + 1 >= len(argv):
            return None
        module_name = argv[index + 1]
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, AttributeError, ModuleNotFoundError, ValueError):
            return None
        origin = str(getattr(spec, "origin", "") or "")
        if not origin or origin in {"built-in", "frozen"}:
            return None
        path = Path(origin)
    else:
        script = next((part for part in argv if part.endswith((".py", ".pyw"))), "")
        if not script:
            return None
        path = Path(script)
        if not path.is_absolute():
            path = Path.cwd() / path
    try:
        if path.stat().st_size > 4 * 1024 * 1024:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _source_declares_accelerator_import(source_text: str) -> bool:
    try:
        module = ast.parse(source_text)
    except (SyntaxError, ValueError):
        return True
    for node in ast.walk(module):
        imported: tuple[str, ...] = ()
        if isinstance(node, ast.Import):
            imported = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported = (str(node.module or ""),)
        if any(name.split(".", 1)[0] in _ACCELERATOR_IMPORT_ROOTS for name in imported):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "__import__" and node.args:
                first = node.args[0]
                if (
                    isinstance(first, ast.Constant)
                    and isinstance(first.value, str)
                    and first.value.split(".", 1)[0] in _ACCELERATOR_IMPORT_ROOTS
                ):
                    return True
    return False


def _dynamic_command_accelerator_use(command: Sequence[str]) -> bool | None:
    """Inspect a dynamic executable without running or importing it.

    ``None`` means the gateway could not establish what the executable is. The
    caller must then provide a concrete model claim or use a reviewed fixed
    command declaration; uncertainty never silently becomes ``none``.
    """

    source_text = _python_source_for_command(command)
    if source_text is not None:
        return _source_declares_accelerator_import(source_text)
    if not command:
        return None
    executable_name = Path(str(command[0])).name.lower()
    if executable_name.startswith("python") or executable_name == Path(sys.executable).name.lower():
        return None
    executable = shutil.which(str(command[0]))
    if executable is None:
        candidate = Path(str(command[0])).expanduser()
        executable = str(candidate) if candidate.is_file() else None
    if executable is None:
        return None
    path = Path(executable)
    try:
        size = path.stat().st_size
        if size <= 0 or size > 64 * 1024 * 1024:
            return None
        payload = path.read_bytes()
    except OSError:
        return None
    if payload.startswith(b"#!"):
        try:
            script_text = payload.decode("utf-8")
        except UnicodeError:
            return None
        return _source_declares_accelerator_import(script_text)
    accelerator_markers = (
        b"Metal.framework",
        b"libmlx",
        b"libtorch",
        b"libtensorflow",
        b"libjax",
        b"MTLCreateSystemDefaultDevice",
    )
    return any(marker in payload for marker in accelerator_markers)


def _resolve_accelerator_claim(
    command: Sequence[str],
    *,
    source: str,
    timeout_s: float,
    accelerator_capability: AcceleratorCapability | str | None,
    model_lane_claim: Any | None = None,
) -> Any | None:
    declaration = _coerce_accelerator_capability(
        accelerator_capability,
        source=source,
    )
    inferred = _inferred_model_lane_claim(
        command,
        source=source,
        timeout_s=timeout_s,
    )
    from core.runtime.model_lane_control import is_registered_non_model_process_command

    registered_probe = is_registered_non_model_process_command(command)
    if declaration is AcceleratorCapability.NONE:
        source_text = _python_source_for_command(command)
        source_uses_accelerator = bool(
            source_text is not None and _source_declares_accelerator_import(source_text)
        )
        if model_lane_claim is not None or inferred is not None or (
            source_uses_accelerator and not registered_probe
        ):
            raise GovernanceViolation(
                f"subprocess_accelerator_capability_contradiction:{source}"
            )
        return None
    if declaration is AcceleratorCapability.MODEL:
        return model_lane_claim or inferred or _declared_model_lane_claim(
            command,
            source=source,
            timeout_s=timeout_s,
        )

    if model_lane_claim is not None or inferred is not None:
        return model_lane_claim or inferred
    if registered_probe:
        return None
    dynamic_accelerator_use = _dynamic_command_accelerator_use(command)
    if dynamic_accelerator_use is None:
        raise GovernanceViolation(
            f"subprocess_accelerator_capability_unresolved:{source}"
        )
    if dynamic_accelerator_use:
        return _declared_model_lane_claim(
            command,
            source=source,
            timeout_s=timeout_s,
        )
    return None


async def _reserve_model_lane_process(
    claim: Any,
) -> tuple[Any, Any]:
    from core.runtime.model_lane_control import prepare_model_lane_claim

    prepared: tuple[Any, Any] = await prepare_model_lane_claim(claim)
    return prepared


async def _cancel_model_lane_process(
    controller: Any,
    decision: Any,
    *,
    reason: str,
) -> None:
    try:
        await controller.cancel(decision, reason=reason)
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        logger.error(
            "Model subprocess reservation cancellation failed transaction=%s: %s",
            getattr(decision, "transaction_id", ""),
            exc,
        )


def _model_command_requires_async(
    command: Sequence[str],
    *,
    source: str,
    accelerator_capability: AcceleratorCapability | str | None,
) -> None:
    claim = _resolve_accelerator_claim(
        command,
        source=source,
        timeout_s=30.0,
        accelerator_capability=accelerator_capability,
    )
    if claim is not None:
        raise RuntimeError(
            "accelerator-owning subprocesses require run_async/spawn_async so "
            "their durable lane reservation can follow the child lifecycle"
        )


def _register_runtime_hygiene_process(
    proc: Any,
    *,
    kind: str,
    source: str,
    command: Sequence[str] | str,
) -> bool:
    """Register gateway-spawned children with runtime hygiene when available."""

    try:
        from core.runtime.runtime_hygiene import get_runtime_hygiene

        if isinstance(command, str):
            command_text = command
        else:
            command_text = " ".join(str(part) for part in command)
        hygiene = get_runtime_hygiene()
        hygiene.register_process_handle(
            proc,
            kind=kind,
            name=source or kind,
            source=f"subprocess_gateway:{source or 'unknown'}",
            command=command_text,
        )
        return bool(hygiene.process_handle_is_registered(proc))
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("runtime hygiene registration skipped for subprocess gateway child: %s", exc)
        return False


def governance_runtime_active() -> bool:
    return bool(_governance_context.governance_runtime_active())


def require_governance(*args: Any, **kwargs: Any) -> Any:
    return _governance_context.require_governance(*args, **kwargs)


def _coerce_argv(argv: Sequence[str]) -> list[str]:
    if not isinstance(argv, (list, tuple)) or not argv:
        raise ValueError("argv must be a non-empty list or tuple")
    coerced = [str(part) for part in argv]
    if any(not part for part in coerced):
        raise ValueError("argv entries must not be empty")
    return coerced


def _coerce_cwd(cwd: str | os.PathLike[str] | None) -> str | None:
    if cwd is None:
        return None
    return str(Path(cwd).expanduser().resolve())


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
    return _truthy_env_value(_effective_env_value(env, "AURA_DESKTOP_RESOURCE_GUARD")) or _truthy_env_value(
        _effective_env_value(env, "AURA_SAFE_BOOT_DESKTOP")
    ) or _truthy_env_value(
        _effective_env_value(env, "AURA_LAUNCHED_FROM_APP")
    )


def _desktop_longrun_override(env: Mapping[str, str] | None) -> bool:
    return _truthy_env_value(_effective_env_value(env, "AURA_ALLOW_DESKTOP_LONGRUNS")) or _truthy_env_value(
        _effective_env_value(env, "AURA_ALLOW_DESKTOP_NETHACK")
    )


def _validate_delegated_governance_environment(
    env: Mapping[str, str] | None,
    *,
    source: str,
) -> None:
    """Bind delegated child provenance to the active in-process receipt."""
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
    try:
        from core.runtime.errors import record_degradation

        record_degradation(
            "subprocess_gateway",
            RuntimeError(detail),
            action=f"{operation}:{source} privilege_inheritance_unnarrowed",
        )
    except Exception:  # noqa: BLE001 — observability must not break spawning
        logger.debug("privilege degradation record failed for %s", source, exc_info=True)


def _open_spawn_stream(path: str | os.PathLike[str], *, text: bool) -> IO[Any]:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(str(target), flags, 0o600)
    try:
        return os.fdopen(fd, "w", encoding="utf-8") if text else os.fdopen(fd, "wb")
    except (OSError, ValueError):
        os.close(fd)
        raise


def _python_process_entrypoint(
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    environment_overrides: dict[str, str],
    scrub_secrets: bool,
) -> None:
    """Pickle-safe child entrypoint that applies the declared environment."""

    if scrub_secrets:
        from core.security.structural_redaction import is_sensitive_key

        for key in tuple(os.environ):
            if is_sensitive_key(key):
                os.environ.pop(key, None)
    os.environ.update(environment_overrides)
    target(*args, **kwargs)


def _terminate_and_reap_python_process(
    process: Any,
    *,
    terminate_timeout_s: float = 1.0,
    kill_timeout_s: float = 1.0,
) -> bool:
    """Bounded handle-based termination; never signal an observed PID directly."""

    try:
        alive = bool(process.is_alive())
    except (AssertionError, AttributeError, OSError, RuntimeError, ValueError):
        alive = True
    if alive:
        try:
            process.terminate()
        except (AttributeError, OSError, RuntimeError, ValueError):
            pass
    try:
        process.join(timeout=max(0.0, float(terminate_timeout_s)))
    except (AssertionError, AttributeError, OSError, RuntimeError, ValueError):
        pass
    try:
        alive = bool(process.is_alive())
    except (AssertionError, AttributeError, OSError, RuntimeError, ValueError):
        alive = True
    if alive:
        try:
            process.kill()
        except (AttributeError, OSError, RuntimeError, ValueError):
            pass
        try:
            process.join(timeout=max(0.0, float(kill_timeout_s)))
        except (AssertionError, AttributeError, OSError, RuntimeError, ValueError):
            pass
    try:
        return not bool(process.is_alive())
    except (AssertionError, AttributeError, OSError, RuntimeError, ValueError):
        return False


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


async def _terminate_async_process_group(
    process: asyncio.subprocess.Process,
    *,
    grace_s: float = 5.0,
) -> tuple[bytes | None, bytes | None]:
    """Terminate and reap an isolated async process and all descendants."""
    process_group_id = int(getattr(process, "_aura_process_group_id", 0) or 0)
    try:
        if process_group_id <= 0:
            process_group_id = int(os.getpgid(process.pid))
    except (OSError, ProcessLookupError, ValueError):
        if bool(getattr(process, "_aura_start_new_session", False)):
            process_group_id = int(process.pid)

    try:
        if process_group_id > 0 and process_group_id != os.getpgrp():
            os.killpg(process_group_id, signal.SIGTERM)
        else:
            process.terminate()
    except (OSError, ProcessLookupError):
        pass
    try:
        return await asyncio.wait_for(process.communicate(), timeout=max(0.1, grace_s))
    except TimeoutError:
        try:
            if process_group_id > 0 and process_group_id != os.getpgrp():
                os.killpg(process_group_id, signal.SIGKILL)
            else:
                process.kill()
        except (OSError, ProcessLookupError):
            pass
        try:
            return await asyncio.wait_for(process.communicate(), timeout=max(0.1, grace_s))
        except TimeoutError:
            logger.error("Async subprocess group could not be reaped pid=%s pgid=%s", process.pid, process_group_id)
            return None, None


def _require_not_shutting_down(
    operation: str,
    *,
    read_only: bool,
    offline_tooling: bool,
    allow_during_shutdown: bool,
    resource_created: bool = False,
    bounded_completion: bool = False,
) -> None:
    """Block new live subprocess work after the process shutdown latch is set."""

    if not is_shutdown_requested():
        return
    # External proof/certification tools run in their own process and therefore
    # do not inherit the stopped runtime's latch. An in-process exception must
    # be explicit and may only be used for non-effectful inspection.
    if allow_during_shutdown and read_only and bounded_completion:
        if not resource_created:
            record_shutdown_admission_event(
                operation,
                resource_kind="subprocess",
                outcome="allowed_read_only",
                detail="explicit_shutdown_probe",
            )
        logger.warning("Allowing explicit shutdown-time subprocess probe: %s", operation)
        return
    record_shutdown_admission_event(
        operation,
        resource_kind="subprocess",
        outcome="crossed" if resource_created else "suppressed",
        detail="shutdown_latch",
    )
    raise GovernanceViolation(f"{operation} refused during runtime shutdown")


class SubprocessGateway:
    """Single owner for subprocess execution and spawning."""

    _PYTHON_PROCESS_CONTRACT_ATTRIBUTE = _PROCESS_CONTRACT_ATTRIBUTE

    def spawn_python_process(
        self,
        spec: PythonProcessSpec,
        *,
        context: Any | None = None,
    ) -> Any:
        """Construct, start, and ownership-commit one multiprocessing child."""

        if not isinstance(spec, PythonProcessSpec):
            raise TypeError("spec must be a PythonProcessSpec")
        source = str(spec.source or "").strip()
        if source in {"", "unknown"} or "\n" in source or "\r" in source:
            raise ValueError("Python process source must be a specific single-line label")
        if not callable(spec.target):
            raise TypeError("Python process target must be callable")
        if not str(spec.name or "").strip():
            raise ValueError("Python process name must be non-empty")
        if not isinstance(spec.role, ProcessRole):
            raise TypeError("Python process role must be a ProcessRole")
        if any(
            not isinstance(privilege, Privilege)
            for privilege in spec.requested_privileges
        ):
            raise TypeError("requested privileges must be Privilege values")

        accelerator = _coerce_accelerator_capability(
            spec.accelerator_capability,
            source=source,
        )
        if accelerator is AcceleratorCapability.AUTO:
            raise GovernanceViolation(
                f"python_process_accelerator_capability_must_be_explicit:{source}"
            )
        if (spec.role is ProcessRole.MODEL_WORKER) != (
            accelerator is AcceleratorCapability.MODEL
        ):
            raise GovernanceViolation(
                f"python_process_role_accelerator_mismatch:{source}"
            )
        decision = check_spawn(
            source,
            set(spec.requested_privileges),
            role=spec.role,
        )
        if not decision.allowed:
            denied = ",".join(
                sorted(privilege.value for privilege in decision.denied)
            )
            raise GovernanceViolation(
                f"python_process_privilege_denied:{source}:{denied}"
            )

        start_method = str(spec.start_method or "").strip().lower()
        if start_method not in {"spawn", "forkserver"}:
            raise ValueError("Python process start_method must be spawn or forkserver")
        selected_context = cast(
            _PythonProcessContext,
            context if context is not None else mp.get_context(start_method),
        )
        try:
            actual_start_method = str(selected_context.get_start_method()).lower()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("Python process context must expose its start method") from exc
        if actual_start_method != start_method:
            raise GovernanceViolation(
                f"python_process_start_method_mismatch:{source}:"
                f"expected={start_method}:actual={actual_start_method}"
            )

        environment_overrides = {
            str(key): str(value)
            for key, value in dict(spec.environment_overrides).items()
        }
        try:
            from core.security.structural_redaction import is_sensitive_key
        except Exception as exc:  # noqa: BLE001 - security dependency fails closed
            raise GovernanceViolation(
                f"python_process_secret_classifier_unavailable:{source}"
            ) from exc
        leaked_overrides = sorted(
            key for key in environment_overrides if is_sensitive_key(key)
        )
        secrets_requested = Privilege.SECRETS in spec.requested_privileges
        if leaked_overrides and not secrets_requested:
            raise GovernanceViolation(
                f"python_process_secret_override_denied:{source}:"
                f"{','.join(leaked_overrides)}"
            )

        operation = f"subprocess_gateway.spawn_python_process:{source}"
        _require_not_shutting_down(
            operation,
            read_only=False,
            offline_tooling=False,
            allow_during_shutdown=False,
        )
        contract = {
            "source": source,
            "name": str(spec.name),
            "role": spec.role.name.lower(),
            "requested_privileges": tuple(
                sorted(privilege.value for privilege in spec.requested_privileges)
            ),
            "accelerator_capability": accelerator.value,
            "start_method": start_method,
        }
        process = selected_context.Process(
            target=_python_process_entrypoint,
            args=(
                spec.target,
                tuple(spec.args),
                dict(spec.kwargs),
                environment_overrides,
                not secrets_requested,
            ),
            name=str(spec.name),
            daemon=bool(spec.daemon),
        )
        setattr(process, self._PYTHON_PROCESS_CONTRACT_ATTRIBUTE, contract)
        start_attempted = False
        try:
            _require_not_shutting_down(
                operation,
                read_only=False,
                offline_tooling=False,
                allow_during_shutdown=False,
            )
            start_attempted = True
            process.start()
            registered = _register_runtime_hygiene_process(
                process,
                kind="multiprocessing",
                source=source,
                command=f"python-multiprocessing:{spec.role.name.lower()}",
            )
            if not registered:
                raise PythonProcessOwnershipError(
                    f"python_process_registration_failed:{source}"
                )
            _require_not_shutting_down(
                operation,
                read_only=False,
                offline_tooling=False,
                allow_during_shutdown=False,
                resource_created=True,
            )
        except (
            AssertionError,
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            GovernanceViolation,
        ):
            if start_attempted:
                reaped = _terminate_and_reap_python_process(process)
                record_shutdown_admission_event(
                    operation,
                    resource_kind="multiprocessing",
                    outcome="reaped" if reaped else "survived",
                    detail=f"pid={getattr(process, 'pid', None)}",
                )
                if reaped or getattr(process, "pid", None) is None:
                    try:
                        process.close()
                    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                        pass
            raise
        return process

    @staticmethod
    def terminate_python_process(
        process: Any,
        *,
        terminate_timeout_s: float = 1.0,
        kill_timeout_s: float = 1.0,
    ) -> bool:
        return _terminate_and_reap_python_process(
            process,
            terminate_timeout_s=terminate_timeout_s,
            kill_timeout_s=kill_timeout_s,
        )

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        read_only: bool = False,
        offline_tooling: bool = False,
        allow_during_shutdown: bool = False,
        capture_output: bool = True,
        input: str | bytes | None = None,
        check: bool = False,
        source: str = "unknown",
        # Byte-exact callers exist and must not be forced through a decode.
        # The recurrent-SFT kernel probe hashes stdout to prove containment;
        # decoding it to str first would launder the very bytes the receipt
        # attests to. An owner that cannot serve that need pushes its callers
        # back to a raw subprocess.run, which is how ownership erodes.
        text: bool = True,
        # A containment probe must not inherit the parent's stdin. Without
        # this the only way to close it was to bypass the gateway.
        stdin_devnull: bool = False,
        # File-backed stdout/stderr keep externally supplied tools from forcing
        # their entire output into Aura's memory. Callers must disable
        # capture_output when either stream is supplied, matching subprocess.run.
        stdout: int | IO[Any] | None = None,
        stderr: int | IO[Any] | None = None,
        accelerator_capability: AcceleratorCapability | str | None = None,
    ) -> subprocess.CompletedProcess[Any]:
        command = _coerce_argv(argv)
        _model_command_requires_async(
            command,
            source=source,
            accelerator_capability=accelerator_capability,
        )
        if read_only and not offline_tooling:
            _validate_read_only_source(source)
        offline_bypass = _validate_offline_tooling_bypass(
            offline_tooling=offline_tooling,
            source=source,
            command=command,
            env=env,
        )
        _validate_delegated_governance_environment(env, source=source)
        _require_not_shutting_down(
            f"subprocess_gateway.run:{source}",
            read_only=read_only,
            offline_tooling=offline_tooling,
            allow_during_shutdown=allow_during_shutdown,
            bounded_completion=True,
        )
        if not read_only and not offline_bypass:
            _require_effect_governance(f"subprocess_gateway.run:{source}")
        _validate_desktop_safe_subprocess(command, env=env, source=source, operation="run")
        resource_token = (
            begin_shutdown_resource_creation_scope()
            if is_shutdown_requested() and allow_during_shutdown and read_only
            else None
        )
        try:
            return subprocess.run(
                command,
                cwd=_coerce_cwd(cwd),
                env=dict(env) if env is not None else None,
                timeout=float(timeout),
                capture_output=bool(capture_output),
                input=input,
                stdin=subprocess.DEVNULL if (stdin_devnull and input is None) else None,
                stdout=stdout,
                stderr=stderr,
                text=bool(text),
                check=bool(check),
                shell=False,
            )
        finally:
            if resource_token is not None:
                end_shutdown_resource_creation_scope(resource_token)

    def run_model_blocking(
        self,
        argv: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        read_only: bool = False,
        offline_tooling: bool = False,
        allow_during_shutdown: bool = False,
        capture_output: bool = True,
        input: str | None = None,
        check: bool = False,
        source: str = "unknown",
        model_lane_claim: Any | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run one governed model child from a synchronous CLI entrypoint.

        The async path performs the reservation, eviction, delegated fencing,
        process-group monitoring, and terminal release. Calling this method from
        an event-loop thread is refused so a synchronous caller cannot stall the
        runtime loop; async code must await ``run_async`` directly.
        """
        command = _coerce_argv(argv)
        claim = _resolve_accelerator_claim(
            command,
            source=source,
            timeout_s=float(timeout),
            accelerator_capability=AcceleratorCapability.MODEL,
            model_lane_claim=model_lane_claim,
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "run_model_blocking cannot block an active event loop; await run_async"
            )

        inherited_owner = str(
            _effective_env_value(env, "AURA_MODEL_LANE_INHERITED_OWNER_ID") or ""
        ).strip()
        inherited_request = str(
            _effective_env_value(env, "AURA_MODEL_LANE_INHERITED_REQUEST_ID") or ""
        ).strip()
        inherited_model = str(
            _effective_env_value(env, "AURA_MODEL_LANE_INHERITED_MODEL_PATH") or ""
        ).strip()
        inherited_purpose = str(
            _effective_env_value(env, "AURA_MODEL_LANE_INHERITED_PURPOSE") or ""
        ).strip()
        inherited_token = str(
            _effective_env_value(env, "AURA_MODEL_LANE_DELEGATION_TOKEN") or ""
        ).strip()
        inherited_fields = (
            inherited_owner,
            inherited_request,
            inherited_model,
            inherited_purpose,
            inherited_token,
        )
        if any(inherited_fields):
            if not all(inherited_fields):
                raise GovernanceViolation("incomplete inherited model-lane delegation")
            if claim is None:
                raise GovernanceViolation(
                    "inherited model-lane delegation requires a child model claim"
                )
            from core.runtime.model_lane_control import get_model_lane_controller

            inherited_controller = get_model_lane_controller()
            inherited_child_request_id = str(claim.request_id)
            inherited_valid = inherited_controller.validate_inherited_child_claim(
                owner_id=inherited_owner,
                request_id=inherited_request,
                model_path=inherited_model,
                purpose=inherited_purpose,
                delegation_token=inherited_token,
                child_pid=os.getpid(),
                parent_pid=os.getppid(),
                requested_gb=float(claim.request_gb),
                child_model_path=str(claim.model_path),
                child_purpose=str(claim.purpose),
                child_request_id=inherited_child_request_id,
                ttl_s=max(60.0, float(timeout) + 30.0),
            )
            if not inherited_valid:
                raise GovernanceViolation("invalid inherited model-child delegation")
            try:
                offline_bypass = _validate_offline_tooling_bypass(
                    offline_tooling=offline_tooling,
                    source=source,
                    command=command,
                    env=env,
                )
                _require_not_shutting_down(
                    f"subprocess_gateway.run_model_blocking:{source}",
                    read_only=read_only,
                    offline_tooling=offline_tooling,
                    allow_during_shutdown=allow_during_shutdown,
                    bounded_completion=True,
                )
                if not read_only and not offline_bypass:
                    _require_effect_governance(
                        f"subprocess_gateway.run_model_blocking:{source}"
                    )
                _validate_desktop_safe_subprocess(
                    command,
                    env=env,
                    source=source,
                    operation="run_model_blocking",
                )
                # Deliberately inherit the outer worker's isolated process group.
                child_env = dict(env) if env is not None else dict(os.environ)
                for key in (*_DELEGATED_GOVERNANCE_ENV_KEYS, *_INHERITED_MODEL_LANE_ENV_KEYS):
                    child_env.pop(key, None)
                child_env["AURA_GOVERNANCE_MODE"] = "delegated_subprocess_child"
                child_env["AURA_REQUIRE_GOVERNANCE"] = "0"
                child_env["AURA_MODEL_LANE_PARENT_ACCOUNTED"] = "1"
                return subprocess.run(
                    command,
                    cwd=_coerce_cwd(cwd),
                    env=child_env,
                    timeout=float(timeout),
                    capture_output=bool(capture_output),
                    input=input,
                    text=True,
                    check=bool(check),
                    shell=False,
                    start_new_session=False,
                )
            finally:
                released = inherited_controller.release_inherited_child_claim(
                    owner_id=inherited_owner,
                    request_id=inherited_request,
                    child_request_id=inherited_child_request_id,
                    child_pid=os.getpid(),
                )
                if not released:
                    raise RuntimeError("inherited_model_child_sublease_release_failed")
        return asyncio.run(
            self.run_async(
                command,
                cwd=cwd,
                env=env,
                timeout=timeout,
                read_only=read_only,
                offline_tooling=offline_tooling,
                allow_during_shutdown=allow_during_shutdown,
                capture_output=capture_output,
                input=input,
                check=check,
                source=source,
                model_lane_claim=claim,
                accelerator_capability=AcceleratorCapability.MODEL,
            )
        )

    async def run_async(
        self,
        argv: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = 30.0,  # noqa: ASYNC109 - forwarded to subprocess.run.
        read_only: bool = False,
        offline_tooling: bool = False,
        allow_during_shutdown: bool = False,
        capture_output: bool = True,
        input: str | None = None,
        check: bool = False,
        source: str = "unknown",
        model_lane_claim: Any | None = None,
        accelerator_capability: AcceleratorCapability | str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = _coerce_argv(argv)
        inferred_claim = _resolve_accelerator_claim(
            command,
            source=source,
            timeout_s=float(timeout),
            accelerator_capability=accelerator_capability,
            model_lane_claim=model_lane_claim,
        )
        # Live effectful work stays in an async-owned process group even when it
        # does not load a model. That gives cancellation and timeout a real
        # descendant-reaping boundary instead of abandoning subprocess.run in
        # a worker thread.
        if inferred_claim is not None or (not read_only and not offline_tooling):
            process = await self.spawn_async(
                command,
                stdin=asyncio.subprocess.PIPE if input is not None else None,
                stdout=asyncio.subprocess.PIPE if capture_output else None,
                stderr=asyncio.subprocess.PIPE if capture_output else None,
                cwd=cwd,
                env=env,
                read_only=read_only,
                offline_tooling=offline_tooling,
                allow_during_shutdown=allow_during_shutdown,
                source=source,
                model_lane_claim=inferred_claim,
                accelerator_capability=(
                    AcceleratorCapability.MODEL
                    if inferred_claim is not None
                    else AcceleratorCapability.NONE
                ),
            )
            input_bytes = input.encode() if input is not None else None
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(input_bytes),
                    timeout=float(timeout),
                )
            except TimeoutError as exc:
                partial_stdout, partial_stderr = await _terminate_async_process_group(
                    process
                )
                stdout_bytes = partial_stdout or b""
                stderr_bytes = partial_stderr or b""
                raise subprocess.TimeoutExpired(
                    command,
                    float(timeout),
                    output=stdout_bytes,
                    stderr=stderr_bytes,
                ) from exc
            except asyncio.CancelledError:
                await _terminate_async_process_group(process)
                raise
            stdout_text = (
                stdout_bytes.decode("utf-8", errors="replace")
                if isinstance(stdout_bytes, bytes)
                else stdout_bytes
            )
            stderr_text = (
                stderr_bytes.decode("utf-8", errors="replace")
                if isinstance(stderr_bytes, bytes)
                else stderr_bytes
            )
            completed = subprocess.CompletedProcess(
                command,
                int(process.returncode or 0),
                stdout_text,
                stderr_text,
            )
            if check:
                completed.check_returncode()
            return completed

        def _run() -> subprocess.CompletedProcess[str]:
            return self.run(
                command,
                cwd=cwd,
                env=env,
                timeout=timeout,
                read_only=read_only,
                offline_tooling=offline_tooling,
                allow_during_shutdown=allow_during_shutdown,
                capture_output=capture_output,
                input=input,
                check=check,
                source=source,
                accelerator_capability=AcceleratorCapability.NONE,
            )

        if is_shutdown_requested() and allow_during_shutdown and read_only:
            result = await run_sync_shutdown_callable(
                _run,
                timeout_s=max(0.1, float(timeout)) + 1.0,
                name=f"read-only-subprocess:{source}",
            )
            if not isinstance(result, subprocess.CompletedProcess):
                raise RuntimeError("shutdown subprocess bridge returned invalid result")
            return result
        return await asyncio.to_thread(_run)

    def spawn(
        self,
        argv: Sequence[str],
        *,
        stdin: Any = None,
        stdout: IO[str] | None = None,
        stderr: IO[str] | None = None,
        stdout_path: str | os.PathLike[str] | None = None,
        stderr_path: str | os.PathLike[str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        text: bool = True,
        start_new_session: bool = True,
        preexec_fn: Callable[[], None] | None = None,
        read_only: bool = False,
        offline_tooling: bool = False,
        allow_during_shutdown: bool = False,
        source: str = "unknown",
        accelerator_capability: AcceleratorCapability | str | None = None,
    ) -> subprocess.Popen[Any]:
        command = _coerce_argv(argv)
        _model_command_requires_async(
            command,
            source=source,
            accelerator_capability=accelerator_capability,
        )
        if read_only and not offline_tooling:
            _validate_read_only_source(source)
        offline_bypass = _validate_offline_tooling_bypass(
            offline_tooling=offline_tooling,
            source=source,
            command=command,
            env=env,
        )
        _validate_delegated_governance_environment(env, source=source)
        _require_not_shutting_down(
            f"subprocess_gateway.spawn:{source}",
            read_only=read_only,
            offline_tooling=offline_tooling,
            allow_during_shutdown=allow_during_shutdown,
        )
        if not read_only and not offline_bypass:
            _require_effect_governance(f"subprocess_gateway.spawn:{source}")
        _validate_desktop_safe_subprocess(command, env=env, source=source, operation="spawn")
        _enforce_process_privilege(env=env, source=source, operation="spawn")
        if stdout is not None and stdout_path is not None:
            raise ValueError("stdout and stdout_path are mutually exclusive")
        if stderr is not None and stderr_path is not None:
            raise ValueError("stderr and stderr_path are mutually exclusive")

        opened_streams: list[IO[Any]] = []
        try:
            if stdout_path is not None:
                stdout = _open_spawn_stream(stdout_path, text=text)
                opened_streams.append(stdout)
            if stderr_path is not None:
                stderr = _open_spawn_stream(stderr_path, text=text)
                opened_streams.append(stderr)

            proc = subprocess.Popen(
                command,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                cwd=_coerce_cwd(cwd),
                env=dict(env) if env is not None else None,
                shell=False,
                text=text,
                start_new_session=start_new_session,
                preexec_fn=preexec_fn,
            )
            try:
                _require_not_shutting_down(
                    f"subprocess_gateway.spawn:{source}",
                    read_only=read_only,
                    offline_tooling=offline_tooling,
                    allow_during_shutdown=allow_during_shutdown,
                    resource_created=True,
                )
            except GovernanceViolation:
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2.0)
                except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
                    record_shutdown_admission_event(
                        f"subprocess_gateway.spawn:{source}",
                        resource_kind="subprocess",
                        outcome="survived",
                        detail=repr(exc),
                    )
                    raise
                record_shutdown_admission_event(
                    f"subprocess_gateway.spawn:{source}",
                    resource_kind="subprocess",
                    outcome="reaped",
                    detail=f"pid={getattr(proc, 'pid', None)}",
                )
                raise
            proc._aura_gateway_streams = tuple(opened_streams)  # type: ignore[attr-defined]
            _register_runtime_hygiene_process(
                proc,
                kind="subprocess",
                source=source,
                command=command,
            )
            return proc
        except (
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            ValueError,
            GovernanceViolation,
        ):
            for stream in opened_streams:
                try:
                    stream.close()
                except OSError as close_exc:
                    logger.debug("failed to close gateway-owned subprocess stream: %s", close_exc)
            raise

    async def spawn_async(
        self,
        argv: Sequence[str],
        *,
        stdin: Any = None,
        stdout: Any = None,
        stderr: Any = None,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        start_new_session: bool = True,
        read_only: bool = False,
        offline_tooling: bool = False,
        allow_during_shutdown: bool = False,
        source: str = "unknown",
        model_lane_claim: Any | None = None,
        accelerator_capability: AcceleratorCapability | str | None = None,
    ) -> asyncio.subprocess.Process:
        command = _coerce_argv(argv)
        if read_only and not offline_tooling:
            _validate_read_only_source(source)
        offline_bypass = _validate_offline_tooling_bypass(
            offline_tooling=offline_tooling,
            source=source,
            command=command,
            env=env,
        )
        _validate_delegated_governance_environment(env, source=source)
        _require_not_shutting_down(
            f"subprocess_gateway.spawn_async:{source}",
            read_only=read_only,
            offline_tooling=offline_tooling,
            allow_during_shutdown=allow_during_shutdown,
        )
        if not read_only and not offline_bypass:
            _require_effect_governance(f"subprocess_gateway.spawn_async:{source}")
        _validate_desktop_safe_subprocess(command, env=env, source=source, operation="spawn_async")
        _enforce_process_privilege(env=env, source=source, operation="spawn_async")
        claim = _resolve_accelerator_claim(
            command,
            source=source,
            timeout_s=300.0,
            accelerator_capability=accelerator_capability,
            model_lane_claim=model_lane_claim,
        )
        if claim is not None and not start_new_session:
            raise RuntimeError("model_subprocess_requires_isolated_process_group")
        model_controller = None
        model_decision = None
        model_delegation_token = ""
        if claim is not None:
            model_controller, model_decision = await _reserve_model_lane_process(claim)
            try:
                model_delegation_token = await model_controller.issue_inherited_claim(
                    model_decision
                )
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
                await _cancel_model_lane_process(
                    model_controller,
                    model_decision,
                    reason="model_subprocess_delegation_failed",
                )
                raise
        process_env = dict(env) if env is not None else None
        if claim is not None:
            if process_env is None:
                process_env = dict(os.environ)
            process_env.update(
                {
                    "AURA_MODEL_LANE_INHERITED_OWNER_ID": str(claim.owner_id),
                    "AURA_MODEL_LANE_INHERITED_REQUEST_ID": str(claim.request_id),
                    "AURA_MODEL_LANE_INHERITED_MODEL_PATH": str(claim.model_path),
                    "AURA_MODEL_LANE_INHERITED_PURPOSE": str(claim.purpose),
                    "AURA_MODEL_LANE_DELEGATION_TOKEN": model_delegation_token,
                }
            )
            # Tell the child WHERE the reservation lives instead of letting it
            # re-derive the path. Both sides used to guess from HOME and the
            # runtime profile, and they can disagree: a child that starts with
            # HOME already redirected cannot detect the redirection, so it
            # applies a profile suffix the parent did not, looks in
            # `.aura-test/` for a record written to `.aura/`, and silently
            # falls back to an uninherited lease — claiming a second lane
            # against a budget that was already spent. The parent knows the
            # real path; passing it makes the two agree by construction.
            controller_state_path = getattr(model_controller, "state_path", None)
            if controller_state_path is not None:
                process_env["AURA_MODEL_LANE_STATE_PATH"] = str(controller_state_path)
            # Which world that path belongs to is handled generally, by
            # core.runtime.state_ownership exporting AURA_LIVE_STATE_ROOT into
            # this process's environment so every child inherits the true
            # identity of the live instance rather than re-inferring it.
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                cwd=_coerce_cwd(cwd),
                env=process_env,
                start_new_session=start_new_session,
            )
            process_pid = int(getattr(proc, "pid", 0) or 0)
            try:
                process_group_id = int(os.getpgid(process_pid)) if process_pid > 0 else 0
            except (OSError, ProcessLookupError, ValueError):
                process_group_id = process_pid if start_new_session else 0
            proc._aura_process_group_id = process_group_id  # type: ignore[attr-defined]
            proc._aura_start_new_session = bool(start_new_session)  # type: ignore[attr-defined]
        except (OSError, RuntimeError, ValueError):
            if model_controller is not None and model_decision is not None:
                await _cancel_model_lane_process(
                    model_controller,
                    model_decision,
                    reason="model_subprocess_spawn_failed",
                )
            raise
        try:
            _require_not_shutting_down(
                f"subprocess_gateway.spawn_async:{source}",
                read_only=read_only,
                offline_tooling=offline_tooling,
                allow_during_shutdown=allow_during_shutdown,
                resource_created=True,
            )
        except GovernanceViolation:
            try:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except TimeoutError:
                    proc.kill()
                    try:
                        # Bounded: a SIGKILLed child that cannot be reaped
                        # in 5s is an OS-level anomaly; leaking one zombie
                        # beats wedging the caller (A1 discipline).
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
                    except TimeoutError:
                        logger.warning(
                            "SIGKILLed subprocess pid=%s not reaped in 5s",
                            getattr(proc, "pid", None),
                        )
            except (OSError, RuntimeError, ProcessLookupError, ValueError) as exc:
                record_shutdown_admission_event(
                    f"subprocess_gateway.spawn_async:{source}",
                    resource_kind="subprocess",
                    outcome="survived",
                    detail=repr(exc),
                )
                raise
            record_shutdown_admission_event(
                f"subprocess_gateway.spawn_async:{source}",
                resource_kind="subprocess",
                outcome="reaped",
                detail=f"pid={getattr(proc, 'pid', None)}",
            )
            if model_controller is not None and model_decision is not None:
                await _cancel_model_lane_process(
                    model_controller,
                    model_decision,
                    reason="shutdown_crossed_model_subprocess_spawn",
                )
            raise
        if model_controller is not None and model_decision is not None:
            from core.runtime.model_lane_control import (
                managed_process_group_alive,
                process_identity_for_pid,
            )

            try:
                process_group_id = int(os.getpgid(proc.pid))
                process_session_id = int(os.getsid(proc.pid))
            except (OSError, ProcessLookupError, ValueError):
                process_group_id = 0
                process_session_id = 0
            committed_process = process_identity_for_pid(proc.pid)
            if start_new_session and (
                process_group_id != proc.pid or process_session_id != proc.pid
            ):
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except (OSError, RuntimeError, ProcessLookupError, TimeoutError, ValueError):
                    logger.error(
                        "Model subprocess survived invalid session identity pid=%s pgid=%s sid=%s",
                        getattr(proc, "pid", None),
                        process_group_id,
                        process_session_id,
                    )
                await _cancel_model_lane_process(
                    model_controller,
                    model_decision,
                    reason="model_subprocess_session_identity_invalid",
                )
                raise RuntimeError("model_subprocess_session_identity_invalid")
            try:
                committed = await model_controller.commit(
                    model_decision,
                    process=committed_process,
                    metadata={
                        "managed_model_process": True,
                        "process_group_id": process_group_id,
                        "process_session_id": process_session_id,
                        "process_group_identity_version": 1,
                        "start_new_session": bool(start_new_session),
                        "source": source,
                        "command": list(command),
                    },
                )
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except (OSError, RuntimeError, ProcessLookupError, TimeoutError, ValueError):
                    logger.error(
                        "Model subprocess survived failed lane commit pid=%s",
                        getattr(proc, "pid", None),
                    )
                await _cancel_model_lane_process(
                    model_controller,
                    model_decision,
                    reason=f"model_subprocess_commit_failed:{type(exc).__name__}",
                )
                raise RuntimeError("model_subprocess_lane_commit_failed") from exc

            async def _release_model_owner_when_done() -> None:
                try:
                    # Re-arming bounded slices: this monitor's LIFETIME is the
                    # worker's lifetime by design, but each individual await
                    # stays bounded (A1) so a wedged wait can never hide.
                    worker_exited = False
                    while not worker_exited:
                        try:
                            await asyncio.wait_for(proc.wait(), timeout=60.0)
                            worker_exited = True
                        except TimeoutError:
                            continue
                    # Descendants have no asyncio completion primitive.
                    while managed_process_group_alive(  # noqa: ASYNC110
                        process_group_id,
                        root_started_at=committed_process.started_at,
                        session_id=process_session_id,
                        root_pid=committed_process.pid,
                    ):
                        await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    if proc.returncode is None or managed_process_group_alive(
                        process_group_id,
                        root_started_at=committed_process.started_at,
                        session_id=process_session_id,
                        root_pid=committed_process.pid,
                    ):
                        logger.info(
                            "Model subprocess monitor cancelled while process tree remains "
                            "live; durable owner retained owner=%s pid=%s pgid=%s",
                            committed.owner_id,
                            proc.pid,
                            process_group_id,
                        )
                    raise
                finally:
                    if proc.returncode is not None and not managed_process_group_alive(
                        process_group_id,
                        root_started_at=committed_process.started_at,
                        session_id=process_session_id,
                        root_pid=committed_process.pid,
                    ):
                        try:
                            await model_controller.release_owner(
                                committed.owner_id,
                                fencing_token=committed.fencing_token,
                                reason=f"model_subprocess_exit:{proc.returncode}",
                            )
                        except (
                            OSError,
                            RuntimeError,
                            AttributeError,
                            TypeError,
                            ValueError,
                        ) as exc:
                            logger.warning(
                                "Model subprocess owner release failed owner=%s: %s",
                                committed.owner_id,
                                exc,
                            )

            from core.utils.task_tracker import get_task_tracker

            monitor_coroutine = _release_model_owner_when_done()
            try:
                monitor = get_task_tracker().create_task(
                    monitor_coroutine,
                    name=f"ModelProcessOwner:{committed.owner_id}",
                )
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                monitor_coroutine.close()
                try:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2.0)
                    except TimeoutError:
                        proc.kill()
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
                except (
                    OSError,
                    RuntimeError,
                    ProcessLookupError,
                    TimeoutError,
                    ValueError,
                ) as reap_exc:
                    logger.error(
                        "Model subprocess monitor failed and child reap was incomplete pid=%s: %s",
                        getattr(proc, "pid", None),
                        reap_exc,
                    )
                finally:
                    await model_controller.release_owner(
                        committed.owner_id,
                        fencing_token=committed.fencing_token,
                        reason="model_subprocess_monitor_registration_failed",
                    )
                raise RuntimeError("model_subprocess_monitor_registration_failed") from exc
            proc._aura_model_lane_owner_id = committed.owner_id  # type: ignore[attr-defined]
            proc._aura_model_lane_fencing_token = committed.fencing_token  # type: ignore[attr-defined]
            proc._aura_model_lane_receipt_id = committed.receipt_id  # type: ignore[attr-defined]
            proc._aura_model_lane_monitor = monitor  # type: ignore[attr-defined]
        _register_runtime_hygiene_process(
            proc,
            kind="subprocess",
            source=source,
            command=command,
        )
        return proc

    async def spawn_shell_async(
        self,
        command: str,
        *,
        stdin: Any = None,
        stdout: Any = None,
        stderr: Any = None,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        start_new_session: bool = True,
        offline_tooling: bool = False,
        allow_during_shutdown: bool = False,
        source: str = "unknown",
        accelerator_capability: AcceleratorCapability | str | None = None,
    ) -> asyncio.subprocess.Process:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("shell command must be a non-empty string")
        if "\x00" in command:
            raise ValueError("shell command must not contain NUL bytes")
        try:
            inspected_argv = shlex.split(command)
        except ValueError as exc:
            raise GovernanceViolation(
                f"subprocess_shell_capability_unparseable:{source}"
            ) from exc
        claim = _resolve_accelerator_claim(
            inspected_argv,
            source=source,
            timeout_s=300.0,
            accelerator_capability=accelerator_capability,
        )
        return await self.spawn_async(
            ["/bin/sh", "-lc", command],
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            cwd=_coerce_cwd(cwd),
            env=env,
            start_new_session=start_new_session,
            read_only=False,
            offline_tooling=offline_tooling,
            allow_during_shutdown=allow_during_shutdown,
            source=source,
            model_lane_claim=claim,
            accelerator_capability=(
                AcceleratorCapability.MODEL
                if claim is not None
                else AcceleratorCapability.NONE
            ),
        )


_gateway: SubprocessGateway | None = None


def get_subprocess_gateway() -> SubprocessGateway:
    global _gateway
    if _gateway is None:
        _gateway = SubprocessGateway()
    return _gateway


__all__ = [
    "AcceleratorCapability",
    "PythonProcessOwnershipError",
    "PythonProcessSpec",
    "SubprocessGateway",
    "get_subprocess_gateway",
]
