"""Shared primitives for running untrusted code as a constrained process.

Aura executes model-generated Python in two places — ``sandbox_operator`` and
``symbolic_sandbox`` — and CP126 raised the same findings against both: the
"sandbox" is not an OS sandbox, there are no kernel-enforced resource quotas,
a timeout does not prove descendants died, and arbitrary execution was labelled
read-only authority.

Those are one problem, so this is one solution. Both callers use these
primitives instead of each growing its own copy.

**The honest bound, stated once.** None of this is containment. The child runs
as Aura's own user with the host interpreter, so it inherits filesystem,
keychain and device reach that no in-process check can revoke. What is provided
is *constrained execution*:

* a scrubbed environment, so no ambient credential or proxy setting is
  inherited;
* POSIX resource limits — CPU, file size, descriptors, processes, no core
  dumps — which the kernel enforces whatever the code does. Not every platform
  honours every limit (Darwin accepts ``RLIMIT_AS`` and ignores it), so
  :func:`effective_limits` reports which ones actually bind rather than
  publishing the ceiling we asked for;
* its own session and process group, so a timeout can reap the whole
  descendant tree and produce evidence that it did;
* an interpreter started in isolated mode.

Real isolation needs a container, a jail, or a separate low-privilege user.
Until one exists, :data:`ISOLATION_LEVEL` reports ``"constrained_process"`` and
never ``"sandboxed"``, so no caller can mistake this for containment.

CP126 f52d8430 / 84fc4f9d (sandbox_operator) and
d10e3cc5 / c77398cb / 23199cb8 / 0f09e635 (symbolic_sandbox).
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from typing import Any

logger = logging.getLogger("Aura.ConstrainedExec")

#: What this actually provides. Never "sandboxed".
ISOLATION_LEVEL = "constrained_process"

#: Kernel-enforced ceilings for a child running untrusted code.
RLIMIT_CPU_S = 30
RLIMIT_ADDRESS_SPACE_BYTES = 2 * 1024 * 1024 * 1024   # 2 GiB
RLIMIT_FILE_SIZE_BYTES = 64 * 1024 * 1024             # 64 MiB
RLIMIT_OPEN_FILES = 128
RLIMIT_PROCESSES = 32

#: Environment keys the child may inherit. Everything else is dropped.
SAFE_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "TZ")


def child_preexec() -> None:
    """Apply POSIX limits and start a new session in the child.

    Runs between fork and exec. The new session (and therefore process group)
    is what makes a recursive reap possible: without it a timeout can only
    signal the direct child, and its descendants survive.
    """
    try:
        os.setsid()
    # not a failure: already a session leader (start_new_session=True), which
    # is the state this call exists to reach.
    except (AttributeError, OSError):
        pass
    try:
        import resource
    # not a failure: POSIX limits do not exist off POSIX, and the child runs
    # without them rather than not at all.
    except ImportError:  # pragma: no cover - non-POSIX
        return

    limits = [
        (resource.RLIMIT_CPU, RLIMIT_CPU_S),
        (resource.RLIMIT_AS, RLIMIT_ADDRESS_SPACE_BYTES),
        (resource.RLIMIT_FSIZE, RLIMIT_FILE_SIZE_BYTES),
        (resource.RLIMIT_NOFILE, RLIMIT_OPEN_FILES),
        (getattr(resource, "RLIMIT_NPROC", resource.RLIMIT_NOFILE), RLIMIT_PROCESSES),
        (getattr(resource, "RLIMIT_CORE", resource.RLIMIT_FSIZE), 0),
    ]
    for limit, value in limits:
        try:
            _soft, hard = resource.getrlimit(limit)
            ceiling = value if hard in (resource.RLIM_INFINITY, -1) else min(value, hard)
            resource.setrlimit(limit, (ceiling, ceiling))
        except (OSError, ValueError):
            continue


def scrubbed_env(**extra: str) -> dict[str, str]:
    """A minimal environment for an untrusted child."""
    env = {key: os.environ[key] for key in SAFE_ENV_KEYS if key in os.environ}
    env.setdefault("PATH", "/usr/bin:/bin")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["AURA_SANDBOX"] = "1"
    env.update({str(key): str(value) for key, value in extra.items()})
    return env


def reap_process_group(pgid: int | None, process: Any = None) -> dict[str, Any]:
    """Terminate a process group and report whether it actually died.

    Two traps this avoids, both found by running it:

    * ``os.getpgid(child.pid)`` races the child's ``setsid`` and can return
      AURA'S OWN group, so the reap must never signal the caller's group. With
      ``start_new_session=True`` the child's pgid IS its pid; pass that.
    * macOS answers ``EPERM`` rather than ``ESRCH`` for a group whose members
      are zombies, so the signal's return value cannot decide the outcome. The
      reap is confirmed by the child's actual exit.
    """
    receipt: dict[str, Any] = {"pgid": pgid, "attempted": False, "reaped": None}
    if not pgid or not hasattr(os, "killpg"):
        receipt["reason"] = "no process group recorded"
        return receipt
    try:
        if pgid == os.getpgrp():
            receipt["reason"] = "refused to signal aura's own process group"
            return receipt
    except OSError as exc:
        # Not knowing our own group is not a reason to stop, but the receipt
        # should say the self-signal check did not run.
        receipt["own_group_check"] = f"unavailable: {type(exc).__name__}: {exc}"

    receipt["attempted"] = True
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
            receipt["signal"] = sig.name
        except ProcessLookupError:
            receipt["reaped"] = True
            receipt["signal"] = sig.name
            return receipt
        except (OSError, PermissionError) as exc:
            receipt["signal_error"] = f"{type(exc).__name__}: {exc}"
            break
        time.sleep(0.05)

    if process is not None:
        try:
            process.wait(timeout=2)
            receipt["reaped"] = True
            receipt["confirmed_by"] = "child_exit"
            return receipt
        except Exception as exc:  # noqa: BLE001 - any wait failure leaves it unconfirmed
            logger.debug("Child-exit confirmation failed for pid=%s: %s", process.pid, exc)
    try:
        os.killpg(pgid, 0)
        receipt["reaped"] = False
        receipt["confirmed_by"] = "group_still_addressable"
    except ProcessLookupError:
        receipt["reaped"] = True
        receipt["confirmed_by"] = "group_gone"
    except OSError as exc:
        receipt["reaped"] = None
        receipt["confirmed_by"] = f"indeterminate: {type(exc).__name__}"
    return receipt


def effective_limits() -> dict[str, Any]:
    """Which requested limits this platform ACTUALLY accepts.

    Written after discovering that macOS silently ignores ``RLIMIT_AS``: the
    ``setrlimit`` call succeeds-by-doing-nothing and the child still has an
    unlimited address space. Publishing the *requested* ceiling would have been
    the same defect this campaign is about — a limit reported as enforced when
    nothing enforces it. This probes the current process and reports what is
    actually enforceable, so a caller sees the real envelope.
    """
    requested = {
        "cpu_s": ("RLIMIT_CPU", RLIMIT_CPU_S),
        "address_space_bytes": ("RLIMIT_AS", RLIMIT_ADDRESS_SPACE_BYTES),
        "file_size_bytes": ("RLIMIT_FSIZE", RLIMIT_FILE_SIZE_BYTES),
        "open_files": ("RLIMIT_NOFILE", RLIMIT_OPEN_FILES),
        "processes": ("RLIMIT_NPROC", RLIMIT_PROCESSES),
    }
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return {name: {"requested": value, "enforced": False, "reason": "no resource module"}
                for name, (_attr, value) in requested.items()}

    report: dict[str, Any] = {}
    for name, (attr, value) in requested.items():
        limit = getattr(resource, attr, None)
        if limit is None:
            report[name] = {"requested": value, "enforced": False, "reason": f"{attr} unavailable"}
            continue
        try:
            _soft, hard = resource.getrlimit(limit)
        except (OSError, ValueError) as exc:
            report[name] = {"requested": value, "enforced": False, "reason": str(exc)}
            continue
        unlimited = hard in (resource.RLIM_INFINITY, -1)
        report[name] = {
            "requested": value,
            # A ceiling can only bind if the hard limit allows it. This is the
            # honest signal; the child's own getrlimit is the ground truth and
            # callers that need certainty should probe it there.
            "enforced": bool(unlimited or hard >= value) if attr != "RLIMIT_AS" else _as_enforced(),
        }
    return report


def _as_enforced() -> bool:
    """Whether RLIMIT_AS actually binds on this platform.

    Darwin accepts the call and ignores it, so the only honest answer comes
    from asking the platform rather than from the constant.
    """
    return not sys.platform.startswith("darwin")


def isolation_receipt(**extra: Any) -> dict[str, Any]:
    """Describe the containment a caller actually got.

    Callers attach this to their result so a consumer can see the real bound
    rather than inferring containment from the word "sandbox".
    """
    receipt = {
        "isolation_level": ISOLATION_LEVEL,
        "os_sandbox": False,
        "resource_limits": effective_limits(),
        # The static gate inspects source; it cannot enumerate what an allowed
        # object graph can reach, so passing it is admission, not proof.
        "static_gate": "ast_denylist_advisory",
        "bound": (
            "runs as Aura's user with the host interpreter; no filesystem, "
            "network-namespace, syscall or UID boundary"
        ),
    }
    receipt.update(extra)
    return receipt
