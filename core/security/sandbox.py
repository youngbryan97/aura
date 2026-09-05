"""A sandbox that is one.

What this class used to do, in full: ``os.makedirs(sandbox_dir)`` and pass
``cwd=sandbox_dir`` to the subprocess gateway. That is a working directory.
The command could still read every file the user can read, write outside the
directory with one absolute path, open sockets, and spawn whatever it liked.
It was named ``LocalCommandSandbox``, its docstring said "restricts command
scopes using directory isolations", :class:`~core.security.consent_kernel.ConsentKernel`
held it as its isolation primitive, and a test named
``test_subsystem_hardening_checkpoint`` asserted that it ran ``python -V``.

Nothing in that arrangement was false about the code and all of it was false
about the property. This is the failure this codebase keeps rediscovering
under different names: the absence of a check reported as a passed check —
here, the absence of isolation reported by a class called Sandbox.

Nothing routes real commands through it yet, so this fix breaks nothing
today. That is exactly why it is worth making now: the first caller who
reaches for the thing named "sandbox" should get one.

The isolation itself already existed, in :mod:`security.sandbox`, which
injects a ``sandbox-exec`` seatbelt profile that denies network outright and
confines writes to the workdir, applies rlimits on CPU/memory/processes/file
size, strips sensitive environment variables through the same classifier the
subprocess gateway enforces with, and validates the binary against a
per-level allowlist. This module now delegates there instead of carrying a
second, weaker idea of what a sandbox is.

The receipt says whether the kernel actually enforced anything, because
``exit_code: 0`` from a sandbox that silently degraded to a plain subprocess
reads identically to one that did not.
"""
from __future__ import annotations

import logging
import shlex
import sys
from collections.abc import Sequence
from subprocess import SubprocessError
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Security.Sandbox")

_SANDBOX_EXECUTION_ERRORS = (
    ImportError,
    OSError,
    RuntimeError,
    SubprocessError,
    TimeoutError,
    TypeError,
    ValueError,
)

#: Wall-clock ceiling. Unchanged from the original so a caller's timing
#: expectations are not quietly altered by this rewrite.
DEFAULT_TIMEOUT_S = 10.0


def _kernel_enforced(level: Any) -> bool:
    """Did the OS itself constrain this process, or only the allowlist?

    macOS gets a seatbelt profile from :mod:`security.sandbox`; elsewhere the
    boundary is the command allowlist plus rlimits, which is real but is not
    the kernel refusing a syscall. Callers deserve to know which one they got.
    """
    try:
        from security.sandbox import SecurityLevel

        return sys.platform == "darwin" and level != SecurityLevel.PRIVILEGED
    except (ImportError, AttributeError):
        return False


class LocalCommandSandbox:
    """Runs a command under OS-enforced isolation, or does not run it."""

    def execute_sandboxed_command(
        self,
        command: str | Sequence[str],
        sandbox_dir: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> dict[str, Any]:
        argv = (
            [str(part) for part in command]
            if isinstance(command, (list, tuple))
            else shlex.split(str(command))
        )
        if not argv:
            return {"exit_code": -1, "error": "empty command", "sandboxed": False}

        logger.info("Executing command under sandbox isolation: %s", argv[0])
        try:
            from security.sandbox import SecureSandbox, SecurityLevel

            level = SecurityLevel.RESTRICTED
            sandbox = SecureSandbox(security_level=level, workdir=sandbox_dir)
            result = sandbox.execute_command(argv, timeout=float(timeout))
            return {
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "sandboxed": True,
                "kernel_enforced": _kernel_enforced(level),
                "security_violations": list(result.security_violations),
                "workdir": str(sandbox.workdir),
            }
        except _SANDBOX_EXECUTION_ERRORS as e:
            record_degradation(
                "security.sandbox",
                e,
                action="refused the command rather than run it unsandboxed",
            )
            logger.error("Sandbox unavailable, command refused: %s", e)
            # Falling back to an unsandboxed run would reinstate the exact
            # defect this module was rewritten to remove, and it would do it
            # on the path where isolation had already failed once.
            return {
                "exit_code": -1,
                "error": f"sandbox unavailable, command refused: {e}",
                "sandboxed": False,
            }
