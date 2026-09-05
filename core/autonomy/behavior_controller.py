"""Autonomous Behavior Controller.

The pre-action hook is the veto point between a decided tool call and a real
effect. Everything here is therefore written so that a check which cannot be
performed refuses, rather than passing.
"""
from __future__ import annotations

import logging
import shlex
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.BehaviorController")

#: Tools whose parameters carry a shell command. CP126 d564d622: the hook
#: passed the raw tool name as ``type`` and validation only inspected commands
#: when that string was literally "terminal" — so shell, run_command,
#: os_automation and every other executor name skipped the policy entirely.
COMMAND_BEARING_TOOLS = frozenset({
    "terminal",
    "shell",
    "bash",
    "sh",
    "zsh",
    "run_command",
    "run_shell",
    "execute_command",
    "os_automation",
    "system",
    "subprocess",
    "code_execution",
    "terminal_command",
})

#: Parameter keys that may contain a command line.
COMMAND_PARAM_KEYS = ("command", "cmd", "shell_command", "script", "args_line")

#: Executables that are never acceptable from an autonomous action.
DENIED_BINARIES = frozenset({
    "mkfs", "mkfs.ext4", "mkfs.xfs", "wipefs", "fdisk", "parted",
    "shred", "shutdown", "halt", "poweroff", "reboot", "init",
    "kexec", "diskutil", "hdiutil", "csrutil", "nvram", "spctl",
})

#: Binaries that are only safe when their arguments are constrained.
GUARDED_BINARIES = frozenset({"rm", "dd", "chmod", "chown", "kill", "pkill", "killall"})

#: Shell metacharacters that turn one vetted command into an arbitrary
#: program. CP126 a37ce790: a substring denylist cannot see through quoting,
#: variables, interpreters, encodings or redirection — refusing composition
#: is what actually bounds the command.
SHELL_COMPOSITION_CHARS = ("|", ";", "&", "`", "$(", ">", "<", "\n", "\r")

#: Paths that must never be the target of a destructive operation.
PROTECTED_TARGETS = ("/", "/*", "~", "~/", "/etc", "/etc/passwd", "/etc/shadow",
                     "/dev", "/dev/sda", "/System", "/usr", "/bin", "/sbin", "/var")


class BehaviorPolicyError(RuntimeError):
    """A behavior-policy check could not be completed, so the action is refused."""


def extract_command(action: dict[str, Any]) -> str:
    """The command text carried by this action, from wherever it lives."""
    for key in COMMAND_PARAM_KEYS:
        value = action.get(key)
        if isinstance(value, str) and value.strip():
            return value
    params = action.get("params") or action.get("args") or action.get("arguments")
    if isinstance(params, dict):
        for key in COMMAND_PARAM_KEYS:
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def classify_command(command: str) -> tuple[bool, str]:
    """Structural command policy. Returns (allowed, reason).

    This parses the command rather than substring-matching it: composition is
    refused, the resolved binary is checked against the denylist, and guarded
    binaries have their targets inspected. It is a BOUND, not a proof — a
    determined operator with an allowed interpreter can still do harm, which
    is why composition and interpreters are refused outright.
    """
    text = str(command or "").strip()
    if not text:
        return True, "no command"
    if "\x00" in text:
        return False, "command contains a NUL byte"
    lowered = text.lower()

    for token in SHELL_COMPOSITION_CHARS:
        if token in text:
            return False, f"shell composition is not permitted ({token!r})"

    try:
        parts = shlex.split(text)
    except ValueError as exc:
        # Unbalanced quotes: we cannot tell what would run.
        return False, f"command could not be parsed ({exc})"
    if not parts:
        return True, "no command"

    binary = parts[0].rsplit("/", 1)[-1].lower()
    arguments = parts[1:]

    if binary in DENIED_BINARIES:
        return False, f"denied binary: {binary}"

    # A fork bomb has no recognizable binary; refuse the shape.
    if ":(){" in lowered.replace(" ", ""):
        return False, "fork bomb pattern"

    if binary in GUARDED_BINARIES:
        blocked = _guarded_binary_reason(binary, arguments)
        if blocked:
            return False, blocked

    return True, "allowed"


def _guarded_binary_reason(binary: str, arguments: list[str]) -> str:
    """Why this guarded binary's arguments are unacceptable, or ''."""
    normalized = [arg.strip() for arg in arguments]
    targets = [arg for arg in normalized if not arg.startswith("-")]
    flags = "".join(arg for arg in normalized if arg.startswith("-")).lower()

    if binary == "rm":
        if "--no-preserve-root" in normalized:
            return "rm --no-preserve-root"
        recursive = "r" in flags or "--recursive" in normalized
        for target in targets:
            if target in PROTECTED_TARGETS or target.rstrip("/") in PROTECTED_TARGETS:
                return f"rm targets a protected path: {target}"
            if recursive and target in {"*", "."}:
                return f"recursive rm of {target}"
        return ""
    if binary == "dd":
        joined = " ".join(normalized).lower()
        if "if=/dev/zero" in joined or "if=/dev/random" in joined or "if=/dev/urandom" in joined:
            return "dd from a device source"
        if any(arg.startswith("of=/dev/") for arg in normalized):
            return "dd writes to a raw device"
        return ""
    if binary in {"chmod", "chown"}:
        for target in targets:
            if target in PROTECTED_TARGETS or target.rstrip("/") in PROTECTED_TARGETS:
                return f"{binary} targets a protected path: {target}"
        return ""
    if binary in {"kill", "pkill", "killall"}:
        joined = " ".join(normalized).lower()
        if "-1" in normalized or "root" in joined or "-9 -1" in joined:
            return f"{binary} would target init or root processes"
        return ""
    return ""


class AutonomousBehaviorController:
    """Controls execution of autonomous behaviors."""

    def __init__(self, orchestrator=None, *, safety_checks_enabled: bool = True):
        self.orchestrator = orchestrator
        # CP126 0f030779: this was a public-looking control that validation
        # never read — a safety-shaped setting that changed nothing. It is now
        # honoured, and turning it OFF is loud.
        self.safety_checks_enabled = bool(safety_checks_enabled)
        if not self.safety_checks_enabled:
            logger.critical(
                "🚨 AutonomousBehaviorController constructed with safety checks DISABLED."
            )

    def validate_action(self, action: dict[str, Any]) -> bool:
        """Validate if an action is safe to execute."""
        if not self.safety_checks_enabled:
            record_degradation(
                "behavior_controller",
                BehaviorPolicyError("safety_checks_disabled"),
                action="allowed an action without safety validation because checks are disabled",
                severity="critical",
            )
            return True

        action_type = str(action.get("type", "") or "").strip().lower()
        command = extract_command(action)
        # CP126 d564d622: inspect the command whenever one is PRESENT, and
        # always for a command-bearing tool — not only when the type string
        # happens to be "terminal".
        if command or action_type in COMMAND_BEARING_TOOLS:
            allowed, reason = classify_command(command)
            if not allowed:
                logger.error(
                    "🚫 Blocked dangerous command (%s): %s", reason, command[:120]
                )
                return False
        return True

    async def execute_tool_call_async(
        self, tool_name: str, arguments: dict[str, Any], *, origin: str = "behavior_controller"
    ) -> Any:
        """Execute a tool through the orchestrator, with its context attached.

        CP126 e20e34cd: the sync path built a ``context`` dict and then never
        passed it, calling execute_tool with no origin, objective, or receipt.
        """
        if not self.orchestrator or not hasattr(self.orchestrator, "execute_tool"):
            return {"ok": False, "error": "No orchestrator available for tool execution"}
        if not self.validate_action({"type": tool_name, "params": arguments}):
            return {"ok": False, "error": "blocked_by_behavior_policy", "tool": tool_name}
        context = {
            "source": "behavior_controller",
            "origin": origin,
            "objective": f"execute {tool_name}",
            "tool": tool_name,
        }
        try:
            return await self.orchestrator.execute_tool(
                tool_name, arguments, context=context
            )
        except TypeError:
            # Older orchestrator signature without a context parameter.
            record_degradation(
                "behavior_controller",
                BehaviorPolicyError(f"execute_tool_context_unsupported:{tool_name}"),
                action="executed a tool without forwarding its authority context",
                severity="warning",
            )
            return await self.orchestrator.execute_tool(tool_name, arguments)

    def execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Synchronous entry point.

        CP126 b7b78f6c: this used ``run_coroutine_threadsafe`` to schedule work
        onto the SAME loop it was called from and then blocked on the result —
        the thread needed to run the coroutine was the thread waiting for it,
        so the call deadlocked until its two-minute timeout. Calling from
        inside a running loop is now refused with a typed error pointing at
        the async entry point.
        """
        import asyncio
        import concurrent.futures

        logger.info("🛠️ Executing tool: %s", tool_name)
        if not self.orchestrator or not hasattr(self.orchestrator, "execute_tool"):
            logger.warning("⚠️ No orchestrator wired — tool %s cannot execute", tool_name)
            return {"ok": False, "error": "No orchestrator available for tool execution"}

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        target_loop = getattr(self.orchestrator, "loop", None)
        try:
            if running is None:
                if target_loop is not None and target_loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(
                        self.execute_tool_call_async(tool_name, arguments), target_loop
                    )
                    return future.result(timeout=120)
                return asyncio.run(self.execute_tool_call_async(tool_name, arguments))

            # We ARE on a running loop.
            if target_loop is not None and target_loop is not running and target_loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self.execute_tool_call_async(tool_name, arguments), target_loop
                )
                return future.result(timeout=120)
            return {
                "ok": False,
                "error": "sync_execute_tool_call_on_running_loop",
                "detail": (
                    "await execute_tool_call_async instead; blocking this loop on "
                    "itself would deadlock"
                ),
            }
        except concurrent.futures.TimeoutError as exc:
            record_degradation("behavior_controller", exc)
            return {"ok": False, "error": "tool_execution_timeout"}
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("behavior_controller", e)
            logger.error("Real tool execution failed for %s: %s", tool_name, e)
            return {"ok": False, "error": str(e)}


# Integration helper
def integrate_behavior_control(orchestrator):
    """Integrate behavior control into orchestrator using formal hooks."""
    controller = AutonomousBehaviorController(orchestrator)

    async def on_pre_action_hook(tool_name: str, params: dict[str, Any]):
        # Return False to veto dangerous actions.
        is_safe = controller.validate_action({"type": tool_name, "params": params})
        if not is_safe:
            return False

        moral = getattr(orchestrator, "moral_reasoning", None)
        if moral is None:
            return True
        action_desc = {
            "type": "tool_call",
            "tool": tool_name,
            "args": params,
            "description": f"Execute tool {tool_name}",
        }
        try:
            assessment = await moral.reason_about_action(
                action_desc, {"type": "execution_check"}
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            # CP126 6369a874 corollary: a moral check that could not run is
            # not a passed moral check for a tool that reaches the world.
            record_degradation(
                "behavior_controller",
                exc,
                action="vetoed a tool call because the moral assessment could not run",
                severity="error",
            )
            return False
        if not isinstance(assessment, dict):
            return False
        if not assessment.get("is_morally_acceptable"):
            # CP126 6369a874: this used to log a warning and return the
            # original safety bit, so the declared "moral alignment
            # responsibility" had no causal effect at all.
            logger.error(
                "🚫 Vetoed tool '%s': moral assessment found it unacceptable (%s)",
                tool_name,
                str(assessment.get("reason") or assessment.get("rationale") or "")[:200],
            )
            return False
        return True

    orchestrator.hooks.register("pre_action", on_pre_action_hook)

    logger.info("✅ Behavior controller integrated via Hook System")
