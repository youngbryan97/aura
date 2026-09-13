"""core/fsw/command_dispatch.py — command dictionary and sequencer.

Clean-room adoption of F Prime's command dispatcher and command sequencer.

A spacecraft does not accept free-form instructions. It has a command
dictionary: each command has an opcode, a name, typed arguments with
ranges, an owner, and a declared effect. A sequence is a list of those
commands with timing and a failure policy, uplinked as one artifact that
can be validated on the ground *before* it runs.

Aura's autonomous action currently looks like the opposite: a plan is a
free-form intention that becomes tool calls as it executes, and whether
the plan was well-formed is discovered by running it. That is fine for a
single tool call and bad for a ten-step plan, because step 7 failing on a
malformed argument means steps 1-6 already happened.

What a dictionary and a sequencer add:

* **Arguments are validated against a declared schema before anything
  runs.** A plan with a bad argument in step 7 is rejected as a plan.
* **A sequence has a declared failure policy per step** — abort the
  sequence, continue, or retry — decided when the sequence is written
  rather than improvised when it fails.
* **Every dispatch is recorded** with its opcode, arguments, outcome, and
  duration, so "what did it actually do" has an answer that does not
  require reconstructing intent from logs.
* **A sequence can be validated without executing it**, which is what
  makes review possible at all.

This composes with the existing Will and admission chain rather than
replacing them: the dictionary says what is *well-formed*, admission says
what is *permitted*, and the Will says what is *chosen*.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Commands")


class ArgType(StrEnum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"
    ENUM = "enum"


class FailurePolicy(StrEnum):
    #: Stop the sequence. The default, because a step that failed usually
    #: invalidates the assumptions of the steps after it.
    ABORT = "abort"
    #: Log and keep going. For genuinely independent steps.
    CONTINUE = "continue"
    #: Retry up to the step's retry count, then abort.
    RETRY = "retry"


@dataclass(frozen=True)
class ArgSpec:
    name: str
    type: ArgType
    description: str = ""
    required: bool = True
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    allowed: tuple[Any, ...] = ()

    def validate(self, value: Any) -> tuple[Any, str]:
        """Returns (coerced value, error). Error empty means valid."""
        try:
            if self.type is ArgType.INT:
                coerced: Any = int(value)
            elif self.type is ArgType.FLOAT:
                coerced = float(value)
            elif self.type is ArgType.BOOL:
                coerced = bool(value) if isinstance(value, bool) else str(value).lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            else:
                coerced = str(value)
        except (TypeError, ValueError):
            return None, f"{self.name}={value!r} is not a valid {self.type}"
        if self.allowed and coerced not in self.allowed:
            return None, f"{self.name}={coerced!r} is not one of {list(self.allowed)}"
        if self.type in (ArgType.INT, ArgType.FLOAT):
            if self.minimum is not None and coerced < self.minimum:
                return None, f"{self.name}={coerced} is below the minimum {self.minimum}"
            if self.maximum is not None and coerced > self.maximum:
                return None, f"{self.name}={coerced} is above the maximum {self.maximum}"
        return coerced, ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": str(self.type),
            "description": self.description,
            "required": self.required,
            "default": self.default,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "allowed": list(self.allowed),
        }


@dataclass(frozen=True)
class CommandSpec:
    opcode: int
    name: str
    description: str
    owner: str
    args: tuple[ArgSpec, ...] = ()
    handler: Callable[..., Any] | None = None
    #: Commands whose effect is hard to undo. The sequencer refuses to
    #: include one in a sequence unless the sequence declares it.
    consequential: bool = False

    def validate(self, args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        resolved: dict[str, Any] = {}
        errors: list[str] = []
        declared = {a.name for a in self.args}
        for extra in set(args) - declared:
            errors.append(f"{extra!r} is not an argument of {self.name!r}")
        for spec in self.args:
            if spec.name not in args:
                if spec.required and spec.default is None:
                    errors.append(f"{spec.name!r} is required")
                else:
                    resolved[spec.name] = spec.default
                continue
            value, error = spec.validate(args[spec.name])
            if error:
                errors.append(error)
            else:
                resolved[spec.name] = value
        return resolved, errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "opcode": self.opcode,
            "name": self.name,
            "description": self.description,
            "owner": self.owner,
            "consequential": self.consequential,
            "args": [a.to_dict() for a in self.args],
            "has_handler": self.handler is not None,
        }


@dataclass
class CommandResult:
    name: str
    opcode: int
    ok: bool
    args: dict[str, Any] = field(default_factory=dict)
    value: Any = None
    error: str = ""
    duration_s: float = 0.0
    at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "opcode": self.opcode,
            "ok": self.ok,
            "args": dict(self.args),
            "error": self.error,
            "duration_ms": round(self.duration_s * 1000, 2),
            "at": self.at,
        }


@dataclass(frozen=True)
class Step:
    command: str
    args: dict[str, Any] = field(default_factory=dict)
    #: Seconds after the previous step completes.
    delay_s: float = 0.0
    on_failure: FailurePolicy = FailurePolicy.ABORT
    retries: int = 0
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "args": dict(self.args),
            "delay_s": self.delay_s,
            "on_failure": str(self.on_failure),
            "retries": self.retries,
            "label": self.label,
        }


@dataclass(frozen=True)
class CommandSequence:
    name: str
    steps: tuple[Step, ...]
    description: str = ""
    #: Must be True for a sequence containing consequential commands.
    allows_consequential: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "allows_consequential": self.allows_consequential,
            "steps": [s.to_dict() for s in self.steps],
        }


class CommandDispatcher:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._commands: dict[str, CommandSpec] = {}
        self._by_opcode: dict[int, str] = {}
        self._log: list[CommandResult] = []
        self.dispatched = 0
        self.rejected = 0

    # ── dictionary ────────────────────────────────────────────────────
    def declare(self, spec: CommandSpec) -> CommandSpec:
        with self._lock:
            existing = self._commands.get(spec.name)
            if existing is not None:
                if existing.opcode != spec.opcode:
                    raise ValueError(
                        f"command {spec.name!r} already has opcode {existing.opcode}"
                    )
                self._commands[spec.name] = spec
                return spec
            claimed = self._by_opcode.get(spec.opcode)
            if claimed is not None and claimed != spec.name:
                raise ValueError(
                    f"opcode {spec.opcode} is already {claimed!r}; opcodes are the "
                    "contract between a plan and what runs it"
                )
            self._commands[spec.name] = spec
            self._by_opcode[spec.opcode] = spec.name
            return spec

    def get(self, name: str) -> CommandSpec | None:
        with self._lock:
            return self._commands.get(name)

    def dictionary(self) -> dict[str, Any]:
        with self._lock:
            commands = [c.to_dict() for c in self._commands.values()]
        return {
            "version": 1,
            "commands": sorted(commands, key=lambda c: c["opcode"]),
            "consequential": [c["name"] for c in commands if c["consequential"]],
        }

    # ── dispatch ──────────────────────────────────────────────────────
    async def dispatch(self, name: str, **args: Any) -> CommandResult:
        spec = self.get(name)
        if spec is None:
            self.rejected += 1
            return self._record(
                CommandResult(name=name, opcode=-1, ok=False, error="no such command")
            )
        resolved, errors = spec.validate(args)
        if errors:
            self.rejected += 1
            return self._record(
                CommandResult(
                    name=name,
                    opcode=spec.opcode,
                    ok=False,
                    args=args,
                    error="; ".join(errors),
                )
            )
        if spec.handler is None:
            self.rejected += 1
            return self._record(
                CommandResult(
                    name=name,
                    opcode=spec.opcode,
                    ok=False,
                    args=resolved,
                    error="command is declared but has no handler",
                )
            )

        started = time.perf_counter()
        try:
            outcome = spec.handler(**resolved)
            if asyncio.iscoroutine(outcome):
                outcome = await outcome
            result = CommandResult(
                name=name,
                opcode=spec.opcode,
                ok=outcome is not False,
                args=resolved,
                value=outcome,
                duration_s=time.perf_counter() - started,
            )
        except Exception as exc:  # noqa: BLE001 — a failed command is a result
            result = CommandResult(
                name=name,
                opcode=spec.opcode,
                ok=False,
                args=resolved,
                error=f"{type(exc).__name__}: {exc}",
                duration_s=time.perf_counter() - started,
            )
        self.dispatched += 1
        return self._record(result)

    def _record(self, result: CommandResult) -> CommandResult:
        with self._lock:
            self._log.append(result)
            if len(self._log) > 256:
                del self._log[:-256]
        try:
            from core.fsw.telemetry_dictionary import EventSeverity, emit_event

            emit_event(
                "command_dispatched",
                severity=EventSeverity.ACTIVITY_HI if result.ok else EventSeverity.WARNING_LO,
                command=result.name,
                opcode=result.opcode,
                ok=result.ok,
                error=result.error,
                duration_ms=round(result.duration_s * 1000, 2),
            )
        except Exception:  # noqa: BLE001
            logger.debug("command telemetry failed", exc_info=True)
        return result

    # ── sequences ─────────────────────────────────────────────────────
    def validate_sequence(self, sequence: CommandSequence) -> list[str]:
        """Check a whole plan before any of it runs.

        This is the point of the whole module: step 7 failing on a
        malformed argument means steps 1-6 already happened.
        """
        problems: list[str] = []
        for index, step in enumerate(sequence.steps):
            spec = self.get(step.command)
            label = step.label or f"step {index + 1}"
            if spec is None:
                problems.append(f"{label}: no such command {step.command!r}")
                continue
            if spec.consequential and not sequence.allows_consequential:
                problems.append(
                    f"{label}: {step.command!r} is consequential but the sequence "
                    "does not declare allows_consequential"
                )
            _, errors = spec.validate(step.args)
            problems.extend(f"{label}: {error}" for error in errors)
            if step.on_failure is FailurePolicy.RETRY and step.retries <= 0:
                problems.append(f"{label}: RETRY policy with no retries declared")
        return problems

    async def run_sequence(
        self, sequence: CommandSequence, *, dry_run: bool = False
    ) -> dict[str, Any]:
        problems = self.validate_sequence(sequence)
        if problems:
            return {
                "sequence": sequence.name,
                "ok": False,
                "validated": False,
                "problems": problems,
                "results": [],
            }
        if dry_run:
            return {
                "sequence": sequence.name,
                "ok": True,
                "validated": True,
                "dry_run": True,
                "problems": [],
                "results": [],
            }

        results: list[CommandResult] = []
        aborted_at: int | None = None
        for index, step in enumerate(sequence.steps):
            if step.delay_s > 0:
                await asyncio.sleep(step.delay_s)
            attempts = 1 + (step.retries if step.on_failure is FailurePolicy.RETRY else 0)
            result = None
            for attempt in range(attempts):
                result = await self.dispatch(step.command, **step.args)
                if result.ok:
                    break
                if attempt + 1 < attempts:
                    logger.info(
                        "sequence %s step %d failed (%s); retry %d/%d",
                        sequence.name,
                        index + 1,
                        result.error,
                        attempt + 1,
                        step.retries,
                    )
            assert result is not None
            results.append(result)
            if not result.ok and step.on_failure is not FailurePolicy.CONTINUE:
                aborted_at = index
                logger.warning(
                    "🛑 sequence %s aborted at step %d (%s): %s",
                    sequence.name,
                    index + 1,
                    step.command,
                    result.error,
                )
                break

        return {
            "sequence": sequence.name,
            "ok": aborted_at is None and all(r.ok for r in results),
            "validated": True,
            "problems": [],
            "aborted_at_step": None if aborted_at is None else aborted_at + 1,
            "completed_steps": len(results),
            "total_steps": len(sequence.steps),
            "results": [r.to_dict() for r in results],
        }

    def report(self) -> dict[str, Any]:
        with self._lock:
            log = [r.to_dict() for r in self._log[-8:]]
            commands = len(self._commands)
        return {
            "commands": commands,
            "dispatched": self.dispatched,
            "rejected": self.rejected,
            "recent": log,
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._commands.clear()
            self._by_opcode.clear()
            self._log.clear()
            self.dispatched = 0
            self.rejected = 0


_DISPATCHER = CommandDispatcher()


def get_dispatcher() -> CommandDispatcher:
    return _DISPATCHER


def command(
    opcode: int,
    name: str,
    *,
    description: str,
    owner: str,
    args: SequenceABC[ArgSpec] = (),
    consequential: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Declare a command next to its handler::

        @command(0x10, "set_lane_count",
                 description="change resident model lane count",
                 owner="core/runtime/model_lane_control.py",
                 args=[ArgSpec("lanes", ArgType.INT, minimum=1, maximum=4)])
        async def set_lane_count(lanes: int) -> bool:
            ...
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        _DISPATCHER.declare(
            CommandSpec(
                opcode=opcode,
                name=name,
                description=description,
                owner=owner,
                args=tuple(args),
                handler=fn,
                consequential=consequential,
            )
        )
        return fn

    return decorate


def command_report() -> dict[str, Any]:
    return _DISPATCHER.report()


def reset_commands_for_test() -> None:
    _DISPATCHER.reset_for_test()


def install_runtime_commands() -> list[str]:
    """Commands the runtime's own disciplines already expose as actions."""

    @command(
        0x01,
        "set_parameter",
        description="change a declared runtime parameter",
        owner="core/runtime/parameters.py",
        args=[
            ArgSpec("name", ArgType.STRING, description="declared parameter name"),
            ArgSpec("value", ArgType.STRING, description="new value, coerced by type"),
            ArgSpec("reason", ArgType.STRING, required=False, default=""),
        ],
    )
    def _set_parameter(name: str, value: str, reason: str = "") -> bool:
        from core.runtime.parameters import set_parameter

        return set_parameter(name, value, source="command", reason=reason).successful

    @command(
        0x02,
        "dump_bus_ring",
        description="write the recent event-bus ring to disk",
        owner="core/observability/bus_recorder.py",
        args=[ArgSpec("seconds", ArgType.FLOAT, required=False, default=60.0, minimum=1.0, maximum=3600.0)],
    )
    async def _dump_bus(seconds: float = 60.0) -> bool:
        from core.observability.bus_recorder import get_bus_recorder

        return await get_bus_recorder().dump(reason="command", seconds=seconds) is not None

    @command(
        0x03,
        "write_trace",
        description="write the buffered trace events to a Perfetto-loadable file",
        owner="core/observability/trace_events.py",
    )
    async def _write_trace() -> bool:
        from core.observability.trace_events import get_tracer

        return await get_tracer().write(reason="command") is not None

    @command(
        0x04,
        "run_verifier",
        description="run the structural verifier over one scope or all",
        owner="core/verify/invariants.py",
        args=[ArgSpec("scope", ArgType.STRING, required=False, default="")],
    )
    def _run_verifier(scope: str = "") -> bool:
        from core.verify.invariants import verify

        return verify(*( (scope,) if scope else () )).ok

    @command(
        0x05,
        "set_pass_bisect_limit",
        description="skip cognitive passes beyond an ordinal, to bisect a regression",
        owner="core/pipeline/pass_manager.py",
        args=[ArgSpec("limit", ArgType.INT, minimum=-1, maximum=100000)],
        consequential=True,
    )
    def _set_bisect(limit: int) -> bool:
        from core.pipeline.pass_manager import get_instrumentation

        get_instrumentation().set_bisect_limit(None if limit < 0 else limit)
        return True

    @command(
        0x06,
        "shed_memory",
        description="run one graded eviction pass",
        owner="core/runtime/eviction.py",
        args=[ArgSpec("dry_run", ArgType.BOOL, required=False, default=True)],
        consequential=True,
    )
    def _shed(dry_run: bool = True) -> bool:
        from core.runtime.eviction import get_eviction_manager

        outcome = get_eviction_manager().enforce(dry_run=dry_run)
        return bool(outcome.get("actions") is not None)

    return sorted(c["name"] for c in _DISPATCHER.dictionary()["commands"])


__all__ = [
    "ArgSpec",
    "ArgType",
    "CommandDispatcher",
    "CommandResult",
    "CommandSequence",
    "CommandSpec",
    "FailurePolicy",
    "Step",
    "command",
    "command_report",
    "get_dispatcher",
    "install_runtime_commands",
    "reset_commands_for_test",
]
