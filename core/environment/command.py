"""Semantic action intents and physical command specs."""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .modal import ModalState

ActionRisk = Literal["safe", "caution", "risky", "irreversible", "forbidden"]


@dataclass
class ActionIntent:
    name: str
    target_id: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    expected_effect: str = ""
    risk: ActionRisk = "safe"
    tags: set[str] = field(default_factory=set)
    requires_authority: bool = False

    def intent_id(self) -> str:
        payload = repr((self.name, self.target_id, sorted(self.parameters.items()), self.risk, sorted(self.tags)))
        return "intent_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class CommandStep:
    kind: Literal["key", "text", "click", "shell", "api", "wait", "observe"]
    value: str
    timeout_s: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CommandSpec:
    command_id: str
    environment_id: str
    intent: ActionIntent
    preconditions: list[str]
    steps: list[CommandStep]
    expected_effects: list[str]
    expected_modal: ModalState | None = None
    rollback: "CommandSpec | None" = None
    trace_id: str = ""
    receipt_id: str | None = None
    created_at: float = field(default_factory=time.time)

    def is_effectful(self) -> bool:
        return any(step.kind not in {"observe", "wait"} for step in self.steps)

    def validate(self) -> None:
        if not self.command_id:
            raise ValueError("command_id_required")
        if not self.environment_id:
            raise ValueError("environment_id_required")
        if not isinstance(self.intent, ActionIntent):
            raise TypeError("intent_must_be_action_intent")
        if not self.steps:
            raise ValueError("command_steps_required")
        if self.intent.risk in {"irreversible", "forbidden"} and not self.intent.requires_authority:
            raise ValueError("irreversible_or_forbidden_intent_requires_authority")
        for step in self.steps:
            if step.timeout_s <= 0:
                raise ValueError("step_timeout_must_be_positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CommandCompiler:
    """Base semantic-intent compiler.

    Environment adapters can subclass or register handlers. Unknown intents
    fail closed so raw LLM text cannot become an effectful command.
    """

    def __init__(self, environment_id: str):
        self.environment_id = environment_id
        self._handlers: dict[str, Any] = {}

    def register(self, intent_name: str, handler: Any) -> None:
        self._handlers[str(intent_name)] = handler

    def compile(self, intent: ActionIntent, *, trace_id: str = "", receipt_id: str | None = None) -> CommandSpec:
        handler = self._handlers.get(intent.name)
        if handler is None:
            raise ValueError(f"unknown_intent:{intent.name}")
        command = handler(intent)
        if not isinstance(command, CommandSpec):
            raise TypeError("command compiler handler must return CommandSpec")
        command.trace_id = trace_id or command.trace_id
        command.receipt_id = receipt_id or command.receipt_id
        command.validate()
        return command


def command_id_for(environment_id: str, intent: ActionIntent) -> str:
    return f"cmd_{environment_id.replace(':', '_')}_{intent.intent_id()[-8:]}"


__all__ = [
    "ActionRisk",
    "ActionIntent",
    "CommandStep",
    "CommandSpec",
    "CommandCompiler",
    "command_id_for",
]
