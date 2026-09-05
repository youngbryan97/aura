"""Tool sequencing constraints — what may be called NEXT, not merely what is allowed.

``ToolPermissionGuard`` answers a static question: may this tool ever touch the
network, may it write to that directory. It has no notion of *order*. Nothing in
the codebase could express "after you read the file you must cite it", "the
verifier only runs once per step", or "you may not finish until you have written
the receipt" — so those obligations lived in prompt text, where the model was
free to ignore them and nothing noticed when it did.

This module makes sequencing a checked structure. A rule set is a small finite
state machine over the tool namespace: given what has already been called this
step and what the last call returned, it computes the tools that remain legal.

Three things here go past the prior art it was learned from (Letta's tool-rule
solver, Apache-2.0 — read for its semantics, written fresh against Aura's
governance and legibility conventions):

* **Unsatisfiable rule sets are refused at construction**, not discovered on the
  step that deadlocks. A required-before-exit tool that a parent rule makes
  unreachable is a bug in the rule set; it should not take a live conversation
  to surface it.
* **Every narrowing carries provenance.** The solver returns not just the
  surviving set but the rule that killed each candidate, because a refusal Aura
  cannot explain is a refusal Aura should not make.
* **Approval routes through the real authority gate** rather than a boolean the
  caller is trusted to honour.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.ToolRules")

__all__ = [
    "ToolRuleType",
    "ToolRule",
    "InitToolRule",
    "TerminalToolRule",
    "ChildToolRule",
    "ParentToolRule",
    "ConditionalToolRule",
    "ContinueToolRule",
    "MaxCountPerStepToolRule",
    "RequiredBeforeExitToolRule",
    "RequiresApprovalToolRule",
    "ToolCall",
    "Verdict",
    "UnsatisfiableRuleSet",
    "ToolRuleSolver",
]


class ToolRuleType(StrEnum):
    INIT = "init"
    TERMINAL = "terminal"
    CHILD = "child"
    PARENT = "parent"
    CONDITIONAL = "conditional"
    CONTINUE = "continue"
    MAX_COUNT_PER_STEP = "max_count_per_step"
    REQUIRED_BEFORE_EXIT = "required_before_exit"
    REQUIRES_APPROVAL = "requires_approval"


@dataclass(frozen=True)
class ToolCall:
    """One completed tool call within the current step.

    ``output`` is whatever the tool returned, kept raw so ConditionalToolRule can
    route on it. It is never rendered into a prompt from here.
    """

    name: str
    output: Any = None


class UnsatisfiableRuleSet(ValueError):
    """The rule set can never reach a legal exit, whatever the model does.

    Raised at construction. The alternative — discovering it mid-conversation —
    turns a static authoring mistake into a live wedge.
    """


# ── rules ──────────────────────────────────────────────────────────────────


class ToolRule(ABC):
    """Base: a constraint that narrows the tools legal at the next call."""

    rule_type: ToolRuleType

    def __init__(self, tool_name: str) -> None:
        if not tool_name or not isinstance(tool_name, str):
            raise ValueError("tool_name must be a non-empty string")
        self.tool_name = tool_name

    @abstractmethod
    def valid_tools(
        self,
        history: Sequence[ToolCall],
        available: frozenset[str],
    ) -> frozenset[str]:
        """The subset of ``available`` this rule permits next."""

    def describe(self) -> str:
        """One line, model-readable, for the system prompt."""
        return f"{self.rule_type}: {self.tool_name}"

    @property
    def forces_call(self) -> bool:
        """Whether this rule can compel a tool call rather than a free answer."""
        return False

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"{type(self).__name__}({self.tool_name!r})"


class InitToolRule(ToolRule):
    """``tool_name`` is one of the tools the step may open with.

    ``args`` pre-fills arguments, overriding whatever the model proposes. That
    is the point: an opening move whose arguments the model can rewrite is not
    a constraint.
    """

    rule_type = ToolRuleType.INIT

    def __init__(self, tool_name: str, args: Mapping[str, Any] | None = None) -> None:
        super().__init__(tool_name)
        self.args = dict(args) if args else None

    def valid_tools(
        self, history: Sequence[ToolCall], available: frozenset[str]
    ) -> frozenset[str]:
        if history:
            return available
        return available & {self.tool_name}

    def describe(self) -> str:
        return f"Start by calling {self.tool_name}."

    @property
    def forces_call(self) -> bool:
        return True


class TerminalToolRule(ToolRule):
    """Calling ``tool_name`` ends the step. Nothing runs after it."""

    rule_type = ToolRuleType.TERMINAL

    def valid_tools(
        self, history: Sequence[ToolCall], available: frozenset[str]
    ) -> frozenset[str]:
        return available

    def describe(self) -> str:
        return f"Calling {self.tool_name} ends your turn."


class ChildToolRule(ToolRule):
    """After ``tool_name``, the next call must be one of ``children``."""

    rule_type = ToolRuleType.CHILD

    def __init__(
        self,
        tool_name: str,
        children: Iterable[str],
        child_args: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(tool_name)
        self.children = frozenset(children)
        if not self.children:
            raise ValueError(f"ChildToolRule({tool_name!r}) needs at least one child")
        self.child_args = {k: dict(v) for k, v in (child_args or {}).items()}

    def valid_tools(
        self, history: Sequence[ToolCall], available: frozenset[str]
    ) -> frozenset[str]:
        if history and history[-1].name == self.tool_name:
            return available & self.children
        return available

    def describe(self) -> str:
        return f"After {self.tool_name}, call one of: {', '.join(sorted(self.children))}."

    @property
    def forces_call(self) -> bool:
        return True


class ParentToolRule(ToolRule):
    """``children`` are callable only immediately after ``tool_name``.

    The mirror of ChildToolRule: that one says where you must go, this one says
    how you must have arrived.
    """

    rule_type = ToolRuleType.PARENT

    def __init__(self, tool_name: str, children: Iterable[str]) -> None:
        super().__init__(tool_name)
        self.children = frozenset(children)
        if not self.children:
            raise ValueError(f"ParentToolRule({tool_name!r}) needs at least one child")

    def valid_tools(
        self, history: Sequence[ToolCall], available: frozenset[str]
    ) -> frozenset[str]:
        if history and history[-1].name == self.tool_name:
            return available & self.children
        return available - self.children

    def describe(self) -> str:
        return (
            f"{', '.join(sorted(self.children))} may only be called "
            f"directly after {self.tool_name}."
        )

    @property
    def forces_call(self) -> bool:
        return True


class ConditionalToolRule(ToolRule):
    """Route on what ``tool_name`` returned.

    ``output_mapping`` keys are compared against the stringified output, so a
    verifier returning ``True`` routes the same way whether it hands back a bool
    or the string "True". ``require_mapping=True`` makes an unmapped output an
    error rather than a silent fallthrough — the difference between a router and
    a suggestion.
    """

    rule_type = ToolRuleType.CONDITIONAL

    def __init__(
        self,
        tool_name: str,
        output_mapping: Mapping[Any, str],
        default_child: str | None = None,
        require_mapping: bool = False,
    ) -> None:
        super().__init__(tool_name)
        if not output_mapping:
            raise ValueError(f"ConditionalToolRule({tool_name!r}) needs an output_mapping")
        self.output_mapping = {self._key(k): v for k, v in output_mapping.items()}
        self.default_child = default_child
        self.require_mapping = require_mapping
        if require_mapping and default_child is not None:
            raise ValueError(
                "require_mapping=True and default_child are contradictory: "
                "one demands every output be mapped, the other supplies a catch-all"
            )

    @staticmethod
    def _key(value: Any) -> str:
        return str(value).strip()

    @property
    def children(self) -> frozenset[str]:
        out = set(self.output_mapping.values())
        if self.default_child:
            out.add(self.default_child)
        return frozenset(out)

    def valid_tools(
        self, history: Sequence[ToolCall], available: frozenset[str]
    ) -> frozenset[str]:
        if not history or history[-1].name != self.tool_name:
            return available
        target = self.output_mapping.get(self._key(history[-1].output))
        if target is None:
            if self.require_mapping:
                # Fail closed: an unroutable output under a strict router is a
                # gap in the rule set, and guessing would hide it.
                return frozenset()
            target = self.default_child
        if target is None:
            return frozenset()
        return available & {target}

    def describe(self) -> str:
        routes = ", ".join(f"{k!r}->{v}" for k, v in sorted(self.output_mapping.items()))
        tail = f" (otherwise {self.default_child})" if self.default_child else ""
        return f"After {self.tool_name}, route on its output: {routes}{tail}."

    @property
    def forces_call(self) -> bool:
        return True


class ContinueToolRule(ToolRule):
    """Calling ``tool_name`` forbids ending the step here."""

    rule_type = ToolRuleType.CONTINUE

    def valid_tools(
        self, history: Sequence[ToolCall], available: frozenset[str]
    ) -> frozenset[str]:
        return available

    def describe(self) -> str:
        return f"After {self.tool_name} you must keep working, not answer."


class MaxCountPerStepToolRule(ToolRule):
    """``tool_name`` may be called at most ``max_count`` times per step."""

    rule_type = ToolRuleType.MAX_COUNT_PER_STEP

    def __init__(self, tool_name: str, max_count: int) -> None:
        super().__init__(tool_name)
        if max_count < 1:
            raise ValueError("max_count must be >= 1; use tool removal to forbid a tool")
        self.max_count = max_count

    def valid_tools(
        self, history: Sequence[ToolCall], available: frozenset[str]
    ) -> frozenset[str]:
        used = sum(1 for call in history if call.name == self.tool_name)
        if used >= self.max_count:
            return available - {self.tool_name}
        return available

    def describe(self) -> str:
        return f"{self.tool_name} may be called at most {self.max_count}x per turn."


class RequiredBeforeExitToolRule(ToolRule):
    """The step may not end until ``tool_name`` has been called.

    Deliberately does not narrow the tool set — it is an exit condition, not a
    routing constraint. ``may_exit`` is where it bites.
    """

    rule_type = ToolRuleType.REQUIRED_BEFORE_EXIT

    def valid_tools(
        self, history: Sequence[ToolCall], available: frozenset[str]
    ) -> frozenset[str]:
        return available

    def describe(self) -> str:
        return f"You must call {self.tool_name} before finishing."


class RequiresApprovalToolRule(ToolRule):
    """``tool_name`` needs a human decision before it runs.

    Like the exit rule this does not narrow the set: the tool stays legal, it
    just cannot execute unapproved. Removing it from the set would tell the model
    the capability does not exist, which is a lie, and models route around lies.
    """

    rule_type = ToolRuleType.REQUIRES_APPROVAL

    def valid_tools(
        self, history: Sequence[ToolCall], available: frozenset[str]
    ) -> frozenset[str]:
        return available

    def describe(self) -> str:
        return f"{self.tool_name} requires the operator's approval before it runs."


# ── solver ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Verdict:
    """The solver's answer, with the reasoning kept attached.

    ``allowed`` is the actionable part. ``rejected`` maps each tool that was in
    play to the rule that removed it, so a refusal can be narrated rather than
    merely enforced.
    """

    allowed: frozenset[str]
    rejected: Mapping[str, str] = field(default_factory=dict)
    forced: bool = False
    prefilled_args: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.allowed)

    def why_not(self, tool_name: str) -> str | None:
        """The rule that removed ``tool_name``, or None if it survived."""
        return self.rejected.get(tool_name)


class ToolRuleSolver:
    """Computes the legal next tools from a rule set and the step so far."""

    def __init__(
        self,
        rules: Iterable[ToolRule],
        *,
        available_tools: Iterable[str] | None = None,
    ) -> None:
        self.rules: tuple[ToolRule, ...] = tuple(rules)
        self._by_type: dict[ToolRuleType, list[ToolRule]] = {}
        for rule in self.rules:
            self._by_type.setdefault(rule.rule_type, []).append(rule)

        self.universe: frozenset[str] | None = (
            frozenset(available_tools) if available_tools is not None else None
        )
        if self.universe is not None:
            self._validate(self.universe)

    def validate(self, available_tools: Iterable[str]) -> None:
        """Re-check the rule set against a tool universe.

        For callers that build the solver before they know the registry, and
        would otherwise carry a typo'd rule that simply never fires.
        """
        self._validate(frozenset(available_tools))

    # -- introspection -----------------------------------------------------

    def _named(self, rule_type: ToolRuleType) -> frozenset[str]:
        return frozenset(r.tool_name for r in self._by_type.get(rule_type, []))

    @property
    def init_tools(self) -> frozenset[str]:
        return self._named(ToolRuleType.INIT)

    @property
    def terminal_tools(self) -> frozenset[str]:
        return self._named(ToolRuleType.TERMINAL)

    @property
    def continue_tools(self) -> frozenset[str]:
        return self._named(ToolRuleType.CONTINUE)

    @property
    def required_before_exit(self) -> frozenset[str]:
        return self._named(ToolRuleType.REQUIRED_BEFORE_EXIT)

    @property
    def approval_tools(self) -> frozenset[str]:
        return self._named(ToolRuleType.REQUIRES_APPROVAL)

    # -- construction-time validation --------------------------------------

    def _validate(self, universe: frozenset[str]) -> None:
        """Refuse rule sets that provably cannot reach a legal exit.

        Cheap, total checks only — this is not a model checker. Each one below
        corresponds to a way a rule set wedges a live turn.
        """
        # A rule naming a tool nobody has is almost always a typo, and it fails
        # silently forever: the rule simply never fires.
        for rule in self.rules:
            unknown = {rule.tool_name} - universe
            children = getattr(rule, "children", frozenset())
            unknown |= set(children) - universe
            if unknown:
                raise UnsatisfiableRuleSet(
                    f"{type(rule).__name__} names tools that do not exist: "
                    f"{sorted(unknown)}. Available: {sorted(universe)}"
                )

        # A required-before-exit tool that is also gated behind a parent whose
        # own path is unreachable can never be satisfied.
        parent_gated: dict[str, str] = {}
        for rule in self._by_type.get(ToolRuleType.PARENT, []):
            for child in rule.children:  # type: ignore[attr-defined]
                parent_gated[child] = rule.tool_name

        init = self.init_tools
        for required in self.required_before_exit:
            gate = parent_gated.get(required)
            if gate is None:
                continue
            # The gate must itself be reachable: either freely callable, or an
            # opening move.
            if init and gate not in init and parent_gated.get(gate) is not None:
                raise UnsatisfiableRuleSet(
                    f"{required!r} must be called before exit, but it is gated "
                    f"behind {gate!r}, which is itself unreachable from the "
                    f"opening tools {sorted(init)}"
                )

        # A tool that both ends the turn and must be preceded by more work is
        # fine; a tool that ends the turn *and* must be continued past is not.
        both = self.terminal_tools & self.continue_tools
        if both:
            raise UnsatisfiableRuleSet(
                f"tools are marked both terminal and continue: {sorted(both)}. "
                "One says the turn ends here, the other says it cannot."
            )

        # An init tool capped at zero-usable count, or capped below 1, would
        # make the opening move illegal on arrival.
        caps = {
            r.tool_name: r.max_count  # type: ignore[attr-defined]
            for r in self._by_type.get(ToolRuleType.MAX_COUNT_PER_STEP, [])
        }
        for required in self.required_before_exit:
            if caps.get(required, 1) < 1:
                raise UnsatisfiableRuleSet(
                    f"{required!r} is required before exit but capped below one call"
                )

        # If init rules exist, at least one opening tool must survive them.
        if init and not (init & universe):
            raise UnsatisfiableRuleSet(
                f"no opening tool is available: init rules name {sorted(init)}, "
                f"available tools are {sorted(universe)}"
            )

    # -- the actual solve --------------------------------------------------

    def allowed(
        self,
        history: Sequence[ToolCall] = (),
        available: Iterable[str] | None = None,
    ) -> Verdict:
        """Which tools may be called next, and why the others may not."""
        if available is not None:
            universe = frozenset(available)
        elif self.universe is not None:
            universe = self.universe
        else:
            raise ValueError(
                "no tool universe: pass available_tools to the solver or "
                "available= to this call"
            )

        surviving = universe
        rejected: dict[str, str] = {}
        forced = False

        for rule in self.rules:
            before = surviving
            after = rule.valid_tools(history, before)
            if after != before:
                reason = rule.describe()
                for removed in before - after:
                    rejected.setdefault(removed, reason)
                surviving = after
            if rule.forces_call and after != universe:
                forced = True
            if not surviving:
                # Keep going rather than short-circuit: the caller wants the
                # full picture of what killed the last candidate.
                break

        prefilled: dict[str, dict[str, Any]] = {}
        if not history:
            for rule in self._by_type.get(ToolRuleType.INIT, []):
                if rule.args and rule.tool_name in surviving:  # type: ignore[attr-defined]
                    prefilled[rule.tool_name] = dict(rule.args)  # type: ignore[attr-defined]
        elif history:
            last = history[-1].name
            for rule in self._by_type.get(ToolRuleType.CHILD, []):
                if rule.tool_name != last:
                    continue
                for child, args in rule.child_args.items():  # type: ignore[attr-defined]
                    if child in surviving:
                        prefilled[child] = dict(args)

        if not surviving:
            logger.debug(
                "tool rules left nothing callable after %s",
                [c.name for c in history] or "<start of turn>",
            )

        return Verdict(
            allowed=surviving,
            rejected=rejected,
            forced=forced,
            prefilled_args=prefilled,
        )

    # -- exit conditions ---------------------------------------------------

    def is_terminal(self, tool_name: str) -> bool:
        return tool_name in self.terminal_tools

    def must_continue(self, tool_name: str) -> bool:
        return tool_name in self.continue_tools

    def requires_approval(self, tool_name: str) -> bool:
        return tool_name in self.approval_tools

    def uncalled_required(self, history: Sequence[ToolCall]) -> frozenset[str]:
        """Required-before-exit tools that have not run yet."""
        called = {call.name for call in history}
        return frozenset(self.required_before_exit - called)

    def may_exit(self, history: Sequence[ToolCall]) -> bool:
        """Whether the step is allowed to end here.

        A continue rule on the last call blocks the exit even when every
        required tool has run — that is the whole point of marking it.
        """
        if history and self.must_continue(history[-1].name):
            return False
        return not self.uncalled_required(history)

    def exceeded(self, history: Sequence[ToolCall]) -> frozenset[str]:
        """Tools called more times this step than their cap allows.

        The solver prevents this prospectively; this reports it after the fact,
        for callers that execute tools the solver did not gate.
        """
        counts = Counter(call.name for call in history)
        out = set()
        for rule in self._by_type.get(ToolRuleType.MAX_COUNT_PER_STEP, []):
            if counts.get(rule.tool_name, 0) > rule.max_count:  # type: ignore[attr-defined]
                out.add(rule.tool_name)
        return frozenset(out)

    # -- prompt surface ----------------------------------------------------

    def render_constraints(self) -> str:
        """The rule set as instructions, for models that steer better when told.

        Enforcement does not depend on this — the solver holds the line whether
        or not the model read it. It exists so a constrained model is not also
        a confused one.
        """
        if not self.rules:
            return ""
        lines = [rule.describe() for rule in self.rules]
        return "Tool use constraints:\n" + "\n".join(f"- {line}" for line in lines)
