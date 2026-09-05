"""core/tools/tool_registry.py — Tool Registry.

Stores, catalogs, and invokes all verified and forged tool classes.

Execution passes two checks the registry did not used to have:

* **Sequencing.** An optional ``ToolRuleSolver`` decides whether the tool may
  run *here* — after what has already run this step, given what the last call
  returned. Without a rule set the registry behaves exactly as before, so this
  is opt-in per caller rather than a new global constraint.
* **Tracing.** Every execution opens a ``execute_tool {name}`` span in the
  GenAI semantic convention, so tool work shows up in the same trace as the
  model call that asked for it.
* **Loop detection.** A :class:`~core.agency.stuck_detector.StuckDetector`
  watches calls and their outcomes. When the same call has produced the same
  result three times, the fourth is refused rather than run again. The refusal
  carries the evidence, and the loop is filed as a no-change impasse so it shows
  up in the same diagnostic as every other kind of deadlock.

  Refusing is the point. The repetition guard this codebase already had appends
  a paragraph to the prompt asking the model to try something else, which leaves
  the loop's fate to the faculty that produced it. A tool that will return the
  same answer for the fourth time is not worth the call.
"""
from __future__ import annotations

import ast
import logging
from typing import Any

from core.agency.stuck_detector import AgentStep, StuckDetector, StuckPattern
from core.observability.genai_semconv import tool_span
from core.sandbox.runner import run_untrusted
from core.tools.tool_rules import ToolCall, ToolRuleSolver

logger = logging.getLogger("Aura.ToolRegistry")

#: Loops where repeating the identical call provably cannot produce a new
#: answer, so the next attempt is refused. The other patterns are reported and
#: left to the caller — see ``_blocked_calls``.
_BLOCKING_PATTERNS = frozenset(
    {StuckPattern.REPEATED_ACTION_OBSERVATION, StuckPattern.REPEATED_ACTION_ERROR}
)


class ToolRegistry:
    """Central directory containing all operational tools."""

    def __init__(
        self,
        *,
        rules: ToolRuleSolver | None = None,
        stuck: StuckDetector | None = None,
    ) -> None:
        self._tools: dict[str, Any] = {}
        self._rules: ToolRuleSolver | None = None
        self._step: list[ToolCall] = []
        self._stuck = stuck if stuck is not None else StuckDetector(scope="tool_registry")
        #: Call identities refused because repeating them cannot produce a new
        #: answer, keyed by ``AgentStep.call_key``. Only the two repetition
        #: patterns populate this. An alternation or a stalled world is reported
        #: and not blocked: the remedy there is a different plan, and the
        #: registry has no basis for deciding which of the alternating calls is
        #: the wrong one.
        self._blocked_calls: dict[str, Any] = {}
        if rules is not None:
            self.set_rules(rules)

    @property
    def stuck_detector(self) -> StuckDetector:
        return self._stuck

    def register_tool(self, name: str, manifest: Any) -> None:
        self._tools[name] = manifest
        logger.info("📦 ToolRegistry: registered '%s'", name)

    def get_tool(self, name: str) -> Any | None:
        return self._tools.get(name)

    # ── sequencing ────────────────────────────────────────────────────────

    def set_rules(self, rules: ToolRuleSolver | None) -> None:
        """Install (or clear) the sequencing constraints.

        Validated against what is actually registered, so a rule naming a tool
        that does not exist is refused here rather than never firing.
        """
        if rules is not None and self._tools:
            rules.validate(self._tools.keys())
        self._rules = rules

    def begin_step(self) -> None:
        """Start a new step. Per-step budgets and child rules reset here.

        The loop window resets too. A new step is a new instruction, and an
        agent asked to read the same file again is doing as it was told rather
        than looping.
        """
        self._step = []
        self._stuck.reset()
        self._blocked_calls.clear()

    @property
    def step_history(self) -> tuple[ToolCall, ...]:
        return tuple(self._step)

    def may_finish(self) -> bool:
        """Whether the step is allowed to end — nothing required is outstanding."""
        if self._rules is None:
            return True
        return self._rules.may_exit(self._step)

    def outstanding(self) -> frozenset[str]:
        """Required-before-exit tools that have not run this step."""
        if self._rules is None:
            return frozenset()
        return self._rules.uncalled_required(self._step)

    # ── execution ─────────────────────────────────────────────────────────

    async def execute_tool(self, name: str, *args, **kwargs) -> dict[str, Any]:
        """Invoke a registered tool in the isolated tool sandbox."""
        manifest = self.get_tool(name)
        if not manifest:
            logger.error("🚫 ToolRegistry: tool '%s' not found", name)
            return {"ok": False, "error": f"tool_not_found:{name}"}

        call_key = AgentStep.of(
            name, arguments={"args": list(args), "kwargs": dict(kwargs)}
        ).call_key
        blocked = self._blocked_calls.get(call_key)
        if blocked is not None:
            logger.info("🔁 ToolRegistry: refusing repeat of '%s' — %s", name, blocked.detail)
            return {
                "ok": False,
                "error": f"tool_call_looping:{name}",
                "reason": blocked.detail,
                "pattern": blocked.pattern.value,
            }

        if self._rules is not None:
            verdict = self._rules.allowed(self._step, available=self._tools.keys())
            if name not in verdict.allowed:
                # The reason travels with the refusal. A tool blocked for an
                # unexplained reason is indistinguishable from one that is
                # broken, and the model will retry it either way.
                reason = verdict.why_not(name) or "no rule permits it at this point"
                logger.info("⛔ ToolRegistry: '%s' not permitted here — %s", name, reason)
                return {
                    "ok": False,
                    "error": f"tool_not_permitted_here:{name}",
                    "reason": reason,
                }

        with tool_span(name) as span:
            try:
                driver = _build_sandbox_driver(str(manifest.code), name, args, kwargs)
                sandbox_result = run_untrusted(driver)
                if sandbox_result.get("status") != "ok":
                    outcome = {
                        "ok": False,
                        "error": sandbox_result.get("repr")
                        or sandbox_result.get("stderr")
                        or sandbox_result.get("status"),
                    }
                elif not (stdout := str(sandbox_result.get("stdout") or "").strip()):
                    outcome = {"ok": False, "error": "tool_returned_no_result"}
                else:
                    outcome = {
                        "ok": True,
                        "result": ast.literal_eval(stdout.splitlines()[-1]),
                    }
            except (AttributeError, SyntaxError, TypeError, ValueError) as e:
                logger.error("Error executing tool %s: %s", name, e)
                outcome = {"ok": False, "error": str(e)}
            span.set_attribute("aura.tool.ok", outcome.get("ok", False))

        # Recorded whether it succeeded or not: a failed call still consumed
        # its budget and still moved the sequencing state. Counting only
        # successes would let a tool retry past its own per-step cap.
        self._step.append(ToolCall(name=name, output=outcome.get("result")))

        ok = bool(outcome.get("ok"))
        self._stuck.observe(
            AgentStep.of(
                name,
                arguments={"args": list(args), "kwargs": dict(kwargs)},
                observation=outcome.get("result") if ok else outcome.get("error"),
                failed=not ok,
                error_kind=str(outcome.get("error") or "")[:80],
            )
        )
        # Assessed after the call rather than before it, so the verdict is based
        # on what this call actually returned. The refusal it produces lands on
        # the next attempt, which is the first one that could have been skipped.
        verdict = self._stuck.assess_once(context={"tool": name})
        if verdict is not None:
            self._stuck.record_to_learner(verdict)
            logger.warning("🔁 ToolRegistry: %s — %s", verdict.pattern.value, verdict.detail)
            if verdict.pattern in _BLOCKING_PATTERNS:
                self._blocked_calls[call_key] = verdict
        return outcome


def _build_sandbox_driver(code: str, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    if not name.isidentifier():
        raise ValueError(f"tool name is not a valid Python identifier: {name}")
    return "\n".join(
        [
            code,
            f"_aura_args = {args!r}",
            f"_aura_kwargs = {kwargs!r}",
            "try:",
            f"    _aura_cls = {name}",
            "except NameError:",
            "    _aura_cls = None",
            "try:",
            "    _aura_main = main",
            "except NameError:",
            "    _aura_main = None",
            "if _aura_cls is not None:",
            "    _aura_result = _aura_cls().run(*_aura_args, **_aura_kwargs)",
            "elif _aura_main is not None:",
            "    _aura_result = _aura_main(*_aura_args, **_aura_kwargs)",
            "else:",
            "    raise Exception('run_method_or_main_not_found')",
            "print(repr(_aura_result))",
            "",
        ]
    )


_tool_registry_instance: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _tool_registry_instance
    if _tool_registry_instance is None:
        _tool_registry_instance = ToolRegistry()
    return _tool_registry_instance
