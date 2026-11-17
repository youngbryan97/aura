"""Ratchet: no NEW unbounded awaits on wedge-capable primitives (roadmap A1).

The recorded loop-wedge class (mind_tick's 2-hour death, the warmup wedge,
12 loop-wedge crash dumps) is one mechanism: an ``await`` that can block
forever with no deadline and no watchdog visibility. This gate makes the
class static, the way the async-write-lane ratchet froze on-loop fsyncs.

Flagged: ``await x.wait()`` / ``.get()`` / ``.acquire()`` / ``.join()`` /
``.drain()`` / ``.wait_closed()`` with NO arguments (the asyncio primitives
that block forever by design), plus peer-controlled stream reads
(``readexactly`` / ``readline`` / ``readuntil``) — unless bounded by
``asyncio.wait_for(...)`` or an enclosing ``asyncio.timeout(...)`` /
``move_on_after(...)`` / ``fail_after(...)`` block.

The allowlist below freezes the historical debt at introduction. Entries
are one of two kinds, and both may only shrink:
  (a) wedge-capable debt — bound it with wait_for + a named timeout
      degradation (the tick_stage_timeout precedent), then delete it here;
  (b) deliberately-infinite consumer loops — migrate to timed waits that
      wake to check shutdown/liveness (the pattern that would have caught
      the mind_tick wedge hours earlier), then delete it here.

If this fails on code you just wrote: wrap the await in
``asyncio.wait_for(...)`` with an explicit budget and record a named
degradation on timeout. Do not add allowlist entries for new code.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_ZERO_ARG_UNBOUNDED = {"wait", "get", "acquire", "join", "drain", "wait_closed"}
_STREAM_READS = {"readexactly", "readline", "readuntil"}
_BOUNDING_WRAPPERS = {"wait_for"}
_BOUNDING_CONTEXTS = {"timeout", "timeout_at", "move_on_after", "fail_after"}

# (file, async function, callee) — historical debt frozen 2026-07-09 (68
# sites). ONLY shrinks: fix a site, delete its entry. Never add for new code.
#
# 2026-08-04: three entries removed, none by changing the code they named. The
# scanner was counting `await self.method()` as a primitive await, so a method
# that delegates to a BOUNDED one in the same class was reported as a wedge
# risk — including CheckedSemaphore.__aenter__, which meant lockdep, the module
# whose entire job is bounding acquires, failed this gate on its own wrapper.
# The callee is scanned in its own right; flagging the delegation counted the
# same await twice.
ALLOWED_LEGACY_OFFENDERS: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("core/adaptation/self_optimizer.py", "_run_gateway_command", "wait"),
        ("core/affect/affect_facade.py", "get", "get"),
        ("core/autonomy/autonomous_initiative_loop.py", "_event_listener_loop", "get"),
        ("core/brain/llm/nucleus_manager.py", "_listen_for_updates", "get"),
        ("core/brain/llm/nucleus_manager.py", "generate_stream_async", "get"),
        ("core/brain/reasoning_queue.py", "_run", "get"),
        ("core/bus/actor_bus.py", "_telemetry_broadcaster", "get"),
        ("core/bus/local_pipe_bus.py", "_dispatch_loop", "get"),
        ("core/bus/sensory_gate.py", "run", "wait"),
        ("core/capability_engine.py", "_execute_wrapped", "get"),
        ("core/cognition/cognitive_loop.py", "_acquire_next_message", "get"),
        ("core/cognitive/state_machine.py", "queue_sentence_generator", "get"),
        ("core/collective/probe_manager.py", "_listen_to_probe", "readline"),
        ("core/collective/swarm_protocol.py", "_handle_peer", "wait_closed"),
        ("core/consciousness/aura_protocol.py", "_handle_connection", "readexactly"),
        ("core/consciousness/constitutive_expression.py", "tick", "get"),
        ("core/consciousness/heartbeat.py", "_gather_state", "get"),
        ("core/consciousness/homeostatic_coupling.py", "_read_affect", "get"),
        ("core/consciousness/parallel_branches.py", "branch_yield", "wait"),
        ("core/consciousness/reasoning_queue.py", "get", "get"),
        ("core/conversation/tagged_reply_queue.py", "get", "get"),
        ("core/cybernetics/omni_tool.py", "_watch_daemon", "wait"),
        ("core/cybernetics/tricorder.py", "_process_empathy", "get"),
        ("core/cybernetics/tricorder.py", "_process_violations", "get"),
        ("core/embodiment/voice_presence.py", "speak", "wait"),
        ("core/managers/vram_manager.py", "acquire", "acquire"),
        ("core/memory/cognitive_vault.py", "on_stop_async", "join"),
        ("core/multimodal/coordinator.py", "subscribe", "get"),
        ("core/networking/hive_node.py", "_close_server", "wait_closed"),
        ("core/networking/hive_node.py", "_handle_peer", "wait_closed"),
        ("core/ops/daemon.py", "_handle_api_connection", "drain"),
        ("core/ops/daemon.py", "_handle_api_connection", "readline"),
        ("core/ops/daemon.py", "run", "wait"),
        ("core/ops/daemon.py", "stop", "wait_closed"),
        ("core/ops/graceful_shutdown.py", "wait_for_shutdown", "wait"),
        # Designed-forever wait: the root signal owner blocks main until
        # SIGTERM/SIGINT arrives — same class as wait_for_shutdown above.
        ("core/runtime/root_signal_owner.py", "wait", "wait"),
        ("core/orchestrator/main.py", "_setup_event_listeners", "get"),
        ("core/orchestrator/main.py", "run", "wait"),
        ("core/resilience/database_coordinator.py", "_process_writes", "get"),
        ("core/resilience/supervisor.py", "_pipe_logger_async", "readline"),
        ("core/resource/rate_limiter.py", "limit_rate", "acquire"),
        ("core/runtime/organ_supervisor.py", "_do", "drain"),
        ("core/runtime/organ_supervisor.py", "_do", "readexactly"),
        ("core/sandbox/bash_daemon.py", "_read_until_delimiter", "readline"),
        ("core/sandbox/bash_daemon.py", "_start", "drain"),
        ("core/sandbox/bash_daemon.py", "execute", "drain"),
        ("core/sandbox/bash_daemon.py", "kill", "wait"),
        ("core/self_modification/safe_pipeline.py", "_run_shadow", "wait"),
        ("core/senses/interaction_signals.py", "_typing_consumer", "get"),
        ("core/senses/interaction_signals.py", "_vision_consumer", "get"),
        ("core/senses/interaction_signals.py", "_voice_consumer", "get"),
        ("core/skills/sovereign_terminal.py", "_open_target", "wait"),
        ("core/state/state_repository.py", "_mutation_consumer_loop", "get"),
        ("core/utils/output_gate.py", "get_secondary_stream", "get"),
        ("core/utils/queues.py", "get", "get"),
    }
)


def _callee_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return ""


def _is_self_call(call: ast.Call) -> bool:
    """True for ``self.method(...)`` — delegation, not a primitive await."""
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "self"
    )


def _scan_offenders() -> set[tuple[str, str, str]]:
    offenders: set[tuple[str, str, str]] = set()
    for path in (PROJECT_ROOT / "core").rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        rel = str(path.relative_to(PROJECT_ROOT))
        local_async_methods = {
            child.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            for child in node.body
            if isinstance(child, ast.AsyncFunctionDef)
        }

        class Visitor(ast.NodeVisitor):
            def __init__(self, rel_path: str, async_methods: set[str]) -> None:
                self.rel_path = rel_path
                self.local_async_methods = async_methods
                self.async_stack: list[str] = []
                self.timeout_ctx_depth = 0

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self.async_stack.append(node.name)
                saved_depth = self.timeout_ctx_depth
                self.timeout_ctx_depth = 0
                self.generic_visit(node)
                self.timeout_ctx_depth = saved_depth
                self.async_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                # A sync def nested in an async def runs off-loop when handed
                # to executors — not this ratchet's target.
                saved = self.async_stack
                self.async_stack = []
                self.generic_visit(node)
                self.async_stack = saved

            def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
                bounding = any(
                    isinstance(item.context_expr, ast.Call)
                    and _callee_name(item.context_expr) in _BOUNDING_CONTEXTS
                    for item in node.items
                )
                if bounding:
                    self.timeout_ctx_depth += 1
                self.generic_visit(node)
                if bounding:
                    self.timeout_ctx_depth -= 1

            def visit_Await(self, node: ast.Await) -> None:
                if not self.async_stack or self.timeout_ctx_depth > 0:
                    self.generic_visit(node)
                    return
                value = node.value
                if isinstance(value, ast.Call):
                    name = _callee_name(value)
                    if name in _BOUNDING_WRAPPERS:
                        return  # bounded; do not descend into the wrapped call
                    # `await self.acquire()` delegating to a method DEFINED IN
                    # THIS FILE is not a primitive await — the callee is scanned
                    # in its own right, so flagging the delegation counts the
                    # same await twice. It reported CheckedSemaphore.__aenter__
                    # as a wedge risk when the acquire it calls is the module's
                    # own `asyncio.wait_for`-bounded wrapper: lockdep, the thing
                    # that exists to bound acquires, failing its own gate.
                    #
                    # Restricted to a `self.` receiver on purpose. `await
                    # queue.get()` stays flagged even in a file that happens to
                    # define a method called `get`.
                    if _is_self_call(value) and name in self.local_async_methods:
                        self.generic_visit(node)
                        return
                    flagged = (
                        name in _ZERO_ARG_UNBOUNDED
                        and not value.args
                        and not value.keywords
                    ) or name in _STREAM_READS
                    if flagged:
                        offenders.add((self.rel_path, self.async_stack[-1], name))
                self.generic_visit(node)

        Visitor(rel, local_async_methods).visit(tree)
    return offenders


def test_no_new_unbounded_awaits():
    offenders = _scan_offenders()
    new = offenders - ALLOWED_LEGACY_OFFENDERS
    assert not new, (
        "NEW unbounded await(s) on wedge-capable primitives — wrap in "
        "asyncio.wait_for(...) with an explicit budget and a named timeout "
        f"degradation (the tick_stage_timeout precedent): {sorted(new)}"
    )


def test_allowlist_carries_no_stale_entries():
    """A fixed site must leave the allowlist so the debt count is honest."""
    offenders = _scan_offenders()
    stale = ALLOWED_LEGACY_OFFENDERS - offenders
    assert not stale, (
        f"These entries are fixed (or moved) — delete them from "
        f"ALLOWED_LEGACY_OFFENDERS: {sorted(stale)}"
    )


def test_bounding_constructs_are_recognized():
    """The scanner must not flag properly bounded awaits (else the ratchet
    would train people to allowlist instead of bound)."""
    import textwrap

    snippet = textwrap.dedent(
        """
        import asyncio

        async def bounded_wait(evt):
            await asyncio.wait_for(evt.wait(), timeout=5.0)

        async def bounded_by_context(q):
            async with asyncio.timeout(2.0):
                await q.get()

        async def timed_get(q):
            await q.get(timeout=1.0)
        """
    )
    tree = ast.parse(snippet)

    flagged: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.async_stack: list[str] = []
            self.timeout_ctx_depth = 0

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.async_stack.append(node.name)
            self.generic_visit(node)
            self.async_stack.pop()

        def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
            bounding = any(
                isinstance(item.context_expr, ast.Call)
                and _callee_name(item.context_expr) in _BOUNDING_CONTEXTS
                for item in node.items
            )
            if bounding:
                self.timeout_ctx_depth += 1
            self.generic_visit(node)
            if bounding:
                self.timeout_ctx_depth -= 1

        def visit_Await(self, node: ast.Await) -> None:
            if self.async_stack and self.timeout_ctx_depth == 0:
                value = node.value
                if isinstance(value, ast.Call):
                    name = _callee_name(value)
                    if name in _BOUNDING_WRAPPERS:
                        return
                    if (
                        name in _ZERO_ARG_UNBOUNDED
                        and not value.args
                        and not value.keywords
                    ) or name in _STREAM_READS:
                        flagged.add(self.async_stack[-1])
            self.generic_visit(node)

    Visitor().visit(tree)
    assert flagged == set(), f"bounded awaits were wrongly flagged: {flagged}"
