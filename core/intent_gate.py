"""core/intent_gate.py — Aura 3.0: IntentGate
==============================================
Replaces the MycelialNetwork regex engine with a clean, two-stage routing
architecture. All intent classification is now explicit, auditable, and
non-blocking.

Design philosophy:
  - "Administrative" commands (the old "Unblockable Root") are now handled
    through an explicit, locked AdminCommandRegistry that requires
    ServiceContainer supervision. No regex magic; only exact-string or
    simple-prefix matching for commands that must never be hijacked.
  - All other intents go through an async LLM-based classifier that runs
    in a background queue and never blocks the main cognitive path.
  - Physarum-style reinforcement is preserved but moved to a separate
    RouteStats ledger so it cannot alter routing logic at runtime.

ZENITH Protocol compliance:
  - Zero synchronous I/O in async methods.
  - Heavy work (LLM classification) runs in a non-blocking asyncio.Queue
    pipeline, decoupled from the message-receive path.
  - All container interactions are read-only during the hot path.
"""

from core.runtime.errors import record_degradation
import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.IntentGate")


# ---------------------------------------------------------------------------
# Enums & Routing Result
# ---------------------------------------------------------------------------

class RouteKind(str, Enum):
    """The tier of routing decision that was made."""
    ADMIN      = "admin"        # Hard-wired administrative/system command
    HARDWIRED  = "hardwired"    # Explicit registered pattern → skill shortcut
    LLM        = "llm"          # Routed after LLM intent classification
    PASSTHROUGH = "passthrough" # No match; falls through to main orchestrator


@dataclass
class RouteResult:
    """The resolved destination for a user message."""
    kind: RouteKind
    skill_name: Optional[str] = None        # Target skill, if applicable
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    matched_pattern: Optional[str] = None   # For audit logs
    latency_ms: float = 0.0

    @property
    def handled(self) -> bool:
        return self.kind != RouteKind.PASSTHROUGH


# ---------------------------------------------------------------------------
# Route Statistics (non-blocking, pure accumulator)
# ---------------------------------------------------------------------------

@dataclass
class RouteStats:
    """Immutable-style stats ledger for a single route. Thread-safe via GIL
    on CPython for integer increments; does NOT influence routing logic."""
    route_id: str
    hit_count: int = 0
    miss_count: int = 0
    total_latency_ms: float = 0.0

    def record(self, success: bool, latency_ms: float = 0.0) -> None:
        if success:
            self.hit_count += 1
        else:
            self.miss_count += 1
        self.total_latency_ms += latency_ms

    @property
    def success_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 1.0

    @property
    def avg_latency_ms(self) -> float:
        total = self.hit_count + self.miss_count
        return self.total_latency_ms / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Admin Command Registry  (replaces "Unblockable Root")
# ---------------------------------------------------------------------------

class AdminCommandRegistry:
    """
    A locked, audited registry for system-level commands that must always
    be reachable — even if the LLM is broken or the orchestrator is stalled.

    Unlike the old MycelialNetwork HardwiredPathway system, admin commands
    here are:
      1. Registered explicitly (no regex auto-discovery).
      2. Supervised: registration requires a container token to prevent
         rogue modules from injecting privileged routes at runtime.
      3. Matched by exact string or simple prefix — never greedy regex.

    The container_token is the string name of the registering service as
    known to the ServiceContainer, providing a lightweight audit trail.
    """

    _LOCKED_AFTER_BOOT = False   # Set to True after on_start() completes

    def __init__(self) -> None:
        self._commands: Dict[str, Tuple[str, Callable, str]] = {}
        # Maps command_key → (skill_name, handler_factory, container_token)
        self._stats: Dict[str, RouteStats] = {}

    def register(
        self,
        command_key: str,
        skill_name: str,
        handler_factory: Callable,
        container_token: str,
    ) -> None:
        """
        Register a privileged administrative command.

        Args:
            command_key: Exact lowercase string to match (e.g. "/shutdown",
                         "/reload", "system:debug"). Prefix matching is
                         supported by ending the key with "*" (e.g. "/dev/*").
            skill_name:  The skill that will handle this command.
            handler_factory: Callable that returns the coroutine to execute.
            container_token: The ServiceContainer service name of the caller,
                             used for audit logging.
        """
        if self._LOCKED_AFTER_BOOT:
            logger.error(
                "SECURITY: Rejected late admin registration for '%s' by '%s'. "
                "Admin commands must be registered during boot.",
                command_key, container_token
            )
            raise RuntimeError(
                f"AdminCommandRegistry is locked. Cannot register '{command_key}' post-boot."
            )
        self._commands[command_key.lower()] = (skill_name, handler_factory, container_token)
        self._stats[command_key.lower()] = RouteStats(route_id=command_key)
        logger.info(
            "AdminCommand registered: '%s' → skill='%s' (token='%s')",
            command_key, skill_name, container_token
        )

    def lock(self) -> None:
        """Call once after boot is complete to prevent further registration."""
        AdminCommandRegistry._LOCKED_AFTER_BOOT = True
        logger.info("AdminCommandRegistry LOCKED. %d commands active.", len(self._commands))

    def match(self, message: str) -> Optional[RouteResult]:
        """
        Exact and prefix match against admin commands.
        Returns None immediately if no match, preserving zero overhead
        for the common case (non-admin messages).
        """
        t0 = time.monotonic()
        msg_lower = message.strip().lower()

        # 1. Exact match (O(1))
        if msg_lower in self._commands:
            skill, factory, token = self._commands[msg_lower]
            self._stats[msg_lower].record(True, (time.monotonic() - t0) * 1000)
            logger.debug("AdminCommand exact match: '%s' → '%s'", msg_lower, skill)
            return RouteResult(
                kind=RouteKind.ADMIN,
                skill_name=skill,
                params={"handler": factory, "token": token},
                matched_pattern=msg_lower,
            )

        # 2. Prefix match for wildcard registrations (e.g. "/dev/*")
        for key, (skill, factory, token) in self._commands.items():
            if key.endswith("*") and msg_lower.startswith(key[:-1]):
                remainder = message[len(key) - 1:].strip()
                self._stats[key].record(True, (time.monotonic() - t0) * 1000)
                logger.debug("AdminCommand prefix match: '%s' → '%s'", key, skill)
                return RouteResult(
                    kind=RouteKind.ADMIN,
                    skill_name=skill,
                    params={"handler": factory, "token": token, "args": remainder},
                    matched_pattern=key,
                )

        return None

    def get_all_stats(self) -> List[Dict[str, Any]]:
        return [
            {
                "command": k,
                "skill": v[0],
                "token": v[2],
                "stats": {
                    "hits": self._stats[k].hit_count,
                    "misses": self._stats[k].miss_count,
                    "success_rate": self._stats[k].success_rate,
                    "avg_latency_ms": self._stats[k].avg_latency_ms,
                },
            }
            for k, v in self._commands.items()
        ]


# ---------------------------------------------------------------------------
# Hardwired Shortcut Registry (replaces HardwiredPathway greedy regexes)
# ---------------------------------------------------------------------------

@dataclass
class ShortcutRoute:
    """
    A single registered shortcut: a compiled regex pattern → skill mapping.

    CRITICAL DIFFERENCE from MycelialNetwork: patterns here are compiled and
    reviewed at registration time. They use non-greedy matching and are
    applied only after admin commands have already been checked. Critically,
    a failed shortcut match falls through gracefully to LLM routing rather
    than producing a robotic canned response.
    """
    route_id: str
    pattern: re.Pattern
    skill_name: str
    param_groups: Dict[str, str] = field(default_factory=dict)
    # Maps named capture group → param name for the skill
    priority: float = 1.0
    direct_response: Optional[str] = None  # Only for truly trivial cases
    stats: RouteStats = field(default_factory=lambda: RouteStats(route_id=""))

    def __post_init__(self) -> None:
        self.stats = RouteStats(route_id=self.route_id)


class ShortcutRegistry:
    """
    Registry for fast, non-LLM shortcut routes.

    All patterns must be non-greedy. Registration enforces that patterns
    cannot match empty strings (preventing the "greedy interception" bug
    from the old MycelialNetwork where short messages hijacked LLM routing).
    """

    def __init__(self) -> None:
        self._routes: List[ShortcutRoute] = []

    def register(
        self,
        route_id: str,
        pattern: str,
        skill_name: str,
        param_groups: Optional[Dict[str, str]] = None,
        priority: float = 1.0,
        direct_response: Optional[str] = None,
    ) -> None:
        compiled = re.compile(pattern, re.IGNORECASE)

        # Safety: reject patterns that match the empty string
        if compiled.match(""):
            raise ValueError(
                f"Pattern '{pattern}' matches empty string. "
                "This would intercept all messages. Rejected."
            )

        route = ShortcutRoute(
            route_id=route_id,
            pattern=compiled,
            skill_name=skill_name,
            param_groups=param_groups or {},
            priority=priority,
            direct_response=direct_response,
        )
        # Insert sorted by descending priority
        self._routes.append(route)
        self._routes.sort(key=lambda r: r.priority, reverse=True)
        logger.debug("ShortcutRoute registered: '%s' → '%s'", route_id, skill_name)

    def match(self, message: str) -> Optional[RouteResult]:
        """Try each route in priority order. Non-greedy: first match wins."""
        t0 = time.monotonic()
        for route in self._routes:
            m = route.pattern.search(message)
            if m:
                params: Dict[str, Any] = {}
                for group_name, param_name in route.param_groups.items():
                    try:
                        params[param_name] = m.group(group_name)
                    except IndexError as _exc:
                        logger.debug("Suppressed IndexError: %s", _exc)

                latency = (time.monotonic() - t0) * 1000
                route.stats.record(True, latency)

                if route.direct_response:
                    return RouteResult(
                        kind=RouteKind.HARDWIRED,
                        skill_name=route.skill_name,
                        params={"direct_response": route.direct_response, **params},
                        confidence=1.0,
                        matched_pattern=route.route_id,
                        latency_ms=latency,
                    )

                return RouteResult(
                    kind=RouteKind.HARDWIRED,
                    skill_name=route.skill_name,
                    params=params,
                    confidence=1.0,
                    matched_pattern=route.route_id,
                    latency_ms=latency,
                )
        return None

    def get_all_stats(self) -> List[Dict[str, Any]]:
        return [
            {
                "route_id": r.route_id,
                "pattern": r.pattern.pattern,
                "skill": r.skill_name,
                "priority": r.priority,
                "hits": r.stats.hit_count,
                "success_rate": r.stats.success_rate,
            }
            for r in self._routes
        ]


# ---------------------------------------------------------------------------
# Async LLM Intent Classification Queue
# ---------------------------------------------------------------------------

class IntentClassifierQueue:
    """
    Non-blocking LLM intent classifier. Classification requests are enqueued
    and processed by a background worker, so the main message-receive path
    is never blocked waiting for an LLM call.

    For the *synchronous* fast-path (where caller awaits the result), this
    still awaits the future — but the I/O itself is isolated in the worker
    so it doesn't hold any locks.

    ZENITH: No sync I/O inside any async method.
    """

    def __init__(self, max_queue: int = 64) -> None:
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._worker_task = get_task_tracker().create_task(
            self._worker_loop(), name="IntentClassifierQueue.worker"
        )
        logger.info("IntentClassifierQueue started.")

    async def stop(self) -> None:
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError as _exc:
                logger.debug("Suppressed asyncio.CancelledError: %s", _exc)
        logger.info("IntentClassifierQueue stopped.")

    async def classify(self, message: str, context: Optional[Dict] = None) -> RouteResult:
        """
        Classify a message via LLM. Awaits the result, but does not block
        the event loop — the actual LLM call happens in the worker coroutine.
        """
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        try:
            self._queue.put_nowait((message, context or {}, future))
        except asyncio.QueueFull:
            logger.warning("IntentClassifierQueue full — passing through without classification.")
            return RouteResult(kind=RouteKind.PASSTHROUGH)

        try:
            return await asyncio.wait_for(future, timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("LLM intent classification timed out for message: '%s...'", message[:40])
            return RouteResult(kind=RouteKind.PASSTHROUGH)

    async def _worker_loop(self) -> None:
        while self._running:
            got_item = False
            future: Optional[asyncio.Future] = None
            try:
                message, context, future = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
                got_item = True
                result = await self._classify_via_llm(message, context)
                if not future.done():
                    future.set_result(result)
            except asyncio.TimeoutError:
                continue  # Normal idle timeout, loop continues
            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation('intent_gate', e)
                if future is not None and not future.done():
                    future.set_result(RouteResult(kind=RouteKind.PASSTHROUGH))
                logger.error("IntentClassifierQueue worker error: %s", e, exc_info=True)
            finally:
                if got_item:
                    self._queue.task_done()

    async def _classify_via_llm(
        self, message: str, context: Dict
    ) -> RouteResult:
        """
        Calls the cognitive engine to classify intent. Falls back to
        PASSTHROUGH on any error so the system never hard-fails on
        classification issues.
        """
        t0 = time.monotonic()
        try:
            from core.container import ServiceContainer
            cognition = ServiceContainer.get("cognitive_engine", default=None)
            skill_registry = ServiceContainer.get("skill_registry", default=None)

            if not cognition or not skill_registry:
                return RouteResult(kind=RouteKind.PASSTHROUGH)

            available_skills = skill_registry.list_skill_names() if skill_registry else []
            skills_str = ", ".join(available_skills[:30])  # Cap to keep prompt tight

            prompt = (
                f"Given the user message below, identify if it clearly maps to one of "
                f"the available skills: [{skills_str}]. "
                f"If yes, respond ONLY with JSON: {{\"skill\": \"<name>\", \"confidence\": 0.0-1.0}}. "
                f"If no clear match, respond ONLY with: {{\"skill\": null, \"confidence\": 0.0}}.\n\n"
                f"Message: {message[:500]}"
            )

            from core.brain.cognitive_engine import ThinkingMode
            response = await cognition.think(
                objective=prompt, context={}, mode=ThinkingMode.FAST
            )
            content = (response.content if hasattr(response, "content") else str(response)).strip()

            import json
            import re as _re
            json_match = _re.search(r"\{[^}]+\}", content)
            if json_match:
                parsed = json.loads(json_match.group(0))
                skill = parsed.get("skill")
                confidence = float(parsed.get("confidence", 0.0))
                if skill and confidence >= 0.65:
                    return RouteResult(
                        kind=RouteKind.LLM,
                        skill_name=skill,
                        confidence=confidence,
                        latency_ms=(time.monotonic() - t0) * 1000,
                    )
        except Exception as e:
            record_degradation('intent_gate', e)
            logger.debug("LLM intent classification failed: %s", e)

        return RouteResult(kind=RouteKind.PASSTHROUGH)


# ---------------------------------------------------------------------------
# IntentGate — The Unified Public Interface
# ---------------------------------------------------------------------------

class IntentGate:
    """
    The single entry point for all intent routing in Aura 3.0.

    Routing happens in strict priority order:
      1. AdminCommandRegistry  — exact/prefix match, zero latency
      2. ShortcutRegistry      — compiled regex shortcuts, sub-ms
      3. IntentClassifierQueue — async LLM classification
      4. PASSTHROUGH           — falls through to the main orchestrator

    This replaces both MycelialNetwork.match_hardwired() and the old
    IntentRouter.classify() flow with a single, auditable pipeline.

    Background AST scanning (the old "Physical Import Graph" from
    mycelium.py) has been completely removed. It served no routing purpose
    and was a pure CPU overhead.
    """

    def __init__(self) -> None:
        self.admin = AdminCommandRegistry()
        self.shortcuts = ShortcutRegistry()
        self.classifier = IntentClassifierQueue()
        self._started = False

    async def on_start_async(self) -> None:
        """Boot sequence. Call from ServiceContainer lifecycle."""
        await self.classifier.start()
        # Lock admin commands so nothing can inject privileged routes at runtime
        self.admin.lock()
        self._started = True
        logger.info("IntentGate ONLINE. Admin: %d cmds, Shortcuts: %d routes.",
                    len(self.admin._commands), len(self.shortcuts._routes))

    async def on_stop_async(self) -> None:
        await self.classifier.stop()
        self._started = False

    async def route(
        self,
        message: str,
        context: Optional[Dict] = None,
        use_llm: bool = True,
    ) -> RouteResult:
        """
        Route a message to the appropriate skill or mark as PASSTHROUGH.

        Args:
            message:  The raw user message text.
            context:  Optional context dict forwarded to LLM classifier.
            use_llm:  If False, skips LLM classification (for hot inner loops
                      where you only want admin/shortcut matching).

        Returns:
            A RouteResult indicating where the message should go.
        """
        if not message or not message.strip():
            return RouteResult(kind=RouteKind.PASSTHROUGH)

        # Stage 1: Admin commands — always first, always unblockable
        admin_result = self.admin.match(message)
        if admin_result:
            logger.info("IntentGate → ADMIN: '%s'", admin_result.skill_name)
            return admin_result

        # Stage 2: Fast shortcut routes (compiled regex, no LLM)
        shortcut_result = self.shortcuts.match(message)
        if shortcut_result:
            logger.debug("IntentGate → SHORTCUT: '%s' (%.1fms)",
                         shortcut_result.skill_name, shortcut_result.latency_ms)
            return shortcut_result

        # Stage 3: LLM-based classification (async, non-blocking)
        if use_llm:
            llm_result = await self.classifier.classify(message, context)
            if llm_result.handled:
                logger.debug("IntentGate → LLM: '%s' (conf=%.2f)",
                             llm_result.skill_name, llm_result.confidence)
                return llm_result

        # Stage 4: No match — let the main orchestrator decide
        return RouteResult(kind=RouteKind.PASSTHROUGH)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Full introspection report for UI / health monitoring."""
        return {
            "admin_commands": self.admin.get_all_stats(),
            "shortcut_routes": self.shortcuts.get_all_stats(),
            "classifier_queue_size": self.classifier._queue.qsize(),
            "started": self._started,
        }


# ---------------------------------------------------------------------------
# Module-level singleton & ServiceContainer registration helper
# ---------------------------------------------------------------------------

_intent_gate: Optional[IntentGate] = None


def get_intent_gate() -> IntentGate:
    """Return the module-level IntentGate singleton."""
    global _intent_gate
    if _intent_gate is None:
        _intent_gate = IntentGate()
    return _intent_gate


def register_intent_gate() -> IntentGate:
    """Register IntentGate with the ServiceContainer. Call during boot."""
    from core.container import ServiceContainer
    gate = get_intent_gate()
    
    # Dynamic Volition Control
    async def _handle_volition(args: str = "0"):
        try:
            level = int(args.strip())
            kernel = ServiceContainer.get("aura_kernel", default=None)
            if kernel:
                kernel.set_volition_level(level)
                return f"✓ Volition Level set to {level}"
            return "❌ Kernel not found in container."
        except Exception as e:
            record_degradation('intent_gate', e)
            return f"❌ Volition error: {e}"

    gate.admin.register(
        command_key="/volition*",
        skill_name="GenesisControl",
        handler_factory=lambda: _handle_volition,
        container_token="bootloader"
    )

    ServiceContainer.register_instance("intent_gate", gate)
    logger.info("IntentGate registered with ServiceContainer (including /volition).")
    return gate
