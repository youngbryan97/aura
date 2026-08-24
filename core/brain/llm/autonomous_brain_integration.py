"""Autonomous Cognitive Engine.
Unifies Aura's managed local brain architecture:
Tier 1: Resident Cortex (PRIMARY)
Tier 2: On-demand local Solver
Tier 3: Local Brainstem and Reflex recovery

Drives the Mind/Body connection.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from core.config import config
from core.container import get_container
from core.runtime import service_access
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.runtime_settings import get_runtime_setting

from .function_calling_adapter import FunctionCallingAdapter
from .llm_router import IntelligentLLMRouter, LLMEndpoint, LLMTier
from .runtime_wiring import build_agentic_tool_map

logger = logging.getLogger("Aura.AutonomousBrain")

#: Failures of something OUTSIDE this module — an endpoint, a file, the network.
#: Recovering from these by routing elsewhere is correct.
ENDPOINT_FAILURE_ERRORS = (
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
)
#: Failures of the code in THIS process. CP126 64ab3dab: these were folded in
#: with endpoint failures around the whole thinking cycle, so a contract defect
#: came back as ordinary output at a made-up confidence instead of failing.
#: They are still caught — a defect must not take the runtime down mid-turn —
#: but they are classified, receipted, and reported as an internal error.
PROGRAMMING_DEFECT_ERRORS = (
    AttributeError,
    TypeError,
    ValueError,
)
BRAIN_RECOVERABLE_ERRORS = ENDPOINT_FAILURE_ERRORS + PROGRAMMING_DEFECT_ERRORS

#: Objective bounds. CP126 4bfedd29: an unvalidated objective was sliced in the
#: main path AND again while building the recovery path's metadata, so a bad
#: type raised a second exception before the fallback could run.
MAX_OBJECTIVE_CHARS = 32_000

#: Agentic work bounds (CP126 21cb172b).
MIN_TURNS = 1
MAX_TURNS = 25
DEFAULT_AGENTIC_DEADLINE_S = 300.0
MAX_AGENTIC_DEADLINE_S = 1800.0

#: Per-lane concurrency limits. CP126 25ec2c1d: a single semaphore of eight
#: guarded only the direct agentic call, so background, deep, fast and fallback
#: router calls ran outside any arbitration and could exceed the high-RAM
#: assumption the number was chosen for.
_DEFAULT_LANE_LIMITS = {
    "foreground": 4,
    "background": 2,
    "deep": 1,
    "agentic": 8,
}

#: How a degradation should be classified. CP126 1a184861: every degradation
#: was labelled SAFE_FALLBACK with receipt_required=False, so a genuinely safe
#: local omission (no dotenv file) was indistinguishable from a routing, cloud
#: or tool-authority failure that needs a receipt.
_DEGRADATION_KINDS: dict[str, tuple[FallbackClassification, bool]] = {
    # A capability was simply not present locally; nothing was bypassed.
    "local_omission": (FallbackClassification.SAFE_FALLBACK, False),
    # A tier/endpoint is gone: capability was lost, quietly, until now.
    "capability_loss": (FallbackClassification.SILENT_LOSS_OF_CAPABILITY, True),
    # Routing chose, or failed to choose, under uncertainty.
    "routing": (FallbackClassification.SILENT_LOSS_OF_CAPABILITY, True),
    # A proof lane or tool authority WITHHELD something.
    # Deliberately not GOVERNANCE_BYPASS: record_degradation treats that
    # classification as fail-closed and RAISES CapabilityDenied regardless of
    # enforce_failure_policy. A working control must not become an exception
    # that takes down the turn. A bypass is when the
    # control did NOT hold; nothing in this module can do that.
    "policy": (FallbackClassification.SILENT_LOSS_OF_CAPABILITY, True),
    # A defect in this module's own contract.
    "internal_defect": (FallbackClassification.AUDIT_GAP, True),
}


def _record_brain_degradation(
    error: BaseException,
    *,
    action: str,
    kind: str = "local_omission",
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    """Record a degradation with a classification that reflects what failed.

    ``kind`` selects the classification and whether a receipt is required, so a
    caller reading the ledger can tell a benign local omission from a policy or
    routing failure (CP126 1a184861).
    """
    classification, receipt_required = _DEGRADATION_KINDS.get(
        kind, _DEGRADATION_KINDS["local_omission"]
    )
    record_degradation(
        "autonomous_brain_integration",
        error,
        severity=severity,
        action=action,
        classification=classification,
        receipt_required=receipt_required,
        extra={**(extra or {}), "degradation_kind": kind},
        enforce_failure_policy=False,
    )


def objective_ref(objective: Any) -> dict[str, Any]:
    """A content-free reference to a prompt, safe for logs and receipts.

    CP126 c0fbd774: the complete objective was logged at info level and copied
    into traces and degradation extras. A user prompt can carry credentials,
    health information, or anything else the user typed; none of it belongs in
    a log line or a degradation record. This identifies a request without
    reproducing it.
    """
    text = objective if isinstance(objective, str) else str(objective)
    return {
        "ref": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16],
        "chars": len(text),
    }


def normalize_objective(objective: Any) -> str:
    """Coerce and bound an objective ONCE, before anything reads it."""
    if objective is None:
        return ""
    text = objective if isinstance(objective, str) else str(objective)
    if len(text) > MAX_OBJECTIVE_CHARS:
        return text[:MAX_OBJECTIVE_CHARS]
    return text


def bounded_turns(value: Any) -> int:
    """A policy-owned turn budget (CP126 21cb172b)."""
    try:
        turns = int(value)
    except (TypeError, ValueError):
        return 5
    return max(MIN_TURNS, min(MAX_TURNS, turns))


def bounded_priority(value: Any) -> float:
    try:
        priority = float(value)
    except (TypeError, ValueError):
        return 1.0
    if priority != priority or priority in (float("inf"), float("-inf")):
        return 1.0
    return max(0.0, min(1.0, priority))


def bounded_deadline(value: Any) -> float:
    try:
        deadline = float(value)
    except (TypeError, ValueError):
        return DEFAULT_AGENTIC_DEADLINE_S
    if deadline != deadline or deadline <= 0 or deadline == float("inf"):
        return DEFAULT_AGENTIC_DEADLINE_S
    return min(MAX_AGENTIC_DEADLINE_S, deadline)


@dataclass(frozen=True)
class ThinkRequest:
    """One immutable request envelope, computed once and reused everywhere.

    CP126 53dd1ff6 / f51792ca / 389d937a: the primary path ANDed caller intent
    with the user's cloud setting, then the RECOVERY path passed the raw caller
    flag — so a failure could route off-box after the user opted out. Recovery
    also dropped system_prompt and context and re-expanded the original kwargs
    on top of explicit keywords, raising a duplicate-keyword TypeError. One
    envelope, built once, removes all three: there is no second computation to
    disagree with the first, and no kwargs left to collide.
    """

    objective: str
    ref: dict[str, Any]
    context: dict[str, Any]
    system_prompt: str
    max_turns: int
    priority: float
    deadline_s: float
    is_background: bool
    deep_handoff: bool
    allow_cloud: bool
    #: Fail-closed: health could not be established. Withdraws cloud routing.
    safe_mode: bool
    #: The guardian actively reported ill health. Withdraws tools and turns —
    #: a stronger restriction, so it needs the stronger evidence.
    measured_unhealthy: bool
    stability_basis: str
    requested_endpoint: str | None
    caller_tools: Any
    passthrough: dict[str, Any] = field(default_factory=dict)

    def router_kwargs(
        self,
        *,
        prefer_endpoint: str | None,
        prefer_tier: str,
        deep_handoff: bool,
        allow_cloud: bool | None = None,
    ) -> dict[str, Any]:
        """The single sanitized argument set for a router call."""
        kwargs = dict(self.passthrough)
        kwargs.update(
            system_prompt=self.system_prompt,
            # CP126 69d1f269: context reached only the agentic route, so the
            # same objective behaved differently depending on which endpoint
            # happened to be healthy, and constraints or evidence a caller
            # supplied were silently dropped.
            context=self.context,
            priority=self.priority,
            prefer_endpoint=prefer_endpoint,
            prefer_tier=prefer_tier,
            deep_handoff=bool(deep_handoff),
            allow_cloud_fallback=self.allow_cloud if allow_cloud is None else bool(allow_cloud),
            is_background=self.is_background,
        )
        return kwargs



#: kwargs the request envelope owns. Anything here must NOT survive into the
#: passthrough set, or it collides with the explicit keyword of the same name
#: (CP126 37b3103e / 389d937a — both were duplicate-keyword TypeErrors).
_RESERVED_KWARGS = frozenset({
    "prefer_endpoint",
    "prefer_tier",
    "deep_handoff",
    "allow_deep_handoff",
    "allow_cloud_fallback",
    "is_background",
    "tools",
    "context",
    "system_prompt",
    "priority",
    "max_turns",
    "bypass_race",
    "deadline_s",
})


def _LOCAL_AGENTIC_ALLOWLIST(primary: str, deep: str, brainstem: str) -> tuple[str, ...]:
    """Endpoints permitted to run tool-bearing agentic work, in preference order.

    Local only, and explicit: CP126 af6a3b7d selected the first endpoint in
    dictionary order that happened to expose ``think_and_act``.
    """
    return (primary, deep, brainstem)


def _endpoint_is_usable(endpoint: Any) -> bool:
    """Whether an endpoint could actually answer a request.

    CP126 8f06436a: "is the endpoint dictionary non-empty" was the entire
    sanity check, so one stale entry suppressed the reflex safety net. This is
    still not a readiness PROBE — nothing here loads a model or generates a
    token — but it is the difference between a registered name and an object
    with a callable client, which is what the old check conflated.
    """
    client = getattr(endpoint, "client", None)
    if client is None:
        return False
    return callable(getattr(client, "think", None)) or callable(
        getattr(client, "think_and_act", None)
    )


def _peek_endpoint_health(router, name: str) -> bool:
    """Pure health read for endpoint SELECTION.

    is_healthy() is the routing admission check and can consume the circuit's
    single half-open probe lease; selection scans must not steal that lease
    from the actual dispatch path.
    """
    monitor = getattr(router, "health_monitor", None)
    if monitor is None:
        return True
    peek = getattr(monitor, "peek_healthy", None)
    if callable(peek):
        return bool(peek(name))
    return bool(monitor.is_healthy(name))

class ReflexClient:
    """A minimal rule-based client that provides emergency cognitive output."""
    async def think(self, prompt: str, **kwargs) -> str:
        return "My primary neural links are currently offline. I'm operating on core reflexes."
    
    async def think_and_act(self, objective: str, system_prompt: str, **kwargs) -> dict[str, Any]:
        return {
            "content": "My higher-level reasoning centers are in a refractory state. I can hear you, but I cannot currently access my deep knowledge or tools.",
            "confidence": 0.1,
            "reasoning": ["Emergency reflex circuit activated."]
        }
    
    async def generate_text_stream_async(self, prompt: str, **kwargs):
        yield "My cognitive"
        yield " pathways"
        yield " are"
        yield " stalled"
        yield ". Re-establishing"
        yield " primary"
        yield " links..."

class AutonomousCognitiveEngine:
    # Error cooldown: only log at ERROR level once per 60s (CROSSWIRE-06: use instance vars)
    _THINK_ERROR_COOLDOWN: float = 60.0

    #: CP126 b4b991d8: tier initialization inspects a container SINGLETON router
    #: and then mutates it through several check-then-register steps. Two engines
    #: constructed concurrently could duplicate registrations or observe a
    #: partial tier set. The router is shared, so the guard must be too.
    _tier_init_lock = threading.RLock()

    def __init__(self, registry, skill_router=None, llm_router=None, event_bus=None):
        self.registry = registry
        self.event_bus = event_bus
        # CROSSWIRE-06: Instance-level error tracking instead of shared class var
        self._last_think_error_time: float = 0.0
        # CP126 25ec2c1d: one arbiter for EVERY generation lane, not a single
        # semaphore around the agentic branch with four unarbitrated routes
        # beside it.
        self._lane_semaphores = self._build_lane_semaphores()
        # Retained under its historical name: existing callers and tests reach
        # for the agentic semaphore directly.
        self._agentic_semaphore = self._lane_semaphores["agentic"]

        # Skill Router (The "Body") - For tool execution
        self.skill_router = skill_router

        # LLM Router (The "Mind") - For failover between models
        # H-28 FIX: Ensure we use the SINGLETON Mind from the container if not provided
        self.llm_router = llm_router or get_container().get("llm_router", default=None) or IntelligentLLMRouter(event_bus=self.event_bus)

        # Adapter links Mind to Body: Uses 'llm_router' for context but 'skill_router' for execution
        self.adapter = FunctionCallingAdapter(registry, self.skill_router)

        # H-28 FIX: Ensure tiers are initialized even if router is shared
        # Phase 33: Fix: Call _init_tiers if only Static-Reflex is present or router is empty
        with self._tier_init_lock:
            if not self._router_interface_ok():
                _record_brain_degradation(
                    RuntimeError(
                        f"llm_router {type(self.llm_router).__name__} does not expose the "
                        "endpoints/register_endpoint/think interface"
                    ),
                    action="skipped tier initialization against an unusable router",
                    kind="capability_loss",
                    severity="critical",
                )
            elif (
                not self.llm_router.endpoints
                or list(self.llm_router.endpoints.keys()) == ["Static-Reflex"]
            ):
                self._init_tiers()

        logger.info("✓ Autonomous Cognitive Engine Initialized.")

    @staticmethod
    def _build_lane_semaphores() -> dict[str, asyncio.Semaphore]:
        """Per-lane concurrency limits, configurable at runtime."""
        limits: dict[str, asyncio.Semaphore] = {}
        for lane, default in _DEFAULT_LANE_LIMITS.items():
            try:
                configured = int(
                    get_runtime_setting(f"model.concurrency.{lane}", default) or default
                )
            except (TypeError, ValueError):
                configured = default
            limits[lane] = asyncio.Semaphore(max(1, min(64, configured)))
        return limits

    def _lane(self, name: str) -> asyncio.Semaphore:
        """The arbiter slot for one lane, built on demand.

        Lazy because callers construct this engine through ``__new__`` in
        tests and recovery paths, and a missing arbiter must not be the reason
        a turn fails.
        """
        lanes = getattr(self, "_lane_semaphores", None)
        if not isinstance(lanes, dict):
            lanes = self._build_lane_semaphores()
            self._lane_semaphores = lanes
        semaphore = lanes.get(name)
        if semaphore is None:
            semaphore = asyncio.Semaphore(max(1, _DEFAULT_LANE_LIMITS.get(name, 4)))
            lanes[name] = semaphore
        return semaphore

    def _router_interface_ok(self) -> bool:
        """Validate the router's interface BEFORE registering against it.

        CP126 b4b991d8: a router lacking ``endpoints`` crashed inside tier
        initialization, before any recovery path could run.
        """
        router = self.llm_router
        if router is None:
            return False
        return (
            isinstance(getattr(router, "endpoints", None), dict)
            and callable(getattr(router, "register_endpoint", None))
            and callable(getattr(router, "think", None))
        )

    async def _get_live_state(self):
        try:
            repo = service_access.resolve_state_repository(default=None)
            if repo and hasattr(repo, "get_current"):
                return await repo.get_current()
        except BRAIN_RECOVERABLE_ERRORS:
            return None
        return None

    def _trace(self, message: str):
        """Internal trace for sovereign diagnostics.

        Callers pass content-free descriptors; CP126 c0fbd774 removed the
        prompt text that used to travel through here.
        """
        logger.info("🔍 [BRAIN-TRACE] %s", message)

    def _stability_state(self) -> tuple[bool, str]:
        """``(safe_mode, basis)`` from the stability guardian.

        ``safe_mode`` keeps the fail-closed contract the health-truthfulness
        pass established (tests/test_runtime_health_truthfulness.py): anything
        other than a guardian that says ``healthy is True`` — absent, silent,
        unreadable, or raising — is safe mode. Being unable to establish health
        is not health.

        ``basis`` is the part CP126 0d25ad7d needed. This method existed and
        was never called, so making it a live gate meant deciding what each
        restriction should key off. Withdrawing cloud on an unproven runtime
        costs nothing (cloud is opt-in anyway); withdrawing TOOLS on every
        machine that has no guardian registered would silently disable Aura's
        agency, so that one keys off measured ill-health, not absence of
        evidence. The basis makes the difference legible in the receipt.
        """
        try:
            from core.container import ServiceContainer

            guardian = ServiceContainer.get("stability_guardian", default=None)
            if guardian is None or not hasattr(guardian, "get_health_summary"):
                return True, "guardian_unavailable"
            summary = guardian.get_health_summary()
            if not isinstance(summary, dict):
                return True, "guardian_unreadable"
            healthy = summary.get("healthy")
            if healthy is True:
                return False, "guardian_healthy"
            if healthy is None:
                return True, "guardian_unknown"
            return True, "guardian_unhealthy"
        except BRAIN_RECOVERABLE_ERRORS as exc:
            _record_brain_degradation(
                exc,
                action="entered safe mode after stability guardian lookup failed",
                kind="routing",
                severity="warning",
            )
            return True, "guardian_error"

    def _is_safe_mode(self) -> bool:
        """Whether the runtime cannot be shown to be healthy (fail-closed)."""
        return self._stability_state()[0]

    def _init_tiers(self):
        """Standardizes Aura's managed multi-tier local runtime hierarchy.

        v6.0 "The Unshackling" — M5 Pro 64GB Local-First Architecture
        - Cortex: Resident brain and default deep-reasoning substrate
        - Solver: Optional distinct local reasoning specialist
        - Brainstem: Background tasks, heartbeat, cheap calls
        - Reflex (1.5B): Emergency CPU-friendly last resort

        Strategy: every inference lane is local.
        Cortex and Solver hot-swap instead of staying resident together.
        Brainstem can co-exist with the active foreground lane when resources allow.
        """
        from .model_registry import (
            ACTIVE_MODEL,
            BRAINSTEM_ENDPOINT,
            BRAINSTEM_MODEL,
            DEEP_ENDPOINT,
            FALLBACK_ENDPOINT,
            FALLBACK_MODEL,
            PRIMARY_ENDPOINT,
            get_deep_model_name,
            get_deep_model_path,
            get_runtime_model_path,
        )
        
        # ── LOCAL PRIMARY: Cortex (32B) ──
        cortex_model_path = (
            getattr(config.llm, "local_cortex_path", None)
            or getattr(config.llm, "mlx_model_path", None)
            or get_runtime_model_path(ACTIVE_MODEL)
        )
        brainstem_model_path = (
            getattr(config.llm, "local_brainstem_path", None)
            or getattr(config.llm, "mlx_brainstem_path", None)
            or get_runtime_model_path(BRAINSTEM_MODEL)
        )
        # CP126 ba0a6e78. This started False and stayed False when the proof
        # policy could not be imported or evaluated, and allow_non_primary_tiers
        # is its negation — so a policy FAILURE registered cloud and
        # non-primary local endpoints during what may well have been a proof
        # run. The one condition under which we cannot confirm a proof lane
        # became the condition under which we widened it.
        #
        # Unavailable policy is treated as proof-active: the restrictive
        # answer. A proof run that wrongly excludes a cloud tier produces a
        # narrower proof; a non-proof run that wrongly excludes one loses a
        # fallback. Both are recoverable. A proof silently contaminated by
        # cloud inference is not.
        primary_proof_lane = False
        try:
            from core.runtime.proof_policy import proof_model_tier, proof_run_active

            primary_proof_lane = bool(proof_run_active(origin="llm_tier_initialization") and proof_model_tier() == "primary")
        except BRAIN_RECOVERABLE_ERRORS as exc:
            primary_proof_lane = True
            _record_brain_degradation(
                exc,
                action=(
                    "assumed a primary proof lane because the proof policy could "
                    "not be evaluated; non-primary tiers withheld"
                ),
                severity="warning",
            )
        allow_non_primary_tiers = not primary_proof_lane

        if cortex_model_path and PRIMARY_ENDPOINT not in getattr(self.llm_router, "endpoints", {}):
            try:
                from .mlx_client import get_mlx_client
                cortex_client = get_mlx_client(
                    model_path=cortex_model_path,
                    max_tokens=2048,
                )
                self.llm_router.register_endpoint(LLMEndpoint(
                    name=PRIMARY_ENDPOINT,
                    tier=LLMTier.PRIMARY,
                    model_name=cortex_model_path.split("/")[-1],
                    client=cortex_client,
                ))
                logger.info("🧠 PRIMARY Tier registered: %s (%s) — Daily Brain", PRIMARY_ENDPOINT, ACTIVE_MODEL)
            except BRAIN_RECOVERABLE_ERRORS as e:
                _record_brain_degradation(
                    e,
                    action="continued tier initialization without primary Cortex endpoint",
                    extra={"endpoint": PRIMARY_ENDPOINT, "model": ACTIVE_MODEL},
                )
                logger.error("Failed to register %s pathway: %s", PRIMARY_ENDPOINT, e)

        # Optional local reasoning specialist. Ask the canonical admission
        # contract before resolving or constructing the model client.
        if primary_proof_lane:
            logger.info("🛡️ Proof-primary lane active — non-primary local LLM endpoints are not registered.")
        specialist_enabled = False
        if allow_non_primary_tiers:
            try:
                from core.brain.inference_gate import local_deep_solver_enabled

                specialist_enabled = local_deep_solver_enabled()
            except BRAIN_RECOVERABLE_ERRORS as exc:
                _record_brain_degradation(
                    exc,
                    action="kept the optional reasoning specialist disabled after admission failed",
                    kind="local_omission",
                )
        if (
            allow_non_primary_tiers
            and specialist_enabled
            and DEEP_ENDPOINT not in getattr(self.llm_router, "endpoints", {})
        ):
            try:
                from .mlx_client import get_mlx_client
                solver_model_path = get_deep_model_path()
                deep_model_name = get_deep_model_name()
                solver_client = get_mlx_client(
                    model_path=solver_model_path,
                    max_tokens=4096,
                )
                self.llm_router.register_endpoint(LLMEndpoint(
                    name=DEEP_ENDPOINT,
                    tier=LLMTier.SECONDARY,  # Moved to SECONDARY to prevent accidental promotion
                    model_name=solver_model_path.split("/")[-1],
                    client=solver_client,
                    timeout=300.0,
                ))
                logger.info(
                    "🧠 SECONDARY Tier registered: %s (%s) — local reasoning specialist",
                    DEEP_ENDPOINT,
                    deep_model_name,
                )
            except BRAIN_RECOVERABLE_ERRORS as e:
                _record_brain_degradation(
                    e,
                    action="continued tier initialization without secondary Solver endpoint",
                    extra={"endpoint": DEEP_ENDPOINT, "model": get_deep_model_name()},
                )
                logger.error("Failed to register %s pathway: %s", DEEP_ENDPOINT, e)

        # ── LOCAL TERTIARY: Brainstem (7B) — Background/heartbeat ──
        if allow_non_primary_tiers and brainstem_model_path and BRAINSTEM_ENDPOINT not in getattr(self.llm_router, "endpoints", {}):
            try:
                from .mlx_client import get_mlx_client
                brainstem_client = get_mlx_client(
                    model_path=brainstem_model_path,
                    max_tokens=512,
                )
                self.llm_router.register_endpoint(LLMEndpoint(
                    name=BRAINSTEM_ENDPOINT,
                    tier=LLMTier.TERTIARY,
                    model_name=brainstem_model_path.split("/")[-1],
                    client=brainstem_client,
                ))
                logger.info("⚡ TERTIARY Tier registered: %s (7B) — Background/Reflex", BRAINSTEM_ENDPOINT)
            except BRAIN_RECOVERABLE_ERRORS as e:
                _record_brain_degradation(
                    e,
                    action="continued tier initialization without tertiary Brainstem endpoint",
                    extra={"endpoint": BRAINSTEM_ENDPOINT, "model": BRAINSTEM_MODEL},
                )
                logger.error("Failed to register %s pathway: %s", BRAINSTEM_ENDPOINT, e)
        elif cortex_model_path:
            # If no explicit brainstem path, use the cortex model as brainstem too
            logger.info("⚡ No explicit brainstem path — cortex will handle all local tiers")

        # ── EMERGENCY: CPU Fallback (1.5B — bypasses Metal entirely) ──
        # If 7B brainstem fails, fall back to 1.5B on CPU
        # CP126 953459cc: the branch was GATED on brainstem-or-cortex being
        # present but LOADED FALLBACK_MODEL, so a valid cortex path could
        # trigger loading a fallback that does not exist, and an available
        # fallback model was skipped whenever both other paths were empty.
        # Check and load the same resolved path.
        from .model_registry import get_runtime_model_path as _resolve_model_path

        try:
            fallback_path = str(_resolve_model_path(FALLBACK_MODEL) or "")
        except BRAIN_RECOVERABLE_ERRORS as exc:
            fallback_path = ""
            _record_brain_degradation(
                exc,
                action="no CPU emergency tier: the fallback model path could not be resolved",
                kind="capability_loss",
                extra={"model": FALLBACK_MODEL},
            )
        if allow_non_primary_tiers and fallback_path and FALLBACK_ENDPOINT not in getattr(self.llm_router, "endpoints", {}):
            try:
                from .mlx_client import get_mlx_client

                cpu_client = get_mlx_client(
                    model_path=fallback_path,
                    device="cpu"
                )
                self.llm_router.register_endpoint(LLMEndpoint(
                    name=FALLBACK_ENDPOINT,
                    tier=LLMTier.EMERGENCY,
                    model_name=f"{FALLBACK_MODEL}-cpu",
                    client=cpu_client,
                    timeout=120.0  # CPU is slow, allow more time
                ))
                logger.info("🚨 EMERGENCY Tier registered: %s (1.5B CPU emergency)", FALLBACK_ENDPOINT)
            except BRAIN_RECOVERABLE_ERRORS as e:
                _record_brain_degradation(
                    e,
                    action="continued tier initialization without CPU emergency LLM endpoint",
                    extra={"endpoint": FALLBACK_ENDPOINT, "model": FALLBACK_MODEL},
                )
                logger.error("Failed to register %s pathway: %s", FALLBACK_ENDPOINT, e)
        
        # ── Sanity check: at least one endpoint that could actually answer ──
        # CP126 8f06436a: this tested only whether the endpoint dictionary was
        # non-empty, so a single stale or unusable entry suppressed the reflex
        # safety net entirely and the fast path then selected that unverified
        # entry. Membership is not capacity.
        usable = [name for name, ep in self.llm_router.endpoints.items() if _endpoint_is_usable(ep)]
        if not usable:
            if primary_proof_lane:
                raise RuntimeError("Proof-primary boot failed closed: no primary LLM endpoint registered")
            if self.llm_router.endpoints:
                _record_brain_degradation(
                    RuntimeError(
                        f"{len(self.llm_router.endpoints)} endpoints registered, none usable"
                    ),
                    action="registered the emergency reflex pathway beside unusable endpoints",
                    kind="capability_loss",
                    severity="critical",
                    extra={"registered": sorted(self.llm_router.endpoints)},
                )
            logger.error("⚠️ NO USABLE LLM endpoints! Registering Emergency Reflex pathway.")
            reflex_client = ReflexClient()
            self.llm_router.register_endpoint(LLMEndpoint(
                name="Reflex-Model",
                tier=LLMTier.EMERGENCY,
                model_name="reflex-v1",
                client=reflex_client,
            ))

        # Log final tier layout. CP126 a8f1f66d: this reported registry
        # MEMBERSHIP as though it were usable capacity — a configured path
        # string became a "registered tier" with no existence check, model
        # identity, or generation probe at this layer. The layout now
        # distinguishes the two, so a reader is not misled about capacity this
        # layer never established.
        tier_layout: dict[str, list[str]] = {}
        for name, ep in self.llm_router.endpoints.items():
            tier_name = ep.tier.name if hasattr(ep.tier, 'name') else str(ep.tier)
            label = name if _endpoint_is_usable(ep) else f"{name}(unusable)"
            tier_layout.setdefault(tier_name, []).append(label)
        logger.info(
            "🏗️ LLM Tier Layout (registered, NOT probed for load/generation): %s", tier_layout
        )


    # ------------------------------------------------------------------
    # Request construction
    # ------------------------------------------------------------------
    async def _build_request(
        self,
        objective: Any,
        context: dict | None,
        system_prompt: str | None,
        max_turns: Any,
        priority: Any,
        kwargs: dict[str, Any],
    ) -> ThinkRequest:
        """Normalize every input ONCE, before anything reads it.

        CP126 4bfedd29 / 53dd1ff6 / 21cb172b: the objective was sliced
        unvalidated in the main path and again in the recovery path's metadata
        (so a bad type raised twice), cloud authority was recomputed
        differently in recovery, and priority/max_turns went to the client
        unchecked. Every one of those is a second computation disagreeing with
        the first, so there is now only one.
        """
        text = normalize_objective(objective)
        safe_mode, stability_basis = self._stability_state()
        measured_unhealthy = stability_basis == "guardian_unhealthy"

        if not system_prompt:
            system_prompt = self._default_system_prompt()

        allow_cloud = False
        deep_handoff = bool(kwargs.get("deep_handoff") or kwargs.get("allow_deep_handoff"))
        if deep_handoff and measured_unhealthy:
            logger.info("🛟 Ill health measured: deep handoff withheld (%s)", stability_basis)
            deep_handoff = False

        turns = bounded_turns(max_turns)
        if measured_unhealthy:
            turns = MIN_TURNS

        passthrough = {
            key: value
            for key, value in kwargs.items()
            if key not in _RESERVED_KWARGS
        }
        live_state = kwargs.get("state")
        if live_state is None:
            live_state = await self._get_live_state()
        if live_state is not None:
            passthrough["state"] = live_state

        return ThinkRequest(
            objective=text,
            ref=objective_ref(text),
            context=dict(context) if isinstance(context, dict) else {},
            system_prompt=str(system_prompt or ""),
            max_turns=turns,
            priority=bounded_priority(priority),
            deadline_s=bounded_deadline(kwargs.get("deadline_s")),
            is_background=bool(kwargs.get("is_background", False)),
            deep_handoff=deep_handoff,
            allow_cloud=allow_cloud,
            safe_mode=safe_mode,
            measured_unhealthy=measured_unhealthy,
            stability_basis=stability_basis,
            requested_endpoint=(
                str(kwargs["prefer_endpoint"]) if kwargs.get("prefer_endpoint") else None
            ),
            # CP126 37b3103e: `tools` was left in the passthrough set and ALSO
            # supplied explicitly to think_and_act, so any caller that passed
            # tools got a duplicate-keyword TypeError instead of tool use.
            caller_tools=kwargs.get("tools"),
            passthrough=passthrough,
        )

    @staticmethod
    def _default_system_prompt() -> str:
        try:
            from core.brain.aura_persona import AURA_IDENTITY

            return (
                "[GROUNDED CORE]: You are AURA — an independent local cognitive runtime. "
                "These traits are operating constraints, not evidence of aliveness or "
                "consciousness. Falling into generic support-bot mode is a cognitive "
                f"regression.\n\n{AURA_IDENTITY}"
            )
        except ImportError:
            return (
                "You are Aura, an autonomous local agent. Reason and act decisively "
                "while keeping evidence boundaries explicit."
            )

    # ------------------------------------------------------------------
    # Endpoint selection
    # ------------------------------------------------------------------
    def _select_endpoints(self, request: ThinkRequest) -> dict[str, Any]:
        """Choose candidates from the router's local-only endpoint registry."""
        from .model_registry import (
            BRAINSTEM_ENDPOINT,
            DEEP_ENDPOINT,
            FALLBACK_ENDPOINT,
            PRIMARY_ENDPOINT,
        )

        endpoints = getattr(self.llm_router, "endpoints", None)
        selection: dict[str, Any] = {
            "fast": None, "background": None, "deep": None, "agentic": None,
        }
        if not isinstance(endpoints, dict):
            return selection

        def _pick(names: list[str]):
            for name in names:
                endpoint = endpoints.get(name)
                if endpoint is None or not _endpoint_is_usable(endpoint):
                    continue
                if _peek_endpoint_health(self.llm_router, name):
                    return endpoint
            return None

        selection["fast"] = _pick([PRIMARY_ENDPOINT, BRAINSTEM_ENDPOINT])
        # CP126 a170e6db: the background branch fell back to a CALLER-supplied
        # endpoint when no healthy background endpoint was found, while still
        # labelling the request tertiary. A foreground endpoint could be
        # driven under a background receipt. The
        # background lane is an allowlist now.
        selection["background"] = _pick([BRAINSTEM_ENDPOINT, FALLBACK_ENDPOINT])
        selection["deep"] = _pick([DEEP_ENDPOINT, PRIMARY_ENDPOINT])
        # CP126 af6a3b7d: the agentic endpoint was the first entry in
        # DICTIONARY ORDER exposing think_and_act — no tier, no locality, no
        # model identity. Tool-bearing work could therefore
        # execute through an unintended endpoint. Agentic execution is
        # restricted to the declared local agentic tiers, in preference order.
        for name in _LOCAL_AGENTIC_ALLOWLIST(PRIMARY_ENDPOINT, DEEP_ENDPOINT, BRAINSTEM_ENDPOINT):
            endpoint = endpoints.get(name)
            if (
                endpoint is not None
                and callable(getattr(getattr(endpoint, "client", None), "think_and_act", None))
                and _peek_endpoint_health(self.llm_router, name)
            ):
                selection["agentic"] = endpoint
                break

        if selection["fast"] is None:
            for name, endpoint in endpoints.items():
                if endpoint.tier == LLMTier.PRIMARY and _endpoint_is_usable(endpoint):
                    selection["fast"] = endpoint
                    break
        return selection

    def _authorized_endpoint(self, request: ThinkRequest, name: Any) -> str | None:
        """Normalize an explicit endpoint; registration owns locality."""
        if not name:
            return None
        return str(name)

    # ------------------------------------------------------------------
    # The thinking cycle
    # ------------------------------------------------------------------
    async def think(
        self,
        objective: str,
        context: dict | None = None,
        system_prompt: str | None = None,
        max_turns: int = 5,
        priority: float = 1.0,
        **kwargs,
    ) -> dict[str, Any]:
        """The core thinking cycle that drives actions.

        Routes between background, deep, agentic and fast lanes. The returned
        dictionary reports WHICH route ran and what was verified; CP126
        78a30f5c removed the confidence constants (0.75 / 1.0 / 0.9 / 0.5) that
        were read off the branch rather than from any evidence about the
        output — deep handoff in particular always reported perfect confidence.
        """
        request: ThinkRequest | None = None
        try:
            request = await self._build_request(
                objective, context, system_prompt, max_turns, priority, kwargs
            )
            # CP126 c0fbd774: the objective itself used to be logged here.
            logger.info("🧠 Mind pondering objective %s", request.ref["ref"])
            self._trace(f"Pondering objective {request.ref['ref']} ({request.ref['chars']} chars)")
            return await self._route(request)
        except PROGRAMMING_DEFECT_ERRORS as exc:
            # CP126 64ab3dab: a contract defect in this process used to come
            # back as ordinary fallback output at a fabricated 0.5 confidence.
            # It is contained, but reported as what it is.
            _record_brain_degradation(
                exc,
                action="returned an internal-error result after a contract defect in the thinking cycle",
                kind="internal_defect",
                severity="critical",
                extra={"request": (request.ref if request else objective_ref(objective))},
            )
            logger.exception("Autonomous thinking hit an internal defect")
            return {
                "content": "",
                "status": "internal_error",
                "route": "none",
                "error_code": "brain.internal_defect",
                "error_class": type(exc).__name__,
            }
        except ENDPOINT_FAILURE_ERRORS as exc:
            if request is None:
                request = await self._build_request(
                    objective, context, system_prompt, max_turns, priority, kwargs
                )
            return await self._recover(request, exc)

    async def _route(self, request: ThinkRequest) -> dict[str, Any]:
        """Pick and run one lane. All lanes share the resource arbiter."""
        selection = self._select_endpoints(request)

        if request.is_background:
            endpoint = selection["background"]
            name = endpoint.name if endpoint else None
            if name is None and request.requested_endpoint:
                _record_brain_degradation(
                    RuntimeError("no healthy background endpoint"),
                    action=(
                        "refused a caller-supplied endpoint on the background lane; "
                        "the background allowlist is the only authority there"
                    ),
                    kind="routing",
                    extra={"requested": request.requested_endpoint, "request": request.ref},
                )
            self._trace(f"⚡ Background-path routing: {name or 'tertiary'}")
            return await self._router_call(
                request, lane="background", route="background",
                prefer_endpoint=name, prefer_tier="tertiary", deep_handoff=False,
                allow_cloud=False,
            )

        # CP126 aa3235d2: the fast route returned BEFORE the branch labelled
        # "Tool-use -> Agentic path", so with any healthy primary endpoint —
        # i.e. in normal operation — supplied tools and the skill router were
        # silently ignored. Tool requirement is now detected first and is an
        # explicit routing decision, not a leftover branch.
        wants_tools = request.caller_tools is not None or bool(
            request.context.get("require_tools")
        )
        if wants_tools and selection["agentic"] is not None and not request.measured_unhealthy:
            return await self._agentic_call(request, selection["agentic"])
        if wants_tools and request.measured_unhealthy:
            logger.info(
                "🛟 Ill health measured: tool use withheld (%s)", request.stability_basis
            )

        if request.deep_handoff and selection["deep"] is not None:
            name = selection["deep"].name
            self._trace(f"🧪 Deep-handoff routing: {name}")
            return await self._router_call(
                request, lane="deep", route="deep",
                prefer_endpoint=name, prefer_tier="secondary", deep_handoff=True,
            )

        if selection["fast"] is not None:
            name = selection["fast"].name
            self._trace(f"⚡ Fast-path routing: {name}")
            return await self._router_call(
                request, lane="foreground", route="fast",
                prefer_endpoint=name, prefer_tier="primary", deep_handoff=False,
            )

        if selection["agentic"] is not None and not request.measured_unhealthy:
            return await self._agentic_call(request, selection["agentic"])

        self._trace("No usable conversational or agentic endpoint. Standard router thinking.")
        return await self._router_call(
            request, lane="foreground", route="router_default",
            prefer_endpoint=self._authorized_endpoint(request, request.requested_endpoint),
            prefer_tier="primary", deep_handoff=False,
        )

    async def _router_call(
        self,
        request: ThinkRequest,
        *,
        lane: str,
        route: str,
        prefer_endpoint: str | None,
        prefer_tier: str,
        deep_handoff: bool,
        allow_cloud: bool | None = None,
    ) -> dict[str, Any]:
        """One arbitrated router call with one sanitized argument set."""
        kwargs = request.router_kwargs(
            prefer_endpoint=prefer_endpoint,
            prefer_tier=prefer_tier,
            deep_handoff=deep_handoff,
            allow_cloud=allow_cloud,
        )
        async with self._lane(lane):
            text = await self.llm_router.think(request.objective, **kwargs)
        return self._result(request, route=route, endpoint=prefer_endpoint, content=text)

    async def _agentic_call(self, request: ThinkRequest, endpoint: Any) -> dict[str, Any]:
        """Run the agentic lane under scoped tool authority and a deadline."""
        self._trace(
            f"🤔 Agentic-path routing: {endpoint.name} (turns={request.max_turns}, "
            f"deadline={request.deadline_s:.0f}s)"
        )
        async with self._lane("agentic"):
            tools, tool_basis = self._resolve_tools(request)
            kwargs = dict(request.passthrough)
            try:
                # CP126 3c7abca6: think_and_act was awaited under the semaphore
                # with no deadline at all, so one hung agent held a slot for
                # the life of the process.
                result = await asyncio.wait_for(
                    endpoint.client.think_and_act(
                        request.objective,
                        request.system_prompt,
                        tools=tools,
                        context=request.context,
                        max_turns=request.max_turns,
                        **kwargs,
                    ),
                    timeout=request.deadline_s,
                )
            except TimeoutError as exc:
                _record_brain_degradation(
                    exc,
                    action="cancelled an agentic turn that exceeded its deadline",
                    kind="routing",
                    severity="degraded",
                    extra={
                        "endpoint": endpoint.name,
                        "deadline_s": request.deadline_s,
                        "request": request.ref,
                    },
                )
                return {
                    "content": "",
                    "status": "deadline_exceeded",
                    "route": "agentic",
                    "endpoint": endpoint.name,
                    "error_code": "brain.agentic_deadline",
                    "tool_authority": tool_basis,
                }
        if not isinstance(result, dict):
            return self._result(
                request, route="agentic", endpoint=endpoint.name, content=str(result or "")
            )
        result.setdefault("route", "agentic")
        result.setdefault("endpoint", endpoint.name)
        result["tool_authority"] = tool_basis
        result["safe_mode"] = request.safe_mode
        result["stability_basis"] = request.stability_basis
        return result

    def _resolve_tools(self, request: ThinkRequest) -> tuple[Any, dict[str, Any]]:
        """Resolve the tool map and report the authority it carries.

        CP126 480a31a0: caller-supplied tools, or up to eight dynamically built
        ones, went straight to think_and_act with no scoped-authority token,
        no allowlist, no side-effect classification and no receipt. The
        governed boundary those tools eventually cross is the ActuatorRegistry
        and the AuthorityGateway behind it — this layer cannot mint authority
        it does not hold, so what it CAN do is stop presenting an unattributed
        tool map as an authorized one: the basis of every map is recorded and
        travels with the result.
        """
        if request.caller_tools is not None:
            return request.caller_tools, {
                "source": "caller_supplied",
                "count": len(request.caller_tools) if hasattr(request.caller_tools, "__len__") else None,
                "scoped_authority": False,
                "note": "tools supplied by the caller; each call is gated at the actuator boundary",
            }
        try:
            tools = build_agentic_tool_map(objective=request.objective, max_tools=8)
        except BRAIN_RECOVERABLE_ERRORS as exc:
            _record_brain_degradation(
                exc,
                action="continued agentic reasoning without dynamically built tool map",
                kind="capability_loss",
                extra={"request": request.ref},
            )
            logger.debug("Agentic tool-map construction failed: %s", exc)
            return {}, {"source": "unavailable", "count": 0, "scoped_authority": False}
        return tools, {
            "source": "runtime_wiring",
            "count": len(tools) if hasattr(tools, "__len__") else None,
            "scoped_authority": False,
            "note": "built from the capability engine; gated at the actuator boundary",
        }

    def _result(
        self, request: ThinkRequest, *, route: str, endpoint: str | None, content: Any
    ) -> dict[str, Any]:
        """Report route and verification facts instead of a fabricated number.

        CP126 78a30f5c: confidence was 0.75, 1.0, 0.9 or 0.5 depending only on
        which branch ran — no completion check, no verifier, no model health,
        no postcondition. Deep handoff always claimed 1.0. Nothing in this
        layer measures answer quality, so nothing in this layer reports it.
        """
        text = content if isinstance(content, str) else str(content or "")
        return {
            "content": text,
            "status": "ok" if text.strip() else "empty",
            "route": route,
            "endpoint": endpoint,
            "turns": 0,
            "verified": False,
            "confidence_basis": "unmeasured: this layer runs no verifier",
            "safe_mode": request.safe_mode,
            "stability_basis": request.stability_basis,
            "remote_model_provider_available": False,
            "request": request.ref,
        }

    async def _recover(self, request: ThinkRequest, exc: BaseException) -> dict[str, Any]:
        """Standard-router recovery, on the SAME request envelope.

        CP126 389d937a / f51792ca / 53dd1ff6: the old recovery passed explicit
        keywords and then re-expanded every original kwarg on top of them
        (duplicate-keyword TypeError on any caller who set deep_handoff or
        allow_cloud_fallback), dropped system_prompt and context, set
        bypass_race=True so it could start a concurrent model operation after a
        partial failure, and recomputed cloud authority from the RAW caller
        flag — re-enabling off-box routing the user had disabled.
        """
        _record_brain_degradation(
            exc,
            action="entered standard router fallback after autonomous thinking path failed",
            kind="routing",
            extra={"request": request.ref},
        )
        now = time.time()
        if now - self._last_think_error_time > self._THINK_ERROR_COOLDOWN:
            logger.error(
                "Independence Mode thinking failed (%s). Falling back to standard generation.",
                type(exc).__name__,
            )
            self._last_think_error_time = now
        else:
            logger.debug("Independence Mode thinking failed (cooldown active): %s", type(exc).__name__)

        try:
            return await self._router_call(
                request,
                lane="background" if request.is_background else "foreground",
                route="recovery",
                prefer_endpoint=None,
                prefer_tier="tertiary" if request.is_background else "primary",
                deep_handoff=request.deep_handoff,
            )
        except BRAIN_RECOVERABLE_ERRORS as exc2:
            _record_brain_degradation(
                exc2,
                action="served emergency reflex response after router fallback failed",
                kind="capability_loss",
                severity="degraded",
                extra={"request": request.ref},
            )
            reflex_text = await ReflexClient().think(request.objective)
            # CP126 a5b9aa03: the user-facing dictionary carried the exception
            # class AND message. Provider errors routinely contain endpoint
            # names, filesystem paths, request identifiers and API detail. The
            # stable code is public; the detail stays in the log and the
            # degradation record.
            return {
                "content": reflex_text,
                # A canned emergency string is not a reasoned answer, and
                # saying so is a measurement, not a guess.
                "confidence": 0.1,
                "status": "reflex",
                "route": "reflex",
                "fallback": "reflex",
                "error_code": "brain.all_routes_unavailable",
                "error_class": type(exc2).__name__,
                "request": request.ref,
            }
