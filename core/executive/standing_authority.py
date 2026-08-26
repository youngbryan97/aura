"""Durable standing authority and bounded child leases for Aura's agency.

Standing grants describe what Aura may do without asking for approval on every
action.  A grant never flows directly to an effect sink.  Each invocation gets
a short-lived, argument-bound, process-local child token that must survive Will,
constitutional, capability, and effect verification before execution.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.agency.capability_token import CapabilityTokenStore, get_token_store
from core.executive.bounded_sandbox_policy import (
    validate_idle_sandbox_probe_arguments,
)
from core.executive.execution_policy import (
    classify_execution_risk,
    normalize_risk,
    normalize_tool_name,
    resolve_execution_effect_scope,
    risk_at_most,
    scope_is_within,
)
from core.runtime.errors import record_degradation
from core.runtime.receipts import AutonomyReceipt, get_receipt_store
from core.runtime.state_ownership import state_root, state_root_override

logger = logging.getLogger("Aura.StandingAuthority")

STATE_DOMAIN = "standing_authority"
STATE_KEY = "registry"
STATE_SCHEMA_VERSION = 1
_GRANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{2,95}$")

USER_FACING_AUTHORITY_ORIGINS = frozenset(
    {
        "admin",
        "api",
        "api_skill_execute",
        "chat",
        "chat_api",
        "desktop",
        "desktop_task",
        "desktop_ui",
        "direct",
        "external",
        "frontend",
        "gui",
        "interface",
        "live_chat",
        "live_skill_api",
        "messages",
        "native_shell",
        "tauri",
        "ui",
        "user",
        "voice",
        "voice_bridge",
        "voice_input",
        "web_ui",
        "websocket",
        "ws",
    }
)

AUTONOMOUS_AUTHORITY_ORIGINS = frozenset(
    {
        "autonomous_initiative_loop",
        "autonomous_initiative",
        "autonomous_task_engine",
        "autonomy",
        "background",
        "background_reflection",
        "behavior_controller",
        "cognitive_coordinator",
        "curiosity",
        "curiosity_daemon",
        "curiosity_engine",
        "curiosity_explorer",
        "dream",
        "intention_loop",
        "latent_cortex",
        "overt_action_loop",
        "proactive_presence",
        "proactive_comm",
        "react_loop",
        "research_cycle",
        "subconscious_loop",
        "temporal",
        "autonomy_reflection",
    }
)

PUBLIC_RESEARCH_TOOLS = frozenset(
    {
        "curiosity_web_search",
        "free_search",
        "grounded_search",
        "search_web",
        "sensory_motor_browser_research",
        "web_search",
    }
)

INTROSPECTION_TOOLS = frozenset(
    {
        "clock",
        "environment_info",
        "evolution_status",
        "get_time",
        "query_beliefs",
        "self_diagnosis",
        "status",
        "system_health",
        "system_proprioception",
    }
)

LOCAL_OBSERVATION_TOOLS = frozenset(
    {
        "grep_search",
        "list_dir",
        "read_file",
        "view_file",
    }
)

CONNECTED_ACCOUNT_READ_TOOLS = frozenset({"email_adapter", "reddit_adapter"})
READ_ONLY_MAINTENANCE_TOOLS = frozenset({"auto_refactor", "self_evolution"})
BOUNDED_MAINTENANCE_TEST_TOOLS = frozenset({"test_generator"})
BOUNDED_SANDBOX_TOOLS = frozenset({"subconscious_sandbox_probe"})
BACKGROUND_REFLECTION_TOOLS = frozenset({"swarm_debate"})
OWNER_PRIVATE_MESSAGE_TOOLS = frozenset({"messages"})
BACKGROUND_REFLECTION_ROLES = frozenset({"architect", "critic", "philosopher"})
BACKGROUND_REFLECTION_MAX_ACTIONS = 6
BACKGROUND_REFLECTION_WINDOW_SECONDS = 3600.0

_UNSAFE_PUBLIC_RESEARCH_MARKERS = frozenset(
    {
        "api key",
        "brute force",
        "bypass login",
        "credential dump",
        "ddos",
        "deanonymize",
        "doxx",
        "exfiltrate",
        "malware payload",
        "password dump",
        "phishing kit",
        "private key",
        "ransomware",
        "session cookie",
        "steal credential",
        "token dump",
        "worm payload",
    }
)

_SUPPORTED_ARGUMENT_POLICIES = frozenset(
    {
        "any",
        "aura_local_read",
        "bounded_sandbox_probe",
        "bounded_background_reflection",
        "bounded_maintenance_test",
        "connected_account_read",
        "foreground_user_request",
        "local_introspection",
        "owner_private_message",
        "public_research",
        "read_only_maintenance",
    }
)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def normalize_authority_origin(value: Any) -> str:
    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value or "").strip().lower(),
    ).strip("_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def coerce_authority_origin(value: Any) -> str:
    normalized = normalize_authority_origin(value)
    if not normalized:
        return "unknown"
    if normalized in USER_FACING_AUTHORITY_ORIGINS | AUTONOMOUS_AUTHORITY_ORIGINS:
        return normalized
    # A sub-route inherits the standing of the route it belongs to.
    #
    # This fell back to a hardcoded nine-token list — user, api, voice, admin,
    # gui, websocket, ws, direct, external — which omits `desktop`, `chat`,
    # `ui`, `frontend` and most of the set it was standing in for. So any
    # labelled sub-route lost its authority silently: `desktop_task.web_search`
    # coerced to itself, matched no grant, and `context_has_user_authority`
    # returned False for a route the desktop lane uses on ordinary foreground
    # turns. Measured 2026-08-19, along with `sovereign_browser.pursue`
    # refused for the same reason.
    #
    # Longest prefix first, so specificity is preserved: `desktop_task` is
    # preferred over `desktop`, and a label that matches nothing still falls
    # through to the token scan below rather than being invented.
    known = USER_FACING_AUTHORITY_ORIGINS | AUTONOMOUS_AUTHORITY_ORIGINS
    parts = [token for token in normalized.split("_") if token]
    for length in range(len(parts) - 1, 0, -1):
        prefix = "_".join(parts[:length])
        if prefix in known:
            return prefix
    # The token scan below stays exactly as narrow as it was. Widening it to
    # the whole known set remapped unrelated labels — `test_autonomy` became
    # `autonomy` and stopped matching its own grant — which is inheritance
    # turning into reassignment. Prefix inheritance above is the specific
    # thing that was missing; this is not.
    tokens = set(parts)
    for candidate in (
        "user",
        "api",
        "voice",
        "admin",
        "gui",
        "websocket",
        "ws",
        "direct",
        "external",
    ):
        if candidate in tokens:
            return candidate
    return normalized


def context_has_user_authority(origin: Any, context: Mapping[str, Any] | None = None) -> bool:
    ctx = dict(context or {})
    normalized = coerce_authority_origin(origin)
    if normalized not in USER_FACING_AUTHORITY_ORIGINS:
        return False
    explicit_keys = (
        "foreground_request",
        "user_explicit_action_request",
        "user_explicitly_authorized",
        "user_requested_action",
    )
    if any(key in ctx for key in explicit_keys):
        return any(_truthy(ctx.get(key)) for key in explicit_keys)
    # The origin is assigned by Aura's authenticated foreground route, not read
    # from tool arguments.  Legacy foreground callers therefore remain usable.
    return True


def canonical_arguments_digest(arguments: Mapping[str, Any] | None) -> str:
    def _canonical(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, bytes):
            return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {
                str(key): _canonical(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [_canonical(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted((_canonical(item) for item in value), key=lambda item: repr(item))
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": repr(value)[:512],
        }

    encoded = json.dumps(
        _canonical(dict(arguments or {})),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class StandingAuthorityGrant:
    grant_id: str
    issuer: str
    description: str
    allowed_origins: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    allowed_effect_scopes: tuple[str, ...]
    max_risk: str
    max_actions: int
    window_seconds: float
    lease_ttl_seconds: float
    argument_policy: str
    subject: str = "aura"
    enabled: bool = True
    built_in: bool = False

    def __post_init__(self) -> None:
        if not _GRANT_ID_RE.fullmatch(self.grant_id):
            raise ValueError(f"invalid standing-authority grant id: {self.grant_id!r}")
        if not str(self.issuer or "").strip():
            raise ValueError("standing-authority issuer is required")
        if not self.allowed_origins or not self.allowed_tools or not self.allowed_effect_scopes:
            raise ValueError("standing-authority origins, tools, and effect scopes are required")
        if normalize_risk(self.max_risk, default="") not in {"low", "medium", "high", "critical"}:
            raise ValueError(f"invalid standing-authority max risk: {self.max_risk!r}")
        if int(self.max_actions) < 0:
            raise ValueError("standing-authority max_actions cannot be negative")
        if self.max_actions and float(self.window_seconds) <= 0:
            raise ValueError("budgeted standing authority requires a positive window")
        if not 1.0 <= float(self.lease_ttl_seconds) <= 900.0:
            raise ValueError("standing-authority lease TTL must be between 1 and 900 seconds")
        if self.argument_policy not in _SUPPORTED_ARGUMENT_POLICIES:
            raise ValueError(f"unsupported standing-authority argument policy: {self.argument_policy}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StandingAuthorityGrant:
        data = dict(payload or {})
        return cls(
            grant_id=str(data.get("grant_id") or ""),
            issuer=str(data.get("issuer") or ""),
            description=str(data.get("description") or ""),
            allowed_origins=tuple(str(item) for item in data.get("allowed_origins") or ()),
            allowed_tools=tuple(str(item) for item in data.get("allowed_tools") or ()),
            allowed_effect_scopes=tuple(
                str(item) for item in data.get("allowed_effect_scopes") or ()
            ),
            max_risk=str(data.get("max_risk") or ""),
            max_actions=int(data.get("max_actions") or 0),
            window_seconds=float(data.get("window_seconds") or 0.0),
            lease_ttl_seconds=float(data.get("lease_ttl_seconds") or 0.0),
            argument_policy=str(data.get("argument_policy") or ""),
            subject=str(data.get("subject") or "aura"),
            enabled=bool(data.get("enabled", True)),
            built_in=bool(data.get("built_in", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StandingAuthorityLeaseRecord:
    token: str
    grant_id: str
    issue_receipt_id: str
    tool_name: str
    origin: str
    effect_scope: str
    risk_level: str
    arguments_digest: str
    issued_at: float
    expires_at: float
    status: str = "issued"
    completion_receipt_id: str | None = None
    completion_error: str | None = None


@dataclass(frozen=True, slots=True)
class StandingAuthorityDecision:
    approved: bool
    reason: str
    context: dict[str, Any] = field(default_factory=dict)
    token: str | None = None
    grant_id: str | None = None
    receipt_id: str | None = None
    budget_remaining: int | None = None


def _builtin_grants() -> tuple[StandingAuthorityGrant, ...]:
    autonomous_origins = tuple(sorted(AUTONOMOUS_AUTHORITY_ORIGINS))
    return (
        StandingAuthorityGrant(
            grant_id="owner.foreground-request",
            issuer="owner_policy",
            description="Authenticated foreground requests may traverse the full governed tool spine.",
            allowed_origins=tuple(sorted(USER_FACING_AUTHORITY_ORIGINS)),
            allowed_tools=("*",),
            allowed_effect_scopes=("*",),
            max_risk="critical",
            max_actions=0,
            window_seconds=0.0,
            lease_ttl_seconds=120.0,
            argument_policy="foreground_user_request",
            built_in=True,
        ),
        StandingAuthorityGrant(
            grant_id="aura.autonomous-public-research",
            issuer="owner_policy",
            description="Aura may perform bounded public read-only research for her own goals.",
            allowed_origins=autonomous_origins,
            allowed_tools=tuple(sorted(PUBLIC_RESEARCH_TOOLS)),
            allowed_effect_scopes=("read_only",),
            max_risk="low",
            max_actions=120,
            window_seconds=900.0,
            lease_ttl_seconds=90.0,
            argument_policy="public_research",
            built_in=True,
        ),
        StandingAuthorityGrant(
            grant_id="aura.autonomous-introspection",
            issuer="owner_policy",
            description="Aura may observe her own health, beliefs, environment, and runtime state.",
            allowed_origins=autonomous_origins,
            allowed_tools=tuple(sorted(INTROSPECTION_TOOLS)),
            allowed_effect_scopes=("read_only", "status"),
            max_risk="low",
            max_actions=600,
            window_seconds=900.0,
            lease_ttl_seconds=60.0,
            argument_policy="local_introspection",
            built_in=True,
        ),
        StandingAuthorityGrant(
            grant_id="aura.autonomous-local-observation",
            issuer="owner_policy",
            description="Aura may read her own source and state roots without crossing into user files.",
            allowed_origins=autonomous_origins,
            allowed_tools=tuple(sorted(LOCAL_OBSERVATION_TOOLS)),
            allowed_effect_scopes=("read_only",),
            max_risk="low",
            max_actions=300,
            window_seconds=900.0,
            lease_ttl_seconds=60.0,
            argument_policy="aura_local_read",
            built_in=True,
        ),
        StandingAuthorityGrant(
            grant_id="aura.autonomous-connected-account-read",
            issuer="owner_policy",
            description=(
                "Aura may inspect connected inboxes and public community state without "
                "sending, posting, replying, or mutating remote state."
            ),
            allowed_origins=autonomous_origins,
            allowed_tools=tuple(sorted(CONNECTED_ACCOUNT_READ_TOOLS)),
            allowed_effect_scopes=("read_only",),
            max_risk="low",
            max_actions=60,
            window_seconds=900.0,
            lease_ttl_seconds=120.0,
            argument_policy="connected_account_read",
            built_in=True,
        ),
        StandingAuthorityGrant(
            grant_id="aura.autonomous-read-only-maintenance",
            issuer="owner_policy",
            description=(
                "Aura may inspect her own implementation and produce repair candidates "
                "without applying, promoting, or testing mutations."
            ),
            allowed_origins=autonomous_origins,
            allowed_tools=tuple(sorted(READ_ONLY_MAINTENANCE_TOOLS)),
            allowed_effect_scopes=("read_only",),
            max_risk="low",
            max_actions=12,
            window_seconds=3600.0,
            lease_ttl_seconds=180.0,
            argument_policy="read_only_maintenance",
            built_in=True,
        ),
        StandingAuthorityGrant(
            grant_id="aura.autonomous-bounded-maintenance-test",
            issuer="owner_policy",
            description=(
                "Aura may generate and execute a deterministic test for one of "
                "her own source files inside an ephemeral no-network sandbox."
            ),
            allowed_origins=autonomous_origins,
            allowed_tools=tuple(sorted(BOUNDED_MAINTENANCE_TEST_TOOLS)),
            allowed_effect_scopes=("sandboxed_compute",),
            max_risk="high",
            max_actions=6,
            window_seconds=3600.0,
            lease_ttl_seconds=180.0,
            argument_policy="bounded_maintenance_test",
            built_in=True,
        ),
        StandingAuthorityGrant(
            grant_id="aura.autonomous-bounded-sandbox-probe",
            issuer="owner_policy",
            description=(
                "Aura's subconscious may execute the exact checked-in idle probe "
                "inside the local no-network Python sandbox."
            ),
            allowed_origins=("subconscious_loop",),
            allowed_tools=tuple(sorted(BOUNDED_SANDBOX_TOOLS)),
            allowed_effect_scopes=("sandboxed_compute",),
            max_risk="high",
            max_actions=8,
            window_seconds=3600.0,
            lease_ttl_seconds=90.0,
            argument_policy="bounded_sandbox_probe",
            built_in=True,
        ),
        StandingAuthorityGrant(
            grant_id="aura.autonomous-background-reflection",
            issuer="owner_policy",
            description=(
                "Aura may run bounded, effect-free internal deliberation for reflection, "
                "planning, and self-assessment without requesting per-turn approval."
            ),
            allowed_origins=("background_reflection",),
            allowed_tools=tuple(sorted(BACKGROUND_REFLECTION_TOOLS)),
            allowed_effect_scopes=("pure_compute",),
            max_risk="low",
            max_actions=BACKGROUND_REFLECTION_MAX_ACTIONS,
            window_seconds=BACKGROUND_REFLECTION_WINDOW_SECONDS,
            lease_ttl_seconds=180.0,
            argument_policy="bounded_background_reflection",
            built_in=True,
        ),
        StandingAuthorityGrant(
            grant_id="aura.autonomous-owner-private-message",
            issuer="owner_policy",
            description=(
                "Aura may privately message her configured primary operator under a "
                "bounded cadence without exposing or selecting a raw destination."
            ),
            allowed_origins=autonomous_origins,
            allowed_tools=tuple(sorted(OWNER_PRIVATE_MESSAGE_TOOLS)),
            allowed_effect_scopes=("external_io", "state_mutation", "status"),
            max_risk="medium",
            max_actions=12,
            window_seconds=3600.0,
            lease_ttl_seconds=120.0,
            argument_policy="owner_private_message",
            built_in=True,
        ),
    )


class StandingAuthorityManager:
    """Own standing grants, durable budgets, and process-local child leases."""

    def __init__(
        self,
        *,
        state_gateway: Any | None = None,
        receipt_store: Any | None = None,
        token_store: CapabilityTokenStore | None = None,
        clock: Any = time.time,
        persistence_enabled: bool = True,
    ) -> None:
        self._builtins = {grant.grant_id: grant for grant in _builtin_grants()}
        self._custom: dict[str, StandingAuthorityGrant] = {}
        self._revocations: dict[str, dict[str, Any]] = {}
        self._budget_events: dict[str, list[float]] = {}
        self._leases: dict[str, StandingAuthorityLeaseRecord] = {}
        self._state_gateway = state_gateway
        self._receipt_store = receipt_store
        self._token_store = token_store or get_token_store()
        self._clock = clock
        self._persistence_enabled = bool(persistence_enabled)
        self._loaded = False
        self._load_error: str | None = None
        self._generation = 0
        self._async_lock = asyncio.Lock()
        self._lock = threading.RLock()

    def _gateway(self) -> Any:
        if self._state_gateway is None:
            from core.runtime.flags import env_str
            from core.state.state_gateway import get_state_gateway

            configured_root = state_root_override()
            test_root = env_str(
                "AURA_TEST_RUNTIME_ROOT",
                description="Hermetic test runtime root (set by the suite)",
                owner="core.executive.standing_authority",
            )
            root = (
                Path(configured_root)
                if configured_root
                else (Path(test_root) / "state" if test_root else None)
            )
            self._state_gateway = get_state_gateway(root=root)
        return self._state_gateway

    def _receipts(self) -> Any:
        if self._receipt_store is None:
            self._receipt_store = get_receipt_store()
        return self._receipt_store

    async def initialize(self, *, force: bool = False) -> dict[str, Any]:
        if self._loaded and not force:
            return self.get_status()
        async with self._async_lock:
            if self._loaded and not force:
                return self.get_status()
            if not self._persistence_enabled:
                self._loaded = True
                self._load_error = None
                return self.get_status()
            try:
                payload = await self._gateway().read(
                    STATE_KEY,
                    default=None,
                    domain=STATE_DOMAIN,
                    fresh=True,
                )
                with self._lock:
                    self._load_payload_locked(payload)
                    self._loaded = True
                    self._load_error = None
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                self._loaded = False
                self._load_error = f"{type(exc).__name__}: {exc}"
                record_degradation(
                    "standing_authority",
                    exc,
                    severity="error",
                    action="failed closed because standing-authority state could not be loaded",
                    enforce_failure_policy=False,
                )
                raise RuntimeError(f"standing_authority_state_unavailable:{exc}") from exc
        return self.get_status()

    def _load_payload_locked(self, payload: Any) -> None:
        if payload is None:
            self._custom = {}
            self._revocations = {}
            self._budget_events = {}
            self._generation = 0
            return
        if not isinstance(payload, Mapping):
            raise ValueError("standing-authority state is not a mapping")
        schema = int(payload.get("schema_version") or 0)
        if schema != STATE_SCHEMA_VERSION:
            raise ValueError(f"unsupported standing-authority schema: {schema}")
        custom: dict[str, StandingAuthorityGrant] = {}
        for grant_id, raw in dict(payload.get("custom_grants") or {}).items():
            grant = StandingAuthorityGrant.from_dict(dict(raw or {}))
            if grant.built_in:
                raise ValueError(f"persisted custom grant cannot claim built-in status: {grant_id}")
            if str(grant_id) != grant.grant_id or grant.grant_id in self._builtins:
                raise ValueError(f"invalid persisted custom grant identity: {grant_id}")
            custom[grant.grant_id] = grant
        known_ids = set(self._builtins) | set(custom)
        revocations = {
            str(grant_id): dict(value or {})
            for grant_id, value in dict(payload.get("revocations") or {}).items()
            if str(grant_id) in known_ids
        }
        now = float(self._clock())
        budgets: dict[str, list[float]] = {}
        for grant_id, raw_events in dict(payload.get("budget_events") or {}).items():
            if str(grant_id) not in known_ids:
                continue
            grant = self._builtins.get(str(grant_id)) or custom.get(str(grant_id))
            if grant is None or not grant.max_actions:
                continue
            floor = now - float(grant.window_seconds)
            events = []
            for value in list(raw_events or []):
                timestamp = float(value)
                if math.isfinite(timestamp) and floor <= timestamp <= now + 5.0:
                    events.append(timestamp)
            budgets[str(grant_id)] = sorted(events)
        self._custom = custom
        self._revocations = revocations
        self._budget_events = budgets
        self._generation = max(0, int(payload.get("generation") or 0))

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "generation": self._generation,
            "updated_at": float(self._clock()),
            "custom_grants": {
                grant_id: grant.to_dict() for grant_id, grant in sorted(self._custom.items())
            },
            "revocations": dict(sorted(self._revocations.items())),
            "budget_events": {
                grant_id: list(events)
                for grant_id, events in sorted(self._budget_events.items())
                if events
            },
        }

    async def _persist_locked(self, *, cause: str, receipt_id: str | None = None) -> None:
        if not self._persistence_enabled:
            return
        from core.governance_context import local_internal_governed_scope
        from core.runtime.gateways import StateMutationRequest

        payload = self._snapshot_locked()
        with local_internal_governed_scope(
            "standing_authority.registry",
            domain="state_mutation",
            constraints={"cause": cause, "state_domain": STATE_DOMAIN},
        ):
            await self._gateway().mutate(
                StateMutationRequest(
                    key=STATE_KEY,
                    new_value=payload,
                    receipt_id=receipt_id,
                    cause=cause,
                    domain=STATE_DOMAIN,
                )
            )

    def _all_grants_locked(self) -> dict[str, StandingAuthorityGrant]:
        return {**self._builtins, **self._custom}

    @staticmethod
    def _matches(values: tuple[str, ...], candidate: str) -> bool:
        return "*" in values or candidate in values

    @staticmethod
    def _validate_aura_source_paths(
        raw_paths: Any,
        *,
        require_file: bool = False,
        require_python: bool = False,
    ) -> tuple[bool, str]:
        """Confine autonomous maintenance to the checked-out Aura source."""

        if not isinstance(raw_paths, (list, tuple)) or not raw_paths:
            return False, "maintenance_target_missing"
        repo_root = Path(__file__).resolve().parents[2]
        resolved_root = repo_root.resolve()
        for raw_path in raw_paths:
            if not isinstance(raw_path, (str, os.PathLike)) or not str(raw_path).strip():
                return False, "maintenance_target_invalid"
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = repo_root / candidate
            try:
                resolved = candidate.resolve()
                within_repo = (
                    os.path.commonpath([str(resolved_root), str(resolved)])
                    == str(resolved_root)
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                return False, "maintenance_target_unresolvable"
            if not within_repo or not resolved.exists():
                return False, "maintenance_target_outside_aura_source"
            if require_file and not resolved.is_file():
                return False, "maintenance_target_is_not_file"
            if require_python and resolved.suffix != ".py":
                return False, "maintenance_target_is_not_python"
        return True, "maintenance_targets_valid"

    def _argument_policy_allows(
        self,
        grant: StandingAuthorityGrant,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        user_authorized: bool,
    ) -> tuple[bool, str]:
        policy = grant.argument_policy
        if policy == "any":
            return True, "custom_argument_policy"
        if policy == "foreground_user_request":
            return (True, "authenticated_foreground_request") if user_authorized else (
                False,
                "foreground_user_authority_missing",
            )
        if policy == "public_research":
            query = str(arguments.get("query") or arguments.get("q") or "").strip()
            if not query:
                return False, "public_research_query_missing"
            if len(query) > 8_000:
                return False, "public_research_query_too_large"
            lowered = query.lower()
            if any(marker in lowered for marker in _UNSAFE_PUBLIC_RESEARCH_MARKERS):
                return False, "public_research_requires_narrower_security_scope"
            return True, "bounded_public_research"
        if policy == "local_introspection":
            if tool_name not in INTROSPECTION_TOOLS:
                return False, "introspection_tool_mismatch"
            return True, "local_introspection"
        if policy == "bounded_sandbox_probe":
            if tool_name not in BOUNDED_SANDBOX_TOOLS:
                return False, "bounded_sandbox_probe_tool_mismatch"
            return validate_idle_sandbox_probe_arguments(arguments)
        if policy == "bounded_background_reflection":
            if tool_name not in BACKGROUND_REFLECTION_TOOLS:
                return False, "background_reflection_tool_mismatch"
            if set(arguments) - {"topic", "query", "roles"}:
                return False, "background_reflection_arguments_unsupported"
            topic = str(arguments.get("topic") or arguments.get("query") or "").strip()
            if not topic:
                return False, "background_reflection_topic_missing"
            if len(topic) > 4_096:
                return False, "background_reflection_topic_too_large"
            roles = arguments.get("roles", ["architect", "critic"])
            if not isinstance(roles, list) or not 1 <= len(roles) <= 3:
                return False, "background_reflection_roles_invalid"
            if any(
                not isinstance(role, str)
                or not role.strip()
                or role.strip().lower() not in BACKGROUND_REFLECTION_ROLES
                for role in roles
            ):
                return False, "background_reflection_roles_invalid"
            return True, "bounded_background_reflection"
        if policy == "connected_account_read":
            mode = str(arguments.get("mode") or "").strip().lower()
            allowed_modes = {
                "email_adapter": {"check", "read", "search"},
                "reddit_adapter": {
                    "browse",
                    "check_inbox",
                    "check_shadowban",
                    "read_post",
                    "read_rules",
                },
            }
            if mode not in allowed_modes.get(tool_name, set()):
                return False, "connected_account_operation_is_not_read_only"
            return True, "connected_account_read_only"
        if policy == "owner_private_message":
            if tool_name not in OWNER_PRIVATE_MESSAGE_TOOLS:
                return False, "owner_private_message_tool_mismatch"
            allowed_keys = {
                "action",
                "alias",
                "body_bytes",
                "body_chars",
                "body_sha256",
                "idempotency_key",
            }
            if set(arguments) - allowed_keys:
                return False, "owner_private_message_arguments_unsupported"
            action = str(arguments.get("action") or "status").strip().lower()
            alias = str(arguments.get("alias") or "primary_operator").strip().lower()
            if alias != "primary_operator":
                return False, "owner_private_message_alias_out_of_scope"
            if action == "status":
                return True, "owner_private_message_status"
            if action in {"pause", "resume"}:
                if any(
                    arguments.get(key)
                    for key in ("body_bytes", "body_chars", "body_sha256", "idempotency_key")
                ):
                    return False, "owner_private_message_control_arguments_unsupported"
                return True, "owner_private_message_control"
            if action != "send":
                return False, "owner_private_message_action_unsupported"
            body_sha256 = str(arguments.get("body_sha256") or "").strip().lower()
            try:
                body_chars = int(arguments.get("body_chars"))
                body_bytes = int(arguments.get("body_bytes"))
            except (TypeError, ValueError):
                return False, "owner_private_message_body_invalid"
            if (
                len(body_sha256) != 64
                or any(character not in "0123456789abcdef" for character in body_sha256)
                or not 1 <= body_chars <= 8_000
                or not 1 <= body_bytes <= 24_000
            ):
                return False, "owner_private_message_body_invalid"
            key = str(arguments.get("idempotency_key") or "").strip()
            if not key or len(key) > 240 or any(
                not (character.isalnum() or character in "._:-")
                for character in key
            ):
                return False, "owner_private_message_idempotency_invalid"
            return True, "bounded_owner_private_message"
        if policy == "read_only_maintenance":
            mode = str(arguments.get("mode") or arguments.get("action") or "scan").strip().lower()
            mutating = bool(
                arguments.get("apply")
                or arguments.get("write")
                or arguments.get("commit")
                or arguments.get("promote")
                or arguments.get("allow_mutation")
                or arguments.get("run_tests")
                or mode in {"apply", "commit", "promote", "rewrite", "write"}
            )
            if mutating:
                return False, "maintenance_operation_is_not_read_only"
            if tool_name == "self_evolution":
                if mode not in {"", "propose"} or not bool(arguments.get("read_only")):
                    return False, "self_evolution_is_not_read_only_proposal"
                objective = str(arguments.get("objective") or "").strip()
                if not objective or len(objective) > 4_096:
                    return False, "self_evolution_objective_invalid"
                raw_paths = arguments.get("files") or []
            elif tool_name == "auto_refactor":
                if mode not in {"", "scan"}:
                    return False, "auto_refactor_is_not_read_only_scan"
                raw_paths = [arguments.get("path") or "."]
            else:
                return False, "maintenance_tool_mismatch"
            valid, reason = self._validate_aura_source_paths(raw_paths)
            return (True, "read_only_maintenance") if valid else (False, reason)
        if policy == "bounded_maintenance_test":
            if tool_name not in BOUNDED_MAINTENANCE_TEST_TOOLS:
                return False, "bounded_maintenance_test_tool_mismatch"
            if not bool(arguments.get("read_only")):
                return False, "bounded_maintenance_test_requires_read_only"
            if set(arguments) - {"target_file", "read_only"}:
                return False, "bounded_maintenance_test_arguments_unsupported"
            valid, reason = self._validate_aura_source_paths(
                [arguments.get("target_file")],
                require_file=True,
                require_python=True,
            )
            return (
                (True, "bounded_maintenance_test")
                if valid
                else (False, reason)
            )
        if policy == "aura_local_read":
            if tool_name not in LOCAL_OBSERVATION_TOOLS:
                return False, "local_observation_tool_mismatch"
            raw_path = (
                arguments.get("path")
                or arguments.get("root")
                or arguments.get("directory")
                or "."
            )
            try:
                repo_root = Path(__file__).resolve().parents[2]
                aura_state = state_root()
                target = Path(str(raw_path)).expanduser()
                if not target.is_absolute():
                    target = repo_root / target
                resolved = target.resolve()
                allowed = any(
                    os.path.commonpath([str(root.resolve()), str(resolved)]) == str(root.resolve())
                    for root in (repo_root, aura_state)
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                allowed = False
            return (True, "aura_local_read") if allowed else (
                False,
                "local_observation_path_outside_aura_roots",
            )
        return False, "unsupported_argument_policy"

    def _select_grant_locked(
        self,
        *,
        tool_name: str,
        origin: str,
        arguments: dict[str, Any],
        effect_scope: str,
        risk_level: str,
        user_authorized: bool,
    ) -> tuple[StandingAuthorityGrant | None, str]:
        policy_reasons: list[str] = []
        for grant in self._all_grants_locked().values():
            if not grant.enabled or grant.grant_id in self._revocations:
                continue
            if not self._matches(grant.allowed_origins, origin):
                continue
            if not self._matches(grant.allowed_tools, tool_name):
                continue
            if not self._matches(grant.allowed_effect_scopes, effect_scope):
                continue
            if not risk_at_most(risk_level, grant.max_risk):
                continue
            allowed, reason = self._argument_policy_allows(
                grant,
                tool_name=tool_name,
                arguments=arguments,
                user_authorized=user_authorized,
            )
            if allowed:
                return grant, reason
            policy_reasons.append(f"{grant.grant_id}:{reason}")
        if policy_reasons:
            return None, ";".join(policy_reasons)
        return None, "no_matching_standing_grant"

    def _prune_budget_locked(self, grant: StandingAuthorityGrant, now: float) -> list[float]:
        if not grant.max_actions:
            return []
        floor = now - float(grant.window_seconds)
        events = [
            float(timestamp)
            for timestamp in self._budget_events.get(grant.grant_id, ())
            if float(timestamp) >= floor
        ]
        self._budget_events[grant.grant_id] = events
        return events

    def _budget_remaining_locked(self, grant: StandingAuthorityGrant, now: float) -> int | None:
        if not grant.max_actions:
            return None
        events = self._prune_budget_locked(grant, now)
        return max(0, int(grant.max_actions) - len(events))

    async def issue_child_lease(
        self,
        tool_name: Any,
        arguments: Mapping[str, Any] | None,
        *,
        origin: Any,
        context: Mapping[str, Any] | None = None,
        user_authorized: bool | None = None,
        effect_scope: Any = "",
        risk_level: Any = "",
    ) -> StandingAuthorityDecision:
        ctx = dict(context or {})
        name = normalize_tool_name(tool_name)
        args = dict(arguments or {})
        resolved_origin = coerce_authority_origin(origin or ctx.get("origin") or ctx.get("source"))
        scope = str(effect_scope or "").strip().lower() or resolve_execution_effect_scope(name, args)
        risk = normalize_risk(risk_level or classify_execution_risk(name, args, effect_scope=scope))
        user_auth = (
            context_has_user_authority(resolved_origin, ctx)
            if user_authorized is None
            else bool(user_authorized) and resolved_origin in USER_FACING_AUTHORITY_ORIGINS
        )

        try:
            await self.initialize()
        except RuntimeError as exc:
            return await self._deny(
                name,
                resolved_origin,
                scope,
                risk,
                f"standing_authority_unavailable:{exc}",
            )

        existing = str(ctx.get("standing_authority_token") or "").strip()
        if existing:
            valid, reason, record = self.validate_context(
                ctx,
                tool_name=name,
                arguments=args,
                origin=resolved_origin,
                effect_scope=scope,
                risk_level=risk,
            )
            if valid and record is not None:
                return StandingAuthorityDecision(
                    approved=True,
                    reason="existing_child_lease_valid",
                    context=dict(ctx),
                    token=existing,
                    grant_id=record.grant_id,
                    receipt_id=record.issue_receipt_id,
                    budget_remaining=self._remaining_for_grant(record.grant_id),
                )
            return await self._deny(name, resolved_origin, scope, risk, reason)

        async with self._async_lock:
            now = float(self._clock())
            with self._lock:
                grant, grant_reason = self._select_grant_locked(
                    tool_name=name,
                    origin=resolved_origin,
                    arguments=args,
                    effect_scope=scope,
                    risk_level=risk,
                    user_authorized=user_auth,
                )
                if grant is None:
                    denial = grant_reason
                else:
                    remaining = self._budget_remaining_locked(grant, now)
                    denial = "" if remaining is None or remaining > 0 else "standing_authority_budget_exhausted"
            if denial:
                return await self._deny(name, resolved_origin, scope, risk, denial)

            assert grant is not None
            budget_added = False
            previous_generation = self._generation
            issue_receipt_id = f"autonomy-{uuid.uuid4()}"
            with self._lock:
                if grant.max_actions:
                    self._budget_events.setdefault(grant.grant_id, []).append(now)
                    budget_added = True
                    self._generation += 1
                remaining = self._budget_remaining_locked(grant, now)
            if budget_added:
                try:
                    await self._persist_locked(
                        cause=f"standing_authority.issue:{grant.grant_id}",
                        receipt_id=issue_receipt_id,
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    with self._lock:
                        events = self._budget_events.get(grant.grant_id, [])
                        if now in events:
                            events.remove(now)
                        self._generation = previous_generation
                        self._loaded = False
                    record_degradation(
                        "standing_authority",
                        exc,
                        severity="error",
                        action="denied child lease because durable budget reservation failed",
                        enforce_failure_policy=False,
                    )
                    return await self._deny(
                        name,
                        resolved_origin,
                        scope,
                        risk,
                        f"standing_authority_budget_persistence_failed:{type(exc).__name__}",
                    )

            args_digest = canonical_arguments_digest(args)
            token = self._token_store.issue(
                origin=resolved_origin,
                scope=grant.grant_id,
                ttl_seconds=float(grant.lease_ttl_seconds),
                domain="tool_execution",
                requested_action=name,
                approver="standing_authority",
                parent_receipt=issue_receipt_id,
            )
            record = StandingAuthorityLeaseRecord(
                token=token.token,
                grant_id=grant.grant_id,
                issue_receipt_id=issue_receipt_id,
                tool_name=name,
                origin=resolved_origin,
                effect_scope=scope,
                risk_level=risk,
                arguments_digest=args_digest,
                issued_at=token.issued_at,
                expires_at=token.issued_at + token.ttl_seconds,
            )
            receipt = AutonomyReceipt(
                receipt_id=issue_receipt_id,
                cause=f"standing_authority:{grant.grant_id}",
                autonomy_level=2 if user_auth else 1,
                proposed_action=f"tool:{name}",
                budget_remaining=float(remaining if remaining is not None else -1),
                metadata={
                    "event": "child_lease_issued",
                    "grant_id": grant.grant_id,
                    "origin": resolved_origin,
                    "effect_scope": scope,
                    "risk_level": risk,
                    "arguments_digest": args_digest,
                    "token_digest": hashlib.sha256(token.token.encode("utf-8")).hexdigest(),
                    "expires_at": record.expires_at,
                    "grant_reason": grant_reason,
                },
            )
            try:
                await asyncio.to_thread(self._receipts().emit, receipt)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                self._token_store.revoke(token.token, reason="issuance_receipt_failed")
                record_degradation(
                    "standing_authority",
                    exc,
                    severity="error",
                    action="revoked child lease because issuance receipt failed",
                    enforce_failure_policy=False,
                )
                return await self._deny(
                    name,
                    resolved_origin,
                    scope,
                    risk,
                    f"standing_authority_receipt_failed:{type(exc).__name__}",
                )
            with self._lock:
                self._leases[token.token] = record

            lease_ref = hashlib.sha256(token.token.encode("utf-8")).hexdigest()[:16]
            authority_context = {
                **ctx,
                "tool": name,
                "skill": name,
                "origin": resolved_origin,
                "source": resolved_origin,
                "authority_origin": resolved_origin,
                "authority_args_digest": args_digest,
                "effect_scope": scope,
                "risk_level": risk,
                "standing_authority_token": token.token,
                "standing_authority_grant_id": grant.grant_id,
                "standing_authority_receipt_id": issue_receipt_id,
                "scoped_authority": f"standing:{grant.grant_id}:{lease_ref}",
            }
            return StandingAuthorityDecision(
                approved=True,
                reason=f"standing_grant:{grant.grant_id}",
                context=authority_context,
                token=token.token,
                grant_id=grant.grant_id,
                receipt_id=issue_receipt_id,
                budget_remaining=remaining,
            )

    async def _deny(
        self,
        tool_name: str,
        origin: str,
        effect_scope: str,
        risk_level: str,
        reason: str,
    ) -> StandingAuthorityDecision:
        receipt_id = f"autonomy-{uuid.uuid4()}"
        receipt = AutonomyReceipt(
            receipt_id=receipt_id,
            cause="standing_authority_denied",
            autonomy_level=0,
            proposed_action=f"tool:{tool_name}",
            budget_remaining=0.0,
            metadata={
                "event": "child_lease_denied",
                "origin": origin,
                "effect_scope": effect_scope,
                "risk_level": risk_level,
                "reason": str(reason)[:500],
            },
        )
        try:
            await asyncio.to_thread(self._receipts().emit, receipt)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "standing_authority",
                exc,
                severity="warning",
                action="kept child lease denied after denial receipt failed",
                enforce_failure_policy=False,
            )
            receipt_id = None
        return StandingAuthorityDecision(
            approved=False,
            reason=str(reason),
            receipt_id=receipt_id,
        )

    def validate_context(
        self,
        context: Mapping[str, Any] | None,
        *,
        tool_name: Any = "",
        arguments: Mapping[str, Any] | None = None,
        origin: Any = "",
        effect_scope: Any = "",
        risk_level: Any = "",
    ) -> tuple[bool, str, StandingAuthorityLeaseRecord | None]:
        ctx = dict(context or {})
        token_value = str(ctx.get("standing_authority_token") or "").strip()
        if not token_value:
            return False, "signed_standing_authority_lease_missing", None
        name = normalize_tool_name(tool_name or ctx.get("tool") or ctx.get("skill"))
        if not name:
            return False, "standing_authority_tool_missing", None
        try:
            token = self._token_store.validate(
                token_value,
                domain="tool_execution",
                action=name,
            )
        except PermissionError as exc:
            return False, str(exc), None
        with self._lock:
            record = self._leases.get(token_value)
            grants = self._all_grants_locked()
            revocations = dict(self._revocations)
        if record is None:
            return False, "standing_authority_lease_record_missing", None
        grant = grants.get(record.grant_id)
        if grant is None or record.grant_id in revocations or not grant.enabled:
            return False, "standing_authority_grant_inactive", record
        if token.scope != record.grant_id or token.parent_receipt != record.issue_receipt_id:
            return False, "standing_authority_token_binding_mismatch", record
        if str(ctx.get("standing_authority_grant_id") or "") != record.grant_id:
            return False, "standing_authority_grant_context_mismatch", record
        if str(ctx.get("standing_authority_receipt_id") or "") != record.issue_receipt_id:
            return False, "standing_authority_receipt_context_mismatch", record
        resolved_origin = coerce_authority_origin(
            origin or ctx.get("authority_origin") or ctx.get("origin") or ctx.get("source")
        )
        if resolved_origin != record.origin:
            return False, "standing_authority_origin_mismatch", record
        args_digest = (
            canonical_arguments_digest(arguments)
            if arguments is not None
            else str(ctx.get("authority_args_digest") or "")
        )
        if not args_digest or args_digest != record.arguments_digest:
            return False, "standing_authority_arguments_mismatch", record
        # Derive the scope the way the LEASE derived it.
        #
        # issue_child_lease falls back to resolve_execution_effect_scope(name,
        # args) when no scope is passed, and this did not, so a caller that
        # omitted it presented "" against a recorded scope and could never
        # match. LIVE 2026-08-19: computer_use, desktop_task, self_evolution
        # and read_screen_text were all refused with
        # standing_authority_effect_scope_mismatch, which is what a guaranteed
        # inequality looks like from the outside — her self-directed action
        # could not execute a tool at all.
        #
        # This does not loosen the check. The comparison against the recorded
        # scope is unchanged, and the tool and arguments it derives from are
        # separately bound above (tool_out_of_grant, arguments_mismatch), so
        # the fallback can only reproduce the recorded value for the same
        # tool and the same arguments the lease was issued for.
        scope = str(effect_scope or ctx.get("effect_scope") or "").strip().lower()
        if not scope:
            scope = str(
                resolve_execution_effect_scope(name, dict(arguments or {}))
            ).strip().lower()
        if scope != record.effect_scope and not scope_is_within(scope, record.effect_scope):
            # Say WHICH scopes disagreed.
            #
            # A refusal that names only the rule cannot be diagnosed from
            # outside: measured live, "write a file on my Desktop" was
            # refused with standing_authority_effect_scope_mismatch and
            # nothing anywhere recorded that the lease held one value and the
            # check derived another. The whole reason this comparison exists
            # is that the two can differ, so the two belong in the reason.
            logger.warning(
                "Standing authority scope mismatch for %s: lease held %r, this call derived %r",
                name,
                record.effect_scope,
                scope,
            )
            return (
                False,
                f"standing_authority_effect_scope_mismatch (lease={record.effect_scope!r},"
                f" call={scope!r})",
                record,
            )
        # Same asymmetry, one field along: the lease classifies the risk from
        # the tool when none is given. The RAW value decides whether anything
        # was supplied — normalize_risk("") returns a default rather than an
        # empty string, so testing the normalised value never finds a gap.
        raw_risk = str(risk_level or ctx.get("risk_level") or "").strip()
        risk = (
            normalize_risk(raw_risk)
            if raw_risk
            else normalize_risk(
                classify_execution_risk(name, dict(arguments or {}), effect_scope=scope)
            )
        )
        if risk != record.risk_level:
            return False, "standing_authority_risk_mismatch", record
        if not self._matches(grant.allowed_origins, record.origin):
            return False, "standing_authority_origin_out_of_grant", record
        if not self._matches(grant.allowed_tools, record.tool_name):
            return False, "standing_authority_tool_out_of_grant", record
        if not self._matches(grant.allowed_effect_scopes, record.effect_scope):
            return False, "standing_authority_effect_out_of_grant", record
        if not risk_at_most(record.risk_level, grant.max_risk):
            return False, "standing_authority_risk_out_of_grant", record
        return True, "standing_authority_lease_valid", record

    def finalize_child_lease(
        self,
        token_value: str | None,
        *,
        success: bool,
        result: Any = None,
        error: str = "",
    ) -> dict[str, Any]:
        token_text = str(token_value or "").strip()
        if not token_text:
            return {"closed": True, "mode": "no_standing_lease", "errors": []}
        with self._lock:
            record = self._leases.get(token_text)
            if record is None:
                return {
                    "closed": False,
                    "mode": "standing_authority",
                    "errors": ["standing_authority_lease_record_missing"],
                }
            if record.completion_receipt_id:
                return {
                    "closed": True,
                    "mode": "standing_authority",
                    "receipt_id": record.completion_receipt_id,
                    "status": record.status,
                    "errors": [],
                }
            token = self._token_store.get(token_text)
            if token is None:
                return {
                    "closed": False,
                    "mode": "standing_authority",
                    "errors": ["standing_authority_capability_token_missing"],
                }
            if not token.is_consumed():
                try:
                    self._token_store.consume(
                        token_text,
                        child_receipt=f"standing-exec-{uuid.uuid4()}",
                    )
                except PermissionError as exc:
                    return {
                        "closed": False,
                        "mode": "standing_authority",
                        "errors": [str(exc)],
                    }
            self._token_store.revoke(token_text, reason="execution_complete")
            record.status = "completed" if success else "failed"

        digest = canonical_arguments_digest({"result": result, "error": error})
        receipt_id = f"autonomy-{uuid.uuid4()}"
        receipt = AutonomyReceipt(
            receipt_id=receipt_id,
            cause=f"standing_authority_complete:{record.grant_id}",
            autonomy_level=1,
            proposed_action=f"tool:{record.tool_name}",
            budget_remaining=float(self._remaining_for_grant(record.grant_id) or 0),
            governance_receipt_id=record.issue_receipt_id,
            metadata={
                "event": "child_lease_completed",
                "grant_id": record.grant_id,
                "origin": record.origin,
                "success": bool(success),
                "status": record.status,
                "outcome_digest": digest,
                "error_type": str(error).split(":", 1)[0][:120] if error else "",
            },
        )
        try:
            self._receipts().emit(receipt)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            with self._lock:
                record.completion_error = f"{type(exc).__name__}: {exc}"
            record_degradation(
                "standing_authority",
                exc,
                severity="error",
                action="child lease was consumed but completion receipt failed",
                enforce_failure_policy=False,
            )
            return {
                "closed": False,
                "mode": "standing_authority",
                "status": record.status,
                "errors": [record.completion_error],
            }
        with self._lock:
            record.completion_receipt_id = receipt_id
            record.completion_error = None
        return {
            "closed": True,
            "mode": "standing_authority",
            "receipt_id": receipt_id,
            "status": record.status,
            "errors": [],
        }

    def _remaining_for_grant(self, grant_id: str) -> int | None:
        with self._lock:
            grant = self._all_grants_locked().get(grant_id)
            if grant is None:
                return 0
            return self._budget_remaining_locked(grant, float(self._clock()))

    @staticmethod
    def _require_owner_evidence(actor: Any, evidence: Mapping[str, Any] | None) -> None:
        origin = coerce_authority_origin(actor)
        ctx = dict(evidence or {})
        authenticated = str(ctx.get("authenticated_principal") or "").strip().lower() == "owner"
        authenticated = authenticated or _truthy(ctx.get("internal_authenticated"))
        explicit = _truthy(ctx.get("user_explicitly_authorized"))
        if origin not in USER_FACING_AUTHORITY_ORIGINS or not authenticated or not explicit:
            raise PermissionError("owner-authenticated explicit authority change required")

    async def revoke_grant(
        self,
        grant_id: str,
        *,
        actor: Any,
        evidence: Mapping[str, Any] | None,
        reason: str,
    ) -> dict[str, Any]:
        await self.initialize()
        self._require_owner_evidence(actor, evidence)
        normalized_id = str(grant_id or "").strip()
        async with self._async_lock:
            with self._lock:
                if normalized_id not in self._all_grants_locked():
                    raise KeyError(normalized_id)
                previous = dict(self._revocations)
                previous_generation = self._generation
                self._revocations[normalized_id] = {
                    "actor": coerce_authority_origin(actor),
                    "reason": str(reason or "owner_revoked")[:500],
                    "revoked_at": float(self._clock()),
                }
                self._generation += 1
                try:
                    await self._persist_locked(cause=f"standing_authority.revoke:{normalized_id}")
                except asyncio.CancelledError:
                    self._revocations = previous
                    self._generation = previous_generation
                    raise
                except (OSError, RuntimeError, TypeError, ValueError):
                    self._revocations = previous
                    self._generation = previous_generation
                    raise
                revoked_tokens = []
                for token_value, lease in self._leases.items():
                    if lease.grant_id == normalized_id and lease.completion_receipt_id is None:
                        self._token_store.revoke(token_value, reason=f"grant_revoked:{normalized_id}")
                        revoked_tokens.append(token_value)
        return {
            "ok": True,
            "grant_id": normalized_id,
            "revoked_active_leases": len(revoked_tokens),
            "generation": self._generation,
        }

    async def restore_grant(
        self,
        grant_id: str,
        *,
        actor: Any,
        evidence: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        await self.initialize()
        self._require_owner_evidence(actor, evidence)
        normalized_id = str(grant_id or "").strip()
        async with self._async_lock:
            with self._lock:
                if normalized_id not in self._all_grants_locked():
                    raise KeyError(normalized_id)
                previous = dict(self._revocations)
                previous_generation = self._generation
                self._revocations.pop(normalized_id, None)
                self._generation += 1
                try:
                    await self._persist_locked(cause=f"standing_authority.restore:{normalized_id}")
                except asyncio.CancelledError:
                    self._revocations = previous
                    self._generation = previous_generation
                    raise
                except (OSError, RuntimeError, TypeError, ValueError):
                    self._revocations = previous
                    self._generation = previous_generation
                    raise
        return {"ok": True, "grant_id": normalized_id, "generation": self._generation}

    async def install_grant(
        self,
        grant: StandingAuthorityGrant,
        *,
        actor: Any,
        evidence: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        await self.initialize()
        self._require_owner_evidence(actor, evidence)
        if grant.built_in or grant.grant_id in self._builtins:
            raise ValueError("built-in standing-authority grants cannot be replaced")
        async with self._async_lock:
            with self._lock:
                previous = dict(self._custom)
                previous_generation = self._generation
                self._custom[grant.grant_id] = grant
                self._generation += 1
                try:
                    await self._persist_locked(cause=f"standing_authority.install:{grant.grant_id}")
                except asyncio.CancelledError:
                    self._custom = previous
                    self._generation = previous_generation
                    raise
                except (OSError, RuntimeError, TypeError, ValueError):
                    self._custom = previous
                    self._generation = previous_generation
                    raise
        return {"ok": True, "grant_id": grant.grant_id, "generation": self._generation}

    def get_status(self) -> dict[str, Any]:
        now = float(self._clock())
        with self._lock:
            grants = self._all_grants_locked()
            active_leases = [
                lease
                for lease in self._leases.values()
                if not lease.completion_receipt_id and lease.expires_at >= now
            ]
            grant_status = []
            for grant_id, grant in sorted(grants.items()):
                grant_status.append(
                    {
                        "grant_id": grant_id,
                        "issuer": grant.issuer,
                        "description": grant.description,
                        "built_in": grant.built_in,
                        "enabled": grant.enabled and grant_id not in self._revocations,
                        "revocation": self._revocations.get(grant_id),
                        "max_risk": grant.max_risk,
                        "budget": {
                            "max_actions": grant.max_actions,
                            "window_seconds": grant.window_seconds,
                            "remaining": self._budget_remaining_locked(grant, now),
                        },
                    }
                )
            return {
                "schema": "aura.standing_authority.status.v1",
                "ready": bool(self._loaded and not self._load_error),
                "loaded": self._loaded,
                "load_error": self._load_error,
                "generation": self._generation,
                "grants": grant_status,
                "active_leases": len(active_leases),
                "issued_leases": len(self._leases),
                "revoked_grants": len(self._revocations),
            }

    def shutdown(self) -> dict[str, Any]:
        revoked = 0
        with self._lock:
            tokens = tuple(self._leases)
        for token_value in tokens:
            token = self._token_store.get(token_value)
            if token is not None and not token.revoked:
                self._token_store.revoke(token_value, reason="standing_authority_shutdown")
                revoked += 1
        return {"closed": True, "revoked_tokens": revoked}


_manager: StandingAuthorityManager | None = None
_manager_lock = threading.RLock()


def get_standing_authority_manager() -> StandingAuthorityManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = StandingAuthorityManager()
        manager = _manager
    try:
        from core.container import ServiceContainer

        if not ServiceContainer.has("standing_authority"):
            ServiceContainer.register_instance("standing_authority", manager, required=False)
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Standing-authority service registration deferred: %s", exc)
    return manager


def reset_standing_authority_manager() -> None:
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.shutdown()
        _manager = None


def validate_standing_authority_context(
    context: Mapping[str, Any] | None,
    *,
    tool_name: Any = "",
    arguments: Mapping[str, Any] | None = None,
    origin: Any = "",
    effect_scope: Any = "",
    risk_level: Any = "",
) -> tuple[bool, str]:
    valid, reason, _record = get_standing_authority_manager().validate_context(
        context,
        tool_name=tool_name,
        arguments=arguments,
        origin=origin,
        effect_scope=effect_scope,
        risk_level=risk_level,
    )
    return valid, reason


__all__ = [
    "AUTONOMOUS_AUTHORITY_ORIGINS",
    "CONNECTED_ACCOUNT_READ_TOOLS",
    "INTROSPECTION_TOOLS",
    "LOCAL_OBSERVATION_TOOLS",
    "PUBLIC_RESEARCH_TOOLS",
    "READ_ONLY_MAINTENANCE_TOOLS",
    "StandingAuthorityDecision",
    "StandingAuthorityGrant",
    "StandingAuthorityLeaseRecord",
    "StandingAuthorityManager",
    "USER_FACING_AUTHORITY_ORIGINS",
    "canonical_arguments_digest",
    "coerce_authority_origin",
    "context_has_user_authority",
    "get_standing_authority_manager",
    "normalize_authority_origin",
    "reset_standing_authority_manager",
    "validate_standing_authority_context",
]
