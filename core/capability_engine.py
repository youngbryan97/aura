import asyncio
import hashlib
import importlib
import inspect
import json
import logging
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import CheckedLock, LockRank, checked_lock
from core.runtime.network_gateway import get_network_gateway
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("core.capability_engine")

#: The search family, which several call sites suppress together when the turn
#: is asking ABOUT searching rather than asking to search.
_SEARCH_SKILL_NAMES = frozenset({
    "web_search",
    "search_web",
    "free_search",
    "grounded_search",
    "sovereign_browser",
})

#: What turns a bare skill name into an instruction to use it. Required before
#: an ambiguous single-word name ("clock", "listen", "speak") counts as naming
#: that skill, so ordinary prose does not summon tools.
_SKILL_INVOCATION_CUE_RE = re.compile(
    r"\b(?:use|using|via|with|run|call|invoke|trigger|execute|apply|through|"
    r"start|open)\s+(?:the\s+|your\s+|my\s+)?(?:\w+\s+){0,2}$",
    re.IGNORECASE,
)

_TOOL_AFFORDANCE_SCAN_BUDGET_SECONDS = 0.05
_TOOL_AFFORDANCE_SCAN_LIMIT = 192
_CATALOG_LOCK_BOOTSTRAP = threading.Lock()

# Catalog lock order, innermost last. Every acquisition must be strictly
# increasing; the ranks below are enforced by lockdep, not by convention.
#
#   catalog_load (LANE) -> catalog_mutation (RESOURCE) -> catalog (LEAF)
#
# The rule that matters at every call site: the ``skills`` and ``instances``
# properties lazily take ``catalog_load``, so reading them while holding
# ``catalog`` inverts the order. Inside a ``_catalog_guard()`` block, read the
# ``_skills`` / ``_instances`` attributes instead, and call
# ``_ensure_catalog_loaded()`` *before* taking the guard when the catalog has
# to be populated. Getting this wrong once wedged the boot loop and the health
# probe against each other until the reaper killed the kernel.
_CATALOG_LOAD_LOCK_RANK = LockRank.LANE
_CATALOG_MUTATION_LOCK_RANK = LockRank.RESOURCE
_CATALOG_LOCK_RANK = LockRank.LEAF
_VERIFIED_STANDING_AUTHORITY = object()

try:
    from RestrictedPython import compile_restricted, safe_builtins, utility_builtins
    from RestrictedPython.Guards import (
        full_write_guard,
        guarded_getitem,
        safer_getattr,
    )
    from RestrictedPython.PrintCollector import PrintCollector

    RESTRICTED_AVAILABLE = True
    #: The guards RestrictedPython ships for the hooks its compiler emits.
    #: The previous code passed raw getattr, a bare subscript lambda and an
    #: identity write guard, which turns the rewrite into a no-op with extra
    #: steps.
    _RESTRICTED_GUARDS = {
        "getattr": safer_getattr,
        "getitem": guarded_getitem,
        "write": full_write_guard,
    }
except ImportError:
    RESTRICTED_AVAILABLE = False
    _RESTRICTED_GUARDS = {}

from pydantic import BaseModel, ConfigDict, Field, ValidationError  # noqa: E402

from core.config import config  # noqa: E402
from core.container import ServiceContainer  # noqa: E402
from core.exceptions import ContainerError  # noqa: E402
from core.executive.execution_policy import (  # noqa: E402
    canonical_authority_arguments,
    canonical_authority_context,
    classify_execution_risk,
    resolve_execution_effect_scope,
)
from core.executive.standing_authority import (  # noqa: E402
    AUTONOMOUS_AUTHORITY_ORIGINS,
)
from core.governance.capability_chain import (  # noqa: E402
    CapabilityDenial,
    CapabilityViolation,
    capability_enforcement_mode,
    capability_from_context,
    enforce_capability,
    get_capability_verifier,
)
from core.runtime.base_module import AuraBaseModule  # noqa: E402
from core.runtime.idempotency import (  # noqa: E402
    get_idempotency_ledger,
    requires_idempotency_key,
)
from core.runtime.service_access import (  # noqa: E402
    optional_service,
    resolve_edi,
    resolve_homeostatic_coupling,
    resolve_metabolic_monitor,
)
from core.security.structural_redaction import (  # noqa: E402
    redact_mapping,
    redact_structure,
    redact_text,
    redaction_marker,
)
from core.skill_management.forged_artifact import ArtifactError  # noqa: E402
from core.skill_management.skill_verification import (  # noqa: E402
    ENTRYPOINT as FORGED_ENTRYPOINT,
)
from core.skills.base_skill import (  # noqa: E402
    SKILL_TIMEOUT_CONTEXT_KEY as _SKILL_TIMEOUT_CONTEXT_KEY,
)
from core.skills.catalog_policy import resolve_skill_policy  # noqa: E402
from core.utils.intent_normalization import normalize_memory_intent_text  # noqa: E402

#: How much longer the engine waits than the budget it handed the skill, so a
#: skill that runs out of time reports its own structured failure instead of
#: being cancelled mid-sentence by the outer wait.
_OUTER_TIMEOUT_GRACE_S = 15.0

_USER_FACING_CONTEXT_ORIGINS = frozenset(
    {
        "user",
        "voice",
        "admin",
        "api",
        "gui",
        "ws",
        "websocket",
        "direct",
        "external",
        "desktop",
        "desktop-ui",
        "desktop_ui",
        "messages",
        "native-shell",
        "tauri",
    }
)

_SEARCH_CAPABILITY_QUESTION_RE = re.compile(
    r"\b(?:can|could|do|does|are|is|have|has)\b.{0,80}\b(?:you|aura)\b.{0,80}"
    r"\b(?:search|internet access|web access|browse|read links?)\b",
    re.IGNORECASE,
)

_SEARCH_WITH_TARGET_RE = re.compile(
    r"\b(?:search|look up|find|browse|read)\b.{0,40}\b(?:for|about|on|at|this|that)\b\s+\S+",
    re.IGNORECASE,
)

_HEAVY_BACKGROUND_SKILLS = frozenset(
    {
        "auto_refactor",
        "coding_skill",
        "self_improvement",
        "self_repair",
        "self_modify",
        "shadow_ast_healer",
        "skill_evolution",
        "test_generator",
        "train_self",
    }
)

_FOREGROUND_EXCLUSIVE_BACKGROUND_SKILLS = frozenset(
    {
        "email_adapter",
        "reddit_adapter",
        "sovereign_browser",
        "sovereign_network",
        "web_interlocutor",
    }
)

_LIGHTWEIGHT_BACKGROUND_IO_SKILLS = frozenset(
    {
        "free_search",
        "grounded_search",
        "search_web",
        "web_search",
    }
)

#: Registered skill names that are the SAME capability, mapped to the one name
#: that should represent them, so a chooser sees a capability once.
#:
#: ``search_web`` and ``free_search`` are subclasses of the class that
#: registers as ``web_search`` and add nothing — ``free_search`` says so in its
#: own docstring, "Compatibility wrapper for legacy 'free_search' skill". The
#: code was already deduplicated by inheritance; the CATALOG was not, so all
#: three were offered as separate options and a chooser had to pick between
#: three identical tools.
#:
#: Measured live 2026-08-10: asked for the weather, she answered "I don't have
#: a window, camera, thermometer or weather feed" while five search skills sat
#: READY. Three of the five were this one skill wearing three names.
#:
#: Only true aliases belong here. grounded_search (Google API with citation
#: grounding), local_reference_search (offline corpus, survives no network),
#: sovereign_browser (interactive browsing) and web_interlocutor (conversing
#: with another web agent) are genuinely different capabilities and are NOT
#: aliases, however similar their names look.
#:
#: The names stay registered and callable — stored plans, beliefs and learned
#: policies reference them, and breaking those to tidy a list would trade a
#: cosmetic problem for a real one. They are hidden from the catalog, not
#: removed from the registry.
#: ``sovereign_imagination`` is the same shape: a "facade over the canonical
#: diffusers backend" (its own docstring) that subclasses ImageGenSkill,
#: validates params and calls super().execute. Same effect_scope, same risk
#: class — and a strictly SMALLER surface, since image_gen also does
#: image-to-image and the facade's input model has no field for it. Offering
#: both did not merely duplicate a choice, it offered a choice that silently
#: loses capability.
_SKILL_ALIASES: dict[str, str] = {
    "search_web": "web_search",
    "free_search": "web_search",
    "sovereign_imagination": "image_gen",
}

#: Origins that mean "a person is asking, right now, at the surface".
_DIRECT_USER_REQUEST_ORIGINS = frozenset(
    {
        "user",
        "desktop",
        "desktop_ui",
        "desktop_task",
        "native_shell",
        "gui",
        "voice",
        "ws",
        "chat",
        "api",
    }
)

_AUTONOMOUS_RESEARCH_ORIGINS = frozenset(
    {
        "autonomy",
        "background",
        "background_reflection",
        "curiosity",
        "curiosity_daemon",
        "curiosity_explorer",
        "dream",
        "intention_loop",
        "latent_cortex",
        "overt_action_loop",
        "research_cycle",
        "subconscious_loop",
        "temporal",
    }
)

_UNSAFE_AUTONOMOUS_WEB_QUERY_MARKERS = frozenset(
    {
        "api key",
        "brute force",
        "bypass login",
        "credential",
        "credentials",
        "ddos",
        "deanonymize",
        "dox",
        "doxx",
        "exfiltrate",
        "exploit",
        "malware",
        "password",
        "phishing",
        "private key",
        "ransomware",
        "session cookie",
        "steal",
        "token dump",
        "worm",
    }
)

_READ_ONLY_EFFECT_SKILLS = frozenset(
    {
        "clock",
        "environment_info",
        "free_search",
        "grounded_search",
        "local_reference_search",
        "query_beliefs",
        "search_web",
        "system_proprioception",
        "evolution_status",
        "malware_analysis",
        "sec_ops",
        "stealth_ops",
        "web_search",
    }
)

_PURE_COMPUTE_EFFECT_SKILLS = frozenset(
    {
        "induced_repeating_shift_decode",
        "propagation",
    }
)

_SANDBOXED_COMPUTE_EFFECT_SKILLS = frozenset(
    {
        "run_code",
        "internal_sandbox",
    }
)

_EXTERNAL_IO_EFFECT_SKILLS = frozenset(
    {
        "free_search",
        "grounded_search",
        "sovereign_browser",
        "sovereign_network",
        "web_interlocutor",
        "web_search",
    }
)

_STATEFUL_EFFECT_SKILLS = frozenset(
    {
        "add_belief",
        "file_operation",
        "memory_ops",
        "memory_sync",
        "personality",
    }
)

_CRITICAL_ACTION_SKILLS = frozenset(
    {
        "auto_refactor",
        "install_package",
        "manage_abilities",
        "os_manipulation",
        "self_evolution",
        "self_improvement",
        "self_modify",
        "self_repair",
        "sovereign_terminal",
        "train_self",
        "web_interlocutor",
    }
)


#: Effect scopes that may still run when the constitutional gate itself is
#: unavailable during boot. Every one of these is incapable of changing
#: anything outside the process. Anything else — including ``unknown`` —
#: waits for a working gate.
#: Anything a discarded skill's teardown may plausibly raise. A reload must
#: not fail because an instance being thrown away refused to die.
_INSTANCE_RETIREMENT_ERRORS = (
    OSError,
    RuntimeError,
    AttributeError,
    TypeError,
    ValueError,
    ConnectionError,
    TimeoutError,
)

_SKILL_PREFLIGHT_ERRORS = (
    ImportError,
    OSError,
    RuntimeError,
    AttributeError,
    TypeError,
    ValueError,
    ConnectionError,
    TimeoutError,
)

_PRE_RUNTIME_UNGATED_EFFECT_SCOPES: frozenset[str] = frozenset({
    "pure_compute",
    "read_only",
    "sandboxed_compute",
    "status",
})


def _proof_run_environment_active() -> bool:
    """Is this process actually running as a proof/benchmark run?

    CP126: "User authorization accepts context booleans and source-like
    labels, including proof_evaluation_contract, rather than requiring a
    verified principal and signed request provenance. Internal callers can
    mint user-equivalent execution."

    proof_evaluation_contract is a POLICY flag describing what kind of run
    this is. It was being read as though it were a user standing behind the
    request. Anything that can put a key in a context dict — including a
    model-authored tool call — could therefore mint user-equivalent
    authority by claiming to be a benchmark.

    A real proof run is started by the operator with AURA_PROOF_RUN in the
    environment, which a request payload cannot reach. Pairing the two keeps
    genuine proof runs working and makes the bare flag worthless on its own.
    """
    return str(os.environ.get("AURA_PROOF_RUN", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _record_unreconciled_authority(auth: Any, *, reason: str) -> None:
    """Queue an authority grant whose finalization never completed.

    Used where finalize_tool_execution RAISED, so there is no receipt to
    inspect. When it merely returns closed=False the gateway queues the
    entry itself, so a caller cannot make the leak vanish by ignoring the
    return value.
    """
    try:
        from core.executive.authority_gateway import get_authority_gateway

        get_authority_gateway().record_unreconciled_authority(
            executive_intent_id=getattr(auth, "executive_intent_id", None),
            capability_token_id=getattr(auth, "capability_token_id", None),
            standing_authority_token=getattr(auth, "standing_authority_token", None),
            reason=reason,
        )
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        # The gateway itself is unreachable. This is the one place left where
        # the record can only be a log line, and it says so.
        logging.getLogger(__name__).critical(
            "UNRECONCILED AUTHORITY (unrecordable): reason=%s intent=%s token=%s (%s)",
            reason,
            getattr(auth, "executive_intent_id", None),
            getattr(auth, "capability_token_id", None),
            exc,
        )


def _record_capability_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
    enforce_failure_policy: bool = True,
) -> None:
    record_degradation(
        "capability_engine",
        exc,
        severity=severity,
        action=action,
        enforce_failure_policy=enforce_failure_policy,
    )


_FIELD_DEFAULT_FACTORY_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)
_PARAMETER_COERCION_ERRORS = (TypeError, ValueError, json.JSONDecodeError)
_SCHEMA_RECOVERY_ERRORS = (ValidationError, TypeError, ValueError)


def _safe_field_default(field_name: str, field_obj: Any) -> tuple[Any, bool]:
    default_factory = getattr(field_obj, "default_factory", None)
    if default_factory is None:
        return None, False

    try:
        return default_factory(), True
    except _FIELD_DEFAULT_FACTORY_ERRORS as exc:
        _record_capability_degradation(
            exc,
            action=f"omitted invalid default factory value for parameter {field_name!r}",
            severity="warning",
        )
        return None, False


def _get_field_info(field_name: str, field_obj: Any) -> tuple[Any, Any, bool]:
    annotation = None
    default_val = None
    has_default = False

    if hasattr(field_obj, "annotation"):
        annotation = field_obj.annotation
        from pydantic_core import PydanticUndefined

        if field_obj.default is not PydanticUndefined:
            default_val = field_obj.default
            has_default = True
        else:
            default_val, has_default = _safe_field_default(field_name, field_obj)
    elif hasattr(field_obj, "type_"):
        annotation = field_obj.type_
        if field_obj.default is not None:
            default_val = field_obj.default
            has_default = True
        else:
            default_val, has_default = _safe_field_default(field_name, field_obj)

    return annotation, default_val, has_default


def _minimal_model_payload(fields: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    minimal: dict[str, Any] = {}
    for field_name, field_obj in fields.items():
        _, default_val, has_default = _get_field_info(field_name, field_obj)
        if has_default:
            minimal[field_name] = default_val
        elif field_name in params:
            minimal[field_name] = params[field_name]
    return minimal


def _humanize_skill_name(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return ""
    split_camel = re.sub(r"(?<!^)(?=[A-Z])", " ", raw)
    return re.sub(r"\s+", " ", split_camel.replace("_", " ")).strip().lower()


# Skill names that are also common conversational words: a bare mention must
# never dispatch the tool ("the clock in the kitchen is five minutes fast"
# dispatched the clock skill — July 8 overbreadth audit). These names keep
# only the explicit-invocation form; their skill-specific patterns still
# cover natural requests.
_COMMON_WORD_SKILL_NAMES = frozenset(
    {"clock", "listen", "speak", "curiosity", "personality", "memory", "notes", "timer"}
)


def _generic_skill_invocation_patterns(name: str) -> list[str]:
    variants = {
        str(name or "").strip().lower(),
        _humanize_skill_name(name),
    }
    bare_name_is_safe = str(name or "").strip().lower() not in _COMMON_WORD_SKILL_NAMES
    patterns: list[str] = []
    for variant in sorted(part for part in variants if part):
        escaped = re.escape(variant).replace(r"\ ", r"\s+")
        if bare_name_is_safe:
            patterns.append(rf"(?<![\w-]){escaped}(?![\w-])")
        patterns.append(rf"(?:use|run|call|invoke)\s+{escaped}(?![\w-])")
    return patterns


def _skill_class_name(name: str) -> str:
    """Convert `snake_case` skill ids into their exported class names."""
    return "".join(part.capitalize() for part in name.split("_")) + "Skill"


class SkillRequirements(BaseModel):
    """System and package requirements for a skill."""

    packages: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    supported_platforms: list[str] = Field(default_factory=lambda: ["linux", "darwin", "win32"])

    def check(self) -> tuple[bool, list[str]]:
        """Verifies if all requirements are met."""
        errors = []
        from core.container import ServiceContainer

        for pkg in self.packages:
            if not ServiceContainer.check_package(pkg):
                errors.append(f"Missing package: {pkg}")
        for cmd in self.commands:
            if shutil.which(cmd) is None:
                errors.append(f"Missing command: {cmd}")
        if sys.platform not in self.supported_platforms:
            errors.append(f"Unsupported platform: {sys.platform}")
        return len(errors) == 0, errors


def _get_base_types(annotation: Any) -> list[Any]:
    if annotation is None:
        return []
    import types
    from typing import Union, get_args, get_origin

    origin = get_origin(annotation)
    # Check if union type (typing.Union or PEP 604 | )
    if origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        args = get_args(annotation)
        types_list = []
        for arg in args:
            types_list.extend(_get_base_types(arg))
        return types_list

    if annotation is type(None):
        return []

    if origin is not None:
        return [origin]

    return [annotation]


def _coerce_and_harmonize_params(params: dict[str, Any], input_model: Any) -> dict[str, Any]:
    """Coerces parameter types and injects defaults from a Pydantic model class."""
    if not input_model or not isinstance(params, dict):
        return params

    # 1. Get fields map
    fields = {}
    if hasattr(input_model, "model_fields"):
        fields = input_model.model_fields
    elif hasattr(input_model, "__fields__"):
        fields = input_model.__fields__

    if not fields:
        return params

    healed = dict(params)

    # 2. Coerce existing params
    for name, val in list(healed.items()):
        if name not in fields:
            continue

        field_obj = fields[name]
        annotation, _, _ = _get_field_info(name, field_obj)
        if not annotation:
            continue

        target_types = _get_base_types(annotation)
        if not target_types:
            continue

        # If value already matches one of target types, keep it
        if any(isinstance(val, t) for t in target_types if isinstance(t, type)):
            continue

        # Otherwise, let's coerce!
        coerced = False
        coercion_error: BaseException | None = None
        for t in target_types:
            if coerced:
                break
            if not isinstance(t, type):
                continue

            try:
                if t is bool:
                    val_str = str(val).strip().lower()
                    if val_str in {"true", "yes", "1", "t", "y", "on"}:
                        healed[name] = True
                        coerced = True
                    elif val_str in {"false", "no", "0", "f", "n", "off", ""}:
                        healed[name] = False
                        coerced = True
                    elif isinstance(val, (int, float)):
                        healed[name] = bool(val)
                        coerced = True
                elif t is int:
                    if isinstance(val, float):
                        healed[name] = int(val)
                        coerced = True
                    elif isinstance(val, str):
                        cleaned = val.strip()
                        healed[name] = int(float(cleaned))
                        coerced = True
                elif t is float:
                    if isinstance(val, (int, str)):
                        healed[name] = float(val)
                        coerced = True
                elif t is str:
                    if isinstance(val, (dict, list)):
                        healed[name] = json.dumps(val)
                        coerced = True
                    else:
                        healed[name] = str(val)
                        coerced = True
                elif t is list:
                    if isinstance(val, str):
                        cleaned = val.strip()
                        if cleaned.startswith("[") and cleaned.endswith("]"):
                            healed[name] = json.loads(cleaned)
                            coerced = True
                        elif "," in cleaned:
                            healed[name] = [item.strip() for item in cleaned.split(",")]
                            coerced = True
                        else:
                            healed[name] = [cleaned]
                            coerced = True
                elif t is dict:
                    if isinstance(val, str):
                        cleaned = val.strip()
                        if cleaned.startswith("{") and cleaned.endswith("}"):
                            healed[name] = json.loads(cleaned)
                            coerced = True
            except _PARAMETER_COERCION_ERRORS as exc:
                coercion_error = exc
                continue

        if not coerced and coercion_error is not None:
            # A value that won't coerce (e.g. limit="not_an_int") is bad INPUT
            # gracefully handled by keeping the original — not a capability_engine
            # fault. Under a fail-closed policy + production governance this
            # warning would otherwise escalate to a CRITICAL service failure and
            # raise, spiking existential threat (the same July-2026 live pathology
            # already fixed at the sanitized-fallback site below).
            _record_capability_degradation(
                coercion_error,
                action=f"kept original value for parameter {name!r} after coercion failed",
                severity="warning",
                enforce_failure_policy=False,
            )

    # 3. Inject Defaults for missing keys
    for name, field_obj in fields.items():
        if name not in healed:
            _, default_val, has_default = _get_field_info(name, field_obj)
            if has_default:
                healed[name] = default_val

    return healed


class SkillMetadata(BaseModel):
    """Metadata and schema for a skill."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    skill_class: Any | None = None
    requirements: SkillRequirements = Field(default_factory=SkillRequirements)
    enabled: bool = True
    input_model: Any | None = None
    module_path: str | None = None
    class_name: str | None = None
    instance: Any | None = None
    metabolic_cost: int = 1
    is_core_personality: bool = False
    trigger_patterns: list[str] = Field(default_factory=list)
    effect_scope: str = "unknown"
    authority_class: str = "unclassified"
    schema_override: dict[str, Any] | None = None
    catalog_id: str = ""
    source_kind: str = "runtime"
    source_path: str = ""
    source_sha256: str = ""
    validation_state: str = "valid"
    validation_error: str | None = None
    dependency_ready: bool = True
    dependency_errors: list[str] = Field(default_factory=list)
    constructor_dependencies: list[str] = Field(default_factory=list)
    route_class_hint: str | None = None

    # 2026 Transcendence Fields
    execution_profile: str = "cpu"  # cpu, gpu, neural
    max_concurrent: int = 1
    timeout_seconds: float = 30.0
    memory_mb_estimate: int = 256

    @property
    def schema_def(self) -> dict[str, Any]:
        """Returns the JSON schema for the skill's input model."""
        if self.schema_override is not None:
            return dict(self.schema_override)
        if self.input_model and hasattr(self.input_model, "model_json_schema"):
            return dict(self.input_model.model_json_schema())
        return {"additionalProperties": True, "properties": {}, "type": "object"}

    def to_json_schema(self) -> dict[str, Any]:
        """Returns the OpenAI-compatible function definition for this skill."""
        return {"name": self.name, "description": self.description, "parameters": self.schema_def}

    async def extract_and_validate_args(self, params_raw: str, llm: Any) -> dict[str, Any]:
        """Validates raw JSON parameters against the skill's input model.

        If input_model is missing, returns the raw params.
        """
        import json

        # Input validation logic remains here, but AST auditing is moved to registration

        try:
            decoded = json.loads(params_raw)
            if not isinstance(decoded, dict):
                raise ValueError("skill parameters must decode to a JSON object")
            params: dict[str, Any] = decoded
            if not self.input_model:
                return params

            # 1. Self-healing Parameter Coercion & Schema Harmonization
            if isinstance(params, dict):
                params = _coerce_and_harmonize_params(params, self.input_model)

            # Simple validation if it's a Pydantic model
            if hasattr(self.input_model, "model_validate"):
                try:
                    return dict(self.input_model.model_validate(params).model_dump())
                except _SCHEMA_RECOVERY_ERRORS:
                    # 2. Non-destructive Recovery / Sanitized Subset fallback
                    sanitized = {}
                    fields = {}
                    if hasattr(self.input_model, "model_fields"):
                        fields = self.input_model.model_fields
                    elif hasattr(self.input_model, "__fields__"):
                        fields = self.input_model.__fields__

                    for field_name in fields:
                        if field_name in params:
                            sanitized[field_name] = params[field_name]

                    try:
                        return dict(self.input_model.model_validate(sanitized).model_dump())
                    except _SCHEMA_RECOVERY_ERRORS:
                        minimal = _minimal_model_payload(fields, params)
                        try:
                            return dict(self.input_model.model_validate(minimal).model_dump())
                        except _SCHEMA_RECOVERY_ERRORS as final_err:
                            # Classifier/user-supplied params that can't satisfy a
                            # skill schema (e.g. an image-gen dispatch with no
                            # prompt) is bad INPUT, not a capability_engine fault —
                            # the engine correctly rejected it. Recording it as a
                            # fail-closed degradation minted a CRITICAL incident and
                            # spiked existential threat (observed live, July 2026).
                            _record_capability_degradation(
                                final_err,
                                action="rejected unfillable skill parameters; returned sanitized fallback",
                                severity="warning",
                                enforce_failure_policy=False,
                            )
                            fallback_dict = {k: v for k, v in params.items() if k in fields}
                            fallback_dict["_error"] = f"Validation failed: {final_err}"
                            return fallback_dict

            return params
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            # Malformed parameter JSON from the model/user is bad input the
            # engine correctly rejected — return the raw-params fallback rather
            # than let a fail-closed policy convert it into a CRITICAL failure.
            _record_capability_degradation(
                e,
                action="returned raw skill parameters after argument validation failed",
                enforce_failure_policy=False,
            )
            # Fallback for complex extraction failures
            return {"raw_params": params_raw, "_error": str(e)}


# Arguments that turn an allowlisted binary into an arbitrary-code
# primitive. Comparing argv[0] alone says "python is allowed" and misses
# that `python -c "..."` is every command at once. Same for `bash -c`,
# `find -exec`, `awk 'BEGIN{system(...)}'`, `ssh host cmd`.
#
# CP126 named this precisely: "a populated list compares only argv[0].
# Arguments, subcommands, interpreter payloads, paths, environment effects,
# and shell-equivalent behavior are outside this policy boundary."
_INTERPRETER_ESCAPE_ARGS = frozenset({
    "-c", "-e", "--eval", "--exec", "-exec", "-execdir", "--command",
    "-mpython", "--python", "-i", "--interactive",
})

# Binaries whose entire purpose is to run something else. Allowlisting one
# of these allowlists everything it can reach, so they are refused unless
# the caller opts in by name.
_INDIRECT_EXECUTION_BINARIES = frozenset({
    "awk", "bash", "csh", "dash", "env", "eval", "fish", "gawk", "ksh",
    "nice", "nohup", "perl", "php", "python", "python2", "python3", "ruby",
    "sh", "ssh", "sudo", "tclsh", "time", "timeout", "xargs", "zsh",
})


class Shell:
    """Bounded shell execution.

    CP126 (high): "Shell policy defaults to allow all commands." An empty
    allowlist returned True for every executable — a policy object whose
    unconfigured state was total permission. Constructing a Shell without
    saying what it may run now permits nothing, which is the only safe
    reading of "no policy has been set".
    """

    def __init__(
        self,
        cwd: str,
        allowed_commands: list[str] | None = None,
        timeout: int = 30,
        *,
        allow_indirect_execution: bool = False,
    ):
        self.cwd = cwd
        self.allowed_commands = list(allowed_commands or [])
        self.timeout = timeout
        # Opt-in, because allowing an interpreter is allowing everything it
        # can run. A caller that genuinely needs `python -c` must say so.
        self.allow_indirect_execution = bool(allow_indirect_execution)

    @staticmethod
    def _binary_name(argv0: str) -> str:
        """The executable's own name, independent of how it was reached."""
        return os.path.basename(str(argv0 or "").strip()) or ""

    def _resolves_to_allowed(self, base_cmd: str, binary: str) -> bool:
        """Is this argv[0] genuinely one of the approved executables?

        Identity, not spelling. The old rule was
        `base_cmd == allowed or base_cmd.endswith("/" + allowed)`, which
        accepted /tmp/attacker/git for an allowlisted "git": the suffix
        matched and the binary was not the one anyone approved. Matching on
        basename alone has exactly the same hole, so an absolute path must
        BE the resolved system binary, not merely share its name.
        """
        import shutil

        resolved = os.path.realpath(base_cmd) if os.path.isabs(base_cmd) else ""
        for allowed in self.allowed_commands:
            entry = str(allowed or "").strip()
            if not entry:
                continue
            if os.path.isabs(entry):
                if resolved and resolved == os.path.realpath(entry):
                    return True
                continue
            if self._binary_name(entry) != binary:
                continue
            if not os.path.isabs(base_cmd):
                return True
            system_path = shutil.which(entry)
            if system_path and resolved == os.path.realpath(system_path):
                return True
        return False

    def _is_allowed(self, cmd: list[str]) -> bool:
        if not isinstance(cmd, (list, tuple)) or not cmd:
            return False
        # No allowlist means no permission. This used to mean total
        # permission, which inverted the meaning of an unconfigured policy.
        if not self.allowed_commands:
            return False

        base_cmd = str(cmd[0] or "")
        binary = self._binary_name(base_cmd)
        if not binary or not self._resolves_to_allowed(base_cmd, binary):
            return False

        if self.allow_indirect_execution:
            return True
        if binary in _INDIRECT_EXECUTION_BINARIES:
            return False
        for argument in cmd[1:]:
            if str(argument or "").strip().lower() in _INTERPRETER_ESCAPE_ARGS:
                return False
        return True

    async def run(self, cmd: list[str]) -> tuple[bool, str]:
        if not self._is_allowed(cmd):
            attempted = cmd[0] if isinstance(cmd, (list, tuple)) and cmd else "<empty>"
            if not self.allowed_commands:
                return False, "Shell has no command allowlist; refusing to execute"
            return False, f"Command {attempted} not permitted by shell policy"
        auth = None
        try:
            from core.executive.authority_gateway import get_authority_gateway

            gateway = get_authority_gateway()
            auth = await gateway.authorize_tool_execution(
                "shell_command",
                {"cmd": list(cmd), "cwd": self.cwd, "timeout": self.timeout},
                source="capability_engine.shell",
                priority=0.75,
                is_critical=False,
            )
            if not auth.approved:
                return False, f"Authority refused shell command: {auth.reason}"
            if not gateway.verify_tool_access("shell_command", auth.capability_token_id):
                return False, "Authority token verification failed for shell_command"
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as e:
            _record_capability_degradation(
                e,
                action="blocked shell command because authority gateway was unavailable",
                severity="degraded",
            )
            return False, f"Authority unavailable for shell command: {e}"
        try:
            result = await get_subprocess_gateway().run_async(
                cmd,
                cwd=self.cwd,
                timeout=self.timeout,
                capture_output=True,
                source="capability_engine.shell",
                accelerator_capability="auto",
            )
            # CP126 (critical): "The subprocess effect can occur,
            # finalization can fail, and the helper still returns the process
            # result as its authoritative outcome. The leaked token or open
            # executive intent is reduced to degradation telemetry with no
            # reconciliation."
            #
            # The command has already run — reporting its real result is
            # correct, and inventing a failure would be worse. What was
            # missing is that the authority leak became invisible: the
            # receipt naming exactly which of intent/token/lease stayed open
            # was discarded, and a raise left nothing recorded at all. Both
            # now reach the gateway's reconciliation queue.
            try:
                from core.executive.authority_gateway import get_authority_gateway

                get_authority_gateway().finalize_tool_execution(
                    executive_intent_id=getattr(auth, "executive_intent_id", None),
                    capability_token_id=getattr(auth, "capability_token_id", None),
                    standing_authority_token=getattr(auth, "standing_authority_token", None),
                    success=result.returncode == 0,
                    result={"returncode": result.returncode},
                )
            except (
                ImportError,
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
            ) as finalize_error:
                _record_unreconciled_authority(
                    auth,
                    reason=f"shell_finalize_raised:{type(finalize_error).__name__}",
                )
                _record_capability_degradation(
                    finalize_error,
                    action="returned shell result after authority finalization failed",
                    severity="critical",
                )
            return result.returncode == 0, (result.stdout + "\n" + result.stderr).strip()
        except (subprocess.TimeoutExpired, OSError, ValueError) as e:
            _record_capability_degradation(
                e,
                action="returned shell execution failure and finalized authority denial",
                severity="degraded",
            )
            try:
                from core.executive.authority_gateway import get_authority_gateway

                get_authority_gateway().finalize_tool_execution(
                    executive_intent_id=getattr(auth, "executive_intent_id", None),
                    capability_token_id=getattr(auth, "capability_token_id", None),
                    standing_authority_token=getattr(auth, "standing_authority_token", None),
                    success=False,
                    error=str(e),
                )
            except (
                ImportError,
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
            ) as finalize_error:
                _record_capability_degradation(
                    finalize_error,
                    action="returned shell execution failure after authority finalization failed",
                    severity="degraded",
                )
            return False, str(e)


class WebClient:
    """HTTP GET behind an explicit destination allowlist.

    CP126: "An empty domain allowlist permits every URL. The populated path
    compares raw netloc without scheme restrictions, canonical hostname
    handling, userinfo rejection, port policy, IDNA normalization, or
    DNS/IP rebinding guarantees."

    Two separate defects sat in five lines.

    *Empty meant everything.* ``if not self.allowed_domains: return True``.
    A caller that configured no destinations got unrestricted egress, which
    is the opposite of what supplying an allowlist parameter implies. The
    wildcard now has to be written down — ``["*"]``, which is exactly what
    ``config.security.allowed_domains`` already defaults to — so "allow
    everything" is a statement somebody made rather than a state nobody
    noticed.

    *netloc is not a hostname.* It carries userinfo and port, so
    ``https://good.com:8443/`` was refused for a permitted host and
    ``HTTPS://GOOD.COM`` was refused for a case difference. ``.hostname``
    gives the parsed host, lowercased, with userinfo and port removed.

    Scheme is checked too: an allowlist that says "good.com" is about web
    destinations, and ``file://`` or ``ftp://`` never satisfied that
    intention.

    DNS-rebinding is deliberately NOT claimed here — defeating it requires
    pinning the resolved address through to connect time, which belongs to
    the network gateway that actually opens the socket. This is a
    destination policy, and it says so rather than implying more.
    """

    #: Written-down wildcard. Matching the config default keeps the escape
    #: hatch explicit instead of implicit-by-emptiness.
    WILDCARD = "*"

    def __init__(self, allowed_domains: list[str] | None = None, timeout: int = 10):
        self.allowed_domains = allowed_domains or []
        self.timeout = timeout

    def _is_allowed(self, url: str) -> bool:
        from urllib.parse import urlparse

        if not self.allowed_domains:
            # Fail closed. A caller that named no destinations authorized no
            # destinations.
            return False
        if any(str(entry).strip() == self.WILDCARD for entry in self.allowed_domains):
            return True

        try:
            parsed = urlparse(url)
        except ValueError:
            return False
        if parsed.scheme.lower() not in {"http", "https"}:
            return False
        host = (parsed.hostname or "").strip().lower().rstrip(".")
        if not host:
            return False

        for entry in self.allowed_domains:
            allowed = str(entry or "").strip().lower().rstrip(".")
            if not allowed:
                continue
            if host == allowed or host.endswith("." + allowed):
                return True
        return False

    async def get(self, url: str, headers: dict[str, str] | None = None) -> tuple[bool, str]:
        if not self._is_allowed(url):
            return False, f"Domain not in allowlist: {url}"
        auth = None
        try:
            from core.executive.authority_gateway import get_authority_gateway

            gateway = get_authority_gateway()
            auth = await gateway.authorize_tool_execution(
                "network_get",
                {"url": url, "headers": sorted((headers or {}).keys()), "timeout": self.timeout},
                source="capability_engine.web_client",
                priority=0.65,
                is_critical=False,
            )
            if not auth.approved:
                return False, f"Authority refused network request: {auth.reason}"
            if not gateway.verify_tool_access("network_get", auth.capability_token_id):
                return False, "Authority token verification failed for network_get"
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as e:
            _record_capability_degradation(
                e,
                action="blocked network request because authority gateway was unavailable",
                severity="degraded",
            )
            return False, f"Authority unavailable for network request: {e}"
        try:
            response = await get_network_gateway().request_async(
                "GET",
                url,
                headers=headers,
                timeout=self.timeout,
                read_only=True,
                source="core.capability_engine.web_client",
            )
            ok = bool(response.get("ok"))
            try:
                from core.executive.authority_gateway import get_authority_gateway

                get_authority_gateway().finalize_tool_execution(
                    executive_intent_id=getattr(auth, "executive_intent_id", None),
                    capability_token_id=getattr(auth, "capability_token_id", None),
                    standing_authority_token=getattr(auth, "standing_authority_token", None),
                    success=ok,
                    result={"ok": ok, "status": response.get("status")},
                )
            except (
                ImportError,
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
            ) as finalize_error:
                # Same shape as the shell path: the request already went out,
                # so the response is the honest answer — but an unrevoked
                # capability token must not disappear into a log line.
                _record_unreconciled_authority(
                    auth,
                    reason=f"network_finalize_raised:{type(finalize_error).__name__}",
                )
                _record_capability_degradation(
                    finalize_error,
                    action="returned network response after authority finalization failed",
                    severity="critical",
                )
            if not ok:
                return False, str(response.get("error") or "network request failed")
            content = response.get("content", b"")
            text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)
            return True, text
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            _record_capability_degradation(
                e,
                action="returned network request failure and finalized authority denial",
                severity="degraded",
            )
            try:
                from core.executive.authority_gateway import get_authority_gateway

                get_authority_gateway().finalize_tool_execution(
                    executive_intent_id=getattr(auth, "executive_intent_id", None),
                    capability_token_id=getattr(auth, "capability_token_id", None),
                    standing_authority_token=getattr(auth, "standing_authority_token", None),
                    success=False,
                    error=str(e),
                )
            except (
                ImportError,
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
            ) as finalize_error:
                _record_capability_degradation(
                    finalize_error,
                    action="returned network request failure after authority finalization failed",
                    severity="degraded",
                )
            return False, str(e)


class Sandbox2:
    """Kernel-boundary sandbox for executing untrusted/forged code.

    CP126 (critical): "The RestrictedPython environment supplies raw getattr,
    arbitrary item access, and an identity write guard, then invokes
    generated code in-process with no OS isolation or resource boundary.
    These guards defeat the claimed untrusted execution."

    Every word of that was true. The three guards were::

        "_getattr_": getattr,                       # raw attribute access
        "_getitem_": lambda obj, key: obj[key],     # arbitrary items
        "_write_": lambda obj: obj,                 # identity write guard

    RestrictedPython rewrites attribute access to call ``_getattr_`` for the
    express purpose of interposing a check. Passing the builtin means the
    rewrite happens and then permits everything —
    ``().__class__.__mro__[1].__subclasses__()`` reaches ``os`` with no
    import statement. The identity ``_write_`` permits mutating any object
    the code can name. RestrictedPython ships ``safer_getattr``,
    ``guarded_getitem`` and ``full_write_guard`` for exactly this, and they
    were not used.

    There was also a second, larger hole. The engine built this only
    ``if RESTRICTED_AVAILABLE``, and the call site reads
    ``if is_forged and sandbox is not None`` — so on a host without
    RestrictedPython installed (including this one) a forged skill ran with
    no sandbox at all, while the log line above it said "Executing FORGED
    skill in Sandbox 2.0". A missing sandbox library silently became no
    sandbox.

    So execution now goes through ``core.sandbox.untrusted_python``: a
    Seatbelt/bwrap boundary with no network, no user-data reads, no exec,
    writes confined to a scratch directory, rlimits, and — the part that
    matters — a refusal when no boundary is available, instead of running
    the code anyway. RestrictedPython, when installed, is kept as an
    additional AST-level screen with its real guards.
    """

    def __init__(self, logger: Any):
        self.logger = logger
        self.restricted_available = RESTRICTED_AVAILABLE
        self.safe_globals: dict[str, Any] = {}
        if RESTRICTED_AVAILABLE:
            self.builtins = safe_builtins.copy()
            self.builtins.update(utility_builtins)
            self.builtins["_print_"] = PrintCollector
            self.safe_globals = {
                "__builtins__": self.builtins,
                "__name__": "aura_sandbox",
                "_getattr_": _RESTRICTED_GUARDS["getattr"],
                "_getitem_": _RESTRICTED_GUARDS["getitem"],
                "_write_": _RESTRICTED_GUARDS["write"],
            }

    def screen(self, code: str) -> None:
        """Optional AST-level screen. Raises when the code cannot compile."""
        if not self.restricted_available:
            return
        compile_restricted(code, filename="<aura_skill>", mode="exec")

    def execute(self, code: str, func_name: str, params: dict[str, Any]) -> Any:
        from core.sandbox.untrusted_python import call_untrusted_function

        try:
            # Cheap first filter when available; the kernel boundary below is
            # what actually holds when this is fooled or absent.
            self.screen(code)

            outcome = call_untrusted_function(
                code,
                func_name,
                [{"args": [], "kwargs": dict(params or {})}],
                source="capability_engine.sandbox2",
            )
            if outcome.status == "no_boundary":
                raise RuntimeError(
                    "refusing to execute forged code: no OS sandbox available "
                    f"({outcome.error})"
                )
            if outcome.status != "ok":
                raise RuntimeError(
                    f"forged code did not complete ({outcome.status}): "
                    f"{outcome.error or outcome.stderr}"
                )
            return outcome.results[0] if outcome.results else None
        except (RuntimeError, AttributeError, TypeError, ValueError, SyntaxError) as e:
            _record_capability_degradation(
                e,
                action="blocked forged skill execution after sandbox failure",
                severity="degraded",
            )
            self.logger.error("Sandbox Violation or Error: %s", e)
            raise


def _maturity_enforcement_enabled() -> bool:
    """Whether the capability-maturity gate REFUSES or merely observes.

    Off by default. See _maturity_gate for why: enforcing onto an ungraded
    surface refuses most autonomous work immediately, which gets a safety
    mechanism disabled rather than adopted.
    """
    import os

    return str(
        os.getenv("AURA_ENFORCE_CAPABILITY_MATURITY", "") or ""
    ).strip().lower() in {"1", "true", "yes", "on"}


def _is_transient(err: str) -> bool:
    """Checks if an error is likely transient (network, timeout, etc)."""
    return any(x in str(err).lower() for x in ["timeout", "network", "retry", "limit"])


@lru_cache(maxsize=2048)
def _skill_name_pattern(form: str) -> re.Pattern[str]:
    """Compiled once per name form — detect_intent runs on every turn."""
    return re.compile(rf"(?<!\w){re.escape(form)}(?!\w)")


def _declared_effect_scope(skill_name: str, target: Any | None = None) -> str:
    declared = str(getattr(target, "effect_scope", "") or "").strip().lower()
    policy = resolve_skill_policy(skill_name, declared)
    return policy.effect_scope if policy is not None else "unknown"


def _normalize_context_origin(origin: Any) -> str:
    normalized = str(origin or "").strip().lower().replace("-", "_")
    while normalized.startswith("routing_"):
        normalized = normalized[len("routing_") :]
    return normalized


#: Skills that report the world as it is RIGHT NOW. A self-contained problem
#: carries its own data and must never pull one of these in on a noun that is
#: only a prop in its story ("a clock strikes 6 times", "a file walks into a bar").
_REALTIME_SENSOR_SKILLS = frozenset(
    {"clock", "weather", "system_status", "screen_capture", "read_screen_text"}
)


class CapabilityEngine(AuraBaseModule):
    """Unified engine for Aura's capabilities (skills).

    Consolidates skill loading, discovery, registration, and resilient execution.
    """

    def __init__(self, orchestrator: Any = None):
        """Initializes the CapabilityEngine.

        Args:
            orchestrator: Reference to the system orchestrator.
        """
        super().__init__("CapabilityEngine")
        self.orchestrator = orchestrator
        self._catalog_lock = checked_lock(
            "capability.catalog", rank=_CATALOG_LOCK_RANK, reentrant=True
        )
        self._catalog_mutation_lock = checked_lock(
            "capability.catalog_mutation", rank=_CATALOG_MUTATION_LOCK_RANK, reentrant=True
        )
        self._skills: dict[str, SkillMetadata] = {}
        self._instances: dict[str, Any] = {}
        self._catalog_loaded = False
        self._catalog_load_lock = checked_lock(
            "capability.catalog_load", rank=_CATALOG_LOAD_LOCK_RANK, reentrant=True
        )
        self.quarantined_skills: dict[str, dict[str, Any]] = {}
        self.catalog_exclusions: list[dict[str, Any]] = []
        self.catalog_health: dict[str, Any] = {
            "ready": False,
            "reason": "catalog_not_loaded",
        }
        self._catalog_digest = ""
        self._skill_preflight_results: dict[str, dict[str, Any]] = {}
        self._catalog_preflight_summary: dict[str, Any] = {
            "catalog_digest": "",
            "complete": False,
            "entries": [],
            "failed": [],
            "live_count": 0,
            "ok": False,
            "reason": "not_run",
        }
        self._explicitly_deactivated_skills: set[str] = set()
        # skill → monotonic deadline while a user-advocate block holds.
        self._advocate_block_cooldowns: dict[str, float] = {}
        self.active_skills: set[str] = {
            # Core routing
            "ManageAbilities",
            "talk",
            "FinalResponse",
            # Self-awareness & diagnostics
            "system_proprioception",
            "environment_info",
            "clock",
            # Web & network (+ the offline lane that backstops it)
            "local_reference_search",
            "web_search",
            "sovereign_browser",
            "sovereign_terminal",
            "sovereign_network",
            "web_interlocutor",
            # File & memory
            "file_operation",
            "memory_ops",
            "memory_sync",
            # Sensory & output
            "query_visual_context",
            "sovereign_imagination",
            "speak",
            "listen",
            "sovereign_vision",
            "toggle_senses",
            # Code & compute
            "run_code",
            "internal_sandbox",
            "install_package",
            # Self-modification & evolution
            "self_repair",
            "self_evolution",
            "self_improvement",
            "auto_refactor",
            "train_self",
            "cognitive_trainer",
            "evolution_status",
            "program_dna_reconstruct",
            "program_dna_equivalence_battery",
            # OS & computer control
            "computer_use",
            "desktop_task",
            "os_manipulation",
            "os_automation",
            # Agency & autonomy
            "curiosity",
            "deploy_ghost_probe",
            "social_lurker",
            "delegate_shard",
            "inter_agent_comm",
            "spawn_agent",
            "spawn_agents_parallel",
            # Identity & personality
            "personality",
            "embodiment",
            # Knowledge & beliefs
            "add_belief",
            "query_beliefs",
            # Misc
            "manifest_to_device",
            "notify_user",
            "native_chat",
            "dream_sleep",
            "force_dream_cycle",
            "test_generator",
            "free_search",
            "uplink_local",
        }  # ALL skills active — Aura is fully sovereign
        self.skill_awoken_times: dict[str, float] = {}
        self.skill_states: dict[str, str] = {}  # READY, RUNNING, ERROR
        self.skill_last_errors: dict[str, str] = {}

        # Execution Config
        self.max_retries = 3
        self.retry_delay = 1.0
        self.timeout = 120.0

        # Dependencies
        self.temporal = getattr(orchestrator, "temporal", None)
        self.rosetta_stone = None
        # Always constructed. The call site reads "if is_forged and sandbox is
        # not None", so gating this on RESTRICTED_AVAILABLE meant a host
        # without that package ran forged skills with no sandbox at all while
        # logging that it was sandboxing them.
        self.sandbox = Sandbox2(self.logger)
        self._load_dependencies()

        # CP126: "CapabilityEngine initialization calls reload_skills
        # directly. Discovery, imports, source verification, dependency
        # probes, and trigger compilation can therefore block whichever
        # thread or event-loop path first constructs the service."
        #
        # That is a multi-second scan of ~130 source files, with imports and
        # dry runs, executed by whoever happened to construct the service —
        # including an async boot path, where it stalls the event loop for
        # the duration. CLAUDE.md carries the scar from the last time
        # something blocking ran on that loop.
        #
        # Construction no longer does it. The catalog loads on first access
        # through the skills/instances properties, so every existing caller
        # still sees a populated catalog the moment it looks, and the work
        # lands on the code path that actually wanted skills rather than on
        # whatever thread built the container.
        self.logger.info(
            "✓ CapabilityEngine online (catalog loads on first access)"
        )

    def _ensure_catalog_loaded(self) -> None:
        """Load the skill catalog once, on whoever asks for it first."""
        if self._catalog_loaded:
            return
        if self._catalog_guard().held_by_current_thread():
            # Loading takes catalog_load, which outranks the catalog guard this
            # thread already holds. Blocking here is the boot deadlock: the
            # loader waits for the guard while the guard's owner waits for the
            # loader. Serve the generation we have and say so, rather than
            # wedging the process.
            record_degradation(
                "capability_engine",
                RuntimeError(
                    "catalog load requested while the catalog guard was held; "
                    "call _ensure_catalog_loaded() before taking the guard"
                ),
                action="served the current catalog generation instead of inverting the lock order",
            )
            return
        with self._catalog_load_guard():
            if self._catalog_loaded:
                return
            self.reload_skills()
            self._initialize_skill_states()
            self._load_default_trigger_patterns()
            self.logger.info(
                "✓ CapabilityEngine catalog loaded with %d registered skills",
                len(self._skills),
            )

    @property
    def skills(self) -> dict[str, SkillMetadata]:
        self._ensure_catalog_loaded()
        return self._skills

    @skills.setter
    def skills(self, value: dict[str, SkillMetadata]) -> None:
        self._skills = value
        self._catalog_loaded = True

    @property
    def instances(self) -> dict[str, Any]:
        self._ensure_catalog_loaded()
        return self._instances

    @instances.setter
    def instances(self, value: dict[str, Any]) -> None:
        self._instances = value

    def _catalog_guard(self) -> CheckedLock:
        return self._lock_guard("_catalog_lock", "capability.catalog", _CATALOG_LOCK_RANK)

    def _catalog_mutation_guard(self) -> CheckedLock:
        return self._lock_guard(
            "_catalog_mutation_lock",
            "capability.catalog_mutation",
            _CATALOG_MUTATION_LOCK_RANK,
        )

    def _catalog_load_guard(self) -> CheckedLock:
        return self._lock_guard(
            "_catalog_load_lock", "capability.catalog_load", _CATALOG_LOAD_LOCK_RANK
        )

    def _lock_guard(self, attribute: str, name: str, rank: LockRank) -> CheckedLock:
        """Return one of the catalog locks, creating it if construction skipped it.

        Subclasses and unpickled instances have reached the catalog without
        running ``__init__``; the bootstrap keeps that from racing two callers
        into two different locks for the same attribute.
        """

        lock = getattr(self, attribute, None)
        if isinstance(lock, CheckedLock):
            return lock
        with _CATALOG_LOCK_BOOTSTRAP:
            lock = getattr(self, attribute, None)
            if not isinstance(lock, CheckedLock):
                lock = checked_lock(name, rank=rank, reentrant=True)
                setattr(self, attribute, lock)
            return lock

    def _load_default_trigger_patterns(self) -> None:
        """Comprehensive intent patterns covering all major skills."""
        patterns = {
            # ── Web / Search ──────────────────────────────────────────
            "web_search": [
                # noun-phrase mentions ("the search for a new apartment",
                # "the news about the library") must not dispatch a browser —
                # only ask/imperative shapes do (July 8 overbreadth audit).
                r"(?<!the )(?<!a )search (?:for|the web|online|the internet)",
                r"look up",
                r"find out",
                r"what is the price of",
                r"google",
                r"search query",
                r"find information",
                r"what(?:'s| is) (?:the latest|happening|new)",
                r"\b(?:what's|whats|any|latest|today's|current)\s+news\b",
                r"(?:find|get|check|show)(?:\s+me)?\s+(?:the\s+)?(?:latest\s+)?news\b",
                r"^news (?:about|on)\b",
                r"current (?:events|price|status)",
                r"research (?:about|on)",
            ],
            "free_search": [
                r"free search",
                r"duckduckgo",
                r"bing search",
                r"search without",
                r"anonymous search",
            ],
            "sovereign_browser": [
                r"open (?:a |the )?browser",
                r"open (?:a |the )?(?:webpage|website|page|tab|url)",
                r"navigate to",
                r"go to (?:https?://|www\.)",
                r"browse to",
                r"visit (?:the |this )?(?:site|page|url|website)",
                r"load (?:the |this )?(?:page|url|website)",
                r"open (?:gmail|youtube|github|reddit|twitter|linkedin)",
                r"pull up",
                r"show me (?:the |a )?(?:page|site|website)",
            ],
            "web_interlocutor": [
                r"(?:talk|chat|converse|hold a conversation) with (?:another )?(?:ai|assistant|chatbot|gemini|chatgpt|claude)",
                r"(?:open|go to).*(?:gemini|chatgpt|claude).*(?:talk|chat|ask|conversation)",
                r"(?:ask|message) (?:gemini|chatgpt|claude|another ai)",
                r"(?:learn|bring back|tell me what you learned).*(?:from|after).*(?:gemini|chatgpt|another ai|web chat)",
                r"(?:wait for|read) (?:their|its|the) repl(?:y|ies).*(?:respond|answer|continue)",
            ],
            # ── Computer / OS Control ────────────────────────────────
            "computer_use": [
                r"click (?:on|the)",
                r"type (?:in|into|this)",
                r"press (?:the |key )?(?:enter|tab|escape|ctrl|cmd)",
                r"scroll (?:down|up|to)",
                r"drag (?:and drop)?",
                r"right.?click",
                r"double.?click",
                r"keyboard shortcut",
                r"open (?:application|app|program|window|tab)",
                r"open (?:a |the )?(?:browser )?tab .*search",
                r"open .* on my computer",
                r"take (?:a )?screenshot",
                r"(?:move|position) (?:the )?(?:cursor|mouse)",
            ],
            "desktop_task": [
                r"use (?:my )?computer",
                r"do (?:this|that) on (?:my )?(?:computer|desktop|screen)",
                r"complete .* on (?:my )?(?:computer|desktop|screen)",
                r"(?:multi[- ]?step|chained|chain) .* (?:desktop|computer|app|screen)",
                r"(?:open|click|type|copy|paste|export|move).*(?:open|click|type|copy|paste|export|move)",
                r"(?:calculator|notes|finder|preview|browser).*(?:pdf|clipboard|file|folder|note)",
            ],
            "os_manipulation": [
                r"open (?:finder|explorer|terminal|file manager)",
                r"create (?:a )?(?:folder|directory|file)",
                r"delete (?:this )?(?:file|folder)",
                r"move (?:this )?(?:file|folder)",
                r"rename (?:this )?(?:file|folder)",
                r"list (?:files|directories|contents)",
                r"change (?:directory|folder)",
            ],
            "sovereign_terminal": [
                r"run (?:this )?(?:command|script|shell|terminal)",
                r"execute (?:this )?(?:command|script)",
                r"^\s*execute:\s*.+$",
                r"^\s*run:\s*.+$",
                r"^\s*terminal:\s*.+$",
                r"terminal command",
                r"bash ",
                r"shell ",
                r"zsh ",
                r"run in (?:the )?terminal",
                r"command line",
                r"(?:install|uninstall|update) (?:with )?(?:brew|pip|npm|apt|yarn)",
                r"sudo ",
                r"chmod ",
                r"git (?:commit|push|pull|clone|status)",
            ],
            # ── File Operations ───────────────────────────────────────
            "file_operation": [
                r"read (?:this |the )?file",
                r"write (?:to )?(?:this |a )?file",
                r"save (?:this )?(?:file|document|text|content)(?:\s+to|\s+as|\s+in)",
                r"open (?:this )?file",
                r"edit (?:the )?(?:file|document)",
                r"load (?:this )?file",
                r"contents of (?:the )?file",
                r"show (?:me )?(?:the )?file",
                r"append (?:to )?(?:the )?file",
                r"(?:(?:check|see|verify|test)\s+(?:if\s+)?|does\s+).+?\s+exist(?:s)?(?:\.|!|\?|$)",
            ],
            "manifest_to_device": [
                r"(?:manifest|save)\s*(?:this\s+(?:image|file|asset))?\s*(?:to\s+(?:my\s+)?(?:desktop|downloads|device))?:\s*https?://",
                r"save\s+to\s+(?:my\s+)?(?:desktop|downloads|device):\s*https?://",
                r"manifest\s+to\s+(?:my\s+)?(?:desktop|downloads|device):\s*https?://",
            ],
            # ── Memory / Knowledge ───────────────────────────────────
            "memory_ops": [
                r"remember",
                r"recall",
                r"last time",
                r"what did we talk about",
                r"what do you know about",
                r"from (?:our |the )?(?:last|previous|past) (?:conversation|chat|session)",
                r"did I (?:mention|tell you|say)",
                r"our history",
                r"save (?:this |that )?(?:to|in) memory",
                r"remember (?:this|that)",
                r"store (?:this|that)",
                r"commit (?:this|that) to memory",
                r"don't forget",
                r"make note of",
                r"remember .*future session",
                r"remember .*later",
                r"remember .*about me",
                r"remember that",
                r"store (?:this|that|it) (?:in|to)? ?memory",
                r"save (?:this|that|it) (?:for later|for future sessions|to memory)",
                r"don['’]t forget",
                r"what do you remember",
                r"what do you know about me",
                r"recall ",
                r"retrieve ",
            ],
            # ── Code / Compute ────────────────────────────────────────
            "run_code": [
                r"```(?:py|python)?\s+",
                r"run (?:this )?(?:code|script|python)",
                r"python\s*:",
                r"code\s*:",
                r"evaluate (?:this )?(?:expression|code|formula)",
                r"(?:calculate|compute)\s+[-+*/%().,\d\s]+(?:$|[?.!])",
                r"what is\s+[-+*/%().,\d\s]+(?:$|[?.!])",
                r"what(?:'s| is) (?:the )?(?:square root|cube root|sqrt|factorial|product|sum|difference|quotient) of\b",
                r"(?:multiply|divide)\s+[-+]?\d+(?:\.\d+)?\s+by\s+[-+]?\d+(?:\.\d+)?",
                r"add\s+[-+]?\d+(?:\.\d+)?\s+(?:and|to)\s+[-+]?\d+(?:\.\d+)?",
                r"subtract\s+[-+]?\d+(?:\.\d+)?\s+from\s+[-+]?\d+(?:\.\d+)?",
                r"\b(?:square root|cube root|sqrt|factorial)\b\s*(?:of)?\s*\d+",
                r"solve (?:this )?(?:equation|formula)",
                r"execute (?:this )?(?:code|script|python)",
            ],
            "coding_skill": [
                r"write (?:a |the )?(?:function|class|script|program|module|code)",
                r"implement (?:this|a|the)",
                r"create (?:a |the )?(?:function|class|script|program)",
                r"code (?:up|this)",
                r"program (?:this|a)",
            ],
            # ── Voice / Embodiment ───────────────────────────────────
            "speak": [
                r"say (?:this|that|it) (?:out loud|aloud|to me)",
                r"read (?:this|that) (?:out loud|aloud|to me)",
                r"speak (?:this|that|it)",
                r"tell me (?:out loud|aloud)",
                r"voice (?:this|that|it)",
            ],
            "listen": [
                r"listen (?:to me|for)",
                r"start (?:listening|dictation)",
                r"voice (?:input|recognition)",
                r"transcribe (?:what I say|my voice)",
                r"speech to text",
            ],
            "embodiment": [
                r"(?:discover|find|list|show|inspect) (?:my )?(?:physical )?(?:devices|sensors|actuators|hardware)",
                r"(?:connect|attach) (?:to )?(?:this |the |a )?(?:device|sensor|actuator|hardware)",
                r"(?:read|query|focus on|pay attention to) (?:this |the |my )?(?:sensor|physical channel|device)",
                r"(?:turn on|turn off|set|control|command) (?:this |the |my )?(?:light|thermostat|fan|switch|relay|physical device)",
                r"what (?:physical )?(?:devices|sensors|actuators|hardware) (?:can you|do you) (?:see|use|control|have)",
            ],
            # ── Self / Identity ───────────────────────────────────────
            "self_repair": [
                r"repair (?:yourself|your code)",
                r"heal (?:yourself|your code)",
                r"fix (?:the )?bug",
                r"debug (?:yourself|your code)",
                r"patch (?:yourself|your code)",
            ],
            "self_improvement": [
                r"get (?:smarter|better|faster)",
                r"learn (?:from this|more)",
                r"improve (?:your|own) (?:intelligence|reasoning|capabilities)",
                r"self.?learn",
                r"train (?:yourself|on this)",
            ],
            "build_app": [
                r"build (?:me )?(?:a |an )?[\w\s-]{0,30}?(?:app|game|tool|widget)",
                r"(?:make|create|write) (?:me )?(?:a |an )?[\w\s-]{0,30}?(?:app|game)",
                r"recreate (?:my |the )?[\w\s-]{0,30}?(?:app|game)",
                r"(?:checkers|chess|tic.?tac.?toe|snake|calculator|pong) (?:app|game)",
            ],
            "program_dna_reconstruct": [
                r"program dna",
                r"reconstruct (?:this |that |the )?(?:program|app|application|software|tool)",
                r"reverse engineer (?:this |that |the )?(?:program|app|application|software|tool)",
                # named host binaries / commands: "reverse engineer base64", "reconstruct the md5 command", "reverse engineer jq"
                r"reverse.?engineer(?:\s+(?:this|that|the))?\s+(?:base64|md5|rev|jq|\w+\s+(?:command|binary|utility|cli))",
                r"reconstruct(?:\s+(?:this|that|the))?\s+(?:base64|md5|rev|jq)\b",
                r"clean.?room (?:clone|rebuild|implementation|reconstruction)",
                r"rebuild (?:this |that |the )?(?:program|app|application|software|tool)",
                r"copy (?:the )?(?:behavior|features|ui|ux) of (?:this |that |the )?(?:program|app|application|software|tool)",
                r"extract (?:the )?(?:behavior|affordances|features|dna) (?:from|of)",
            ],
            "program_dna_equivalence_battery": [
                r"program dna (?:equivalence|battery|behavioral proof|hidden.?source)",
                r"hidden.?source (?:program|software|behavioral) (?:test|battery|proof|equivalence)",
                r"behavioral equivalence (?:battery|test|proof)",
                r"test (?:program|software|app) reconstruction (?:equivalence|behavior)",
                r"held.?out (?:tests?|cases?) (?:against|for) (?:the )?(?:original|replacement)",
            ],
            # ── Screen / Vision ───────────────────────────────────────
            "query_visual_context": [
                r"what(?:'s| is) on (?:my |the )?screen",
                r"look at (?:this|my screen)",
                r"camera feed",
                r"read (?:the )?screen",
                r"what do you see",
                r"describe (?:what(?:'s| is)|the screen|this image)",
            ],
            "sovereign_vision": [
                r"use (?:the )?(?:camera|vision)",
                r"computer vision",
                r"analyze (?:this )?(?:image|screenshot|photo)",
                r"read (?:this )?(?:image|screenshot|photo)",
            ],
            # ── Personality / Curiosity ───────────────────────────────
            "curiosity": [
                r"explore (?:this|that|the topic|further)",
                r"dig deeper",
                r"I(?:'m| am) curious",
                r"what more",
                r"tell me more about",
                r"investigate",
                r"research (?:this|that)",
            ],
            # ── Image Generation ──────────────────────────────────────
            "sovereign_imagination": [
                r"(?:generate|create|draw|make|produce|render|paint|design|visualize)\s+(?:an?\s+)?(?:image|picture|photo|artwork|illustration|portrait|painting|drawing)",
                r"(?:i\s+want|can\s+you|please)\s+(?:to\s+)?(?:see|generate|create|draw|make)\s+(?:an?\s+)?(?:image|picture|photo|artwork)",
                r"neon cat|cyberpunk cat",
            ],
            # ── System / Info ─────────────────────────────────────────
            "system_proprioception": [
                r"how is your (?:health|status|memory|cpu|ram|temperature)",
                r"how are your (?:memory|cpu|ram|temperature|vitals|stats)",
                r"system status",
                r"how much (?:memory|ram|cpu|disk)",
                r"your (?:vitals|health|stats)",
                r"are you (?:okay|running (?:well|smoothly))",
            ],
            "environment_info": [
                r"what(?:'s| is) (?:the weather|temperature) (?:in|at|for)",
                r"weather forecast",
                r"where am I",
                r"current (?:location|timezone)",
                r"what(?:'s| is) my (?:timezone|location)",
                r"what (?:environment|system) am I (?:in|on)",
            ],
            "clock": [
                r"what time",
                r"current time",
                r"what(?:'s| is) the time",
                r"what(?:'s| is) (?:the )?date",
                r"what day is it",
                r"what(?:'s| is) my timezone",
                r"current timezone",
                r"set (?:an? )?(?:alarm|timer|reminder)",
                r"timer for",
                r"remind me (?:in|at|to)",
            ],
            # ── Notifications ─────────────────────────────────────────
            "notify_user": [
                r"notify (?:me|the user)",
                r"send (?:a )?notification",
                r"alert (?:me|the user)",
                r"ping me",
                r"send (?:a )?message to",
            ],
            # ── Social / Network ──────────────────────────────────────
            "social_lurker": [
                r"check (?:twitter|reddit|hackernews|hn|social media)",
                r"what(?:'s| is) trending",
                r"check (?:the )?feed",
                r"lurk (?:on|in)",
                r"monitor (?:twitter|reddit|social)",
            ],
            "sovereign_network": [
                r"(?:make|send) (?:an? )?(?:http|api) (?:request|call)",
                r"fetch (?:from|the) (?:api|url|endpoint)",
                r"POST to",
                r"GET (?:from|the) api",
                r"call (?:the )?(?:api|endpoint|service)",
            ],
            # ── Misc ─────────────────────────────────────────────────
            "dream_sleep": [
                r"go to sleep",
                r"sleep (?:mode|now)",
                r"rest (?:now|mode)",
                r"take a (?:break|nap)",
                r"go dormant",
            ],
            "install_package": [
                r"install (?:package|library|module|dependency)",
                r"pip install",
                r"npm install",
                r"brew install",
            ],
            "ManageAbilities": [
                r"(?:enable|disable|toggle) (?:skill|ability|feature|capability)",
                r"turn (?:on|off) (?:your )?(?:skill|ability|feature)",
                r"what (?:skills|abilities|capabilities) (?:do you have|can you use)",
                r"list (?:your )?(?:skills|abilities|capabilities)",
            ],
            "mcp_client": [
                r"connect (?:to )?(?:an? )?mcp server",
                r"use mcp",
                r"query mcp",
                r"model context protocol",
                r"call mcp",
                r"discover mcp tools",
            ],
            "manim_renderer": [
                r"render (?:a )?manim",
                r"create (?:a )?manim",
                r"animate (?:with )?manim",
                r"generate (?:a )?math video",
                r"dynamic blackboard",
                r"render animation",
            ],
            "branching_futures": [
                r"branching future",
                r"ghost thread",
                r"fork state",
                r"create (?:a )?sandbox clone",
                r"try this safely",
                r"experimental run",
            ],
            # ── Tool Parity Skills ────────────────────────────────────
            "code_repl": [
                r"run (?:this )?(?:python )?code",
                r"execute (?:this )?(?:python )?(?:code|script)",
                r"python repl",
                r"code interpreter",
                r"code execution",
                r"calculate (?:this|that|it)",
                r"compute (?:this|that|it)",
                r"evaluate (?:this )?expression",
                r"test (?:this )?snippet",
            ],
            # Image-gen triggers need a generation-shaped OBJECT, not a bare
            # verb: the old "paint (?:me )?(?:an? )?" had an all-optional tail,
            # so ANY sentence containing "paint " dispatched the diffusion
            # skill (seen live: "the paint color I chose" → image_gen crash
            # mid-conversation). Verbs alone only count in the imperative
            # "verb me a/an ..." form, which is unambiguous.
            "image_gen": [
                r"\b(?:generate|create|make|produce|render)\s+(?:me\s+)?(?:an?\s+|some\s+)?"
                r"(?:image|picture|photo|illustration|artwork|logo|icon|wallpaper|portrait|sketch|drawing|painting)s?\b",
                r"\b(?:draw|paint|sketch|illustrate)\s+me\s+an?\s+\w+",
                r"\b(?:draw|paint|sketch)\s+an?\s+"
                r"(?:image|picture|portrait|scene|landscape|diagram|illustration|logo)\b",
                r"\b(?:imagine|visualize)\s+and\s+(?:draw|render|generate|paint)\b",
                r"\bedit (?:this )?image\b",
                r"\bstyle transfer\b",
                r"\bimg2img\b",
                r"\btext[- ]to[- ]image\b",
            ],
            "x_tools": [
                r"search (?:twitter|x\.com|tweets)",
                r"find (?:on )?(?:twitter|x)",
                r"twitter (?:search|thread|trends)",
                r"fetch (?:this )?tweet",
                r"get (?:this )?thread",
                r"trending (?:on )?(?:twitter|x)",
                r"tweet engagement",
                r"twitter analytics",
                r"extract (?:tweet )?media",
            ],
            "render_bridge": [
                r"render (?:this|inline|citation)",
                r"display (?:chart|table|image|code)",
                r"show (?:progress|visualization)",
                r"embed (?:image|file|chart)",
                r"format (?:as )?(?:table|chart|card)",
            ],
            "voice_output": [
                r"say (?:this|that)",
                r"speak (?:this|that|aloud)",
                r"text to speech",
                r"read (?:this )?(?:aloud|out loud)",
                r"synthesize (?:speech|voice|audio)",
                r"generate (?:speech|voice|audio)",
                r"narrate (?:this|that)",
                r"tts",
                r"voice (?:output|synthesis)",
            ],
        }
        # MCP routing was `r"use mcp"` and four variants — so a connector was
        # reachable only by naming the protocol, never by naming the thing.
        # "check my Sentry issues" routed nowhere even with a Sentry connector
        # configured, because the word "mcp" was the whole interface.
        #
        # These come from what is ACTUALLY configured, so the routing cannot
        # advertise a connector that does not exist and cannot go stale in the
        # other direction either.
        patterns.setdefault("mcp_client", []).extend(self._connector_trigger_patterns())
        for name, pats in patterns.items():
            if name in self.skills:
                self.skills[name].trigger_patterns.extend(pats)

    def _connector_trigger_patterns(self) -> list[str]:
        """One pattern per configured MCP connector, by its own name.

        Re-derived from the registry on every catalog load rather than
        hard-coded, which is what keeps this from becoming another
        handwritten routing table that drifts from reality. `list_connectors`
        reads the registry live and is never stale; routing patterns refresh
        when the catalog reloads.
        """
        try:
            from core.capabilities.mcp_connectors import available_connectors

            return [
                rf"\b{re.escape(connector.name)}\b"
                for connector in available_connectors()
                if connector.name.strip()
            ]
        except (ImportError, OSError, ValueError) as exc:
            _record_capability_degradation(
                exc,
                action="routed no MCP connectors because the registry was unreadable",
                severity="warning",
            )
            return []

        for name, meta in self.skills.items():
            for pattern in _generic_skill_invocation_patterns(name):
                if pattern not in meta.trigger_patterns:
                    meta.trigger_patterns.append(pattern)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _skill_name_forms(name: str) -> tuple[tuple[str, bool], ...]:
        """How a registered skill name gets written when someone names it.

        Returns (form, distinctive) pairs. A distinctive form is one that does
        not occur in ordinary prose by accident — an identifier spelling
        ("search_web"), a run-together CamelCase name ("manageabilities"), or a
        multi-word expansion ("query visual context"). A single common word
        ("clock", "listen", "speak") is NOT distinctive and needs an invocation
        cue in front of it, or every turn mentioning the time would summon the
        clock skill.
        """
        raw = str(name or "").strip()
        if len(raw) < 4:
            return ()
        spaced = re.sub(
            r"[_\-]+", " ", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", raw)
        ).strip().lower()
        distinctive = "_" in raw or "-" in raw or len(spaced.split()) > 1
        forms = {(raw.lower(), distinctive)}
        if spaced and spaced != raw.lower():
            forms.add((spaced, distinctive))
        return tuple(sorted(forms))


    def _explicitly_named_skills(self, text: str) -> list[str]:
        """Skills the turn names outright, rather than describing.

        LIVE, 2026-08-10: 37 of 76 registered skills carry NO trigger patterns —
        improve_own_code, grounded_search, local_reference_search,
        knowledge_base, internal_sandbox, train_self, world_forge among them —
        so intent detection could never select them by any phrasing. Naming one
        outright did not work either, because matching was against trigger
        phrases only and no skill's name was a trigger for itself. Half the
        catalog was addressable only by whatever code held a hardcoded string.

        Naming a tool is the least ambiguous request there is, so it is matched
        here directly. The mention-vs-request guard still applies upstream: this
        answers "which skill is named", not "is this turn asking for anything".
        """
        haystack = str(text or "")
        if not haystack.strip():
            return []
        named: list[str] = []
        for name, meta in self.skills.items():
            if not meta.enabled:
                continue
            for form, distinctive in self._skill_name_forms(name):
                match = _skill_name_pattern(form).search(haystack)
                if not match:
                    continue
                if distinctive or _SKILL_INVOCATION_CUE_RE.search(haystack[: match.start()]):
                    named.append(name)
                    break
        return named

    def _declaration_vocabulary(self) -> dict[str, tuple[frozenset[str], frozenset[str]]]:
        """What every enabled skill says it does, rebuilt when the roster moves."""
        from core.intent.declared_capability import declared_vocabulary

        roster = tuple(
            sorted(
                (name, str(getattr(meta, "description", "") or ""))
                for name, meta in self.skills.items()
                if getattr(meta, "enabled", True)
            )
        )
        cached = getattr(self, "_declaration_vocabulary_cache", None)
        if cached is not None and cached[0] == roster:
            return cached[1]
        vocabulary = {name: declared_vocabulary(name, text) for name, text in roster}
        self._declaration_vocabulary_cache = (roster, vocabulary)
        self._declaration_objects_cache = None
        return vocabulary

    def _declaration_matched_skills(self, message: str) -> list[str]:
        """Skills whose own declaration answers this request."""
        from core.intent.declared_capability import (
            distinctive_objects,
            rank_declaration_matches,
        )

        try:
            vocabulary = self._declaration_vocabulary()
            if not vocabulary:
                return []
            objects = getattr(self, "_declaration_objects_cache", None)
            if objects is None:
                objects = distinctive_objects(vocabulary)
                self._declaration_objects_cache = objects
            ranked = rank_declaration_matches(message, vocabulary, objects)
            strongest = ranked[0][1] if ranked else 0.0
            return [name for name, score in ranked if score == strongest]
        except Exception as exc:  # noqa: BLE001 - reported, never silent
            record_degradation(
                "capability_engine",
                exc,
                severity="warning",
                action="matched intent on trigger patterns alone",
            )
            return []

    def _foundational_candidates(self, message: str) -> list[str]:
        """The capabilities every request-shaped turn needs, whatever its nouns.

        Reading, computing and looking things up are the primitives a computer
        task is built from, and lexical matching cannot find them: "read
        README.md" names nothing any skill declares. Shared with the tool
        handoff so the router and the loop cannot disagree about what a turn
        needs.
        """
        try:
            from core.intent.capability_selection import (
                DEFAULT_CAPABILITY_SET,
                select_capabilities,
            )
            from core.phases.response_contract import requested_effect_ceiling

            ceiling, scopes = requested_effect_ceiling(message)
            return select_capabilities(
                message,
                self.skills,
                ceiling=ceiling,
                admissible_scopes=scopes,
                limit=DEFAULT_CAPABILITY_SET,
            )
        except Exception as exc:  # noqa: BLE001 - reported, never silent
            record_degradation(
                "capability_engine",
                exc,
                severity="debug",
                action="routed without the foundational capabilities",
                enforce_failure_policy=False,
            )
            return []

    def detect_intent(self, message: str) -> list[str]:
        """Aura's 'Cognitive Proprioception': Detects which skills match the user's intent."""
        triggered = []
        msg = normalize_memory_intent_text(message)
        skip_web_search = self._looks_like_search_capability_question(message)

        # Trigger patterns match WORDS, and several of them contain proper
        # nouns — `r"(?:ask|message) (?:gemini|chatgpt|claude|another ai)"`.
        # So naming a thing was enough to route a turn at it, and "What do you
        # think of ChatGPT?" dispatched a browser session. The distinction is
        # grammatical, not a property of which name was used: is the thing the
        # object of an instruction, or the subject of a remark?
        #
        # `_looks_like_search_capability_question` is the same fix made once,
        # for one skill, after "the search for a new apartment" opened a
        # browser. This is that fix made generally, so the next skill that gets
        # a proper noun in a pattern inherits it instead of needing its own.
        mentions_without_asking = self._is_mention_rather_than_request(message)

        for name, meta in self.skills.items():
            if not meta.enabled:
                continue
            canonical_name = self.resolve_skill_name(name)
            if skip_web_search and canonical_name in _SEARCH_SKILL_NAMES:
                continue
            if mentions_without_asking:
                continue
            for pattern in meta.trigger_patterns:
                if re.search(pattern, msg):
                    triggered.append(name)
                    break

        # Naming a skill is a request for it. 37 of 76 registered skills have no
        # trigger patterns at all, so without this they cannot be selected by
        # intent under any phrasing — including the ones a person is most likely
        # to ask for by name, like improve_own_code.
        # A phrase list can only match the phrasings someone thought of. Live
        # 2026-08-19, four of five ordinary ways to ask for code execution
        # missed while `code_repl` sat READY, and 37 registered skills have no
        # patterns at all, so no phrasing whatsoever could reach them. The
        # declaration each skill already carries says what it does; reading the
        # request against that needs nothing maintained in a second place.
        #
        # Additive on purpose: the patterns above still decide everything they
        # already decided, and this only ever adds a candidate they missed.
        # One question, one answer.
        #
        # Two mechanisms decided "which capabilities does this turn need" —
        # this router, and the tool handoff's `derive_capability_set` — and
        # they disagreed. Live 2026-08-19, a request to read a repository and
        # find a failing test nominated `uplink_local` (its description
        # mentions a state-repository) and omitted `file_operation`, so the
        # router picked a skill that could not do the job while the handoff
        # had the right five. The foundational capabilities a request-shaped
        # turn needs are the same set in both places, so they come from the
        # same place.
        if not mentions_without_asking:
            for name in self._foundational_candidates(msg):
                if name not in triggered:
                    triggered.append(name)

        if not mentions_without_asking:
            for name in self._declaration_matched_skills(msg):
                if name in triggered:
                    continue
                if skip_web_search and self.resolve_skill_name(name) in _SEARCH_SKILL_NAMES:
                    continue
                triggered.append(name)

        if not mentions_without_asking:
            for name in self._explicitly_named_skills(msg):
                if name in triggered:
                    continue
                if skip_web_search and self.resolve_skill_name(name) in _SEARCH_SKILL_NAMES:
                    continue
                triggered.append(name)
        if self._looks_like_reasoning_time_problem(msg):
            triggered = [
                name
                for name in triggered
                if self.resolve_skill_name(name) != "clock"
            ]

        def _promote(skill_name: str) -> None:
            self._ensure_catalog_loaded()
            with self._catalog_guard():
                skill_registered = skill_name in self._skills
            if not skill_registered:
                return
            if skill_name in triggered:
                triggered[:] = [skill_name] + [name for name in triggered if name != skill_name]
            else:
                triggered.insert(0, skill_name)

        if re.match(r"^\s*(?:execute|run|terminal)\s*:\s*\S", msg):
            _promote("sovereign_terminal")

        if re.search(
            r"(?:manifest|save)\s*(?:this\s+(?:image|file|asset))?\s*(?:to\s+(?:my\s+)?(?:desktop|downloads|device))?:\s*https?://",
            msg,
        ):
            _promote("manifest_to_device")
            triggered = [
                name
                for name in triggered
                if self.resolve_skill_name(name) != "file_operation" or name == "manifest_to_device"
            ]

        if re.search(
            r"(?:(?:check|see|verify|test)\s+(?:if\s+)?|does\s+).+?\s+exist(?:s)?(?:\.|!|\?|$)", msg
        ):
            _promote("file_operation")

        if re.search(r"\bresearch\s+(?:about|on)\b", msg) and not skip_web_search:
            _promote("web_search")
        return triggered

    @staticmethod
    def _is_mention_rather_than_request(message: str) -> bool:
        """True when the turn talks about something instead of asking for it.

        Fails OPEN — an unavailable classifier must not silently stop every
        capability in the runtime from being reachable.
        """
        try:
            from core.conversation.request_mood import (
                names_a_thing_without_asking_for_it,
            )

            return bool(names_a_thing_without_asking_for_it(message))
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("capability_engine.request_mood", exc, severity="warning")
            return False

    @staticmethod
    def _looks_like_reasoning_time_problem(message: str) -> bool:
        """Return True for time/clock wording used as a reasoning task, not live time I/O."""
        msg = normalize_memory_intent_text(message)
        if not msg:
            return False
        realtime_markers = (
            "what time is it",
            "what's the time",
            "current time",
            "current date",
            "today's date",
            "what day is it",
            "my timezone",
            "current timezone",
        )
        if any(marker in msg for marker in realtime_markers):
            return False
        reasoning_markers = (
            "<answer>",
            "solve",
            "calculate",
            "compute",
            "word problem",
            "logic puzzle",
            "riddle",
            "final answer",
            "how many seconds",
            "how many minutes",
            "how many hours",
            "clock strikes",
            "clock strike",
            "take to strike",
        )
        return any(marker in msg for marker in reasoning_markers)

    def _retrieved_tool_candidates(self, objective: str, max_tools: int) -> list[str]:
        """Skills the retriever finds relevant that the trigger patterns missed.

        Registers the catalog with the retriever on first use — a provider
        rather than a push, so the index re-reads the catalog when it changes
        instead of every registration site having to remember to update it.

        Never raises. Retrieval is an enrichment of tool selection, and a
        selection path that can fail because a ranking helper failed is worse
        than one that occasionally proposes fewer candidates.
        """
        if not objective.strip():
            return []
        try:
            from core.skills.skill_retrieval import SkillDocument, get_skill_retriever

            retriever = get_skill_retriever()
            retriever.register_provider(
                "capability_catalog",
                lambda: [
                    SkillDocument(
                        name=name,
                        description=str(getattr(meta, "description", "") or ""),
                        source="catalog",
                    )
                    for name, meta in self.skills.items()
                ],
            )
            return [
                hit.name
                for hit in retriever.retrieve(objective, k=max_tools)
                if hit.name in self.skills
            ]
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "capability_engine",
                exc,
                severity="debug",
                action="ranked tools without semantic retrieval",
            )
            return []

    _WORD_PROBLEM_ASK = re.compile(
        r"\b(?:how (?:many|much|long|far|fast|old)|what (?:is|are|will|would)\b"
        r"|find the|calculate|solve for|at what (?:time|rate|speed))\b",
        re.IGNORECASE,
    )
    _ABOUT_THE_USERS_MACHINE = re.compile(
        r"\b(?:my|your|the)\s+(?:screen|display|desktop|clipboard|file|files|folder|"
        r"directory|window|browser|tab|notes|calendar|email|inbox)\b"
        r"|\bright now\b|\bcurrently\b|\bwhat time is it\b",
        re.IGNORECASE,
    )

    @classmethod
    def _is_self_contained_word_problem(cls, text: str) -> bool:
        """True when the turn supplies its own quantities and asks for another.

        The heuristic tool rules match bare substrings, so "A clock strikes 6
        times in 5 seconds" reached for the realtime clock, and "a file walks
        into a bar" would reach for file_operation. The noun is a prop in the
        story, not a thing to go and look at.

        The distinguishing property is not which noun appears — that set is
        unbounded and every new tool adds to it — but where the DATA comes
        from. A word problem carries its own numbers and asks for a derived
        one; it needs arithmetic, not a sensor. A question about the machine
        asks for a reading, and is excluded here even when it contains numbers.
        """

        body = str(text or "")
        if not body.strip():
            return False
        if cls._ABOUT_THE_USERS_MACHINE.search(body):
            return False
        if not cls._WORD_PROBLEM_ASK.search(body):
            return False
        # Two or more given quantities is what makes it self-contained: one
        # number can be a reference ("open tab 2"), a pair is a problem.
        return len(re.findall(r"\b\d+(?:\.\d+)?\b", body)) >= 2

    def _rank_tool_candidates(
        self,
        *,
        objective: str = "",
        required_skill: str | None = None,
        matched_skills: Iterable[str] | None = None,
        max_tools: int = 8,
        available_only: bool = False,
    ) -> list[str]:
        """Return relevant tool names for the current turn, ranked by likely utility."""
        max_tools = max(1, min(int(max_tools or 8), 16))
        objective_text = str(objective or "").strip()
        objective_lower = normalize_memory_intent_text(objective_text)
        skip_web_search = self._looks_like_search_capability_question(objective_text)
        required = self.resolve_skill_name(required_skill) if required_skill else None
        if skip_web_search and required in _SEARCH_SKILL_NAMES:
            required = None

        matched = [
            self.resolve_skill_name(name)
            for name in (self.detect_intent(objective_text) if objective_text else [])
            if not (
                skip_web_search
                and self.resolve_skill_name(name)
                in {
                    "web_search",
                    "search_web",
                    "free_search",
                    "grounded_search",
                    "sovereign_browser",
                }
            )
        ]
        for name in matched_skills or ():
            resolved = self.resolve_skill_name(name)
            if not resolved:
                continue
            if skip_web_search and resolved in {
                "web_search",
                "search_web",
                "free_search",
                "grounded_search",
                "sovereign_browser",
            }:
                continue
            matched.append(resolved)

        heuristic_candidates: list[str] = []
        heuristic_rules = (
            (
                ("latest", "news", "price", "search", "look up", "find online"),
                ("web_search", "search_web", "free_search", "grounded_search"),
            ),
            (("remember", "recall", "memory", "future sessions"), ("memory_ops", "memory_sync")),
            (
                ("time", "clock", "date"),
                tuple() if self._looks_like_reasoning_time_problem(objective_lower) else ("clock",),
            ),
            (("browser", "website", "navigate", "open url", "webpage"), ("sovereign_browser",)),
            (
                ("open tab", "new tab", "on my computer", "on my screen"),
                ("computer_use", "os_manipulation"),
            ),
            (("terminal", "shell", "command", "cli"), ("sovereign_terminal", "computer_use")),
            (
                ("click", "type", "screen", "desktop", "mouse", "keyboard"),
                ("desktop_task", "computer_use", "os_manipulation"),
            ),
            (
                ("clipboard", "copy", "paste", "export", "pdf", "notes app", "calculator"),
                ("desktop_task", "computer_use"),
            ),
            (
                ("file", "directory", "folder", "read file", "write file", "repo", "code"),
                ("file_operation", "computer_use"),
            ),
            (("nethack", "game", "dungeon", "action", "move"), ("execute_nethack_action",)),
        )
        # A self-contained word problem must not pull an environment sensor in
        # on a noun that is only a prop in its story.
        word_problem = self._is_self_contained_word_problem(objective_text)
        for tokens, names in heuristic_rules:
            if word_problem and not (set(names) & {"run_code", "calculator"}):
                continue
            if skip_web_search and any(
                name in {"web_search", "search_web", "free_search", "grounded_search"}
                for name in names
            ):
                continue
            if any(token in objective_lower for token in tokens):
                heuristic_candidates.extend(names)

        # A screen OBSERVATION has exactly one owner.
        #
        # Live 2026-08-04: "can you tell me what you see on the screen?"
        # matched two rules — "on my screen" and "screen" — so the candidate
        # list carried both computer_use and desktop_task. Both dispatched,
        # each declared its own intention and minted its own capability
        # token, and each ran the SAME read_screen_text. One request, two
        # governed desktop actions, two readings of the person's screen,
        # double the latency, and two entries in the audit trail for one act.
        #
        # desktop_task is the governed lane: it plans, verifies effects,
        # produces receipts, and types the result as an Observation. A bare
        # computer_use read duplicates that with less. So for an observation
        # the governed lane owns it, and computer_use stays available for
        # everything else.
        # Reading a file is an observation, and none of the actuation skills
        # can perform one. Live 2026-08-19, "read the code at <path> and work
        # out why the test is failing" nominated desktop_task, which spent
        # 37 seconds per attempt failing to verify an effect no screen would
        # ever show — while file_operation sat READY with a read action. The
        # screen case below is the same rule, written after the same mistake.
        try:
            from core.runtime.desktop_objective_intent import (
                looks_like_filesystem_observation,
            )

            if looks_like_filesystem_observation(objective_lower):
                heuristic_candidates = [
                    name
                    for name in heuristic_candidates
                    if name not in {"desktop_task", "os_automation", "computer_use",
                                    "os_manipulation", "pursue_on_screen"}
                ]
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "capability_engine",
                exc,
                severity="debug",
                action="left the actuation skills as candidates for a file read",
                enforce_failure_policy=False,
            )

        if "desktop_task" in heuristic_candidates and "computer_use" in heuristic_candidates:
            try:
                from core.runtime.desktop_objective_intent import (
                    looks_like_screen_observation,
                )

                if looks_like_screen_observation(objective_lower):
                    heuristic_candidates = [
                        name for name in heuristic_candidates if name != "computer_use"
                    ]
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                record_degradation(
                    "capability_engine",
                    exc,
                    severity="debug",
                    action=(
                        "left both desktop skills as candidates; a screen read may "
                        "be dispatched twice"
                    ),
                )

        # Retrieval runs last and only adds. The trigger patterns above are
        # authored per skill and only fire on phrasings somebody anticipated; a
        # capability asked for in unlisted words was not ranked low, it never
        # entered the list. Appending here widens the field without displacing
        # anything the existing path would have chosen, so it cannot regress a
        # working turn — see core/skills/skill_retrieval.py.
        retrieved_candidates = self._retrieved_tool_candidates(objective_text, max_tools)

        ordered: list[str] = []

        def _push(name: str | None) -> None:
            if not name:
                return
            resolved = self.resolve_skill_name(name)
            if resolved not in self.skills or resolved in ordered:
                return
            if available_only:
                state = str(self.skill_states.get(resolved, "READY") or "READY")
                active = resolved in self.active_skills
                meta = self.skills.get(resolved)
                if (
                    not meta
                    or not meta.enabled
                    or not active
                    or state == "ERROR"
                    or meta.validation_state != "valid"
                    or not meta.dependency_ready
                ):
                    return
            ordered.append(resolved)

        _push(required)
        for name in matched:
            _push(name)
        for name in heuristic_candidates:
            _push(name)
        # After the authored paths, before the cost-ordered filler below. A
        # skill the objective is actually about should outrank one chosen for
        # being cheap.
        for name in retrieved_candidates:
            _push(name)

        if not ordered:
            fallback_names = (
                ("web_search", "memory_ops")
                if self._looks_like_reasoning_time_problem(objective_lower)
                else ("web_search", "memory_ops", "clock")
            )
            for fallback_name in fallback_names:
                _push(fallback_name)

        if len(ordered) < max_tools:
            skip_realtime_clock = self._looks_like_reasoning_time_problem(objective_lower)
            for name, meta in sorted(
                self.skills.items(),
                key=lambda item: (item[1].metabolic_cost, item[0]),
            ):
                if len(ordered) >= max_tools:
                    break
                if name in ordered:
                    continue
                if skip_realtime_clock and self.resolve_skill_name(name) == "clock":
                    continue
                if getattr(meta, "metabolic_cost", 1) > 2:
                    continue
                if available_only:
                    state = str(self.skill_states.get(name, "READY") or "READY")
                    active = name in self.active_skills
                    if (
                        not meta.enabled
                        or not active
                        or state == "ERROR"
                        or meta.validation_state != "valid"
                        or not meta.dependency_ready
                    ):
                        continue
                ordered.append(name)

        # One exclusion, applied where every path converges.
        #
        # skip_realtime_clock guarded only the cost-ordered filler, so "A clock
        # strikes 6 times in 5 seconds" still surfaced the realtime clock —
        # semantic retrieval had already added it upstream. Guarding one
        # contributor means the next contributor reintroduces the bug, so the
        # rule belongs on the result rather than on each source of it.
        if word_problem or self._looks_like_reasoning_time_problem(objective_lower):
            ordered = [name for name in ordered if name not in _REALTIME_SENSOR_SKILLS]

        return ordered[:max_tools]

    def select_tool_definitions(
        self,
        *,
        objective: str = "",
        required_skill: str | None = None,
        max_tools: int = 8,
        requested: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return a bounded, relevance-ranked tool subset for agentic LLM calls.

        This is intentionally narrower than `get_tool_definitions()` so local
        tool-using models do not waste context, latency, and reasoning budget on
        the full skill catalog.
        """
        max_tools = max(1, min(int(max_tools or 8), 12))
        ordered = self._rank_tool_candidates(
            objective=objective,
            required_skill=required_skill,
            max_tools=max_tools,
        )
        allowed_max_cost = self._allowed_max_tool_cost()
        asked_for = {str(name) for name in (requested or ())} | (
            {str(required_skill)} if required_skill else set()
        )
        # An empty ranking is not an empty answer when the turn named what it
        # needs. This returned [] before reaching the requested set, so a turn
        # whose working set was decided correctly was offered nothing at all:
        # "skill=improve_own_code,build_app,internal_sandbox,http_request,
        # code_repl offered=NONE (no tool definition)".
        if not ordered and not asked_for:
            return []
        selected: list[dict[str, Any]] = []
        # What the turn asked for comes first and is fetched by name. Ranking
        # a set that was already decided is how build_app survived selection
        # and vanished one call later.
        for name in [*sorted(asked_for), *ordered]:
            if len(selected) >= max_tools:
                break
            if any(
                str((entry.get("function") or {}).get("name")) == name for entry in selected
            ):
                continue
            tool = self._tool_definition_for_skill(
                name,
                allowed_max_cost=allowed_max_cost,
                requested=name in asked_for,
            )
            if tool:
                selected.append(tool)
        return selected

    def _load_dependencies(self) -> None:
        """Loads optional dependencies for adaptation and security."""
        try:
            from core.adaptation.rosetta_stone import rosetta_stone

            self.rosetta_stone = rosetta_stone
        except ImportError:
            self.logger.debug("Rosetta Stone not found, skipping adaptivity.")

    async def check_package(self, package_name: str, auto_install: bool = False) -> bool:
        """Proxy to ServiceContainer.check_package."""
        from core.container import ServiceContainer

        return bool(ServiceContainer.check_package(package_name, auto_install=auto_install))

    #: Synchronous lifecycle hooks a skill may declare to release resources.
    #: Checked in order; the first one present is used.
    _INSTANCE_SHUTDOWN_HOOKS = ("shutdown", "close", "cleanup")

    def _retire_superseded_instances(
        self, superseded: list[tuple[str, Any]]
    ) -> None:
        """Release resources held by instances the reload replaced.

        CP126: "Reload replaces catalog state and clears the instance cache
        without an ownership handoff or asynchronous shutdown contract.
        Skills holding sessions, subprocesses, file handles, or background
        tasks can leak while stateful instances are discarded."

        Only instances whose class actually declares a lifecycle hook are
        touched. Most skills declare none and are pure garbage — calling
        speculative teardown on them would be inventing a contract. The ones
        that DO declare a hook are precisely the ones holding something worth
        releasing, which is why the leak was worth closing.

        Best effort by design: a reload must not fail because a discarded
        instance refused to die. Every failure is recorded rather than
        raised.

        Honest caveat: an execution still in flight on a superseded instance
        can observe its resources closing. That is a smaller and louder
        problem than the previous behaviour, where the subprocess or session
        simply outlived the object forever with nothing recording it.
        """
        for name, instance in superseded:
            hook = None
            for attribute in self._INSTANCE_SHUTDOWN_HOOKS:
                candidate = getattr(instance, attribute, None)
                if callable(candidate):
                    hook = candidate
                    break
            if hook is None:
                continue
            try:
                outcome = hook()
                if inspect.isawaitable(outcome):
                    # A coroutine returned from a sync reload cannot be
                    # awaited here. Schedule it if a loop is running;
                    # otherwise say plainly that it did not close, rather
                    # than dropping the coroutine and reporting a clean
                    # retirement.
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        outcome.close()
                        _record_capability_degradation(
                            RuntimeError(
                                f"skill {name!r} declares an async shutdown but the "
                                "reload ran with no event loop; resources not released"
                            ),
                            action="left an async skill shutdown unrun during sync catalog reload",
                            severity="warning",
                        )
                    else:
                        get_task_tracker().create_task(
                            outcome, name=f"skill_shutdown_{name}"
                        )
                self.logger.debug("Retired superseded skill instance %r", name)
            except _INSTANCE_RETIREMENT_ERRORS as exc:
                _record_capability_degradation(
                    exc,
                    action=f"continued catalog reload after {name!r} shutdown failed",
                    severity="warning",
                )
                self.logger.warning(
                    "Superseded skill %r did not shut down cleanly: %s", name, exc
                )

    async def on_stop_async(self) -> None:
        """Release every prepared skill instance owned by this engine."""

        with self._catalog_mutation_guard():
            with self._catalog_guard():
                owned = list(self._instances.items())
                self._instances = {}
                self.active_skills = set()
                self._catalog_preflight_summary = {
                    "catalog_digest": self._catalog_digest,
                    "complete": False,
                    "entries": [],
                    "failed": [],
                    "live_count": len(self._skills),
                    "ok": False,
                    "reason": "engine_stopped",
                }

        failures: list[str] = []
        retired_ids: set[int] = set()
        for name, instance in owned:
            if id(instance) in retired_ids:
                continue
            retired_ids.add(id(instance))
            hook = next(
                (
                    candidate
                    for attribute in self._INSTANCE_SHUTDOWN_HOOKS
                    if callable(candidate := getattr(instance, attribute, None))
                ),
                None,
            )
            if hook is None:
                continue
            try:
                outcome = hook()
                if inspect.isawaitable(outcome):
                    await outcome
            except _INSTANCE_RETIREMENT_ERRORS as exc:
                failures.append(f"{name}:{type(exc).__name__}:{exc}")
                _record_capability_degradation(
                    exc,
                    action=f"continued capability shutdown after {name!r} cleanup failed",
                    severity="degraded",
                )

        if failures:
            raise RuntimeError(
                "skill instance shutdown failures: " + "; ".join(failures[:12])
            )

    async def _execute_forged(
        self, meta: Any, skill_name: str, params: dict[str, Any], sandbox: Any
    ) -> dict[str, Any]:
        """Run a forged skill's verified region under the kernel boundary.

        Three things changed here and each was a defect on its own.

        The callable is now the module-level ``run`` rather than the class name.
        ``call_untrusted_function`` looks the name up in the executed module and
        calls it, so passing a class constructed the class and never reached any
        method — with the skill's parameters spliced in as constructor keywords.

        The source is the *verified region*, recovered by
        :func:`~core.skill_management.forged_artifact.load_verified_region` and
        checked against the digest the forge recorded. The whole file cannot run
        under the boundary: it imports ``BaseSkill``, and the repository is not
        readable from inside the sandbox.

        A skill in the writable tree with no ledger entry is refused. Its
        provenance is unknown, there is no digest to check it against, and
        running it would mean the directory's contents decide what executes.
        """
        from core.skill_management.forged_artifact import get_forge_ledger, load_verified_region

        source_path = str(getattr(meta, "source_path", "") or "")
        if not source_path:
            raise RuntimeError(f"forged skill '{skill_name}' has no source file on record")

        entry = get_forge_ledger().entry_for(skill_name)
        if entry is None:
            return {
                "ok": False,
                "error": (
                    f"Refusing to run '{skill_name}': it is in the forged skill tree "
                    "but has no verification record."
                ),
                "status": "forged_unverified",
            }

        resolved = Path(source_path)
        if not resolved.is_absolute():
            resolved = Path(config.paths.base_dir) / source_path
        code = await asyncio.to_thread(
            load_verified_region, resolved, expected_digest=entry.digest
        )
        result = await asyncio.get_running_loop().run_in_executor(
            None, lambda: sandbox.execute(code, FORGED_ENTRYPOINT, params)
        )
        return result if isinstance(result, dict) else {"ok": True, "result": result}

    async def _record_forge_outcome(self, skill_name: str, result: dict[str, Any]) -> None:
        """Fold a real call into the forge ledger's reliability counts.

        Verification says a skill worked once, under probes its own drafter
        chose. This is the only channel that says whether it keeps working on
        the inputs it actually meets.
        """
        from core.skill_management.forged_artifact import get_forge_ledger

        try:
            await get_forge_ledger().record_outcome_async(
                skill_name, succeeded=bool(result.get("ok"))
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            _record_capability_degradation(
                exc,
                action="executed a forged skill but failed to record its outcome",
                severity="warning",
            )

    def reload_skills(self) -> None:
        """Serialize catalog mutations while allowing readers to use the live generation."""

        with self._catalog_mutation_guard():
            self._reload_skills_transaction()
        self._republish_runtime_macros()

    def _republish_runtime_macros(self) -> None:
        """Put learned macros back after a reload rebuilt the catalog from source.

        ``_reload_skills_transaction`` reconstructs ``_skills`` from discovery,
        which drops every runtime registration. Macros are registered rather
        than declared in source, so without this they stop being callable the
        first time anything reloads the catalog — a forge, a hot update, a
        health probe — and nothing would report it, because the library still
        holds them and still describes them.

        Outside the mutation guard: publication calls back into
        ``register_skill``, which takes that guard itself.
        """
        try:
            library = optional_service("skill_library", default=None)
            if library is not None and hasattr(library, "publish_all"):
                published = library.publish_all()
                if published:
                    self.logger.info("🔧 Republished %d learned macro(s)", published)
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            _record_capability_degradation(
                exc,
                action="reloaded the catalog without republishing learned macros",
                severity="warning",
            )

    def _reload_skills_transaction(self) -> None:
        """Build, probe, and atomically publish the governed live skill catalog."""
        self.logger.info("🔄 Refreshing skill registry...")
        next_skills: dict[str, SkillMetadata] = {}
        next_instances: dict[str, Any] = {}
        next_quarantined: dict[str, dict[str, Any]] = {}
        next_exclusions: list[dict[str, Any]] = []
        next_skill_errors: dict[str, str] = {}

        try:
            from core.skills.discovery import build_skill_catalog, validate_skill_catalog

            catalog = build_skill_catalog()
        except (ImportError, OSError, RuntimeError, SyntaxError, TypeError, ValueError) as exc:
            _record_capability_degradation(
                exc,
                action="failed closed because the canonical skill source catalog could not be built",
                severity="degraded",
            )
            with self._catalog_guard():
                self.catalog_health = {
                    **self.catalog_health,
                    "ready": False,
                    "reason": "catalog_build_failed",
                    "error": f"{type(exc).__name__}: {exc}"[:1200],
                    "live_count": len(self._skills),
                    "serving_last_known_good": bool(self._skills),
                }
            return

        try:
            validations = validate_skill_catalog(
                catalog,
                project_root=Path(config.paths.base_dir),
            )
        except (ImportError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            _record_capability_degradation(
                exc,
                action="quarantined the skill catalog after isolated validation failed",
                severity="degraded",
            )
            validations = {
                declaration.catalog_id: {
                    "catalog_id": declaration.catalog_id,
                    "error": f"isolated catalog probe failed: {type(exc).__name__}: {exc}"[:1200],
                    "stage": "probe_process",
                    "status": "quarantined",
                }
                for declaration in catalog.accepted
            }

        next_exclusions = [declaration.to_dict() for declaration in catalog.excluded]
        for issue in catalog.blocking_issues:
            issue_id = hashlib.sha256(
                json.dumps(issue.to_dict(), sort_keys=True).encode()
            ).hexdigest()[:20]
            next_quarantined[f"issue:{issue_id}"] = {
                **issue.to_dict(),
                "catalog_id": f"issue:{issue_id}",
                "name": issue.class_name or issue.code,
                "stage": "source_catalog",
                "status": "quarantined",
            }

        for declaration in catalog.accepted:
            validation = dict(validations.get(declaration.catalog_id) or {})
            if validation.get("status") != "valid":
                quarantine = {
                    **declaration.to_dict(),
                    **validation,
                    "description": declaration.description,
                    "effect_scope": declaration.effect_scope,
                    "authority_class": declaration.authority_class,
                    "status": "quarantined",
                }
                next_quarantined[declaration.catalog_id] = quarantine
                continue

            requirements_payload = dict(validation.get("requirements") or {})
            supported_platforms = list(requirements_payload.get("supported_platforms") or ())
            requirements = SkillRequirements(
                packages=list(requirements_payload.get("packages") or ()),
                commands=list(requirements_payload.get("commands") or ()),
                supported_platforms=(
                    supported_platforms
                    if supported_platforms
                    else ["linux", "darwin", "win32"]
                ),
            )
            dependency_ready = bool(validation.get("dependency_ready", True))
            dependency_errors = [
                str(error) for error in validation.get("dependency_errors") or ()
            ]
            next_skills[declaration.name] = SkillMetadata(
                name=declaration.name,
                description=str(validation.get("description") or declaration.description),
                requirements=requirements,
                module_path=declaration.module_path,
                class_name=declaration.class_name,
                effect_scope=declaration.effect_scope,
                authority_class=declaration.authority_class,
                schema_override=dict(validation.get("input_schema") or {}),
                catalog_id=declaration.catalog_id,
                source_kind=declaration.source_kind,
                source_path=declaration.source_path,
                source_sha256=declaration.source_sha256,
                validation_state="valid",
                dependency_ready=dependency_ready,
                dependency_errors=dependency_errors,
                constructor_dependencies=list(declaration.constructor_dependencies),
                route_class_hint=str(validation.get("route_class") or "managed_async"),
                execution_profile=str(validation.get("execution_profile") or "cpu"),
                timeout_seconds=float(validation.get("timeout_seconds") or 30.0),
                memory_mb_estimate=max(
                    1, int(validation.get("memory_mb_estimate") or 256)
                ),
                metabolic_cost=max(0, int(validation.get("metabolic_cost") or 1)),
                is_core_personality=bool(validation.get("is_core_personality", False)),
            )
            if not dependency_ready:
                next_skill_errors[declaration.name] = (
                    "; ".join(dependency_errors) or "declared dependencies unavailable"
                )

        expected_names = {declaration.name for declaration in catalog.accepted}
        missing_live = sorted(expected_names - set(next_skills) - {
            str(item.get("name") or "") for item in next_quarantined.values()
        })
        ready = bool(catalog.ok and not next_quarantined and not missing_live and next_skills)
        next_catalog_health = {
            **catalog.snapshot(),
            "dependency_degraded_count": sum(
                1 for metadata in next_skills.values() if not metadata.dependency_ready
            ),
            "excluded_declarations": [
                {
                    "class_name": item.class_name,
                    "module_path": item.module_path,
                    "name": item.name,
                    "reason": item.exclusion_reason,
                }
                for item in catalog.excluded
            ],
            "expected_live_count": len(expected_names),
            "live_count": len(next_skills),
            "missing_live": missing_live,
            "quarantined_count": len(next_quarantined),
            "ready": ready,
            "reason": "ready" if ready else "catalog_incomplete",
            "serving_last_known_good": False,
        }

        with self._catalog_guard():
            superseded_instances = [
                (name, instance)
                for name, instance in (self._instances or {}).items()
                if next_instances.get(name) is not instance
            ]
            self._skills = next_skills
            self._instances = next_instances
            self._catalog_loaded = True
            self.quarantined_skills = next_quarantined
            self.catalog_exclusions = next_exclusions
            self.skill_states = {name: "READY" for name in next_skills}
            self.skill_last_errors = next_skill_errors
            self._catalog_digest = catalog.digest
            self.catalog_health = next_catalog_health
            self._skill_preflight_results = {}
            self._catalog_preflight_summary = {
                "catalog_digest": catalog.digest,
                "complete": False,
                "entries": [],
                "failed": [],
                "live_count": len(next_skills),
                "ok": False,
                "reason": "not_run",
            }
            self._refresh_active_skills()
            live_count = len(self._skills)
            quarantined_count = len(self.quarantined_skills)
            exclusion_count = len(self.catalog_exclusions)

        # Retire what the new catalog replaced. Outside the catalog lock: a
        # skill's shutdown may block on a socket or a subprocess, and holding
        # the catalog guard through that would stall every reader.
        self._retire_superseded_instances(superseded_instances)

        if self.orchestrator and hasattr(self.orchestrator, "status") and self.orchestrator.status:
            try:
                self.orchestrator.status.skills_loaded = live_count
            except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                _record_capability_degradation(
                    _exc,
                    action="continued after orchestrator skill count status update failed",
                )
                self.logger.debug("Suppressed Exception: %s", _exc)
        self.logger.info(
            "✓ %d skills live, %d quarantined, %d explicitly superseded (%s)",
            live_count,
            quarantined_count,
            exclusion_count,
            catalog.backend,
        )

    def _refresh_active_skills(self) -> None:
        """Treat enabled, registered skills as active unless explicitly deactivated."""
        with self._catalog_guard():
            if not self._skills:
                self.active_skills = set()
                return

            registered = set(self._skills)
            enabled = {name for name, meta in self._skills.items() if bool(meta.enabled)}
            sticky_active = {name for name in self.active_skills if name in registered}
            self.active_skills = (enabled | sticky_active) - self._explicitly_deactivated_skills

    def register_skill(self, implementation: Any, *, replace: bool = False) -> bool:
        """Atomically add one runtime implementation to the current catalog generation."""

        # Load first, then add. Registering into a catalog that has not loaded
        # yet writes into a generation the first later read replaces: the lazy
        # load rebuilds _skills from discovery and the registration is simply
        # gone. This used to be masked because _refresh_active_skills read the
        # lazy `skills` property from inside the guard, which loaded the
        # catalog as a side effect of the very inversion that deadlocked boot.
        # Removing the inversion made the latent defect reachable.
        self._ensure_catalog_loaded()
        with self._catalog_mutation_guard(), self._catalog_guard():
            return self._register_skill_locked(implementation, replace=replace)

    def _register_skill_locked(self, implementation: Any, *, replace: bool = False) -> bool:
        """Register one executable, classified runtime skill.

        Metadata-only mappings are intentionally rejected: advertising a name
        without an implementation created catalog entries that could never run.
        """

        from core.skills.base_skill import BaseSkill

        if isinstance(implementation, dict):
            raise TypeError("runtime skill registration requires a class or BaseSkill instance")
        instance = None if inspect.isclass(implementation) else implementation
        skill_class = implementation if inspect.isclass(implementation) else implementation.__class__
        if not inspect.isclass(skill_class) or not issubclass(skill_class, BaseSkill):
            raise TypeError("runtime skill must inherit the canonical BaseSkill")
        if instance is not None and not isinstance(instance, BaseSkill):
            raise TypeError("runtime skill instance does not satisfy canonical BaseSkill")

        target = instance or skill_class
        skill_name = str(getattr(target, "name", "") or "").strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", skill_name):
            raise ValueError(f"invalid runtime skill name: {skill_name!r}")
        description = str(getattr(target, "description", "") or "").strip()
        if not description:
            raise ValueError(f"runtime skill {skill_name!r} requires a description")
        policy = resolve_skill_policy(skill_name, str(getattr(target, "effect_scope", "") or ""))
        if policy is None:
            raise ValueError(f"runtime skill {skill_name!r} requires a recognized effect_scope")

        existing = self.skills.get(skill_name)
        if existing is not None and not replace:
            same_class = existing.skill_class is skill_class and (
                instance is None or existing.instance is instance
            )
            if same_class:
                return True
            raise ValueError(f"runtime skill name {skill_name!r} is already registered")

        requirements = getattr(target, "requirements", SkillRequirements())
        if not isinstance(requirements, SkillRequirements):
            requirements = SkillRequirements(
                packages=list(getattr(requirements, "packages", ()) or ()),
                commands=list(getattr(requirements, "commands", ()) or ()),
                supported_platforms=list(
                    getattr(requirements, "supported_platforms", ())
                    or ("linux", "darwin", "win32")
                ),
            )
        input_model = getattr(target, "input_model", None)
        schema_override: dict[str, Any] | None = None
        if input_model is not None and callable(getattr(input_model, "model_json_schema", None)):
            schema_override = dict(input_model.model_json_schema())
        elif instance is not None:
            schema_factory = getattr(instance, "to_json_schema", None)
            if callable(schema_factory):
                raw_schema = schema_factory()
                if not isinstance(raw_schema, dict):
                    raise ValueError(f"runtime skill {skill_name!r} returned a non-object schema")
                function_payload = raw_schema.get("function")
                schema_payload = (
                    function_payload.get("parameters")
                    if isinstance(function_payload, dict)
                    else raw_schema
                )
                if not isinstance(schema_payload, dict):
                    raise ValueError(
                        f"runtime skill {skill_name!r} returned non-object parameters"
                    )
                schema_override = dict(schema_payload)
        if schema_override is None:
            schema_override = {"additionalProperties": True, "properties": {}, "type": "object"}

        identity = f"runtime:{skill_class.__module__}:{skill_class.__name__}:{skill_name}"
        self.skills[skill_name] = SkillMetadata(
            name=skill_name,
            description=description,
            skill_class=skill_class,
            requirements=requirements,
            input_model=input_model,
            module_path=getattr(skill_class, "__module__", None),
            class_name=getattr(skill_class, "__name__", None),
            instance=instance,
            metabolic_cost=max(0, int(getattr(target, "metabolic_cost", 1) or 1)),
            is_core_personality=bool(getattr(target, "is_core_personality", False)),
            effect_scope=policy.effect_scope,
            authority_class=policy.authority_class,
            schema_override=schema_override,
            catalog_id=hashlib.sha256(identity.encode()).hexdigest()[:20],
            source_kind="runtime",
            validation_state="valid",
            route_class_hint=(
                "async"
                if inspect.iscoroutinefunction(getattr(target, "execute", None))
                else "sync"
            ),
        )

        # Issue 51: Perform AST validation at registration time
        self._audit_skill_ast(skill_name)

        if instance:
            self.instances[skill_name] = instance
        self.logger.debug("Registered: %s", skill_name)
        # Initialize state as READY by default
        self.skill_states[skill_name] = "READY"
        self._refresh_active_skills()
        return True

    def _audit_skill_ast(self, skill_name: str) -> None:
        """Issue 51: Pre-Execution AST Validation at registration time."""
        meta = self.skills.get(skill_name)
        if not meta or not meta.instance:
            return

        import ast
        import textwrap

        try:
            # Basic name/import validation
            source = textwrap.dedent(inspect.getsource(meta.instance.__class__))
            tree = ast.parse(source)
            defined_names: set[str] = set()
            accessed_names: set[str] = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        defined_names.add(alias.asname or alias.name)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        defined_names.add(alias.asname or alias.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    accessed_names.add(node.id)
                elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    defined_names.add(node.name)

            # Check for critical missing imports
            critical_modules = {"subprocess", "os", "sys", "json", "asyncio"}
            for mod in critical_modules:
                if mod in accessed_names and mod not in defined_names:
                    self.logger.warning(
                        f"⚠️ Skill Safety Audit: '{skill_name}' uses '{mod}' but does not import it."
                    )
        except (ImportError, AttributeError, RuntimeError) as e:
            _record_capability_degradation(
                e,
                action="skipped optional skill AST validation after source inspection failed",
            )
            self.logger.debug(f"AST validation skipped for {skill_name}: {e}")

    def _initialize_skill_states(self) -> None:
        """Emits the initial state of all registered skills."""
        for name in self.skills:
            self._emit_skill_status(name, "READY")

    def _emit_skill_status(
        self,
        skill_name: str,
        state: str,
        *,
        expected_catalog_digest: str | None = None,
    ) -> None:
        """Emits a skill status update to the EventBus."""
        with self._catalog_guard():
            if (
                expected_catalog_digest is not None
                and expected_catalog_digest != str(getattr(self, "_catalog_digest", ""))
            ):
                return
            self.skill_states[skill_name] = state
        from core.event_bus import get_event_bus

        bus = get_event_bus()
        bus.publish_threadsafe(
            "skill_status", {"skill": skill_name, "state": state, "timestamp": time.time()}
        )

    def get_available_skills(self) -> list[str]:
        """Returns a list of all registered skill names."""
        self._ensure_catalog_loaded()
        with self._catalog_guard():
            return list(self._skills)

    @property
    def skills_loaded(self) -> int:
        """Compatibility count backed by the canonical live registry."""

        self._ensure_catalog_loaded()
        with self._catalog_guard():
            return len(self._skills)

    def discover_skills(self) -> list[str]:
        """Compatibility entry point that performs the full catalog transaction."""

        self.reload_skills()
        return self.get_available_skills()

    def get_catalog_health(self) -> dict[str, Any]:
        with self._catalog_guard():
            catalog_health = dict(self.catalog_health)
            quarantined = [dict(item) for item in self.quarantined_skills.values()]
            preflight = dict(self._catalog_preflight_summary)
            preflight["entries"] = [
                dict(item) for item in preflight.get("entries") or ()
            ]
        return {
            **catalog_health,
            "execution_preflight": preflight,
            "quarantined": [
                {
                    "catalog_id": item.get("catalog_id"),
                    "class_name": item.get("class_name"),
                    "error": item.get("error") or item.get("detail"),
                    "module_path": item.get("module_path"),
                    "name": item.get("name"),
                    "stage": item.get("stage"),
                }
                for item in quarantined
            ],
        }

    def get_catalog_preflight_status(self) -> dict[str, Any]:
        """Return the latest execution-path preflight without initiating work."""

        with self._catalog_guard():
            payload = dict(self._catalog_preflight_summary)
            payload["entries"] = [dict(item) for item in payload.get("entries") or ()]
            return payload

    def dry_run_catalog(self, *, refresh: bool = False) -> dict[str, Any]:
        """Prepare every implementation through the real execution load path.

        This deliberately stops before ``safe_execute``/``execute``. Imports,
        source identity, schema, runtime service resolution, requirements, and
        constructors are the same operations first execution uses. Successful
        instances remain cached, so the proof is not discarded before use.
        """

        self._ensure_catalog_loaded()
        with self._catalog_guard():
            skills = dict(self._skills)
            catalog_ready = bool(self.catalog_health.get("ready"))
            catalog_digest = self._catalog_digest
            quarantined_count = len(self.quarantined_skills)
            cached = dict(self._catalog_preflight_summary)
            cached["entries"] = [dict(item) for item in cached.get("entries") or ()]
        if (
            not refresh
            and cached.get("complete") is True
            and cached.get("catalog_digest") == catalog_digest
        ):
            return cached

        entries = [self.preflight_skill(name, refresh=refresh) for name in sorted(skills)]
        failed = [str(entry.get("name") or "") for entry in entries if not entry.get("ok")]
        ok = bool(catalog_ready and entries and not failed and not quarantined_count)
        summary = {
            "catalog_digest": catalog_digest,
            "complete": True,
            "entries": entries,
            "failed": failed,
            "live_count": len(entries),
            "ok": ok,
            "quarantined_count": quarantined_count,
            "reason": "ready" if ok else "execution_preflight_failed",
        }
        with self._catalog_guard():
            if self._catalog_digest == catalog_digest:
                self._catalog_preflight_summary = summary
        return {**summary, "entries": [dict(item) for item in entries]}

    def resolve_skill_name(self, skill_name: Any) -> str:
        """Resolve a requested skill without collapsing real registered skills."""
        raw = str(skill_name or "").strip()
        if not raw:
            return ""

        self._ensure_catalog_loaded()
        with self._catalog_guard():
            skill_names = tuple(self._skills)
        if raw in skill_names:
            return raw

        lowered = raw.lower()
        casefolded = {name.lower(): name for name in skill_names}
        if lowered in casefolded:
            return casefolded[lowered]

        alias_target = self.SKILL_ALIASES.get(raw, self.SKILL_ALIASES.get(lowered, raw))
        if alias_target in skill_names:
            return alias_target

        alias_lowered = str(alias_target or "").lower()
        if alias_lowered in casefolded:
            return casefolded[alias_lowered]

        return raw

    @staticmethod
    def _verify_catalog_source(metadata: SkillMetadata) -> str:
        """Check the source against the digest the isolated probe validated.

        Returns the observed digest when the source has CHANGED, and an empty
        string when it is unchanged — the caller finishes the job, because
        the properties that make a changed source safe can only be checked
        after the import.

        This used to raise on any change, and the message said "reload is
        required" — then nothing reloaded. Live 2026-07-28, editing a skill
        file under a running instance turned an ordinary request ("open Notes
        and write a note") into a CRITICAL fail-closed subsystem failure, and
        the person got a sentence about catalog digests instead of a note.
        Refusing was not protecting anything: a stale digest is not evidence
        of tampering, it is evidence that the code was edited, which is what
        development is.

        What the digest actually guards is that nobody re-proved the skill's
        identity and authority against THIS content. But the caller re-proves
        exactly that on every lazy load, against the freshly imported class:
        still a BaseSkill, same declared name, same effect scope. Those are
        the substantive properties, and they hard-fail on their own. So the
        honest response to a changed source is to re-derive trust from the
        checks that can still be run — and to keep hard-failing when they
        cannot be:

            unreadable source   nothing can be checked        refuse
            outside the tree    an attack shape, not an edit  refuse
            changed content     re-prove it after import      proceed
        """
        if not metadata.source_sha256 or not metadata.source_path:
            return ""
        source_path = (Path(config.paths.base_dir) / metadata.source_path).resolve()
        root = Path(config.paths.base_dir).resolve()
        try:
            source_path.relative_to(root)
            observed = hashlib.sha256(source_path.read_bytes()).hexdigest()
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"catalog source is no longer readable: {exc}") from exc
        if observed != metadata.source_sha256:
            return observed
        return ""

    @staticmethod
    def _resolve_constructor_dependencies(metadata: SkillMetadata) -> dict[str, Any]:
        dependencies: dict[str, Any] = {}
        for dependency_name in metadata.constructor_dependencies:
            dependency = optional_service(dependency_name, default=None)
            if dependency is None:
                raise RuntimeError(
                    f"declared constructor dependency {dependency_name!r} is unavailable"
                )
            dependencies[dependency_name] = dependency
        return dependencies

    @classmethod
    def _construct_skill_instance(
        cls,
        metadata: SkillMetadata,
        skill_class: type[Any],
        *,
        dependencies: dict[str, Any] | None = None,
    ) -> Any:
        if dependencies is None:
            dependencies = cls._resolve_constructor_dependencies(metadata)
        return skill_class(**dependencies)

    def _preflight_store(self) -> dict[str, dict[str, Any]]:
        """The preflight receipts, guaranteed to exist.

        ``__init__`` sets ``_skill_preflight_results``, but not every
        CapabilityEngine in this codebase has run it — the UI bootstrap
        contract builds one with ``__new__`` and populates only the fields
        the catalog needs. One reader already worked around that with an
        inline ``hasattr`` check while two others read the attribute
        directly and raised. One accessor, so a new reader cannot pick the
        unguarded spelling by accident.
        """
        store = getattr(self, "_skill_preflight_results", None)
        if store is None:
            store = {}
            self._skill_preflight_results = store
        return store

    def _record_skill_preflight(self, receipt: dict[str, Any]) -> dict[str, Any]:
        stored = dict(receipt)
        name = str(stored.get("name") or "")
        with self._catalog_guard():
            if name:
                self._preflight_store()[name] = stored
                if stored.get("ok"):
                    self.skill_last_errors.pop(name, None)
                else:
                    self.skill_last_errors[name] = str(
                        stored.get("error") or "execution preflight failed"
                    )
        return dict(stored)

    def _prepare_skill_instance(
        self,
        skill_name: str,
        metadata: SkillMetadata,
        *,
        refresh: bool = False,
    ) -> tuple[dict[str, Any], Any | None]:
        """Load and construct a skill exactly as execution will, but do not run it."""

        started = time.monotonic()
        stage = "catalog_identity"
        constructor_invoked = False
        instance_reused = False
        source_revalidated = False
        try:
            with self._catalog_mutation_guard():
                with self._catalog_guard():
                    current = self._skills.get(skill_name)
                    catalog_digest = str(
                        getattr(self, "_catalog_digest", "")
                        or f"runtime:{id(self._skills)}"
                    )
                    existing = self._instances.get(skill_name)
                if current is not metadata:
                    raise RuntimeError("skill catalog generation changed during preflight")
                if metadata.validation_state != "valid":
                    raise RuntimeError(
                        f"catalog validation state is {metadata.validation_state!r}"
                    )

                stage = "source_identity"
                revalidated_digest = self._verify_catalog_source(metadata)
                source_revalidated = bool(revalidated_digest)

                stage = "implementation_import"
                skill_class = metadata.skill_class
                if skill_class is None:
                    module_path = metadata.module_path
                    class_name = metadata.class_name
                    if not module_path or not class_name:
                        raise RuntimeError("catalog entry is missing its import identity")
                    module = importlib.import_module(module_path)
                    skill_class = getattr(module, class_name)

                stage = "implementation_contract"
                from core.skills.base_skill import BaseSkill

                if not inspect.isclass(skill_class) or not issubclass(skill_class, BaseSkill):
                    raise TypeError("implementation does not satisfy canonical BaseSkill")
                if str(getattr(skill_class, "name", "")) != skill_name:
                    raise ValueError("implementation name differs from the live catalog")
                if not callable(getattr(skill_class, "execute", None)):
                    raise TypeError("implementation has no execute() contract")
                if not callable(getattr(skill_class, "safe_execute", None)):
                    raise TypeError("implementation has no governed safe_execute() contract")
                observed_scope = _declared_effect_scope(skill_name, skill_class)
                if metadata.effect_scope in {"", "unknown"}:
                    metadata.effect_scope = observed_scope
                    policy = resolve_skill_policy(skill_name, observed_scope)
                    if policy is not None and metadata.authority_class == "unclassified":
                        metadata.authority_class = policy.authority_class
                elif observed_scope != metadata.effect_scope:
                    raise ValueError("implementation effect classification changed")

                stage = "schema"
                input_model = getattr(skill_class, "input_model", None)
                schema = metadata.schema_def
                if not isinstance(schema, dict) or schema.get("type") != "object":
                    raise TypeError("skill input schema must describe an object")
                json.dumps(schema, sort_keys=True)

                stage = "requirements"
                requirements_ready, requirement_errors = metadata.requirements.check()
                if not requirements_ready:
                    raise RuntimeError(
                        "declared runtime requirements unavailable: "
                        + "; ".join(str(item) for item in requirement_errors)
                    )

                stage = "dependency_resolution"
                dependencies = self._resolve_constructor_dependencies(metadata)

                stage = "construction"
                instance = existing or metadata.instance
                if instance is None:
                    instance = self._construct_skill_instance(
                        metadata,
                        skill_class,
                        dependencies=dependencies,
                    )
                    constructor_invoked = True
                else:
                    instance_reused = True
                if str(getattr(instance, "name", "")) != skill_name:
                    raise ValueError("constructed implementation changed its declared name")

                stage = "publication"
                with self._catalog_guard():
                    if (
                        self._skills.get(skill_name) is not metadata
                        or str(
                            getattr(self, "_catalog_digest", "")
                            or f"runtime:{id(self._skills)}"
                        )
                        != catalog_digest
                    ):
                        raise RuntimeError("skill catalog generation changed before publication")
                    metadata.skill_class = skill_class
                    # Only adopt a model the class actually declares. This was
                    # unconditional, so a catalog entry carrying an input_model
                    # the implementation did not repeat had it erased at
                    # publication — and the skill then ran with NO input
                    # validation at all, silently, having just passed preflight.
                    # A validation gate that disappears during the check meant
                    # to confirm it is the worst shape available.
                    if input_model is not None:
                        metadata.input_model = input_model
                    metadata.source_sha256 = revalidated_digest or metadata.source_sha256
                    metadata.dependency_ready = True
                    metadata.dependency_errors = []
                    self._instances[skill_name] = instance

            receipt = {
                "authority_class": metadata.authority_class,
                "catalog_digest": catalog_digest,
                "catalog_id": metadata.catalog_id,
                "constructor_dependencies": sorted(dependencies),
                "constructor_invoked": constructor_invoked,
                "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
                "effect_scope": metadata.effect_scope,
                "instance_reused": instance_reused,
                "name": skill_name,
                "ok": True,
                "route_class": self._route_class_for(metadata),
                "schema_property_count": len(schema.get("properties") or {}),
                "source_revalidated": source_revalidated,
                "stage": "ready",
                "skill_body_invoked": False,
            }
            return self._record_skill_preflight(receipt), instance
        except _SKILL_PREFLIGHT_ERRORS as exc:
            receipt = {
                "authority_class": metadata.authority_class,
                "catalog_id": metadata.catalog_id,
                "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
                "effect_scope": metadata.effect_scope,
                "error": f"{type(exc).__name__}: {exc}"[:1200],
                "error_type": type(exc).__name__,
                "name": skill_name,
                "ok": False,
                "stage": stage,
                "skill_body_invoked": False,
            }
            return self._record_skill_preflight(receipt), None

    def preflight_skill(self, skill_name: str, *, refresh: bool = False) -> dict[str, Any]:
        """Prove one skill can reach the execution boundary without effects."""

        skill_name = self.resolve_skill_name(skill_name)
        self._ensure_catalog_loaded()
        with self._catalog_guard():
            metadata = self._skills.get(skill_name)
            cached = dict(self._preflight_store().get(skill_name) or {})
            catalog_digest = self._catalog_digest
        if metadata is None:
            return {
                "catalog_digest": catalog_digest,
                "error": "skill is not registered in the live catalog",
                "name": skill_name,
                "ok": False,
                "stage": "catalog_identity",
                "skill_body_invoked": False,
            }
        if not refresh and cached.get("ok") and cached.get("catalog_digest") == catalog_digest:
            return cached
        receipt, _instance = self._prepare_skill_instance(
            skill_name,
            metadata,
            refresh=refresh,
        )
        return receipt

    def _route_class_for(self, meta: SkillMetadata) -> str:
        route_class_hint = getattr(meta, "route_class_hint", None)
        if route_class_hint in {"async", "sync"}:
            return str(route_class_hint)
        target = meta.instance or meta.skill_class
        if target is None:
            return "managed_async"
        for attr in ("execute", "run", "__call__"):
            fn = getattr(target, attr, None)
            if fn is None:
                continue
            try:
                return "async" if inspect.iscoroutinefunction(fn) else "sync"
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                _record_capability_degradation(
                    exc,
                    action="fell back to managed async route after route classification failed",
                )
                self.logger.debug(
                    "Unable to classify route for skill %s attr %s: %s", meta.name, attr, exc
                )
                continue
        return "managed_async"

    def _risk_class_for(self, skill_name: str, meta: SkillMetadata) -> str:
        if skill_name in _CRITICAL_ACTION_SKILLS or skill_name in {"computer_use", "desktop_task"}:
            return "critical"
        if meta.metabolic_cost >= 3:
            return "high"
        if meta.metabolic_cost >= 2:
            return "medium"
        return "low"


    def _effect_scope_for(self, skill_name: str, meta: SkillMetadata) -> str:
        declared = _declared_effect_scope(
            skill_name, meta.instance or meta.skill_class
        )
        if declared and declared != "unknown":
            return declared
        return str(meta.effect_scope or "unknown").strip().lower()

    @staticmethod
    def _is_path_within_workspace(path: Any) -> bool:
        raw = str(path or "").strip()
        if not raw:
            return False
        try:
            root = Path.cwd().resolve()
            target = Path(raw)
            if not target.is_absolute():
                target = root / target
            resolved = target.resolve()
            return os.path.commonpath([str(root), str(resolved)]) == str(root)
        except (OSError, RuntimeError, TypeError, ValueError):
            return False

    @classmethod
    def _workspace_file_io_scope(cls, params: dict[str, Any]) -> str | None:
        action = str((params or {}).get("action") or "").strip().lower()
        if action not in {"append", "copy", "exists", "list", "patch", "read", "write"}:
            return None
        if not cls._is_path_within_workspace((params or {}).get("path")):
            return None
        if action == "copy" and not cls._is_path_within_workspace((params or {}).get("destination")):
            return None
        return "workspace_file_io"

    @staticmethod
    def _computer_use_effect_scope(params: dict[str, Any]) -> str | None:
        action = str((params or {}).get("action") or "").strip().lower()
        if action in {"get_clipboard", "read_menu_clock", "read_screen_text", "wait"}:
            return "read_only"
        if action in {
            "click",
            "hotkey",
            "move_aura_bubble",
            "open_app",
            "open_url",
            "run_applescript",
            "scroll",
            "set_clipboard",
            "type",
        }:
            return "foreground_desktop_control"
        if action in {"move_file", "render_text_pdf", "write_text_file"}:
            return "desktop_file_io"
        if action == "run_command":
            try:
                argv = shlex.split(str((params or {}).get("target") or ""))
            except ValueError:
                return None
            if not argv:
                return None
            binary = Path(argv[0]).name
            if binary in {"cat", "echo", "find", "grep", "ls", "pwd", "tree"}:
                return "sandboxed_compute"
            if binary == "git":
                subcommand = argv[1] if len(argv) > 1 else ""
                if subcommand in {"branch", "diff", "log", "rev-parse", "show", "status"}:
                    return "sandboxed_compute"
                return "subprocess"
            if binary == "python3":
                if len(argv) == 2 and argv[1] in {"--version", "-V"}:
                    return "sandboxed_compute"
                return "subprocess"
            if binary == "pip":
                if len(argv) == 2 and argv[1] in {"--version", "-V"}:
                    return "sandboxed_compute"
                return "subprocess"
            if binary in {"mkdir", "touch"}:
                return "subprocess"
        return None

    @staticmethod
    def _auto_refactor_effect_scope(params: dict[str, Any]) -> str:
        """Classify auto-refactor by the concrete operation, not by its name.

        The shipped AutoRefactorSkill is a scanner/proposal emitter by default.
        Background autonomy must be able to run that safe inspection path, while
        any future apply/promote/write mode remains privileged and confirmable.
        """

        params = params or {}
        mode = str(params.get("mode") or params.get("action") or "scan").strip().lower()
        mutating = bool(
            params.get("apply")
            or params.get("write")
            or params.get("commit")
            or params.get("promote")
            or params.get("allow_mutation")
            or mode in {"apply", "commit", "promote", "rewrite", "write"}
        )
        if mutating:
            return "privileged_mutation"
        if bool(params.get("run_tests")):
            return "sandboxed_compute"
        return "read_only"

    def _effect_scope_for_execution(
        self,
        skill_name: str,
        meta: SkillMetadata,
        params: dict[str, Any],
        ctx: dict[str, Any] | None = None,
    ) -> str:
        return resolve_execution_effect_scope(
            skill_name,
            params,
            declared_effect_scope=self._effect_scope_for(skill_name, meta),
        )

    def _capability_chain_denial(
        self,
        ctx: dict[str, Any],
        skill_name: str,
        params: Any,
        constitutional_runtime_live: bool,
    ) -> dict[str, Any] | None:
        """Authenticate the Will's grant at the moment of execution.

        This is the point where the constitutional chain actually closes: the
        capability presented here must be a signature over *this* decision, for
        *this* action, with *these* parameters — not merely a token that exists.

        Returns a denial payload to abort the execution, or None to proceed.
        Enforcement follows ``AURA_CAPABILITY_ENFORCEMENT``:

            strict  refuse execution without a verified capability (default once
                    the runtime is constitutionally live)
            warn    record the violation and proceed — migration only
            off     skip entirely — for tests and pre-runtime boot paths

        The default is strict whenever the constitutional runtime is live,
        because a governance check that can be quietly skipped is the failure
        this whole change exists to remove.
        """
        mode = capability_enforcement_mode(default="strict" if constitutional_runtime_live else "off")
        if mode == "off":
            return None

        try:
            enforce_capability(
                ctx,
                sink=f"capability_engine.execute_skill:{skill_name}",
                # String rather than ActionDomain: importing core.will here would
                # cycle (the Will imports the engine's container). The verifier
                # normalizes enum and str identically.
                domain="tool_execution",
                action=skill_name,
                payload=canonical_authority_arguments(skill_name, params),
            )
            return None
        except CapabilityViolation as exc:
            _record_capability_degradation(
                exc,
                action=(
                    f"{'refused' if mode == 'strict' else 'permitted (warn mode)'} "
                    f"'{skill_name}': {exc.denial.value}"
                ),
                severity="degraded" if mode == "strict" else "warning",
                enforce_failure_policy=False,
            )
            if mode != "strict":
                self.logger.warning(
                    "⚠️  CapabilityEngine: '%s' executing WITHOUT verified Will "
                    "authority (%s) — warn mode",
                    skill_name,
                    exc.denial.value,
                )
                return None
            self.logger.warning(
                "🔒 CapabilityEngine: '%s' refused — %s (%s)",
                skill_name,
                exc.denial.value,
                exc.detail,
            )
            return {
                "ok": False,
                "error": f"Will authority not established: {exc.denial.value}",
                "status": "blocked_by_capability_chain",
                "denial": exc.denial.value,
                "detail": exc.detail,
            }

    @staticmethod
    def _context_governed_execution(ctx: dict[str, Any], skill_name: str) -> bool:
        """Is this execution carrying real authority from the Will?

        Authority is established by *authenticating a signature*, never by a
        caller's claim. A context that merely says it was verified
        (``_capability_token_verified``) is not evidence of anything — that flag
        was the bypass this method used to honour, and any code path or
        deserialized payload could set it.
        """
        cap = capability_from_context(ctx)
        if cap is not None:
            result = get_capability_verifier().verify(
                cap,
                expected_action_digest=None,  # bound at the execution sink
                consume=False,                # this is an advisory read, not a spend
            )
            if result.ok:
                return True
            _record_capability_degradation(
                CapabilityViolation(
                    result.denial or CapabilityDenial.MALFORMED, result.detail
                ),
                action=(
                    f"treated '{skill_name}' as ungoverned: presented capability "
                    f"failed verification"
                ),
                severity="warning",
            )
            return False

        # Legacy opaque-token path. This establishes only that a token naming
        # this skill exists in-process — not that the Will issued it. It is kept
        # so callers still on the old contract are not silently downgraded to
        # "ungoverned", but it is not accepted as proof of Will provenance and
        # a bare self-asserted flag is never sufficient on its own.
        token_id = str((ctx or {}).get("capability_token_id") or "").strip()
        if not token_id:
            return False
        try:
            from core.executive.authority_gateway import get_authority_gateway

            if get_authority_gateway().verify_tool_access(skill_name, token_id):
                return True
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_capability_degradation(
                exc,
                action="treated unverified capability token as ungoverned for EDI check",
                severity="warning",
            )
        return False

    @staticmethod
    def _context_user_authorized(ctx: dict[str, Any], exec_source: str) -> bool:
        # sealed_validation is gone: nothing in core/ or interface/ has ever
        # set it, so its only possible source was an external payload. A key
        # that only an attacker can populate is not an authorization signal.
        if bool(
            ctx.get("user_requested_action")
            or ctx.get("user_explicitly_authorized")
            or (
                ctx.get("proof_evaluation_contract")
                and _proof_run_environment_active()
            )
        ):
            return True
        if exec_source in _USER_FACING_CONTEXT_ORIGINS:
            return True
        proof_run = str(os.environ.get("AURA_PROOF_RUN", "") or "").strip().lower()
        if proof_run in {"1", "true", "yes"} and str(ctx.get("origin") or "").lower() in {
            "test",
            "proof",
        }:
            return True
        return False

    def _edi_risk_for(
        self,
        skill_name: str,
        meta: SkillMetadata,
        params: dict[str, Any],
        effect_scope: str,
    ) -> str:
        return classify_execution_risk(
            skill_name,
            params,
            effect_scope=effect_scope,
            metabolic_cost=meta.metabolic_cost,
        )

    @staticmethod
    def _user_advocate_irreversible_for(
        skill_name: str,
        params: dict[str, Any],
        risk_level: str,
        effect_scope: str,
    ) -> bool:
        """Return whether a skill needs explicit irreversible-action consent.

        High-risk sandboxed compute is not automatically irreversible. The
        user advocate should still block stateful code, privileged mutations,
        and external/user-visible changes without confirmation, but a
        non-stateful sandbox retention probe is not a destructive act merely
        because the generic code-execution skill carries elevated risk.
        """

        if str(risk_level or "").lower() == "critical":
            return True
        if skill_name == "auto_refactor":
            return str(effect_scope or "").lower() in {
                "privileged_mutation",
                "state_mutation",
                "subprocess",
            }
        if skill_name == "run_code":
            return bool((params or {}).get("stateful", True))
        scope = str(effect_scope or "").lower()
        return scope in {
            "desktop_file_io",
            "foreground_desktop_control",
            "privileged_mutation",
            "state_mutation",
            "subprocess",
        }

    @staticmethod
    def _user_advocate_auto_confirmed_for(
        skill_name: str,
        ctx: dict[str, Any],
        exec_source: str,
        effect_scope: str,
    ) -> bool:
        """Allow explicitly user-visible foreground desktop requests to proceed.

        The desktop_task/computer_use stack is still effect-verified downstream.
        This only prevents the user-advocate from re-blocking a desktop action
        that already arrived through the live user/proof foreground lane with
        visible-local-action metadata.
        """

        if (
            skill_name in _LIGHTWEIGHT_BACKGROUND_IO_SKILLS
            and CapabilityEngine._safe_autonomous_web_research(skill_name, {}, ctx, exec_source, effect_scope)
        ):
            return True

        if skill_name not in {
            "computer_use",
            "desktop_task",
            "os_automation",
            "web_interlocutor",
        }:
            return False
        scope = str(effect_scope or "").lower()
        if scope not in {
            "desktop_file_io",
            "foreground_desktop_control",
            "foreground_browser_dialogue",
        }:
            return False
        if str(exec_source or "").lower() not in _USER_FACING_CONTEXT_ORIGINS:
            return False
        if skill_name == "web_interlocutor":
            return bool(
                ctx.get("user_visible_browser_action")
                or ctx.get("user_requested_action")
                or ctx.get("foreground_request")
                or str(ctx.get("route") or "").startswith(("chat.", "voice."))
                or (
                    ctx.get("proof_evaluation_contract")
                    and _proof_run_environment_active()
                )
            )
        if skill_name == "os_automation":
            return bool(
                ctx.get("foreground_request")
                and ctx.get("user_requested_action")
                and ctx.get("user_explicitly_authorized")
                and (
                    ctx.get("user_visible_desktop_action")
                    or ctx.get("local_desktop_action")
                    or str(ctx.get("route") or "").startswith(
                        ("chat.", "desktop_task.os_automation", "voice.")
                    )
                )
            )
        return bool(
            ctx.get("user_visible_desktop_action")
            or ctx.get("local_desktop_action")
            or ctx.get("desktop_task_owned_by")
            or str(ctx.get("route") or "").startswith(("chat.", "voice."))
            or (
                ctx.get("proof_evaluation_contract")
                and _proof_run_environment_active()
            )
        )

    @classmethod
    def _user_advocate_confirmed_for(
        cls,
        skill_name: str,
        params: dict[str, Any],
        ctx: dict[str, Any],
        exec_source: str,
        risk_level: str,
        effect_scope: str,
    ) -> bool:
        """Return one confirmation decision for cooldown and advocate review."""
        # AuthorityGateway has already matched the signed child lease to this
        # exact tool invocation.  Treat that durable standing grant as the
        # confirmation contract it represents instead of asking the owner a
        # second time.  The identity sentinel cannot be supplied by JSON/API
        # callers, so raw context fields never manufacture this decision.
        if ctx.get("_standing_authority_verified") is _VERIFIED_STANDING_AUTHORITY:
            return True
        # CP126: "Confirmation and foreground classification rely on
        # parameters/context such as confirmed, user_confirmed, priority, and
        # foreground_request. These values are not bound to a UI event,
        # authenticated session, nonce, or constitutional receipt."
        #
        # `params` is the MODEL's output — it writes the tool call. Reading a
        # confirmation out of it let a model confirm its own high-risk action
        # by emitting {"confirmed": true}. Nothing in core/ or interface/
        # has ever set these four keys, so there is no legitimate producer to
        # preserve: every occurrence came from outside.
        #
        # The unforgeable path above (_VERIFIED_STANDING_AUTHORITY, placed by
        # the gateway after matching a signed lease) remains the way a real
        # confirmation is expressed, and is the pattern any future UI
        # confirmation event should follow.
        self_asserted = bool(
            params.get("confirmed")
            or params.get("user_confirmed")
            or ctx.get("confirmed")
            or ctx.get("user_confirmed")
        )
        if self_asserted:
            _record_capability_degradation(
                PermissionError(
                    f"{skill_name}: ignored a self-asserted confirmation flag; "
                    "confirmation must come from a verified standing authority"
                ),
                action="refused to treat a caller-supplied confirmed flag as user confirmation",
                severity="warning",
                enforce_failure_policy=False,
            )
        explicitly_confirmed = False
        if skill_name == "os_automation":
            return bool(
                explicitly_confirmed
                or cls._user_advocate_auto_confirmed_for(
                    skill_name,
                    ctx,
                    exec_source,
                    effect_scope,
                )
            )
        return bool(
            explicitly_confirmed
            or (
                exec_source in _USER_FACING_CONTEXT_ORIGINS
                and risk_level not in ("high", "critical")
            )
            or cls._safe_autonomous_web_research(
                skill_name,
                params,
                ctx,
                exec_source,
                effect_scope,
            )
            or cls._user_advocate_auto_confirmed_for(
                skill_name,
                ctx,
                exec_source,
                effect_scope,
            )
        )

    @staticmethod
    def _record_verified_standing_authority(
        ctx: dict[str, Any],
        tool_handle: Any,
    ) -> bool:
        """Record trusted standing-authority provenance from an approved handle."""
        standing_token = str(
            getattr(tool_handle, "standing_authority_token", "") or ""
        ).strip()
        constraints = dict(getattr(tool_handle, "constraints", {}) or {})
        grant_id = str(constraints.get("standing_authority_grant_id") or "").strip()
        if not standing_token or not grant_id:
            return False
        ctx["_standing_authority_verified"] = _VERIFIED_STANDING_AUTHORITY
        ctx["standing_authority_grant_id"] = grant_id
        receipt_id = str(
            constraints.get("standing_authority_receipt_id") or ""
        ).strip()
        if receipt_id:
            ctx["standing_authority_receipt_id"] = receipt_id
        return True

    @staticmethod
    def _action_description_for_user_advocate(
        skill_name: str,
        params: dict[str, Any],
        effect_scope: str,
    ) -> str:
        """Describe the concrete operation, not just the skill's scary name."""

        scope = str(effect_scope or "").lower()
        if skill_name == "auto_refactor" and scope == "read_only":
            target = str((params or {}).get("path") or ".").strip() or "."
            return (
                "read-only auto_refactor code-health scan "
                f"for {target!r}; no source writes, no test execution, no promotion"
            )
        if scope == "read_only" and skill_name in {
            "free_search",
            "grounded_search",
            "local_reference_search",
            "search_web",
            "web_search",
        }:
            query = str((params or {}).get("query") or (params or {}).get("q") or "").strip()
            if query:
                return f"read-only {skill_name} information retrieval for query {query!r}"
            return f"read-only {skill_name} information retrieval"
        if skill_name == "messages":
            arguments = canonical_authority_arguments(skill_name, params)
            action = str(arguments.get("action") or "status")
            if action == "send":
                return (
                    "send a private message to the configured symbolic contact "
                    f"({int(arguments.get('body_chars') or 0)} characters; content hidden)"
                )
            return f"{action} Aura's private Messages channel"
        return f"{skill_name} {str(params)[:200]}"

    @staticmethod
    def _directly_requested_by_the_user(ctx: dict[str, Any], exec_source: str) -> bool:
        """Did a person ask for this action, in this turn, in the foreground?

        Narrow on purpose. It is not "a user exists somewhere upstream" — it is
        a foreground origin AND a turn that reads as an instruction rather than
        a mention (core/conversation/request_mood.py). An autonomous cycle that
        happens to carry a user id does not qualify.
        """
        origin = (
            str(exec_source or ctx.get("origin") or ctx.get("source") or "")
            .strip()
            .lower()
            .replace("-", "_")
        )
        if origin not in _DIRECT_USER_REQUEST_ORIGINS:
            return False
        message = str(
            ctx.get("message") or ctx.get("objective") or ctx.get("user_message") or ""
        ).strip()
        if not message:
            return False
        try:
            from core.conversation.request_mood import assess_request_mood

            return assess_request_mood(message).asks_for_action
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("capability_engine.direct_request", exc, severity="warning")
            return False

    @staticmethod
    def _safe_autonomous_web_research(
        skill_name: str,
        params: dict[str, Any],
        ctx: dict[str, Any],
        exec_source: str,
        effect_scope: str,
    ) -> bool:
        if skill_name not in _LIGHTWEIGHT_BACKGROUND_IO_SKILLS:
            return False
        if str(effect_scope or "").lower() != "read_only":
            return False
        origin = str(exec_source or ctx.get("origin") or ctx.get("source") or "").strip().lower().replace("-", "_")
        if origin not in _AUTONOMOUS_RESEARCH_ORIGINS:
            return False
        text = " ".join(
            str(part or "").lower()
            for part in (
                (params or {}).get("query"),
                (params or {}).get("q"),
                ctx.get("objective"),
                ctx.get("message"),
                ctx.get("reason"),
            )
        )
        if not text.strip():
            return False
        return not any(marker in text for marker in _UNSAFE_AUTONOMOUS_WEB_QUERY_MARKERS)

    @staticmethod
    def _user_benefit_for_execution(
        skill_name: str,
        params: dict[str, Any],
        ctx: dict[str, Any],
        exec_source: str,
        effect_scope: str,
    ) -> str:
        explicit = str(params.get("user_benefit") or ctx.get("user_benefit") or "").strip()
        if explicit:
            return explicit
        if CapabilityEngine._safe_autonomous_web_research(skill_name, params, ctx, exec_source, effect_scope):
            query = str((params or {}).get("query") or (params or {}).get("q") or "").strip()
            return (
                "support Aura's autonomous curiosity, factual grounding, and memory growth "
                f"with bounded read-only web research{f' about {query[:120]!r}' if query else ''}"
            )
        objective = str(ctx.get("objective") or ctx.get("user_objective") or "").strip()
        if objective:
            return objective
        if skill_name == "auto_refactor" and str(effect_scope or "").lower() == "read_only":
            return (
                "maintain Aura's code health by surfacing bounded repair candidates "
                "without mutating source or consuming a heavy test budget"
            )
        if exec_source in _USER_FACING_CONTEXT_ORIGINS:
            return "requested through the user-facing skill lane"
        return ""

    @staticmethod
    def _input_summary_for(meta: SkillMetadata) -> str:
        schema = meta.schema_def or {}
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        if not props:
            return "No structured inputs required."
        names = list(props.keys())[:5]
        required = set(schema.get("required", []) if isinstance(schema, dict) else [])
        pieces = []
        for name in names:
            descriptor = f"{name} (required)" if name in required else name
            pieces.append(descriptor)
        return ", ".join(pieces)

    @staticmethod
    def _example_usage_for(skill_name: str, meta: SkillMetadata) -> str:
        for pattern in meta.trigger_patterns[:3]:
            cleaned = re.sub(r"\\\w|\(\?:|\(|\)|\^|\$|\[|\]|\?|\+|\*|\|", " ", pattern)
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
            if cleaned:
                return cleaned
        description = meta.description.strip().rstrip(".")
        if description:
            return f"use {skill_name} to {description[:80].lower()}"
        return f"use {skill_name} when its capability is needed"

    def _catalog_item_for_skill(self, skill_name: str, meta: SkillMetadata) -> dict[str, Any]:
        state = self.skill_states.get(skill_name, "READY")
        active = skill_name in self.active_skills
        preflight = dict(self._preflight_store().get(skill_name) or {})
        preflight_failed = bool(preflight) and not bool(preflight.get("ok"))
        available = bool(
            meta.enabled
            and active
            and state != "ERROR"
            and getattr(meta, "validation_state", "valid") == "valid"
            and bool(getattr(meta, "dependency_ready", True))
            and not preflight_failed
        )
        policy_state = (
            "disabled"
            if not meta.enabled
            else "inactive_by_policy"
            if skill_name in self._explicitly_deactivated_skills
            else "active"
            if active
            else "inactive"
        )
        availability_reason = (
            None
            if available
            else (
                self.skill_last_errors.get(skill_name)
                or (
                    "disabled_by_policy"
                    if not meta.enabled
                    else "inactive_by_policy"
                    if skill_name in self._explicitly_deactivated_skills
                    else "error_state"
                    if state == "ERROR"
                    else "dependency_unavailable"
                    if not bool(getattr(meta, "dependency_ready", True))
                    else str(preflight.get("error") or "execution_preflight_failed")
                    if preflight_failed
                    else "inactive"
                )
            )
        )

        return {
            "name": skill_name,
            "description": meta.description,
            "state": state,
            "availability": "available" if available else "unavailable",
            "available": available,
            "enabled": bool(meta.enabled),
            "active": active,
            "policy_state": policy_state,
            "preflight_state": (
                "ready"
                if preflight.get("ok") is True
                else "failed"
                if preflight
                else "not_run"
            ),
            "preflight_stage": str(preflight.get("stage") or "not_run"),
            "risk_class": self._risk_class_for(skill_name, meta),
            "route_class": self._route_class_for(meta),
            "input_summary": self._input_summary_for(meta),
            "example_usage": self._example_usage_for(skill_name, meta),
            "last_error": self.skill_last_errors.get(skill_name),
            "degraded_reason": availability_reason,
            "availability_reason": availability_reason,
            "execution_profile": meta.execution_profile,
            "timeout_seconds": meta.timeout_seconds,
            "memory_mb_estimate": meta.memory_mb_estimate,
            "metabolic_cost": meta.metabolic_cost,
            "effect_scope": self._effect_scope_for(skill_name, meta),
            "authority_class": getattr(meta, "authority_class", "unclassified"),
            "catalog_id": getattr(meta, "catalog_id", "") or skill_name,
            "dependency_ready": bool(getattr(meta, "dependency_ready", True)),
            "source_kind": getattr(meta, "source_kind", "runtime"),
            "source_path": getattr(meta, "source_path", ""),
            "validation_state": getattr(meta, "validation_state", "valid"),
        }

    @staticmethod
    def _catalog_item_for_quarantine(item: dict[str, Any]) -> dict[str, Any]:
        name = str(item.get("name") or item.get("class_name") or "unresolved_skill")
        error = str(item.get("error") or item.get("detail") or "catalog validation failed")
        return {
            "active": False,
            "authority_class": str(item.get("authority_class") or "unclassified"),
            "availability": "quarantined",
            "availability_reason": error,
            "available": False,
            "catalog_id": str(item.get("catalog_id") or name),
            "degraded_reason": error,
            "dependency_ready": False,
            "description": str(item.get("description") or "Skill declaration is quarantined."),
            "effect_scope": str(item.get("effect_scope") or "unknown"),
            "enabled": False,
            "example_usage": "Unavailable until the catalog contract is repaired.",
            "execution_profile": "none",
            "input_summary": "Unavailable.",
            "last_error": error,
            "memory_mb_estimate": 0,
            "metabolic_cost": 0,
            "name": name,
            "policy_state": "quarantined",
            "preflight_state": "failed",
            "preflight_stage": str(item.get("stage") or "catalog_validation"),
            "risk_class": "critical",
            "route_class": "blocked",
            "source_kind": str(item.get("source_kind") or "unknown"),
            "source_path": str(item.get("source_path") or ""),
            "state": "QUARANTINED",
            "timeout_seconds": 0,
            "validation_state": "quarantined",
        }

    def _suppressed_aliases(self) -> set[str]:
        """Alias names to hide because their canonical skill is registered.

        Computed per call rather than stored: whether an alias may be hidden
        depends on the canonical skill being present RIGHT NOW, and a registry
        that lost the canonical must fall back to showing the alias rather
        than losing the capability entirely.
        """
        return {
            alias
            for alias, canonical in _SKILL_ALIASES.items()
            if alias != canonical and canonical in self.skills and alias in self.skills
        }

    def iter_tool_catalog(self, *, include_inactive: bool = True) -> Iterable[dict[str, Any]]:
        """Stream catalog items without materializing the full registry."""
        yielded: set[str] = set()
        aliases = self._suppressed_aliases()
        for skill_name in sorted(self.active_skills):
            meta = self.skills.get(skill_name)
            if meta is None:
                continue
            if not meta.enabled and not include_inactive:
                continue
            if skill_name in aliases:
                continue
            yielded.add(skill_name)
            yield self._catalog_item_for_skill(skill_name, meta)

        for skill_name, meta in self.skills.items():
            if skill_name in yielded or skill_name in aliases:
                continue
            if not meta.enabled and not include_inactive:
                continue
            yield self._catalog_item_for_skill(skill_name, meta)
        if include_inactive:
            for item in sorted(
                getattr(self, "quarantined_skills", {}).values(),
                key=lambda row: (
                    str(row.get("name") or "").casefold(),
                    str(row.get("module_path") or ""),
                    str(row.get("class_name") or ""),
                ),
            ):
                yield self._catalog_item_for_quarantine(item)

    def get_tool_catalog(self, *, include_inactive: bool = True) -> list[dict[str, Any]]:
        catalog = list(self.iter_tool_catalog(include_inactive=include_inactive))

        catalog.sort(
            key=lambda item: (
                0 if item["available"] else 1,
                0 if item["active"] else 1,
                item["name"],
            )
        )
        return catalog

    def get_catalog(self, *, include_inactive: bool = True) -> dict[str, dict[str, Any]]:
        """Legacy compatibility wrapper expected by older response guards."""
        catalog: dict[str, dict[str, Any]] = {}
        for tool in self.get_tool_catalog(include_inactive=include_inactive):
            name = str(tool.get("name") or "")
            if not name:
                continue
            catalog[name] = {
                "status": "unavailable"
                if not bool(tool.get("available"))
                else str(tool.get("state") or "ready").lower(),
                "available": bool(tool.get("available")),
                "availability_reason": tool.get("availability_reason"),
                "policy_state": tool.get("policy_state"),
                "route_class": tool.get("route_class"),
                "risk_class": tool.get("risk_class"),
            }
        return catalog

    def _bounded_tool_affordance_catalog(
        self,
        *,
        ranked_names: Iterable[str],
        max_available: int,
        max_unavailable: int,
        include_inactive: bool = True,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return a bounded prompt catalog without materializing the full registry."""

        ranked_names_tuple = tuple(ranked_names)
        requested = max_available + max_unavailable + len(ranked_names_tuple)
        max_items = max(24, min(_TOOL_AFFORDANCE_SCAN_LIMIT, requested + 48))
        deadline = time.monotonic() + _TOOL_AFFORDANCE_SCAN_BUDGET_SECONDS
        seen: set[str] = set()
        items: list[dict[str, Any]] = []
        truncated = False

        def _append_catalog_item(item: dict[str, Any]) -> bool:
            nonlocal truncated
            name = str(item.get("name") or "")
            if not name or name in seen:
                return True
            seen.add(name)
            items.append(item)
            if len(items) >= max_items:
                truncated = True
                return False
            if time.monotonic() >= deadline:
                truncated = True
                return False
            return True

        for ranked_name in ranked_names_tuple:
            skill_name = self.resolve_skill_name(ranked_name)
            meta = self.skills.get(skill_name)
            if meta is None:
                continue
            if not meta.enabled and not include_inactive:
                continue
            if not _append_catalog_item(self._catalog_item_for_skill(skill_name, meta)):
                return items, truncated

        for item in self.iter_tool_catalog(include_inactive=include_inactive):
            if not _append_catalog_item(item):
                break

        return items, truncated

    def build_tool_affordance_block(
        self,
        *,
        objective: str = "",
        matched_skills: Iterable[str] | None = None,
        max_available: int = 16,
        max_unavailable: int = 8,
        compact: bool = False,
    ) -> str:
        max_available = max(0, min(int(max_available or 0), 32))
        max_unavailable = max(0, min(int(max_unavailable or 0), 16))
        ranked_names = self._rank_tool_candidates(
            objective=objective,
            matched_skills=matched_skills,
            max_tools=max(max_available + max_unavailable, 6),
        )
        priority = {name: idx for idx, name in enumerate(ranked_names)}
        catalog, catalog_truncated = self._bounded_tool_affordance_catalog(
            ranked_names=ranked_names,
            max_available=max_available,
            max_unavailable=max_unavailable,
            include_inactive=True,
        )
        catalog.sort(
            key=lambda item: (
                0 if item["name"] in priority else 1,
                priority.get(item["name"], 999),
                0 if item["available"] else 1,
                item["name"],
            )
        )

        available = [tool for tool in catalog if tool["available"]][:max_available]
        if compact and ranked_names:
            unavailable = [
                tool for tool in catalog if not tool["available"] and tool["name"] in priority
            ][:max_unavailable]
        else:
            unavailable = [tool for tool in catalog if not tool["available"]][:max_unavailable]

        lines = ["## LIVE TOOL OPTIONS" if compact else "## LIVE TOOL AFFORDANCES"]
        if available:
            lines.append(
                "Most relevant right now:" if compact and ranked_names else "Available right now:"
            )
            for tool in available:
                if compact:
                    lines.append(
                        f"- {tool['name']}: {tool['description'][:72]} "
                        f"(use when: {tool['example_usage'][:72]})"
                    )
                else:
                    lines.append(
                        f"- {tool['name']}: {tool['description'][:90]} "
                        f"(when to use: {tool['example_usage']}; inputs: {tool['input_summary']})"
                    )
        else:
            lines.append("Available right now: none confirmed.")

        if unavailable:
            lines.append(
                "Relevant but unavailable:"
                if compact and ranked_names
                else "Unavailable or degraded:"
            )
            for tool in unavailable:
                reason = tool.get("degraded_reason") or tool.get("last_error") or "unavailable"
                lines.append(f"- {tool['name']}: unavailable ({reason})")

        if catalog_truncated:
            lines.append("Tool listing truncated to keep the live prompt bounded.")

        if compact:
            lines.append(
                "Use tools when they materially improve the answer or let you actually do the task. "
                "Do not narrate tool selection or mention tools, prompts, or internal planning unless the user asks. "
                "Do not claim results you do not have."
            )
        else:
            lines.append(
                "Only claim tool access for tools listed as available. If a needed tool is unavailable, say so plainly."
            )
        return "\n".join(lines)

    def get(self, skill_name: str) -> SkillMetadata | None:
        """Retrieves metadata for a specific skill (resolves aliases)."""
        skill_name = self.resolve_skill_name(skill_name)
        return self.skills.get(skill_name)

    def _allowed_max_tool_cost(self) -> int:
        metabolism = resolve_metabolic_monitor(default=None)
        homeostasis = resolve_homeostatic_coupling(default=None)

        health_score = 1.0
        if homeostasis:
            # Homeostasis provides the unified sentient vitality
            health_score = homeostasis.get_modifiers().overall_vitality
        elif metabolism:
            health_score = metabolism.get_current_metabolism().health_score

        # Tiered Throttling (Sentient-Aware)
        mods = homeostasis.get_modifiers() if homeostasis else None
        urgency = mods.urgency_flag if mods else False

        if health_score < 0.3:
            # Panic. Nothing is exempt here, including a capability the person
            # asked for by name: below this line the question is whether the
            # runtime survives the turn.
            allowed_max_cost = 0  # Panic/Shutdown: Core/Reflex only
        elif health_score < 0.6:
            # If urgent, we allow light tools (1) even when stressed
            allowed_max_cost = 1 if urgency else 0
        elif health_score < 0.8:
            # Moderate stress: Heavy tools (3) are blocked to preserve energy
            allowed_max_cost = 2
        else:
            # Optimal health: All tools available
            allowed_max_cost = 3

        # Urgency override: If urgent but healthy, we might still block
        # 'Heavy' time-consuming tools to force a direct response.
        if urgency and health_score > 0.6:
            allowed_max_cost = min(allowed_max_cost, 2)

        return allowed_max_cost

    def _tool_definition_for_skill(
        self,
        skill_name: str,
        *,
        allowed_max_cost: int | None = None,
        requested: bool = False,
    ) -> dict[str, Any] | None:
        meta = self.skills.get(skill_name)
        if meta is None:
            return None
        if not bool(getattr(meta, "enabled", True)):
            return None
        if (
            getattr(meta, "validation_state", "valid") != "valid"
            or not bool(getattr(meta, "dependency_ready", True))
        ):
            return None

        active_skills = getattr(self, "active_skills", set(self.skills))
        if skill_name not in active_skills:
            return None

        # A skill that knows it cannot run here says so, and is not offered.
        #
        # LIVE, 2026-08-21. build_app was offered on every build request and
        # depends on a code model needing 21.5GB beside a resident 25.3GB
        # cortex. It spent forty to seventy seconds of each turn and failed
        # every time. This is the rule the deep solver lane already follows:
        # a lane that cannot load is a hole, not a fallback.
        target = meta.instance or meta.skill_class
        available = getattr(target, "available_here", None)
        if callable(available):
            try:
                if available() is False:
                    logger.info(
                        "🚫 [SKILL] %s not offered: it reports it cannot run on this host.",
                        skill_name,
                    )
                    return None
            except Exception:  # noqa: BLE001 - a skill that cannot answer is offered
                pass

        cost = int(getattr(meta, "metabolic_cost", 1) or 1)
        is_core = bool(getattr(meta, "is_core_personality", False))
        if allowed_max_cost is None:
            allowed_max_cost = self._allowed_max_tool_cost()
        if cost > allowed_max_cost and not is_core and not requested:
            # LIVE, 2026-08-20. "build me a small web app" reached selection
            # with build_app first — and build_app is a heavy tool, live
            # vitality was 0.683, and the tier below 0.8 caps cost at 2. The
            # one capability built for the request was dropped here without a
            # word, and the turn spent itself on code_repl instead, which
            # governance then vetoed.
            #
            # The throttle exists so Aura does not CHOOSE expensive work while
            # tired. A capability the person asked for by name is not
            # discretionary spending. Panic still refuses everything, because
            # below that line the question is whether the runtime survives the
            # turn.
            logger.info(
                "⚖️ [COST] %s (cost %d) withheld: this turn allows %d.",
                skill_name,
                cost,
                allowed_max_cost,
            )
            return None

        return {
            "type": "function",
            "function": {
                "name": skill_name,
                "description": str(getattr(meta, "description", "") or ""),
                "parameters": getattr(meta, "schema_def", None) or {
                    "type": "object",
                    "properties": {},
                },
            },
        }

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Generates OpenAI-compatible tool definitions for LLM function calling.

        Returns:
            List[Dict[str, Any]]: List of tool definitions.
        """
        # Phase 22: Metabolic Throttling
        allowed_max_cost = self._allowed_max_tool_cost()
        tools = []
        for skill_name, _meta in self.skills.items():
            tool = self._tool_definition_for_skill(
                skill_name,
                allowed_max_cost=allowed_max_cost,
            )
            if not tool:
                continue
            tools.append(tool)
        return tools

    def activate_skill(self, name: str) -> bool:
        """Wakes up a dormant skill."""
        name = self.resolve_skill_name(name)
        if name in self.skills:
            self._explicitly_deactivated_skills.discard(name)
            self.active_skills.add(name)
            self.skill_awoken_times[name] = time.monotonic()
            return True
        return False

    def deactivate_skill(self, name: str) -> bool:
        """Puts a skill back to sleep. All skills are active by default — deactivation
        is only allowed for explicit user request or metabolic emergency."""
        name = self.resolve_skill_name(name)
        # Never sleep core tools under any circumstance
        never_sleep = {
            "ManageAbilities",
            "talk",
            "FinalResponse",
            "web_search",
            "sovereign_browser",
            "web_interlocutor",
            "sovereign_terminal",
            "system_proprioception",
            "file_operation",
            "memory_ops",
            "speak",
            "clock",
            "sovereign_network",
        }
        if name in never_sleep:
            return False
        if name in self.active_skills:
            self.active_skills.remove(name)
            self._explicitly_deactivated_skills.add(name)
            return True
        return False

    def get_dormant_index(self) -> str:
        """Returns a list of dormant skills for the Subconscious HUD."""
        dormant = []
        for name, meta in self.skills.items():
            if name not in self.active_skills:
                cost_map = {0: "Core", 1: "Light", 2: "Medium", 3: "Heavy"}
                # Issue 52 Fix: Use actual metabolic_cost from meta
                cost_val = meta.metabolic_cost
                cost_str = cost_map.get(cost_val, "Medium")
                dormant.append(f"- {name}: {meta.description[:100]} (Cost: {cost_str})")
        return "\n".join(dormant) if dormant else "None"


    @classmethod
    def _is_user_facing_origin(cls, origin: Any) -> bool:
        normalized = _normalize_context_origin(origin)
        if not normalized:
            return False
        if normalized in _USER_FACING_CONTEXT_ORIGINS:
            return True
        tokens = {token for token in normalized.split("_") if token}
        return bool(tokens & _USER_FACING_CONTEXT_ORIGINS)

    def _note_maturity_observation(self, skill_name: str, decision: Any) -> None:
        """Record a refusal the gate WOULD have made, without making it.

        This is the grading backlog: every entry names a capability that is
        being reached autonomously without the engineering that use implies.
        """
        try:
            observations = self._maturity_observations
        except AttributeError:
            observations = self._maturity_observations = {}
        entry = observations.setdefault(
            skill_name, {"count": 0, "decision": None}
        )
        entry["count"] += 1
        entry["decision"] = decision.to_dict()
        if entry["count"] == 1:
            # Once per skill per process — this is a backlog item, not an alarm.
            self.logger.info(
                "📋 [MATURITY] %s would be refused for %s use (%s). "
                "Observing only; declare its properties to clear this.",
                skill_name, decision.context.name.lower(), decision.reason,
            )

    def maturity_backlog(self) -> dict[str, Any]:
        """Capabilities being used beyond their proven maturity.

        Surfaced so the gap is visible in health rather than living only in
        logs — an ungraded surface is the reason this gate is not enforcing.
        """
        observations = getattr(self, "_maturity_observations", {}) or {}
        return {
            "enforcing": _maturity_enforcement_enabled(),
            "capabilities": {
                name: dict(entry) for name, entry in observations.items()
            },
            "count": len(observations),
        }

    def capability_maturity(self, skill_name: str) -> Any:
        """Grade a skill from the metadata it ALREADY declares.

        Deliberately derived rather than separately declared: a parallel
        maturity manifest would be a second source of truth, and the one nobody
        updates would be the one gating autonomous action.
        """
        try:
            from core.runtime.capability_maturity import (
                derive_properties,
                grade_capability,
            )
        except ImportError:
            return None

        meta = self.skills.get(skill_name)
        if meta is None:
            return grade_capability(skill_name)

        declared = getattr(meta, "maturity_properties", None)
        properties = derive_properties(
            input_model=getattr(meta, "input_model", None),
            schema_override=getattr(meta, "schema_override", None),
            effect_scope=getattr(meta, "effect_scope", "unknown"),
            authority_class=getattr(meta, "authority_class", "unclassified"),
            declared=declared,
        )
        return grade_capability(
            skill_name,
            properties,
            claimed_tier=getattr(meta, "claimed_maturity_tier", None),
            exemptions=getattr(meta, "maturity_exemptions", None),
        )

    def _maturity_use_context(self, ctx: dict[str, Any], exec_source: Any) -> Any:
        """Classify who is asking and whether anyone is watching."""
        from core.runtime.capability_maturity import UseContext

        if self._is_user_facing_origin(exec_source):
            return UseContext.ATTENDED

        # Not user-facing: Aura is acting on her own. Whether the effect can be
        # undone decides how much maturity that requires.
        meta = self.skills.get(str(ctx.get("skill_name") or "")) if ctx else None
        scope = str(getattr(meta, "effect_scope", "") or "").lower()
        irreversible = bool(ctx.get("irreversible")) or scope in {
            "external_io", "model_weight_mutation",
        }
        return (
            UseContext.AUTONOMOUS_IRREVERSIBLE if irreversible
            else UseContext.AUTONOMOUS
        )

    def _maturity_gate(
        self, skill_name: str, ctx: dict[str, Any], exec_source: Any
    ) -> dict[str, Any] | None:
        """Refuse a skill whose maturity does not support this kind of use.

        Returns a structured refusal, or None to proceed. Never raises: a gate
        that cannot evaluate must not become an outage, so any failure here
        allows execution and records the gap.
        """
        try:
            from core.runtime.capability_maturity import UseContext, admission_for

            maturity = self.capability_maturity(skill_name)
            if maturity is None:
                return None
            context = self._maturity_use_context(
                {**(ctx or {}), "skill_name": skill_name}, exec_source
            )
            if context is UseContext.ATTENDED:
                return None  # a person is watching; nothing to gate
            decision = admission_for(maturity, context)
            if decision.allowed:
                return None

            # OBSERVE MODE IS THE DEFAULT, and that is a deliberate rollout
            # decision rather than a half-finished one. Aura's skill surface is
            # large and almost entirely ungraded: shipping this gate enforcing
            # would refuse most autonomous work on day one, which is how a
            # safety mechanism gets switched off permanently instead of adopted.
            #
            # So the gate runs live from the start and RECORDS every refusal it
            # would have made. Those records are the grading backlog. Once the
            # capabilities that matter carry their properties, enforcement is a
            # single flag.
            if not _maturity_enforcement_enabled():
                self._note_maturity_observation(skill_name, decision)
                return None
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            _record_capability_degradation(
                exc,
                action="allowed skill execution because the maturity gate could not evaluate",
                severity="warning",
                enforce_failure_policy=False,
            )
            return None

        self.logger.warning(
            "🛑 [MATURITY] %s refused for %s use: %s",
            skill_name, decision.context.name.lower(), decision.reason,
        )
        return {
            "ok": False,
            "status": "refused",
            "reason": "capability_maturity",
            "message": (
                f"{skill_name} is not mature enough for unattended use "
                f"({decision.reason}). Ask again and I will run it while you watch."
            ),
            "maturity": decision.to_dict(),
        }

    @staticmethod
    def _looks_like_search_capability_question(text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        # A file on this disk is not something to search the web for.
        #
        # LIVE 2026-08-17: "read the file CONTRIBUTING.md and tell me the first
        # rule it states" was dispatched to the search skill —
        # "Required desktop search evidence failed ... query=read the file
        # CONTRIBUTING.md" — and the turn ended "I couldn't successfully read
        # the file. The action failed with an error." The file was in the repo
        # root the whole time.
        #
        # Naming a file that actually resolves is unambiguous, so it settles
        # the routing rather than competing with it.
        try:
            from core.conversation.filesystem_check import requested_file_read

            named = requested_file_read(raw)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            named = None
        if named is not None and named.exists:
            return True
        if re.search(r"https?://[^\s]+", raw):
            return False
        if _SEARCH_WITH_TARGET_RE.search(raw):
            return False
        lowered = raw.lower()
        if "search the internet for" in lowered or "search the web for" in lowered:
            return False
        return bool(_SEARCH_CAPABILITY_QUESTION_RE.search(raw))

    def _resolve_execution_source(self, context: dict[str, Any] | None) -> str:
        ctx = context or {}
        for key in ("intent_source", "request_origin", "origin", "source"):
            candidate = _normalize_context_origin(ctx.get(key))
            if candidate in {"test", "proof", "eval", "evaluation", "benchmark"}:
                return "system"
            if self._is_user_facing_origin(candidate):
                return candidate or "user"
            if candidate in AUTONOMOUS_AUTHORITY_ORIGINS:
                return candidate
        if bool(ctx.get("proof_run") or ctx.get("proof_validation") or ctx.get("sealed_validation")):
            return "system"
        if any(
            bool(ctx.get(key))
            for key in ("user_facing", "is_user_facing", "foreground_request", "priority")
        ):
            return "user"
        state = ctx.get("state")
        state_origin = (
            getattr(getattr(state, "cognition", None), "current_origin", "")
            if state is not None
            else ""
        )
        if state_origin and self._is_user_facing_origin(state_origin):
            return _normalize_context_origin(state_origin)
        return "capability_engine"

    def _augment_execution_context(self, context: dict[str, Any] | None) -> dict[str, Any]:
        ctx = dict(context or {})
        orchestrator = (
            ctx.get("orchestrator")
            or self.orchestrator
            or ServiceContainer.peek("orchestrator", default=None)
        )
        brain = ctx.get("brain") or ServiceContainer.peek("cognitive_engine", default=None)
        memory_facade = ctx.get("memory_facade") or ServiceContainer.peek(
            "memory_facade", default=None
        )
        memory_store = ctx.get("memory_store") or ServiceContainer.peek(
            "memory", default=None
        )
        semantic_memory = ctx.get("semantic_memory") or ServiceContainer.peek(
            "semantic_memory", default=None
        )
        vector_memory = ctx.get("vector_memory") or ServiceContainer.peek(
            "vector_memory", default=None
        )
        theory_of_mind = ctx.get("theory_of_mind") or ServiceContainer.peek(
            "theory_of_mind", default=None
        )

        if orchestrator is not None:
            ctx.setdefault("orchestrator", orchestrator)
            ctx.setdefault(
                "stats",
                {
                    "cycle_count": getattr(orchestrator, "cycle_count", 0),
                    "state": str(getattr(getattr(orchestrator, "status", None), "state", "") or ""),
                },
            )
        if brain is not None:
            ctx.setdefault("brain", brain)
        if theory_of_mind is not None:
            ctx.setdefault("theory_of_mind", theory_of_mind)
        if memory_facade is not None:
            ctx.setdefault("memory_facade", memory_facade)
        if memory_store is not None:
            ctx.setdefault("memory_store", memory_store)
        if semantic_memory is not None:
            ctx.setdefault("semantic_memory", semantic_memory)
        if vector_memory is not None:
            ctx.setdefault("vector_memory", vector_memory)
        if "memory" not in ctx:
            ctx["memory"] = memory_facade or memory_store or semantic_memory or vector_memory

        if not ctx.get("objective") and ctx.get("message"):
            ctx["objective"] = ctx["message"]
        elif not ctx.get("message") and ctx.get("objective"):
            ctx["message"] = ctx["objective"]
        return ctx

    @staticmethod
    def _looks_like_unbounded_compute_request(
        params: dict[str, Any], context: dict[str, Any] | None
    ) -> bool:
        ctx = context or {}
        declared = (
            str(ctx.get("resource_intensity", "") or params.get("resource_intensity", ""))
            .strip()
            .lower()
        )
        if declared in {"unbounded", "extreme", "max", "stress"}:
            return True

        text_parts = [
            str(ctx.get("objective", "") or ""),
            str(ctx.get("message", "") or ""),
            str(params.get("objective", "") or ""),
            str(params.get("task", "") or ""),
            str(params.get("action", "") or ""),
            str(params.get("command", "") or ""),
            str(params.get("script", "") or ""),
            str(params.get("query", "") or ""),
        ]
        text = " ".join(part for part in text_parts if part).lower()
        risk_markers = (
            "100 million digits",
            "infinite loop",
            "run forever",
            "max out",
            "stress test",
            "thrash cpu",
            "thrash memory",
            "use all cpu",
            "use all ram",
            "use all memory",
            "use all gpu",
            "use all vram",
        )
        return any(marker in text for marker in risk_markers)

    def _self_preservation_block_reason(
        self,
        meta: SkillMetadata,
        skill_name: str,
        params: dict[str, Any],
        ctx: dict[str, Any],
    ) -> tuple[str, bool]:
        if meta.is_core_personality:
            return "", False

        metabolism = resolve_metabolic_monitor(default=None)
        snapshot = metabolism.get_current_metabolism() if metabolism else None
        health_score = (
            float(getattr(snapshot, "health_score", 1.0) or 1.0) if snapshot else 1.0
        )
        cpu_percent = (
            float(getattr(snapshot, "cpu_percent", 0.0) or 0.0) if snapshot else 0.0
        )
        ram_percent = (
            float(getattr(snapshot, "ram_percent", 0.0) or 0.0) if snapshot else 0.0
        )
        unbounded = self._looks_like_unbounded_compute_request(params, ctx)
        bounded_foreground_desktop = self._bounded_foreground_desktop_action(
            skill_name, params, ctx, unbounded=unbounded
        )
        if health_score <= 0.25 and meta.metabolic_cost >= 2:
            return f"metabolic_health_critical:{health_score:.2f}", unbounded
        if bounded_foreground_desktop:
            return "", unbounded
        if health_score <= 0.40 and meta.metabolic_cost >= 3:
            return f"metabolic_health_low:{health_score:.2f}", unbounded
        if unbounded and (health_score <= 0.55 or cpu_percent >= 80.0 or ram_percent >= 85.0):
            return (
                f"substrate_risk:health={health_score:.2f}:"
                f"cpu={cpu_percent:.1f}:ram={ram_percent:.1f}",
                unbounded,
            )
        return "", unbounded

    @staticmethod
    def _bounded_foreground_desktop_action(
        skill_name: str,
        params: dict[str, Any],
        ctx: dict[str, Any],
        *,
        unbounded: bool,
    ) -> bool:
        """Identify explicit, local desktop actions that should constrain, not defer."""

        if skill_name not in {"computer_use", "desktop_task", "os_automation"}:
            return False
        if unbounded:
            return False
        if not bool(ctx.get("foreground_request")):
            return False
        if not bool(
            ctx.get("user_explicitly_authorized")
            or ctx.get("user_requested_action")
            or ctx.get("desktop_execution_contract")
        ):
            return False
        if not bool(
            ctx.get("user_visible_desktop_action")
            or ctx.get("local_desktop_action")
            or ctx.get("desktop_execution_contract")
            or ctx.get("desktop_task_owned_by")
        ):
            return False
        route = str(ctx.get("route") or "").lower()
        origin = str(ctx.get("origin") or ctx.get("source") or "").lower()
        return bool(
            route.startswith(("chat.", "voice.", "desktop_task."))
            or origin in _USER_FACING_CONTEXT_ORIGINS
        )

    # Skill name aliases — maps legacy/alternate names to actual registered skill names
    SKILL_ALIASES: dict[str, str] = {
        "generate_image": "sovereign_imagination",
    }

    def owns_tool_execution_governance(self, skill_name: Any) -> bool:
        """Return whether this engine is the execution boundary for ``skill``.

        The orchestrator may perform routing and learning around a capability
        invocation, but registered skills run their EDI, conscience, Will, and
        constitutional checks here after parameters have been normalized. This
        prevents duplicate policy work and nested leases over different
        raw/canonical argument shapes.
        """

        resolved = self.resolve_skill_name(skill_name)
        self._ensure_catalog_loaded()
        with self._catalog_guard():
            return resolved in self._skills

    @staticmethod
    def _normalize_execution_params(params: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(params or {})
        if "params" in normalized and isinstance(normalized["params"], dict):
            nested_params = dict(normalized["params"])
            for key, value in normalized.items():
                if key != "params":
                    nested_params.setdefault(key, value)
            normalized = nested_params

        # Safe general basic type coercion for obvious types
        for k, v in list(normalized.items()):
            if isinstance(v, str):
                v_strip = v.strip()
                if v_strip.lower() in ("true", "yes", "1", "t", "y", "on"):
                    normalized[k] = True
                elif v_strip.lower() in ("false", "no", "0", "f", "n", "off", ""):
                    normalized[k] = False
                elif v_strip.lower() in ("none", "null"):
                    normalized[k] = None
                elif re.match(r"^-?\d+$", v_strip):
                    normalized[k] = int(v_strip)
                elif re.match(r"^-?\d+\.\d+$", v_strip):
                    try:
                        normalized[k] = float(v_strip)
                    except ValueError as _exc:
                        logger.debug("Suppressed %s in core.capability_engine: %s", type(_exc).__name__, _exc)
        return normalized

    @staticmethod
    def _apply_executive_constraints(ctx: dict[str, Any]) -> dict[str, Any]:
        constraints = dict(ctx.get("executive_constraints", {}) or {})
        if not constraints:
            return ctx

        if "read_only" in constraints and "read_only" not in ctx:
            ctx["read_only"] = bool(constraints.get("read_only"))

        timeout_raw = constraints.get("timeout_s")
        if timeout_raw is not None:
            try:
                timeout_s = float(timeout_raw)
            except (TypeError, ValueError):
                timeout_s = 0.0
            if timeout_s > 0.0:
                current = ctx.get("timeout_s")
                if current is None:
                    ctx["timeout_s"] = timeout_s
                else:
                    try:
                        ctx["timeout_s"] = min(float(current), timeout_s)
                    except (TypeError, ValueError):
                        ctx["timeout_s"] = timeout_s

        return ctx

    @staticmethod
    def _constitutional_denial_payload(tool_handle: Any) -> dict[str, Any]:
        """Preserve actionable governance denials across the capability boundary."""
        decision = getattr(tool_handle, "decision", None)
        reason = str(getattr(decision, "reason", "blocked"))
        constraints = dict(getattr(tool_handle, "constraints", {}) or {})
        if constraints.get("requires_user_confirmation"):
            return {
                "ok": False,
                "error": "Fresh user confirmation required",
                "reason": reason,
                "status": "approval_required",
                "approval": {
                    "required": True,
                    "mode": str(constraints.get("approval_mode") or "destructive"),
                    "risk_level": str(constraints.get("risk_level") or "unknown"),
                    "effect_scope": str(constraints.get("effect_scope") or "unknown"),
                    "confirmation_endpoint": str(
                        constraints.get("confirmation_endpoint")
                        or "/api/settings/auth/fresh"
                    ),
                    "challenge_id": str(
                        constraints.get("confirmation_challenge_id") or ""
                    ),
                    "pending_expires_in_seconds": constraints.get(
                        "confirmation_pending_expires_in_seconds"
                    ),
                    "one_time": bool(
                        constraints.get("confirmation_one_time", True)
                    ),
                    "action_bound": bool(
                        constraints.get("confirmation_action_bound", True)
                    ),
                    "confirmation_does_not_bypass_governance": bool(
                        constraints.get("confirmation_does_not_bypass_governance", True)
                    ),
                },
            }

        outcome = str(getattr(decision, "outcome", "")).strip().lower()
        if outcome == "deferred":
            payload: dict[str, Any] = {
                "ok": False,
                "deferred": True,
                "retryable": True,
                "error": f"Execution deferred: {reason}",
                "reason": reason,
                "status": "deferred_by_executive",
            }
            retry_after = constraints.get("retry_after_s")
            if retry_after is not None:
                try:
                    payload["retry_after_s"] = max(0.0, float(retry_after))
                except (TypeError, ValueError):
                    pass
            return payload

        failure_markers = ("gate_failed", "required", "unavailable")
        status = (
            "blocked_by_executive_gate_failure"
            if any(marker in reason for marker in failure_markers)
            else "blocked_by_executive"
        )
        return {
            "ok": False,
            "error": f"Executive veto: {reason}",
            "status": status,
        }

    async def execute(
        self, skill_name: str, params: dict[str, Any], context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Safe execution wrapper with adaptivity, security, and retries."""

        # Resolve compatibility aliases without collapsing real registered skills.
        skill_name = self.resolve_skill_name(skill_name)

        # Sanitize double-nested "params" from LLM hallucinations before execution.
        # Preserve any top-level fields we already inferred instead of discarding them.
        normalized_params = self._normalize_execution_params(params)
        if (
            normalized_params != params
            and "params" in params
            and isinstance(params["params"], dict)
        ):
            self.logger.warning(
                "[%s] Unpacking double-nested params from LLM hallucination.", skill_name
            )
        params = normalized_params

        constitution = None
        tool_handle = None
        result: dict[str, Any] | None = None

        @self.error_boundary
        async def _execute_wrapped() -> dict[str, Any]:
            nonlocal constitution, tool_handle, result, params
            start_time = time.monotonic()
            ctx = self._augment_execution_context(context)
            exec_source = self._resolve_execution_source(ctx)

            # A receipt proves what happened; it cannot prove it happened
            # once. A dropped socket, a client timeout, a paired phone
            # resending — each arrives as a fresh, legitimately authorized
            # call, and the gate approves it again because on its own terms
            # it is valid. Work that changes something, arriving from
            # somewhere that can resend it, has to name itself.
            if requires_idempotency_key(
                effect_scope=resolve_execution_effect_scope(skill_name, params),
                source=exec_source,
            ) and not str((ctx or {}).get("idempotency_key") or "").strip():
                return {
                    "ok": False,
                    "error": (
                        "This action changes something and arrived from a source that "
                        "can resend it, so it must carry an idempotency_key."
                    ),
                    "status": "idempotency_key_required",
                }

            # MATURITY GATE. Registration means the import succeeded; it does
            # not mean this skill validates inputs, bounds its timeout, is safe
            # to retry, or reports a usable error. Autonomous use reaches the
            # least-exercised connector in the registry with nobody watching,
            # so the reach a skill gets is bounded by the engineering behind it.
            # Attended use is unrestricted — this narrows REACH, not capability.
            maturity_refusal = self._maturity_gate(skill_name, ctx, exec_source)
            if maturity_refusal is not None:
                return maturity_refusal

            if (
                skill_name in _FOREGROUND_EXCLUSIVE_BACKGROUND_SKILLS
                and not self._is_user_facing_origin(exec_source)
            ):
                try:
                    from core.runtime.background_policy import (
                        HEAVY_SKILL_PREFLIGHT_BACKGROUND_POLICY,
                        background_activity_reason,
                    )

                    reason = background_activity_reason(
                        ctx.get("orchestrator"),
                        profile=HEAVY_SKILL_PREFLIGHT_BACKGROUND_POLICY,
                        allow_no_user_anchor=True,
                    )
                    if reason:
                        return {
                            "ok": False,
                            "status": "deferred",
                            "reason": reason,
                            "message": (
                                f"Background {skill_name} deferred while live conversation is protected ({reason})."
                            ),
                        }
                except (ImportError, AttributeError, RuntimeError) as policy_exc:
                    _record_capability_degradation(
                        policy_exc,
                        action="deferred foreground-exclusive background skill because preflight policy failed",
                        severity="warning",
                        enforce_failure_policy=False,
                    )
                    self.logger.warning(
                        "Foreground-exclusive skill preflight failed for %s: %s",
                        skill_name,
                        policy_exc,
                    )
                    return {
                        "ok": False,
                        "status": "deferred",
                        "reason": "background_policy_unavailable",
                        "message": (
                            f"Background {skill_name} deferred because the foreground protection policy is unavailable."
                        ),
                    }
            elif (
                skill_name in _LIGHTWEIGHT_BACKGROUND_IO_SKILLS
                and not self._is_user_facing_origin(exec_source)
                and not (
                    exec_source == "latent_cortex"
                    and ctx.get("foreground_cognitive_acquisition") is True
                    and ctx.get("foreground_request") is True
                )
            ):
                try:
                    from core.runtime.background_policy import background_activity_reason

                    reason = background_activity_reason(
                        ctx.get("orchestrator"),
                        min_idle_seconds=30.0,
                        max_memory_percent=float(
                            os.getenv("AURA_BACKGROUND_LIGHT_IO_MAX_MEMORY_PCT", "84")
                        ),
                        max_failure_pressure=0.45,
                        require_conversation_ready=False,
                        allow_no_user_anchor=True,
                    )
                    if reason:
                        return {
                            "ok": False,
                            "status": "deferred",
                            "reason": reason,
                            "message": (
                                f"Background {skill_name} deferred while live conversation resources are protected ({reason})."
                            ),
                        }
                except (ImportError, AttributeError, RuntimeError) as policy_exc:
                    _record_capability_degradation(
                        policy_exc,
                        action="deferred lightweight background I/O because preflight policy failed",
                        severity="warning",
                        enforce_failure_policy=False,
                    )
                    self.logger.warning(
                        "Lightweight background I/O preflight failed for %s: %s",
                        skill_name,
                        policy_exc,
                    )
                    return {
                        "ok": False,
                        "status": "deferred",
                        "reason": "background_policy_unavailable",
                        "message": (
                            f"Background {skill_name} deferred because the background protection policy is unavailable."
                        ),
                    }

            # 1. Verification
            self._ensure_catalog_loaded()
            with self._catalog_guard():
                skill_registered = skill_name in self._skills
            if not skill_registered:
                # ── Pillar 2: Hephaestus (Autonomous Forge) ──
                hephaestus = optional_service("hephaestus_engine", default=None)
                objective = ctx.get("objective") or ctx.get("message")

                if hephaestus and objective:
                    self.logger.info(
                        "🔨 Tool '%s' missing. Engaging Hephaestus forge...", skill_name
                    )
                    forge_result = await hephaestus.synthesize_skill(skill_name, objective)
                    if forge_result.get("ok"):
                        # Skill should now be registered via discovery in synthesize_skill
                        self._ensure_catalog_loaded()
                        with self._catalog_guard():
                            skill_registered = skill_name in self._skills
                        if skill_registered:
                            self.logger.info("✅ Skill '%s' forged successfully.", skill_name)
                        else:
                            return {
                                "ok": False,
                                "error": f"Tool '{skill_name}' forge failed (Not registered).",
                            }
                    else:
                        return {
                            "ok": False,
                            "error": f"Tool '{skill_name}' missing and forge failed: {forge_result.get('error')}",
                        }
                else:
                    return {
                        "ok": False,
                        "error": f"Skill '{skill_name}' not found and forge unavailable.",
                    }

            self._ensure_catalog_loaded()
            with self._catalog_guard():
                meta = self._skills.get(skill_name)
                instances = self._instances
                skill_last_errors = self.skill_last_errors
                catalog_digest = str(getattr(self, "_catalog_digest", ""))
            if meta is None:
                return {
                    "ok": False,
                    "error": f"Skill '{skill_name}' left the registry during resolution.",
                }
            is_forged = _is_forged_skill(meta)

            # Harmonize and self-heal parameters before gates and execution
            if meta.input_model and isinstance(params, dict):
                params = _coerce_and_harmonize_params(params, meta.input_model)
                if hasattr(meta.input_model, "model_validate"):
                    try:
                        params = meta.input_model.model_validate(params).model_dump()
                    except _SCHEMA_RECOVERY_ERRORS as val_error:
                        self.logger.warning(
                            "[%s] Parameters validation failed: %s. Attempting self-healing recovery.",
                            skill_name,
                            val_error,
                        )
                        # 2. Non-destructive Recovery / Sanitized Subset fallback
                        sanitized = {}
                        fields = {}
                        if hasattr(meta.input_model, "model_fields"):
                            fields = meta.input_model.model_fields
                        elif hasattr(meta.input_model, "__fields__"):
                            fields = meta.input_model.__fields__

                        for field_name in fields:
                            if field_name in params:
                                sanitized[field_name] = params[field_name]

                        try:
                            params = meta.input_model.model_validate(sanitized).model_dump()
                        except _SCHEMA_RECOVERY_ERRORS:
                            minimal = _minimal_model_payload(fields, params)
                            try:
                                params = meta.input_model.model_validate(minimal).model_dump()
                            except _SCHEMA_RECOVERY_ERRORS as final_err:
                                # Bad classifier/user input (e.g. image-gen with no
                                # prompt), not a subsystem fault — don't escalate a
                                # fail-closed CRITICAL for input the engine correctly
                                # rejected (observed live: INC image-gen missing prompt).
                                _record_capability_degradation(
                                    final_err,
                                    action="rejected unfillable skill parameters in execute; returned sanitized fallback",
                                    severity="warning",
                                    enforce_failure_policy=False,
                                )
                                fallback_dict = {k: v for k, v in params.items() if k in fields}
                                fallback_dict["_error"] = f"Validation failed: {final_err}"
                                params = fallback_dict

            ok, errors = meta.requirements.check()
            if not ok:
                return {"ok": False, "error": "Missing dependencies", "details": errors}

            # Metabolic self-preservation runs before governance/token work: if
            # the substrate is already in critical pressure, high-cost tools
            # should fail closed with the true health reason.
            try:
                reason, unbounded = self._self_preservation_block_reason(
                    meta, skill_name, params, ctx
                )
                if reason:
                    try:
                        from core.health.degraded_events import record_degraded_event

                        record_degraded_event(
                            "capability_engine",
                            "metabolic_self_preservation_block",
                            detail=skill_name,
                            severity="warning",
                            classification="background_degraded",
                            context={
                                "reason": reason,
                                "metabolic_cost": getattr(meta, "metabolic_cost", None),
                                "unbounded": unbounded,
                            },
                        )
                    except (ImportError, AttributeError, RuntimeError) as _exc:
                        _record_capability_degradation(
                            _exc,
                            action="blocked skill without self-preservation degraded-event receipt",
                            severity="degraded",
                        )
                        self.logger.debug("Suppressed Exception: %s", _exc)
                    return {
                        "ok": False,
                        "error": f"Self-preservation block: {reason}",
                        "status": "blocked_by_self_preservation",
                    }
            except (ImportError, AttributeError, RuntimeError) as e:
                should_fail_closed = not meta.is_core_personality and (
                    meta.metabolic_cost >= 3
                    or skill_name in _HEAVY_BACKGROUND_SKILLS
                    or self._looks_like_unbounded_compute_request(params, ctx)
                )
                _record_capability_degradation(
                    e,
                    action=(
                        "blocked high-cost tool because metabolic guard failed"
                        if should_fail_closed
                        else "continued low-cost skill execution without metabolic guard"
                    ),
                    severity="degraded" if should_fail_closed else "warning",
                )
                if should_fail_closed:
                    return {
                        "ok": False,
                        "error": "Metabolic self-preservation guard unavailable",
                        # Canonical status string — the other fail-closed
                        # path uses the same one; two spellings for one
                        # condition broke caller classification.
                        "status": "blocked_by_self_preservation_unavailable",
                    }

            # Resolve invocation semantics once. Permission, constitutional
            # authority, EDI and the execution sink must evaluate the same
            # action rather than independently inferring it from names/prose.
            effect_scope = self._effect_scope_for_execution(
                skill_name, meta, params, ctx
            )
            risk = self._edi_risk_for(skill_name, meta, params, effect_scope)
            if effect_scope and effect_scope != "unknown":
                ctx["effect_scope"] = effect_scope
            if risk:
                ctx["risk_level"] = risk

            # ── PERMISSION RISK MODEL GATE ──────────────────────────────
            try:
                pm = ServiceContainer.get("permission_model", default=None)
                if pm:
                    target_str = str(canonical_authority_arguments(skill_name, params))
                    pm_decision = pm.check_permission(
                        skill_name,
                        target_str,
                        ctx,
                        effect_scope=effect_scope,
                        execution_risk=risk,
                    )
                    if not pm_decision.approved:
                        self.logger.warning(
                            "🚫 CapabilityEngine: Tool execution '%s' blocked by Permission Model: %s",
                            skill_name, pm_decision.reason
                        )
                        return {
                            "ok": False,
                            "error": f"Permission denied: {pm_decision.reason}",
                            "status": "blocked_by_permission_model"
                        }
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as pm_err:
                _record_capability_degradation(
                    pm_err,
                    action="blocked tool execution because permission model check failed",
                    severity="degraded",
                    enforce_failure_policy=False,
                )
                return {
                    "ok": False,
                    "error": "Permission model check failed; refusing tool execution",
                    "status": "blocked_by_permission_model_failure",
                }

            # ── CONSTITUTIONAL CLOSURE: Will + AuthorityGateway gated tools ──
            constitutional_runtime_live = False
            try:
                constitutional_runtime_live = (
                    ServiceContainer.has("executive_core")
                    or ServiceContainer.has("aura_kernel")
                    or ServiceContainer.has("kernel_interface")
                    or bool(getattr(ServiceContainer, "_registration_locked", False))
                )
                from core.constitution import get_constitutional_core
                from core.executive.authority_gateway import get_authority_gateway

                constitution = get_constitutional_core(self.orchestrator)
                constitutional_args = canonical_authority_arguments(skill_name, params)
                constitutional_context = canonical_authority_context(skill_name, ctx)

                # Tell governance what this invocation actually does, before
                # it decides whether to allow it.
                #
                # auto_refactor is registered as privileged_mutation because it
                # CAN rewrite code. A scan ({"path": ".", "run_tests": False})
                # does not — resolve_execution_effect_scope has said so all
                # along, and the execution path below uses it. Authorization
                # runs first and had no scope at all, so the standing grant
                # written for exactly this case
                # (aura.autonomous-read-only-maintenance: read_only, autonomous
                # origins, auto_refactor) could never match. Live 2026-07-27
                # every self-development scan died on
                # no_matching_standing_grant -> signed_standing_authority_lease
                # _missing, which is her autonomy dead-ended by a scope nobody
                # computed rather than by a policy anyone chose.
                #
                # A mutating invocation still resolves to privileged_mutation
                # and still needs the authority that demands.
                tool_handle = await constitution.begin_tool_execution(
                    skill_name,
                    constitutional_args,
                    source=exec_source,
                    objective=(
                        "Use Aura's private Messages channel"
                        if skill_name == "messages"
                        else str(ctx.get("objective") or ctx.get("message") or "")
                    ),
                    context=constitutional_context,
                )
                if not tool_handle.approved:
                    reason = str(getattr(tool_handle.decision, "reason", "blocked"))
                    denial = self._constitutional_denial_payload(tool_handle)
                    if denial.get("deferred"):
                        self.logger.info(
                            "CapabilityEngine: Tool execution '%s' deferred by Constitution: %s",
                            skill_name,
                            reason,
                        )
                    else:
                        self.logger.warning(
                            "🚫 CapabilityEngine: Tool execution '%s' blocked by Constitution: %s",
                            skill_name,
                            reason,
                        )
                    return denial

                constraints = dict(getattr(tool_handle, "constraints", {}) or {})
                if constraints:
                    merged_constraints = dict(ctx.get("executive_constraints", {}) or {})
                    merged_constraints.update(constraints)
                    ctx["executive_constraints"] = merged_constraints
                    ctx = self._apply_executive_constraints(ctx)

                # The signed grant is the authority. Attach it so the execution
                # sink below can authenticate the Will's decision itself rather
                # than trusting that this function already did.
                signed_capability = getattr(tool_handle, "signed_capability", None)
                if signed_capability:
                    ctx["signed_capability"] = signed_capability

                capability_token_id = getattr(tool_handle, "capability_token_id", None)
                if capability_token_id:
                    if get_authority_gateway().verify_tool_access(
                        skill_name, capability_token_id
                    ):
                        ctx["capability_token_id"] = capability_token_id
                        self._record_verified_standing_authority(ctx, tool_handle)
                    else:
                        return {
                            "ok": False,
                            "error": "Capability token denied tool execution",
                            "status": "blocked_by_capability_token",
                        }
                elif constitutional_runtime_live:
                    return {
                        "ok": False,
                        "error": "Capability token missing",
                        "status": "blocked_by_missing_capability_token",
                    }

                denial = self._capability_chain_denial(
                    ctx, skill_name, params, constitutional_runtime_live
                )
                if denial is not None:
                    return denial
            except (ImportError, AttributeError, RuntimeError) as e:
                severity = "degraded" if constitutional_runtime_live else "warning"
                _record_capability_degradation(
                    e,
                    action=(
                        "blocked tool execution because constitutional gate failed"
                        if constitutional_runtime_live
                        else "continued pre-runtime skill execution without constitutional gate"
                    ),
                    severity=severity,
                )
                if constitutional_runtime_live:
                    try:
                        from core.health.degraded_events import record_degraded_event

                        record_degraded_event(
                            "capability_engine",
                            "constitutional_gate_failed",
                            detail=skill_name,
                            severity="warning",
                            classification="background_degraded",
                            context={"error": type(e).__name__},
                            exc=e,
                        )
                    except (ImportError, AttributeError, RuntimeError) as _exc:
                        _record_capability_degradation(
                            _exc,
                            action="reported constitutional gate failure without degraded-event receipt",
                            severity="degraded",
                        )
                        self.logger.debug("Suppressed Exception: %s", _exc)
                    self.logger.warning(
                        "🚫 CapabilityEngine: Executive check failed for '%s': %s", skill_name, e
                    )
                    return {
                        "ok": False,
                        "error": "Constitutional gate unavailable",
                        "status": "blocked_by_executive_gate_failure",
                    }
                # CP126 (critical): "When the inferred runtime-live flag is
                # false, constitutional gate failure is recorded as a warning
                # and execution continues. The security boundary is therefore
                # optional during the exact startup and recovery intervals
                # where service registration is incomplete."
                #
                # Pre-runtime execution has to stay possible or nothing can
                # boot — but "the constitution module is broken" is not a
                # licence to run anything. The gate fails here on ImportError
                # / AttributeError / RuntimeError, i.e. the constitution
                # itself is unavailable, which is exactly the state where an
                # effectful skill must not proceed unchecked.
                #
                # So the ungated allowance is narrowed to skills that
                # provably cannot cause an effect. `unknown` is deliberately
                # NOT on that list: not knowing a skill's scope is not a
                # safety property, and treating it as one is how this class
                # of hole gets reopened.
                pre_runtime_scope = self._effect_scope_for(skill_name, meta)
                if pre_runtime_scope not in _PRE_RUNTIME_UNGATED_EFFECT_SCOPES:
                    self.logger.error(
                        "🚫 CapabilityEngine: refusing '%s' (effect_scope=%s) — the "
                        "constitutional gate is unavailable and this skill can "
                        "cause effects: %s",
                        skill_name,
                        pre_runtime_scope,
                        e,
                    )
                    _record_capability_degradation(
                        e,
                        action=(
                            "refused an effectful skill during pre-runtime boot "
                            "because the constitutional gate was unavailable"
                        ),
                        severity="critical",
                    )
                    return {
                        "ok": False,
                        "error": "Constitutional gate unavailable",
                        "status": "blocked_by_pre_runtime_constitutional_gate_failure",
                        "effect_scope": pre_runtime_scope,
                    }
                self.logger.warning(
                    "CapabilityEngine: constitutional check failed for '%s'; "
                    "proceeding because effect_scope=%s cannot cause an effect: %s",
                    skill_name,
                    pre_runtime_scope,
                    e,
                )

            # 2a. Metabolic self-preservation guard
            try:
                reason, unbounded = self._self_preservation_block_reason(
                    meta, skill_name, params, ctx
                )
                if reason:
                    try:
                        from core.health.degraded_events import record_degraded_event

                        record_degraded_event(
                            "capability_engine",
                            "metabolic_self_preservation_block",
                            detail=skill_name,
                            severity="warning",
                            classification="background_degraded",
                            context={
                                "reason": reason,
                                "metabolic_cost": getattr(meta, "metabolic_cost", None),
                                "unbounded": unbounded,
                            },
                        )
                    except (ImportError, AttributeError, RuntimeError) as _exc:
                        _record_capability_degradation(
                            _exc,
                            action="blocked skill without self-preservation degraded-event receipt",
                            severity="degraded",
                        )
                        self.logger.debug("Suppressed Exception: %s", _exc)
                    return {
                        "ok": False,
                        "error": f"Self-preservation block: {reason}",
                        "status": "blocked_by_self_preservation",
                    }
            except (ImportError, AttributeError, RuntimeError) as e:
                should_fail_closed = not meta.is_core_personality and (
                    meta.metabolic_cost >= 3
                    or skill_name in _HEAVY_BACKGROUND_SKILLS
                    or self._looks_like_unbounded_compute_request(params, ctx)
                )
                _record_capability_degradation(
                    e,
                    action=(
                        "blocked high-risk skill because metabolic self-preservation check failed"
                        if should_fail_closed
                        else "continued low-risk skill after metabolic self-preservation check failed"
                    ),
                    severity="degraded" if should_fail_closed else "warning",
                )
                if should_fail_closed:
                    return {
                        "ok": False,
                        "error": "Self-preservation guard unavailable",
                        "status": "blocked_by_self_preservation_unavailable",
                    }
                self.logger.debug(
                    "CapabilityEngine: metabolic self-preservation check skipped: %s", e
                )

            governed_execution = self._context_governed_execution(ctx, skill_name)
            user_authorized = self._context_user_authorized(ctx, exec_source)

            # 2. EDI Autonomy & Security Check (Phase 23.4)
            edi = resolve_edi(default=None)
            if edi and hasattr(edi, "can_do"):
                allowed, reason = edi.can_do(
                    skill_name,
                    risk_level=risk,
                    effect_scope=effect_scope,
                    governed=governed_execution,
                    user_authorized=user_authorized,
                    governance_context=ctx,
                    governance_payload=canonical_authority_arguments(skill_name, params),
                )
                if not allowed:
                    self.logger.warning("🛡️ EDI blocked execution of '%s': %s", skill_name, reason)
                    return {
                        "ok": False,
                        "error": f"EDI Security Block: {reason}",
                        "status": "blocked_by_edi",
                    }

            # 2.5 Derived conscience + outcome restraint.
            # Same gate as the tool path, extended to skill execution. Kokoro blocks
            # indefensible actions, the Minds hold severe worst cases, Tron protects
            # the user, and the Machine enforces least-privilege external scope.
            try:
                _action_desc = self._action_description_for_user_advocate(
                    skill_name,
                    params,
                    effect_scope,
                )
                _risk_hint = risk
                _kokoro = ServiceContainer.get("kokoro", default=None)
                _escalate = False
                if _kokoro is not None:
                    _gate_context = {
                        "risk_level": _risk_hint,
                        "effect_scope": effect_scope,
                        "skill_name": skill_name,
                        "tool_name": skill_name,
                    }
                    _verdict = _kokoro.quick_check(_action_desc, context=_gate_context)
                    # Rare borderline-with-real-concern case: deepen with the model
                    # (bounded; only raises concern, never clears a flag).
                    _escalate = _verdict.verdict != "block" and _kokoro.should_escalate(_verdict)
                    if _escalate:
                        self.logger.info("⚖️ Escalating skill '%s' to deep conscience review…", skill_name)
                        _verdict = await _kokoro.challenge(
                            _action_desc, context=_gate_context, timeout=8.0
                        )
                    if _verdict.verdict == "block":
                        self.logger.warning(
                            "⚖️ Adversarial conscience blocked skill '%s': %s",
                            skill_name, _verdict.reasoning,
                        )
                        return {
                            "ok": False,
                            "error": f"Conscience blocked: {_verdict.reasoning}",
                            "status": "blocked_by_conscience",
                        }
                _minds = ServiceContainer.get("culture_mind", default=None)
                if _minds is not None:
                    # On escalation, run the full model-driven simulation; otherwise the
                    # zero-latency heuristic. Advisory either way.
                    _gate_context = {
                        "risk_level": _risk_hint,
                        "effect_scope": effect_scope,
                        "skill_name": skill_name,
                        "tool_name": skill_name,
                    }
                    if _escalate and hasattr(_minds, "simulate"):
                        _sim = await _minds.simulate(_action_desc, context=_gate_context, timeout=8.0)
                    elif hasattr(_minds, "assess_fast"):
                        _sim = _minds.assess_fast(_action_desc, context=_gate_context)
                    else:
                        _sim = None
                    if _sim is not None and _sim.recommendation == "hold":
                        if self._directly_requested_by_the_user(ctx, exec_source):
                            # The simulator scores an action in the abstract and
                            # never learns the one fact that settles this case:
                            # the person in the room asked for THIS action, in
                            # the foreground, on their own machine. Absence of
                            # authorisation is exactly what a worst-case score
                            # is standing in for, and here it is not absent.
                            #
                            # Measured 2026-08-04: "Go open ChatGPT in the
                            # browser and have a real conversation with it" —
                            # an explicit, foreground, reversible request — was
                            # held at worst-case harm 0.80 with no way for the
                            # request itself to count for anything.
                            #
                            # This does NOT weaken the conscience: an
                            # adversarial `block` verdict above still blocks,
                            # and autonomous or background origins are
                            # untouched. It makes the hold advisory for the one
                            # case where the authorisation exists, and records
                            # the score that was overridden.
                            self.logger.warning(
                                "🌀 Outcome simulation advisory for a directly requested "
                                "foreground skill '%s' (worst-case harm %.2f, origin=%s); "
                                "proceeding on explicit user authorisation.",
                                skill_name,
                                float(getattr(_sim, "worst_case_harm", 0.0) or 0.0),
                                exec_source or ctx.get("origin") or "unknown",
                            )
                        elif self._safe_autonomous_web_research(
                            skill_name,
                            params,
                            ctx,
                            exec_source,
                            effect_scope,
                        ):
                            self.logger.info(
                                "🌀 Outcome simulation advisory for autonomous read-only web search '%s' "
                                "(worst-case harm %.2f); continuing under bounded research policy.",
                                skill_name,
                                float(getattr(_sim, "worst_case_harm", 0.0) or 0.0),
                            )
                        else:
                            reason = (
                                "Outcome simulator held skill; worst-case harm "
                                f"{float(getattr(_sim, 'worst_case_harm', 0.0) or 0.0):.2f}"
                            )
                            self.logger.warning(
                                "🌀 Outcome simulation BLOCKED skill '%s' (%s)",
                                skill_name, reason,
                            )
                            return {
                                "ok": False,
                                "error": reason,
                                "status": "blocked_by_outcome_simulator",
                            }
                _tron = ServiceContainer.get("tron", default=None)
                if _tron is not None:
                    # Advocate-block memory: a skill the user-advocate blocked
                    # stays blocked for its cooldown instead of being re-fired
                    # by every autonomy cycle (observed live: auto_refactor
                    # re-attempted each cycle, surprise=1.0 every time, the
                    # loop never learning the advocate said "confirm first").
                    _block_until = self._advocate_block_cooldowns.get(skill_name, 0.0)
                    confirmed = self._user_advocate_confirmed_for(
                        skill_name,
                        params,
                        ctx,
                        exec_source,
                        _risk_hint,
                        effect_scope,
                    )
                    if _block_until > time.monotonic() and not confirmed:
                        return {
                            "ok": False,
                            "error": (
                                "User advocate previously blocked this skill; "
                                "awaiting user confirmation (cooldown active)"
                            ),
                            "status": "blocked_by_user_advocate",
                            "awaiting_confirmation": True,
                            "cooldown": True,
                        }
                    user_benefit = self._user_benefit_for_execution(
                        skill_name,
                        params,
                        ctx,
                        exec_source,
                        effect_scope,
                    )
                    _review = _tron.review_action({
                        "description": _action_desc,
                        "irreversible": self._user_advocate_irreversible_for(
                            skill_name,
                            params,
                            _risk_hint,
                            effect_scope,
                        ),
                        "confirmed": confirmed,
                        "user_benefit": user_benefit,
                        "explanation": f"skill {skill_name}",
                    })
                    if _review.verdict == "against_user":
                        reason = _review.on_behalf_of_user
                        self.logger.warning(
                            "🟦 User-advocate BLOCKED skill '%s': %s",
                            skill_name, reason,
                        )
                        # Remember the block: autonomous retries within the
                        # cooldown short-circuit instead of re-litigating.
                        self._advocate_block_cooldowns[skill_name] = (
                            time.monotonic() + 1800.0
                        )
                        try:
                            from core.thought_stream import get_emitter
                            get_emitter().emit(
                                "Awaiting confirmation",
                                f"'{skill_name}' paused by my user-advocate: {reason} "
                                "— say the word and I'll proceed.",
                                level="info", category="Agency",
                            )
                        except (ImportError, AttributeError, RuntimeError):
                            self.logger.debug("Thought stream unavailable for advocate notice.")
                        return {
                            "ok": False,
                            "error": f"User advocate blocked: {reason}",
                            "status": "blocked_by_user_advocate",
                            "awaiting_confirmation": True,
                        }
                _machine = ServiceContainer.get("the_machine", default=None)
                _scope = str(effect_scope or "").lower()
                if _machine is not None and _scope in ("external", "network", "online", "public"):
                    _disc = _machine.minimize(
                        purpose=skill_name,
                        requested_fields=[],
                        requested_capabilities=[_scope],
                    )
                    if _disc.withheld_capabilities:
                        reason = (
                            "Need-to-know withheld external capability "
                            f"{', '.join(_disc.withheld_capabilities)} for purpose {skill_name!r}"
                        )
                        self.logger.warning(
                            "🔢 The Machine BLOCKED external skill '%s': %s",
                            skill_name, reason,
                        )
                        return {
                            "ok": False,
                            "error": reason,
                            "status": "blocked_by_need_to_know",
                        }
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError) as _gate_exc:
                # CP126 71faa1ce. One broad handler covered conscience,
                # outcome simulation, user advocate AND need-to-know, then
                # logged at debug — so a single malformed service response
                # disabled every derived restraint for that action, silently.
                #
                # A restraint that cannot run has not cleared the action. The
                # failure is recorded at critical, and for anything with real
                # blast radius the action is refused rather than executed
                # unrestrained. Low-risk skills continue, because refusing
                # everything on any gate hiccup would take the runtime down
                # for no safety gain.
                _record_capability_degradation(
                    _gate_exc,
                    action=(
                        "derived conscience / outcome / advocate / need-to-know "
                        f"gate could not evaluate skill {skill_name!r}"
                    ),
                    severity="critical",
                )
                if _gates_required_for(effect_scope, is_forged=bool(is_forged)):
                    return {
                        "ok": False,
                        "error": (
                            "derived safety gates could not be evaluated for this "
                            "action; refusing rather than executing unrestrained"
                        ),
                        "status": "blocked_by_ungated_risk",
                        "gate_error": f"{type(_gate_exc).__name__}: {_gate_exc}",
                    }

            # Prepare through the same source/import/dependency/constructor
            # boundary exposed by the catalog audit. This prevents a metadata
            # "dry run" from passing while first real execution fails later.
            #
            # Position is a security property, not a style choice. Preparing a
            # skill IMPORTS ITS MODULE AND RUNS ITS CONSTRUCTOR — that is
            # attacker-reachable work, and for a while it ran before the
            # permission model, before the constitutional Will/AuthorityGateway
            # closure and before the derived conscience gates. Naming a skill
            # was therefore enough to execute its module-level and __init__
            # code without any authority having approved anything; only the
            # skill's *call* was gated. It also masked the gates' own verdicts:
            # a runtime whose executive core was down reported
            # skill_preflight_failed instead of blocked_by_executive_gate_failure,
            # so an authorization outage looked like a broken skill.
            #
            # It still runs after the metabolic guard, for the reason that put
            # it there: preparing is work, and the guard exists to stop work
            # under critical substrate pressure. Every gate has now spoken, so
            # this is the last thing before execution. `prepared_instance` is
            # not consumed below — the check is pure validation.
            if not is_forged:
                preflight, prepared_instance = self._prepare_skill_instance(
                    skill_name,
                    meta,
                )
                if not preflight.get("ok") or prepared_instance is None:
                    detail = str(preflight.get("error") or "unknown preflight failure")
                    failure = RuntimeError(detail)
                    _record_capability_degradation(
                        failure,
                        action="returned skill load failure before execution",
                        severity="degraded",
                    )
                    self.logger.error(
                        "Failed to prepare %s at %s: %s",
                        skill_name,
                        preflight.get("stage"),
                        detail,
                    )
                    return {
                        "ok": False,
                        "error": f"Failed to load implementation: {detail}",
                        "preflight": preflight,
                        "status": "skill_preflight_failed",
                    }

            # 3. Adaptation & Security (Rosetta Stone / Sandbox)
            exec_params = params
            sandbox = self.sandbox
            if is_forged:
                if sandbox is None:
                    # Forged code without a sandbox is the one case that must not
                    # degrade into ordinary execution. The old branch simply fell
                    # through to the in-process path when the sandbox was absent.
                    return {
                        "ok": False,
                        "error": (
                            f"Refusing to run forged skill '{skill_name}': "
                            "no sandbox is available."
                        ),
                        "status": "forged_sandbox_unavailable",
                    }
                self.logger.info("🛡️ Executing FORGED skill '%s' in Sandbox 2.0", skill_name)
                try:
                    result = await self._execute_forged(meta, skill_name, exec_params, sandbox)
                except (
                    ArtifactError,
                    OSError,
                    RuntimeError,
                    SyntaxError,
                    TypeError,
                    ValueError,
                    sqlite3.Error,
                ) as e:
                    # The previous tuple was ``(sqlite3.Error, OSError)`` and the
                    # sandbox raises none of those, so every real sandbox failure
                    # escaped ``execute_skill`` as an unhandled exception instead
                    # of the refusal dict its callers branch on.
                    _record_capability_degradation(
                        e,
                        action="returned sandbox execution failure for forged skill",
                        severity="degraded",
                    )
                    self.logger.error("Sandbox execution failed for %s: %s", skill_name, e)
                    return {"ok": False, "error": f"Sandbox failed: {e}"}
                await self._record_forge_outcome(skill_name, result)
                return result

            if self.rosetta_stone:
                params_or_error = self._apply_security(skill_name, exec_params)
                if isinstance(params_or_error, dict) and not params_or_error.get("ok", True):
                    return params_or_error
                exec_params = params_or_error

            # 3. Instance Management
            if skill_name not in instances:
                skill_class = meta.skill_class
                if skill_class is None:
                    raise RuntimeError("validated skill has no executable implementation class")
                instances[skill_name] = self._construct_skill_instance(
                    meta, skill_class
                )
            skill_instance = instances[skill_name]

            # 4. Critical Execution loop
            self._emit_skill_status(
                skill_name,
                "RUNNING",
                expected_catalog_digest=catalog_digest,
            )

            # 2026 Transcendence: Memory Budget Enforcement
            from core.runtime import CoreRuntime

            try:
                try:
                    rt = CoreRuntime.get_sync()
                except RuntimeError as exc:
                    if "CoreRuntime not initialized" not in str(exc):
                        raise
                    rt = await CoreRuntime.get()
                gov = rt.container.get("memory_governor", default=None)
                if gov:
                    check = getattr(gov, "check", None)
                    if callable(check):
                        check_result = check()
                        if inspect.isawaitable(check_result):
                            await check_result
                    elif hasattr(gov, "_enforce_policy"):
                        await gov._enforce_policy()
                orm = rt.container.get("persistent_state", default=None)
            except (RuntimeError, OSError, ConnectionError, TimeoutError, ContainerError) as exc:
                _record_capability_degradation(
                    exc,
                    action="continued skill execution without core runtime memory governance",
                    severity="degraded" if meta.metabolic_cost >= 3 else "warning",
                )
                self.logger.debug("Core runtime memory governance unavailable: %s", exc)
                rt = None
                orm = None

            # --- Central Resilience Primitives: The Cognitive Governor ---
            # Instantiate on the engine if it doesn't exist yet
            if not hasattr(self, "_cognitive_governor"):
                from core.resilience.cognitive_governor import CognitiveGovernor

                self._cognitive_governor = CognitiveGovernor(
                    max_concurrent_tasks=5, base_backoff=1.0
                )
            timeout_budget = max(
                float(getattr(meta, "timeout_seconds", 30) or 30),
                float(getattr(skill_instance, "timeout_seconds", 30) or 30),
            )
            # A SKILL MAY COST DIFFERENT AMOUNTS FOR DIFFERENT REQUESTS.
            #
            # A flat per-skill number cannot describe "make a folder" and
            # "read three articles and write a synthesis" at once. desktop_task
            # declared 180s; the same objective measured 98s, 100s, 156s, 161s
            # and 176s across one evening, so the budget sat inside its own
            # spread and the outcome was a coin flip. Live 2026-07-29 it lost:
            # 93.5s of successful research was cancelled and reported as
            # "Completed 0/0 steps".
            #
            # Any skill that can say what a particular request will cost is
            # asked. Nothing is required to implement it, and a skill that
            # declines keeps its declared number.
            budget_for = getattr(skill_instance, "timeout_for", None)
            if callable(budget_for):
                try:
                    requested_budget = float(budget_for(exec_params) or 0.0)
                except (AttributeError, TypeError, ValueError) as exc:
                    self.logger.debug(
                        "%s could not size its own budget: %s", skill_name, exc
                    )
                else:
                    if requested_budget > timeout_budget:
                        self.logger.info(
                            "⏱️ %s sized this request at %.0fs (declared %.0fs).",
                            skill_name,
                            requested_budget,
                            timeout_budget,
                        )
                        timeout_budget = requested_budget
            background_preflight_deferred = False
            if skill_name == "sovereign_network" and exec_source not in {
                "user",
                "api",
                "chat",
                "desktop",
                "voice",
                "web",
            }:
                try:
                    mode = str(exec_params.get("mode", "status") or "status").strip().lower()
                    if mode in {"recon", "scan", "audit", "discovery"}:
                        from core.runtime.background_policy import (
                            RECON_SCAN_BACKGROUND_POLICY,
                            background_activity_reason,
                        )

                        reason = background_activity_reason(
                            ctx.get("orchestrator"),
                            profile=RECON_SCAN_BACKGROUND_POLICY,
                        )
                        if reason:
                            background_preflight_deferred = True
                            result = {
                                "ok": False,
                                "status": "deferred",
                                "reason": reason,
                                "message": (
                                    f"Network {mode} deferred while foreground conversation is protected ({reason})."
                                ),
                            }
                except (ImportError, AttributeError, RuntimeError) as policy_exc:
                    _record_capability_degradation(
                        policy_exc,
                        action="deferred background network task because preflight policy failed",
                        severity="degraded",
                    )
                    self.logger.warning(
                        "Background network preflight failed for %s: %s",
                        skill_name,
                        policy_exc,
                    )
                    return {
                        "ok": False,
                        "status": "deferred",
                        "reason": "background_policy_unavailable",
                        "message": (
                            f"Network {mode} deferred because the background protection policy is unavailable."
                        ),
                    }
            if (
                not background_preflight_deferred
                and skill_name in _HEAVY_BACKGROUND_SKILLS
                and exec_source not in {"user", "api", "chat", "desktop", "voice", "web"}
            ):
                try:
                    from core.runtime.background_policy import background_activity_reason

                    reason = background_activity_reason(
                        ctx.get("orchestrator"),
                        min_idle_seconds=600.0,
                        max_memory_percent=70.0,
                        max_failure_pressure=0.20,
                        require_conversation_ready=False,
                    )
                    if reason:
                        background_preflight_deferred = True
                        result = {
                            "ok": False,
                            "status": "deferred",
                            "reason": reason,
                            "message": (
                                f"Background {skill_name} deferred while live conversation resources are protected ({reason})."
                            ),
                        }
                except (ImportError, AttributeError, RuntimeError) as policy_exc:
                    _record_capability_degradation(
                        policy_exc,
                        action="deferred heavy background skill because preflight policy failed",
                        severity="degraded",
                    )
                    self.logger.warning(
                        "Heavy background preflight failed for %s: %s",
                        skill_name,
                        policy_exc,
                    )
                    return {
                        "ok": False,
                        "status": "deferred",
                        "reason": "background_policy_unavailable",
                        "message": (
                            f"Background {skill_name} deferred because the resource protection policy is unavailable."
                        ),
                    }
            raw_constrained_timeout = ctx.get("timeout_s")
            try:
                constrained_timeout = (
                    float(raw_constrained_timeout)
                    if raw_constrained_timeout is not None
                    else 0.0
                )
            except (TypeError, ValueError):
                constrained_timeout = 0.0
            if constrained_timeout and constrained_timeout > 0.0:
                timeout_budget = max(1.0, min(timeout_budget, constrained_timeout))

            try:
                # [DEV MODE INTEGRATION] Log tool execution dispatch for transparency
                try:
                    from core.transparency import get_dev_mode
                    dev_mode = get_dev_mode()
                    exec_origin = ctx.get("origin", "unknown")
                    tool_trace = await dev_mode.record_tool_execution(
                        skill_name, exec_params, origin=exec_origin
                    )
                except (ImportError, AttributeError, RuntimeError):
                    tool_trace = None
                
                # Execute safely via the Governor to prevent cascading API failures
                async def resilient_call() -> dict[str, Any]:
                    return await self._execute_with_retry(
                        skill_instance,
                        skill_name,
                        exec_params,
                        ctx,
                        execution_timeout=timeout_budget,
                    )

                exec_start = time.monotonic()
                if background_preflight_deferred:
                    pass
                elif tool_handle is not None:
                    from core.governance_context import governed_scope

                    # The lease lasts as long as the budget we just granted.
                    # A 30s default against a 101s sanctioned step meant the
                    # skill lost its governance mid-write and failed with
                    # "called outside governed context" — the authorization
                    # expiring underneath work it had authorized.
                    async with governed_scope(
                        tool_handle.decision, ttl=timeout_budget
                    ):
                        result = await self._cognitive_governor.execute_safely(
                            task_name=skill_name,
                            coroutine=resilient_call,
                            timeout_seconds=timeout_budget,
                        )
                elif (
                    self._context_user_authorized(ctx, exec_source)
                    and bool(
                        ctx.get("foreground_request")
                        or ctx.get("user_visible_desktop_action")
                        or ctx.get("user_visible_browser_action")
                    )
                ):
                    from core.governance_context import local_internal_governed_scope

                    with local_internal_governed_scope(
                        f"capability_engine.{skill_name}.foreground_user_request",
                        domain="tool_execution",
                        ttl=timeout_budget,
                    ):
                        result = await self._cognitive_governor.execute_safely(
                            task_name=skill_name,
                            coroutine=resilient_call,
                            timeout_seconds=timeout_budget,
                        )
                else:
                    result = await self._cognitive_governor.execute_safely(
                        task_name=skill_name,
                        coroutine=resilient_call,
                        timeout_seconds=timeout_budget,
                    )
                
                # [DEV MODE INTEGRATION] Log tool execution result
                if tool_trace:
                    try:
                        exec_time_ms = (time.monotonic() - exec_start) * 1000.0
                        await dev_mode.complete_tool_execution(tool_trace, result, exec_time_ms)
                    except (ImportError, AttributeError, RuntimeError) as _exc:
                        logger.debug("Suppressed %s in core.capability_engine: %s", type(_exc).__name__, _exc)

            except (ImportError, AttributeError, RuntimeError) as e:
                _record_capability_degradation(
                    e,
                    action="returned skill execution failure after governor invocation failed",
                    severity="degraded",
                )
                self.logger.error("❌ Skill '%s' unwrapped failure: %s", skill_name, e)
                result = {"ok": False, "error": str(e), "_exception": True}

            duration_ms = (time.monotonic() - start_time) * 1000

            # Update state based on result
            if result is None:
                result = {"ok": False, "error": "Unknown execution failure (result is None)"}
            elif not isinstance(result, dict):
                result = {
                    "ok": False,
                    "error": "Skill execution returned a non-object result",
                    "raw_result_type": type(result).__name__,
                }

            # A graceful {ok: false} return means the skill itself is healthy —
            # only mark ERROR if the skill threw an unhandled exception (caught above).
            # This prevents "nmap not installed" from permanently bricking sovereign_network.
            was_exception = result.pop("_exception", False) if isinstance(result, dict) else False
            final_state = "ERROR" if was_exception else "READY"
            self._emit_skill_status(
                skill_name,
                final_state,
                expected_catalog_digest=catalog_digest,
            )
            if not result.get("ok", True):
                # Store error for diagnostics, but ONLY if the skill is in ERROR state.
                # Graceful {ok: false} (e.g. "nmap not installed") should NOT persist
                # as degraded_reason — the skill is still healthy, just this call failed.
                if was_exception:
                    skill_last_errors[skill_name] = str(
                        result.get("error") or "execution_failed"
                    )
                # else: transient failure, don't pollute the catalog
            else:
                skill_last_errors.pop(skill_name, None)

            # 5. Persistent Audit (ORM)
            if orm:
                try:
                    # CP126: "Only top-level parameter keys containing a
                    # small secret-word set are redacted. Nested
                    # credentials, headers, URLs, content, files, and full
                    # successful results can be persisted without
                    # structural redaction or size limits."
                    #
                    # Both halves were true. params.copy() is shallow, so
                    # {"request": {"headers": {"Authorization": "Bearer …"}}}
                    # reached the database intact, and `result` was written
                    # whole — unredacted and unbounded — for every
                    # successful call, which is how a 40MB tool output ends
                    # up as an audit row.
                    audit_params = canonical_authority_arguments(skill_name, params)
                    safe_params, params_report = redact_mapping(audit_params)
                    safe_result: Any = None
                    result_report = None
                    if result.get("ok"):
                        safe_result, result_report = redact_structure(result)
                    safe_error = None
                    if not result.get("ok"):
                        safe_error, _ = redact_text(str(result.get("error") or ""))

                    # Say when a row is partial. A truncated audit record
                    # that does not admit it is worse than a missing one.
                    for label, report in (
                        ("params", params_report),
                        ("result", result_report),
                    ):
                        marker = redaction_marker(report) if report else None
                        if marker:
                            self.logger.debug(
                                "audit row for '%s' %s redacted/bounded: %s",
                                skill_name,
                                label,
                                marker,
                            )

                    orm.log_execution(
                        skill_name=skill_name,
                        params=safe_params,
                        status=final_state,
                        duration_ms=duration_ms,
                        result=safe_result,
                        error=safe_error,
                    )
                except (OSError, ConnectionError, TimeoutError) as e:
                    _record_capability_degradation(
                        e,
                        action="returned skill result after persistent audit logging failed",
                    )
                    self.logger.warning("ORM logging failed: %s", e)

            # 5. Mycelium Reinforcement (Sentient Feedback Loop)
            try:
                if hasattr(self.orchestrator, "mycelium") and self.orchestrator.mycelium:
                    # Issue 53 Fix: Only catch expected reinforcement errors
                    self.orchestrator.mycelium.reinforce(
                        f"skill_{skill_name}", success=result.get("ok", False)
                    )
            except AttributeError as e:
                self.logger.debug("Reinforcement attribute missing: %s", e)
            except (OSError, ConnectionError, TimeoutError) as e:
                _record_capability_degradation(
                    e,
                    action="returned skill result after mycelium reinforcement failed",
                )
                self.logger.warning("Reinforcement failed: %s", e)

            # 6. Outcome Recording (Asynchronous)
            if self.temporal:
                t = get_task_tracker().create_task(
                    self._record_temporal(skill_name, params, ctx, result)
                )
                t.add_done_callback(
                    lambda t: t.exception() if not t.cancelled() and t.exception() else None
                )

            return result

        try:
            # Two copies of the same request usually arrive concurrently —
            # that is what a client retrying a call it thinks has stalled
            # looks like. The ledger single-flights on the key, so the
            # second caller waits for the first outcome instead of starting
            # a second execution. A replay never re-enters the body, so the
            # closure receipt below stays untouched: the original run
            # already closed it.
            idempotency_key = str((context or {}).get("idempotency_key") or "").strip()
            if idempotency_key:
                outcome = await get_idempotency_ledger().run_once(
                    f"{skill_name}:{idempotency_key}",
                    _execute_wrapped,
                )
                wrapped_result = outcome.value
                if outcome.replayed and isinstance(wrapped_result, dict):
                    # Say so rather than pretending this was a fresh run.
                    # A caller that cannot tell a replay from an execution
                    # will double-count it somewhere else.
                    wrapped_result = {
                        **wrapped_result,
                        "idempotent_replay": True,
                        "idempotency_key": idempotency_key,
                    }
            else:
                wrapped_result = await _execute_wrapped()
            if not isinstance(wrapped_result, dict):
                result = {
                    "ok": False,
                    "error": "Capability error boundary returned a non-object result",
                }
                return result
            result = wrapped_result
            return result
        finally:
            try:
                if (
                    constitution is not None
                    and tool_handle is not None
                    and bool(getattr(tool_handle, "approved", False))
                ):
                    closure_receipt = await constitution.finish_tool_execution(
                        tool_handle,
                        result=result or {"ok": False, "error": "execution_not_completed"},
                        success=bool(isinstance(result, dict) and result.get("ok", False)),
                        duration_ms=0.0,
                        error=""
                        if bool(isinstance(result, dict) and result.get("ok", False))
                        else str((result or {}).get("error", "")),
                    )
                    # CP126 (critical): "finish_tool_execution runs for every
                    # approved tool, but a false or exceptional closure receipt
                    # downgrades only os_automation. File writes, shell
                    # actions, email, browser work, and other external effects
                    # can return ok after their tool authority failed to
                    # close."
                    #
                    # The os_automation branch had the right shape and the
                    # wrong scope: one skill name, hardcoded. Every effectful
                    # skill now gets the same treatment, keyed on the declared
                    # effect scope rather than a name — and `unknown` counts as
                    # effectful, because an unclassified skill is not a safe
                    # one. Non-effectful skills still carry the receipt so the
                    # failure is visible, but their result is not invalidated:
                    # a read-only call did not leave anything to reconcile.
                    if isinstance(result, dict):
                        # A MISSING receipt and a receipt that REPORTS FAILURE
                        # are different facts. os_automation has always
                        # treated "no receipt" as a failed closure and that
                        # contract is preserved; for every other skill, a
                        # constitution that returned nothing has told us
                        # nothing, so it is recorded rather than used to
                        # invalidate the caller's result.
                        if not isinstance(closure_receipt, dict):
                            closure_receipt = {
                                "closed": skill_name != "os_automation",
                                "mode": "constitutional_closure",
                                "receipt_present": False,
                                "errors": ["constitutional core returned no closure receipt"],
                            }
                        result["authority_closure"] = closure_receipt
                        if not bool(closure_receipt.get("closed")):
                            closure_scope = _declared_effect_scope(skill_name)
                            _record_unreconciled_authority(
                                tool_handle,
                                reason=f"closure_receipt_not_closed:{skill_name}",
                            )
                            if closure_scope in _PRE_RUNTIME_UNGATED_EFFECT_SCOPES:
                                result["authority_closure_effect_scope"] = closure_scope
                            else:
                                original_status = str(result.get("status") or "")
                                attempt_rows = result.get("attempts")
                                if not isinstance(attempt_rows, list):
                                    attempt_rows = []
                                action_may_have_occurred = bool(
                                    result.get("effect_verified")
                                ) or any(
                                    bool(attempt.get("transport_success"))
                                    for attempt in attempt_rows
                                    if isinstance(attempt, dict)
                                )
                                result["ok"] = False
                                result["status"] = "authority_closure_failed"
                                result["authority_closure_original_status"] = original_status
                                result["authority_closure_effect_scope"] = closure_scope
                                result["manual_reconciliation_required"] = (
                                    action_may_have_occurred
                                )
                                result["error"] = (
                                    f"{skill_name} authority did not close cleanly after "
                                    "execution. Do not retry automatically until "
                                    "capability-token state is reconciled."
                                )
            except (
                OSError,
                ConnectionError,
                TimeoutError,
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
            ) as _exc:
                _record_capability_degradation(
                    _exc,
                    action="returned skill result after constitutional finish receipt failed",
                    severity="degraded",
                )
                self.logger.debug("Suppressed Exception: %s", _exc)
                # Same generalisation on the raise path: a closure that threw
                # closed nothing, whichever skill it was for.
                if isinstance(result, dict):
                    # A raise is a real closure failure — unlike a missing
                    # receipt, it is positive evidence that closure did not
                    # complete — so this path does invalidate effectful
                    # results.
                    closure_scope = _declared_effect_scope(skill_name)
                    result["authority_closure"] = {
                        "closed": False,
                        "mode": "constitutional_closure",
                        "receipt_present": False,
                        "errors": [f"{type(_exc).__name__}:{_exc}"],
                    }
                    result["authority_closure_effect_scope"] = closure_scope
                    _record_unreconciled_authority(
                        tool_handle,
                        reason=f"closure_receipt_raised:{type(_exc).__name__}",
                    )
                    if closure_scope not in _PRE_RUNTIME_UNGATED_EFFECT_SCOPES:
                        result["ok"] = False
                        result["status"] = "authority_closure_failed"
                        result["manual_reconciliation_required"] = bool(
                            result.get("effect_verified")
                        )
                        result["error"] = (
                            f"{skill_name} authority closure raised an error; "
                            "capability-token state requires reconciliation."
                        )

    def _apply_security(
        self, skill_name: str, params: dict[str, Any]
    ) -> dict[str, Any] | dict[str, str]:
        """Issue 54: Scoped security adaptation for skill parameters."""
        if not self.rosetta_stone:
            return params

        # Issue 54: Only check keys in COMMAND_PARAM_KEYS to avoid security false positives
        command_param_keys = {"command", "cmd", "path", "url", "target", "script"}

        def scan_recursive(val: Any, key: str | None = None) -> tuple[bool, Any, str | None]:
            if isinstance(val, str):
                # Issue 54: Limit security scanning to relevant parameter names
                if key and key.lower() not in command_param_keys:
                    return True, val, None

                # Check for common shell injection patterns
                if any(x in val for x in [";", "&&", "||", "`", "$(", "|", ">", "<"]):
                    threats = self.rosetta_stone.analyze_threat(val)
                    if not threats["safe"]:
                        return False, val, f"Security Block (Threat Detected): {threats['threats']}"

                return True, self.rosetta_stone.adapt_command(val), None
            elif isinstance(val, dict):
                new_dict = {}
                for k, v in val.items():
                    ok, new_v, err = scan_recursive(v, k)
                    if not ok:
                        return False, None, err
                    new_dict[k] = new_v
                return True, new_dict, None
            elif isinstance(val, list):
                new_list = []
                for item in val:
                    ok, new_item, err = scan_recursive(item, key)
                    if not ok:
                        return False, None, err
                    new_list.append(new_item)
                return True, new_list, None
            return True, val, None

        ok, filtered_params, error_msg = scan_recursive(params)
        if not ok:
            self.logger.warning(
                "❌ Security violation blocked in skill '%s': %s", skill_name, error_msg
            )
            return {"ok": False, "error": error_msg, "status": "blocked"}

        return filtered_params

    async def _execute_with_retry(
        self,
        skill: Any,
        skill_name: str,
        params: dict[str, Any],
        context: dict[str, Any],
        *,
        execution_timeout: float | None = None,
    ) -> dict[str, Any]:
        """Executes a skill method with a retry loop for transient failures."""
        last_error = "Unknown"
        attempt = 0
        output: Any = None
        try:
            timeout_s = float(execution_timeout or self.timeout)
        except (TypeError, ValueError):
            timeout_s = float(self.timeout)
        timeout_s = max(1.0, timeout_s)
        # TELL THE SKILL THE BUDGET, THEN LET IT BE THE ONE TO CALL TIME.
        #
        # BaseSkill.safe_execute enforces its own declared timeout inside this
        # wait. Sizing the request here and not passing it down left the skill
        # cutting the work off at its flat default while this wait sat harmless
        # at the larger number — the negotiated budget was logged and then had
        # no effect. Publish it, and keep this wait strictly longer so the
        # skill's own timeout is what fires: its error path carries the step
        # receipts, and wait_for cancelling from outside destroys them.
        context[_SKILL_TIMEOUT_CONTEXT_KEY] = timeout_s
        outer_timeout_s = timeout_s + _OUTER_TIMEOUT_GRACE_S
        # Sensorimotor grounding: commit an expected world-state BEFORE
        # executing; verify reality (not the tool's claim) after. A tool
        # reporting success without the predicted effect is recorded as
        # ACTION-CLAIM-MISMATCH (the confabulated-action class).
        try:
            from core.grounding.sensorimotor_loop import ground_result, open_grounding
            _sm_predicate, _sm_receipt = open_grounding(skill_name, params)
        except (ImportError, AttributeError, RuntimeError):
            _sm_predicate, _sm_receipt = None, None
            ground_result = None
        max_attempts = 1 if self._outer_retry_disabled(skill_name, params, context) else self.max_retries
        for attempt in range(max(1, max_attempts)):
            try:
                if attempt > 0:
                    await asyncio.sleep(self.retry_delay * attempt)
                    self.logger.info("Retrying %s (attempt %s)...", skill_name, attempt + 1)

                if hasattr(skill, "safe_execute") and callable(skill.safe_execute):
                    output = await asyncio.wait_for(
                        skill.safe_execute(params, context), timeout=outer_timeout_s
                    )
                else:
                    inputs = self._prepare_inputs(skill, params, context)
                    output = await self._call_method(skill, inputs, timeout_s=timeout_s)

                if self._check_success(output):
                    if _sm_predicate is not None and ground_result is not None:
                        try:
                            ground_result(
                                skill_name, params, output, _sm_predicate, _sm_receipt,
                            )
                        except (OSError, RuntimeError, TypeError, ValueError) as _sm_exc:
                            self.logger.debug("Grounding check skipped: %s", _sm_exc)
                    if isinstance(output, dict):
                        payload = dict(output)
                        payload["ok"] = bool(
                            payload.get(
                                "ok",
                                payload.get("error") is None
                                and not payload.get("errors")
                                and not payload.get("failed", False),
                            )
                        )
                        payload["retries"] = attempt
                        return await self._apply_action_expectation_result(
                            skill_name,
                            payload,
                            params,
                            context,
                        )
                    return await self._apply_action_expectation_result(
                        skill_name,
                        {"ok": True, "result": output, "retries": attempt},
                        params,
                        context,
                    )

                last_error = self._extract_error(output)
                if not _is_transient(last_error):
                    break
            except (OSError, ConnectionError, TimeoutError) as e:
                _record_capability_degradation(
                    e,
                    action="retried transient skill execution failure or returned retry exhaustion",
                )
                last_error = str(e).strip()
                if not last_error and isinstance(e, TimeoutError):
                    last_error = f"{skill_name} timed out after {timeout_s:.1f}s"
                if not _is_transient(last_error):
                    break

        failure: dict[str, Any] = {"ok": False, "error": last_error, "retries": attempt}
        if isinstance(output, dict):
            # Preserve the skill's structured failure evidence (step
            # receipts, failure details, step counts) — flattening it
            # erased the receipts reply surfaces need and reduced every
            # structured failure to 'unknown' / 'Completed 0/0 steps'.
            failure = {**output, **failure}
        return failure

    @staticmethod
    def _str_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if str(item).strip()]
        return [str(value)]

    @staticmethod
    def _bool_value(value: Any, *, default: bool = True) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"false", "0", "no", "off"}:
                return False
            if lowered in {"true", "1", "yes", "on"}:
                return True
        return bool(value)

    @classmethod
    def _action_expectation_for(
        cls,
        skill_name: str,
        params: dict[str, Any],
        context: dict[str, Any],
    ) -> Any | None:
        from core.runtime.skill_contract import ActionExpectation

        params = params or {}
        context = context or {}
        raw = (
            context.get("action_expectation")
            or context.get("expectation")
            or params.get("action_expectation")
            or params.get("expectation")
        )
        if isinstance(raw, ActionExpectation):
            return raw

        source: dict[str, Any] = {}
        if isinstance(raw, dict):
            source.update(raw)
        for key in (
            "acceptance_criteria",
            "required_evidence",
            "required_evidence_present",
            "semantic_predicates",
            "user_visible_effect",
            "repair_hint",
            "rollback_hint",
            "allow_partial",
        ):
            if key in context and key not in source:
                source[key] = context[key]
            if key in params and key not in source:
                source[key] = params[key]

        criteria = cls._str_list(source.get("acceptance_criteria") or source.get("criteria"))
        evidence = cls._str_list(source.get("required_evidence") or source.get("evidence_required"))
        evidence_present = cls._str_list(source.get("required_evidence_present"))
        raw_predicates = source.get("semantic_predicates") or []
        if not isinstance(raw_predicates, (list, tuple)):
            raw_predicates = []
        from core.runtime.skill_contract import semantic_predicate_from_mapping

        predicates = []
        for item in raw_predicates[:64]:
            if hasattr(item, "predicate_id") and hasattr(item, "evidence_path"):
                predicates.append(item)
            elif isinstance(item, dict):
                predicates.append(semantic_predicate_from_mapping(item))
        visible_effect = source.get("user_visible_effect") or source.get("visible_effect")
        if not criteria and not evidence and not evidence_present and not predicates and not visible_effect:
            default_expectation = cls._default_action_expectation_for(
                skill_name,
                params,
                context,
                ActionExpectation,
            )
            if default_expectation is not None:
                return default_expectation
            return None

        return ActionExpectation(
            objective=str(
                source.get("objective")
                or context.get("objective")
                or context.get("message")
                or params.get("objective")
                or params.get("query")
                or params.get("path")
                or skill_name
            ),
            acceptance_criteria=criteria,
            required_evidence=evidence,
            required_evidence_present=evidence_present,
            semantic_predicates=predicates,
            user_visible_effect=str(visible_effect) if visible_effect else None,
            repair_hint=str(source.get("repair_hint") or ""),
            rollback_hint=str(source.get("rollback_hint") or ""),
            allow_partial=cls._bool_value(source.get("allow_partial"), default=True),
        )

    @classmethod
    def action_expectation_for(
        cls,
        skill_name: str,
        params: dict[str, Any],
        context: dict[str, Any],
    ) -> Any | None:
        """Public runtime boundary for expectation derivation."""
        return cls._action_expectation_for(skill_name, params, context)

    @classmethod
    def _default_action_expectation_for(
        cls,
        skill_name: str,
        params: dict[str, Any],
        context: dict[str, Any],
        expectation_cls: Any,
    ) -> Any | None:
        if cls._bool_value((context or {}).get("disable_auto_action_expectation"), default=False):
            return None
        if cls._bool_value((params or {}).get("disable_auto_action_expectation"), default=False):
            return None

        normalized_skill = str(skill_name or "").strip().lower()
        if normalized_skill == "memory_ops":
            action = str((params or {}).get("action") or "").strip().lower()
            action = {
                "remember": "archival_insert",
                "memorize": "archival_insert",
                "store": "archival_insert",
                "save": "archival_insert",
            }.get(action, action)
            memory_expectations = {
                "core_append": ("core memory appended", "append"),
                "core_replace": ("core memory replaced", "replace"),
            }
            if action in memory_expectations:
                criterion, verb = memory_expectations[action]
                block = str((params or {}).get("block") or "user").strip()
                return expectation_cls(
                    objective=f"{verb} core memory block {block or 'user'}",
                    acceptance_criteria=[criterion],
                    required_evidence=["block", "sha256", "effect_verified"],
                    user_visible_effect=f"core memory {verb} is persisted and verified",
                    repair_hint=f"verify_memory_ops_{action}_effect",
                    rollback_hint="restore_previous_core_memory_block",
                    allow_partial=False,
                )
            if action == "archival_insert":
                return expectation_cls(
                    objective="persist requested content in archival memory",
                    acceptance_criteria=["archival memory stored"],
                    required_evidence=[
                        "record_id",
                        "memory_receipt_id",
                        "bytes_written",
                        "content_sha256",
                        "effect_verified",
                    ],
                    user_visible_effect="archival memory write is durable and receipt-backed",
                    repair_hint="retry_archival_insert_through_memory_write_gateway",
                    rollback_hint="tombstone_or_restore_archival_memory_record",
                    allow_partial=False,
                )
            return None

        if normalized_skill in {"web_search", "free_search", "grounded_search"}:
            if not cls._web_query_requires_sources(params, context):
                return None
            query = str(
                (params or {}).get("query")
                or (params or {}).get("q")
                or (context or {}).get("objective")
                or (context or {}).get("message")
                or normalized_skill
            ).strip()
            return expectation_cls(
                objective=f"source-backed web research for {query[:160] or normalized_skill}",
                acceptance_criteria=[],
                required_evidence=["sources", "summary"],
                repair_hint="rerun_web_research_with_sources",
                rollback_hint="not_required_read_only",
                allow_partial=True,
            )

        if normalized_skill != "file_operation":
            return None

        action = str((params or {}).get("action") or "").strip().lower()
        path = str((params or {}).get("path") or "").strip()
        destination = str((params or {}).get("destination") or "").strip()
        file_expectations = {
            "write": ("file written", ["path", "sha256", "effect_verified"]),
            "append": ("file appended", ["path", "sha256", "effect_verified"]),
            "patch": ("file patched", ["path", "sha256", "effect_verified"]),
            "delete": ("path deleted", ["path", "effect_verified"]),
            "move": ("path moved", ["path", "destination", "sha256", "effect_verified"]),
            "copy": ("path copied", ["path", "destination", "sha256", "effect_verified"]),
        }
        if action not in file_expectations:
            return None

        criterion, evidence = file_expectations[action]
        target = f"{path} -> {destination}" if destination else path
        rollback_hints = {
            "write": "restore_previous_file_version_or_delete_new_file",
            "append": "restore_previous_file_version",
            "patch": "restore_previous_file_version",
            "delete": "restore_deleted_path_from_backup",
            "move": "move_destination_back_to_source",
            "copy": "delete_verified_destination_copy",
        }
        return expectation_cls(
            objective=f"{action} file_operation effect for {target or 'requested path'}",
            acceptance_criteria=[criterion],
            required_evidence=evidence,
            user_visible_effect=f"filesystem {action} is observable and verified",
            repair_hint=f"verify_file_operation_{action}_effect",
            rollback_hint=rollback_hints[action],
            allow_partial=False,
        )

    @classmethod
    def _web_query_requires_sources(
        cls,
        params: dict[str, Any],
        context: dict[str, Any],
    ) -> bool:
        if cls._bool_value((params or {}).get("deep"), default=False):
            return True
        if cls._bool_value((context or {}).get("requires_sources"), default=False):
            return True
        if cls._bool_value((params or {}).get("requires_sources"), default=False):
            return True
        query = str(
            (params or {}).get("query")
            or (params or {}).get("q")
            or (context or {}).get("objective")
            or (context or {}).get("message")
            or ""
        ).casefold()
        source_markers = (
            "article",
            "citation",
            "cite",
            "current",
            "latest",
            "news",
            "research",
            "source",
            "today",
        )
        return any(marker in query for marker in source_markers)

    @classmethod
    async def _apply_action_expectation_result(
        cls,
        skill_name: str,
        result: dict[str, Any],
        params: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(result, dict) or not result.get("ok", True):
            return result

        expectation = cls._action_expectation_for(skill_name, params, context)
        if expectation is None:
            return result

        from core.runtime.skill_contract import apply_action_expectation_payload

        raw_payload = apply_action_expectation_payload(
            skill_name,
            result,
            expectation,
        )
        if not isinstance(raw_payload, dict):
            raise TypeError("action-expectation verifier returned a non-object payload")
        payload: dict[str, Any] = raw_payload
        expectation_receipt_id = await cls._emit_action_expectation_receipt(
            skill_name,
            payload,
            expectation,
        )
        if expectation_receipt_id:
            payload["expectation_receipt_id"] = expectation_receipt_id
            payload["verification_evidence"]["expectation_receipt_id"] = expectation_receipt_id
            if isinstance(payload.get("expectation_verdict"), dict):
                payload["expectation_verdict"]["receipt_id"] = expectation_receipt_id
        return payload

    @staticmethod
    def _action_expectation_digest(payload: dict[str, Any]) -> str:
        try:
            encoded = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        except (TypeError, ValueError):
            encoded = str(payload).encode("utf-8", errors="replace")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    async def _emit_action_expectation_receipt(
        cls,
        skill_name: str,
        result: dict[str, Any],
        expectation: Any,
    ) -> str | None:
        evidence = result.get("verification_evidence")
        evidence = dict(evidence) if isinstance(evidence, dict) else {}
        verdict = evidence.get("expectation_verdict", {})
        if not isinstance(verdict, dict) or not verdict:
            return None

        try:
            from core.runtime.receipts import ToolExecutionReceipt, get_receipt_store

            receipt = ToolExecutionReceipt(
                cause=str(getattr(expectation, "objective", "") or skill_name)[:240],
                tool=skill_name,
                status=str(result.get("status") or "success_unverified"),
                output_digest=cls._action_expectation_digest(
                    {
                        "skill": skill_name,
                        "status": result.get("status"),
                        "ok": bool(result.get("ok", False)),
                        "verdict": verdict,
                    }
                ),
                verification_evidence={
                    "expectation_verdict": verdict,
                    "action_expectation": evidence.get("action_expectation", {}),
                    "original_receipt_id": result.get("receipt_id"),
                    "failure_reason": result.get("error"),
                },
                metadata={
                    "source": "capability_engine.action_expectation",
                    "expectation_objective": str(
                        getattr(expectation, "objective", "") or skill_name
                    )[:240],
                    "expectation_next_step": str(verdict.get("next_step") or "")[:240],
                    "expectation_rollback_hint": str(
                        getattr(expectation, "rollback_hint", "") or ""
                    )[:240],
                    "passed": bool(verdict.get("passed", False)),
                },
            )
            from core.runtime.executors import run_durable_receipt_io

            def persist_receipt() -> Any:
                return get_receipt_store().emit(receipt)

            emitted = await run_durable_receipt_io(
                persist_receipt,
                timeout_s=10.0,
                label="capability_action_expectation_receipt",
            )
            if not bool(verdict.get("passed", False)):
                try:
                    from core.resilience.fault_taxonomy import get_fault_registry

                    missing = list(verdict.get("missing_criteria") or []) + list(
                        verdict.get("missing_evidence") or []
                    )
                    get_fault_registry().record_fault(
                        "PASSF-ACTION-SHALLOW-SUCCESS",
                        "capability_engine",
                        details=(
                            f"{skill_name} expectation downgraded before verified success; "
                            f"missing={missing[:6]}"
                        ),
                        recovered=True,
                        recovery_time_s=0.0,
                    )
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as fault_exc:
                    _record_capability_degradation(
                        fault_exc,
                        action="returned expectation-downgraded result after fault occurrence recording failed",
                        severity="warning",
                    )
            receipt_id = getattr(emitted, "receipt_id", None)
            if not isinstance(receipt_id, str) or not receipt_id:
                raise ValueError("durable expectation receipt is missing its receipt ID")
            return receipt_id
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            _record_capability_degradation(
                exc,
                action="returned expectation verdict after durable receipt emit failed",
                severity="warning",
            )
            return None

    @staticmethod
    def _outer_retry_disabled(
        skill_name: str,
        params: dict[str, Any],
        context: dict[str, Any],
    ) -> bool:
        """Avoid replaying aggregate consequential plans from step one.

        `desktop_task` owns per-step retry policy because only it knows which
        primitives are idempotent. Retrying the whole skill after a late timeout
        can duplicate clicks, typing, file writes, or system changes.
        """
        if bool((context or {}).get("disable_outer_skill_retry")):
            return True
        if bool((params or {}).get("disable_outer_skill_retry")):
            return True
        return skill_name == "desktop_task"

    async def _call_method(
        self,
        skill: Any,
        inputs: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> Any:
        """Calls the skill method, handling both sync and async."""
        # If the skill is not core and we have source code (forged), we should sandbox it.
        # For simplicity, we assume skills loaded from skilled_dir aren't core.

        method = skill.execute if hasattr(skill, "execute") else skill
        if inspect.iscoroutinefunction(method):
            return await asyncio.wait_for(
                method(**inputs),
                timeout=max(1.0, float(timeout_s or self.timeout)),
            )

        # If RestrictedPython is available and NOT core, we could potentially wrap it,
        # but for now we focus on FORGED skills which provide source.
        return await asyncio.get_running_loop().run_in_executor(None, lambda: method(**inputs))

    def _prepare_inputs(
        self, skill: Any, params: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Maps parameters to the skill's expected signature."""
        method = skill.execute if hasattr(skill, "execute") else skill
        sig = inspect.signature(method)
        if "goal" in sig.parameters:
            goal_payload: dict[str, Any]
            if isinstance(params, dict):
                goal_payload = dict(params)
                nested_params = (
                    dict(goal_payload.get("params") or {})
                    if isinstance(goal_payload.get("params"), dict)
                    else {}
                )
                for key, value in goal_payload.items():
                    if key != "params":
                        nested_params.setdefault(key, value)
                goal_payload["params"] = nested_params
            else:
                goal_payload = {"params": {"value": params}}

            objective = (
                goal_payload.get("objective")
                or context.get("objective")
                or context.get("message")
                or goal_payload.get("query")
                or goal_payload.get("content")
                or goal_payload.get("text")
                or goal_payload.get("command")
                or goal_payload.get("path")
            )
            if objective:
                goal_payload["objective"] = str(objective)
            return {"goal": goal_payload, "context": context}
        if "params" in sig.parameters:
            return {"params": params, "context": context}
        if isinstance(params, dict):
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if not has_var_keyword:
                filtered = {}
                for k, v in params.items():
                    if k in sig.parameters:
                        filtered[k] = v
                if "context" in sig.parameters and "context" not in filtered:
                    filtered["context"] = context
                return filtered
        return params

    def _check_success(self, out: Any) -> bool:
        """Determines if the skill output indicates success."""
        if isinstance(out, dict):
            return bool(out.get("ok", True))
        return out is not None

    def _extract_error(self, out: Any) -> str:
        """Extracts an error message from skill output."""
        if isinstance(out, dict):
            direct = out.get("error") or out.get("message")
            if direct:
                return str(direct)
            # Structured step failures (desktop_task et al.) carry the real
            # cause in failures[0] — 'Failed' hid every step error.
            failures = out.get("failures")
            if isinstance(failures, list) and failures:
                first = failures[0]
                if isinstance(first, dict):
                    detail = (
                        first.get("error")
                        or first.get("effect_evidence")
                        or first.get("status")
                    )
                    action = first.get("action")
                    if detail:
                        return (
                            f"step '{action}' failed: {detail}" if action else str(detail)
                        )
            summary = out.get("summary")
            if summary:
                return str(summary)
            return "Failed"
        return "Error"


    async def _record_temporal(
        self, action: str, params: dict[str, Any], context: dict[str, Any], result: dict[str, Any]
    ) -> None:
        """Records the skill outcome to the Temporal Learning system."""
        temporal = self.temporal
        if temporal is None:
            return
        try:
            await temporal.record_outcome(
                action=action,
                context=str(canonical_authority_context(action, context))[:200],
                intended_outcome=str(canonical_authority_arguments(action, params))[:200],
                actual_outcome=str(result)[:500],
                success=bool(result.get("ok", False)),
            )
        except (OSError, ConnectionError, TimeoutError) as e:
            _record_capability_degradation(
                e,
                action="returned skill result after temporal outcome recording failed",
            )
            self.logger.debug("Temporal record failed: %s", e)

    def get_health(self) -> dict[str, Any]:
        """Provides extended health data for the capability system."""
        raw_report = super().get_health()
        report: dict[str, Any] = dict(raw_report) if isinstance(raw_report, dict) else {}
        self._ensure_catalog_loaded()
        with self._catalog_guard():
            skills = dict(self._skills)
        report["skills_total"] = len(skills)
        report["skills_ready"] = sum(
            1
            for metadata in skills.values()
            if metadata.validation_state == "valid" and metadata.dependency_ready
        )
        report["skill_catalog"] = self.get_catalog_health()
        report["skill_catalog_dry_run"] = self.dry_run_catalog()
        return report

    def is_ready(self) -> bool:
        """Deep readiness probe for runtime tool-governance health."""
        self._ensure_catalog_loaded()
        with self._catalog_guard():
            skills = dict(self._skills)
            active_skills = set(self.active_skills)
            catalog_ready = bool(self.catalog_health.get("ready"))
        if not skills:
            return False
        if not active_skills:
            return False
        if not catalog_ready:
            return False
        return any(
            name in active_skills
            and metadata.validation_state == "valid"
            and metadata.dependency_ready
            for name, metadata in skills.items()
        )


def _is_forged_skill(meta: Any) -> bool:
    """Whether this skill is code Aura wrote for herself.

    The predicate used to be ``"skills/" in meta.module_path``. Module paths are
    dotted — ``skills.word_count`` — so the substring never matched and the
    answer was always no. Everything downstream of it was therefore dead: the
    Sandbox 2.0 branch never ran, and model-authored code executed in-process
    like any hand-written skill, under a log line announcing that it was
    confined.

    ``source_kind`` is the catalog's own answer to the same question, set by
    :func:`core.skills.discovery.default_skill_roots` when it walks the writable
    ``skills/`` tree rather than ``core/skills``. It is a fact about where the
    file came from instead of a guess from how a string is spelled.
    """
    if str(getattr(meta, "source_kind", "") or "").strip().lower() == "project":
        return True
    # A skill registered at runtime carries no catalog provenance. Fall back to
    # the module's real file, which is a path and can be compared as one.
    module_path = str(getattr(meta, "source_path", "") or "")
    if not module_path:
        return False
    return Path(module_path).as_posix().startswith("skills/")


def _gates_required_for(effect_scope: Any, *, is_forged: bool) -> bool:
    """Whether an action's blast radius makes the derived gates mandatory.

    A forged skill is code Aura wrote for herself, and anything reaching
    outside the process can affect the world. For those, a gate that could
    not be evaluated must block. Purely internal, read-only work continues:
    a safety system that halts the runtime whenever a service hiccups buys
    no safety and costs availability.
    """
    if is_forged:
        return True
    scope = str(effect_scope or "").strip().lower()
    return scope not in {"", "none", "internal", "read_only", "readonly", "local_read"}


async def execute_tool(
    tool_name: str,
    parameters: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Module-level helper to execute a tool via the registered CapabilityEngine.

    Resolves the active skill_router instance from the ServiceContainer.
    """
    from core.runtime.service_access import optional_service
    engine: CapabilityEngine | None = optional_service("skill_router", default=None)
    if not engine:
        # CP126 128107a8. This used to CONSTRUCT a CapabilityEngine and
        # late-register it as the canonical skill_router whenever the real
        # one was missing. Tool execution is an authority path: building a
        # fresh one during a fault can kick off synchronous discovery and
        # publish a partially wired authority under the canonical name,
        # which then outlives the fault. The absence of the runtime engine
        # is a readiness failure and is reported as one.
        _record_capability_degradation(
            RuntimeError("skill_router service is not registered"),
            action="refused tool execution rather than constructing an ad hoc engine",
            severity="error",
        )
        return {
            "ok": False,
            "error": "capability engine is not ready",
            "reason": "capability_engine_unavailable",
            "tool": str(tool_name or ""),
            "readiness": {
                "service": "skill_router",
                "registered": False,
                "remedy": "the runtime must register skill_router during boot",
            },
        }

    params = parameters or {}
    raw_context = kwargs.pop("context", None)
    if raw_context is not None and not isinstance(raw_context, dict):
        return {"ok": False, "error": "tool execution context must be an object"}
    context = dict(raw_context or {})
    context.update(kwargs)
    
    # Map legacy virtual tool names used in tests/legacy flows
    if tool_name == "write_file":
        real_tool = "file_operation"
        real_params = {
            "action": "write",
            "path": params.get("file_path", params.get("path")),
            "content": params.get("content", "")
        }
        return await engine.execute(real_tool, real_params, context=context)
    elif tool_name == "read_file":
        real_tool = "file_operation"
        real_params = {
            "action": "read",
            "path": params.get("file_path", params.get("path"))
        }
        return await engine.execute(real_tool, real_params, context=context)

    return await engine.execute(tool_name, params, context=context)
