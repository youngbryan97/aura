"""Central model/runtime registry for Aura's local cognition lanes.

This module is the single source of truth for:
  - the logical Aura model lanes (Cortex / Solver / Brainstem / Reflex)
  - local artifact paths for Aura's MLX runtime
  - the active local backend selection, forced to MLX for live Aura
"""
import copy
import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.brain.llm.context_window_evidence import (
    ContextWindowEvidence,
    assumed,
    derived,
    measured,
    note_assumption,
)
from core.runtime.flags import FlagKind as _FlagKind
from core.runtime.flags import declare as _declare_flag
from core.runtime.runtime_settings import get_runtime_setting

# Declared flags (migrated from raw os.environ reads so the knobs are
# inventoried and reportable). STRING kind with the original literal
# default keeps read semantics byte-identical to os.environ.get.
_FLAG_BRAINSTEM_MODEL = _declare_flag(
    "AURA_BRAINSTEM_MODEL",
    kind=_FlagKind.STRING,
    # Qwen3.5-9B replaced Qwen2.5-7B on 12 Aug 2026. Unlike the 1.5B reflex
    # model — which is the speculative draft and contrastive amateur for the
    # cortex and therefore locked to the cortex's own distribution — nothing
    # is keyed to this tier's weights. Verified: no draft/amateur/contrastive
    # path references the brainstem, so the generation gap was free to close.
    default="Qwen3.5-9B-4bit",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_DEEP_MODEL = _declare_flag(
    "AURA_DEEP_MODEL",
    kind=_FlagKind.STRING,
    default=None,
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_DEEP_SPECIALIST_CERTIFICATE = _declare_flag(
    "AURA_DEEP_SPECIALIST_CERTIFICATE",
    kind=_FlagKind.STRING,
    default=None,
    description="Externally attested comparative-evidence certificate for the optional local specialist.",
    owner="model-registry",
)
_FLAG_DEEP_SPECIALIST_TRUST_ROOT = _declare_flag(
    "AURA_DEEP_SPECIALIST_TRUST_ROOT",
    kind=_FlagKind.STRING,
    default=None,
    description="Pinned Ed25519 public key that may attest optional specialist qualification.",
    owner="model-registry",
)
_FLAG_FALLBACK_MODEL = _declare_flag(
    "AURA_FALLBACK_MODEL",
    kind=_FlagKind.STRING,
    default="Qwen2.5-1.5B-Instruct-4bit",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_LANE_AUDIT_CACHE_TTL_S = _declare_flag(
    "AURA_LANE_AUDIT_CACHE_TTL_S",
    kind=_FlagKind.STRING,
    default="30",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_LLM__MLX_BRAINSTEM_PATH = _declare_flag(
    "AURA_LLM__MLX_BRAINSTEM_PATH",
    kind=_FlagKind.STRING,
    default=None,
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_LLM__MLX_DEEP_MODEL_PATH = _declare_flag(
    "AURA_LLM__MLX_DEEP_MODEL_PATH",
    kind=_FlagKind.STRING,
    default=None,
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_LLM__MLX_MODEL_PATH = _declare_flag(
    "AURA_LLM__MLX_MODEL_PATH",
    kind=_FlagKind.STRING,
    default=None,
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_LOCAL_BACKEND = _declare_flag(
    "AURA_LOCAL_BACKEND",
    kind=_FlagKind.STRING,
    default="mlx",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_LORA_PATH = _declare_flag(
    "AURA_LORA_PATH",
    kind=_FlagKind.STRING,
    default="",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_LORA_TARGET_MODEL = _declare_flag(
    "AURA_LORA_TARGET_MODEL",
    kind=_FlagKind.STRING,
    default="",
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)
_FLAG_MODEL = _declare_flag(
    "AURA_MODEL",
    kind=_FlagKind.STRING,
    default=None,
    description="Migrated from a raw environment read; see owner for the lane.",
    owner="flag-migration",
)


logger = logging.getLogger("Aura.ModelRegistry")

BASE_DIR = Path(os.getenv("AURA_ROOT", Path(__file__).resolve().parents[3]))
LOCAL_BACKEND = str(_FLAG_LOCAL_BACKEND.value()).strip().lower()


def get_models_dir() -> Path:
    """Return the model artifact root independently of the source checkout.

    Worktree-built desktop apps execute source from the worktree but share the
    large, immutable model inventory in the primary checkout.  Conflating those
    two roots made a valid Hugging Face repository ID get reinterpreted as a
    nonexistent path below the worktree.
    """

    configured = str(os.getenv("AURA_MODELS_DIR", "")).strip()
    return Path(configured).expanduser() if configured else BASE_DIR / "models"


def get_fused_model_root() -> Path:
    """Return the runtime-wide model promotion root.

    Promotion state belongs to the running Aura installation, not to an
    individual source worktree.  A worktree must therefore observe the same
    active manifest as the primary checkout or it can silently substitute an
    unqualified base checkpoint for the promoted cortex.
    """

    configured = str(os.getenv("AURA_FUSED_MODEL_ROOT", "")).strip()
    if configured:
        return Path(configured).expanduser()
    return BASE_DIR / "training" / "fused-model"


@dataclass(frozen=True)
class ActiveCortexSpec:
    """Immutable, validated view of the one active cortex pointer.

    JSON dictionaries are retained as canonical strings so callers cannot
    mutate the registry's cached authority object.  Accessors return fresh
    values when a subsystem needs the complete contract.
    """

    manifest_path: Path
    pointer_sha256: str
    schema_version: int
    model_path: Path
    base_model: str
    tag: str
    size_class: str
    descriptor_sha256: str
    repository_id: str
    revision: str
    serving_profile_sha256: str
    migration_contract_sha256: str
    evaluation_sha256: str
    exact_identity: bool
    promotion_qualified: bool
    predecessor_pointer_sha256: str = ""
    identity_transition_sha256: str = ""
    identity_transition_verified: bool = False
    _artifact_descriptor_json: str = ""
    _serving_profile_json: str = ""
    _migration_contract_json: str = ""

    @staticmethod
    def _decode(value: str) -> dict[str, object] | None:
        if not value:
            return None
        decoded = json.loads(value)
        return decoded if isinstance(decoded, dict) else None

    def artifact_descriptor(self) -> dict[str, object] | None:
        return self._decode(self._artifact_descriptor_json)

    def serving_profile(self) -> dict[str, object] | None:
        return self._decode(self._serving_profile_json)

    def migration_contract(self) -> dict[str, object] | None:
        return self._decode(self._migration_contract_json)


@dataclass(frozen=True)
class CortexServingLaneLimits:
    """One qualified input/output envelope from the active cortex profile."""

    name: str
    max_input_tokens: int
    max_output_tokens: int


@dataclass(frozen=True)
class CortexServingLimits:
    """Immutable serving limits bound to one exact active model artifact."""

    model_path: Path
    descriptor_sha256: str
    profile_sha256: str
    source: str
    qualified: bool
    served_context_tokens: int
    prefill_chunk_tokens: int
    lanes: tuple[CortexServingLaneLimits, ...]

    def lane(self, name: str) -> CortexServingLaneLimits | None:
        normalized = str(name or "").strip().lower()
        return next((lane for lane in self.lanes if lane.name == normalized), None)


_ACTIVE_CORTEX_SPEC_TTL_S = 5.0
_active_cortex_spec_cache: tuple[float, Path, ActiveCortexSpec | None] | None = None
_DEEP_SPECIALIST_STATUS_TTL_S = 30.0
_deep_specialist_status_cache: dict[
    str,
    tuple[float, tuple[object, ...], object],
] = {}


def _canonical_contract_json(value: dict[str, object] | None) -> str:
    if not isinstance(value, dict):
        return ""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _contract_digest(value: dict[str, object], *, digest_key: str) -> str:
    material = dict(value)
    material.pop(digest_key, None)
    return hashlib.sha256(_canonical_contract_json(material).encode("ascii")).hexdigest()


def _validate_identity_transition(
    *,
    manifest: Path,
    pointer: dict[str, object],
    active: Path,
    descriptor_sha256: str,
) -> tuple[str, str, bool]:
    transition = pointer.get("identity_transition")
    if transition is None:
        return "", "", False
    required = {
        "schema",
        "kind",
        "previous_pointer_sha256",
        "active_model_path",
        "model_descriptor_sha256",
        "transition_sha256",
    }
    if not isinstance(transition, dict) or set(transition) != required:
        raise ValueError("active_cortex_identity_transition_schema_invalid")
    previous_sha256 = str(transition.get("previous_pointer_sha256") or "")
    transition_sha256 = str(transition.get("transition_sha256") or "")
    if (
        transition.get("schema") != "aura.cortex_upgrade.identity_transition.v1"
        or transition.get("kind") != "model_identity_normalization"
        or not re.fullmatch(r"[0-9a-f]{64}", previous_sha256)
        or not re.fullmatch(r"[0-9a-f]{64}", transition_sha256)
        or transition_sha256
        != _contract_digest(transition, digest_key="transition_sha256")
        or transition.get("active_model_path") != str(active)
        or transition.get("model_descriptor_sha256") != descriptor_sha256
    ):
        raise ValueError("active_cortex_identity_transition_invalid")

    backup = manifest.with_name("active.json.identity-backup")
    if not backup.is_file() or backup.is_symlink() or backup.stat().st_size > 64 * 1024:
        raise ValueError("active_cortex_identity_backup_invalid")
    predecessor_raw = backup.read_bytes()
    if hashlib.sha256(predecessor_raw).hexdigest() != previous_sha256:
        raise ValueError("active_cortex_identity_backup_digest_mismatch")
    predecessor = json.loads(predecessor_raw)
    if not isinstance(predecessor, dict):
        raise ValueError("active_cortex_identity_backup_invalid")
    predecessor_raw_path = str(predecessor.get("active_model_path") or "").strip()
    if not predecessor_raw_path:
        raise ValueError("active_cortex_identity_backup_model_missing")
    predecessor_active = Path(predecessor_raw_path).expanduser().resolve(strict=True)
    if predecessor_active != active:
        raise ValueError("active_cortex_identity_backup_model_mismatch")

    stripped = dict(pointer)
    stripped.pop("artifact_descriptor", None)
    stripped.pop("identity_transition", None)
    stripped["schema_version"] = predecessor.get("schema_version")
    if stripped != predecessor:
        raise ValueError("active_cortex_identity_transition_not_narrow")
    return previous_sha256, transition_sha256, True


def _read_active_cortex_spec(manifest: Path | None = None) -> ActiveCortexSpec | None:
    manifest = manifest or (get_fused_model_root() / "active.json")
    try:
        raw = manifest.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("active_cortex_pointer_not_object")
        active_raw = str(payload.get("active_model_path") or "").strip()
        if not active_raw:
            raise ValueError("active_cortex_path_missing")
        active = Path(active_raw).expanduser().resolve(strict=True)
        if not active.is_dir():
            raise ValueError("active_cortex_path_not_directory")

        schema_version = int(payload.get("schema_version") or 0)
        descriptor: dict[str, object] | None = None
        descriptor_sha256 = ""
        repository_id = ""
        revision = ""
        exact_identity = False
        if schema_version >= 3:
            candidate = payload.get("artifact_descriptor")
            if not isinstance(candidate, dict):
                raise ValueError("active_model_descriptor_missing")
            from core.brain.llm.model_artifact_profile import (
                validate_model_artifact_descriptor,
            )

            validate_model_artifact_descriptor(candidate, model_path=active)
            descriptor = copy.deepcopy(candidate)
            descriptor_sha256 = str(descriptor.get("descriptor_sha256") or "")
            repository_id = str(descriptor.get("repository_id") or "")
            revision = str(descriptor.get("revision") or "")
            exact_identity = True

        (
            predecessor_pointer_sha256,
            identity_transition_sha256,
            identity_transition_verified,
        ) = _validate_identity_transition(
            manifest=manifest,
            pointer=payload,
            active=active,
            descriptor_sha256=descriptor_sha256,
        )

        serving = payload.get("serving_profile")
        migration = payload.get("migration_contract")
        evaluation = payload.get("evaluation")
        promotion_qualified = False
        contract_keys = tuple(
            name in payload
            for name in ("serving_profile", "migration_contract", "evaluation")
        )
        contract_presence = tuple(
            isinstance(value, dict) for value in (serving, migration, evaluation)
        )
        if any(contract_keys) and not all(contract_keys):
            raise ValueError("active_cortex_promotion_contract_partial")
        if all(contract_keys) and not all(contract_presence):
            raise ValueError("active_cortex_promotion_contract_invalid")
        if all(isinstance(value, dict) for value in (descriptor, serving, migration, evaluation)):
            assert isinstance(descriptor, dict)
            assert isinstance(serving, dict)
            assert isinstance(migration, dict)
            assert isinstance(evaluation, dict)
            from core.brain.llm.model_artifact_profile import (
                validate_model_serving_profile,
            )
            from core.learning.cortex_generation_upgrade import (
                validate_migration_contract,
                validate_upgrade_evaluation,
            )

            validate_model_serving_profile(serving, descriptor)
            validate_migration_contract(migration, descriptor)
            validate_upgrade_evaluation(
                evaluation,
                descriptor_sha256=descriptor_sha256,
            )
            promotion_qualified = True

        return ActiveCortexSpec(
            manifest_path=manifest.resolve(strict=False),
            pointer_sha256=hashlib.sha256(raw).hexdigest(),
            schema_version=schema_version,
            model_path=active,
            base_model=str(payload.get("base_model") or ""),
            tag=str(payload.get("tag") or active.name),
            size_class=str(payload.get("size") or ""),
            descriptor_sha256=descriptor_sha256,
            repository_id=repository_id,
            revision=revision,
            serving_profile_sha256=str(
                serving.get("profile_sha256") if isinstance(serving, dict) else ""
            ),
            migration_contract_sha256=str(
                migration.get("migration_contract_sha256")
                if isinstance(migration, dict)
                else ""
            ),
            evaluation_sha256=str(
                evaluation.get("evaluation_sha256")
                if isinstance(evaluation, dict)
                else ""
            ),
            exact_identity=exact_identity,
            promotion_qualified=promotion_qualified,
            predecessor_pointer_sha256=predecessor_pointer_sha256,
            identity_transition_sha256=identity_transition_sha256,
            identity_transition_verified=identity_transition_verified,
            _artifact_descriptor_json=_canonical_contract_json(descriptor),
            _serving_profile_json=_canonical_contract_json(
                serving if isinstance(serving, dict) else None
            ),
            _migration_contract_json=_canonical_contract_json(
                migration if isinstance(migration, dict) else None
            ),
        )
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.error("Active cortex pointer is invalid: %s", exc)
        return None


def get_active_cortex_spec(*, force_refresh: bool = False) -> ActiveCortexSpec | None:
    """Return the single validated active-cortex authority object."""
    global _active_cortex_spec_cache
    manifest = get_fused_model_root() / "active.json"
    now = time.monotonic()
    cached = _active_cortex_spec_cache
    if (
        not force_refresh
        and cached is not None
        and cached[1] == manifest
        and (now - cached[0]) < _ACTIVE_CORTEX_SPEC_TTL_S
    ):
        return cached[2]
    observed = _read_active_cortex_spec(manifest)
    _active_cortex_spec_cache = (now, manifest, observed)
    return observed


def get_active_cortex_serving_limits(
    model_path: str | Path | None = None,
    *,
    force_refresh: bool = False,
) -> CortexServingLimits | None:
    """Return the measured serving envelope for the exact active artifact.

    An old active pointer remains observable as ``legacy_unqualified`` but has
    no enforceable lane limits. This preserves its existing runtime behavior
    while making the absence of qualification explicit. A supplied path must
    identify the active artifact exactly; another local checkpoint cannot
    borrow its evidence.
    """

    spec = get_active_cortex_spec(force_refresh=force_refresh)
    if spec is None:
        return None
    active_path = spec.model_path.resolve(strict=False)
    if model_path is not None:
        requested_path = Path(str(model_path)).expanduser().resolve(strict=False)
        if requested_path != active_path:
            return None

    if not spec.promotion_qualified:
        descriptor = spec.artifact_descriptor() or {}
        artifact_profile = descriptor.get("artifact_profile")
        native_context = 0
        if isinstance(artifact_profile, dict):
            try:
                native_context = int(artifact_profile.get("native_context_window") or 0)
            except (TypeError, ValueError, OverflowError):
                native_context = 0
        return CortexServingLimits(
            model_path=active_path,
            descriptor_sha256=spec.descriptor_sha256,
            profile_sha256="",
            source="legacy_unqualified",
            qualified=False,
            served_context_tokens=max(0, native_context),
            prefill_chunk_tokens=0,
            lanes=(),
        )

    profile = spec.serving_profile()
    if not isinstance(profile, dict):
        return None
    raw_lanes = profile.get("lanes")
    if not isinstance(raw_lanes, dict):
        return None
    lanes: list[CortexServingLaneLimits] = []
    for name, raw_limits in sorted(raw_lanes.items()):
        if not isinstance(raw_limits, dict):
            return None
        try:
            lane = CortexServingLaneLimits(
                name=str(name).strip().lower(),
                max_input_tokens=int(raw_limits.get("max_input_tokens") or 0),
                max_output_tokens=int(raw_limits.get("max_output_tokens") or 0),
            )
        except (TypeError, ValueError, OverflowError):
            return None
        if not lane.name or lane.max_input_tokens <= 0 or lane.max_output_tokens <= 0:
            return None
        lanes.append(lane)
    try:
        served_context = int(profile.get("served_context_tokens") or 0)
        prefill_chunk = int(profile.get("prefill_chunk_tokens") or 0)
    except (TypeError, ValueError, OverflowError):
        return None
    if served_context <= 0 or prefill_chunk <= 0 or not lanes:
        return None
    return CortexServingLimits(
        model_path=active_path,
        descriptor_sha256=spec.descriptor_sha256,
        profile_sha256=spec.serving_profile_sha256,
        source="qualified_profile",
        qualified=True,
        served_context_tokens=served_context,
        prefill_chunk_tokens=prefill_chunk,
        lanes=tuple(lanes),
    )


def read_active_cortex_spec(manifest_path: str | Path) -> ActiveCortexSpec | None:
    """Validate one explicit pointer without consulting process-global roots."""

    return _read_active_cortex_spec(Path(manifest_path).expanduser())


PRIMARY_ENDPOINT = "Cortex"
DEEP_ENDPOINT = "Solver"
BRAINSTEM_ENDPOINT = "Brainstem"
FALLBACK_ENDPOINT = "Reflex"

# ── Recurrent Depth: Mythos-inspired layer looping per lane ──────────────
# Controls how many times the middle transformer layers loop before output.
# Higher loops = more "thinking" time in latent space before committing.
# This is the core architectural change: the model iterates on its hidden
# representation instead of doing a single pass through all layers.
RECURRENT_DEPTH_DEFAULTS = {
    PRIMARY_ENDPOINT:   2,   # Resident cortex; serving qualification owns the live ceiling.
    DEEP_ENDPOINT:      1,   # Optional specialist starts conservative until measured.
    BRAINSTEM_ENDPOINT: 1,   # Background lane; standard pass is the latency contract.
    FALLBACK_ENDPOINT:  1,   # Emergency lane; speed and availability take priority.
}

LEGACY_ENDPOINT_ALIASES = {
    "Local-MLX": PRIMARY_ENDPOINT,
    "MLX-Cortex": PRIMARY_ENDPOINT,
    "MLX-Solver": DEEP_ENDPOINT,
    "MLX-Brainstem": BRAINSTEM_ENDPOINT,
    "Reflex-CPU": FALLBACK_ENDPOINT,
}


_EXTERNAL_CORTEX_QUERIED = False


def external_llama_cortex_allowed() -> bool:
    """External server Cortex is retired for live Aura.

    Normal Aura desktop/runtime operation uses the in-process MLX lane so the
    substrate, affect, steering, memory, and response gates can act on the same
    live model path.  Keep the public function for compatibility, but make the
    answer permanently false — and leave a breadcrumb if anything still asks,
    so a lingering caller of the retired lane is visible instead of silent.
    """

    global _EXTERNAL_CORTEX_QUERIED
    if not _EXTERNAL_CORTEX_QUERIED:
        _EXTERNAL_CORTEX_QUERIED = True
        logging.getLogger("Aura.ModelRegistry").debug(
            "external_llama_cortex_allowed() queried; the external server Cortex "
            "lane is retired and permanently disabled."
        )
    return False


def _effective_local_backend() -> str:
    # MLX is the only supported local backend; any configured value coerces to
    # it so retired backends can never be resurrected through env/config drift.
    return "mlx"


def _normalize_backend_name(value: str | None) -> str:
    del value  # every backend name normalizes to the sole supported lane
    return "mlx"


def normalize_runtime_model_name(model_name: str | None, *, backend: str | None = None) -> str:
    """Map backend-incompatible logical names onto runnable local artifacts."""
    name = str(model_name or "").strip()
    if not name:
        return name

    normalized_backend = _normalize_backend_name(backend)
    if normalized_backend == "mlx":
        return {
            # Retired Q4 alias; the MLX runtime uses the 4-bit artifact/layout.
            "Qwen2.5-72B-Instruct-Q4": "Qwen2.5-72B-Instruct-4bit",
        }.get(name, name)
    return name


def _default_deep_model_name(*, backend: str | None = None) -> str:
    _normalize_backend_name(backend)
    # Deep reasoning is a cognition contract, not a parameter-count tier. The
    # resident cortex and Aura's reasoning systems own it unless an operator
    # deliberately configures a distinct local specialist.
    return str(_FLAG_MODEL.value() or "Qwen2.5-32B-Instruct-8bit")


# The logical name remains stable while the promoted artifact pointer changes.
# Deep reasoning defaults to this same resident model; a separate specialist is
# opt-in and must have a distinct model or artifact identity.
ACTIVE_MODEL = _FLAG_MODEL.value() or "Qwen2.5-32B-Instruct-8bit"
DEEP_MODEL = normalize_runtime_model_name(
    _FLAG_DEEP_MODEL.value() or _default_deep_model_name()
)
BRAINSTEM_MODEL = _FLAG_BRAINSTEM_MODEL.value()
FALLBACK_MODEL = _FLAG_FALLBACK_MODEL.value()

# Env-override for the primary (Cortex), solver, and brainstem model paths so
# a .env swap actually takes effect.  This is how we point Aura at the fused
# weight artifact from training/fused-model/ after a LoRA fuse — previously
# the hard-coded dict below made AURA_LLM__MLX_MODEL_PATH a no-op.
def _resolve_active_fused_model() -> str | None:
    """Read training/fused-model/active.json if present.

    The train_and_fuse pipeline writes this manifest after every successful
    fuse so Aura picks up the newest weights on next boot without anyone
    editing .env. An explicit AURA_LLM__MLX_MODEL_PATH still wins, so
    operators can pin a specific build for diagnostics.
    """
    spec = get_active_cortex_spec()
    return str(spec.model_path) if spec is not None else None


def get_active_model_artifact_descriptor(
    model_path: str | Path | None = None,
) -> dict[str, object] | None:
    """Return the exact identity bound to the active promoted cortex.

    Schema-v2 pointers predate exact basis identity and honestly return None.
    Schema-v3 pointers must validate and must name the model the caller has
    actually loaded; a same-width but different checkpoint is not compatible.
    """

    spec = get_active_cortex_spec()
    if spec is None or not spec.exact_identity:
        return None
    try:
        requested = Path(model_path or spec.model_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if requested != spec.model_path:
        logger.error("Active cortex artifact descriptor path mismatch: %s", requested)
        return None
    return spec.artifact_descriptor()


_CORTEX_NAME = "Qwen2.5-32B-Instruct-8bit"
_CORTEX_MANIFEST_TTL_S = 5.0
_cortex_path_cache: tuple[float, Path] | None = None

HF_FALLBACKS = {
    "Qwen2.5-1.5B-Instruct-4bit": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "Qwen3.5-9B-4bit": "mlx-community/Qwen3.5-9B-4bit",
    "Qwen2.5-7B-Instruct-4bit": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "Qwen2.5-32B-Instruct-8bit": "mlx-community/Qwen2.5-32B-Instruct-8bit",
    "Qwen2.5-32B-Instruct-4bit": "mlx-community/Qwen2.5-32B-Instruct-4bit",
    "Qwen2.5-72B-Instruct-4bit": "mlx-community/Qwen2.5-72B-Instruct-4bit",
    "Qwen2.5-72B-Instruct-Q4": "mlx-community/Qwen2.5-72B-Instruct-4bit",
    "QwQ-32B-4bit": "mlx-community/QwQ-32B-4bit",
    "DeepSeek-R1-Distill-Qwen-32B-4bit": (
        "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit"
    ),
    "DeepSeek-R1-Distill-Qwen-32B-8bit": (
        "mlx-community/DeepSeek-R1-Distill-Qwen-32B-MLX-8Bit"
    ),
}
_KNOWN_MODEL_REPOSITORY_IDS = frozenset(HF_FALLBACKS.values())


def is_model_repository_id(value: str | Path | None) -> bool:
    """Return whether *value* is a governed remote artifact identifier.

    This deliberately recognizes only registry-owned repositories.  Treating
    every relative ``owner/name`` string as remote would turn malformed local
    paths into implicit network fetches.
    """

    return str(value or "").strip() in _KNOWN_MODEL_REPOSITORY_IDS


def _current_cortex_path() -> Path:
    """Resolve the cortex artifact FRESH: env pin > fused manifest > default.

    The import-time constant this replaces froze the fused-model manifest at
    boot, so a weight promotion (training/fused-model/active.json) reached
    only the lane it was hot-swapped into — any later worker respawn
    re-resolved the stale path and silently resurrected the previous
    generation's weights. Model identity is a runtime decision; a short TTL
    keeps spawn paths cheap without letting them lag a promotion.
    """
    global _cortex_path_cache
    now = time.monotonic()
    if _cortex_path_cache is not None and (now - _cortex_path_cache[0]) < _CORTEX_MANIFEST_TTL_S:
        return _cortex_path_cache[1]
    resolved = Path(
        _FLAG_LLM__MLX_MODEL_PATH.value()
        or _resolve_active_fused_model()
        or str(get_models_dir() / _CORTEX_NAME)
    )
    _cortex_path_cache = (now, resolved)
    return resolved


_CORTEX_PATH = _current_cortex_path()  # legacy import-time view; do not add consumers
_SOLVER_PATH = Path(
    _FLAG_LLM__MLX_DEEP_MODEL_PATH.value()
    or str(get_models_dir() / "Qwen2.5-72B-Instruct-4bit")
)
_BRAINSTEM_PATH = Path(
    _FLAG_LLM__MLX_BRAINSTEM_PATH.value()
    or str(get_models_dir() / "Qwen3.5-9B-4bit")
)

MODEL_PATHS = {
    "Qwen2.5-1.5B-Instruct-4bit": BASE_DIR / "models" / "Qwen2.5-1.5B-Instruct-4bit",
    "Qwen3.5-9B-4bit":            _BRAINSTEM_PATH,
    "Qwen2.5-7B-Instruct-4bit":   BASE_DIR / "models" / "Qwen2.5-7B-Instruct-4bit",  # legacy
    "Qwen2.5-14B-Instruct-4bit":  BASE_DIR / "models" / "Qwen2.5-14B-Instruct-4bit",
    "Qwen2.5-32B-Instruct-8bit":  _CORTEX_PATH,
    "Qwen2.5-32B-Instruct-4bit":  BASE_DIR / "models" / "Qwen2.5-32B-Instruct-4bit",  # legacy
    "Qwen2.5-72B-Instruct-4bit":  _SOLVER_PATH,
    "QwQ-32B-4bit":               BASE_DIR / "models" / "QwQ-32B-4bit",
    "DeepSeek-R1-Distill-Qwen-32B-4bit": BASE_DIR / "models" / "DeepSeek-R1-Distill-Qwen-32B-4bit",
    "DeepSeek-R1-Distill-Qwen-32B-8bit": BASE_DIR / "models" / "DeepSeek-R1-Distill-Qwen-32B-8bit",
    "Qwen3-72B-Instruct":         BASE_DIR / "models" / "Qwen3-72B-Instruct",
    "Qwen2.5-72B-Instruct-Q4":    BASE_DIR / "models" / "Qwen2.5-72B-Instruct-Q4",
}

ADAPTER_PATH = BASE_DIR / "data" / "adapters"


# Tokenizer sentinel values (1e30) and corrupt metadata can advertise an
# impossible allocation; a too-small value would break every request. The
# registry bounds what an artifact is allowed to claim about itself.
_MIN_CONTEXT_WINDOW = 2048
_MAX_CONTEXT_WINDOW = 262144

#: Used only when nothing about the artifact is readable. It is a GUESS and
#: is always labelled WindowSource.ASSUMED so no caller can mistake it for a
#: measurement — the previous bare ``return 32768`` was indistinguishable
#: from a 32,768 read off config.json.
_DEFAULT_CONTEXT_WINDOW = 32768


def bounded_context_window(value: Any) -> int:
    """Clamp any claimed context window into the range this runtime will serve.

    Public because the registry is the only place that gets to say what a
    legal context window is. Callers that read an operator override — the
    inference gate's foreground budget, for one — were clamping the minimum
    and leaving the maximum open, so an environment typo could hand a prompt
    budget of a billion tokens to the compactor.
    """
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = _DEFAULT_CONTEXT_WINDOW
    return max(_MIN_CONTEXT_WINDOW, min(number, _MAX_CONTEXT_WINDOW))


#: Retained name for the registry's own internal call sites.
_bounded_context_window = bounded_context_window


def _safe_positive_int(value: Any) -> int:
    """Non-negative int from arbitrary JSON, never raising."""
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0
    return parsed if parsed > 0 else 0


def _coerce_bool(value: Any) -> bool:
    """Interpret JSON booleans AND their string spellings.

    bool("false") is True, so a string-valued config flag silently enabled
    behavior the artifact declared OFF.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_model_identity(value: str | None) -> str:
    text = os.path.basename(str(value or "").strip()).lower()
    if text.endswith(".gguf"):
        text = text[:-5]
    return text


def _model_identity_variants(value: str | None) -> set[str]:
    normalized = _normalize_model_identity(value)
    if not normalized:
        return set()

    variants = {normalized}
    size_tag = _extract_size_tag(normalized)
    if size_tag:
        variants.add(size_tag)

    # Drop common quantization / backend suffixes so the same base family can
    # match across MLX and retired external artifacts when explicitly intended.
    family = re.sub(r"-(?:q\d.*|[248]bit.*)$", "", normalized)
    if family:
        variants.add(family)

    return {item for item in variants if item}


def model_identities_compatible(expected_model: str | None, candidate_model: str | None) -> bool:
    expected = _model_identity_variants(expected_model)
    candidate = _model_identity_variants(candidate_model)
    if not expected or not candidate:
        return False

    expected_size = _extract_size_tag(str(expected_model or ""))
    candidate_size = _extract_size_tag(str(candidate_model or ""))
    exact_match = _normalize_model_identity(expected_model) == _normalize_model_identity(candidate_model)
    if exact_match:
        return True
    if expected_size and candidate_size and expected_size == candidate_size and expected.intersection(candidate):
        return True
    return False


def _read_adapter_target_model(adapter_dir: Path) -> str:
    config_path = Path(adapter_dir) / "adapter_config.json"
    if not config_path.exists():
        return ""
    try:
        payload = json.loads(config_path.read_text())
    except (json.JSONDecodeError, TypeError, ValueError):
        return ""
    return str(payload.get("model") or "").strip()


def get_model_path(model_name: str | None = None) -> str:
    """Resolve the path for a model. Returns absolute path if local, else HF repo ID."""
    raw_name = str(model_name or ACTIVE_MODEL).strip() or ACTIVE_MODEL
    if is_model_repository_id(raw_name):
        return raw_name
    explicit_path = Path(raw_name).expanduser()
    if explicit_path.is_absolute() or explicit_path.exists():
        return str(explicit_path.resolve() if explicit_path.exists() else explicit_path)

    name = normalize_runtime_model_name(raw_name)

    # The cortex lane resolves FRESH so a promoted fused artifact serves
    # every consumer (including worker respawns), not just the hot-swapped
    # client. See _current_cortex_path.
    if name == _CORTEX_NAME:
        cortex = _current_cortex_path().expanduser()
        if cortex.exists():
            return str(cortex.resolve())
        return HF_FALLBACKS.get(name, str(cortex))

    local_path = MODEL_PATHS.get(name, get_models_dir() / name)

    # If it's a Path object, check if it exists
    if isinstance(local_path, Path):
        local_path = local_path.expanduser()
        if local_path.exists():
            return str(local_path.resolve())
        # Fallback to repo ID if missing locally
        shared_path = get_models_dir() / name
        if shared_path != local_path and shared_path.exists():
            return str(shared_path.resolve())
        return HF_FALLBACKS.get(name, str(local_path))

    return str(local_path)


def get_local_backend() -> str:
    """Return Aura's effective local backend.

    This is intentionally dynamic.  Older builds captured ``AURA_LOCAL_BACKEND``
    at module import time, so stale launch environments could route live
    conversation into a generic external server. Treat every non-MLX value as
    MLX.
    """

    return _effective_local_backend()


def local_backend_is_mlx() -> bool:
    return get_local_backend() == "mlx"


def normalize_endpoint_name(endpoint_name: str | None) -> str | None:
    if endpoint_name is None:
        return None
    normalized = str(endpoint_name).strip()
    if not normalized:
        return normalized
    return LEGACY_ENDPOINT_ALIASES.get(normalized, normalized)


def _extract_size_tag(value: str | None) -> str:
    match = re.search(r"(\d+\.?\d*b)", str(value or "").lower())
    return match.group(1) if match else ""


def get_lane_model_name(endpoint_name: str | None) -> str:
    normalized = normalize_endpoint_name(endpoint_name) or PRIMARY_ENDPOINT
    if normalized == PRIMARY_ENDPOINT:
        return ACTIVE_MODEL
    if normalized == DEEP_ENDPOINT:
        return get_deep_model_name()
    if normalized == BRAINSTEM_ENDPOINT:
        return BRAINSTEM_MODEL
    return FALLBACK_MODEL


def _configured_deep_model_name() -> str:
    return normalize_runtime_model_name(str(_FLAG_DEEP_MODEL.value() or "").strip())


def _configured_deep_model_path() -> str:
    configured = str(_FLAG_LLM__MLX_DEEP_MODEL_PATH.value() or "").strip()
    if not configured:
        configured = str(get_runtime_setting("model.deep_path", "") or "").strip()
    return configured


def get_deep_model_name() -> str:
    """Return the distinct specialist name, or the resident model role.

    The old registry always returned a 72B name, which made lifecycle and
    health code treat that optional 38GB artifact as part of every Aura
    installation. An absent specialist now means resident deep reasoning.
    """

    configured_name = _configured_deep_model_name()
    if configured_name:
        return configured_name
    configured_path = _configured_deep_model_path()
    if configured_path:
        return Path(configured_path).name or ACTIVE_MODEL
    return ACTIVE_MODEL


def _canonical_model_locator(value: str | Path | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    expanded = os.path.expanduser(text)
    if os.path.isabs(expanded) or text.startswith(("~/", "./", "../")):
        # Role resolution runs on model status and admission paths. Lexical
        # normalization avoids realpath/stat work on those hot reads; clients
        # receive resolved artifact paths from this registry at construction.
        return os.path.normcase(os.path.abspath(expanded))
    return text.lower().rstrip("/")


def deep_solver_is_distinctly_configured() -> bool:
    """Whether a second local model was deliberately assigned to deep work."""

    configured_name = _configured_deep_model_name()
    configured_path = _configured_deep_model_path()
    if not configured_name and not configured_path:
        return False

    active_path = _canonical_model_locator(get_runtime_model_path(ACTIVE_MODEL))
    candidate_path = _canonical_model_locator(
        configured_path
        or get_runtime_model_path(configured_name)
    )
    if candidate_path and active_path and candidate_path == active_path:
        return False
    if configured_name and configured_name.lower() == ACTIVE_MODEL.lower() and not configured_path:
        return False
    return bool(candidate_path or configured_name)


def deep_solver_artifact_is_ready() -> bool:
    """Whether the configured specialist is a measured local model artifact."""

    if not deep_solver_is_distinctly_configured():
        return False
    try:
        from core.brain.llm.model_artifact_profile import get_model_artifact_profile

        profile = get_model_artifact_profile(get_deep_model_path())
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return bool(
        profile.exists
        and profile.measured
        and profile.weight_bytes > 0
        and profile.total_parameters > 0
    )


def get_deep_specialist_certificate_path() -> Path:
    configured = str(_FLAG_DEEP_SPECIALIST_CERTIFICATE.value() or "").strip()
    if configured:
        return Path(configured).expanduser()
    return get_fused_model_root() / "specialists" / "deep" / "admission.json"


def get_deep_specialist_trust_root_path() -> Path:
    configured = str(_FLAG_DEEP_SPECIALIST_TRUST_ROOT.value() or "").strip()
    if configured:
        return Path(configured).expanduser()
    return get_fused_model_root() / "specialists" / "deep" / "admission.pub.pem"


def _path_stat_signature(path: Path) -> tuple[object, ...]:
    try:
        stat = path.stat()
        return (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    except OSError:
        return (str(path), 0, 0, 0, 0)


def _deep_specialist_identity_signature(
    *,
    certificate_path: Path,
    trust_root_path: Path,
    specialist_path: Path,
) -> tuple[object, ...]:
    """Cheap invalidation key; a cache miss performs the full model hash."""

    signature: list[object] = [
        _path_stat_signature(certificate_path),
        _path_stat_signature(trust_root_path),
    ]
    try:
        children = sorted(
            child
            for child in specialist_path.iterdir()
            if child.is_file() or child.is_symlink()
        )
    except OSError:
        children = []
    signature.extend(_path_stat_signature(child) for child in children)
    try:
        from core.learning.specialist_cortex_admission import REQUIRED_SOURCE_CLOSURE

        signature.extend(
            _path_stat_signature(BASE_DIR / relative)
            for relative in sorted(REQUIRED_SOURCE_CLOSURE)
        )
    except ImportError:
        signature.append(("specialist_admission_import", 0))
    return tuple(signature)


def _current_specialist_source_commit() -> str:
    expected = str(os.getenv("AURA_LAUNCH_EXPECTED_COMMIT", "") or "").strip().lower()
    if expected:
        return expected
    from core.runtime.release_certificate import current_commit

    return str(current_commit() or "").strip().lower()


def get_deep_solver_admission_status(
    requested_domain: str | None = None,
    *,
    force_refresh: bool = False,
):
    """Return evidence qualification for the optional specialist.

    This is the one model-free admission authority used by startup, routing,
    health, and the API.  Host availability remains a live check in the
    inference gate; this status supplies the measured minima it must enforce.
    """

    from core.learning.specialist_cortex_admission import (
        SpecialistAdmissionError,
        denied_status,
        verify_specialist_qualification_certificate,
    )

    if not deep_solver_is_distinctly_configured():
        return denied_status("specialist_not_configured")
    if not deep_solver_artifact_is_ready():
        return denied_status("specialist_artifact_unmeasured")
    resident = get_active_cortex_spec(force_refresh=force_refresh)
    if (
        resident is None
        or not resident.exact_identity
        or not resident.promotion_qualified
        or not resident.descriptor_sha256
        or not resident.pointer_sha256
    ):
        return denied_status("resident_cortex_unqualified")

    certificate_path = get_deep_specialist_certificate_path()
    trust_root_path = get_deep_specialist_trust_root_path()
    specialist_path = Path(get_deep_model_path()).expanduser().resolve(strict=False)
    cache_key = str(requested_domain or "").strip().lower()
    signature = _deep_specialist_identity_signature(
        certificate_path=certificate_path,
        trust_root_path=trust_root_path,
        specialist_path=specialist_path,
    )
    now = time.monotonic()
    cached = _deep_specialist_status_cache.get(cache_key)
    cached_status = cached[2] if cached is not None else None
    cached_expiry = getattr(cached_status, "expires_at", None)
    cached_time_valid = not bool(getattr(cached_status, "admitted", False)) or (
        isinstance(cached_expiry, (int, float)) and time.time() < float(cached_expiry)
    )
    if (
        not force_refresh
        and cached is not None
        and (now - cached[0]) < _DEEP_SPECIALIST_STATUS_TTL_S
        and cached[1] == signature
        and cached_time_valid
    ):
        return cached_status
    try:
        status = verify_specialist_qualification_certificate(
            certificate_path,
            trusted_public_key_path=trust_root_path,
            source_root=BASE_DIR,
            current_source_commit=_current_specialist_source_commit(),
            resident_descriptor_sha256=resident.descriptor_sha256,
            resident_pointer_sha256=resident.pointer_sha256,
            specialist_model_path=specialist_path,
            requested_domain=requested_domain,
            verify_full_model_hash=True,
        )
    except SpecialistAdmissionError as exc:
        status = denied_status(exc.code)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("Optional specialist qualification failed closed: %s", exc)
        status = denied_status("specialist_qualification_error")
    _deep_specialist_status_cache[cache_key] = (now, signature, status)
    return status


def reset_deep_solver_admission_cache() -> None:
    _deep_specialist_status_cache.clear()


def get_lane_runtime_model_path(endpoint_name: str | None) -> str:
    return get_runtime_model_path(get_lane_model_name(endpoint_name))


def get_model_lane_role(model_path: str | Path | None) -> str | None:
    """Return the configured serving role for an exact model locator.

    Parameter count describes resource cost, not competence or authority.
    Only the registry can assign a serving artifact to Cortex, Solver,
    Brainstem, or Reflex. An unregistered artifact therefore has no semantic
    role even when its directory name contains one of those words.
    """

    candidate = _canonical_model_locator(model_path)
    if not candidate:
        return None
    assignments = (
        (PRIMARY_ENDPOINT, "cortex", True),
        (DEEP_ENDPOINT, "solver", deep_solver_is_distinctly_configured()),
        (BRAINSTEM_ENDPOINT, "brainstem", True),
        (FALLBACK_ENDPOINT, "reflex", True),
    )
    for endpoint, role, active in assignments:
        if not active:
            continue
        assigned = _canonical_model_locator(get_lane_runtime_model_path(endpoint))
        if assigned and candidate == assigned:
            return role
    return None


def _artifact_signature(model_path: Path) -> tuple:
    """Cheap identity of the on-disk artifact for cache invalidation.

    The context-window cache used to key on the LOGICAL model name alone, so
    promoting a new fused artifact under the same name kept serving the old
    limit for the life of the process. Including the config/tokenizer
    mtime+size means a promotion invalidates the entry naturally.
    """
    signature: list[Any] = [str(model_path)]
    for child in ("config.json", "tokenizer_config.json"):
        try:
            stat = (model_path / child).stat()
            signature.append((child, stat.st_mtime_ns, stat.st_size))
        except OSError:
            signature.append((child, 0, 0))
    return tuple(signature)


def get_model_context_window(model_name: str | None = None) -> int:
    """Return the effective context window for a local model.

    Prefer the model's true architectural limit from ``config.json``.
    Some tokenizers advertise larger theoretical windows in
    ``tokenizer_config.json`` that require explicit rope/scaling settings to
    be enabled; those should not silently become Aura's live runtime budget.
    """
    return int(get_context_window_evidence(model_name).tokens)


def get_context_window_evidence(model_name: str | None = None) -> ContextWindowEvidence:
    """The window, with the provenance of the number attached.

    Three dead ends in this resolution return the same default, and the
    caller could not tell any of them from a 32,768 that was actually read
    off the artifact. Sizing the whole prompt budget from a number nobody
    measured is the shape of the indefinite-coherence defect; this makes
    the difference legible.
    """
    name = normalize_runtime_model_name(model_name or ACTIVE_MODEL)
    model_path = MODEL_PATHS.get(name, BASE_DIR / "models" / str(name))
    if not isinstance(model_path, Path):
        return note_assumption(
            assumed(_DEFAULT_CONTEXT_WINDOW, model=name, detail="model path is not a filesystem path")
        )
    return _context_window_evidence_for_artifact(name, _artifact_signature(model_path))


@lru_cache(maxsize=32)
def _context_window_for_artifact_cached(name: str, signature: tuple) -> ContextWindowEvidence:
    model_path = Path(signature[0])

    config_path = model_path / "config.json"
    tokenizer_config_path = model_path / "tokenizer_config.json"

    max_position_embeddings = 0
    sliding_window = 0
    use_sliding_window = False
    tokenizer_model_max = 0

    try:
        if config_path.exists():
            config_payload = json.loads(config_path.read_text())
            if not isinstance(config_payload, dict):
                raise ValueError("model config.json is not an object")
            max_position_embeddings = _safe_positive_int(
                config_payload.get("max_position_embeddings")
            )
            sliding_window = _safe_positive_int(config_payload.get("sliding_window"))
            # JSON carries these as real booleans OR as strings. bool("false")
            # is True, so a string-valued flag silently ENABLED sliding-window
            # expansion and the registry then advertised the larger window.
            use_sliding_window = _coerce_bool(config_payload.get("use_sliding_window"))
    except (
        OSError,
        ConnectionError,
        TimeoutError,
        # A partial or malformed artifact raises these, and they used to
        # escape context resolution entirely instead of degrading to the
        # documented default.
        json.JSONDecodeError,
        UnicodeDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        logger.warning("Unreadable model config at %s: %s", config_path, exc)
        max_position_embeddings = 0
        sliding_window = 0
        use_sliding_window = False

    try:
        if tokenizer_config_path.exists():
            tokenizer_payload = json.loads(tokenizer_config_path.read_text())
            if not isinstance(tokenizer_payload, dict):
                raise ValueError("tokenizer_config.json is not an object")
            tokenizer_model_max = _safe_positive_int(
                tokenizer_payload.get("model_max_length")
            )
    except (
        OSError,
        ConnectionError,
        TimeoutError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        logger.warning(
            "Unreadable tokenizer config at %s: %s", tokenizer_config_path, exc
        )
        tokenizer_model_max = 0

    if max_position_embeddings > 0:
        # Respect the on-disk config unless sliding/YaRN is explicitly enabled.
        if use_sliding_window and sliding_window > max_position_embeddings:
            return derived(
                _bounded_context_window(max(sliding_window, max_position_embeddings)),
                model=name,
                detail="sliding_window explicitly enabled and larger than max_position_embeddings",
            )
        return measured(
            _bounded_context_window(max_position_embeddings), model=name
        )

    if sliding_window > 0 and use_sliding_window:
        return derived(
            _bounded_context_window(sliding_window),
            model=name,
            detail="sliding_window only; config.json carried no max_position_embeddings",
        )

    if tokenizer_model_max > 0:
        return derived(
            _bounded_context_window(tokenizer_model_max),
            model=name,
            detail="tokenizer_config.json model_max_length; may require rope scaling",
        )

    return note_assumption(
        assumed(
            _DEFAULT_CONTEXT_WINDOW,
            model=name,
            detail="no readable max_position_embeddings, sliding_window or tokenizer maximum",
        )
    )


def _context_window_evidence_for_artifact(name: str, signature: tuple) -> ContextWindowEvidence:
    """Artifact-keyed context resolution (thin wrapper over the LRU)."""
    return _context_window_for_artifact_cached(name, signature)


def _context_window_for_artifact(name: str, signature: tuple) -> int:
    return int(_context_window_for_artifact_cached(name, signature).tokens)


# get_model_context_window is a public entry point that callers (and this
# module's own test-reset helper) invalidate via .cache_clear(). The cache now
# lives on the artifact-keyed inner function, so the attribute is re-exposed
# here to keep that contract intact.
get_model_context_window.cache_clear = _context_window_for_artifact_cached.cache_clear
get_model_context_window.cache_info = _context_window_for_artifact_cached.cache_info


def get_lane_context_window(endpoint_name: str | None) -> int:
    architectural = get_model_context_window(get_lane_model_name(endpoint_name))
    if (normalize_endpoint_name(endpoint_name) or PRIMARY_ENDPOINT) != PRIMARY_ENDPOINT:
        return architectural
    limits = get_active_cortex_serving_limits()
    if limits is None or not limits.qualified:
        return architectural
    return min(architectural, limits.served_context_tokens)


def guard_solver_request(
    prefer_endpoint: str | None,
    *,
    deep_handoff: bool,
) -> dict[str, Any]:
    normalized = normalize_endpoint_name(prefer_endpoint)
    if normalized != DEEP_ENDPOINT or deep_handoff:
        return {
            "endpoint": normalized,
            "redirected": False,
            "reason": "",
        }
    return {
        "endpoint": PRIMARY_ENDPOINT,
        "redirected": True,
        "reason": "solver_redirected_without_explicit_deep_handoff",
    }


def get_endpoint_name_for_model(model_name: str | None) -> str:
    """Map a model name to its logical lane based on the configured tier layout."""
    name = str(model_name or ACTIVE_MODEL)
    lowered = name.lower()

    # Exact identity wins. Size alone is not a role: two unrelated checkpoints
    # can have the same parameter count, and a newer smaller resident can be
    # more useful than an older larger specialist.
    active_lower = ACTIVE_MODEL.lower()
    deep_name = get_deep_model_name()
    deep_lower = deep_name.lower()
    brainstem_lower = BRAINSTEM_MODEL.lower()
    fallback_lower = FALLBACK_MODEL.lower()

    if lowered == active_lower:
        return PRIMARY_ENDPOINT
    if deep_solver_is_distinctly_configured() and lowered == deep_lower:
        return DEEP_ENDPOINT
    if lowered == brainstem_lower:
        return BRAINSTEM_ENDPOINT
    if lowered == fallback_lower:
        return FALLBACK_ENDPOINT

    # Extract the core model identifier (e.g. "72b" from "qwen2.5-72b-instruct-q3_k_m-00001...")
    size_match = re.search(r'(\d+\.?\d*b)', lowered)
    model_size = size_match.group(1) if size_match else ""

    active_size = _extract_size_tag(ACTIVE_MODEL)
    deep_size = _extract_size_tag(deep_name) if deep_solver_is_distinctly_configured() else ""
    brainstem_size = _extract_size_tag(BRAINSTEM_MODEL)
    fallback_size = _extract_size_tag(FALLBACK_MODEL)

    # Match by model size across configured model identifiers.
    if model_size and model_size == active_size:
        return PRIMARY_ENDPOINT
    if model_size and model_size == deep_size:
        return DEEP_ENDPOINT
    if model_size and model_size == brainstem_size:
        return BRAINSTEM_ENDPOINT
    if model_size and model_size == fallback_size:
        return FALLBACK_ENDPOINT

    return PRIMARY_ENDPOINT


def _user_model_path_override(name: str) -> str | None:
    """Honor a user-set explicit model path (model.local_path / model.deep_path).

    Returns the configured path ONLY when it is set and exists on disk, so a
    blank or stale setting safely falls through to normal resolution. Maps the
    primary cortex lane to ``model.local_path`` and the deep solver lane to
    ``model.deep_path`` (docs/SETTINGS_WIRING_AUDIT.md).
    """
    if name == ACTIVE_MODEL:
        key = "model.local_path"
    elif deep_solver_is_distinctly_configured() and name == get_deep_model_name():
        key = "model.deep_path"
    else:
        return None
    value = str(get_runtime_setting(key, "") or "").strip()
    if value.lower().endswith(".gguf"):
        return None
    if value and Path(value).exists():
        return value
    return None


def get_runtime_model_path(model_name: str | None = None) -> str:
    """Resolve the active MLX runtime artifact for a lane."""
    name = model_name or ACTIVE_MODEL
    override = _user_model_path_override(name)
    if override:
        return override
    return get_model_path(name)


def get_runtime_download_target(model_name: str | None = None) -> dict[str, str]:
    _ = model_name
    return {}


def get_brainstem_path() -> str:
    """Resolve path for the brainstem (small/fast) model."""
    return get_runtime_model_path(BRAINSTEM_MODEL)


def get_deep_model_path() -> str:
    """Resolve a distinct specialist, or return the resident cortex path."""

    configured_path = _configured_deep_model_path()
    if configured_path:
        path = Path(configured_path).expanduser()
        return str(path.resolve()) if path.exists() else str(path)
    configured_name = _configured_deep_model_name()
    if configured_name:
        return get_runtime_model_path(configured_name)
    return get_runtime_model_path(ACTIVE_MODEL)


def get_fallback_path() -> str:
    """Resolve path for the emergency fallback (1.5B) model."""
    return get_runtime_model_path(FALLBACK_MODEL)


def get_active_model() -> str:
    """Return the name of the currently active cortex model."""
    return ACTIVE_MODEL


_LANE_AUDIT_CACHE_LOCK = threading.Lock()
_LANE_AUDIT_CACHE: dict[str, Any] = {"key": None, "at": 0.0, "result": None}


def reset_model_registry_caches_for_test() -> None:
    global _EXTERNAL_CORTEX_QUERIED, _active_cortex_spec_cache, _cortex_path_cache
    _EXTERNAL_CORTEX_QUERIED = False
    _active_cortex_spec_cache = None
    _cortex_path_cache = None
    get_model_context_window.cache_clear()
    with _LANE_AUDIT_CACHE_LOCK:
        _LANE_AUDIT_CACHE.update(key=None, at=0.0, result=None)


def _lane_audit_cache_ttl_s() -> float:
    try:
        return max(0.0, float(_FLAG_LANE_AUDIT_CACHE_TTL_S.value() or 30))
    except (TypeError, ValueError):
        return 30.0


def audit_lane_assignments(*, force_refresh: bool = False) -> dict[str, Any]:
    """Detect role drift so callers can surface it in health before runtime churn begins.

    The audit result is cached for a short TTL keyed by the lane→model
    assignment. Health probes call this on every is_ready()/UI bootstrap,
    and the uncached path does directory globs plus realpath per lane —
    blocking disk work that takes seconds exactly when the host is
    swapping (observed in stall dumps during the 110GB incident).
    """
    cache_key = "|".join(
        f"{endpoint}={get_lane_model_name(endpoint)}@{get_lane_runtime_model_path(endpoint)}"
        for endpoint in (
            PRIMARY_ENDPOINT,
            DEEP_ENDPOINT,
            BRAINSTEM_ENDPOINT,
            FALLBACK_ENDPOINT,
        )
    )
    ttl = _lane_audit_cache_ttl_s()
    now = time.monotonic()
    if not force_refresh and ttl > 0:
        with _LANE_AUDIT_CACHE_LOCK:
            cached = _LANE_AUDIT_CACHE
            if (
                cached["result"] is not None
                and cached["key"] == cache_key
                and (now - float(cached["at"])) < ttl
            ):
                return copy.deepcopy(cached["result"])

    result = _audit_lane_assignments_uncached()
    with _LANE_AUDIT_CACHE_LOCK:
        _LANE_AUDIT_CACHE.update(key=cache_key, at=now, result=copy.deepcopy(result))
    return result


def _audit_lane_assignments_uncached() -> dict[str, Any]:
    def _artifact_key(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        return os.path.realpath(text) if text.startswith("/") else text.lower()

    lanes: dict[str, dict[str, Any]] = {}
    for endpoint_name in (
        PRIMARY_ENDPOINT,
        DEEP_ENDPOINT,
        BRAINSTEM_ENDPOINT,
        FALLBACK_ENDPOINT,
    ):
        model_name = get_lane_model_name(endpoint_name)
        runtime_path = get_lane_runtime_model_path(endpoint_name)
        active = endpoint_name != DEEP_ENDPOINT or deep_solver_is_distinctly_configured()
        lanes[endpoint_name] = {
            "model": model_name,
            "runtime_path": runtime_path,
            "size_tag": _extract_size_tag(model_name),
            "active": active,
            "role_mode": (
                "distinct_specialist"
                if endpoint_name == DEEP_ENDPOINT and active
                else "resident_systems"
                if endpoint_name == DEEP_ENDPOINT
                else "model_lane"
            ),
        }

    issues: list[dict[str, Any]] = []
    seen_models: dict[str, str] = {}
    seen_paths: dict[str, str] = {}

    for endpoint_name, payload in lanes.items():
        if not bool(payload.get("active", True)):
            continue
        model_key = str(payload["model"]).strip().lower()
        if model_key:
            other_lane = seen_models.get(model_key)
            if other_lane and other_lane != endpoint_name:
                issues.append(
                    {
                        "kind": "duplicate_model_assignment",
                        "lanes": [other_lane, endpoint_name],
                        "detail": f"{payload['model']} is assigned to multiple lanes.",
                    }
                )
            else:
                seen_models[model_key] = endpoint_name

        path_key = _artifact_key(str(payload["runtime_path"]))
        if path_key:
            other_lane = seen_paths.get(path_key)
            if other_lane and other_lane != endpoint_name:
                issues.append(
                    {
                        "kind": "duplicate_runtime_path",
                        "lanes": [other_lane, endpoint_name],
                        "detail": f"{payload['runtime_path']} is serving multiple lanes.",
                    }
                )
            else:
                seen_paths[path_key] = endpoint_name

    cortex_size = str(lanes[PRIMARY_ENDPOINT].get("size_tag") or "")
    solver_size = str(lanes[DEEP_ENDPOINT].get("size_tag") or "")
    if (
        bool(lanes[DEEP_ENDPOINT].get("active", False))
        and cortex_size
        and solver_size
        and cortex_size == solver_size
    ):
        issues.append(
            {
                "kind": "cortex_solver_size_collision",
                "lanes": [PRIMARY_ENDPOINT, DEEP_ENDPOINT],
                "detail": f"Cortex and Solver are both configured as {cortex_size}.",
            }
        )

    return {
        "ok": not issues,
        "lanes": lanes,
        "issues": issues,
    }


def get_adapter_path() -> Path:
    """Return the LoRA adapter directory."""
    return ADAPTER_PATH


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _personality_lora_disabled(backend: str) -> bool:
    backend_key = "AURA_DISABLE_MLX_LORA"
    return _env_truthy("AURA_DISABLE_PERSONALITY_LORA") or _env_truthy(backend_key)


def _default_personality_lora_enabled(backend: str) -> bool:
    backend_key = "AURA_ENABLE_MLX_LORA"
    return _env_truthy("AURA_ENABLE_PERSONALITY_LORA") or _env_truthy(backend_key)


def resolve_personality_adapter(
    target_model: str | None,
    *,
    backend: str = "mlx",
) -> str | None:
    """Return a compatible Aura personality adapter for the requested model.

    MLX overrides are supported:
      - `AURA_LORA_PATH`, `AURA_LORA_TARGET_MODEL`

    The bundled personality adapter is opt-in. It has historically been useful
    for experiments, but live Cortex already runs against Aura-tuned/fused
    weights and the extra adapter can pollute user-visible prose. Explicit
    adapter paths still work; the default adapter only loads when an enable env
    flag is set.
    """
    normalized_backend = str(backend or "mlx").strip().lower()
    if normalized_backend != "mlx":
        return None
    target_model = str(target_model or "").strip()

    if _personality_lora_disabled(normalized_backend):
        return None

    adapter_dir = _FLAG_LORA_PATH.value().strip()
    if not adapter_dir:
        if not _default_personality_lora_enabled(normalized_backend):
            return None
        default_dir = BASE_DIR / "training" / "adapters" / "aura-personality"
        if (default_dir / "adapters.safetensors").exists():
            adapter_dir = str(default_dir)
    if not adapter_dir or not Path(adapter_dir).is_dir():
        return None

    configured_target = (
        _FLAG_LORA_TARGET_MODEL.value().strip()
        or _read_adapter_target_model(Path(adapter_dir))
        or ACTIVE_MODEL
    )
    if target_model and configured_target and not model_identities_compatible(configured_target, target_model):
        return None
    return adapter_dir
