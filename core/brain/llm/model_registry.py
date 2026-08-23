"""Central model/runtime registry for Aura's local cognition lanes.

This module is the single source of truth for:
  - the logical Aura model lanes (Cortex / Solver / Brainstem / Reflex)
  - local artifact paths for Aura's MLX runtime
  - the active local backend selection, forced to MLX for live Aura
"""
import copy
import json
import logging
import os
import re
import threading
import time
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
    PRIMARY_ENDPOINT:   2,   # Cortex (32B) — 2 loops, meaningful improvement
    DEEP_ENDPOINT:      1,   # Solver (72B) — standard pass by default on 64GB-class desktops
    BRAINSTEM_ENDPOINT: 1,   # Brainstem (7B) — standard pass, too small to benefit
    FALLBACK_ENDPOINT:  1,   # Reflex (1.5B) — standard pass, speed is priority
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
    return "Qwen2.5-72B-Instruct-4bit"


# 32B Q5 as Cortex (fast, stable ~20s responses); 72B Q4 as Solver (deep reasoning, hot-swap)
# 72B Q4 is too slow (~84s) for primary use with Aura's background task architecture
# [STABILITY v53.9] Use 8-bit base model + LoRA adapter at runtime.
# Re-quantized fused models degrade quality (repetition loops, wrong answers).
# The separate adapter has intermittent float32 errors but most generations
# succeed — the worker catches and retries on failure.
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
    manifest = get_fused_model_root() / "active.json"
    try:
        if not manifest.exists():
            return None
        data = json.loads(manifest.read_text())
        path = str(data.get("active_model_path") or "").strip()
        if not path:
            return None
        if not Path(path).exists():
            return None
        if int(data.get("schema_version") or 0) >= 3:
            from core.brain.llm.model_artifact_profile import (
                validate_model_artifact_descriptor,
            )

            descriptor = data.get("artifact_descriptor")
            if not isinstance(descriptor, dict):
                logger.error("Active cortex schema v3 has no artifact descriptor")
                return None
            validate_model_artifact_descriptor(descriptor, model_path=path)
        return path
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def get_active_model_artifact_descriptor(
    model_path: str | Path | None = None,
) -> dict[str, object] | None:
    """Return the exact identity bound to the active promoted cortex.

    Schema-v2 pointers predate exact basis identity and honestly return None.
    Schema-v3 pointers must validate and must name the model the caller has
    actually loaded; a same-width but different checkpoint is not compatible.
    """

    manifest = get_fused_model_root() / "active.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or int(payload.get("schema_version") or 0) < 3:
            return None
        descriptor = payload.get("artifact_descriptor")
        if not isinstance(descriptor, dict):
            raise ValueError("active_model_descriptor_missing")
        active_path = str(payload.get("active_model_path") or "").strip()
        requested = Path(model_path or active_path).expanduser().resolve(strict=True)
        active = Path(active_path).expanduser().resolve(strict=True)
        if requested != active:
            raise ValueError("active_model_descriptor_path_mismatch")
        from core.brain.llm.model_artifact_profile import (
            validate_model_artifact_descriptor,
        )

        validate_model_artifact_descriptor(descriptor, model_path=active)
        return copy.deepcopy(descriptor)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.error("Active cortex artifact descriptor is invalid: %s", exc)
        return None


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
        return DEEP_MODEL
    if normalized == BRAINSTEM_ENDPOINT:
        return BRAINSTEM_MODEL
    return FALLBACK_MODEL


def get_lane_runtime_model_path(endpoint_name: str | None) -> str:
    return get_runtime_model_path(get_lane_model_name(endpoint_name))


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
    return get_model_context_window(get_lane_model_name(endpoint_name))


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

    # Match against configured tier assignments (not hardcoded sizes)
    active_lower = ACTIVE_MODEL.lower()
    deep_lower = DEEP_MODEL.lower()
    brainstem_lower = BRAINSTEM_MODEL.lower()
    fallback_lower = FALLBACK_MODEL.lower()

    # Extract the core model identifier (e.g. "72b" from "qwen2.5-72b-instruct-q3_k_m-00001...")
    size_match = re.search(r'(\d+\.?\d*b)', lowered)
    model_size = size_match.group(1) if size_match else ""

    active_size = _extract_size_tag(ACTIVE_MODEL)
    deep_size = _extract_size_tag(DEEP_MODEL)
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

    # Exact name match fallback
    if lowered == active_lower:
        return PRIMARY_ENDPOINT
    if lowered == deep_lower:
        return DEEP_ENDPOINT
    if lowered == brainstem_lower:
        return BRAINSTEM_ENDPOINT
    if lowered == fallback_lower:
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
    elif name == DEEP_MODEL:
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
    """Resolve path for the deep solver (72B) model."""
    return get_runtime_model_path(DEEP_MODEL)


def get_fallback_path() -> str:
    """Resolve path for the emergency fallback (1.5B) model."""
    return get_runtime_model_path(FALLBACK_MODEL)


def get_active_model() -> str:
    """Return the name of the currently active cortex model."""
    return ACTIVE_MODEL


_LANE_AUDIT_CACHE_LOCK = threading.Lock()
_LANE_AUDIT_CACHE: dict[str, Any] = {"key": None, "at": 0.0, "result": None}


def reset_model_registry_caches_for_test() -> None:
    global _EXTERNAL_CORTEX_QUERIED, _cortex_path_cache
    _EXTERNAL_CORTEX_QUERIED = False
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
        f"{endpoint}={get_lane_model_name(endpoint)}"
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
        lanes[endpoint_name] = {
            "model": model_name,
            "runtime_path": runtime_path,
            "size_tag": _extract_size_tag(model_name),
        }

    issues: list[dict[str, Any]] = []
    seen_models: dict[str, str] = {}
    seen_paths: dict[str, str] = {}

    for endpoint_name, payload in lanes.items():
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
    if cortex_size and solver_size and cortex_size == solver_size:
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
