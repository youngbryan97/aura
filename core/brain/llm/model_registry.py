"""Central model/runtime registry for Aura's local cognition lanes.

This module is the single source of truth for:
  - the logical Aura model lanes (Cortex / Solver / Brainstem / Reflex)
  - local artifact paths for both MLX and GGUF runtimes
  - the active local backend selection
"""
import json
import os
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

BASE_DIR = Path(os.getenv("AURA_ROOT", Path(__file__).resolve().parents[3]))
LOCAL_BACKEND = str(os.getenv("AURA_LOCAL_BACKEND", "llama_cpp")).strip().lower()

PRIMARY_ENDPOINT = "Cortex"
DEEP_ENDPOINT = "Solver"
BRAINSTEM_ENDPOINT = "Brainstem"
FALLBACK_ENDPOINT = "Reflex"

LEGACY_ENDPOINT_ALIASES = {
    "Local-MLX": PRIMARY_ENDPOINT,
    "MLX-Cortex": PRIMARY_ENDPOINT,
    "MLX-Solver": DEEP_ENDPOINT,
    "MLX-Brainstem": BRAINSTEM_ENDPOINT,
    "Reflex-CPU": FALLBACK_ENDPOINT,
}

# ── Change these lines to upgrade Aura's brain ──
# Auto-detect: Use 72B Q4 if downloaded, otherwise fall back to stable 32B Q5
def _detect_72b_q4() -> bool:
    shard1 = BASE_DIR / "models_gguf" / "qwen2.5-72b-instruct-q4_k_m-00001-of-00012.gguf"
    try:
        return shard1.exists() and shard1.stat().st_size > 3_500_000_000
    except Exception:
        return False
_72B_READY = _detect_72b_q4()
# 32B Q5 as Cortex (fast, stable ~20s responses); 72B Q4 as Solver (deep reasoning, hot-swap)
# 72B Q4 is too slow (~84s) for primary use with Aura's background task architecture
# [STABILITY v53] Use the fused personality model directly instead of
# base + separate LoRA adapter. The separate adapter causes intermittent
# float32 type errors in MLX when LoRA weights interact with 8-bit
# quantized compute graphs. Fused model has personality in the weights.
_FUSED_MODEL_DIR = BASE_DIR / "training" / "fused-model" / "Aura-32B-v4"
_FUSED_AVAILABLE = _FUSED_MODEL_DIR.is_dir() and (_FUSED_MODEL_DIR / "config.json").exists()
ACTIVE_MODEL = os.getenv("AURA_MODEL") or ("Aura-32B-v4" if _FUSED_AVAILABLE else "Qwen2.5-32B-Instruct-8bit")
DEEP_MODEL = os.getenv("AURA_DEEP_MODEL") or ("Qwen2.5-72B-Instruct-Q4" if _72B_READY else "Qwen2.5-72B-Instruct-4bit")
BRAINSTEM_MODEL = os.getenv("AURA_BRAINSTEM_MODEL", "Qwen2.5-7B-Instruct-4bit")
FALLBACK_MODEL = os.getenv("AURA_FALLBACK_MODEL", "Qwen2.5-1.5B-Instruct-4bit")

GGUF_DIR = BASE_DIR / "models_gguf"
MODEL_PATHS = {
    "Aura-32B-v4":                 BASE_DIR / "training" / "fused-model" / "Aura-32B-v4",
    "Qwen2.5-1.5B-Instruct-4bit": BASE_DIR / "models" / "Qwen2.5-1.5B-Instruct-4bit",
    "Qwen2.5-7B-Instruct-4bit":   BASE_DIR / "models" / "Qwen2.5-7B-Instruct-4bit",
    "Qwen2.5-14B-Instruct-4bit":  BASE_DIR / "models" / "Qwen2.5-14B-Instruct-4bit",
    "Qwen2.5-32B-Instruct-8bit":  BASE_DIR / "models" / "Qwen2.5-32B-Instruct-8bit",
    "Qwen2.5-32B-Instruct-4bit":  BASE_DIR / "models" / "Qwen2.5-32B-Instruct-4bit",  # legacy
    "Qwen2.5-72B-Instruct-4bit":  BASE_DIR / "models" / "Qwen2.5-72B-Instruct-4bit",
    "Qwen3-72B-Instruct":         BASE_DIR / "models" / "Qwen3-72B-Instruct",
    "Qwen2.5-72B-Instruct-Q4":    BASE_DIR / "models" / "Qwen2.5-72B-Instruct-Q4",
}

GGUF_MODEL_PATHS = {
    "Qwen2.5-1.5B-Instruct-4bit": Path(
        os.getenv(
            "AURA_FALLBACK_GGUF",
            str(GGUF_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        )
    ),
    "Qwen2.5-7B-Instruct-4bit": Path(
        os.getenv(
            "AURA_BRAINSTEM_GGUF",
            str(GGUF_DIR / "qwen2.5-7b-instruct-q4_k_m.gguf"),
        )
    ),
    "Qwen2.5-32B-Instruct-8bit": Path(
        os.getenv(
            "AURA_CORTEX_GGUF",
            str(GGUF_DIR / "qwen2.5-32b-instruct-q5_k_m.gguf"),
        )
    ),
    "Qwen2.5-32B-Instruct-4bit": Path(
        os.getenv(
            "AURA_CORTEX_GGUF",
            str(GGUF_DIR / "qwen2.5-32b-instruct-q5_k_m.gguf"),
        )
    ),
    "Qwen2.5-72B-Instruct-4bit": Path(
        os.getenv(
            "AURA_SOLVER_GGUF",
            str(GGUF_DIR / "qwen2.5-72b-instruct-q3_k_m.gguf"),
        )
    ),
    "Qwen3-72B-Instruct": Path(
        os.getenv(
            "AURA_SOLVER_GGUF",
            str(GGUF_DIR / "Qwen3-72B-Instruct.Q4_K_M.gguf"),
        )
    ),
    "Qwen2.5-72B-Instruct-Q4": Path(
        os.getenv(
            "AURA_SOLVER_GGUF",
            str(GGUF_DIR / "qwen2.5-72b-instruct-q4_k_m.gguf"),
        )
    ),
}

GGUF_DOWNLOAD_TARGETS = {
    "Qwen2.5-1.5B-Instruct-4bit": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "pattern": "qwen2.5-1.5b-instruct-q4_k_m*.gguf",
    },
    "Qwen2.5-7B-Instruct-4bit": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "pattern": "qwen2.5-7b-instruct-q4_k_m*.gguf",
    },
    "Qwen2.5-32B-Instruct-8bit": {
        "repo": "Qwen/Qwen2.5-32B-Instruct-GGUF",
        "pattern": "qwen2.5-32b-instruct-q5_k_m*.gguf",
    },
    "Qwen2.5-32B-Instruct-4bit": {
        "repo": "Qwen/Qwen2.5-32B-Instruct-GGUF",
        "pattern": "qwen2.5-32b-instruct-q5_k_m*.gguf",
    },
    "Qwen2.5-72B-Instruct-4bit": {
        "repo": "Qwen/Qwen2.5-72B-Instruct-GGUF",
        "pattern": "qwen2.5-72b-instruct-q3_k_m*.gguf",
    },
    "Qwen3-72B-Instruct": {
        "repo": "mradermacher/Qwen3-72B-Instruct-GGUF",
        "pattern": "Qwen3-72B-Instruct.Q4_K_M.gguf",
    },
}

ADAPTER_PATH = BASE_DIR / "data" / "adapters"


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
    # match across MLX and GGUF artifacts when explicitly intended.
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
    except Exception:
        return ""
    return str(payload.get("model") or "").strip()


def get_model_path(model_name: str | None = None) -> str:
    """Resolve the path for a model. Returns absolute path if local, else HF repo ID."""
    name = model_name or ACTIVE_MODEL
    
    # Mapping of local names to HF repo IDs for auto-download fallback
    HF_FALLBACKS = {
        "Qwen2.5-1.5B-Instruct-4bit": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "Qwen2.5-7B-Instruct-4bit":   "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "Qwen2.5-32B-Instruct-8bit":  "mlx-community/Qwen2.5-32B-Instruct-8bit",
        "Qwen2.5-32B-Instruct-4bit":  "mlx-community/Qwen2.5-32B-Instruct-4bit",
        "Qwen2.5-72B-Instruct-4bit":  "mlx-community/Qwen2.5-72B-Instruct-4bit",
    }
    
    local_path = MODEL_PATHS.get(name, BASE_DIR / "models" / name)
    
    # If it's a Path object, check if it exists
    if isinstance(local_path, Path):
        if local_path.exists():
            return str(local_path)
        # Fallback to repo ID if missing locally
        return HF_FALLBACKS.get(name, name)
        
    return str(local_path)


def local_backend_is_mlx() -> bool:
    return LOCAL_BACKEND == "mlx"


def get_local_backend() -> str:
    return LOCAL_BACKEND


def find_llama_server_bin() -> str | None:
    explicit = os.getenv("AURA_LLAMA_SERVER_BIN")
    if explicit:
        return explicit
    discovered = shutil.which("llama-server")
    if discovered:
        return discovered
    for candidate in ("/opt/homebrew/bin/llama-server", "/usr/local/bin/llama-server"):
        if Path(candidate).exists():
            return candidate
    return None


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


@lru_cache(maxsize=16)
def get_model_context_window(model_name: str | None = None) -> int:
    """Return the effective context window for a local model.

    Prefer the model's true architectural limit from ``config.json``.
    Some tokenizers advertise larger theoretical windows in
    ``tokenizer_config.json`` that require explicit rope/scaling settings to
    be enabled; those should not silently become Aura's live runtime budget.
    """
    name = model_name or ACTIVE_MODEL
    model_path = MODEL_PATHS.get(name, BASE_DIR / "models" / str(name))
    if not isinstance(model_path, Path):
        return 32768

    config_path = model_path / "config.json"
    tokenizer_config_path = model_path / "tokenizer_config.json"

    max_position_embeddings = 0
    sliding_window = 0
    use_sliding_window = False
    tokenizer_model_max = 0

    try:
        if config_path.exists():
            config_payload = json.loads(config_path.read_text())
            max_position_embeddings = int(config_payload.get("max_position_embeddings") or 0)
            sliding_window = int(config_payload.get("sliding_window") or 0)
            use_sliding_window = bool(config_payload.get("use_sliding_window"))
    except Exception:
        max_position_embeddings = 0
        sliding_window = 0
        use_sliding_window = False

    try:
        if tokenizer_config_path.exists():
            tokenizer_payload = json.loads(tokenizer_config_path.read_text())
            tokenizer_model_max = int(tokenizer_payload.get("model_max_length") or 0)
    except Exception:
        tokenizer_model_max = 0

    if max_position_embeddings > 0:
        # Respect the on-disk config unless sliding/YaRN is explicitly enabled.
        if use_sliding_window and sliding_window > max_position_embeddings:
            return max(sliding_window, max_position_embeddings)
        return max_position_embeddings

    if sliding_window > 0 and use_sliding_window:
        return sliding_window

    if tokenizer_model_max > 0:
        return tokenizer_model_max

    return 32768


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

    # Match by model size (most reliable for GGUF filenames)
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

    return PRIMARY_ENDPOINT


def get_runtime_model_path(model_name: str | None = None) -> str:
    """Resolve the active local-runtime artifact for a lane."""
    name = model_name or ACTIVE_MODEL
    if local_backend_is_mlx():
        return get_model_path(name)
    path = GGUF_MODEL_PATHS.get(name, GGUF_DIR / name)
    shard_prefix = f"{path.stem}-00001-of-*.gguf"
    shard_matches = sorted(path.parent.glob(shard_prefix))
    if shard_matches:
        return str(shard_matches[0])

    if path.exists():
        return str(path)

    wildcard_matches = sorted(path.parent.glob(f"{path.stem}*.gguf"))
    if wildcard_matches:
        return str(wildcard_matches[0])
    return str(path)


def get_runtime_download_target(model_name: str | None = None) -> dict[str, str]:
    name = model_name or ACTIVE_MODEL
    return dict(GGUF_DOWNLOAD_TARGETS.get(name, {}))


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


def audit_lane_assignments() -> dict[str, Any]:
    """Detect role drift so callers can surface it in health before runtime churn begins."""

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


def resolve_personality_adapter(
    target_model: str | None,
    *,
    backend: str = "mlx",
) -> str | None:
    """Return a compatible Aura personality adapter for the requested model.

    Backend-specific overrides are supported so MLX and GGUF can be pinned
    differently when needed:
      - `AURA_LORA_PATH`, `AURA_LORA_TARGET_MODEL`
      - `AURA_GGUF_LORA_PATH`, `AURA_GGUF_LORA_TARGET_MODEL`
    """
    normalized_backend = str(backend or "mlx").strip().lower()
    target_model = str(target_model or "").strip()

    if normalized_backend == "gguf":
        adapter_path = os.getenv("AURA_GGUF_LORA_PATH", "").strip()
        if not adapter_path:
            default_path = (
                BASE_DIR / "training" / "adapters" / "aura-personality" / "aura-personality-lora.gguf"
            )
            if default_path.exists():
                adapter_path = str(default_path)
        if not adapter_path or not Path(adapter_path).is_file():
            return None

        configured_target = (
            os.getenv("AURA_GGUF_LORA_TARGET_MODEL", "").strip()
            or os.getenv("AURA_LORA_TARGET_MODEL", "").strip()
            or ACTIVE_MODEL
        )
        if target_model and configured_target and not model_identities_compatible(configured_target, target_model):
            return None
        return adapter_path

    adapter_dir = os.getenv("AURA_LORA_PATH", "").strip()
    if not adapter_dir:
        default_dir = BASE_DIR / "training" / "adapters" / "aura-personality"
        if (default_dir / "adapters.safetensors").exists():
            adapter_dir = str(default_dir)
    if not adapter_dir or not Path(adapter_dir).is_dir():
        return None

    configured_target = (
        os.getenv("AURA_LORA_TARGET_MODEL", "").strip()
        or _read_adapter_target_model(Path(adapter_dir))
        or ACTIVE_MODEL
    )
    if target_model and configured_target and not model_identities_compatible(configured_target, target_model):
        return None
    return adapter_dir
