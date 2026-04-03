"""Central model/runtime registry for Aura's local cognition lanes.

This module is the single source of truth for:
  - the logical Aura model lanes (Cortex / Solver / Brainstem / Reflex)
  - local artifact paths for both MLX and GGUF runtimes
  - the active local backend selection
"""
import os
import shutil
import re
from pathlib import Path
from typing import Any, Dict, Optional

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
def _detect_72b_q4():
    shard1 = BASE_DIR / "models_gguf" / "qwen2.5-72b-instruct-q4_k_m-00001-of-00012.gguf"
    try:
        return shard1.exists() and shard1.stat().st_size > 3_500_000_000
    except Exception:
        return False
_72B_READY = _detect_72b_q4()
# 32B Q5 as Cortex (fast, stable ~20s responses); 72B Q4 as Solver (deep reasoning, hot-swap)
# 72B Q4 is too slow (~84s) for primary use with Aura's background task architecture
ACTIVE_MODEL = os.getenv("AURA_MODEL") or "Qwen2.5-32B-Instruct-8bit"
DEEP_MODEL = os.getenv("AURA_DEEP_MODEL") or ("Qwen2.5-72B-Instruct-Q4" if _72B_READY else "Qwen2.5-72B-Instruct-4bit")
BRAINSTEM_MODEL = os.getenv("AURA_BRAINSTEM_MODEL", "Qwen2.5-7B-Instruct-4bit")
FALLBACK_MODEL = os.getenv("AURA_FALLBACK_MODEL", "Qwen2.5-1.5B-Instruct-4bit")

GGUF_DIR = BASE_DIR / "models_gguf"
MODEL_PATHS = {
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


def get_model_path(model_name: Optional[str] = None) -> str:
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


def find_llama_server_bin() -> Optional[str]:
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


def normalize_endpoint_name(endpoint_name: Optional[str]) -> Optional[str]:
    if endpoint_name is None:
        return None
    normalized = str(endpoint_name).strip()
    if not normalized:
        return normalized
    return LEGACY_ENDPOINT_ALIASES.get(normalized, normalized)


def _extract_size_tag(value: Optional[str]) -> str:
    match = re.search(r"(\d+\.?\d*b)", str(value or "").lower())
    return match.group(1) if match else ""


def get_lane_model_name(endpoint_name: Optional[str]) -> str:
    normalized = normalize_endpoint_name(endpoint_name) or PRIMARY_ENDPOINT
    if normalized == PRIMARY_ENDPOINT:
        return ACTIVE_MODEL
    if normalized == DEEP_ENDPOINT:
        return DEEP_MODEL
    if normalized == BRAINSTEM_ENDPOINT:
        return BRAINSTEM_MODEL
    return FALLBACK_MODEL


def get_lane_runtime_model_path(endpoint_name: Optional[str]) -> str:
    return get_runtime_model_path(get_lane_model_name(endpoint_name))


def guard_solver_request(
    prefer_endpoint: Optional[str],
    *,
    deep_handoff: bool,
) -> Dict[str, Any]:
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


def get_endpoint_name_for_model(model_name: Optional[str]) -> str:
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


def get_runtime_model_path(model_name: Optional[str] = None) -> str:
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


def get_runtime_download_target(model_name: Optional[str] = None) -> dict[str, str]:
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


def audit_lane_assignments() -> Dict[str, Any]:
    """Detect role drift so callers can surface it in health before runtime churn begins."""

    def _artifact_key(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        return os.path.realpath(text) if text.startswith("/") else text.lower()

    lanes: Dict[str, Dict[str, Any]] = {}
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

    issues: list[Dict[str, Any]] = []
    seen_models: Dict[str, str] = {}
    seen_paths: Dict[str, str] = {}

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
