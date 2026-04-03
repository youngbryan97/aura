"""Aura model fetcher for both MLX and managed GGUF runtimes."""

from __future__ import annotations

from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.model_registry import (
    ACTIVE_MODEL,
    BRAINSTEM_MODEL,
    DEEP_MODEL,
    FALLBACK_MODEL,
    GGUF_DIR,
    GGUF_MODEL_PATHS,
    MODEL_PATHS,
    get_local_backend,
    get_runtime_download_target,
)


MODEL_PLAN = [
    (BRAINSTEM_MODEL, "⚡ Brainstem (Background/Reflex)"),
    (ACTIVE_MODEL, "🧠 Cortex (Daily Executive Brain)"),
    (DEEP_MODEL, "🔮 Solver (Hot-Swap Deep Thinker)"),
    (FALLBACK_MODEL, "🚨 Fallback (Emergency Last Resort)"),
]

MLX_REPOS = {
    "Qwen2.5-1.5B-Instruct-4bit": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "Qwen2.5-7B-Instruct-4bit": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "Qwen2.5-32B-Instruct-4bit": "mlx-community/Qwen2.5-32B-Instruct-4bit",
    "Qwen2.5-32B-Instruct-8bit": "mlx-community/Qwen2.5-32B-Instruct-8bit",
    "Qwen2.5-72B-Instruct-4bit": "mlx-community/Qwen2.5-72B-Instruct-4bit",
}


_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def _repair_sharded_layout(target: Path, pattern: str) -> list[Path]:
    matches = sorted(target.parent.glob(pattern))
    sharded = [path for path in matches if _SHARD_RE.search(path.name)]
    if target.exists() and sharded and not any("-00001-of-" in path.name for path in sharded):
        shard_match = _SHARD_RE.search(sharded[0].name)
        if shard_match:
            shard_one = target.parent / f"{target.stem}-00001-of-{shard_match.group(2)}.gguf"
            if not shard_one.exists():
                target.rename(shard_one)
            matches = sorted(target.parent.glob(pattern))
    return matches


def _gguf_artifact_ready(target: Path, pattern: str) -> tuple[bool, list[Path]]:
    matches = _repair_sharded_layout(target, pattern)
    if target.exists() and not matches:
        return True, [target]
    sharded = [path for path in matches if _SHARD_RE.search(path.name)]
    if sharded:
        totals = {int(_SHARD_RE.search(path.name).group(2)) for path in sharded if _SHARD_RE.search(path.name)}
        if len(totals) == 1:
            total = next(iter(totals))
            present = {
                int(_SHARD_RE.search(path.name).group(1))
                for path in sharded
                if _SHARD_RE.search(path.name)
            }
            return present == set(range(1, total + 1)), matches
    return bool(target.exists() or matches), matches


def _ensure_gguf_artifact(target: Path, pattern: str) -> list[Path]:
    ready, matches = _gguf_artifact_ready(target, pattern)
    if ready:
        return matches
    if matches:
        return matches
    raise FileNotFoundError(f"No GGUF matched {pattern} in {target.parent}")


def _fetch_mlx_models() -> None:
    from huggingface_hub import snapshot_download

    base_dir = Path(__file__).resolve().parent.parent / "models"
    base_dir.mkdir(parents=True, exist_ok=True)

    print("🧠 Aura Model Fetcher — MLX artifact mode")
    print("=" * 45)

    for index, (model_name, description) in enumerate(MODEL_PLAN, 1):
        repo = MLX_REPOS[model_name]
        target = MODEL_PATHS[model_name]
        if target.exists() and any(target.iterdir()):
            print(f"  [{index}/{len(MODEL_PLAN)}] {description}: Already exists, skipping")
            continue

        print(f"  [{index}/{len(MODEL_PLAN)}] {description}: Downloading {repo}...")
        snapshot_download(
            repo_id=repo,
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )
        print(f"    ✅ Downloaded to {target}")


def _fetch_gguf_models() -> None:
    from huggingface_hub import snapshot_download

    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    print("🧠 Aura Model Fetcher — Managed local runtime (GGUF mode)")
    print("=" * 58)

    for index, (model_name, description) in enumerate(MODEL_PLAN, 1):
        spec = get_runtime_download_target(model_name)
        target = GGUF_MODEL_PATHS[model_name]
        ready, existing_matches = _gguf_artifact_ready(target, pattern=spec["pattern"]) if spec else (target.exists(), [])
        if ready:
            print(f"  [{index}/{len(MODEL_PLAN)}] {description}: Already exists, skipping")
            continue
        if not spec:
            print(f"  [{index}/{len(MODEL_PLAN)}] {description}: No GGUF target configured, skipping")
            continue

        repo = spec["repo"]
        pattern = spec["pattern"]
        print(f"  [{index}/{len(MODEL_PLAN)}] {description}: Downloading {repo} ({pattern})...")
        snapshot_download(
            repo_id=repo,
            local_dir=str(target.parent),
            local_dir_use_symlinks=False,
            allow_patterns=[pattern],
        )
        resolved_matches = _ensure_gguf_artifact(target, pattern)
        if target.exists():
            resolved_path = target
        elif resolved_matches:
            resolved_path = resolved_matches[0]
        else:
            resolved_path = target
        print(f"    ✅ Downloaded to {resolved_path}")


if __name__ == "__main__":
    backend = get_local_backend()
    if backend == "mlx":
        _fetch_mlx_models()
    else:
        _fetch_gguf_models()

    print()
    print("✅ Model fetch complete.")
