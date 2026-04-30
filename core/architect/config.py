"""Configuration for Aura's Autonomous Architecture Governor."""
from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

from core.architect.models import MutationTier


DEFAULT_PROTECTED_PATHS: tuple[str, ...] = (
    "aura_main.py",
    "core/will.py",
    "core/constitution.py",
    "core/executive/authority_gateway.py",
    "core/executive/executive_core.py",
    "core/agency/capability_system.py",
    "core/agency/capability_token.py",
    "core/capability_engine.py",
    "core/memory/**gateway*.py",
    "core/state/**",
    "core/runtime/gateways.py",
    "core/runtime/atomic_writer.py",
    "core/runtime/conformance.py",
    "core/runtime/capability_tokens.py",
    "core/runtime/service_manifest.py",
    "core/runtime/activation_audit.py",
    "core/runtime/self_healing.py",
    "core/self_modification/**",
    "core/brain/llm/llm_router.py",
    "core/brain/llm/model_registry.py",
    "core/brain/llm/mlx_client.py",
    "core/identity/**",
    "core/self/canonical_self.py",
    "core/security/**",
    "core/guardians/**",
)


DEFAULT_SEALED_PATHS: tuple[str, ...] = (
    "core/architect/**",
    "core/self_modification/formal_verifier.py",
    "core/self_modification/mutation_tiers.py",
    "core/self_modification/mutation_safety.py",
    "core/runtime/atomic_writer.py",
    "core/runtime/conformance.py",
    "core/runtime/backup_restore.py",
    "core/runtime/restore_drill.py",
    "core/constitution.py",
    "core/will.py",
)


DEFAULT_EXCLUDES: tuple[str, ...] = (
    ".git",
    ".claude",
    ".agents",
    ".aura_architect",
    ".venv",
    ".venv_aura",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".aura_architect/shadow_runs",
    "data",
    "logs",
    "models",
    "model_weights",
    "checkpoints",
    "training/runs",
    "training/checkpoints",
    "*.sqlite",
    "*.sqlite3",
    "*.db",
    "*.log",
    "*.safetensors",
    "*.gguf",
    "*.mlx",
)


def default_safe_boot_command() -> tuple[str, ...]:
    return (sys.executable or "python3", "-B", "-m", "core.architect.safe_boot_harness")


@dataclass(frozen=True)
class ASAConfig:
    repo_root: Path
    enabled: bool = False
    autopromote: bool = False
    max_tier: MutationTier = MutationTier.T1_CLEANUP
    shadow_timeout: float = 30.0
    observation_window: float = 10.0
    artifact_root: Path | None = None
    protected_paths: tuple[str, ...] = DEFAULT_PROTECTED_PATHS
    sealed_paths: tuple[str, ...] = DEFAULT_SEALED_PATHS
    excludes: tuple[str, ...] = DEFAULT_EXCLUDES
    retain_shadow_runs: int = 10
    god_file_lines: int = 500
    god_class_lines: int = 300
    high_fan_in: int = 25
    high_fan_out: int = 35
    safe_boot_command: tuple[str, ...] = field(default_factory=default_safe_boot_command)
    runtime_receipt_limit: int = 2000
    coverage_hit_limit: int = 20000
    broader_pytest: bool = False
    env: dict[str, str] = field(default_factory=dict)

    @property
    def artifacts(self) -> Path:
        return self.artifact_root or (self.repo_root / ".aura_architect")

    @classmethod
    def from_env(cls, repo_root: str | Path | None = None) -> "ASAConfig":
        root = Path(
            repo_root
            or os.environ.get("AURA_ASA_REPO_ROOT")
            or os.getcwd()
        ).resolve()
        enabled = _env_bool("AURA_ASA_ENABLED", True)
        autopromote = _env_bool("AURA_ASA_AUTOPROMOTE", False)
        max_tier = MutationTier.parse(os.environ.get("AURA_ASA_MAX_TIER", "T1"))
        timeout = float(os.environ.get("AURA_ASA_SHADOW_TIMEOUT", "30"))
        observation = float(os.environ.get("AURA_ASA_OBSERVATION_WINDOW", "10"))
        protected = _merge_patterns(DEFAULT_PROTECTED_PATHS, os.environ.get("AURA_ASA_PROTECTED_PATHS", ""))
        safe_boot = _safe_boot_command_from_env(os.environ.get("AURA_ASA_SAFE_BOOT_COMMAND", ""))
        runtime_receipt_limit = int(os.environ.get("AURA_ASA_RECEIPT_LIMIT", "2000"))
        coverage_hit_limit = int(os.environ.get("AURA_ASA_COVERAGE_HIT_LIMIT", "20000"))
        return cls(
            repo_root=root,
            enabled=enabled,
            autopromote=autopromote,
            max_tier=max_tier,
            shadow_timeout=timeout,
            observation_window=observation,
            protected_paths=protected,
            safe_boot_command=safe_boot,
            runtime_receipt_limit=runtime_receipt_limit,
            coverage_hit_limit=coverage_hit_limit,
            env=dict(os.environ),
        )

    def rel(self, path: str | Path) -> str:
        target = Path(path)
        if target.is_absolute():
            return target.resolve().relative_to(self.repo_root).as_posix()
        return _clean_rel(target.as_posix())

    def is_excluded(self, path: str | Path) -> bool:
        rel = _clean_rel(str(path).replace("\\", "/"))
        parts = set(Path(rel).parts)
        for pattern in self.excludes:
            normalized = pattern.replace("\\", "/")
            if normalized in parts or fnmatch(rel, normalized) or fnmatch(Path(rel).name, normalized):
                return True
        return False

    def is_protected(self, path: str | Path) -> bool:
        rel = _clean_rel(str(path).replace("\\", "/"))
        return any(fnmatch(rel, pattern) for pattern in self.protected_paths)

    def is_sealed(self, path: str | Path) -> bool:
        rel = _clean_rel(str(path).replace("\\", "/"))
        return any(fnmatch(rel, pattern) for pattern in self.sealed_paths)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _merge_patterns(base: tuple[str, ...], extra: str) -> tuple[str, ...]:
    additions = tuple(part.strip() for part in extra.split(os.pathsep) if part.strip())
    return tuple(dict.fromkeys(base + additions))


def _safe_boot_command_from_env(raw: str) -> tuple[str, ...]:
    value = raw.strip()
    if not value:
        return default_safe_boot_command()
    if value.lower() in {"0", "false", "off", "disabled", "none"}:
        return ()
    return tuple(shlex.split(value))


def _clean_rel(path: str) -> str:
    if path == ".":
        return "."
    if path.startswith("./"):
        return path[2:]
    return path
