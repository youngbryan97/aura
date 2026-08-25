"""Configuration for Aura's Autonomous Architecture Governor."""
from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

from core.architect.models import MutationTier
from core.runtime.flags import FlagKind, declare

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
    "scratch",
    "training/adapters",
    "training/data",
    "training/datasets",
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


#: The architect's operator knobs, declared rather than parsed at each call
#: site. A declared flag has a type, a default, a description and an owner,
#: and shows up in flag_report(); a raw os.environ.get has none of those and
#: is discoverable only by grep.
_REPO_ROOT_FLAG = declare(
    "AURA_ASA_REPO_ROOT",
    kind=FlagKind.STRING,
    default="",
    description="Repository the self-architect operates on; empty means the working directory",
    owner="core.architect.config",
)
_ENABLED_FLAG = declare(
    "AURA_ASA_ENABLED",
    kind=FlagKind.BOOL,
    default=True,
    description="Whether the autonomous self-architect runs at all",
    owner="core.architect.config",
)
_AUTOPROMOTE_FLAG = declare(
    "AURA_ASA_AUTOPROMOTE",
    kind=FlagKind.BOOL,
    default=False,
    description="Whether a shadow mutation may promote itself without a human",
    owner="core.architect.config",
)
_MAX_TIER_FLAG = declare(
    "AURA_ASA_MAX_TIER",
    kind=FlagKind.STRING,
    default="T1",
    description="Highest mutation tier the architect may attempt",
    owner="core.architect.config",
)
_SHADOW_TIMEOUT_FLAG = declare(
    "AURA_ASA_SHADOW_TIMEOUT",
    kind=FlagKind.FLOAT,
    default=30.0,
    description="Seconds a shadow run may take before it is abandoned",
    owner="core.architect.config",
)
_OBSERVATION_WINDOW_FLAG = declare(
    "AURA_ASA_OBSERVATION_WINDOW",
    kind=FlagKind.FLOAT,
    default=10.0,
    description="Seconds a promoted mutation is watched before it is accepted",
    owner="core.architect.config",
)
_PROTECTED_PATHS_FLAG = declare(
    "AURA_ASA_PROTECTED_PATHS",
    kind=FlagKind.STRING,
    default="",
    description="Extra path patterns the architect may never modify, os.pathsep separated",
    owner="core.architect.config",
)
_SAFE_BOOT_COMMAND_FLAG = declare(
    "AURA_ASA_SAFE_BOOT_COMMAND",
    kind=FlagKind.STRING,
    default="",
    description="Command proving the runtime still boots after a mutation",
    owner="core.architect.config",
)
_RECEIPT_LIMIT_FLAG = declare(
    "AURA_ASA_RECEIPT_LIMIT",
    kind=FlagKind.INT,
    default=2000,
    description="Runtime receipts retained per architect session",
    owner="core.architect.config",
)
_COVERAGE_HIT_LIMIT_FLAG = declare(
    "AURA_ASA_COVERAGE_HIT_LIMIT",
    kind=FlagKind.INT,
    default=20000,
    description="Coverage hits retained per architect session",
    owner="core.architect.config",
)


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
    def from_env(cls, repo_root: str | Path | None = None) -> ASAConfig:
        root = Path(
            repo_root
            or str(_REPO_ROOT_FLAG.value() or "")
            or os.getcwd()
        ).resolve()
        enabled = bool(_ENABLED_FLAG.value())
        autopromote = bool(_AUTOPROMOTE_FLAG.value())
        max_tier = MutationTier.parse(str(_MAX_TIER_FLAG.value()))
        timeout = float(_SHADOW_TIMEOUT_FLAG.value())
        observation = float(_OBSERVATION_WINDOW_FLAG.value())
        protected = _merge_patterns(
            DEFAULT_PROTECTED_PATHS, str(_PROTECTED_PATHS_FLAG.value() or "")
        )
        safe_boot = _safe_boot_command_from_env(
            str(_SAFE_BOOT_COMMAND_FLAG.value() or "")
        )
        runtime_receipt_limit = int(_RECEIPT_LIMIT_FLAG.value())
        coverage_hit_limit = int(_COVERAGE_HIT_LIMIT_FLAG.value())
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
