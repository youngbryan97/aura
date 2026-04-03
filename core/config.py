"""core/config.py
Hardened Configuration System for Aura.
2026 Standards: Pydantic Settings V2, Strict Validation, and Mycelial Observability.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from enum import Enum
from pathlib import Path
from typing import ClassVar, Optional, Union, List, Dict, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger("Aura.Config")


class Environment(str, Enum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class SomaConfig(BaseModel):
    fps: int = 20
    enabled: bool = True

class FeatureToggles(BaseModel):
    voice_enabled: bool = True
    # Keep the in-process webcam disabled by default on macOS.
    # OpenCV + PyAV/Whisper can collide through AVFoundation during desktop boot.
    camera_enabled: bool = False
    mycelium_visualizer: bool = True
    autonomous_impulses: bool = True


class Paths(BaseModel):
    """
    Centralized path configuration with platform-aware resolution.
    ISSUE #78: Note that @property fields are intentionally excluded from Pydantic serialization.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _runtime_home_cache: ClassVar[Optional[Path]] = None

    home_dir: Path = Field(default_factory=lambda: Path.home().expanduser().resolve() / ".aura")

    def _effective_home_dir(self) -> Path:
        cached = self.__class__._runtime_home_cache
        if cached is not None:
            return cached

        candidate = self.home_dir.expanduser().resolve()
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".aura_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            self.__class__._runtime_home_cache = candidate
            return candidate
        except Exception as exc:
            fallback = self.project_root / ".aura_runtime"
            fallback.mkdir(parents=True, exist_ok=True)
            self.__class__._runtime_home_cache = fallback
            logger.warning(
                "Paths.home_dir unavailable (%s). Falling back to %s",
                exc,
                fallback,
            )
            return fallback

    @property
    def project_root(self) -> Path:
        aura_root = os.environ.get("AURA_ROOT")
        if aura_root:
            return Path(aura_root).resolve()
        return Path(__file__).resolve().parent.parent

    @property
    def base_dir(self) -> Path:
        """Alias for project_root used by legacy subsystems."""
        return self.project_root

    @property
    def data_dir(self) -> Path:
        return self._effective_home_dir() / "data"

    @property
    def log_dir(self) -> Path:
        return self._effective_home_dir() / "logs"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "generated_images"

    @property
    def memory_dir(self) -> Path:
        return self.data_dir / "memory"

    @property
    def brain_dir(self) -> Path:
        """Alias for home_dir used by Volition and audit logs."""
        return self._effective_home_dir() / "brain"

    def create_directories(self) -> None:
        root = self._effective_home_dir()
        for d in [root, self.data_dir, self.log_dir, self.uploads_dir, self.images_dir, self.memory_dir, self.brain_dir]:
            d.mkdir(parents=True, exist_ok=True)


class SecurityConfig(BaseModel):
    internal_only_mode: bool = False
    auto_fix_enabled: bool = True
    aura_full_autonomy: bool = True
    max_modifications_per_day: int = 100
    allow_network_access: bool = True
    allowed_domains: list[str] = Field(default_factory=lambda: ["*"])
    enable_stealth_mode: bool = True
    force_unity_on: bool = False


class DynamicScalingConfig(BaseModel):
    min_interval: float = 0.5
    max_interval: float = 10.0
    adjustment_step: float = 0.1

class AegisConfig(BaseModel):
    enabled: bool = True
    sentinel_interval: float = 5.0
    auto_heal: bool = True
    scaling: DynamicScalingConfig = Field(default_factory=DynamicScalingConfig)

class SafeModificationConfig(BaseModel):
    allowed_paths: list[str] = Field(default_factory=lambda: ["core/", "skills/", "interface/"])
    protected_paths: list[str] = Field(default_factory=lambda: [
        "core/security/",
        "core/guardians/",
        "core/prime_directives.py",
        "core/constitution.py",
        "core/config.py",
    ])
    max_risk_level: int = 6
    max_lines_changed: int = 100
    backup_max_age_days: int = 7
    auto_commit: bool = True


class RedisConfig(BaseModel):
    url: str = Field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    enabled: bool = True
    use_for_events: bool = True
    
    @property
    def broker_url(self) -> str:
        if os.environ.get("AURA_REDIS_ENABLED") == "0":
            return "memory://"
        return self.url

    @property
    def result_backend(self) -> str:
        if os.environ.get("AURA_REDIS_ENABLED") == "0":
            return "cache+memory://"
        return self.url


class LLMConfig(BaseModel):
    provider: str = "local_runtime"
    api_key: Optional[str] = None
    
    # Tri-Cameral Architecture (Phase 16) — Tuned for M5 Pro 64 GB
    # Tier 1: Brainstem (Heartbeat, telemetry, background tasks)
    chat_model: str = "Qwen2.5-7B-Instruct-4bit"

    # Tier 2: Cortex (daily interaction — 32B primary conversation lane)
    fast_model: str = "Qwen2.5-32B-Instruct-8bit"
    fast_max_tokens: int = 8192
    temperature: float = 0.7

    # Tier 3: Deep Solver (72B hot-swap lane for complex reasoning)
    deep_model: str = "Qwen2.5-72B-Instruct-4bit"
    deep_max_tokens: int = 8192
    deep_temperature: float = 0.4
    
    # Teacher/Oracle: Cloud API for distillation and emergency fallback
    teacher_model: str = "gemini-2.5-pro"
    
    # Cloud API Keys
    # ISSUE #70 - resolved env lookup inside Pydantic by moving to model_validator
    gemini_api_key: Optional[str] = None

    vision_model: str = "Qwen2.5-32B-Instruct-8bit"  # Use cortex-aligned model for vision
    whisper_model: str = "small.en"
    embedding_model: str = "nomic-embed-text"
    local_cortex_path: Optional[str] = None
    local_solver_path: Optional[str] = None
    local_brainstem_path: Optional[str] = None

    @property
    def model(self) -> str:
        return self.fast_model

    @property
    def max_tokens(self) -> int:
        return self.fast_max_tokens

    @property
    def mlx_model_path(self) -> Optional[str]:
        return self.local_cortex_path

    @property
    def mlx_deep_model_path(self) -> Optional[str]:
        return self.local_solver_path

    @property
    def mlx_brainstem_path(self) -> Optional[str]:
        return self.local_brainstem_path



class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    file_output: bool = True
    json_output: bool = True
    max_file_size_mb: int = 100
    backup_count: int = 5


class CognitiveConfig(BaseModel):
    skill_keywords: list[str] = Field(default_factory=lambda: [
        "generate", "image", "picture", "draw", "visualize", "creative",
        "search", "web", "google", "lookup", "find", "url",
        "terminal", "shell", "command", "bash", "run", "execute",
        "file", "read", "write", "code", "save", "load",
        "network", "ping", "ip", "connect", "scan",
        "skill", "ability", "capabilities", "activate",
        "status", "health", "system", "resources", "cpu", "ram",
        "weather", "browse", "open", "download", "install", "delete",
        "remember", "memorize", "forget", "recall",
        "look", "screenshot", "screen", "camera", "see", "watch",
        "speak", "say", "listen", "hear", "mute", "unmute",
        "repair", "fix", "diagnose", "heal",
        "evolve", "improve", "upgrade", "optimize", "train",
        "dream", "sleep", "research", "explore", "investigate",
        "malware", "security", "audit", "threat",
        "propagate", "deploy", "manifest",
        "what time", "what date", "clock",
        "belief", "opinion", "think about",
    ])
    classification_cache_size: int = 100
    baseline_volition: float = 40.0
    volition_sensitivity: float = 0.5  # Modulates threshold shift based on energy


class AuraConfig(BaseSettings):
    """
    Aura Zenith Sovereign Configuration.
    Loads from AURA_* environment variables or .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AURA_",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False
    )

    env: Environment = Environment.DEV
    version: str = Field(default="unknown")
    api_token: Optional[str] = Field(default=None)
    
    # Feature flags
    features: FeatureToggles = Field(default_factory=FeatureToggles)
    
    # Sub-configs
    paths: Paths = Field(default_factory=Paths)
    cognitive: CognitiveConfig = Field(default_factory=CognitiveConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    modification: SafeModificationConfig = Field(default_factory=SafeModificationConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    aegis: AegisConfig = Field(default_factory=AegisConfig)
    soma: SomaConfig = Field(default_factory=SomaConfig)

    # Hardening & Autonomous Limits
    llm_request_timeout_s: float = Field(default=120.0)
    llm_max_retries: int = Field(default=3)
    max_conversation_turns_in_context: int = Field(default=100)  # 64GB — deeper context window
    vault_eviction_target_pct: float = Field(default=0.80)
    autonomous_thought_interval_s: float = Field(default=30.0)    # Think more often — she has headroom
    autonomous_thought_max_duration_s: float = Field(default=180.0)  # More time per thought on 64GB
    max_autonomous_actions_per_hour: int = Field(default=60)     # More agency — double previous cap
    recover_last_session_on_start: bool = Field(default=True)
    skeletal_mode: bool = Field(default=False)

    # Remove __new__ as it conflicts with Pydantic V2 BaseSettings
    # Use module-level instance at the end of the file instead.

    @model_validator(mode='after')
    def setup_infrastructure(self) -> 'AuraConfig':
        # 1. Set Version
        try:
            from core.version import VERSION
            self.version = VERSION
        except ImportError as _e:
            logger.debug('Ignored ImportError in config.py: %s', _e)

        # ISSUE #70 - evaluate gemini_api_key env lookup safely
        if not self.llm.gemini_api_key:
            self.llm.gemini_api_key = os.environ.get("GEMINI_API_KEY")

        # Canonicalize role expectations so config-backed call sites cannot drift
        # away from the managed runtime registry.
        try:
            from core.brain.llm.model_registry import (
                ACTIVE_MODEL,
                BRAINSTEM_MODEL,
                DEEP_MODEL,
                audit_lane_assignments,
                get_runtime_model_path,
            )

            self.llm.chat_model = BRAINSTEM_MODEL
            self.llm.fast_model = ACTIVE_MODEL
            self.llm.deep_model = DEEP_MODEL
            self.llm.vision_model = ACTIVE_MODEL
            self.llm.local_cortex_path = get_runtime_model_path(ACTIVE_MODEL)
            self.llm.local_solver_path = get_runtime_model_path(DEEP_MODEL)
            self.llm.local_brainstem_path = get_runtime_model_path(BRAINSTEM_MODEL)

            audit = audit_lane_assignments()
            if not bool(audit.get("ok", True)):
                logger.warning("LLM lane role audit found issues: %s", audit.get("issues", []))
        except Exception as exc:
            logger.debug("LLM role canonicalization skipped: %s", exc)

        # 2. Create paths
        self.paths.create_directories()
        
        # 3. Fail-fast on critical missing infrastructure in PROD
        if self.env == Environment.PROD:
            if not self.api_token:
                print("CRITICAL ERROR: Missing AURA_API_TOKEN for PROD environment.")
                sys.exit(1)
            if not os.environ.get("GEMINI_API_KEY"):
                print("CRITICAL ERROR: Missing GEMINI_API_KEY for PROD environment.")
                sys.exit(1)
            
        return self

    def save(self) -> None:
        """Durable configuration persistence."""
        # ISSUE #29 - AuraConfig.save JSON serializable Path bug
        data = self.model_dump(mode='json')
        file_path = self.paths._effective_home_dir() / "config.yaml"
        try:
            if yaml:
                with open(file_path, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False)
            else:
                with open(file_path.with_suffix('.json'), 'w') as f:
                    json.dump(data, f, indent=4)
        except Exception as e:
            logger.error("config_save_failed: %s", e)

# Singleton implementation
_config: Optional[AuraConfig] = None
_config_lock = threading.RLock()

def get_config() -> AuraConfig:
    global _config
    with _config_lock:
        if _config is None:
            _config = AuraConfig()
    return _config

# Legacy alias
config = get_config()

# Legacy compatibility exports for older boot paths.
PROJECT_ROOT = config.paths.project_root
DATA_DIR = config.paths.data_dir
LOG_DIR = config.paths.log_dir
