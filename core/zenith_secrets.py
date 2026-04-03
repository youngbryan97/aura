"""
Secrets management for Aura.
Hierarchy (highest priority first):
  1. macOS Keychain (production)
  2. Environment variable
  3. .env file (development)
  4. config.yaml (least secure — warns if used for secrets)
"""
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("Aura.Secrets")

_KEYCHAIN_SERVICE = "AuraAutonomyEngine"


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieve a secret by key. Never logs the value.
    """
    # 1. Environment variable (also catches .env loaded by pydantic-settings)
    value = os.environ.get(key)
    if value:
        return value

    # 2. macOS Keychain
    value = _keychain_get(key)
    if value:
        return value

    if default is None:
        logger.debug("Secret '%s' not found in any store.", key)
    return default


def set_secret(key: str, value: str, store: str = "keychain"):
    """
    Persist a secret. Default target is macOS Keychain.
    """
    if store == "keychain":
        success = _keychain_set(key, value)
        if success:
            logger.info("Secret '%s' stored in Keychain.", key)
            return
        logger.warning("Keychain unavailable — storing '%s' in environment only.", key)

    os.environ[key] = value


def _keychain_get(key: str) -> Optional[str]:
    """Retrieve from macOS Keychain using Security framework."""
    try:
        import subprocess
        result = subprocess.run(
            [
                "security", "find-generic-password",
                "-s", _KEYCHAIN_SERVICE,
                "-a", key,
                "-w",                    # Output password only
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as _e:
        logger.debug('Ignored Exception in secrets.py: %s', _e)
    return None


def _keychain_set(key: str, value: str) -> bool:
    """Store in macOS Keychain."""
    try:
        import subprocess
        # Try to update existing; if it fails, add new
        result = subprocess.run(
            [
                "security", "add-generic-password",
                "-s", _KEYCHAIN_SERVICE,
                "-a", key,
                "-w", value,
                "-U",                    # Update if exists
            ],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def load_dotenv(path: Optional[str] = None):
    """Load a .env file into environment variables. Dev convenience only."""
    dot_env = Path(path or ".env")
    if not dot_env.exists():
        return
    logger.info("Loading .env from %s (dev mode)", dot_env)
    with open(dot_env) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
