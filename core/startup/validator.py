"""
Pre-flight startup validation.
Checks every hard dependency before accepting any user input.
Fails fast with actionable error messages.
"""
from core.runtime.errors import record_degradation
import asyncio
import importlib
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger("Aura.StartupValidator")


@dataclass
class ValidationResult:
    name: str
    passed: bool
    message: str
    severity: str = "error"   # "error" (blocks start) | "warn" (logs only)
    fix_hint: str = ""


class StartupValidator:
    """
    Run before any subsystem initializes.
    All CRITICAL checks must pass or the process exits with a clear message.
    """

    def __init__(self):
        self._checks: List[Callable] = []

    def check(self, fn: Callable):
        """Decorator: register a validation check."""
        self._checks.append(fn)
        return fn

    async def run_all(self) -> bool:
        """
        Run every registered check.
        Returns True if all CRITICAL checks passed.
        Prints a formatted report regardless.
        """
        results: List[ValidationResult] = []

        for check_fn in self._checks:
            try:
                if asyncio.iscoroutinefunction(check_fn):
                    result = await check_fn()
                else:
                    result = check_fn()
                if isinstance(result, list):
                    results.extend(result)
                elif result:
                    results.append(result)
            except Exception as e:
                record_degradation('validator', e)
                results.append(ValidationResult(
                    name=check_fn.__name__,
                    passed=False,
                    message=f"Check raised exception: {e}",
                    severity="error",
                ))

        # Print report
        divider = "═" * 60
        logger.info("\n" + divider)
        logger.info("  AURA STARTUP VALIDATION REPORT")
        logger.info(divider)

        critical_failures = 0
        warnings = 0

        for r in results:
            icon = "✅" if r.passed else ("⚠️ " if r.severity == "warn" else "❌")
            logger.info(f"  {icon}  {r.name}")
            if not r.passed:
                logger.info(f"       → {r.message}")
                if r.fix_hint:
                    logger.info(f"       💡 Fix: {r.fix_hint}")
                if r.severity == "error":
                    critical_failures += 1
                else:
                    warnings += 1

        logger.info(divider)
        if critical_failures == 0:
            logger.info(f"  ✅ All checks passed ({warnings} warnings)")
        else:
            logger.error(f"  ❌ {critical_failures} critical failure(s) — Aura cannot start")
        logger.info(divider + "\n")

        return critical_failures == 0


# ── Registered Checks ────────────────────────────────────────────────────────

validator = StartupValidator()


@validator.check
def check_python_version() -> ValidationResult:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 11)
    return ValidationResult(
        name="Python >= 3.11",
        passed=ok,
        message=f"Found Python {major}.{minor}. Aura requires 3.11+.",
        fix_hint="Install Python 3.11 or later: https://python.org",
    )


@validator.check
def check_required_packages() -> List[ValidationResult]:
    required = [
        ("numpy",           "pip install numpy"),
        ("pydantic",        "pip install pydantic>=2.0"),
        ("pydantic_settings","pip install pydantic-settings"),
        ("aiohttp",         "pip install aiohttp"),
        ("aiosqlite",       "pip install aiosqlite"),
    ]
    results = []
    for pkg, fix in required:
        try:
            importlib.import_module(pkg.replace("-", "_"))
            results.append(ValidationResult(name=f"Package: {pkg}", passed=True, message=""))
        except ImportError:
            results.append(ValidationResult(
                name=f"Package: {pkg}",
                passed=False,
                message=f"Missing required package: {pkg}",
                severity="error",
                fix_hint=fix,
            ))
    return results


@validator.check
def check_optional_packages() -> List[ValidationResult]:
    from core.brain.llm.model_registry import find_llama_server_bin, get_local_backend

    backend = get_local_backend()
    optional = [
        ("pyaudio",      "pip install pyaudio",           "Voice input unavailable"),
        ("webrtcvad",    "pip install webrtcvad-wheels",  "Voice activity detection unavailable"),
        ("cryptography", "pip install cryptography",      "Ed25519 signatures unavailable (using HMAC fallback)"),
        ("astor",        "pip install astor",             "Fictional Engine Synthesis unavailable"),
        ("mlx_whisper",  "pip install mlx-whisper",       "Voice transcription unavailable"),
        ("yaml",         "pip install PyYAML",             "YAML config loading unavailable"),
    ]
    if backend == "mlx":
        optional.append(("mlx", "pip install mlx", "Local MLX inference unavailable"))
    results = []
    for pkg, fix, impact in optional:
        try:
            importlib.import_module(pkg)
            results.append(ValidationResult(name=f"Optional: {pkg}", passed=True, message=""))
        except ImportError:
            results.append(ValidationResult(
                name=f"Optional: {pkg}",
                passed=False,
                message=f"{impact}",
                severity="warn",
                fix_hint=fix,
            ))
    if backend == "llama_cpp":
        binary = find_llama_server_bin()
        if binary:
            results.append(ValidationResult(name="Optional: llama-server", passed=True, message=""))
        else:
            results.append(
                ValidationResult(
                    name="Optional: llama-server",
                    passed=False,
                    message="Managed local runtime unavailable",
                    severity="warn",
                    fix_hint="Install llama.cpp and ensure llama-server is on PATH",
                )
            )
    return results


@validator.check
def check_data_directories() -> List[ValidationResult]:
    # Dynamic import to avoid circularity during early boot
    from core.config import config
    results = []
    dirs = [
        config.paths.home_dir,
        config.paths.data_dir,
        config.paths.log_dir,
    ]
    for d in dirs:
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
            results.append(ValidationResult(name=f"Directory: {d}", passed=True, message=""))
        except PermissionError as e:
            results.append(ValidationResult(
                name=f"Directory: {d}",
                passed=False,
                message=f"Cannot create directory: {e}",
                severity="error",
                fix_hint=f"Check permissions on {Path(d).parent}",
            ))
    return results


@validator.check
def check_audio_device() -> ValidationResult:
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        count = p.get_device_count()
        p.terminate()
        if count == 0:
            return ValidationResult(
                name="Audio input device",
                passed=False,
                message="No audio devices found. Voice input will be unavailable.",
                severity="warn",
                fix_hint="Connect a microphone or check macOS Privacy > Microphone settings",
            )
        return ValidationResult(
            name="Audio input device",
            passed=True,
            message=f"{count} audio device(s) found",
        )
    except Exception as e:
        record_degradation('validator', e)
        return ValidationResult(
            name="Audio input device",
            passed=False,
            message=str(e),
            severity="warn",
        )


def get_validator() -> StartupValidator:
    """Convenience factory for the startup validator."""
    return validator
