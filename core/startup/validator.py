"""
Pre-flight startup validation.
Checks every hard dependency before accepting any user input.
Fails fast with actionable error messages.
"""
import importlib
import importlib.util
import inspect
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.runtime.errors import record_degradation

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
        self._checks: list[Callable] = []

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
        results: list[ValidationResult] = []

        for check_fn in self._checks:
            try:
                if inspect.iscoroutinefunction(check_fn):
                    result = await check_fn()
                else:
                    result = check_fn()
                if isinstance(result, list):
                    results.extend(result)
                elif result:
                    results.append(result)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation(
                    "validator",
                    e,
                    severity="warning",
                    action="marked startup validation check failed and continued remaining checks",
                    extra={"check": check_fn.__name__},
                )
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
            logger.info("  %s  %s", icon, r.name)
            if not r.passed:
                logger.info("       → %s", r.message)
                if r.fix_hint:
                    logger.info("       💡 Fix: %s", r.fix_hint)
                if r.severity == "error":
                    critical_failures += 1
                else:
                    warnings += 1

        logger.info(divider)
        if critical_failures == 0:
            logger.info("  ✅ All checks passed (%s warnings)", warnings)
        else:
            logger.error("  ❌ %s critical failure(s) — Aura cannot start", critical_failures)
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
def check_required_packages() -> list[ValidationResult]:
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
def check_optional_packages() -> list[ValidationResult]:
    from core.brain.llm.model_registry import get_local_backend

    def _module_available(module_name: str) -> bool:
        """Check optional package presence without executing native imports."""
        try:
            return importlib.util.find_spec(module_name) is not None
        except (ImportError, AttributeError, ModuleNotFoundError, ValueError) as exc:
            logger.debug(
                "the optional package could not be located (%s: %s)", type(exc).__name__, exc
            )
            return False

    backend = get_local_backend()
    optional = [
        ("sounddevice",  "pip install sounddevice",       "Voice capture unavailable"),
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
        if _module_available(pkg):
            results.append(ValidationResult(name=f"Optional: {pkg}", passed=True, message=""))
        else:
            results.append(ValidationResult(
                name=f"Optional: {pkg}",
                passed=False,
                message=f"{impact}",
                severity="warn",
                fix_hint=fix,
            ))
    return results


@validator.check
def check_data_directories() -> list[ValidationResult]:
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
        from core.senses.sensory_registry import get_capabilities

        hearing_enabled = bool(get_capabilities().hearing_enabled)
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("Audio capability probe unavailable: %s", exc)
        hearing_enabled = False

    voice_required = os.environ.get("AURA_REQUIRE_VOICE_INPUT", "0").strip().lower() in {"1", "true", "yes", "on"}
    if not hearing_enabled and not voice_required:
        return ValidationResult(
            name="Audio input device",
            passed=True,
            message="Voice input disabled by capability flags",
            severity="warn",
        )

    try:
        import sounddevice as sd

        devices = sd.query_devices()
        if isinstance(devices, dict):
            device_list = [devices]
        else:
            device_list = list(devices or [])
        input_count = sum(
            1
            for device in device_list
            if int(device.get("max_input_channels", 0) or 0) > 0
        )
        if input_count == 0:
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
            message=f"{input_count} audio input device(s) found",
        )
    except (ImportError, AttributeError, RuntimeError) as e:
        if voice_required:
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
