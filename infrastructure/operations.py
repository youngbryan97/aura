"""Core operational helpers for dependency, web, and file access."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from core.runtime.errors import record_degradation

logger = logging.getLogger("Infra.Operations")

Installer = Callable[[Sequence[str]], int]
VersionLookup = Callable[[str], str]
Fetcher = Callable[[str, float, int], str]

_DEFAULT_REQUIRED_PACKAGES = (
    "beautifulsoup4",
    "cryptography",
    "numpy",
    "psutil",
    "pydantic",
    "pydantic-settings",
    "requests",
    "setproctitle",
    "sounddevice",
)
_INSTALL_RECOVERABLE_ERRORS = (
    ImportError,
    ModuleNotFoundError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_WEB_RECOVERABLE_ERRORS = (
    HTTPError,
    TimeoutError,
    UnicodeDecodeError,
    URLError,
    ValueError,
)
_FILE_RECOVERABLE_ERRORS = (
    FileNotFoundError,
    IsADirectoryError,
    OSError,
    PermissionError,
    UnicodeDecodeError,
)


@dataclass(frozen=True)
class DependencyReport:
    installed: tuple[str, ...]
    missing: tuple[str, ...]
    attempted_install: bool
    install_exit_code: int | None = None

    @property
    def ok(self) -> bool:
        return not self.missing or self.install_exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "installed": list(self.installed),
            "missing": list(self.missing),
            "attempted_install": self.attempted_install,
            "install_exit_code": self.install_exit_code,
        }


class DepOps:
    @staticmethod
    def check_and_install(
        *,
        required_packages: Sequence[str] | None = None,
        auto_install: bool | None = None,
        installer: Installer | None = None,
        version_lookup: VersionLookup | None = None,
    ) -> dict[str, Any]:
        """Check runtime dependencies and optionally install missing packages.

        Runtime installation is opt-in through ``auto_install=True`` or
        ``AURA_ALLOW_RUNTIME_PIP=1``. This keeps production boots observable
        and avoids surprise environment mutation.
        """
        required = tuple(required_packages or _DEFAULT_REQUIRED_PACKAGES)
        version_lookup = version_lookup or metadata.version
        installed, missing = _classify_dependencies(required, version_lookup)
        allow_install = auto_install if auto_install is not None else os.environ.get("AURA_ALLOW_RUNTIME_PIP") == "1"

        if not missing:
            logger.info("All dependencies satisfied.")
            return DependencyReport(installed, missing, attempted_install=False).to_dict()
        if not allow_install:
            logger.warning("Missing dependencies detected without runtime install permission: %s", ", ".join(missing))
            return DependencyReport(installed, missing, attempted_install=False).to_dict()

        package_installer = installer or _pip_install
        try:
            exit_code = package_installer(missing)
        except _INSTALL_RECOVERABLE_ERRORS as exc:
            record_degradation("dependency_operations", exc)
            logger.error("Dependency installation failed: %s", exc)
            exit_code = 1

        report = DependencyReport(installed, missing, attempted_install=True, install_exit_code=exit_code)
        if report.ok:
            logger.info("Dependency installation completed.")
        else:
            logger.error("Dependency installation returned exit code %s.", exit_code)
        return report.to_dict()


class WebOps:
    @staticmethod
    async def fetch_page_text(
        url: str,
        timeout_seconds: int | float = 10,
        *,
        max_bytes: int = 1_000_000,
        fetcher: Fetcher | None = None,
    ) -> str | None:
        """Fetch a HTTP(S) page without blocking the event loop."""
        try:
            _validate_http_url(url)
            bounded_timeout = _bounded_float(timeout_seconds, minimum=0.1, maximum=60.0)
            byte_limit = _bounded_int(max_bytes, minimum=1, maximum=10_000_000)
            return await asyncio.to_thread(fetcher or _fetch_url_text, url, bounded_timeout, byte_limit)
        except _WEB_RECOVERABLE_ERRORS as exc:
            record_degradation("web_operations", exc)
            logger.warning("Web fetch failed for %s: %s", url, exc)
            return None


class FileOps:
    @staticmethod
    def timestomp(filepath: Path, *, timestamp: float | None = None, allow: bool | None = None) -> bool:
        """Explicitly set file times only when the caller or environment permits it."""
        permitted = allow if allow is not None else os.environ.get("AURA_ALLOW_FILE_MTIME_EDIT") == "1"
        if not permitted:
            logger.warning("Refusing file mtime edit without explicit permission: %s", filepath)
            return False
        target_time = float(timestamp if timestamp is not None else time.time())
        try:
            os.utime(filepath, (target_time, target_time))
            return True
        except _FILE_RECOVERABLE_ERRORS as exc:
            record_degradation("file_operations", exc)
            logger.warning("Failed to set file mtime for %s: %s", filepath, exc)
            return False

    @staticmethod
    def load_file(path: str | Path, *, max_bytes: int = 5_000_000) -> str:
        """Load a text file with a byte cap and typed recovery."""
        target = Path(path)
        if not target.exists() or not target.is_file():
            return ""
        byte_limit = _bounded_int(max_bytes, minimum=1, maximum=100_000_000)
        try:
            with target.open("rb") as handle:
                data = handle.read(byte_limit + 1)
            if len(data) > byte_limit:
                logger.warning("File read truncated at %d bytes: %s", byte_limit, target)
                data = data[:byte_limit]
            return data.decode("utf-8")
        except _FILE_RECOVERABLE_ERRORS as exc:
            record_degradation("file_operations", exc)
            logger.warning("File load failed for %s: %s", target, exc)
            return ""


def _classify_dependencies(
    required_packages: Sequence[str],
    version_lookup: VersionLookup,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    installed: list[str] = []
    missing: list[str] = []
    for package in required_packages:
        try:
            version_lookup(package)
            installed.append(package)
        except metadata.PackageNotFoundError:
            missing.append(package)
    return tuple(installed), tuple(missing)


def _pip_install(packages: Sequence[str]) -> int:
    from pip._internal.cli.main import main as pip_main

    return int(pip_main(["install", *packages]))


def _validate_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("only absolute http(s) URLs are supported")


def _fetch_url_text(url: str, timeout_seconds: float, max_bytes: int) -> str:
    request = Request(url, headers={"User-Agent": "Aura/3.5"})
    with urlopen(request, timeout=timeout_seconds) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        payload = payload[:max_bytes]
    return payload.decode(charset, errors="replace")


def _bounded_int(value: int | str, *, minimum: int, maximum: int) -> int:
    parsed = int(value)
    return max(minimum, min(maximum, parsed))


def _bounded_float(value: int | float | str, *, minimum: float, maximum: float) -> float:
    parsed = float(value)
    return max(minimum, min(maximum, parsed))
