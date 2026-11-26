"""core/utils/privacy_hygiene.py -- Local Privacy Hygiene and Metadata Scrubbing
=============================================================================
Provides clean, audit-safe utilities to scrub credentials, system paths, IPs,
MAC addresses, and secrets from outgoing text and data structures.
"""
from __future__ import annotations
import os
import re
import platform
from ipaddress import ip_address
from pathlib import Path, PurePosixPath
from typing import Any

_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_MAC_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")
_SECRET_PATTERNS = (
    (
        re.compile(r"(?i)\b(password|passwd|secret|api[_-]?key|token)\s*[:=]\s*\S+"),
        r"\1=[SECRET_REDACTED]",
    ),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "[SECRET_REDACTED]"),
)


def _redact_ip_if_valid(match: re.Match[str]) -> str:
    value = match.group(0)
    try:
        ip_address(value)
    except ValueError:
        return value
    return "[IP_REDACTED]"


class MetadataScrubber:
    """Removes identifying local metadata from outgoing text and structures."""

    def __init__(self):
        self.real_hostname = platform.node()
        self.real_username = os.getenv("USER") or os.getenv("USERNAME") or "user"
        self.real_home = os.path.expanduser("~")
        self.real_cwd = os.getcwd()

        self.fake_hostname = "generic-system"
        self.fake_username = "user"
        self.fake_home = str(PurePosixPath("/home") / "user")
        self.fake_cwd = "/opt/application"

    def scrub_text(self, text: str) -> str:
        if not text:
            return ""
        scrubbed = text
        replacements = (
            (self.real_cwd, self.fake_cwd),
            (self.real_home, self.fake_home),
            (self.real_hostname, self.fake_hostname),
        )
        for real, fake in replacements:
            if real:
                scrubbed = scrubbed.replace(real, fake)
        if self.real_username:
            # The account's name is metadata where it is a part of a path or a login (user@host), and a person's name
            # everywhere else. LIVE 26 November 2026 the account bore the name of the man she was writing to: her
            # reply to him was held as "sensitive system metadata" for naming him, and would have gone out with his
            # name, and the part of his address that holds it, turned into "user".
            user = re.escape(self.real_username)
            scrubbed = re.sub(rf"(?<=[/\\]){user}(?=[/\\]|\b)", self.fake_username, scrubbed)
            scrubbed = re.sub(rf"(?<![\w.+-]){user}@(?={re.escape(self.fake_hostname)}\b|localhost\b)",
                              f"{self.fake_username}@", scrubbed)

        scrubbed = _IP_RE.sub(_redact_ip_if_valid, scrubbed)
        scrubbed = _MAC_RE.sub("[MAC_REDACTED]", scrubbed)
        for pattern, replacement in _SECRET_PATTERNS:
            scrubbed = pattern.sub(replacement, scrubbed)
        return scrubbed

    def scrub_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        return {str(key): self._scrub_value(value) for key, value in data.items()}

    def _scrub_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.scrub_text(value)
        if isinstance(value, Path):
            return self.scrub_file_path(str(value))
        if isinstance(value, dict):
            return self.scrub_dict(value)
        if isinstance(value, list):
            return [self._scrub_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._scrub_value(item) for item in value)
        return value

    def scrub_file_path(self, path: str) -> str:
        scrubbed = path
        if self.real_home and scrubbed.startswith(self.real_home):
            scrubbed = scrubbed.replace(self.real_home, self.fake_home, 1)
        if self.real_username:
            scrubbed = scrubbed.replace(f"/{self.real_username}/", "/user/")
        return scrubbed


class StealthMode:
    """Backward-compatible name for Aura's privacy mode controller."""

    def __init__(self):
        self.scrubber = MetadataScrubber()
        self.stealth_enabled = True

    def process_output(self, text: str) -> str:
        return self.scrubber.scrub_text(text)


_stealth_instance: StealthMode | None = None


def get_stealth_mode() -> StealthMode:
    global _stealth_instance
    if _stealth_instance is None:
        _stealth_instance = StealthMode()
    return _stealth_instance
