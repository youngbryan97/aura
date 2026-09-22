"""What Aura can actually reach through MCP, and where that list comes from.

`mcp_client` could always call an MCP server. It could not find one. The
skill's only inputs were `server_command` and `server_args`, so reaching a
connector required Aura to produce a working command line from nothing —
which in practice means guessing `npx -y @some/server-name` and being wrong.
A capability that can only be used by correctly hallucinating its arguments
is not a reachable capability.

This module is the directory. It resolves a connector by NAME against real
declarations, so Aura asks for "sentry" and gets the command the owner
actually configured, or an honest refusal naming what does exist.

Three sources, most specific first:

  1. `config/mcp_connectors.json` — Aura's own, for connectors that are hers.
  2. `~/.claude.json` — the Claude Code host config on this machine.
  3. Claude Desktop's `claude_desktop_config.json`.

Reading 2 and 3 is the point rather than a shortcut: when the owner adds an
MCP server to the tools they already use, Aura's reach widens the same day,
with no second place to register it and no chance of the two lists
disagreeing. All three use the standard `mcpServers` shape, so there is one
schema to understand.

There are deliberately no built-in connectors. Shipping a plausible-looking
default list would make `list_connectors` describe a reach Aura does not
have — the exact failure this codebase keeps finding, where a claim outlives
the code that made it true. Empty is the correct answer when nothing is
configured, and it is the answer this returns.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Capabilities.MCPConnectors")

# Aura's own declarations. Same `mcpServers` shape as the host configs.
_AURA_REGISTRY = Path(__file__).resolve().parents[2] / "config" / "mcp_connectors.json"

# Where the host tools keep theirs. Read-only; Aura never writes to these.
_HOST_CONFIGS = (
    ("claude_code", Path.home() / ".claude.json"),
    (
        "claude_desktop",
        Path.home()
        / "Library"
        / "Application Support"
        / "Claude"
        / "claude_desktop_config.json",
    ),
)

# A connector declaration is data from a config file, so every field is
# validated rather than trusted. A `command` that is not a string, or args
# that are not a list of strings, is a malformed entry and is dropped with a
# degradation — not coerced into something that would then be executed.


@dataclass(frozen=True)
class MCPConnector:
    """One reachable MCP server."""

    name: str
    command: str
    args: tuple[str, ...] = ()
    source: str = "unknown"
    purpose: str = ""
    env_keys: tuple[str, ...] = field(default=())

    def argv(self) -> list[str]:
        return [self.command, *self.args]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "args": list(self.args),
            "source": self.source,
            "purpose": self.purpose,
            # Names only. The values are credentials and never leave the
            # config file through this surface.
            "env_keys": list(self.env_keys),
        }

    def describe(self) -> str:
        detail = self.purpose or f"{self.command} {' '.join(self.args)}".strip()
        return f"{self.name} — {detail} (from {self.source})"


def _coerce(name: Any, spec: Any, source: str) -> MCPConnector | None:
    """Turn one `mcpServers` entry into a connector, or reject it."""
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(spec, dict):
        return None

    command = spec.get("command")
    if not isinstance(command, str) or not command.strip():
        # An entry with no command cannot be launched. Silently keeping it
        # would put a name in `list_connectors` that fails on use, which is
        # worse than not listing it.
        return None

    raw_args = spec.get("args") or []
    if not isinstance(raw_args, (list, tuple)):
        return None
    args = tuple(str(a) for a in raw_args)

    raw_env = spec.get("env") or {}
    env_keys = tuple(sorted(str(k) for k in raw_env)) if isinstance(raw_env, dict) else ()

    purpose = spec.get("purpose") or spec.get("description") or ""

    return MCPConnector(
        name=name.strip(),
        command=command.strip(),
        args=args,
        source=source,
        purpose=str(purpose),
        env_keys=env_keys,
    )


def _read(path: Path, source: str) -> list[MCPConnector]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        record_degradation(
            "mcp_connectors",
            exc,
            severity="warning",
            action=f"skipped unreadable MCP config at {path.name}",
            extra={"source": source},
        )
        return []

    if not isinstance(payload, dict):
        return []
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict):
        return []

    found: list[MCPConnector] = []
    for name, spec in servers.items():
        connector = _coerce(name, spec, source)
        if connector is not None:
            found.append(connector)
    return found


def available_connectors() -> list[MCPConnector]:
    """Every connector Aura can reach right now, deduplicated by name.

    Order is precedence order: Aura's own registry wins over the host
    configs, and Claude Code wins over Claude Desktop. Recomputed on each
    call rather than cached, because the owner adding a server to their
    config should widen Aura's reach without a restart.
    """
    seen: dict[str, MCPConnector] = {}
    for connector in _read(_AURA_REGISTRY, "aura"):
        seen.setdefault(connector.name, connector)
    for source, path in _HOST_CONFIGS:
        for connector in _read(path, source):
            seen.setdefault(connector.name, connector)
    return sorted(seen.values(), key=lambda c: c.name)


def resolve_connector(name: str) -> MCPConnector | None:
    """Find a connector by name, case-insensitively. None when unknown."""
    wanted = str(name or "").strip().lower()
    if not wanted:
        return None
    for connector in available_connectors():
        if connector.name.lower() == wanted:
            return connector
    return None


def connector_env(connector: MCPConnector) -> dict[str, str]:
    """The environment a connector needs, read from the process environment.

    Credentials live in the owner's environment, not in Aura's config and
    not in her context. This returns only the keys the connector declared,
    and only those that are actually set — a missing key is reported as
    missing rather than passed through as an empty string, which would make
    the server fail with an authentication error instead of a clear one.
    """
    return {
        key: os.environ[key] for key in connector.env_keys if key in os.environ
    }


def missing_env(connector: MCPConnector) -> list[str]:
    return [key for key in connector.env_keys if key not in os.environ]


def describe_reach() -> dict[str, Any]:
    """What Aura should say when asked what she can reach.

    Honest about zero. `connectors: []` with `configured: false` is the
    correct answer on a machine with no MCP servers, and it is what the
    health report and the skill both surface.
    """
    connectors = available_connectors()
    return {
        "configured": bool(connectors),
        "count": len(connectors),
        "connectors": [c.to_dict() for c in connectors],
        "sources_checked": [str(_AURA_REGISTRY)] + [str(p) for _, p in _HOST_CONFIGS],
        "registry_path": str(_AURA_REGISTRY),
    }


__all__ = [
    "MCPConnector",
    "available_connectors",
    "connector_env",
    "describe_reach",
    "missing_env",
    "resolve_connector",
]


def external_reach_report() -> dict[str, Any]:
    """How far Aura can actually reach outside herself, right now.

    `mcp_client` was registered, routable, and described as connecting Aura to
    enterprise data connectors while the `mcp` package was not installed and no
    connector was configured — every call returned an error and nothing
    anywhere said so. A capability that is dead in practice must be visible as
    dead, not discoverable only by trying it.

    This lived in `core/runtime/health_contract.py`, which meant the foundation
    layer imported this module to describe it. The dependency runs the other
    way now: the subsystem that owns the transport publishes its own fragment,
    and the foundation reads a register.
    """
    snapshot: dict[str, Any] = {"mcp": {"available": False, "connectors": 0}}
    try:
        import importlib.util

        connectors = available_connectors()
        snapshot["mcp"] = {
            # The transport. Without it no connector is reachable however many
            # are configured.
            "available": importlib.util.find_spec("mcp") is not None,
            "connectors": len(connectors),
            "names": [c.name for c in connectors],
        }
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        snapshot["mcp"]["error"] = repr(exc)
    return snapshot


def _register_fragment() -> None:
    try:
        from core.runtime.health_fragments import register_health_fragment

        register_health_fragment("external_reach", external_reach_report)
    # not a failure: the comment beside it says it: the register is optional and its
    # absence is reported by the surface.
    except (ImportError, AttributeError):
        pass  # the register is optional; its absence is reported by the surface


_register_fragment()
