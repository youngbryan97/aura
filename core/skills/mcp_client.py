"""core/skills/mcp_client.py

Model Context Protocol (MCP) Client Skill.

Aura connects to MCP servers over stdio, discovers their tools, and calls
them. This is the widest reach in the system after the terminal: one MCP
server is a database, another is an issue tracker, another is a design tool,
and every one of them arrives as a program Aura is asked to launch and a
tool name Aura is asked to invoke.

Two things were wrong with the first version, and both matter more here than
almost anywhere else in the codebase:

  1. It spawned a caller-supplied `server_command` through `stdio_client`,
     which does its own process creation and therefore never touched
     `subprocess_gateway`. Every shutdown check, privilege check, and
     desktop-safety check the gateway performs was simply not on this path.
  2. It declared `requires_approval = False` and called no authorization at
     all — so "run this arbitrary program" was the one skill in the system
     that asked nobody.

Both are now routed through `core.security.execution_authority`, the single
gate in front of every surface that runs a caller-supplied program. The
server launch is authorized as `mcp_server`; each tool invocation is
authorized separately as `mcp_tool`, because "you may start this connector"
and "you may call `delete_issue` on it" are different questions and a
connector that was safe to start is not thereby safe to use for anything it
exposes.

Discovery is deliberately NOT free: discovery still starts the server
process, so it carries the same spawn authorization. Only the per-tool
question is skipped for discovery, since listing tools invokes none of them.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.capabilities.mcp_connectors import (
    available_connectors,
    connector_env,
    describe_reach,
    missing_env,
    resolve_connector,
)
from core.runtime.errors import record_degradation
from core.security.execution_authority import (
    KIND_MCP_SERVER,
    KIND_MCP_TOOL,
    authorize_execution,
    release_execution,
)
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.MCP")


class MCPInput(BaseModel):
    action: str = Field(
        ...,
        description=(
            "'list_connectors' to see what is reachable, 'discover' to list a "
            "server's tools, 'execute' to run one"
        ),
    )
    connector: Optional[str] = Field(
        None,
        description=(
            "Name of a configured connector (preferred). Resolves to the "
            "command the owner configured; use 'list_connectors' to see them."
        ),
    )
    server_command: Optional[str] = Field(
        None,
        description="Explicit command to start a server, when no connector is configured.",
    )
    server_args: List[str] = Field(default_factory=list, description="Arguments for the server command")
    tool_name: Optional[str] = Field(None, description="The name of the tool to execute")
    tool_args: Optional[Dict[str, Any]] = Field(None, description="Arguments for the tool")


class MCPClientSkill(BaseSkill):
    name = "mcp_client"
    description = "Connects to Model Context Protocol (MCP) servers to execute external foundation models and tools."
    input_model = MCPInput
    timeout_seconds = 120.0  # External MCP execution can take time (e.g., Chronos inference)
    metabolic_cost = 2
    # Launching a caller-named program and calling arbitrary tools on it is a
    # consequential action. The authorization below is the real gate; this
    # flag is the declaration that matches it.
    requires_approval = True

    async def execute(self, params: MCPInput, context: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(params, dict):
            try:
                params = MCPInput(**params)
            except (TypeError, ValueError) as exc:
                record_degradation("mcp_client", exc)
                return {"ok": False, "error": f"Invalid input: {exc}"}

        if params.action not in ("list_connectors", "discover", "execute"):
            return {"ok": False, "error": f"Unknown action: {params.action}"}

        # Answering "what can you reach?" must not require launching
        # anything, and must be truthful when the answer is nothing.
        if params.action == "list_connectors":
            reach = describe_reach()
            if not reach["configured"]:
                reach["summary"] = (
                    "No MCP connectors are configured on this machine. Adding a "
                    "server to ~/.claude.json, Claude Desktop's config, or "
                    f"{reach['registry_path']} makes it reachable immediately."
                )
            else:
                reach["summary"] = "Reachable connectors: " + "; ".join(
                    c.describe() for c in available_connectors()
                )
            reach["ok"] = True
            return reach

        if params.action == "execute" and not params.tool_name:
            return {"ok": False, "error": "tool_name is required for execute action"}

        # Resolve the connector by name before anything else. Requiring a
        # correct command line was what made this skill unreachable: the
        # model had to guess `npx -y @scope/server` and be exactly right.
        env: Dict[str, str] = {}
        if params.connector:
            connector = resolve_connector(params.connector)
            if connector is None:
                known = [c.name for c in available_connectors()]
                return {
                    "ok": False,
                    "error": (
                        f"No MCP connector named '{params.connector}'. "
                        + (
                            f"Configured connectors: {', '.join(known)}."
                            if known
                            else "No connectors are configured on this machine."
                        )
                    ),
                    "connectors": known,
                }
            absent = missing_env(connector)
            if absent:
                # Fail with the real reason. Launching without the
                # credentials produces an opaque auth error from the server
                # and a wrong diagnosis at Aura's end.
                return {
                    "ok": False,
                    "error": (
                        f"Connector '{connector.name}' needs these environment "
                        f"variables, which are not set: {', '.join(absent)}."
                    ),
                }
            server_command = connector.command
            server_args = list(connector.args)
            env = connector_env(connector)
        elif params.server_command:
            server_command = params.server_command
            server_args = [str(a) for a in params.server_args]
        else:
            known = [c.name for c in available_connectors()]
            return {
                "ok": False,
                "error": (
                    "Specify a 'connector' name or an explicit 'server_command'. "
                    + (
                        f"Configured connectors: {', '.join(known)}."
                        if known
                        else "No connectors are configured on this machine; use "
                        "action='list_connectors' for where to add one."
                    )
                ),
                "connectors": known,
            }

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            return {
                "ok": False,
                "error": "The 'mcp' Python package is not installed. Please run: pip install mcp"
            }

        argv = [server_command, *server_args]

        # Authorize the process launch. `stdio_client` creates the process
        # itself, so this is the only place the launch can be refused —
        # there is no subprocess_gateway check downstream to fall back on.
        spawn_verdict = await authorize_execution(
            KIND_MCP_SERVER,
            argv,
            source="tool_execution:mcp_client.server",
            extra={"mcp_action": params.action},
        )
        if not spawn_verdict.approved:
            return spawn_verdict.as_error()

        tool_verdict = None
        if params.action == "execute":
            # A second, separate decision. Starting a connector and invoking
            # a named capability on it are different asks, and collapsing
            # them would mean one approval to launch bought every tool the
            # server happens to expose.
            tool_verdict = await authorize_execution(
                KIND_MCP_TOOL,
                f"{server_command}::{params.tool_name}",
                source="tool_execution:mcp_client.tool",
                extra={
                    "tool_name": params.tool_name,
                    "tool_args": dict(params.tool_args or {}),
                    "server": server_command,
                },
            )
            if not tool_verdict.approved:
                release_execution(
                    spawn_verdict,
                    source="mcp_client.server",
                    success=False,
                    error="tool authorization refused",
                )
                return tool_verdict.as_error()

        server_params = StdioServerParameters(
            command=server_command,
            args=server_args,
            # Only the keys the connector declared, only when actually set.
            # Credentials come from the owner's environment, never from
            # Aura's context or config.
            env={**os.environ, **env} if env else None,
        )

        succeeded = False
        failure = ""
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Handshake and capability negotiation
                    await session.initialize()

                    if params.action == "discover":
                        tools_result = await session.list_tools()
                        succeeded = True
                        return {
                            "ok": True,
                            "summary": f"Discovered {len(tools_result.tools)} tools from MCP server",
                            "tools": [t.model_dump() for t in tools_result.tools],
                            "governance": spawn_verdict.receipt(),
                        }

                    logger.info(
                        "Executing MCP Tool '%s' on %s",
                        params.tool_name,
                        server_command,
                    )
                    exec_result = await session.call_tool(
                        params.tool_name,
                        arguments=params.tool_args or {},
                    )
                    succeeded = True
                    return {
                        "ok": True,
                        "summary": f"Executed MCP Tool '{params.tool_name}'",
                        "result": exec_result.model_dump(),
                        "governance": {
                            "server": spawn_verdict.receipt(),
                            "tool": tool_verdict.receipt() if tool_verdict else None,
                        },
                    }

        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as e:
            failure = str(e)
            record_degradation("mcp_client", e)
            logger.error("MCP Execution failed: %s", str(e), exc_info=True)
            return {"ok": False, "error": f"MCP Execution Error: {str(e)}"}
        finally:
            # Close both grants regardless of how the body exited. A token
            # left open after the server process is gone is a standing grant
            # for a connector that no longer exists.
            if tool_verdict is not None:
                release_execution(
                    tool_verdict,
                    source="mcp_client.tool",
                    success=succeeded,
                    error=failure,
                )
            release_execution(
                spawn_verdict,
                source="mcp_client.server",
                success=succeeded,
                error=failure,
            )
