"""core/skills/mcp_client.py

Model Context Protocol (MCP) Client Skill.
Allows Aura to dynamically connect to MCP servers (like specialized scientific 
models or enterprise data connectors) using standard I/O, discover their tools, 
and execute them natively.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill
from core.runtime.errors import record_degradation

logger = logging.getLogger("Skills.MCP")


class MCPInput(BaseModel):
    server_command: str = Field(..., description="Command to start the MCP server (e.g. 'npx', 'python', 'docker')")
    server_args: List[str] = Field(default_factory=list, description="Arguments for the server command")
    action: str = Field(..., description="Action to perform: 'discover' to list tools, 'execute' to run a tool")
    tool_name: Optional[str] = Field(None, description="The name of the tool to execute")
    tool_args: Optional[Dict[str, Any]] = Field(None, description="Arguments for the tool")


class MCPClientSkill(BaseSkill):
    name = "mcp_client"
    description = "Connects to Model Context Protocol (MCP) servers to execute external foundation models and tools."
    input_model = MCPInput
    timeout_seconds = 120.0  # External MCP execution can take time (e.g., Chronos inference)
    metabolic_cost = 2
    requires_approval = False

    async def execute(self, params: MCPInput, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            return {
                "ok": False,
                "error": "The 'mcp' Python package is not installed. Please run: pip install mcp"
            }

        server_params = StdioServerParameters(
            command=params.server_command,
            args=params.server_args
        )

        try:
            # We use standard IO to connect to the MCP server natively.
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Handshake and capability negotiation
                    await session.initialize()
                    
                    if params.action == "discover":
                        tools_result = await session.list_tools()
                        return {
                            "ok": True,
                            "summary": f"Discovered {len(tools_result.tools)} tools from MCP server",
                            "tools": [t.model_dump() for t in tools_result.tools]
                        }
                    
                    elif params.action == "execute":
                        if not params.tool_name:
                            return {"ok": False, "error": "tool_name is required for execute action"}
                            
                        logger.info(f"Executing MCP Tool '{params.tool_name}' on {params.server_command}")
                        exec_result = await session.call_tool(
                            params.tool_name, 
                            arguments=params.tool_args or {}
                        )
                        
                        return {
                            "ok": True,
                            "summary": f"Executed MCP Tool '{params.tool_name}'",
                            "result": exec_result.model_dump()
                        }
                    
                    else:
                        return {"ok": False, "error": f"Unknown action: {params.action}"}
                        
        except Exception as e:
            record_degradation("mcp_client", e)
            logger.error("MCP Execution failed: %s", str(e), exc_info=True)
            return {"ok": False, "error": f"MCP Execution Error: {str(e)}"}
