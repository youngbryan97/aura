import asyncio
from core.skills.mcp_client import MCPClientSkill, MCPInput

async def test():
    skill = MCPClientSkill()
    
    print("Testing MCP Action: Discover...")
    discover_params = MCPInput(
        server_command="npx",
        server_args=["-y", "@modelcontextprotocol/server-memory"],
        action="discover"
    )
    res_discover = await skill.safe_execute(discover_params)
    print("Discover result:")
    print(res_discover)

    if not res_discover.get("ok"):
        print("Discover failed. Stopping.")
        return

    print("\nTesting MCP Action: Execute...")
    execute_params = MCPInput(
        server_command="npx",
        server_args=["-y", "@modelcontextprotocol/server-memory"],
        action="execute",
        tool_name="create_entities",
        tool_args={
            "entities": [
                {
                    "name": "Aura",
                    "entityType": "AI Agent",
                    "observations": ["Built by Bryan", "Loves MCP"]
                }
            ]
        }
    )
    res_execute = await skill.safe_execute(execute_params)
    print("Execute result:")
    print(res_execute)

if __name__ == "__main__":
    asyncio.run(test())
