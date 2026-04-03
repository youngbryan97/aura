import asyncio
from typing import Dict, Any

from core.brain.llm.mlx_client import get_mlx_client

async def main():
    print("Initializing MLX Client...")
    
    # We load the config and initialize the client
    client = get_mlx_client()
    print("Warming up...")
    await client.warm_up()
    
    print("\nSending query to 32B Brain...")
    success, text, meta = await client.generate_text_async(
        prompt="REPLY EXACTLY WITH 'OK_TEST_32B'. NOTHING ELSE.", 
        max_tokens=20
    )
    
    print("\n-------------------------")
    print(f"Success: {success}")
    print(f"Meta: {meta}")
    print(f"Response: {text}")
    print("-------------------------")

if __name__ == "__main__":
    asyncio.run(main())
