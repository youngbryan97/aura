import asyncio
import os
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("list_models")

async def list_models():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found")
        return

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code == 200:
            models = response.json().get("models", [])
            for m in models:
                print(f"Name: {m['name']}, Display: {m['displayName']}")
        else:
            print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    asyncio.run(list_models())

