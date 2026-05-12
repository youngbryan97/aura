import asyncio
import json
import os
import httpx

async def test_key():
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY before running this archived smoke test.")
    model = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "Hello! Reply with 'Key works!'"}]}]
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, timeout=10)
        print(f"Status: {response.status_code}")
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text)

asyncio.run(test_key())
