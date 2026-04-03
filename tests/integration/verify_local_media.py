################################################################################

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from skills.local_media_generation import LocalMediaGenerationSkill

async def test_local_generation():
    print("--- Testing Local Media Generation ---")
    
    skill = LocalMediaGenerationSkill()
    
    # Test model loading
    print("Loading model (this may take time on first run)...")
    if not skill._load_model():
        print("❌ Model failed to load. Check PyTorch/Diffusers installation.")
        return

    print("✓ Model loaded successfully.")
    
    # Test generation
    prompt = "A futuristic city with flying cars, cyberpunk style, intense colors."
    print(f"Generating image for: '{prompt}'")
    
    result = await skill.execute({"objective": prompt}, {})
    
    if result["ok"]:
        print(f"✓ Generation successful!")
        print(f"  URL: {result['url']}")
        print(f"  Path: {result['path']}")
    else:
        print(f"❌ Generation failed: {result.get('error')}")

if __name__ == "__main__":
    asyncio.run(test_local_generation())


##
