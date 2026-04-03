import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.append("/Users/bryan/Desktop/aura")

from core.brain.llm.mlx_client import get_mlx_client
from core.brain.llm.model_registry import get_model_path

async def main():
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("Diag.MLX")
    
    model_path = get_model_path()
    logger.info(f"Target model path: {model_path}")
    
    client = get_mlx_client(model_path)
    logger.info("Initializing MLX client...")
    
    try:
        # Step 1: Warm up (spawns worker)
        await client.warm_up()
        logger.info("Warm up successful.")
        
        # Step 2: Simple generation
        logger.info("Starting test generation...")
        success, text, meta = await client.generate_text_async("Hello, identify yourself.", temp=0.0)
        
        if success:
            logger.info(f"✅ Success! Response: {text}")
            logger.info(f"Meta: {meta}")
        else:
            logger.error(f"❌ Generation failed: {text}")
            
    except Exception as e:
        logger.error(f"💥 Diagnostic crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
