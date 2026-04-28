
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch
    from diffusers import AutoPipelineForText2Image, DiffusionPipeline
except ImportError:
    torch = None
    DiffusionPipeline = None

from core.config import config
from infrastructure import BaseSkill

logger = logging.getLogger("Skills.LocalMedia")

class LocalMediaGenerationSkill(BaseSkill):
    name = "local_media_generation"
    description = "Generate images locally using Stable Diffusion (Offline)."
    inputs = {
        "prompt": "Description of the image to generate.",
        "negative_prompt": "Optional. What to avoid in the image.",
        "style": "Optional style guidance."
    }
    
    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.device = "mps" if torch and torch.backends.mps.is_available() else "cpu"
        self.model_id = "runwayml/stable-diffusion-v1-5" # Faster/smaller for initial test
        # self.model_id = "stabilityai/stable-diffusion-xl-base-1.0" # Better but heavier
        
        self.output_dir = Path(config.paths.data_dir) / "generated_images"
        get_task_tracker().create_task(get_storage_gateway().create_dir(self.output_dir, cause='LocalMediaGenerationSkill.__init__'))
        
    def _load_model(self):
        """Lazy load the model to save RAM until needed."""
        if self.pipeline:
            return True
            
        if not torch:
            logger.error("Torch/Diffusers not installed.")
            return False
            
        logger.info("Loading Local Diffusion Model (%s) on %s...", self.model_id, self.device)
        try:
            # Choose appropriate pipeline class
            pipeline_cls = None
            if 'AutoPipelineForText2Image' in globals() and AutoPipelineForText2Image is not None:
                pipeline_cls = AutoPipelineForText2Image
            elif 'DiffusionPipeline' in globals() and DiffusionPipeline is not None:
                pipeline_cls = DiffusionPipeline
            else:
                logger.error("No suitable Diffusers pipeline class available.")
                return False

            # Use float16 only on CUDA devices; MPS/CPU should use float32 to avoid issues
            torch_dtype = torch.float16 if (hasattr(self, 'device') and str(self.device).startswith('cuda')) else torch.float32

            self.pipeline = pipeline_cls.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True
            )
            # Move to device (some pipelines require .to on underlying torch modules)
            try:
                self.pipeline.to(self.device)
            except Exception:
                # Some pipeline classes may not support direct .to() — ignore if so
                pass

            # Enable attention slicing for lower memory usage when supported
            try:
                if hasattr(self.pipeline, 'enable_attention_slicing'):
                    self.pipeline.enable_attention_slicing()
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)
            logger.info("✓ Local Diffusion Model Loaded.")
            return True
        except Exception as e:
            logger.error("Failed to load local model: %s", e)
            return False

    async def execute(self, goal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image locally."""
        prompt = goal.get("objective") or goal.get("params", {}).get("prompt")
        
        if not prompt:
            return {"ok": False, "error": "No prompt provided."}
            
        # 1. Load Model (Lazy)
        if not self._load_model():
            return {
                "ok": False, 
                "error": "Local AI Model failed to load. Check dependencies (torch, diffusers).",
                "message": "I tried to initialize my local imagination engine, but the neural weights are missing."
            }
            
        # 2. Generate
        logger.info("Dreaming locally: '%s'...", prompt)
        try:
            # Run in executor to avoid blocking async loop (generation takes seconds)
            import asyncio
            loop = asyncio.get_event_loop()
            
            def _generate():
                # UPGRADE: Automatic Prompt Engineering for High Fidelity
                enhanced_prompt = f"{prompt}, masterpiece, best quality, 8k, HDR, cinematic lighting, sharp focus, detailed texture"
                negative = "blur, low quality, distortion, watermark, text, ugly, bad anatomy"
                
                return self.pipeline(
                    prompt=enhanced_prompt, 
                    negative_prompt=negative,
                    num_inference_steps=40, # Increased for quality
                    guidance_scale=8.0      # Slightly higher adherence
                ).images[0]
                
            image = await loop.run_in_executor(None, _generate)
            
            # 3. Save
            timestamp = int(time.time())
            filename = f"gen_{timestamp}.png"
            filepath = self.output_dir / filename
            image.save(filepath)
            
            # URL for frontend
            # Assuming server serves /api/images or static files
            # For now, we can just give the local path or a relative URL if we set up static serving
            # Let's assume we'll serve 'data/generated_images' as '/images'
            relative_url = f"/data/generated_images/{filename}" 
            
            return {
                "ok": True,
                "url": relative_url,
                "path": str(filepath),
                "message": f"I painted this for you (locally): {relative_url}",
                "type": "image"
            }
            
        except Exception as e:
            logger.error("Local generation failed: %s", e)
            return {"ok": False, "error": f"Generation failed: {e}"}
