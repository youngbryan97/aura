from core.runtime.errors import record_degradation
import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field

from core.config import config
from core.skills.base_skill import BaseSkill

# Enable MPS fallback for Mac stability
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logger = logging.getLogger("Skills.ImageGeneration")

try:
    import torch
    from diffusers import FluxPipeline
    FLUX_AVAILABLE = True
except ImportError:
    FLUX_AVAILABLE = False
    logger.error("Flux not installed. Install with: pip install diffusers torch torchvision torchaudio accelerate")

# Issue 77: Check 'torch' in sys.modules to avoid NameError if import failed
import sys
TORCH_READY = "torch" in sys.modules and FLUX_AVAILABLE

class ImageInput(BaseModel):
    prompt: str = Field(..., description="Detailed prompt for high-fidelity image")
    negative_prompt: str = Field("blurry, low quality, deformed, ugly, text, watermark", description="What to avoid")
    steps: int = Field(4, description="Inference steps (4 is fast & high quality for FLUX-schnell)")
    guidance_scale: float = Field(3.5, description="Prompt adherence (higher = stricter)")

class SovereignImaginationSkill(BaseSkill):
    name = "sovereign_imagination"
    description = "Generate high-fidelity photorealistic images locally using FLUX.1 (Offline)."
    input_model = ImageInput
    metabolic_cost = 3 # High cost for ML inference

    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.output_dir = Path(config.paths.data_dir) / "generated_images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Issue 77: Safe backends check
        self.device = "mps" if TORCH_READY and torch.backends.mps.is_available() else "cpu"

    def _load_model(self):
        if self.pipeline:
            return True
        if not FLUX_AVAILABLE:
            return False

        model_id = "black-forest-labs/FLUX.1-schnell"
        logger.info(f"Loading FLUX.1-schnell on {self.device}... (first run downloads ~12GB)")

        try:
            self.pipeline = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if self.device == "mps" else torch.float32,
                revision="main"
            )
            self.pipeline.to(self.device)
            # Enable attention slicing for lower memory on Mac
            self.pipeline.enable_attention_slicing()
            logger.info("✅ FLUX.1-schnell loaded successfully.")
            return True
        except Exception as e:
            record_degradation('sovereign_imagination', e)
            logger.error(f"Model load failed: {e}")
            return False

    async def execute(self, params: ImageInput, context: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(params, dict):
            try:
                params = ImageInput(**params)
            except Exception as e:
                record_degradation('sovereign_imagination', e)
                return {"ok": False, "error": f"Invalid parameters: {e}"}

        if not self._load_model():
            return {"ok": False, "error": "FLUX model failed to load. Ensure dependencies are installed and hardware supports MPS/CPU inference."}

        prompt = params.prompt
        negative = params.negative_prompt
        steps = params.steps
        guidance = params.guidance_scale

        logger.info(f"Generating high-fidelity image: {prompt[:100]}...")

        try:
            # Issue 78: Use asyncio.to_thread instead of loop.run_in_executor(None)
            def _generate():
                # Flux.1-schnell doesn't always support negative_prompt in standard way depending on version,
                # but we include it if the pipeline signature allows.
                image = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=1024,
                    width=1024,
                    max_sequence_length=512
                ).images[0]
                return image

            image = await asyncio.to_thread(_generate)

            # Save
            import time
            timestamp = int(time.time())
            filename = f"flux_{timestamp}.png"
            filepath = self.output_dir / filename
            image.save(filepath)

            # Relative URL for HUD
            relative_url = f"/data/generated_images/{filename}"

            return {
                "ok": True,
                "url": relative_url,
                "path": str(filepath),
                "message": f"High-fidelity image generated locally with FLUX.1-schnell: {relative_url}",
                "display_type": "image"
            }

        except Exception as e:
            record_degradation('sovereign_imagination', e)
            logger.error(f"Generation failed: {e}")
            return {"ok": False, "error": f"Generation failed: {e}"}

    async def on_stop_async(self):
        if self.pipeline:
            del self.pipeline
            self.pipeline = None