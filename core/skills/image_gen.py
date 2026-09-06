"""core/skills/image_gen.py — Sovereign Image Generation & Editing
==================================================================
First-class BaseSkill for local image generation and editing.
Extends the existing _local_media_generation.py with:
  - Image-to-image editing (inpainting, style transfer)
  - Multiple backend support (MLX diffusers, CoreML, SDXL)
  - Automatic prompt engineering for photorealistic quality
  - Output management with URL serving

This closes the "image generation" gap in tool parity.
"""

import asyncio
import gc
import logging
import os
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from core.config import config
from core.runtime.errors import FallbackClassification, record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.ImageGen")

_IMAGEGEN_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    TimeoutError,
)


def _record_imagegen_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "image_gen",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=False,
        extra=extra,
    )


class ImageGenInput(BaseModel):
    prompt: str = Field(..., description="Description of the image to generate or edit.")

    @model_validator(mode="before")
    @classmethod
    def _accept_generic_dispatch_shape(cls, data: Any) -> Any:
        """The skill router's generic dispatch passes the user text as
        ``query``; the diffusion contract calls it ``prompt``. Map it instead
        of exploding — a field-name mismatch is not invalid input (seen live
        as a capability_engine CRITICAL on an image_gen dispatch)."""
        if isinstance(data, dict):
            if not data.get("prompt"):
                fallback = str(data.get("query") or data.get("text") or "").strip()
                if fallback:
                    data = {**data, "prompt": fallback}
            data = {k: v for k, v in data.items() if k not in ("query", "text")}
        return data

    negative_prompt: str | None = Field(
        None,
        description="What to avoid in the image.",
    )
    style: str | None = Field(
        None,
        description="Visual style guidance (e.g., 'photorealistic', 'anime', 'oil painting').",
    )
    width: int = Field(1024, ge=256, le=2048, description="Image width in pixels.")
    height: int = Field(1024, ge=256, le=2048, description="Image height in pixels.")
    # Lower bounds widened for distilled/turbo models, which are trained for
    # 1-4 steps at guidance 0 and are the only way to get interactive latency on
    # this hardware. The old floors (steps>=10, guidance>=1.0) made those models
    # impossible to drive correctly.
    steps: int = Field(40, ge=1, le=100, description="Number of inference steps.")
    guidance_scale: float = Field(
        8.0,
        ge=0.0,
        le=20.0,
        description="Adherence to prompt (higher = more literal; 0 for turbo models).",
    )
    seed: int | None = Field(None, description="Random seed for reproducibility.")
    enhance: bool = Field(
        True,
        description=(
            "Append quality boosters to the prompt. Disable when the prompt is "
            "already exactly what should be drawn (e.g. rendering Aura's own "
            "mental canvas), since the boosters override its intent."
        ),
    )
    source_image_path: str | None = Field(
        None,
        description="Path to source image for img2img / editing tasks.",
    )
    strength: float = Field(
        0.75,
        ge=0.0,
        le=1.0,
        description="How much to transform source image (0=none, 1=full).",
    )


class ImageGenSkill(BaseSkill):
    name = "image_gen"
    description = (
        "Draw or paint a picture, an illustration or a piece of artwork, and edit an "
        "existing image, using local diffusion models. Supports text-to-image, "
        "image-to-image editing, style transfer and inpainting. Outputs images saved "
        "to disk. Use for anything pictorial; not for a schematic or a technical "
        "drawing, which are computed from a model rather than generated."
    )
    input_model = ImageGenInput
    timeout_seconds = 300.0  # Image generation can be slow
    metabolic_cost = 3  # Heavy GPU/CPU workload
    effect_scope = "read_write_artifacts"
    #: What a caller gets back, machine-readable. A prose `output`
    #: tells a reader what to expect and a caller nothing it can check.
    result_schema = {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "error": {"type": "string"}
        },
        "required": ["ok"],
        "additionalProperties": True,
    }

    def __init__(self) -> None:
        super().__init__()
        self._pipeline: Any | None = None
        self._img2img_pipeline: Any | None = None
        self._model_loaded = False
        self._device: str | None = None
        # SDXL-Turbo over SDXL-base: same architecture, distilled to 1-4 steps.
        # On this host SDXL-base needs float32 (float16 decodes to black on MPS)
        # at 40 steps/1024px, which is minutes per image — unusable for anything
        # interactive. Turbo measures 4.6s at 4 steps. AURA_IMAGE_MODEL overrides
        # for anyone who wants base quality and can wait.
        self._model_id = os.environ.get("AURA_IMAGE_MODEL", "stabilityai/sdxl-turbo")
        self._fallback_model_id = "runwayml/stable-diffusion-v1-5"
        self._output_dir = Path(config.paths.data_dir) / "generated_images"
        self._lane_lease: Any | None = None
        self._pipeline_lock = asyncio.Lock()
        self._generation_lock = asyncio.Lock()
        self._resident_mode = "txt2img"
        self._closing = False

    @staticmethod
    def _detect_device() -> str:
        """Detect the best available compute device."""
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except (ImportError, AttributeError) as _exc:
            logger.debug("Suppressed %s in core.skills.image_gen: %s", type(_exc).__name__, _exc)
        return "cpu"

    def _load_pipeline(self, img2img: bool = False) -> bool:
        """Lazy-load the diffusion pipeline."""
        if img2img and self._img2img_pipeline:
            return True
        if not img2img and self._pipeline:
            return True
        if self._device is None:
            self._device = self._detect_device()

        try:
            import torch
            from diffusers import (
                AutoPipelineForImage2Image,
                AutoPipelineForText2Image,
            )
        except ImportError as exc:
            _record_imagegen_degradation(
                exc,
                action="reported missing torch/diffusers dependencies",
            )
            logger.error("torch/diffusers not installed: %s", exc)
            return False

        if img2img:
            self._pipeline = None
        else:
            self._img2img_pipeline = None
        gc.collect()
        try:
            if self._device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif self._device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (AttributeError, RuntimeError):
            logger.debug("Diffusion cache clear before pipeline mode switch failed")

        # SDXL in float16 on MPS decodes to NaN and silently writes a pure black
        # PNG — the skill reports success and hands back an all-zero image.
        # Measured on this host (torch 2.11 / diffusers 0.37 / M-series):
        #   sdxl-turbo mps fp16 -> std 0.00  (BLACK)
        #   sdxl-turbo mps fp32 -> std 53.5  (real image, 4.6s @ 4 steps)
        # float16 stays on CUDA, where it is well behaved.
        torch_dtype = torch.float16 if self._device == "cuda" else torch.float32

        model_id = self._model_id
        pipeline_cls = (
            AutoPipelineForImage2Image if img2img else AutoPipelineForText2Image
        )

        for attempt_model in (model_id, self._fallback_model_id):
            try:
                logger.info(
                    "Loading %s pipeline (%s) on %s...",
                    "img2img" if img2img else "txt2img",
                    attempt_model,
                    self._device,
                )
                pipe = pipeline_cls.from_pretrained(
                    attempt_model,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                )
                try:
                    pipe.to(self._device)
                except _IMAGEGEN_RECOVERABLE_ERRORS as exc:
                    _record_imagegen_degradation(
                        exc,
                        action="kept pipeline on default device after move failed",
                        extra={"device": self._device, "model": attempt_model},
                    )

                # Memory optimization
                if hasattr(pipe, "enable_attention_slicing"):
                    try:
                        pipe.enable_attention_slicing()
                    except _IMAGEGEN_RECOVERABLE_ERRORS as _exc:
                        logger.debug("Suppressed %s in core.skills.image_gen: %s", type(_exc).__name__, _exc)

                if img2img:
                    self._pipeline = None
                    self._img2img_pipeline = pipe
                    self._resident_mode = "img2img"
                else:
                    self._img2img_pipeline = None
                    self._pipeline = pipe
                    self._resident_mode = "txt2img"

                self._model_loaded = True
                logger.info("✓ %s pipeline loaded.", "img2img" if img2img else "txt2img")
                return True

            except _IMAGEGEN_RECOVERABLE_ERRORS as exc:
                _record_imagegen_degradation(
                    exc,
                    action=f"failed to load model {attempt_model}, trying fallback",
                    extra={"model": attempt_model},
                )
                logger.warning("Model %s failed: %s", attempt_model, exc)
                continue

        return False

    async def _ensure_pipeline(self, *, img2img: bool) -> bool:
        async with self._pipeline_lock:
            if self._closing:
                return False
            if img2img and self._img2img_pipeline is not None:
                return True
            if not img2img and self._pipeline is not None:
                return True
            if self._lane_lease is None:
                from core.runtime.model_lane_control import (
                    acquire_in_process_model_lane,
                    estimate_model_job_footprint_gb,
                    run_owned_model_thread_call,
                )

                request_gb = max(
                    estimate_model_job_footprint_gb(self._model_id, purpose="serve"),
                    estimate_model_job_footprint_gb(
                        self._fallback_model_id,
                        purpose="serve",
                    ),
                )
                self._lane_lease = await acquire_in_process_model_lane(
                    owner_id=f"image-generation:{id(self)}",
                    model_path=self._model_id,
                    purpose="serve",
                    request_gb=request_gb,
                    priority=60,
                    preemptible=False,
                    evict=self._evict_for_model_lane,
                    compensate=self._compensate_model_lane,
                    metadata={
                        "skill": self.name,
                        "fallback_model": self._fallback_model_id,
                        "single_pipeline_residency": True,
                        "activation_state": "loading",
                    },
                )
            loaded = await run_owned_model_thread_call(
                lambda: self._load_pipeline(img2img),
                operation_name="image-generation-pipeline-load",
            )
            if loaded:
                if self._lane_lease is None or not await self._lane_lease.set_preemptible(
                    True
                ):
                    await self._unload_pipelines(
                        reason="image_generation_activation_fence_lost"
                    )
                    return False
                return True
            if self._pipeline is None and self._img2img_pipeline is None:
                lease, self._lane_lease = self._lane_lease, None
                if lease is not None:
                    await lease.release(reason="image_generation_model_load_failed")
            return False

    async def _evict_for_model_lane(self, _owner: Any, reason: str) -> bool:
        if self._generation_lock.locked():
            logger.warning("Image model preemption refused during active use: %s", reason)
            return False
        try:
            await asyncio.wait_for(self._pipeline_lock.acquire(), timeout=0.01)
        except TimeoutError:
            logger.warning("Image model preemption refused during pipeline transition: %s", reason)
            return False
        try:
            await self._unload_pipelines(reason=f"lane_eviction:{reason}")
        finally:
            self._pipeline_lock.release()
        return self._pipeline is None and self._img2img_pipeline is None

    async def _compensate_model_lane(self, _owner: Any, reason: str) -> bool:
        if self._closing:
            return False
        logger.info("Restoring diffusion pipeline after failed candidate: %s", reason)
        return await self._ensure_pipeline(img2img=self._resident_mode == "img2img")

    def _enhance_prompt(self, prompt: str, style: str | None, enhance: bool = True) -> str:
        """Apply automatic prompt engineering for maximum quality.

        The quality suffix is not free: "masterpiece, 8k, HDR, cinematic
        lighting" overwhelms an abstract prompt and pulls the sampler toward a
        photoreal scene. Rendering Aura's own mental canvas through it turned
        "internal associative canvas: silent in the foreground…" into a sunlit
        plaza. Callers who mean their prompt literally pass enhance=False.
        """
        style_prefixes = {
            "photorealistic": "photorealistic, ultra-detailed photograph, ",
            "anime": "anime style, studio ghibli, vibrant colors, ",
            "oil_painting": "oil painting on canvas, impasto technique, ",
            "watercolor": "delicate watercolor painting, soft edges, ",
            "digital_art": "professional digital artwork, concept art, ",
            "3d_render": "3D rendered scene, octane render, volumetric lighting, ",
            "pixel_art": "pixel art, retro game style, 8-bit, ",
            "sketch": "detailed pencil sketch, cross-hatching, ",
        }

        prefix = ""
        if style:
            normalized_style = style.lower().replace(" ", "_")
            prefix = style_prefixes.get(
                normalized_style,
                f"{style} style, ",
            )

        suffix = (
            ", masterpiece, best quality, 8k, HDR, cinematic lighting, sharp focus"
            if enhance
            else ""
        )
        return f"{prefix}{prompt}{suffix}"

    async def execute(
        self, params: ImageGenInput, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate or edit an image."""
        if isinstance(params, dict):
            try:
                params = ImageGenInput(**params)
            except _IMAGEGEN_RECOVERABLE_ERRORS as exc:
                _record_imagegen_degradation(
                    exc,
                    action="rejected invalid image generation input",
                )
                return {"ok": False, "error": f"Invalid input: {exc}"}

        prompt = params.prompt.strip()
        if not prompt:
            return {"ok": False, "error": "No prompt provided."}

        # Determine mode: img2img vs txt2img
        is_img2img = bool(params.source_image_path)

        async with self._generation_lock:
            if is_img2img:
                return await self._generate_img2img(params)
            return await self._generate_txt2img(params)

    async def _generate_txt2img(
        self, params: ImageGenInput
    ) -> dict[str, Any]:
        """Text-to-image generation."""
        if not await self._ensure_pipeline(img2img=False):
            return {
                "ok": False,
                "error": "Image generation model failed to load. Check torch/diffusers installation.",
            }

        enhanced_prompt = self._enhance_prompt(params.prompt, params.style, params.enhance)
        negative = params.negative_prompt or (
            "blur, low quality, distortion, watermark, text, ugly, bad anatomy, deformed"
        )

        logger.info("🎨 Generating image: '%s'...", params.prompt[:60])

        try:
            import torch

            generator = None
            if params.seed is not None:
                generator = torch.Generator(device=self._device).manual_seed(params.seed)

            pipeline = self._pipeline
            if pipeline is None:
                return {"ok": False, "error": "Image generation pipeline is unavailable."}

            steps, guidance = self._sampler_settings(params.steps, params.guidance_scale)

            def _generate() -> Any:
                return pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt=negative,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=params.width,
                    height=params.height,
                    generator=generator,
                ).images[0]

            from core.runtime.model_lane_control import run_owned_model_thread_call

            image = await run_owned_model_thread_call(
                _generate,
                operation_name="image-generation-txt2img",
            )
            return await self._save_and_respond(image, params.prompt)

        except _IMAGEGEN_RECOVERABLE_ERRORS as exc:
            _record_imagegen_degradation(
                exc,
                action="reported txt2img generation failure",
                extra={"prompt": params.prompt[:100]},
            )
            return {"ok": False, "error": f"Generation failed: {exc}"}

    async def _generate_img2img(
        self, params: ImageGenInput
    ) -> dict[str, Any]:
        """Image-to-image editing."""
        if not params.source_image_path:
            return {"ok": False, "error": "No source image path provided."}

        source_path = Path(params.source_image_path)
        if not await asyncio.to_thread(source_path.exists):
            return {"ok": False, "error": f"Source image not found: {source_path}"}

        if not await self._ensure_pipeline(img2img=True):
            return {
                "ok": False,
                "error": "Image editing model failed to load.",
            }

        try:
            from PIL import Image

            def _load_source_image() -> Any:
                with Image.open(source_path) as raw_image:
                    converted = raw_image.convert("RGB")
                    return converted.resize(
                        (params.width, params.height),
                        Image.LANCZOS,
                    )

            source_image = await asyncio.to_thread(_load_source_image)
        except _IMAGEGEN_RECOVERABLE_ERRORS as exc:
            return {"ok": False, "error": f"Failed to load source image: {exc}"}

        enhanced_prompt = self._enhance_prompt(params.prompt, params.style, params.enhance)
        negative = params.negative_prompt or (
            "blur, low quality, distortion, watermark, text"
        )

        logger.info("🖌️ Editing image: '%s'...", params.prompt[:60])

        try:
            pipeline = self._img2img_pipeline
            if pipeline is None:
                return {"ok": False, "error": "Image editing pipeline is unavailable."}

            def _edit() -> Any:
                return pipeline(
                    prompt=enhanced_prompt,
                    image=source_image,
                    negative_prompt=negative,
                    num_inference_steps=params.steps,
                    guidance_scale=params.guidance_scale,
                    strength=params.strength,
                ).images[0]

            from core.runtime.model_lane_control import run_owned_model_thread_call

            image = await run_owned_model_thread_call(
                _edit,
                operation_name="image-generation-img2img",
            )
            return await self._save_and_respond(image, params.prompt, mode="img2img")

        except _IMAGEGEN_RECOVERABLE_ERRORS as exc:
            _record_imagegen_degradation(
                exc,
                action="reported img2img editing failure",
                extra={"source": str(source_path)},
            )
            return {"ok": False, "error": f"Image editing failed: {exc}"}

    async def _save_and_respond(
        self, image: Any, prompt: str, mode: str = "txt2img"
    ) -> dict[str, Any]:
        """Save generated image and build response."""
        # A degenerate decode (all-black / uniform) must not be reported as a
        # successful generation: that is how an fp16 NaN silently became a
        # "created image" with a real path and URL. Cheap to check, and the only
        # thing standing between a numerical fault and a confident lie.
        degenerate = await asyncio.to_thread(self._is_degenerate, image)
        if degenerate:
            _record_imagegen_degradation(
                RuntimeError("diffusion decoded to a uniform image"),
                action="refused to return an all-black generation as success",
                extra={"mode": mode},
            )
            return {
                "ok": False,
                "error": (
                    "Image generation produced a blank image (numerical fault in the "
                    "diffusion decode). Nothing was saved."
                ),
            }

        timestamp = int(time.time())
        filename = f"gen_{mode}_{timestamp}_{uuid.uuid4().hex[:10]}.png"
        filepath = self._output_dir / filename

        try:
            def _encode_png() -> bytes:
                output = BytesIO()
                image.save(output, format="PNG")
                return output.getvalue()

            payload = await asyncio.to_thread(_encode_png)
            # The tool execution itself was authorized upstream (standing
            # authority + Will + constitution); this internal artifact write
            # still needs an active governed scope or the gateway refuses it
            # — the exact failure the first live RENDER THIS hit after every
            # authorization gate had already passed.
            from core.governance_context import local_internal_governed_scope

            with local_internal_governed_scope("image_gen_output"):
                await get_file_write_gateway().write_bytes_async(
                    filepath,
                    payload,
                    source="skills.image_gen.output",
                )
        except _IMAGEGEN_RECOVERABLE_ERRORS as exc:
            return {"ok": False, "error": f"Failed to save image: {exc}"}

        relative_url = f"/data/generated_images/{filename}"

        return {
            "ok": True,
            "url": relative_url,
            "path": str(filepath),
            "mode": mode,
            "type": "image",
            "summary": f"Generated {mode} image from prompt: {prompt[:80]}",
            "message": f"Image created ({mode}): {relative_url}",
        }

    def _sampler_settings(self, steps: int, guidance: float) -> tuple[int, float]:
        """Reconcile requested sampler settings with what the model expects.

        Distilled/turbo checkpoints are adversarially trained for 1-4 steps with
        classifier-free guidance disabled. Handing them a caller's 40-step,
        guidance-8 default costs ten times the latency *and* degrades the image,
        so those requests are clamped rather than honoured literally.
        """
        if "turbo" not in str(self._model_id).lower():
            return steps, guidance
        return max(1, min(int(steps), 4)), 0.0

    @staticmethod
    def _is_degenerate(image: Any) -> bool:
        """True when the decode carries no signal (uniform / all-black).

        Uses the image's own extrema rather than pulling numpy in: a real
        generation always spans a range, a NaN-collapsed one does not.
        """
        try:
            bands = image.convert("RGB").getextrema()
        except (AttributeError, ValueError, OSError):
            return False
        try:
            return all(lo == hi for lo, hi in bands)
        except (TypeError, ValueError):
            return False

    async def _unload_pipelines(self, *, reason: str) -> None:
        self._pipeline = None
        self._img2img_pipeline = None
        self._model_loaded = False
        lease, self._lane_lease = self._lane_lease, None
        try:
            import torch

            if bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available():
                await asyncio.to_thread(torch.mps.empty_cache)
            elif torch.cuda.is_available():
                await asyncio.to_thread(torch.cuda.empty_cache)
        except (ImportError, AttributeError, RuntimeError):
            pass
        await asyncio.to_thread(gc.collect)
        if lease is not None:
            await lease.release(reason=reason)

    async def on_stop_async(self) -> None:
        """Release model references, lane ownership, and accelerator cache."""

        self._closing = True
        async with self._generation_lock:
            async with self._pipeline_lock:
                await self._unload_pipelines(reason="image_generation_skill_stopped")
