import base64
import io
import logging
import os

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service

# Aura Imports
try:
    from core.config import config
# not a failure: the module is optional here, and its absence is the answer.
except ImportError:
    config = None

logger = logging.getLogger("Senses.Vision")


def _screen_capture_preflight() -> bool:
    """Return True only when the current app identity already has screen permission."""
    try:
        import Quartz  # type: ignore

        preflight = getattr(Quartz, "CGPreflightScreenCaptureAccess", None)
        if callable(preflight):
            return bool(preflight())
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation('screen_vision', exc)
        logger.debug("Quartz screen preflight unavailable in LocalVision: %s", exc)
        return True
    return os.getenv("AURA_ASSUME_SCREEN_PERMISSION", "0") == "1"

def _process_image_for_vlm(img):
    """Picklable top-level function for process pool."""
    img.thumbnail((672, 672))
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

class LocalVision:
    """The 'Eyes' of the machine.
    Redirected to Aura's Brain (Cognitive Engine) for unified inference.
    """

    def __init__(self, model="vision-fallback"):
        self.model = model
        self._last_failure_time = 0
        self._cooling_period = 60  # seconds
        
        # v7.0 HARDENING: Eagerly initialize Vision Circuit Breaker (Issue 25)
        from core.resilience.resilience import SmartCircuitBreaker
        self._circuit_breaker = SmartCircuitBreaker(
            name="Vision", failure_threshold=2, base_recovery_timeout=300
        )

    async def capture_screen(self):
        """Take a screenshot of the primary monitor."""
        from core.security.screen_capture_policy import (
            evaluate_screen_capture_admission,
            evaluate_screen_capture_admission_async,
        )

        admission = await evaluate_screen_capture_admission_async()
        if not admission.allowed:
            logger.info("👁️ Screen capture skipped: %s", admission.public_error)
            return None
        try:
            from core.security.permission_guard import PermissionType

            guard = get_runtime_service("permission_guard", default=None)
            if guard:
                check = await guard.check_permission(PermissionType.SCREEN)
                if not check.get("granted", False):
                    logger.info("👁️ Screen capture skipped: screen permission not active for this app identity.")
                    return None
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('screen_vision', exc)
            logger.debug("Screen permission preflight failed before capture: %s", exc)

        from core.utils.executor import run_in_thread
        
        def _safe_screenshot():
            try:
                # The foreground may change while the async permission probe
                # is in flight.  Re-check in the same thread and immediately
                # before the backend acquires pixels.
                if not evaluate_screen_capture_admission().allowed:
                    return None
                if not _screen_capture_preflight():
                    logger.info("👁️ Screen capture preflight denied for LocalVision.")
                    return None
                import pyautogui

                return pyautogui.screenshot()
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('screen_vision', e)
                logger.error("Screenshot failed (check screen recording permissions): %s", e)
                return None
        
        return await run_in_thread(_safe_screenshot)

    async def capture_desktop(self):
        """Compatibility shim for code paths that expect a desktop-capture interface."""
        return await self.capture_screen()

    async def analyze_moment(self, prompt="What is on the user's screen?"):
        """The visual cortex loop.
        Captures screen -> Sends to primary Brain -> Returns description.
        """
        from core.security.permission_guard import PermissionType
        from core.utils.executor import run_in_process
        
        # v7.0 HARDENING: Formally wrap Vision in a SmartCircuitBreaker

        async def _vision_payload():
            # 1. Pre-flight Permission Check
            guard = get_runtime_service("permission_guard", default=None)
            if guard:
                check = await guard.check_permission(PermissionType.SCREEN)
                if not check["granted"]:
                    logger.warning("👁️ Vision Blocked: %s", check["guidance"])
                    return f"I cannot see your screen. {check['guidance']}"

            # 2. Capture
            image = await self.capture_screen()
            if not image:
                raise RuntimeError("Screen capture returned empty (likely permission issue)")
            
            # Offload CPU-heavy image processing to Process Pool
            img_str = await run_in_process(_process_image_for_vlm, image)

            # 3. Analyze using primary Brain (Cognitive Engine)
            brain = get_runtime_service("cognitive_engine", default=None)
            if not brain:
                return "Vision brain unavailable."

            full_prompt = f"IMAGE_DATA_ATTACHED: [Base64 Encoded JPEG]\n\nUSER_REQUEST: {prompt}"
            
            logger.info("🧠 Processing visual data via primary Brain...")
            response = await brain.think(full_prompt, images=[img_str])
            return response.content if hasattr(response, 'content') else str(response)

        try:
            return await self._circuit_breaker.call(_vision_payload)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('screen_vision', e)
            logger.error("👁️ Vision Circuit Tripped: %s", e)
            return "Vision subsystem is offline due to repeated failures (check macOS Screen Recording permissions)."
