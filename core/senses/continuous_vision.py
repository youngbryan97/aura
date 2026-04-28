from core.runtime.errors import record_degradation
import asyncio
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

from core.runtime.boot_safety import main_process_camera_policy

logger = logging.getLogger(__name__)


class ContinuousSensoryBuffer:
    """Maintains a rolling buffer of screen captures for real-time spatial awareness."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.sct = None
        self._mss_module = None
        self._screen_probe_cooldown_until = 0.0
        self._screen_permission_notice_at = 0.0
        self._screen_permission_notice_interval_s = 300.0
        try:
            import mss

            self._mss_module = mss
        except (ImportError, ModuleNotFoundError):
            logger.warning("👁️ [VISION] mss not found. Continuous Sensory Buffer will be disabled.")

        self.frame_buffer = deque(maxlen=6)
        self._capture_task = None
        self._is_active = False
        self.cap: Optional[Any] = None

        from core.config import get_config

        requested_camera = get_config().features.camera_enabled
        if os.environ.get("AURA_FORCE_CAMERA") == "1":
            requested_camera = True

        self.camera_enabled, camera_reason = main_process_camera_policy(requested_camera)
        if self.camera_enabled and os.environ.get("AURA_FORCE_CAMERA") == "1":
            logger.info("👁️ [VISION] Camera FORCED ON via AURA_FORCE_CAMERA=1.")
        elif requested_camera and not self.camera_enabled:
            logger.warning("👁️ [VISION] %s", camera_reason)
        elif not self.camera_enabled:
            logger.info(
                "👁️ [VISION] Camera disabled by default (Metal Conflict Safety). "
                "Use AURA_FORCE_CAMERA=1 plus "
                "AURA_ALLOW_UNSAFE_MAIN_PROCESS_CAMERA=1 to override."
            )

        self.monitor = None

    def start(self):
        """Starts the background rolling capture loop."""
        if not self._is_active:
            if self._mss_module is None and not self.camera_enabled:
                logger.warning("👁️ Continuous Sensory Buffer not started: no capture backends are available.")
                return
            self._is_active = True
            loop = asyncio.get_running_loop()
            self._capture_task = loop.create_task(
                self._capture_loop(), name="ContinuousSensoryCapture"
            )
            logger.info("👁️ Continuous Sensory Buffer Online.")

    def stop(self):
        """Stops the capture loop."""
        self._is_active = False
        if self.cap and self.cap.isOpened():
            try:
                self.cap.release()
            except Exception:
                logger.debug("ContinuousSensoryBuffer: camera release skipped", exc_info=True)
            self.cap = None
        if self._capture_task:
            self._capture_task.cancel()
            self._capture_task = None
            logger.info("👁️ Continuous Sensory Buffer Offline.")

    async def _screen_permission_active(self) -> bool:
        try:
            from core.container import ServiceContainer
            from core.security.permission_guard import PermissionType

            guard = ServiceContainer.get("permission_guard", default=None)
            if not guard:
                return os.getenv("AURA_ASSUME_SCREEN_PERMISSION", "0") == "1"
            check = await guard.check_permission(PermissionType.SCREEN)
            granted = bool(check.get("granted", False))
            if not granted:
                now = time.monotonic()
                if (
                    self._screen_permission_notice_at <= 0.0
                    or (now - self._screen_permission_notice_at) >= self._screen_permission_notice_interval_s
                ):
                    logger.info(
                        "👁️ [VISION] Continuous screen buffer deferred: screen permission is not active for this app identity."
                    )
                    self._screen_permission_notice_at = now
            else:
                self._screen_permission_notice_at = 0.0
            return granted
        except Exception as exc:
            record_degradation('continuous_vision', exc)
            logger.debug("ContinuousSensoryBuffer permission probe failed: %s", exc)
            return False

    async def _ensure_screen_backend(self) -> bool:
        if self.sct is not None and self.monitor is not None:
            return True
        if self._mss_module is None:
            return False
        if time.monotonic() < self._screen_probe_cooldown_until:
            return False
        if not await self._screen_permission_active():
            self._screen_probe_cooldown_until = time.monotonic() + 15.0
            return False
        try:
            sct = await asyncio.to_thread(self._mss_module.mss)
            try:
                monitor = sct.monitors[1]
            except IndexError:
                monitor = sct.monitors[0]
            self.sct = sct
            self.monitor = monitor
            self._screen_probe_cooldown_until = 0.0
            logger.info("👁️ [VISION] Continuous screen capture backend initialized.")
            return True
        except Exception as exc:
            record_degradation('continuous_vision', exc)
            self._screen_probe_cooldown_until = time.monotonic() + 15.0
            logger.warning("👁️ [VISION] Continuous screen capture backend unavailable: %s", exc)
            return False

    async def _capture_loop(self):
        """Runs continuously in the background, updating Aura's visual working memory."""
        while self._is_active:
            try:
                if self.sct is None or self.monitor is None:
                    await self._ensure_screen_backend()
                if self.sct and self.monitor:
                    sct_img = await asyncio.get_running_loop().run_in_executor(
                        None, self.sct.grab, self.monitor
                    )

                    import mss.tools

                    png_bytes = mss.tools.to_png(sct_img.rgb, sct_img.size)
                    self.frame_buffer.append(("image/png", png_bytes))

                if self.camera_enabled:
                    if self.cap is None or not self.cap.isOpened():
                        import cv2

                        self.cap = cv2.VideoCapture(0)
                        if self.cap:
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                    if self.cap and self.cap.isOpened():
                        ret, frame = self.cap.read()
                        if ret:
                            import cv2

                            _, jpeg_bytes = cv2.imencode('.jpg', frame)
                            self.frame_buffer.append(("image/jpeg", jpeg_bytes.tobytes()))
            except Exception as e:
                record_degradation('continuous_vision', e)
                logger.error(f"Sensory Buffer capture failed: {e}")

            await asyncio.sleep(2.0)

    def get_visual_context_parts(self) -> list:
        """Retrieves the rolling visual buffer formatted for the Gemini API."""
        if not self.frame_buffer:
            return []

        return [
            {"mime_type": mime_type, "data": frame_bytes}
            for mime_type, frame_bytes in self.frame_buffer
        ]

    async def query_visual_context(self, prompt: str, brain: Any) -> str:
        """
        Sends the current frame buffer and the prompt to the brain for visual reasoning.

        Args:
            prompt: The specific question or directive for the visual context.
            brain: The CognitiveEngine (or GeminiAdapter) instance capable of multimodal logic.
        """
        if not self.frame_buffer:
            return "I don't have any visual frames in my buffer yet."

        parts = self.get_visual_context_parts()
        parts.insert(0, {"text": prompt})

        try:
            if hasattr(brain, "think"):
                from core.brain.types import ThinkingMode

                thought = await brain.think(prompt, mode=ThinkingMode.FAST, parts=parts)
                return thought.content if hasattr(thought, "content") else str(thought)
            elif hasattr(brain, "call"):
                success, text, _ = await brain.call(prompt, parts=parts)
                return text if success else "I failed to process the visual data."
            else:
                return "My cognitive systems are not equipped for that visual request."
        except Exception as e:
            record_degradation('continuous_vision', e)
            logger.error(f"Visual reasoning failed: {e}")
            return f"I had an error analyzing my vision: {e}"
