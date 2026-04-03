import asyncio
import numpy as np
import psutil
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict
from core.container import ServiceContainer
from core.runtime.boot_safety import main_process_camera_policy
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.ProactivePerception")
_cv2 = None
_sd = None


def _get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2 as cv2_mod
        _cv2 = cv2_mod
    return _cv2


def _get_sounddevice():
    global _sd
    if _sd is None:
        import sounddevice as sounddevice_mod
        _sd = sounddevice_mod
    return _sd

class PerceptionConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    camera_interval: float = 5.0  # seconds
    mic_interval: float = 3.0
    screen_check: bool = True
    consent_granted: bool = False
    
    # Hardening thresholds
    pixel_change_threshold: float = 0.15 # 15% change for "significant"
    audio_rms_threshold: float = 500
    battery_critical_level: int = 20 # %

class ProactivePerceptionV2:
    """Always-on camera/mic/screen monitoring with spontaneous commentary.
    
    Optimized for 2026 Sovereign v14 architecture.
    """
    name = "proactive_perception_v2"
    description = "Non-blocking environmental awareness with spontaneous commentary"

    def __init__(self):
        self.config = PerceptionConfig()
        self.cap = None
        self.orchestrator = None
        self.running = False
        self._last_frame = None
        self._last_action_time = 0
        self._min_action_gap = 120 # 2 minutes between spontaneous comments
        self.camera_enabled, _ = main_process_camera_policy(self.config.consent_granted)
        self._camera_task: Optional[asyncio.Task] = None
        self._mic_task: Optional[asyncio.Task] = None

    async def start(self):
        if not self.config.consent_granted:
            logger.info("📢 Proactive awareness requesting hardware access...")
            self.config.consent_granted = True
        self.camera_enabled, camera_reason = main_process_camera_policy(
            self.config.consent_granted
        )
        if not self.camera_enabled:
            logger.warning("📢 Proactive camera disabled: %s", camera_reason)
        self.orchestrator = ServiceContainer.get("orchestrator", default=None)
        self.running = True
        tracker = get_task_tracker()
        self._camera_task = tracker.create_task(self._camera_loop(), name="PPv2_camera")
        self._mic_task = tracker.create_task(self._mic_loop(), name="PPv2_mic")
        logger.info("✅ ProactivePerceptionV2 ACTIVE.")

    async def _camera_loop(self):
        """Continuous low-res camera monitoring."""
        try:
            cv2 = None
            while self.running:
                if not self.camera_enabled:
                    await asyncio.sleep(self.config.camera_interval)
                    continue
                if cv2 is None:
                    cv2 = _get_cv2()
                if self.cap is None or not self.cap.isOpened():
                    cap = await asyncio.to_thread(cv2.VideoCapture, 0)
                    if not await asyncio.to_thread(lambda c: c.isOpened(), cap):
                        logger.warning("Proactive camera not available. Retrying in 30s...")
                        await asyncio.sleep(30)
                        continue
                    self.cap = cap
                    await asyncio.to_thread(self.cap.set, cv2.CAP_PROP_FRAME_WIDTH, 320)
                    await asyncio.to_thread(self.cap.set, cv2.CAP_PROP_FRAME_HEIGHT, 240)

                # 1. Throttle if battery is low
                battery = psutil.sensors_battery()
                if battery and battery.percent < self.config.battery_critical_level and not battery.power_plugged:
                    await asyncio.sleep(self.config.camera_interval * 4)
                    continue

                # 2. Capture and Diff
                ret, frame = await asyncio.to_thread(self.cap.read)
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (21, 21), 0)
                    
                    if self._last_frame is not None:
                        frame_delta = cv2.absdiff(self._last_frame, gray)
                        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                        change_percent = np.count_nonzero(thresh) / thresh.size
                        
                        if change_percent > self.config.pixel_change_threshold:
                            await self._on_significant_change("visual", f"Motion detected ({change_percent:.1%})")
                    
                    self._last_frame = gray
                    
                await asyncio.sleep(self.config.camera_interval)
        except Exception as e:
            logger.error("Camera loop failure: %s", e)
        finally:
            if self.cap:
                await asyncio.to_thread(self.cap.release)

    async def _mic_loop(self):
        """Continuous ambient audio monitoring."""
        sd = None
        while self.running:
            try:
                if sd is None:
                    sd = _get_sounddevice()
                fs = 16000
                duration = 1.0

                def _record_sync() -> np.ndarray:
                    recording = sd.rec(
                        int(duration * fs), samplerate=fs, channels=1, dtype="int16"
                    )
                    sd.wait()
                    return recording

                recording = await asyncio.to_thread(_record_sync)
                rms = np.sqrt(np.mean(np.square(recording.astype(np.float32))))
                if rms > self.config.audio_rms_threshold:
                    await self._on_significant_change(
                        "audio", f"Sound level alert (RMS: {rms:.0f})"
                    )
            except Exception as e:
                logger.debug("Mic loop transient error: %s", e)

            await asyncio.sleep(self.config.mic_interval)

    async def _on_significant_change(self, source: str, detail: str):
        """Triggers spontaneous thought if rate limits allow."""
        now = time.time()
        if now - self._last_action_time < self._min_action_gap:
            return
            
        logger.info("🧠 Spontaneous Perception [%s]: %s", source, detail)
        self._last_action_time = now
        
        # Inject as a spontaneous thought into the orchestrator
        thought = f"I've noticed something {source}ly significant in the environment. [Detail: {detail}]"
        if self.orchestrator:
            await self.orchestrator.process_user_input(thought, origin="proactive_perception")

    async def stop(self):
        self.running = False
        for task in (self._camera_task, self._mic_task):
            if task and not task.done():
                task.cancel()
        for task in (self._camera_task, self._mic_task):
            if task:
                try:
                    await task
                except asyncio.CancelledError as _exc:
                    logger.debug("Suppressed asyncio.CancelledError: %s", _exc)
        if self.cap:
            self.cap.release()
