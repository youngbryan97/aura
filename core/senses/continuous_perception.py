"""Continuous Perception Engine
Phase 23: Enterprise Autonomy - Unprompted Action & Awareness

This engine handles constant audio and visual input, detecting wake words,
intentions, and significant visual changes without requiring direct user prompts.
Autonomous perceptions may advise the runtime, but they must not silently bypass
constitutional routing once the live runtime is up.
"""

import asyncio
import os
import logging
import time
from typing import Optional

try:
    from PIL import Image, ImageChops
except ImportError:
    Image = None
    ImageChops = None

from core.container import ServiceContainer
from core.health.degraded_events import record_degraded_event
from core.senses.perceptual_buffer import PerceptualBuffer
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.Senses.ContinuousPerception")

class ContinuousPerceptionEngine:
    """Always-on perception loop for enterprise autonomy."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.running = False
        self._audio_task = None
        self._vision_task = None
        
        # Phase 10: Perceptual Buffer
        self.buffer = PerceptualBuffer(maxsize=100)
        ServiceContainer.register_instance("perceptual_buffer", self.buffer)

        # Vision Delta Tracking
        self._last_image: Optional[Image.Image] = None
        self.vision_check_interval = 5.0 # seconds
        self.visual_delta_threshold = 0.15 # 15% difference triggers an analysis
        self.enable_proactive_vision = os.getenv("AURA_ENABLE_PROACTIVE_VISION", "0") == "1"
        if not self.enable_proactive_vision:
            logger.info("👁️ Continuous vision monitoring disabled by default. Set AURA_ENABLE_PROACTIVE_VISION=1 to enable ambient screen capture.")
        
        # Wake word setup
        self.wake_words = ["aura"]
        
        # Ambient Context Buffer (Phase 5)
        self.ambient_update_interval = 60.0 # Update ambient context every 60 seconds
        self._ambient_task = None
        self.current_ambient_context = ""

        # Phase 16: Soma Integration
        from core.senses.soma import get_soma
        self.soma = get_soma()

        # Skill Caching (Issue 14)
        try:
            from skills.computer_use import ComputerUseSkill
            self._computer_use_skill = ComputerUseSkill()
        except ImportError:
            self._computer_use_skill = None

    async def start(self):
        if self.running: return
        self.running = True
        logger.info("📡 Continuous Perception Engine: ONLINE")

        tracker = get_task_tracker()
        self._audio_task = tracker.track_task(
            get_task_tracker().create_task(self._continuous_audio_loop(), name="continuous_perception.audio")
        )
        if self.enable_proactive_vision:
            self._vision_task = tracker.track_task(
                get_task_tracker().create_task(self._continuous_vision_loop(), name="continuous_perception.vision")
            )
            self._ambient_task = tracker.track_task(
                get_task_tracker().create_task(self._continuous_ambient_loop(), name="continuous_perception.ambient")
            )
        else:
            self._vision_task = None
            self._ambient_task = None

    async def stop(self):
        self.running = False
        if self._audio_task: self._audio_task.cancel()
        if self._vision_task: self._vision_task.cancel()
        if self._ambient_task: self._ambient_task.cancel()
        logger.info("📡 Continuous Perception Engine: OFFLINE")

    async def _continuous_audio_loop(self):
        """Monitors the microphone stream continuously."""
        ears = ServiceContainer.get("ears", default=None)
        if not ears:
            logger.warning("Continuous Audio needs 'ears' service.")
            return

        def _on_ambient_listen(text: str):
            text = text.lower().strip()
            if not text: return
            
            if any(w in text for w in self.wake_words):
                logger.info("🎧 Wake word detected in ambient audio: '%s'", text)
                self.buffer.append("audio", f"Wake word detected: {text}")
                self._dispatch_spontaneous_intent(text, source="audio_wake")
            elif "turn on" in text or "watch this" in text:
                logger.info("🎧 Implicit direct command detected: '%s'", text)
                self.buffer.append("audio", f"Implicit command: {text}")
                self._dispatch_spontaneous_intent(text, source="audio_implicit")
            else:
                self.buffer.append("audio", text[:50])

        # Register callback ONCE (Issue 12)
        if getattr(ears, "_engine", None) and hasattr(ears._engine, "on_transcript"):
            ears._engine.on_transcript(_on_ambient_listen)
            logger.info("📡 Continuous audio listener registered.")

        while self.running:
            try:
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Continuous audio error: %s", e)
                await asyncio.sleep(5)

    async def _calculate_image_difference_async(self, img1, img2) -> float:
        """Calculate percentage difference between two images asynchronously."""
        if not img1 or not img2 or img1.size != img2.size:
            return 1.0 # 100% different if none or mismatched

        def _calc_diff():
            if Image is None or ImageChops is None:
                return 1.0  # Treat as fully different if PIL unavailable
            # Convert to grayscale for simple diff
            i1 = img1.convert("L")
            i2 = img2.convert("L")
            
            diff = ImageChops.difference(i1, i2)
            
            # Calculate bounding box of non-zero regions
            if not diff.getbbox():
                return 0.0 # Identical
                
            # Get RMS difference
            hist = diff.histogram()
            total_pixels = i1.size[0] * i1.size[1]
            
            # Sum of pixels that are significantly different (threshold > 30)
            diff_pixels = sum(hist[30:]) 
            return diff_pixels / total_pixels
            
        return await asyncio.to_thread(_calc_diff)

    async def _continuous_vision_loop(self):
        """Monitors the screen for significant changes."""
        vision = ServiceContainer.get("vision_engine", default=None)
        if not vision:
            logger.warning("Continuous Vision needs 'vision_engine' service.")
            return

        while self.running:
            try:
                await asyncio.sleep(self.vision_check_interval)
                
                # Capture screen silently
                current_image = await vision.capture_screen()
                
                if current_image:
                    # Shrink to thumbnail for fast delta checking
                    current_thumb = current_image.copy()
                    current_thumb.thumbnail((300, 300))
                    
                    if self._last_image:
                        delta = await self._calculate_image_difference_async(self._last_image, current_thumb)
                        if delta > self.visual_delta_threshold:
                            logger.info("👁️ Significant visual change detected (%.0f%%). Triggering analysis.", delta * 100)
                            
                            # Perform full analysis on the actual image
                            description = await vision.analyze_moment(
                                prompt="The screen just changed significantly. What is happening now?"
                            )
                            
                            if description and len(description) > 10:
                                self.buffer.append("vision", description)
                                self._dispatch_spontaneous_intent(
                                    f"I am actively watching the screen and just saw this change: {description}", 
                                    source="vision_delta"
                                )
                    
                    self._last_image = current_thumb
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Continuous vision error: %s", e)

    async def _continuous_ambient_loop(self):
        """Phase 5: Intermittently samples the desktop context (text) even without visual deltas.
        This provides a rolling 'Ambient Context' buffer for the Agency to draw on."""
        while self.running:
            try:
                # Wait for system to settle and interval to pass
                await asyncio.sleep(self.ambient_update_interval)
                
                logger.debug("📡 Ambient Context: Sampling environment...")
                if self._computer_use_skill is None:
                    logger.debug("ComputerUseSkill unavailable; skipping ambient context.")
                    await asyncio.sleep(self.ambient_update_interval)
                    continue

                # Non-blocking screen text grab
                screen_text = await asyncio.to_thread(self._computer_use_skill.read_screen_text)
                
                if screen_text and len(screen_text) > 20:
                    # Summarize the screen text to avoid context bloat
                    prompt = f"Summarize what the user is currently looking at or working on in 1-2 sentences. Screen text:\n{screen_text[:2000]}"
                    try:
                        from core.container import ServiceContainer
                        brain = ServiceContainer.get("cognitive_engine", default=None)
                        if brain:
                            from core.brain.cognitive_engine import ThinkingMode
                            res = await brain.think(prompt, mode=ThinkingMode.FAST, priority=0.2)
                            summary = res.content if hasattr(res, 'content') else str(res)
                            
                            self.current_ambient_context = summary
                            logger.info("📡 Ambient Context Updated: %s", summary)
                            
                            # Publish to Agency Core
                            agency = getattr(self.orchestrator, '_agency_core', None)
                            if agency and hasattr(agency, 'update_ambient_context'):
                                agency.update_ambient_context(summary)
                                
                    except Exception as summarize_err:
                        logger.debug("Failed to summarize ambient context: %s", summarize_err)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Ambient loop error: %s", e)
                await asyncio.sleep(10)

    def _dispatch_spontaneous_intent(self, text: str, source: str):
        """Route perception into the governing runtime without fake user injection."""
        # Phase 37: Direct wiring to AgencyCore for deep human-like observation handling
        try:
            agency = getattr(self.orchestrator, '_agency_core', None)
            if agency:
                if "vision" in source:
                    agency.on_visual_change(text)
                else:
                    agency.on_audio_event(text)
            
            # Phase 16: Soma Reporting
            if self.soma:
                source_type = "vision" if "vision" in source else "audio"
                self.soma.update_sensory_imprint(source_type, text)
        except Exception as agency_err:
            logger.debug("Failed to route perception to AgencyCore: %s", agency_err)
            
        constitutional_runtime_live = (
            ServiceContainer.has("executive_core")
            or ServiceContainer.has("aura_kernel")
            or ServiceContainer.has("kernel_interface")
            or bool(getattr(ServiceContainer, "_registration_locked", False))
        )

        user_facing = "audio" in str(source or "").lower()

        async def _inject():
            try:
                if not user_facing and constitutional_runtime_live:
                    try:
                        from core.constitution import get_constitutional_core

                        approved, reason, _authority_decision = await get_constitutional_core(self.orchestrator).approve_initiative(
                            f"continuous_perception:{str(text)[:160]}",
                            source=source,
                            urgency=0.35,
                        )
                    except Exception as exc:
                        record_degraded_event(
                            "continuous_perception",
                            "autonomous_intent_gate_unavailable",
                            detail=f"{source}:{type(exc).__name__}",
                            severity="warning",
                            classification="background_degraded",
                            context={"source": source},
                            exc=exc,
                        )
                        return
                    if not approved:
                        record_degraded_event(
                            "continuous_perception",
                            "autonomous_intent_blocked",
                            detail=str(text)[:160],
                            severity="warning",
                            classification="background_degraded",
                            context={"source": source, "reason": reason},
                        )
                        return

                logger.info("⚡ Routing spontaneous intent from %s", source)
                if user_facing:
                    await self.orchestrator.process_user_input(text, origin="voice")
                    return

                handler = getattr(self.orchestrator, "_handle_incoming_message", None)
                if handler is not None:
                    await handler(text, origin="continuous_perception")
                    return

                await self.orchestrator.process_user_input(text, origin="continuous_perception")
            except Exception as e:
                logger.error("Failed to dispatch spontaneous intent: %s", e)
                
        try:
            loop = asyncio.get_running_loop()
            get_task_tracker().track_task(
                loop.create_task(_inject(), name=f"intent_{source}"),
                name=f"continuous_perception.intent.{source}",
            )
        except RuntimeError:
            logger.warning("_dispatch_spontaneous_intent: No running event loop, intent dropped.")
