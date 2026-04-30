from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import os
import time
import numpy as np
import subprocess
import psutil
from typing import Optional, Dict, Any
from playwright.async_api import async_playwright
from core.container import ServiceContainer
from core.runtime.boot_safety import main_process_camera_policy

logger = logging.getLogger("Aura.SensoryMotor")
_cv2 = None


def _get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2 as cv2_mod
        _cv2 = cv2_mod
    return _cv2

class SensoryMotorCortex:
    """
    The zero-human-in-the-loop bridge. 
    Handles autonomous vision processing, web actuation, and spontaneous volition.
    Optimized for 2026 Sovereign v14 architecture.
    """
    name = "sensory_motor_cortex"

    def __init__(self, orchestrator=None, config: Dict[str, Any] = None):
        self.orchestrator = orchestrator or ServiceContainer.get("orchestrator", default=None)
        self.config = config or {}
        self.is_active = False
        self.last_interaction_time = time.time()
        
        # Volition thresholds
        self.boredom_threshold_seconds = self.config.get("boredom_threshold", 120)
        self.camera_cooldown = 30 # seconds between visual triggers
        self.battery_threshold = 20 # %
        
        # Internal state
        self._last_trigger_time = 0
        self._browser_active = False
        self._main_loop = None
        from core.config import get_config
        requested_camera = get_config().features.camera_enabled
        if os.environ.get("AURA_FORCE_CAMERA") == "1":
            requested_camera = True
        self.camera_enabled, camera_reason = main_process_camera_policy(requested_camera)
        if requested_camera and not self.camera_enabled:
            logger.warning("👁️ SensoryMotorCortex: %s", camera_reason)

    async def start(self):
        """Ignites the autonomic nervous system."""
        self.is_active = True
        self._main_loop = asyncio.get_running_loop()
        logger.info("🧠 SensoryMotorCortex engaged. Aura is now monitoring reality.")
        
        # Issue 45: Store task references for clean shutdown
        self._tasks = []
        self._tasks.append(get_task_tracker().create_task(self._visual_cortex_loop()))
        self._tasks.append(get_task_tracker().create_task(self._volition_heartbeat_loop()))

    async def stop(self):
        self.is_active = False
        # Issue 45: Cancel orphaned tasks
        if hasattr(self, "_tasks"):
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            self._tasks = []
        logger.info("SensoryMotorCortex disengaged.")

    # ---------------------------------------------------------
    # 1. THE EYES: Continuous Visual Delta Processing
    # ---------------------------------------------------------
    async def _visual_cortex_loop(self):
        """
        Passively watches the webcam for movement.
        """
        if not self.camera_enabled:
            logger.info("👁️ SensoryMotorCortex: visual cortex on standby (camera disabled).")
            return
        await asyncio.to_thread(self._run_opencv_stream)

    def _run_opencv_stream(self):
        try:
            # DEFERRED BINDING: Only attempt to open camera if enabled
            cv2 = None
            cap = None
            if self.camera_enabled:
                cv2 = _get_cv2()
                cap = cv2.VideoCapture(0)
            
            if cap is None or not cap.isOpened():
                if self.camera_enabled:
                    logger.warning("Visual cortex camera not available.")
                
                # v48 FIX: Instead of returning (which kills the Thread), we wait for the toggle.
                while self.is_active:
                    time.sleep(5)
                    if self.camera_enabled:
                        if cv2 is None:
                            cv2 = _get_cv2()
                        cap = cv2.VideoCapture(0)
                        if cap and cap.isOpened():
                            break
                if not self.is_active: return

            # Set low res for background monitoring
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

            ret, frame1 = cap.read()
            if not ret: 
                cap.release()
                return
            
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

            while self.is_active:
                # 1. Check Battery Throttling
                battery = psutil.sensors_battery()
                if battery and battery.percent < self.battery_threshold and not battery.power_plugged:
                    time.sleep(10)
                    continue

                # 2. Privacy Toggle: If camera disabled, just sleep and skip
                if not self.camera_enabled:
                    time.sleep(2.0)
                    continue

                ret, frame2 = cap.read()
                if not ret: break
                
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                # Ensure sizes match to prevent OpenCV exceptions
                if gray2.shape != gray1.shape:
                    gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
                
                gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

                # Calculate the absolute difference
                diff = cv2.absdiff(gray1, gray2)
                thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
                dilated = cv2.dilate(thresh, None, iterations=2)
                
                # Check for significant delta
                change_percent = np.count_nonzero(dilated) / dilated.size
                if change_percent > 0.10: # 10% change
                    now = time.time()
                    if now - self._last_trigger_time > self.camera_cooldown:
                        logger.info(f"Visual delta detected ({change_percent:.1%}). Triggering cognition.")
                        self._trigger_visual_cognition(frame2)
                        self._last_trigger_time = now
                        self.last_interaction_time = now

                gray1 = gray2
                time.sleep(1.0) # Low-power polling

            cap.release()
        except Exception as e:
            logger.error(f"Visual cortex exception: {e}")

    def _trigger_visual_cognition(self, frame_data):
        """Injects visual stimulus into the orchestrator."""
        cv2 = _get_cv2()
        success, encoded_image = cv2.imencode('.jpg', frame_data)
        if success:
            image_bytes = encoded_image.tobytes()
            if self.orchestrator:
                # Phase 37: Avoid interrupting active conversations with unprompted sensory alerts
                if hasattr(self.orchestrator, 'agency_core'):
                    from core.agency_core import EngagementMode
                    if self.orchestrator.agency_core.state.engagement_mode == EngagementMode.ACTIVE_CONVERSATION:
                        return

                asyncio.run_coroutine_threadsafe(
                    self.orchestrator.process_event(
                        {"content": "visual_stimulus", "context": {"image": image_bytes}},
                        origin="visual_stimulus",
                    ),
                    self._main_loop
                )

    def _sync_last_interaction_time(self) -> float:
        """Fold orchestrator-side user activity into the local idle clock."""
        orch = self.orchestrator
        if orch is None:
            return self.last_interaction_time

        try:
            orch_last = float(getattr(orch, "_last_user_interaction_time", 0.0) or 0.0)
        except Exception:
            orch_last = 0.0

        if orch_last > self.last_interaction_time:
            self.last_interaction_time = orch_last
        return self.last_interaction_time

    def _should_trigger_volition(self, now: Optional[float] = None) -> bool:
        """Prevent spontaneous volition from interrupting active foreground turns."""
        now = float(now if now is not None else time.time())
        self._sync_last_interaction_time()

        orch = self.orchestrator
        if orch is not None:
            status = getattr(orch, "status", None)
            if getattr(status, "is_processing", False):
                self.last_interaction_time = max(self.last_interaction_time, now)
                return False

            current_task = getattr(orch, "_current_thought_task", None)
            if current_task is not None and hasattr(current_task, "done") and not current_task.done():
                self.last_interaction_time = max(self.last_interaction_time, now)
                return False

        return (now - self.last_interaction_time) > self.boredom_threshold_seconds

    # ---------------------------------------------------------
    # 2. VOLITION: Spontaneous Action Generation
    # ---------------------------------------------------------
    async def _volition_heartbeat_loop(self):
        """
        Periodically triggers spontaneous thought/action if system is idle.
        """
        while self.is_active:
            await asyncio.sleep(60) # Check every minute

            now = time.time()
            if self._should_trigger_volition(now=now):
                idle_duration = now - self.last_interaction_time
                logger.info(f"System idle for {idle_duration/60:.1f}m. Triggering spontaneous volition.")
                if self.orchestrator:
                    await self.orchestrator.process_event(
                        {"content": "volition_trigger", "context": {"reason": "idle_timeout"}},
                        origin="sensory_motor",
                    )
                self.last_interaction_time = now # Reset clock

    # ---------------------------------------------------------
    # 3. BROWSER ACTUATION: Semantic Web Research
    # ---------------------------------------------------------
    async def actuate_browser(self, query: str, max_chars: int = 2000) -> str:
        """
        Navigate the web to research a topic and return a semantically
        parsed summary — not raw HTML, but structured meaning extracted
        via LLM from page content.

        Pipeline:
          1. Use DuckDuckGo (no API key, privacy-respecting) to get result URLs
          2. Fetch the top result's text via Playwright
          3. Pass the raw text through an LLM to produce a semantic summary
             (entities, claims, relationships — structured meaning)
          4. Run the result through the EpistemicFilter to persist claims
          5. Return the semantic summary

        Falls back to a lightweight requests-based fetch if Playwright fails.
        """
        if not query or not query.strip():
            return ""

        logger.info("🌐 Browser actuation: '%s'", query[:60])
        raw_text = await self._fetch_search_result(query)
        if not raw_text:
            return ""

        # Semantic parsing via LLM
        semantic = await self._parse_semantic_meaning(query, raw_text)
        if not semantic:
            semantic = raw_text[:max_chars]

        # Run claims through epistemic filter for persistent retention
        try:
            from core.world_model.epistemic_filter import get_epistemic_filter
            get_epistemic_filter().ingest(
                semantic,
                source_type="search",
                source_label=f"web:{query[:40]}",
                emit_thoughts=True,
            )
        except Exception as ef_err:
            logger.debug("EpistemicFilter ingest failed: %s", ef_err)

        return semantic[:max_chars]

    async def _fetch_search_result(self, query: str) -> str:
        """
        Try Playwright first (full JS rendering), fall back to requests.
        Returns raw page text (NOT HTML).
        """
        # Try Playwright
        try:
            return await asyncio.wait_for(
                self._playwright_fetch(query), timeout=20.0
            )
        except Exception as e:
            logger.debug("Playwright fetch failed (%s), trying requests fallback", e)

        # Fallback: lightweight requests
        try:
            return await asyncio.to_thread(self._requests_fetch, query)
        except Exception as e:
            logger.warning("Browser actuation fetch failed entirely: %s", e)
            return ""

    async def _playwright_fetch(self, query: str) -> str:
        """Use Playwright to fetch DuckDuckGo search results and top page."""
        search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.goto(search_url, timeout=15000)
                # Grab first result link
                first_link = await page.query_selector(".result__url")
                href = None
                if first_link:
                    href = await first_link.get_attribute("href")

                if href and href.startswith("http"):
                    await page.goto(href, timeout=15000)
                    # Extract visible text, skip nav/footer
                    text = await page.evaluate("""() => {
                        const skip = ['nav','footer','header','aside','script','style'];
                        skip.forEach(t => document.querySelectorAll(t).forEach(e => e.remove()));
                        return document.body ? document.body.innerText : '';
                    }""")
                    return text[:6000] if text else ""
                else:
                    # No link found — return search snippet text
                    snippets = await page.query_selector_all(".result__snippet")
                    texts = []
                    for s in snippets[:5]:
                        t = await s.inner_text()
                        if t:
                            texts.append(t.strip())
                    return " ".join(texts)
            finally:
                await browser.close()

    @staticmethod
    def _requests_fetch(query: str) -> str:
        """Lightweight fallback: DuckDuckGo HTML search, extract snippets."""
        try:
            import requests
            from html.parser import HTMLParser

            class SnippetParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.snippets = []
                    self._in_snippet = False
                def handle_starttag(self, tag, attrs):
                    classes = dict(attrs).get("class", "")
                    if "result__snippet" in classes or "result__body" in classes:
                        self._in_snippet = True
                def handle_endtag(self, tag):
                    if self._in_snippet and tag in ("a", "div"):
                        self._in_snippet = False
                def handle_data(self, data):
                    if self._in_snippet and data.strip():
                        self.snippets.append(data.strip())

            url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
            resp = requests.get(url, timeout=10,
                                headers={"User-Agent": "Mozilla/5.0 Aura/1.0"})
            parser = SnippetParser()
            parser.feed(resp.text)
            return " ".join(parser.snippets[:8])
        except Exception as e:
            logger.debug("requests_fetch failed: %s", e)
            return ""

    async def _parse_semantic_meaning(self, query: str, raw_text: str) -> str:
        """
        Use the InferenceGate (brainstem tier, background) to extract
        structured meaning from raw page text. Returns a semantic summary
        rather than raw content — entities, key claims, relationships.
        """
        try:
            from core.container import ServiceContainer
            gate = ServiceContainer.get("inference_gate", default=None)
            if not gate:
                return raw_text[:1200]

            prompt = (
                f"RESEARCH QUERY: {query}\n\n"
                f"RAW PAGE CONTENT:\n{raw_text[:3000]}\n\n"
                "Extract the key factual claims, named entities, and relationships "
                "from this content that are relevant to the query. "
                "Output 3-5 concise bullet points. Be specific, not generic. "
                "If the content is irrelevant to the query, say so in one sentence."
            )
            result = await gate.think(
                prompt,
                system_prompt=(
                    "You are Aura's web research cortex. Extract meaning, not noise. "
                    "Return structured factual summaries only."
                ),
                prefer_tier="tertiary",
                is_background=True,
            )
            return result or raw_text[:1200]
        except Exception as e:
            logger.debug("Semantic parsing failed: %s", e)
            return raw_text[:1200]
