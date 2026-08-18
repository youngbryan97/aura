"""core/perception/screen_perception.py — Visual Desktop Perception
=====================================================================
Gives Aura eyes good enough to recover when the UI changes.

Can answer: "What app is open? Did the file appear? Did the Google Doc
load? Is the note written? Did the wallpaper change?"

Uses screencapture + OCR (pytesseract or macOS Vision) for text,
and AppleScript for structured queries (app name, window title).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.ScreenPerception")


@dataclass
class ScreenSnapshot:
    """A structured snapshot of the current screen state."""
    active_app: str = ""
    window_title: str = ""
    frontmost_window_bounds: str = ""
    focused_role: str = ""
    focused_name: str = ""
    focused_description: str = ""
    focused_value: str = ""
    accessibility_text: str = ""    # frontmost app accessibility tree text
    screen_text: str = ""           # OCR text
    #: Why screen_text is what it is. An empty screen_text meant two entirely
    #: different things and said neither: "there are no words on this screen"
    #: and "the pixels were never read". Through the whole life of the
    #: governance defect that blocked take_screenshot, OCR never ran once, and
    #: every consumer of screen_text saw "" and could only conclude the screen
    #: was blank. Absence of a reading is not a reading of absence.
    screen_text_status: str = "not_attempted"
    #: Every window on the screen, front to back, with geometry and how much
    #: of each survives the windows above it. The fields above describe the
    #: ONE window that happens to be in front; a person sees the whole desk.
    #: Without this, "is Notes open" and "can I see Notes" are the same
    #: unanswerable question — see core/perception/screen_blueprint.py.
    window_layout: str = ""
    open_apps: tuple[str, ...] = ()
    text_hash: str = ""             # for change detection
    screenshot_path: str = ""       # saved screenshot file
    has_modal: bool = False         # dialog/alert detected
    modal_text: str = ""
    has_loading: bool = False       # spinner/progress bar detected
    timestamp: float = field(default_factory=time.time)
    capture_denied: bool = False
    unavailable_reason: str = ""
    capture_admission: dict[str, str | bool] = field(default_factory=dict)


class ScreenPerception:
    """Visual perception of the desktop.

    Usage:
        perc = get_screen_perception()
        snap = await perc.capture()
        print(snap.active_app, snap.window_title)

        found = await perc.find_text_on_screen("Save")
        changed = await perc.detect_change(previous_hash)
    """

    def __init__(self) -> None:
        self._last_hash: str = ""
        self._capture_count: int = 0
        self._started = False

    @staticmethod
    def _prepare_screenshot_path(capture_count: int) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_dir = state_root() / "data" / "screenshots"
        save_dir.mkdir(parents=True, exist_ok=True)
        return str(save_dir / f"screen_{ts}_{capture_count}.png")

    @staticmethod
    def _path_exists(path: str) -> bool:
        return Path(path).exists()

    @staticmethod
    def _compare_screenshot_files(before_path: str, after_path: str) -> dict[str, Any]:
        before = Path(before_path)
        after = Path(after_path)

        if not before.exists() or not after.exists():
            return {"error": "Screenshot not found", "change_magnitude": 1.0}

        before_bytes = before.read_bytes()
        after_bytes = after.read_bytes()
        before_hash = hashlib.sha256(before_bytes).hexdigest()
        after_hash = hashlib.sha256(after_bytes).hexdigest()

        if before_hash == after_hash:
            return {"change_magnitude": 0.0, "identical": True}

        before_size = len(before_bytes)
        after_size = len(after_bytes)
        size_diff = abs(before_size - after_size)
        magnitude = min(1.0, size_diff / max(1, max(before_size, after_size)))

        return {
            "change_magnitude": magnitude,
            "identical": False,
            "before_size": before_size,
            "after_size": after_size,
        }

    @staticmethod
    def _ocr_screenshot_sync(screenshot_path: str) -> str:
        if not screenshot_path or not Path(screenshot_path).exists():
            return ""

        # macOS Vision FIRST. It is native, always present on this host, more
        # accurate than the alternative, and the only one of the two that also
        # reports where each run of text sat — see
        # HostAutomationProvider._ocr_image_regions.
        #
        # The order used to be reversed, with pytesseract preferred and Vision
        # as its fallback. pytesseract is an optional third-party install, so
        # in practice the first branch raised ImportError and Vision ran
        # anyway — meaning the ordering only ever mattered on a machine where
        # someone HAD installed it, where it silently downgraded perception.
        try:
            text = ScreenPerception._ocr_screenshot_with_macos_vision(screenshot_path)
            if text and text.strip():
                return text.strip()
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("macOS Vision OCR failed for %s: %s", screenshot_path, exc)

        try:
            import pytesseract
            from PIL import Image

            with Image.open(screenshot_path) as img:
                return str(pytesseract.image_to_string(img) or "").strip()
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("pytesseract OCR failed for %s: %s", screenshot_path, exc)

        # Return NOTHING rather than a sentence about OCR.
        #
        # This used to return "[OCR not available — install pytesseract or
        # enable macOS Vision OCR]", and that string was assigned to
        # snap.screen_text — the field holding what is ON the screen. Every
        # consumer downstream reads that field as screen content, so a failed
        # read presented as a screen that literally says OCR is unavailable.
        # She could then report it as something she saw.
        #
        # screen_text_status exists for precisely this and already says
        # read / read_empty / not_attempted: "Absence of a reading is not a
        # reading of absence." The reason belongs there, not in the words.
        return ""

    @staticmethod
    def _ocr_screenshot_with_macos_vision(screenshot_path: str) -> str:
        import threading

        import Foundation
        import Vision

        done = threading.Event()
        lines: list[str] = []
        errors: list[str] = []

        def _completion(request: Any, error: Any) -> None:
            if error is not None:
                errors.append(str(error))
            try:
                for observation in request.results() or []:
                    candidates = observation.topCandidates_(1)
                    if candidates:
                        text = str(candidates[0].string() or "").strip()
                        if text:
                            lines.append(text)
            except (AttributeError, TypeError, ValueError) as exc:
                errors.append(str(exc))
            finally:
                done.set()

        request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(_completion)
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)
        url = Foundation.NSURL.fileURLWithPath_(str(screenshot_path))
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
        ok, error = handler.performRequests_error_([request], None)
        if not ok:
            raise RuntimeError(str(error or "Vision OCR request failed"))
        if not done.wait(12.0):
            raise TimeoutError("Vision OCR timed out")
        if errors and not lines:
            raise RuntimeError("; ".join(errors))
        return "\n".join(lines)

    @staticmethod
    def _bounded_text(value: str, limit: int = 6000) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        half = max(1, limit // 2)
        return text[:half] + "\n... [TRUNCATED] ...\n" + text[-half:]

    async def _run_osascript(
        self,
        script: str,
        *,
        source: str,
        timeout_s: float = 3.0,
    ) -> str:
        """Run a bounded, read-only AppleScript probe and always reap it."""

        proc = None
        try:
            proc = await get_subprocess_gateway().spawn_async(
                ["osascript", "-e", script],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                read_only=True,
                source=source,
                accelerator_capability="none",
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            if proc.returncode == 0 and stdout:
                return stdout.decode("utf-8", errors="replace").strip()
            return ""
        except TimeoutError as exc:
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except (TimeoutError, OSError, RuntimeError) as kill_exc:
                    reaper_registered = False
                    try:
                        from core.reaper import register_reaper_pid

                        pid = int(getattr(proc, "pid", 0) or 0)
                        if pid > 0:
                            register_reaper_pid(pid)
                            reaper_registered = True
                    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as reaper_exc:
                        logger.error(
                            "Failed to register timed-out AppleScript child for reaping: %s",
                            reaper_exc,
                        )
                    reap_action = (
                        "registered the timed-out child PID with Aura's process reaper "
                        "for supervised cleanup"
                        if reaper_registered
                        else "reported the unreaped child explicitly after kill and reaper registration failed"
                    )
                    record_degradation(
                        f"{source}.reap",
                        kill_exc,
                        severity="degraded",
                        action=reap_action,
                    )
            record_degradation(source, exc)
            return ""
        except (OSError, RuntimeError) as exc:
            record_degradation(source, exc)
            return ""

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("screen_perception", self, required=False)
        self._started = True
        logger.info("ScreenPerception ONLINE")

    async def capture(self, save_screenshot: bool = False) -> ScreenSnapshot:
        """Capture a full snapshot of the current screen state."""
        snap = ScreenSnapshot()
        self._capture_count += 1

        from core.security.screen_capture_policy import (
            evaluate_screen_capture_admission_async,
        )

        admission = await evaluate_screen_capture_admission_async()
        snap.capture_admission = admission.to_receipt()
        if not admission.allowed:
            snap.capture_denied = True
            snap.unavailable_reason = admission.public_error
            return snap

        if save_screenshot:
            active = await self.get_active_window()
            summary = {
                "active_app": active.get("app", ""),
                "window_title": active.get("title", ""),
                "frontmost_window_bounds": active.get("bounds", ""),
                "focused_role": "",
                "focused_name": "",
                "focused_description": "",
                "focused_value": "",
                "accessibility_text": "",
            }
        else:
            summary = await self._frontmost_accessibility_summary()
        snap.active_app = summary.get("active_app", "")
        snap.window_title = summary.get("window_title", "")[:200]
        snap.frontmost_window_bounds = summary.get("frontmost_window_bounds", "")
        snap.focused_role = summary.get("focused_role", "")
        snap.focused_name = summary.get("focused_name", "")
        snap.focused_description = summary.get("focused_description", "")
        snap.focused_value = summary.get("focused_value", "")
        snap.accessibility_text = self._bounded_text(summary.get("accessibility_text", ""))

        # The whole desk, not just the window on top of it. Cheap enough
        # (in-process, cached for a moment) to belong on every snapshot, and
        # it is what makes "Chrome is covering your note" sayable at all.
        try:
            from core.perception.screen_blueprint import capture_blueprint

            blueprint = await asyncio.to_thread(capture_blueprint)
            if not blueprint.unavailable:
                snap.window_layout = blueprint.describe()
                snap.open_apps = blueprint.apps
                if not snap.active_app and blueprint.frontmost_app:
                    snap.active_app = blueprint.frontmost_app
        except Exception as exc:  # noqa: BLE001 - a snapshot may never fail on extras
            logger.debug("Screen blueprint unavailable for snapshot: %s", exc)

        # Take screenshot
        if save_screenshot:
            snap.screenshot_path = await self._take_screenshot()

        # OCR (if screenshot was taken or we need text)
        if snap.screenshot_path:
            snap.screen_text = await self._ocr_screenshot(snap.screenshot_path)
            snap.screen_text_status = "read" if snap.screen_text else "read_empty"
        elif save_screenshot:
            # A capture was asked for and did not arrive. That is the case
            # that spent this defect's lifetime masquerading as a blank screen.
            snap.screen_text_status = (
                f"unreadable:{snap.unavailable_reason}"
                if snap.unavailable_reason
                else "unreadable:capture_failed"
            )
        else:
            snap.screen_text_status = "not_attempted"
        if snap.screen_text:
            snap.text_hash = hashlib.sha256(snap.screen_text.encode()).hexdigest()[:16]
        elif snap.accessibility_text:
            snap.text_hash = hashlib.sha256(snap.accessibility_text.encode()).hexdigest()[:16]

        # Modal/loading detection from window title heuristics
        title_lower = snap.window_title.lower()
        snap.has_modal = any(
            kw in title_lower for kw in ("alert", "error", "warning", "permission", "allow")
        )
        snap.has_loading = any(
            kw in title_lower for kw in ("loading", "saving", "progress", "processing")
        )

        self._last_hash = snap.text_hash
        return snap

    def capture_sync(self, save_screenshot: bool = False) -> ScreenSnapshot:
        """Synchronous wrapper for tool fallbacks running outside the event loop."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.capture(save_screenshot=save_screenshot))
        raise RuntimeError("capture_sync cannot run inside an active event loop; await capture() instead")

    async def _frontmost_accessibility_summary(self) -> dict[str, str]:
        """Return frontmost app, window, focus, bounds, and accessibility text."""

        output = await self._run_osascript(
            r'''tell application "System Events"
    set frontApp to first application process whose frontmost is true
    set appName to name of frontApp
    set winTitle to ""
    set boundsText to ""
    set focusRole to ""
    set focusName to ""
    set focusDescription to ""
    set focusValue to ""
    set contentsText to ""
    try
        set winTitle to name of front window of frontApp
    end try
    try
        tell front window of frontApp
            set winPosition to position
            set winSize to size
            set boundsText to ((item 1 of winPosition) as string) & "," & ((item 2 of winPosition) as string) & "," & ((item 1 of winSize) as string) & "," & ((item 2 of winSize) as string)
        end tell
    end try
    try
        set focusedElement to first UI element of frontApp whose focused is true
        try
            set focusRole to role of focusedElement as string
        end try
        try
            set focusName to name of focusedElement as string
        end try
        try
            set focusDescription to description of focusedElement as string
        end try
        try
            set focusValue to value of focusedElement as string
        end try
    end try
    try
        set contentsText to entire contents of frontApp as string
    end try
    return appName & linefeed & winTitle & linefeed & boundsText & linefeed & focusRole & linefeed & focusName & linefeed & focusDescription & linefeed & focusValue & linefeed & contentsText
end tell''',
            source="screen_perception.accessibility_summary",
            timeout_s=3.0,
        )
        parts = output.split("\n", 7) if output else []
        parts += [""] * max(0, 8 - len(parts))
        return {
            "active_app": parts[0].strip(),
            "window_title": parts[1].strip(),
            "frontmost_window_bounds": parts[2].strip(),
            "focused_role": parts[3].strip(),
            "focused_name": parts[4].strip(),
            "focused_description": parts[5].strip(),
            "focused_value": parts[6].strip(),
            "accessibility_text": parts[7].strip(),
        }

    async def get_active_window(self) -> dict[str, str]:
        """Get just the active window info (fast, no screenshot)."""
        result = {"app": "", "title": "", "bounds": ""}
        output = await self._run_osascript(
            '''tell application "System Events"
                    set frontApp to name of first application process whose frontmost is true
                    set winTitle to name of front window of process frontApp
                    return frontApp & "|" & winTitle
                end tell''',
            source="screen_perception.active_window",
        )
        if output:
            parts = output.split("|", 1)
            result["app"] = parts[0] if parts else ""
            result["title"] = parts[1] if len(parts) > 1 else ""
        return result

    async def find_text_on_screen(self, target: str) -> dict[str, Any]:
        """Check if specific text appears on screen (via OCR)."""
        screenshot = await self._take_screenshot()
        if not screenshot:
            return {"found": False, "error": "Screenshot failed"}

        text = await self._ocr_screenshot(screenshot)
        found = target.lower() in text.lower() if target and text else False
        return {
            "found": found,
            "text_length": len(text),
            "screenshot": screenshot,
        }

    async def detect_change(self, previous_hash: str = "") -> dict[str, Any]:
        """Detect if the screen content changed since a previous capture."""
        if not previous_hash:
            previous_hash = self._last_hash

        screenshot = await self._take_screenshot()
        text = await self._ocr_screenshot(screenshot) if screenshot else ""
        current_hash = hashlib.sha256(text.encode()).hexdigest()[:16] if text else ""

        changed = current_hash != previous_hash if previous_hash and current_hash else True
        self._last_hash = current_hash

        return {
            "changed": changed,
            "previous_hash": previous_hash,
            "current_hash": current_hash,
        }

    async def compare_screenshots(
        self, before_path: str, after_path: str
    ) -> dict[str, Any]:
        """Compare two screenshots for differences."""
        return await asyncio.to_thread(
            self._compare_screenshot_files,
            before_path,
            after_path,
        )

    async def _take_screenshot(self) -> str:
        """Take a screenshot and return the file path."""
        from core.security.screen_capture_policy import (
            evaluate_screen_capture_admission_async,
        )

        admission = await evaluate_screen_capture_admission_async()
        if not admission.allowed:
            return ""
        save_path = await asyncio.to_thread(
            self._prepare_screenshot_path,
            self._capture_count,
        )

        try:
            proc = await get_subprocess_gateway().spawn_async(
                ["screencapture", "-x", save_path],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                source="screen_perception.take_screenshot",
                accelerator_capability="none",
            )
            await asyncio.wait_for(proc.communicate(), timeout=5.0)
            if proc.returncode == 0 and await asyncio.to_thread(self._path_exists, save_path):
                return save_path
        except (TimeoutError, OSError, RuntimeError) as exc:
            record_degradation("screen_perception.take_screenshot", exc)
        return ""

    async def _ocr_screenshot(self, screenshot_path: str) -> str:
        """Extract text from a screenshot via OCR."""
        try:
            return await asyncio.to_thread(self._ocr_screenshot_sync, screenshot_path)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as e:
            record_degradation("screen_perception.ocr", e)
            logger.debug("pytesseract OCR failed: %s", e)

        # Same rule as the sync reader below it: an unreadable screen produces
        # no words, not a sentence about why. This string was assigned to
        # snap.screen_text, the field describing what is ON the screen, so a
        # failed read became a screen that says "[OCR not available]" and could
        # be reported as something she saw. screen_text_status carries the
        # reason; screen_text carries only what was read.
        return ""

    def get_status(self) -> dict[str, Any]:
        return {
            "running": self._started,
            "started": self._started,
            "captures": self._capture_count,
            "last_hash": self._last_hash,
        }


_instance: ScreenPerception | None = None


def get_screen_perception() -> ScreenPerception:
    global _instance
    if _instance is None:
        _instance = ScreenPerception()
    return _instance


__all__ = ["ScreenPerception", "ScreenSnapshot", "get_screen_perception"]
