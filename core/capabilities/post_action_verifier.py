"""core/capabilities/post_action_verifier.py — Step-by-Step Verification
========================================================================
This is what separates Aura from a brittle macro.

After EVERY action step, the verifier checks the actual state of the
world to confirm the action succeeded. If verification fails, the
RecoveryEngine is invoked to retry, fallback, or honestly report failure.

Verification predicates are concrete checks — not LLM opinion.
They use AppleScript queries, filesystem checks, OCR, and screenshot
comparison to produce evidence-backed VerificationResults.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.security.execution_authority import (
    KIND_SHELL,
    authorize_execution,
    release_execution,
)

logger = logging.getLogger("Aura.PostActionVerifier")


@dataclass
class VerificationResult:
    """Evidence-backed result of a post-action verification."""
    predicate: str
    args: Dict[str, Any]
    success: bool
    evidence: str = ""              # what was observed
    expected: str = ""              # what was expected
    screenshot_path: str = ""       # optional verification screenshot
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicate": self.predicate,
            "success": self.success,
            "evidence": self.evidence[:300],
            "expected": self.expected[:200],
            "screenshot": self.screenshot_path,
            "duration_ms": round(self.duration_ms, 1),
        }


class PostActionVerifier:
    """Verifies action outcomes against concrete predicates.

    Usage:
        verifier = get_post_action_verifier()
        result = await verifier.verify("app_is_frontmost", {"name": "Notes"})
        if not result.success:
            # invoke recovery
    """

    def __init__(self) -> None:
        self._verification_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("post_action_verifier", self, required=False)
        self._started = True
        logger.info("PostActionVerifier ONLINE")

    async def verify(self, predicate: str, args: Optional[Dict[str, Any]] = None) -> VerificationResult:
        """Execute a verification predicate and return evidence-backed result."""
        args = args or {}
        start = time.time()
        self._verification_count += 1

        try:
            handler = self._get_handler(predicate)
            if handler is None:
                result = VerificationResult(
                    predicate=predicate, args=args, success=False,
                    evidence=f"Unknown predicate: {predicate}",
                    duration_ms=(time.time() - start) * 1000,
                )
            else:
                result = await handler(args)
                result.predicate = predicate
                result.args = args
                result.duration_ms = (time.time() - start) * 1000
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            result = VerificationResult(
                predicate=predicate, args=args, success=False,
                evidence=f"Verification error: {e}",
                duration_ms=(time.time() - start) * 1000,
            )

        if result.success:
            self._success_count += 1
        else:
            self._failure_count += 1
            logger.info(
                "Verification FAILED: %s(%s) — evidence: %s",
                predicate, args, result.evidence[:100],
            )

        return result

    def _get_handler(self, predicate: str) -> Optional[Callable]:
        """Map predicate name to handler function."""
        handlers = {
            "app_is_frontmost": self._verify_app_frontmost,
            "app_is_running": self._verify_app_running,
            "file_exists": self._verify_file_exists,
            "file_has_content": self._verify_file_has_content,
            "folder_exists": self._verify_folder_exists,
            "file_is_pdf": self._verify_file_is_pdf,
            "file_is_image": self._verify_file_is_image,
            "file_in_folder": self._verify_file_in_folder,
            "window_title_contains": self._verify_window_title,
            "screen_contains_text": self._verify_screen_text,
            "wallpaper_is": self._verify_wallpaper,
            "wallpaper_changed": self._verify_wallpaper_changed,
            "browser_has_tabs": self._verify_browser_tabs,
            "clipboard_contains": self._verify_clipboard,
            "command_succeeded": self._verify_command,
            "true": self._verify_always_true,
        }
        return handlers.get(predicate)

    # ------------------------------------------------------------------
    # Verification predicates — all concrete, evidence-producing
    # ------------------------------------------------------------------

    async def _verify_app_frontmost(self, args: Dict[str, Any]) -> VerificationResult:
        """Check if an app is the currently frontmost application."""
        expected = str(args.get("name", ""))
        try:
            from core.capabilities.host_automation import get_host_automation
            receipt = await get_host_automation().get_frontmost_app()
            actual = str(receipt.result or "").strip()
            match = expected.lower() in actual.lower() if expected and actual else False
            return VerificationResult(
                predicate="app_is_frontmost", args=args,
                success=match,
                evidence=f"Frontmost app: '{actual}'",
                expected=f"Expected: '{expected}'",
            )
        except (ImportError, AttributeError, RuntimeError) as e:
            return VerificationResult(
                predicate="app_is_frontmost", args=args,
                success=False, evidence=f"Check failed: {e}",
                expected=expected,
            )

    async def _verify_app_running(self, args: Dict[str, Any]) -> VerificationResult:
        """Check if an app is currently running."""
        expected = str(args.get("name", ""))
        try:
            from core.capabilities.host_automation import get_host_automation
            receipt = await get_host_automation().get_running_apps()
            apps = receipt.result if isinstance(receipt.result, list) else []
            running = any(expected.lower() in str(a).lower() for a in apps)
            return VerificationResult(
                predicate="app_is_running", args=args,
                success=running,
                evidence=f"Running apps: {', '.join(str(a) for a in apps[:10])}",
                expected=f"Expected '{expected}' in running apps",
            )
        except (ImportError, AttributeError, RuntimeError) as e:
            return VerificationResult(
                predicate="app_is_running", args=args,
                success=False, evidence=f"Check failed: {e}",
            )

    async def _verify_file_exists(self, args: Dict[str, Any]) -> VerificationResult:
        """Check if a file exists and has content."""
        path = Path(str(args.get("path", ""))).expanduser()
        if not path.is_absolute():
            # Try common roots
            for root in [Path.home() / "Documents" / "Aura",
                         Path.home() / "Desktop" / "Aura",
                         Path.home() / "Downloads"]:
                candidate = root / path
                if candidate.exists():
                    path = candidate
                    break

        exists = path.exists() and path.is_file()
        size = path.stat().st_size if exists else 0
        return VerificationResult(
            predicate="file_exists", args=args,
            success=exists and size > 0,
            evidence=f"{'Exists' if exists else 'NOT FOUND'}, size={size} bytes" if exists else f"File not found: {path}",
            expected=f"File exists at {path}",
        )

    async def _verify_file_has_content(self, args: Dict[str, Any]) -> VerificationResult:
        """Check if a file contains expected content or matches expected hash."""
        path = Path(str(args.get("path", ""))).expanduser()
        expected_hash = str(args.get("hash", ""))
        expected_text = str(args.get("contains", ""))
        min_size = int(args.get("min_size", 1))

        if not path.exists():
            return VerificationResult(
                predicate="file_has_content", args=args,
                success=False, evidence=f"File not found: {path}",
            )

        size = path.stat().st_size
        if size < min_size:
            return VerificationResult(
                predicate="file_has_content", args=args,
                success=False, evidence=f"File too small: {size} < {min_size}",
            )

        if expected_hash:
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            match = actual_hash == expected_hash
            return VerificationResult(
                predicate="file_has_content", args=args,
                success=match,
                evidence=f"Hash: {actual_hash}" + (" (match)" if match else f" (expected {expected_hash})"),
            )

        if expected_text:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                found = expected_text.lower() in content.lower()
                return VerificationResult(
                    predicate="file_has_content", args=args,
                    success=found,
                    evidence=f"Content length: {len(content)}, contains '{expected_text[:50]}': {found}",
                )
            except (OSError, UnicodeDecodeError) as e:
                return VerificationResult(
                    predicate="file_has_content", args=args,
                    success=False, evidence=f"Read error: {e}",
                )

        # Just check it has content
        return VerificationResult(
            predicate="file_has_content", args=args,
            success=size > 0,
            evidence=f"File size: {size} bytes",
        )

    async def _verify_folder_exists(self, args: Dict[str, Any]) -> VerificationResult:
        """Check if a folder exists."""
        path = Path(str(args.get("path", ""))).expanduser()
        exists = path.exists() and path.is_dir()
        return VerificationResult(
            predicate="folder_exists", args=args,
            success=exists,
            evidence=f"{'Exists' if exists else 'NOT FOUND'}: {path}",
        )

    async def _verify_file_is_pdf(self, args: Dict[str, Any]) -> VerificationResult:
        """Check if a file is a valid PDF."""
        path = Path(str(args.get("path", ""))).expanduser()
        if not path.exists():
            return VerificationResult(
                predicate="file_is_pdf", args=args,
                success=False, evidence=f"File not found: {path}",
            )

        try:
            with open(path, "rb") as f:
                header = f.read(5)
            is_pdf = header == b"%PDF-"
            size = path.stat().st_size
            return VerificationResult(
                predicate="file_is_pdf", args=args,
                success=is_pdf and size > 50,
                evidence=f"Header: {header!r}, size: {size}",
                expected="Valid PDF file",
            )
        except OSError as e:
            return VerificationResult(
                predicate="file_is_pdf", args=args,
                success=False, evidence=f"Read error: {e}",
            )

    async def _verify_file_is_image(self, args: Dict[str, Any]) -> VerificationResult:
        """Check if a file is a valid image."""
        path = Path(str(args.get("path", ""))).expanduser()
        if not path.exists():
            return VerificationResult(
                predicate="file_is_image", args=args,
                success=False, evidence=f"File not found: {path}",
            )

        try:
            with open(path, "rb") as f:
                header = f.read(16)
            # Check magic bytes
            is_jpeg = header[:2] == b"\xff\xd8"
            is_png = header[:8] == b"\x89PNG\r\n\x1a\n"
            is_webp = header[8:12] == b"WEBP"
            is_gif = header[:4] == b"GIF8"
            is_image = is_jpeg or is_png or is_webp or is_gif
            fmt = "JPEG" if is_jpeg else "PNG" if is_png else "WebP" if is_webp else "GIF" if is_gif else "unknown"
            size = path.stat().st_size

            return VerificationResult(
                predicate="file_is_image", args=args,
                success=is_image and size > 100,
                evidence=f"Format: {fmt}, size: {size}",
                expected="Valid image file",
            )
        except OSError as e:
            return VerificationResult(
                predicate="file_is_image", args=args,
                success=False, evidence=f"Read error: {e}",
            )

    async def _verify_file_in_folder(self, args: Dict[str, Any]) -> VerificationResult:
        """Check if a file exists inside a specific folder."""
        file_name = str(args.get("file", ""))
        folder_path = Path(str(args.get("folder", ""))).expanduser()

        if not folder_path.exists():
            return VerificationResult(
                predicate="file_in_folder", args=args,
                success=False, evidence=f"Folder not found: {folder_path}",
            )

        # Look for the file in the folder
        found = False
        found_path = ""
        try:
            for entry in folder_path.iterdir():
                if file_name.lower() in entry.name.lower():
                    found = True
                    found_path = str(entry)
                    break
        except PermissionError:
            return VerificationResult(
                predicate="file_in_folder", args=args,
                success=False, evidence=f"Permission denied: {folder_path}",
            )

        return VerificationResult(
            predicate="file_in_folder", args=args,
            success=found,
            evidence=f"Found: {found_path}" if found else f"'{file_name}' not in {folder_path}",
        )

    async def _verify_window_title(self, args: Dict[str, Any]) -> VerificationResult:
        """Check if the current window title contains expected text."""
        expected = str(args.get("text", ""))
        app = str(args.get("app", ""))
        try:
            from core.capabilities.host_automation import get_host_automation
            receipt = await get_host_automation().get_window_title(app)
            actual = str(receipt.result or "").strip()
            match = expected.lower() in actual.lower() if expected and actual else False
            return VerificationResult(
                predicate="window_title_contains", args=args,
                success=match,
                evidence=f"Window title: '{actual}'",
                expected=f"Contains: '{expected}'",
            )
        except (ImportError, AttributeError, RuntimeError) as e:
            return VerificationResult(
                predicate="window_title_contains", args=args,
                success=False, evidence=f"Check failed: {e}",
            )

    async def _verify_screen_text(self, args: Dict[str, Any]) -> VerificationResult:
        """Check if the screen contains specific text (via OCR)."""
        expected = str(args.get("text", ""))
        try:
            from core.capabilities.host_automation import get_host_automation
            receipt = await get_host_automation().get_screen_text()
            screen_text = str(receipt.result or "")
            found = expected.lower() in screen_text.lower() if expected else False
            return VerificationResult(
                predicate="screen_contains_text", args=args,
                success=found,
                evidence=f"Screen text length: {len(screen_text)}, contains '{expected[:30]}': {found}",
                expected=f"Screen shows: '{expected[:50]}'",
            )
        except (ImportError, AttributeError, RuntimeError) as e:
            return VerificationResult(
                predicate="screen_contains_text", args=args,
                success=False, evidence=f"OCR failed: {e}",
            )

    async def _verify_wallpaper(self, args: Dict[str, Any]) -> VerificationResult:
        """Check if the wallpaper is set to a specific path."""
        expected_path = str(args.get("path", ""))
        try:
            proc = await get_subprocess_gateway().spawn_async(
                ["osascript", "-e", 'tell application "System Events" to get picture of current desktop'],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                read_only=True,
                source="post_action_verifier.wallpaper_is",
                accelerator_capability="none",
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
            actual = stdout.decode("utf-8", errors="replace").strip() if stdout else ""
            match = expected_path in actual if expected_path and actual else False
            return VerificationResult(
                predicate="wallpaper_is", args=args,
                success=match,
                evidence=f"Current wallpaper: {actual}",
                expected=f"Expected: {expected_path}",
            )
        except (OSError, asyncio.TimeoutError) as e:
            return VerificationResult(
                predicate="wallpaper_is", args=args,
                success=False, evidence=f"Check failed: {e}",
            )

    async def _verify_wallpaper_changed(self, args: Dict[str, Any]) -> VerificationResult:
        """Check if the wallpaper changed from a previous value."""
        previous_path = str(args.get("previous", ""))
        try:
            proc = await get_subprocess_gateway().spawn_async(
                ["osascript", "-e", 'tell application "System Events" to get picture of current desktop'],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                read_only=True,
                source="post_action_verifier.wallpaper_changed",
                accelerator_capability="none",
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
            actual = stdout.decode("utf-8", errors="replace").strip() if stdout else ""
            changed = actual != previous_path if previous_path and actual else bool(actual)
            return VerificationResult(
                predicate="wallpaper_changed", args=args,
                success=changed,
                evidence=f"Current: {actual}, previous: {previous_path}",
            )
        except (OSError, asyncio.TimeoutError) as e:
            return VerificationResult(
                predicate="wallpaper_changed", args=args,
                success=False, evidence=f"Check failed: {e}",
            )

    async def _verify_browser_tabs(self, args: Dict[str, Any]) -> VerificationResult:
        """Check if the browser has a minimum number of tabs."""
        min_count = int(args.get("min_count", 1))
        browser = str(args.get("browser", "Google Chrome"))
        try:
            script = f'tell application "{browser}" to count of tabs of front window'
            proc = await get_subprocess_gateway().spawn_async(
                ["osascript", "-e", script],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                read_only=True,
                source="post_action_verifier.browser_tabs",
                accelerator_capability="none",
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
            count = int(stdout.decode().strip()) if stdout else 0
            return VerificationResult(
                predicate="browser_has_tabs", args=args,
                success=count >= min_count,
                evidence=f"{browser} has {count} tab(s)",
                expected=f"At least {min_count} tab(s)",
            )
        except (OSError, asyncio.TimeoutError, ValueError) as e:
            return VerificationResult(
                predicate="browser_has_tabs", args=args,
                success=False, evidence=f"Check failed: {e}",
            )

    async def _verify_clipboard(self, args: Dict[str, Any]) -> VerificationResult:
        """Check if the clipboard contains expected text."""
        expected = str(args.get("text", ""))
        try:
            proc = await get_subprocess_gateway().spawn_async(
                ["pbpaste"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                read_only=True,
                source="post_action_verifier.clipboard_contains",
                accelerator_capability="none",
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
            content = stdout.decode("utf-8", errors="replace") if stdout else ""
            found = expected.lower() in content.lower() if expected else bool(content)
            return VerificationResult(
                predicate="clipboard_contains", args=args,
                success=found,
                evidence=f"Clipboard length: {len(content)}, contains target: {found}",
            )
        except (OSError, asyncio.TimeoutError) as e:
            return VerificationResult(
                predicate="clipboard_contains", args=args,
                success=False, evidence=f"Check failed: {e}",
            )

    async def _verify_command(self, args: Dict[str, Any]) -> VerificationResult:
        """Check if a command succeeds (return code 0).

        "Verify that this command succeeds" is arbitrary execution wearing a
        verification hat: the predicate runs whatever string it is handed,
        with the same reach as the terminal skill and none of its scrutiny.
        A check that can run anything is a capability, so it asks like one.
        """
        command = str(args.get("command", ""))
        if not command:
            return VerificationResult(
                predicate="command_succeeded", args=args,
                success=False, evidence="No command specified",
            )

        verdict = await authorize_execution(
            KIND_SHELL,
            command,
            source="tool_execution:post_action_verifier.command_succeeded",
            extra={"predicate": "command_succeeded"},
        )
        if not verdict.approved:
            return VerificationResult(
                predicate="command_succeeded", args=args,
                success=False, evidence=verdict.reason,
            )

        try:
            proc = await get_subprocess_gateway().spawn_shell_async(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                source="post_action_verifier.command_succeeded",
                accelerator_capability="auto",
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            success = proc.returncode == 0
            return VerificationResult(
                predicate="command_succeeded", args=args,
                success=success,
                evidence=f"Return code: {proc.returncode}" +
                         (f", stdout: {stdout.decode()[:100]}" if stdout else ""),
            )
        except (OSError, asyncio.TimeoutError) as e:
            return VerificationResult(
                predicate="command_succeeded", args=args,
                success=False, evidence=f"Command failed: {e}",
            )
        finally:
            release_execution(
                verdict, source="post_action_verifier.command_succeeded"
            )

    async def _verify_always_true(self, args: Dict[str, Any]) -> VerificationResult:
        """Always-true predicate for steps that don't need verification."""
        return VerificationResult(
            predicate="true", args=args, success=True,
            evidence="No verification needed",
        )

    # ------------------------------------------------------------------
    # Composite verification
    # ------------------------------------------------------------------

    async def verify_all(self, predicates: List[Tuple[str, Dict[str, Any]]]) -> List[VerificationResult]:
        """Run multiple verification predicates and return all results."""
        results = []
        for pred_name, pred_args in predicates:
            result = await self.verify(pred_name, pred_args)
            results.append(result)
        return results

    def get_status(self) -> Dict[str, Any]:
        return {
            "total_verifications": self._verification_count,
            "successes": self._success_count,
            "failures": self._failure_count,
            "success_rate": round(
                self._success_count / max(1, self._verification_count), 3
            ),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[PostActionVerifier] = None


def get_post_action_verifier() -> PostActionVerifier:
    global _instance
    if _instance is None:
        _instance = PostActionVerifier()
    return _instance


__all__ = [
    "PostActionVerifier",
    "VerificationResult",
    "get_post_action_verifier",
]
