"""Taking a picture of the screen, and turning it back into words.

A screenshot she keeps is evidence; a screenshot she keeps forever is a
disclosure risk sitting on disk, so the retention sweep is here beside the
capture rather than somewhere that might not run. The OCR paths are the other
half: text with the regions it came from, so a later claim about what was on
screen can say where on screen it was.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only; that module imports this one
    from .host_automation import AutomationReceipt

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from core.governance.will import ActionDomain
from core.governance_context import local_internal_governed_scope
from core.runtime.action_executor import ActionExecutor
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root
from core.runtime.subprocess_gateway import get_subprocess_gateway


class _ReadsTheScreen:
    """Lifted whole from HostAutomationProvider; see host_automation.py."""

    @staticmethod
    def _retention_limit(name: str, default: int, minimum: int) -> int:
        try:
            return max(minimum, int(os.getenv(name, str(default)) or str(default)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _screenshot_retention_candidates(
        directory: Path,
        keep_path: Path | None,
    ) -> list[tuple[Path, os.stat_result, bool]]:
        candidates: list[tuple[Path, os.stat_result, bool]] = []
        resolved_keep = keep_path.resolve() if keep_path is not None else None
        try:
            for path in directory.iterdir():
                if path.suffix.lower() != ".png" or not path.is_file():
                    continue
                try:
                    candidates.append(
                        (
                            path,
                            path.stat(),
                            resolved_keep is not None and path.resolve() == resolved_keep,
                        )
                    )
                except OSError:
                    continue
        except OSError:
            return []
        candidates.sort(key=lambda item: item[1].st_mtime, reverse=True)
        return candidates

    @classmethod
    async def _enforce_screenshot_retention(
        cls,
        directory: Path,
        *,
        keep_path: Path | None = None,
    ) -> dict[str, int]:
        """Bound retained captures by age, count, and total bytes.

        The ephemeral directory gets its own, far tighter budget. A retained
        screenshot is something Aura is meant to still have; an ephemeral one
        is deleted immediately after its OCR, so ANY file at rest there is the
        residue of a failure. Sharing the 200-file retained budget meant 67
        orphans from a few hours of refused cleanups — 112MB of full-screen
        captures — sat inside the limit and were never reclaimed.
        """
        ephemeral = directory.name == "ephemeral"
        max_files = cls._retention_limit(
            "AURA_SCREENSHOT_RETENTION_MAX_FILES", 4 if ephemeral else 200, 1
        )
        if ephemeral:
            max_age_seconds = cls._retention_limit(
                "AURA_EPHEMERAL_SCREENSHOT_RETENTION_MAX_SECONDS", 300, 30
            )
        else:
            max_age_seconds = (
                cls._retention_limit("AURA_SCREENSHOT_RETENTION_MAX_DAYS", 14, 1)
                * 86400
            )
        max_bytes = cls._retention_limit(
            "AURA_SCREENSHOT_RETENTION_MAX_BYTES",
            (16 if ephemeral else 512) * 1024 * 1024,
            8 * 1024 * 1024,
        )
        cutoff = time.time() - max_age_seconds
        candidates = await asyncio.to_thread(
            cls._screenshot_retention_candidates,
            directory,
            keep_path,
        )

        kept = 0
        kept_bytes = 0
        deleted = 0
        bytes_deleted = 0
        for path, stat_result, is_keep in candidates:
            expired = stat_result.st_mtime < cutoff
            over_count = kept >= max_files
            over_bytes = kept_bytes + stat_result.st_size > max_bytes
            if not is_keep and (expired or over_count or over_bytes):
                try:
                    # Same missing-scope defect as the capture directory: with
                    # no declared scope the Will refused every deletion, so
                    # retention silently kept everything and the capture
                    # directory grew without bound.
                    with local_internal_governed_scope(
                        "host_automation.screenshot_retention",
                        domain=ActionDomain.FILE_WRITE.value,
                        constraints={"path": str(path), "op": "delete"},
                    ):
                        deletion = await ActionExecutor.execute(
                            domain=ActionDomain.FILE_WRITE,
                            action_name="host_automation.screenshot_retention_delete",
                            params={"path": str(path), "op": "delete"},
                            source="host_automation.screenshot_retention",
                        )
                    if cls._action_completed(deletion):
                        deleted += 1
                        bytes_deleted += stat_result.st_size
                        continue
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    record_degradation(
                        "host_automation.screenshot_retention",
                        exc,
                        action="retained a screenshot after governed retention cleanup failed",
                        severity="warning",
                    )
            kept += 1
            kept_bytes += stat_result.st_size
        return {"kept": kept, "deleted": deleted, "bytes_deleted": bytes_deleted}

    async def take_screenshot(
        self,
        save_path: str = "",
        region: tuple[int, int, int, int] | None = None,
        *,
        retain_capture: bool = True,
    ) -> AutomationReceipt:
        """Take a screenshot and optionally save to path.

        Args:
            save_path: Where to save (auto-generated if empty).
            region: Optional (x, y, w, h) to capture a region.

        Returns:
            Receipt with save_path as result.
        """
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .host_automation import (
            AutomationReceipt,
            _resolve_screenshot_path_policy,
            logger,
        )

        start = time.time()
        from core.security.screen_capture_policy import (
            evaluate_screen_capture_admission_async,
        )

        admission = await evaluate_screen_capture_admission_async()
        if not admission.allowed:
            return AutomationReceipt(
                action="take_screenshot",
                target="",
                adapter="screen_capture_policy",
                success=False,
                error=admission.public_error,
                duration_ms=(time.time() - start) * 1000,
                evidence={"capture_admission": admission.to_receipt()},
            )
        if not save_path:
            ts = time.strftime("%Y%m%d_%H%M%S")
            unique = f"{time.time_ns() % 1_000_000_000:09d}"
            folder = "screenshots" if retain_capture else "ephemeral"
            save_dir = state_root() / "data" / folder
            # Creating Aura's OWN capture directory under her state root is
            # internal maintenance, not a user-directed write to the host.
            # Without a declared scope the Will refused this on every ambient
            # perception tick and take_screenshot failed before it ever
            # reached screencapture — screen perception was dead behind a
            # warning card. The scope is narrow on purpose: one domain, one
            # path, released as soon as the directory exists.
            with local_internal_governed_scope(
                "host_automation.screenshot_directory",
                domain=ActionDomain.FILE_WRITE.value,
                constraints={"path": str(save_dir), "op": "ensure_directory"},
            ):
                directory_result = await ActionExecutor.execute(
                    domain=ActionDomain.FILE_WRITE,
                    action_name="host_automation.ensure_screenshot_directory",
                    params={"path": str(save_dir), "op": "ensure_directory"},
                    source="host_automation.screenshot_directory",
                )
            if not self._action_completed(directory_result):
                return AutomationReceipt(
                    action="take_screenshot",
                    target=str(save_dir),
                    adapter="screencapture",
                    success=False,
                    error=(
                        "Screenshot directory could not be created through the governed "
                        "file transaction lane."
                    ),
                    duration_ms=(time.time() - start) * 1000,
                )
            save_path = str(save_dir / f"screenshot_{ts}_{unique}.png")
        else:
            # A caller-supplied path must resolve inside an allowed screenshot
            # root and be an image file — otherwise screencapture would write an
            # arbitrary host path (symlink/traversal) with no boundary.
            resolved, allowed_roots = await asyncio.to_thread(
                _resolve_screenshot_path_policy, save_path
            )
            in_root = resolved is not None and any(
                resolved == r or str(resolved).startswith(str(r) + os.sep) for r in allowed_roots
            )
            # `in_root` already requires resolved is not None, but that is a
            # correlation mypy cannot follow — and a reader cannot either.
            # Test the thing directly rather than relying on a prior clause.
            if (
                resolved is None
                or not in_root
                or resolved.suffix.lower() not in {".png", ".jpg", ".jpeg"}
            ):
                return AutomationReceipt(
                    action="take_screenshot",
                    target=str(save_path),
                    adapter="screencapture",
                    success=False,
                    error="Screenshot save_path is outside the allowed roots or not an image file.",
                    duration_ms=(time.time() - start) * 1000,
                )
            save_path = str(resolved)

        try:
            capture_started_at = time.time()
            cmd = ["screencapture", "-x"]  # -x = no sound
            if region:
                x, y, w, h = region
                cmd.extend(["-R", f"{x},{y},{w},{h}"])
            cmd.append(save_path)

            proc = await get_subprocess_gateway().spawn_async(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                source="host_automation.screenshot",
                accelerator_capability="auto",
            )
            await asyncio.wait_for(proc.communicate(), timeout=5.0)

            def _fresh_capture() -> bool:
                p = Path(save_path)
                if not p.exists():
                    return False
                st = p.stat()
                # A pre-existing file satisfies mere existence — require a
                # nonempty file written at/after this capture began.
                return st.st_size > 0 and st.st_mtime >= (capture_started_at - 1.0)

            fresh = await asyncio.to_thread(_fresh_capture)
            success = proc.returncode == 0 and fresh

            receipt = AutomationReceipt(
                action="take_screenshot", target=save_path,
                adapter="screencapture", success=success,
                result=save_path if success else "",
                duration_ms=(time.time() - start) * 1000,
            )
        except (TimeoutError, OSError) as e:
            receipt = AutomationReceipt(
                action="take_screenshot", target=save_path,
                adapter="screencapture", success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

        self._log_receipt(receipt)
        # Retention runs for the ephemeral directory too.
        #
        # It used to be gated on retain_capture, so the ONLY thing bounding
        # ~/.aura/data/ephemeral was the per-call cleanup in get_screen_text —
        # a single point of failure guarding the directory with by far the
        # highest churn, roughly one full-screen capture every 7 seconds. When
        # that cleanup was refused for hours on 2026-08-10 nothing noticed and
        # nothing pruned. A backstop is the difference between a bug that
        # leaves stale files and one that fills a disk with pictures of the
        # person's screen.
        if receipt.success:
            try:
                retention = await self._enforce_screenshot_retention(
                    Path(save_path).parent,
                    keep_path=Path(save_path),
                )
                if retention["deleted"]:
                    logger.info(
                        "Screenshot retention removed %d captures (%d bytes)",
                        retention["deleted"],
                        retention["bytes_deleted"],
                    )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "host_automation.screenshot_retention",
                    exc,
                    action="returned screenshot after retention enforcement failed",
                    severity="warning",
                )
        return receipt

    @staticmethod
    def _ocr_image_regions(image_path: str) -> list[dict[str, Any]]:
        """Recognized text WITH the position of each run.

        macOS Vision already computes this. Every VNRecognizedTextObservation
        carries a boundingBox, and _ocr_image_text read `.string()` off the top
        candidate and discarded the geometry, returning "\n".join(lines).

        So the flat text a caller received was not what the OS produced — it
        was what survived. Anything laid out in two dimensions became a column
        of strings in reading order: a table lost its columns, a form lost
        which label went with which field, and a grid lost the grid. Tasks
        that need to know WHERE something is were unreachable, and the reason
        was invisible because OCR appeared to work.

        Boxes are normalized 0..1 with a TOP-left origin, matching how screen
        coordinates are expressed everywhere else in this codebase. Vision's
        own origin is bottom-left, and leaving that mismatch for each caller to
        remember is how a click lands at the wrong end of the screen.

        Returns [] rather than raising: a caller that wants flat text still has
        _ocr_image_text, and losing layout is not a reason to lose the words.
        """
        try:
            from Foundation import NSURL
            from Quartz import (
                CGImageSourceCreateImageAtIndex,
                CGImageSourceCreateWithURL,
            )
            from Vision import (
                VNImageRequestHandler,
                VNRecognizeTextRequest,
                VNRequestTextRecognitionLevelAccurate,
            )

            image_url = NSURL.fileURLWithPath_(str(image_path))
            image_source = CGImageSourceCreateWithURL(image_url, None)
            if image_source is None:
                return []
            image = CGImageSourceCreateImageAtIndex(image_source, 0, None)
            if image is None:
                return []
            request = VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)
            request.setUsesLanguageCorrection_(True)
            handler = VNImageRequestHandler.alloc().initWithCGImage_options_(image, {})
            succeeded, _error = handler.performRequests_error_([request], None)
            if not succeeded:
                return []
            regions: list[dict[str, Any]] = []
            for observation in list(request.results() or []):
                candidates = list(observation.topCandidates_(1) or [])
                if not candidates:
                    continue
                value = str(candidates[0].string() or "").strip()
                if not value:
                    continue
                try:
                    box = observation.boundingBox()
                    x = float(box.origin.x)
                    y = float(box.origin.y)
                    w = float(box.size.width)
                    h = float(box.size.height)
                    confidence = float(candidates[0].confidence())
                except (AttributeError, TypeError, ValueError):
                    continue
                regions.append(
                    {
                        "text": value,
                        # Vision measures up from the bottom; everything else
                        # here measures down from the top.
                        "x": round(x, 5),
                        "y": round(1.0 - (y + h), 5),
                        "width": round(w, 5),
                        "height": round(h, 5),
                        "center_x": round(x + (w / 2.0), 5),
                        "center_y": round(1.0 - (y + (h / 2.0)), 5),
                        "confidence": round(confidence, 4),
                    }
                )
            return regions
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return []

    @staticmethod
    def _ocr_image_text(image_path: str) -> str:
        """Recognize text with native macOS Vision, then optional Tesseract."""
        native_error = ""
        try:
            from Foundation import NSURL
            from Quartz import (
                CGImageSourceCreateImageAtIndex,
                CGImageSourceCreateWithURL,
            )
            from Vision import (
                VNImageRequestHandler,
                VNRecognizeTextRequest,
                VNRequestTextRecognitionLevelAccurate,
            )

            image_url = NSURL.fileURLWithPath_(str(image_path))
            image_source = CGImageSourceCreateWithURL(image_url, None)
            if image_source is None:
                raise ValueError("Vision could not open the screenshot image source")
            image = CGImageSourceCreateImageAtIndex(image_source, 0, None)
            if image is None:
                raise ValueError("Vision could not decode the screenshot image")
            request = VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)
            request.setUsesLanguageCorrection_(True)
            handler = VNImageRequestHandler.alloc().initWithCGImage_options_(image, {})
            succeeded, error = handler.performRequests_error_([request], None)
            if not succeeded:
                raise RuntimeError(f"Vision OCR request failed: {error}")
            lines: list[str] = []
            for observation in list(request.results() or []):
                candidates = list(observation.topCandidates_(1) or [])
                if not candidates:
                    continue
                value = str(candidates[0].string() or "").strip()
                if value:
                    lines.append(value)
            if lines:
                return "\n".join(lines)
            native_error = "Vision returned no recognized text"
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            native_error = f"{type(exc).__name__}:{exc}"

        try:
            import pytesseract
            from PIL import Image

            return str(pytesseract.image_to_string(Image.open(str(image_path))) or "").strip()
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            fallback_error = f"{type(exc).__name__}:{exc}"
            raise RuntimeError(
                f"OCR unavailable (native={native_error or 'unavailable'}; "
                f"fallback={fallback_error})"
            ) from exc

    async def get_screen_text(
        self,
        region: tuple[int, int, int, int] | None = None,
        *,
        retain_screenshot: bool = True,
    ) -> AutomationReceipt:
        """Take a screenshot and extract text via OCR.

        Verification callers set ``retain_screenshot=False`` so repeated
        desktop actions do not accumulate private screen captures indefinitely.
        """
        from .host_automation import (
            _HOST_AUTOMATION_ERRORS,
            AutomationReceipt,
            logger,
        )

        start = time.time()
        # Take screenshot first
        ss = await self.take_screenshot(
            region=region,
            retain_capture=retain_screenshot,
        )
        if not ss.success or not ss.result:
            return AutomationReceipt(
                action="get_screen_text", target="",
                adapter=ss.adapter or "ocr", success=False,
                error=f"Screenshot failed: {ss.error}",
                duration_ms=(time.time() - start) * 1000,
                evidence=dict(ss.evidence),
            )

        text = ""
        ocr_error = ""
        regions: list[dict[str, Any]] = []
        try:
            text = await asyncio.to_thread(self._ocr_image_text, str(ss.result))
            # Read the layout in the SAME pass. Vision computes the position of
            # every text run whether or not anyone asks, so taking it here
            # costs one more traversal of a result set already in memory —
            # against a second full screenshot and a second OCR, which is
            # the difference between a loop that can watch something
            # change and one that cannot.
            regions = await asyncio.to_thread(self._ocr_image_regions, str(ss.result))
        except _HOST_AUTOMATION_ERRORS as e:
            ocr_error = str(e)

        if not retain_screenshot:
            try:
                # THIRD instance of the missing-scope defect, after the capture
                # directory and screenshot retention. Both of those were fixed
                # by declaring the scope; this path was not, so every ephemeral
                # capture survived its own deletion.
                #
                # LIVE, 2026-08-10, immediately after the capture fix went live:
                # ~/.aura/data/ephemeral grew to 10 files / 15MB in the first
                # minutes, one roughly every 7 seconds — about 770MB an hour of
                # full-screen captures that nothing could ever remove. The
                # refusal reads "permission_model_blocked: Modality
                # 'file_delete' is disabled", and that rule is correct: Aura
                # may not delete a person's files, and file_delete is
                # deliberately ungrantable. This file is one she created
                # herself, seconds ago, under her own state root,
                # explicitly as ephemeral — and leaving it there is the privacy
                # harm the rule exists to prevent, not a way of avoiding one.
                with local_internal_governed_scope(
                    "host_automation.ephemeral_ocr_cleanup",
                    domain=ActionDomain.FILE_WRITE.value,
                    constraints={"path": str(ss.result), "op": "delete"},
                ):
                    cleanup = await ActionExecutor.execute(
                        domain=ActionDomain.FILE_WRITE,
                        action_name="host_automation.ephemeral_ocr_cleanup",
                        params={"path": str(ss.result), "op": "delete"},
                        source="host_automation.ephemeral_ocr_cleanup",
                    )
                if not self._action_completed(cleanup):
                    raise RuntimeError(
                        str(cleanup.get("error") or "ephemeral OCR cleanup was not verified")
                    )
            except _HOST_AUTOMATION_ERRORS as exc:
                record_degradation(
                    "host_automation.ocr_cleanup",
                    exc,
                    action="retained an ephemeral OCR screenshot after governed cleanup failed",
                    severity="warning",
                )
                logger.warning("Ephemeral OCR screenshot cleanup failed: %s", exc)

        receipt = AutomationReceipt(
            action="get_screen_text",
            target=str(ss.result) if retain_screenshot else "ephemeral_verification_capture",
            adapter="ocr", success=bool(text),
            result=text[:2000],
            error=ocr_error[:500],
            duration_ms=(time.time() - start) * 1000,
            layout=regions,
        )
        return receipt
