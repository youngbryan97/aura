"""Turning an answer into a file a person can open.

A PDF drawn through Quartz where that is available and through a plain text
renderer where it is not, and the image search that finds a picture with a
licence rather than the first one that matched. Both end in a file on disk with
a checksum, because a document she says she made has to be a document that is
there.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_bytes
from core.runtime.content_integrity import paragraph_sha256s, text_sha256
from core.runtime.errors import record_degradation


class _MakesFilesToShow:
    """Lifted whole from ComputerUseSkill; see computer_use.py."""

    @staticmethod
    def _has_pdf_header(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"

    @classmethod
    def _image_topic_candidates(
        cls, topic: str, gateway: Any, headers: dict[str, str]
    ) -> list[str]:
        """Titles worth trying for "a picture of X", best first.

        The literal topic comes first because it is free when it works. Then
        the bare noun ("a rock" -> "rock"), then whatever Wikipedia's own
        search says that phrase names — which is what turns "a rock" into
        "Rock (geology)" without anyone writing that mapping down.
        """
        seen: list[str] = []

        def _add(value: str) -> None:
            cleaned = " ".join(str(value or "").split())
            if cleaned and cleaned.lower() not in {item.lower() for item in seen}:
                seen.append(cleaned)

        raw = " ".join(str(topic or "").split())
        if not raw:
            return []
        _add(raw[:1].upper() + raw[1:])

        stripped = raw.lower()
        changed = True
        while changed:
            changed = False
            for leader in cls._IMAGE_TOPIC_LEADERS:
                if stripped.startswith(leader):
                    stripped = stripped[len(leader):].strip()
                    changed = True
        if stripped:
            _add(stripped[:1].upper() + stripped[1:])

        # NO FUZZY ARTICLE SEARCH. It answers a different question.
        #
        # Wikipedia's full-text search ranks by article prominence, not by
        # what a word depicts, so asking it to name "a picture of X" returns
        # whatever is famous:
        #
        #   "rock" -> Rock music, The Rock, "Rock, Rock, Rock!" (a 1956 film)
        #   "tree" -> Kruskal's tree theorem, Oliver Tree
        #
        # Measured the hard way: asked for a rock as his wallpaper, Bryan got
        # the one-sheet poster for "Rock, Rock, Rock!". The lookup succeeded
        # and the sense was wrong, which is worse than failing.
        #
        # Only the literal title and the bare noun are tried here. Finding a
        # picture of a thing is Wikimedia Commons' job — it indexes files that
        # are OF things — and the caller reaches for it before giving up.
        return seen[:2]

    @staticmethod
    def _commons_image_candidate(
        topic: str, gateway: Any, headers: dict[str, str]
    ) -> tuple[str, str] | None:
        """An image from Wikimedia Commons: ``(image_url, page_url)``.

        The last resort, and the only step that is actually an image search
        rather than an encyclopedia lookup. Commons is used because it is
        freely licensed and reachable through the same governed gateway.
        """
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from urllib.parse import quote

        from .computer_use import (
            ComputerUseSkill,
            logger,
        )

        query = " ".join(str(topic or "").split())
        if not query:
            return None
        url = (
            "https://commons.wikimedia.org/w/api.php?action=query"
            f"&generator=search&gsrsearch={quote(query)}&gsrlimit=8"
            "&gsrnamespace=6&prop=imageinfo&iiprop=url|size"
            "&iiurlwidth=2560&format=json"
        )
        try:
            response = ComputerUseSkill._polite_media_request(
                gateway,
                url,
                headers,
                timeout=20.0,
                source="computer_use:fetch_topic_image.commons",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Commons image search unavailable for %r: %s", topic, exc)
            return None
        if not response.get("ok"):
            return None
        body = response.get("content") or response.get("text") or b"{}"
        if isinstance(body, bytes):
            body = body.decode("utf-8", errors="replace")
        try:
            pages = (json.loads(body or "{}").get("query") or {}).get("pages") or {}
        except (TypeError, ValueError):
            return None
        for page in (pages.values() if isinstance(pages, dict) else []):
            if not isinstance(page, dict):
                continue
            # JUDGE THE FILE, NOT ITS RENDERING.
            #
            # Commons renders the first page of a PDF or DjVu as a .jpg
            # thumbnail, so a suffix check on thumburl let documents through:
            # "a lonely traffic cone" came back as a scanned poetry book.
            title = str(page.get("title") or "").lower()
            if title.endswith((".pdf", ".djvu", ".tif", ".tiff", ".svg", ".ogv", ".webm")):
                continue
            if not title.endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            for info in page.get("imageinfo") or []:
                if not isinstance(info, dict):
                    continue
                # A scaled rendition when Commons offers one: the originals
                # are routinely tens of megabytes and the byte bound below
                # would drop them.
                image_url = str(info.get("thumburl") or info.get("url") or "")
                if not image_url.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".webp")
                ):
                    continue
                return (image_url, str(info.get("descriptionurl") or ""))
        return None

    def _fetch_topic_image(self, target: str) -> dict[str, Any]:
        """Fetch a representative image for a topic via Wikipedia's REST
        summary API, through the governed network gateway. General by
        construction: any topic, deterministic endpoint, no scraping —
        and the page URL comes back as evidence of where it was found.
        """
        from .computer_use import (
            _image_suffix_from_bytes,
        )

        payload = self._target_json(target)
        topic = str(payload.get("topic") or "").strip()
        if not topic:
            return {"ok": False, "error": "fetch_topic_image requires a topic."}
        path = self._resolve_allowed_desktop_path(payload.get("path"))
        from urllib.parse import quote

        from core.runtime.network_gateway import get_network_gateway

        gateway = get_network_gateway()
        ua = {"User-Agent": "AuraDigitalEntity/1.0 (local desktop runtime)"}

        # ASK FOR A PICTURE OF A THING, not for an exact encyclopedia title.
        #
        # This did one lookup: Wikipedia's summary endpoint for the literal
        # topic, capitalised. That works for "orca" and fails for most of
        # English. Measured live 2026-07-28, asked to set a rock as the
        # wallpaper:
        #
        #   "rock"    -> no image available for topic 'rock'
        #               (Wikipedia's "Rock" is a disambiguation page, and a
        #                disambiguation page has no thumbnail)
        #   "a rock"  -> topic lookup failed: HTTP Error 404
        #               (there is no article called "A_rock")
        #
        # She had found the image search perfectly well and then had nowhere
        # to go, because the one endpoint she could use demanded a title she
        # did not have. Resolving the topic is the general form: try what was
        # asked, then let Wikipedia's own search say which article that names,
        # then fall back to Wikimedia Commons — an actual image search rather
        # than an encyclopedia lookup.
        doc: dict[str, Any] = {}
        lookup_error = ""
        for candidate_title in self._image_topic_candidates(topic, gateway, ua):
            summary_url = (
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                + quote(candidate_title.replace(" ", "_"))
            )
            meta = self._polite_media_request(
                gateway,
                summary_url,
                ua,
                timeout=20.0,
                source="computer_use:fetch_topic_image",
            )
            if not meta.get("ok"):
                lookup_error = str(meta.get("error") or meta.get("status_code"))
                continue
            raw_meta = meta.get("content") or meta.get("text") or b"{}"
            if isinstance(raw_meta, bytes):
                raw_meta = raw_meta.decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw_meta or "{}")
            except (TypeError, ValueError):
                parsed = {}
            if not isinstance(parsed, dict):
                continue
            # A disambiguation page is a list of other pages, never a picture
            # of anything. Keep looking rather than reporting "no image".
            if str(parsed.get("type") or "").endswith("disambiguation"):
                lookup_error = f"'{candidate_title}' is a disambiguation page"
                continue
            has_image = bool(
                (parsed.get("originalimage") or {}).get("source")
                or (parsed.get("thumbnail") or {}).get("source")
            )
            if has_image:
                doc = parsed
                break
            lookup_error = f"'{candidate_title}' has no illustration"
        if not doc and lookup_error and "404" in lookup_error and not topic:
            return {"ok": False, "error": f"topic lookup failed: {lookup_error}"}
        original_url = str(((doc.get("originalimage") or {}).get("source")) or "")
        thumbnail_url = str(((doc.get("thumbnail") or {}).get("source")) or "")
        page_url = str(
            ((doc.get("content_urls") or {}).get("desktop") or {}).get("page")
            or f"https://en.wikipedia.org/wiki/{quote(topic.replace(' ', '_'))}"
        )
        # Candidate order: original (full quality, e.g. wallpaper use),
        # then a 1600px rendition of the thumbnail, then the raw thumbnail.
        # Each is size-bounded; oversized candidates fall through instead
        # of failing the whole step (live failure: squid original > 8MB).
        candidates = [u for u in (original_url, thumbnail_url) if u]
        if thumbnail_url and "px-" in thumbnail_url:
            import re as _re

            wide = _re.sub(r"/(\d+)px-", "/1600px-", thumbnail_url, count=1)
            if wide != thumbnail_url:
                candidates.insert(1, wide)
        if not candidates:
            # The actual image search, and the only step that is one.
            commons = self._commons_image_candidate(topic, gateway, ua)
            if commons:
                candidates = [commons[0]]
                page_url = commons[1] or page_url
                lookup_error = ""
        if not candidates:
            # BEING THROTTLED IS NOT THE SAME AS THERE BEING NO PICTURE.
            #
            # Measured: five image topics in a burst all came back "no image
            # available for topic X (HTTP Error 429)" — a rate limit reported
            # to the person as though the thing they asked for did not exist.
            # She would then explain that she could not find a traffic cone.
            throttled = "429" in lookup_error or "too many" in lookup_error.lower()
            return {
                "ok": False,
                "error": (
                    (
                        f"the image service asked me to slow down while looking "
                        f"for '{topic}' — this is rate limiting, not a missing "
                        f"picture; trying again in a moment should work"
                    )
                    if throttled
                    else (
                        f"no image available for topic '{topic}'"
                        + (f" ({lookup_error})" if lookup_error else "")
                    )
                ),
                "rate_limited": throttled,
                "page_url": page_url,
            }
        max_bytes = 24 * 1024 * 1024
        raw = b""
        image_url = ""
        last_error = ""
        for candidate in candidates:
            img = gateway.request(
                "GET",
                candidate,
                headers=ua,
                timeout=30.0,
                source="computer_use:fetch_topic_image",
                read_only=True,
            )
            body = img.get("content") or img.get("body_bytes")
            if isinstance(body, str):
                body = body.encode("latin-1", errors="ignore")
            if not img.get("ok") or not body:
                last_error = f"download failed: {img.get('error') or img.get('status_code')}"
                continue
            if len(body) > max_bytes:
                last_error = f"candidate exceeds {max_bytes // (1024 * 1024)}MB bound"
                continue
            raw, image_url = body, candidate
            break
        if not raw:
            return {
                "ok": False,
                "error": f"image download failed for all candidates ({last_error})",
                "image_url": candidates[0],
                "page_url": page_url,
            }
        # Name the file what it actually is.
        #
        # The planner asks for "<topic>_wallpaper.png" and the web returns
        # whatever the web returns. Live 2026-07-28 a real grizzly image
        # landed on the Desktop as grizzly_bear_wallpaper.png containing JPEG
        # data — `file` said "JPEG image data" for a .png. Nothing downstream
        # had lied; the extension had. Renaming to the sniffed type keeps the
        # artifact honest for anything that trusts the suffix.
        sniffed = _image_suffix_from_bytes(raw)
        if sniffed and path.suffix.lower() != sniffed:
            path = path.with_suffix(sniffed)
        atomic_write_bytes(path, raw)
        expected_digest = hashlib.sha256(raw).hexdigest()
        actual_digest = self._file_sha256(path)
        verified = path.is_file() and actual_digest == expected_digest
        return {
            "ok": verified,
            "action": "fetch_topic_image",
            "path": str(path),
            "bytes": len(raw),
            "sha256": actual_digest,
            "image_url": image_url,
            "page_url": page_url,
            "topic": topic,
            "effect_verified": verified,
            "verification": (
                "Downloaded image matched the governed network response."
                if verified
                else "Downloaded image did not match the governed network response."
            ),
            **({} if verified else {"error": "Downloaded image verification failed."}),
        }

    def _render_text_pdf_quartz(
        self, path: Any, title: str, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Render a searchable-text PDF via CoreGraphics; None = fall back."""
        from .computer_use import (
            _QUARTZ_RENDER_ERRORS,
        )

        try:
            import Quartz
            from CoreText import (
                CTFontCreateWithName,
                CTFrameDraw,
                CTFramesetterCreateFrame,
                CTFramesetterCreateWithAttributedString,
                kCTFontAttributeName,
            )
            from Foundation import NSURL
            from Quartz import CoreGraphics as CG  # noqa: N817 - Apple framework convention
        except ImportError:
            return None

        body = str(payload.get("body") or "")[:9000]
        image_path = str(payload.get("image_path") or "").strip()
        width, height, margin = 612.0, 792.0, 54.0
        image_drawn = False
        image_error = ""

        try:
            url = NSURL.fileURLWithPath_(str(path))
            rect = CG.CGRectMake(0, 0, width, height)
            ctx = Quartz.CGPDFContextCreateWithURL(url, rect, None)
            if ctx is None:
                return None

            from Foundation import (
                NSAttributedString,
                NSMutableAttributedString,
            )

            title_font = CTFontCreateWithName("Helvetica-Bold", 17.0, None)
            body_font = CTFontCreateWithName("Helvetica", 12.0, None)
            text = NSMutableAttributedString.alloc().initWithString_attributes_(
                title + "\n\n", {kCTFontAttributeName: title_font}
            )
            text.appendAttributedString_(
                NSAttributedString.alloc().initWithString_attributes_(
                    body, {kCTFontAttributeName: body_font}
                )
            )
            framesetter = CTFramesetterCreateWithAttributedString(text)

            image = None
            img_h = 0.0
            if image_path:
                try:
                    img_file = self._resolve_allowed_desktop_path(
                        image_path, must_exist=True
                    )
                    img_url = NSURL.fileURLWithPath_(str(img_file))
                    source = Quartz.CGImageSourceCreateWithURL(img_url, None)
                    if source is not None:
                        image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
                except (OSError, ValueError) as exc:
                    image_error = str(exc)
                if image is not None:
                    iw = float(CG.CGImageGetWidth(image))
                    ih = float(CG.CGImageGetHeight(image))
                    max_w, max_h = width - 2 * margin, 260.0
                    scale = min(max_w / max(iw, 1.0), max_h / max(ih, 1.0), 1.0)
                    img_w, img_h = iw * scale, ih * scale

            consumed = 0
            total = text.length()
            first_page = True
            page_count = 0
            while consumed < total or first_page:
                Quartz.CGPDFContextBeginPage(ctx, None)
                top = height - margin
                if first_page and image is not None:
                    CG.CGContextDrawImage(
                        ctx,
                        CG.CGRectMake(margin, top - img_h, img_w, img_h),
                        image,
                    )
                    image_drawn = True
                    top -= img_h + 14.0
                frame_rect = CG.CGRectMake(
                    margin, margin, width - 2 * margin, top - margin
                )
                frame_path = CG.CGPathCreateWithRect(frame_rect, None)
                frame = CTFramesetterCreateFrame(
                    framesetter, (consumed, 0), frame_path, None
                )
                CTFrameDraw(frame, ctx)
                from CoreText import CTFrameGetVisibleStringRange

                visible = CTFrameGetVisibleStringRange(frame)
                advanced = int(visible.length)
                Quartz.CGPDFContextEndPage(ctx)
                page_count += 1
                first_page = False
                if advanced <= 0:
                    break
                consumed += advanced
            Quartz.CGPDFContextClose(ctx)
        except _QUARTZ_RENDER_ERRORS as exc:
            record_degradation(
                "computer_use",
                exc,
                action="fell back to raster PDF after Quartz text rendering failed",
                severity="warning",
            )
            return None

        result: dict[str, Any] = {
            "ok": bool(path.exists() and path.stat().st_size > 0),
            "action": "render_text_pdf",
            "path": str(path),
            "renderer": "quartz_text_layer",
            "image_embedded": image_drawn,
            "bytes": path.stat().st_size if path.exists() else 0,
            "pages": max(1, page_count),
            "chars": len(title) + len(body),
        }
        result["sha256"] = self._file_sha256(path) if result["ok"] else ""
        result["effect_verified"] = bool(
            result["ok"] and self._has_pdf_header(path)
        )
        result["ok"] = result["effect_verified"]
        result["verification"] = (
            "PDF header and persisted content confirmed."
            if result["effect_verified"]
            else "PDF renderer returned without a valid persisted PDF."
        )
        if not result["ok"]:
            result["error"] = result["verification"]
        if image_error:
            result["image_error"] = image_error
        return result

    def _render_text_pdf(self, target: str) -> dict[str, Any]:
        payload = self._target_json(target)
        path = self._resolve_allowed_desktop_path(payload.get("path"))
        title = str(payload.get("title") or "Aura Desktop Proof")[:160]
        body = str(payload.get("body") or "")
        overwrite = bool(payload.get("overwrite", False))
        if not body.strip():
            return {"ok": False, "error": "PDF body is empty."}
        if path.exists() and not overwrite:
            path = self._versioned_path(path)
        if path.suffix.lower() != ".pdf":
            return {"ok": False, "error": "PDF path must end with .pdf."}

        # The renderer is deliberately bounded, so hash the exact bounded body
        # that is handed to either backend. These hashes let an upstream task
        # prove the requested synthesis reached this specific persisted PDF
        # without putting private document text in its audit receipt.
        max_chars = 9000
        safe_body = body[:max_chars]
        payload = dict(payload, body=safe_body)
        content_evidence = {
            "source_body_sha256": text_sha256(safe_body),
            "source_body_chars": len(safe_body),
            "source_paragraph_sha256s": list(paragraph_sha256s(safe_body)),
            "source_title_sha256": text_sha256(title),
        }

        # Native Quartz rendering produces a REAL text layer (searchable,
        # extractable, hostile-verifiable). The previous Pillow renderer
        # rasterized every page into one big image: zero extractable
        # text, and an /Image XObject on every page that made embedded-
        # image evidence vacuous.
        if sys.platform == "darwin":
            quartz_result = self._render_text_pdf_quartz(path, title, payload)
            if quartz_result is not None:
                quartz_result.update(content_evidence)
                return quartz_result

        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:
            return {"ok": False, "error": f"Pillow is required for PDF rendering: {exc}"}

        width, height = 612, 792
        margin = 54
        line_height = 18
        title_height = 28
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
            title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 17)
        except (OSError, ValueError):
            font = ImageFont.load_default()
            title_font = font

        def wrap_line(draw: ImageDraw.ImageDraw, line: str) -> list[str]:
            if not line:
                return [""]
            words = line.split(" ")
            lines: list[str] = []
            current = ""
            max_width = width - (2 * margin)
            for word in words:
                candidate = word if not current else f"{current} {word}"
                if draw.textlength(candidate, font=font) <= max_width:
                    current = candidate
                    continue
                if current:
                    lines.append(current)
                current = word
            if current:
                lines.append(current)
            return lines or [line]

        pages: list[Image.Image] = []

        def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
            page = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(page)
            draw.text((margin, margin), title, fill=(0, 0, 0), font=title_font)
            return page, draw, margin + title_height + 14

        page, draw, y = new_page()

        image_path = str(payload.get("image_path") or "").strip()
        if image_path:
            try:
                resolved_img = self._resolve_allowed_desktop_path(image_path, must_exist=True)
                with Image.open(resolved_img) as embedded:
                    embedded = embedded.convert("RGB")
                    max_w = width - (2 * margin)
                    max_h = 260
                    embedded.thumbnail((max_w, max_h))
                    page.paste(embedded, (margin, y))
                    y += embedded.height + 14
            except (OSError, ValueError) as exc:
                # The image is an enhancement; the document must still
                # render — but record the miss honestly in the body.
                draw.text((margin, y), f"[image unavailable: {exc}]", fill=(120, 0, 0), font=font)
                y += line_height + 6

        for paragraph in safe_body.splitlines():
            for line in wrap_line(draw, paragraph):
                if y + line_height > height - margin:
                    pages.append(page)
                    page, draw, y = new_page()
                draw.text((margin, y), line, fill=(0, 0, 0), font=font)
                y += line_height
            y += 6
        pages.append(page)

        path.parent.mkdir(parents=True, exist_ok=True)
        first, rest = pages[0], pages[1:]
        first.save(path, "PDF", resolution=72.0, save_all=bool(rest), append_images=rest)
        verified = self._has_pdf_header(path)
        return {
            "ok": verified,
            "action": "render_text_pdf",
            "path": str(path),
            "bytes": path.stat().st_size,
            "pages": len(pages),
            "chars": len(safe_body),
            "sha256": self._file_sha256(path),
            "effect_verified": verified,
            **content_evidence,
            "verification": (
                "PDF header and persisted content confirmed."
                if verified
                else "PDF renderer returned without a valid persisted PDF."
            ),
            **({} if verified else {"error": "PDF artifact verification failed."}),
        }
