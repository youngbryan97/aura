"""core/capabilities/web_asset_handler.py — Image Search, Download & Validation
================================================================================
General-purpose web asset handler. Searches, downloads, validates, and
manages images and other web assets.

NOT hardcoded for any specific content. The LLM + planner decides what
to search for; this layer handles the HOW.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.network_gateway import get_network_gateway
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.WebAssetHandler")
# ── Image admission limits ──────────────────────────────────────────────
#
# CP126 (critical): "MIME rejection is fail-open and magic-byte checks do
# not validate an image." Both halves were true. A non-image Content-Type
# only logged a warning and processing continued ("Try anyway — some
# servers don't set correct type"), and the validator accepted a few
# leading bytes with no structural decode, so a hostile or malformed file
# was persisted as a valid image.
#
# What a magic-byte prefix cannot detect, and now is checked:
#   * polyglots — a file that is a valid JPEG header AND valid HTML, JS or
#     a shell script, which is how "just an image" becomes executable
#     content once something downstream opens it by extension
#   * decompression bombs — a few KB of PNG that decodes to gigapixels
#   * truncated or corrupt bodies that decode into garbage
#   * animation frame floods
#   * arbitrary payloads appended after a well-formed image
_MAX_IMAGE_PIXELS = 50_000_000          # ~50MP; a 100MB RGBA bitmap decoded
_MAX_IMAGE_FRAMES = 512                 # animated GIF/WEBP frame ceiling
_MAX_TRAILING_BYTES = 4096              # slack for legitimate metadata trailers
_ALLOWED_IMAGE_FORMATS = frozenset({"jpeg", "png", "webp", "gif", "bmp"})
# Content-Type values that map to a format we accept. A server may be
# sloppy about which image type it names, but it may not name a non-image.
_ALLOWED_IMAGE_MIME_PREFIX = "image/"


@dataclass(frozen=True)
class ImageAdmission:
    """The verdict on one candidate image, and how much was actually proven.

    ``structurally_verified`` is the field that keeps this honest. Pillow is
    an optional dependency; when it is absent no structural decode happened,
    and saying so is not the same as saying the image passed. A caller that
    needs a real guarantee reads this rather than ``ok``.
    """

    ok: bool
    fmt: str = "unknown"
    reason: str = ""
    structurally_verified: bool = False
    width: int = 0
    height: int = 0
    frames: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.ok,
            "format": self.fmt,
            "reason": self.reason,
            "structurally_verified": self.structurally_verified,
            "width": self.width,
            "height": self.height,
            "frames": self.frames,
        }


def _container_end_offset(data: bytes, fmt: str) -> int:
    """Where this image container legitimately ends, or -1 if unknown.

    A decoder answers "are the pixels valid", which is not the same question
    as "is this file ONLY an image". Pillow decodes a well-formed PNG with a
    shell script stapled to the end without complaint — that is the polyglot,
    and it is only visible by comparing the container's own declared extent
    against the file length.
    """
    try:
        if fmt == "png":
            # IEND chunk type, plus its 4-byte CRC.
            marker = data.rfind(b"IEND")
            return marker + 8 if marker >= 0 else -1
        if fmt == "jpeg":
            marker = data.rfind(b"\xff\xd9")  # EOI
            return marker + 2 if marker >= 0 else -1
        if fmt == "gif":
            marker = data.rfind(b"\x3b")  # trailer
            return marker + 1 if marker >= 0 else -1
        if fmt in {"webp", "bmp"}:
            # Both declare their own total size in the header.
            offset, base = (4, 8) if fmt == "webp" else (2, 0)
            declared = int.from_bytes(data[offset:offset + 4], "little")
            return declared + base if declared > 0 else -1
    except (IndexError, ValueError):
        return -1
    return -1


def _structural_image_check(data: bytes, declared_fmt: str) -> ImageAdmission:
    """Decode the image for real and hold it to the admission limits."""
    try:
        import io

        from PIL import Image
    except ImportError:
        # No decoder available. Report the magic-byte result as exactly what
        # it is — unproven — rather than promoting it to verified.
        return ImageAdmission(
            ok=True,
            fmt=declared_fmt,
            reason="pillow_unavailable_structure_unverified",
            structurally_verified=False,
        )

    # Pillow's own bomb guard. Set below our ceiling so it fires first and
    # raises rather than merely warning.
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS
    try:
        # verify() checks structural integrity but leaves the file unusable
        # afterwards, so the image is opened twice on purpose.
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(data)) as image:
            fmt = str(image.format or "").lower()
            width, height = int(image.width), int(image.height)
            frames = int(getattr(image, "n_frames", 1) or 1)
            # Force a real decode; verify() alone accepts bodies that fail
            # when the pixels are actually read.
            image.load()
    except Exception as exc:  # noqa: BLE001 - Pillow failures vary by codec/version.
        logger.debug(
            "Rejected structurally invalid image payload (%s)",
            type(exc).__name__,
            exc_info=True,
        )
        return ImageAdmission(
            ok=False,
            fmt=declared_fmt,
            reason=f"structural_decode_failed:{type(exc).__name__}",
            structurally_verified=True,
        )
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit

    if fmt not in _ALLOWED_IMAGE_FORMATS:
        return ImageAdmission(
            False, fmt, f"format_not_allowed:{fmt}", True, width, height, frames,
        )
    if fmt != declared_fmt:
        # The magic bytes and the decoder disagree about what this is. That
        # is the signature of a crafted container, not a sloppy server.
        return ImageAdmission(
            False, fmt, f"format_mismatch:magic={declared_fmt},decoded={fmt}",
            True, width, height, frames,
        )
    if width <= 0 or height <= 0:
        return ImageAdmission(False, fmt, "empty_dimensions", True, width, height, frames)
    if width * height > _MAX_IMAGE_PIXELS:
        return ImageAdmission(
            False, fmt, f"pixel_count_exceeds_limit:{width * height}",
            True, width, height, frames,
        )
    if frames > _MAX_IMAGE_FRAMES:
        return ImageAdmission(
            False, fmt, f"frame_count_exceeds_limit:{frames}", True, width, height, frames,
        )

    # Polyglot check. A decodable image says nothing about what was appended
    # after it; only the container's declared extent does.
    container_end = _container_end_offset(data, fmt)
    if container_end > 0:
        trailing = len(data) - container_end
        if trailing > _MAX_TRAILING_BYTES:
            return ImageAdmission(
                False,
                fmt,
                f"trailing_data_after_image:{trailing}",
                True,
                width,
                height,
                frames,
            )
    return ImageAdmission(True, fmt, "", True, width, height, frames)



class WebAssetHandler:
    """Search, download, validate, and manage web assets.

    Usage:
        handler = get_web_asset_handler()
        results = await handler.search_images("mountain landscape")
        path = await handler.download_image(results[0]["url"], "~/Documents/Aura/images")
        validation = await handler.validate_image(path)
    """

    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

    def __init__(self) -> None:
        self._download_count = 0
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("web_asset_handler", self, required=False)
        self._started = True
        logger.info("WebAssetHandler ONLINE")

    async def search_images(
        self, query: str, count: int = 5, min_width: int = 800, min_height: int = 600
    ) -> list[dict[str, str]]:
        """Search for images using DuckDuckGo image search.

        Returns list of {url, title, source, thumbnail} dicts.
        """
        try:
            results = await self._ddg_image_search(query, count * 2)

            # Filter by minimum dimensions if info available
            filtered = []
            for r in results:
                # Accept all initially — we'll validate after download
                filtered.append(r)
                if len(filtered) >= count:
                    break

            logger.info("Image search '%s': %d results", query[:30], len(filtered))
            return filtered

        except (OSError, RuntimeError) as e:
            record_degradation("web_asset.search", e)
            return []

    async def _ddg_image_search(self, query: str, count: int) -> list[dict[str, str]]:
        """Scrape DuckDuckGo image search results."""
        url = f"https://duckduckgo.com/?q={quote_plus(query)}&iax=images&ia=images"
        try:
            response = await get_network_gateway().request_async(
                "GET",
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                },
                timeout=10,
                read_only=True,
                source="web_asset_handler.ddg_image_search",
            )
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error") or response.get("status_code")))
            html = bytes(response.get("content", b"")).decode("utf-8", errors="replace")

            # Extract vqd token for API call
            vqd_match = re.search(r"vqd=['\"]([^'\"]+)", html)
            if not vqd_match:
                return await self._fallback_image_search(query, count)

            vqd = vqd_match.group(1)

            # Call DuckDuckGo image API
            api_url = (
                f"https://duckduckgo.com/i.js?l=us-en&o=json&q={quote_plus(query)}"
                f"&vqd={vqd}&f=size:Large&p=1"
            )
            response = await get_network_gateway().request_async(
                "GET",
                api_url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Referer": "https://duckduckgo.com/",
                },
                timeout=10,
                read_only=True,
                source="web_asset_handler.ddg_image_api",
            )
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error") or response.get("status_code")))
            data = bytes(response.get("content", b"")).decode("utf-8", errors="replace")

            parsed = json.loads(data)
            results = []
            for item in parsed.get("results", [])[:count]:
                img_url = item.get("image", "")
                if img_url and img_url.startswith("http"):
                    results.append({
                        "url": img_url,
                        "title": item.get("title", ""),
                        "source": item.get("source", urlparse(img_url).netloc),
                        "thumbnail": item.get("thumbnail", ""),
                        "width": str(item.get("width", "")),
                        "height": str(item.get("height", "")),
                    })

            return results

        except (json.JSONDecodeError, OSError, RuntimeError, TypeError, ValueError) as e:
            record_degradation("web_asset.ddg_image_search", e)
            logger.debug("DDG image API failed: %s", e)
            return await self._fallback_image_search(query, count)

    async def _fallback_image_search(self, query: str, count: int) -> list[dict[str, str]]:
        """Fallback: use Unsplash Source for free images."""
        results = []
        for i in range(min(count, 5)):
            results.append({
                "url": f"https://source.unsplash.com/1920x1080/?{quote_plus(query)}&sig={i}",
                "title": f"{query} (Unsplash #{i+1})",
                "source": "unsplash.com",
            })
        return results

    async def download_image(
        self, url: str, save_dir: str = "", filename: str = ""
    ) -> str:
        """Download an image, validate it, and save locally.

        Returns the local file path, or empty string on failure.
        """
        save_root = save_dir or str(state_root() / "data" / "assets")

        try:
            response = await get_network_gateway().request_async(
                "GET",
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                },
                timeout=30,
                read_only=True,
                source="web_asset_handler.download_image",
            )
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error") or response.get("status_code")))

            # Content-Type must claim an image. The previous behaviour
            # logged a warning and continued ("Try anyway — some servers
            # don't set correct type"), which meant a server returning
            # text/html or application/javascript had its body persisted
            # under an image filename. Tolerating a server that names the
            # WRONG image type is reasonable; tolerating one that names a
            # non-image is how a download becomes an execution primitive.
            headers = response.get("headers", {})
            content_type = str(headers.get("Content-Type", "")).strip().lower()
            declared_mime = content_type.split(";", 1)[0].strip()
            if declared_mime and not declared_mime.startswith(_ALLOWED_IMAGE_MIME_PREFIX):
                logger.warning(
                    "Refusing non-image Content-Type from %s (type=%s)",
                    url[:60],
                    declared_mime,
                )
                record_degradation(
                    "web_asset.download",
                    ValueError(f"non_image_content_type:{declared_mime}"),
                    severity="warning",
                    action="refused the download instead of saving it as an image",
                    enforce_failure_policy=False,
                )
                return ""

            # Read data with size limit
            data = bytes(response.get("content", b""))
            if len(data) > self.MAX_FILE_SIZE:
                logger.warning("Image too large: %s (%d bytes)", url[:60], len(data))
                return ""

            if len(data) < 100:
                logger.warning("Image too small: %s (%d bytes)", url[:60], len(data))
                return ""

            # Magic bytes first (cheap), then a real decode under the
            # admission limits. The prefix check alone accepts polyglots,
            # bombs and truncated bodies.
            is_valid, fmt = self._validate_image_header(data)
            if not is_valid:
                logger.warning("Invalid image data from %s", url[:60])
                return ""

            admission = await asyncio.to_thread(_structural_image_check, data, fmt)
            if not admission.ok:
                logger.warning(
                    "Refusing image from %s: %s", url[:60], admission.reason,
                )
                record_degradation(
                    "web_asset.download",
                    ValueError(f"image_admission_refused:{admission.reason}"),
                    severity="warning",
                    action="refused to persist a file that did not decode as a valid image",
                    enforce_failure_policy=False,
                )
                return ""
            if not admission.structurally_verified:
                # Honest about what was NOT checked, rather than silent.
                logger.info(
                    "Image from %s passed magic-byte checks only (%s)",
                    url[:60],
                    admission.reason,
                )
            fmt = admission.fmt or fmt

            save_path = Path(
                await get_file_write_gateway().ensure_directory_async(
                    save_root,
                    source="web_asset_handler.download_image",
                )
            )

            # Determine filename
            if not filename:
                # Generate from URL or hash
                parsed = urlparse(url)
                url_path = parsed.path.split("/")[-1] if parsed.path else ""
                if url_path and "." in url_path:
                    filename = url_path
                else:
                    ext = {"jpeg": ".jpg", "png": ".png", "webp": ".webp", "gif": ".gif"}.get(fmt, ".jpg")
                    filename = f"image_{hashlib.sha256(data).hexdigest()[:12]}{ext}"

            # Sanitize the filename. It can come straight from the URL
            # path, so it is attacker-influenced: strip separators and
            # control characters, then refuse anything that still resolves
            # somewhere other than the save directory.
            filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)[:200]
            filename = filename.strip().lstrip(".") or "image"
            if filename in {".", ".."} or os.sep in filename:
                filename = f"image_{hashlib.sha256(data).hexdigest()[:12]}.{fmt}"

            # Save
            file_path = save_path / filename
            if await asyncio.to_thread(file_path.exists):
                # Version it
                stem = file_path.stem
                suffix = file_path.suffix
                for v in range(2, 20):
                    candidate = save_path / f"{stem}_v{v}{suffix}"
                    if not await asyncio.to_thread(candidate.exists):
                        file_path = candidate
                        break

            await get_file_write_gateway().write_bytes_async(
                file_path,
                data,
                source="web_asset_handler.download_image",
            )
            self._download_count += 1
            logger.info("Downloaded image: %s (%d bytes, %s)", file_path.name, len(data), fmt)
            return str(file_path)

        except (OSError, RuntimeError, TypeError, ValueError) as e:
            record_degradation("web_asset.download", e)
            logger.debug("Image download failed from %s: %s", url[:60], e)
            return ""

    async def validate_image(self, path: str) -> dict[str, Any]:
        """Validate an image file, structurally — not just its first bytes."""
        p = Path(path)
        try:
            size_bytes = await asyncio.to_thread(lambda: p.stat().st_size)
        # not a failure: a file that is not there is the state this reads for.
        except FileNotFoundError:
            return {"valid": False, "error": "File not found"}
        except OSError as exc:
            return {"valid": False, "error": f"stat_failed:{type(exc).__name__}"}
        if size_bytes > self.MAX_FILE_SIZE:
            return {
                "valid": False,
                "error": f"file_exceeds_max_size:{size_bytes}",
                "size_bytes": size_bytes,
                "path": str(p),
            }

        # Reading and decoding are both blocking and unbounded in CPU; a
        # 50MB read plus a full decode on the event loop is a loop stall.
        data = await asyncio.to_thread(p.read_bytes)
        is_valid, fmt = self._validate_image_header(data)
        if not is_valid:
            return {
                "valid": False,
                "error": "magic_bytes_not_an_image",
                "format": fmt,
                "size_bytes": size_bytes,
                "path": str(p),
                "structurally_verified": False,
            }

        admission = await asyncio.to_thread(_structural_image_check, data, fmt)
        result: dict[str, Any] = admission.to_dict()
        result.update(size_bytes=size_bytes, path=str(p))
        if not admission.ok:
            result["error"] = admission.reason
        return result

    @staticmethod
    def _validate_image_header(data: bytes) -> tuple[bool, str]:
        """Check image magic bytes — a cheap prefilter, not a validation.

        This answers "could this plausibly be an image", and nothing more.
        Anything persisted must also clear ``_structural_image_check``.

        The WEBP rule previously appeared twice: once correctly as
        ``RIFF`` + ``WEBP`` at offset 8, and once as a bare ``data[8:12] ==
        b"WEBP"`` that matched FIRST. Any file at all whose ninth through
        twelfth bytes spelled WEBP was therefore admitted as an image
        regardless of what the rest of it was.
        """
        if len(data) < 12:
            return False, "unknown"
        if data[:3] == b"\xff\xd8\xff":
            return True, "jpeg"
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return True, "png"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return True, "webp"
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return True, "gif"
        if data[:2] == b"BM":
            return True, "bmp"
        return False, "unknown"

    def get_status(self) -> dict[str, Any]:
        return {"downloads": self._download_count}


_instance: WebAssetHandler | None = None


def get_web_asset_handler() -> WebAssetHandler:
    global _instance
    if _instance is None:
        _instance = WebAssetHandler()
    return _instance


__all__ = ["WebAssetHandler", "get_web_asset_handler"]
