from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from core.runtime.errors import record_degradation
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlsplit
from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.network_gateway import get_network_gateway

logger = logging.getLogger("Skills.ManifestToDevice")

_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "application/pdf": ".pdf",
    "text/plain": ".txt",
}


def _safe_flat_filename(value: str) -> str:
    """Return a caller-visible filename that cannot escape the target folder."""

    filename = str(value or "").strip()
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
        or Path(filename).name != filename
    ):
        raise ValueError("filename must be one non-empty file name without a path")
    if len(filename.encode("utf-8")) > 240:
        raise ValueError("filename is too long")
    return filename


def _response_header(headers: object, name: str) -> str:
    if not isinstance(headers, dict):
        return ""
    wanted = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == wanted:
            return str(value or "").strip()
    return ""


def _default_filename(url: str, content_type: str) -> str:
    parsed = urlsplit(url)
    source_name = unquote(Path(parsed.path).name).strip()
    if source_name:
        try:
            return _safe_flat_filename(source_name)
        except ValueError:
            pass

    media_type = content_type.partition(";")[0].strip().lower()
    extension = _CONTENT_TYPE_EXTENSIONS.get(media_type, ".bin")
    return f"aura_manifest_{int(time.time())}{extension}"

class ManifestInput(BaseModel):
    url: str = Field(..., description="The remote URL of the asset to manifest/save.")
    filename: Optional[str] = Field(None, description="Optional custom filename for the saved asset.")

class ManifestToDeviceSkill(BaseSkill):
    """Downloads remote assets to the host's Desktop for permanent storage."""
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT


    name = "manifest_to_device"
    description = "Save a remote image or file to the host device's Desktop. Use this when the user explicitly asks to 'save' or 'download' an image seen in chat."
    input_model = ManifestInput

    def __init__(self):
        super().__init__()
        # Construction happens during catalog discovery, before a Will decision
        # exists. Keep it side-effect free; execute() owns all host mutation.
        self.desktop_path = Path(os.path.expanduser("~/Desktop")) / "Aura_Manifests"

    async def execute(self, params: ManifestInput, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the manifest action."""
        if isinstance(params, dict):
            try:
                params = ManifestInput(**params)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('manifest_to_device', e)
                return {"ok": False, "error": f"Invalid parameters: {e}"}

        url = params.url.strip()
        try:
            parsed = urlsplit(url)
            if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
                raise ValueError("url must be an absolute http(s) URL")
            requested_filename = (
                _safe_flat_filename(params.filename) if params.filename else None
            )
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": f"Invalid manifest request: {exc}"}

        logger.info("💾 Downloading asset from %s", url)

        try:
            response = await get_network_gateway().request_async(
                "GET",
                url,
                timeout=60,
                source="skills.manifest_to_device.download",
                read_only=True,
                max_response_bytes=_MAX_MANIFEST_BYTES,
                public_network_only=True,
            )

            status_code = int(response.get("status_code") or 0)
            if status_code < 200 or status_code >= 300 or not response.get("ok", False):
                detail = str(response.get("error") or "HTTP request did not succeed")
                return {
                    "ok": False,
                    "error": f"Asset retrieval failed ({status_code}): {detail}",
                }

            payload = bytes(response.get("content") or b"")
            if not payload:
                return {"ok": False, "error": "Asset retrieval returned an empty body."}
            if len(payload) > _MAX_MANIFEST_BYTES:
                return {
                    "ok": False,
                    "error": (
                        f"Asset is {len(payload)} bytes; the governed download limit is "
                        f"{_MAX_MANIFEST_BYTES} bytes."
                    ),
                }

            content_type = _response_header(response.get("headers"), "content-type")
            filename = requested_filename or _default_filename(
                str(response.get("url") or url), content_type
            )
            filepath = self.desktop_path / filename
            if filepath.parent != self.desktop_path:
                raise ValueError("resolved manifest path escaped the target directory")

            file_gateway = get_file_write_gateway()
            await file_gateway.ensure_directory_async(
                self.desktop_path,
                source="skills.manifest_to_device.desktop_path",
            )

            await file_gateway.write_bytes_async(
                filepath,
                payload,
                source="skills.manifest_to_device.asset",
            )

            return {
                "ok": True,
                "path": str(filepath),
                "source_url": str(response.get("url") or url),
                "bytes_written": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "content_type": content_type or None,
                "summary": f"Downloaded {len(payload)} bytes to {filepath}",
                "message": f"The asset is saved at {filepath}.",
            }

        except (OSError, ConnectionError, RuntimeError, TimeoutError, TypeError, ValueError) as e:
            record_degradation('manifest_to_device', e)
            logger.error("Manifest failed: %s", e)
            return {"ok": False, "error": str(e)}
