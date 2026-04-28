from core.runtime.errors import record_degradation
import logging
import httpx
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.ManifestToDevice")

class ManifestInput(BaseModel):
    url: str = Field(..., description="The remote URL of the asset to manifest/save.")
    filename: Optional[str] = Field(None, description="Optional custom filename for the saved asset.")

class ManifestToDeviceSkill(BaseSkill):
    """Downloads remote assets to the host's Desktop for permanent storage."""

    name = "manifest_to_device"
    description = "Save a remote image or file to the host device's Desktop. Use this when the user explicitly asks to 'save' or 'download' an image seen in chat."
    input_model = ManifestInput

    def __init__(self):
        super().__init__()
        # Target: ~/Desktop/Aura_Manifests/
        self.desktop_path = Path(os.path.expanduser("~/Desktop")) / "Aura_Manifests"
        try:
            self.desktop_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            record_degradation('manifest_to_device', e)
            logger.warning("Could not create Manifests directory on Desktop: %s", e)

    async def execute(self, params: ManifestInput, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the manifest action."""
        if isinstance(params, dict):
            try:
                params = ManifestInput(**params)
            except Exception as e:
                record_degradation('manifest_to_device', e)
                return {"ok": False, "error": f"Invalid parameters: {e}"}

        url = params.url
        
        # Determine filename
        if params.filename:
            filename = params.filename
        else:
            # Extract from URL or use timestamp
            timestamp = int(time.time())
            ext = ".jpg" # Default for image generation
            if ".png" in url.lower(): ext = ".png"
            elif ".webp" in url.lower(): ext = ".webp"
            filename = f"aura_manifest_{timestamp}{ext}"

        filepath = self.desktop_path / filename

        logger.info("💾 Manifesting asset from %s to %s", url, filepath)
        
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.get(url)

            if response.status_code != 200:
                return {"ok": False, "error": f"Asset retrieval failed: {response.status_code}"}

            with open(filepath, 'wb') as f:
                f.write(response.content)

            return {
                "ok": True,
                "path": str(filepath),
                "summary": f"I've manifested that asset to your desktop: {filepath}",
                "message": f"Asset secured. You can find it at: {filepath}"
            }

        except Exception as e:
            record_degradation('manifest_to_device', e)
            logger.error("Manifest failed: %s", e)
            return {"ok": False, "error": str(e)}