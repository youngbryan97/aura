from core.runtime.errors import record_degradation
import logging
import requests
import time
from typing import Tuple, Optional

logger = logging.getLogger("Aura.GpsInterface")

class GpsInterface:
    """Live IP-Based GPS Interface for Location-Awareness."""
    
    def __init__(self):
        import os

        self._fallback_lat = float(os.environ.get("AURA_FALLBACK_LAT", "0.0"))
        self._fallback_lon = float(os.environ.get("AURA_FALLBACK_LON", "0.0"))
        self.latitude = self._fallback_lat
        self.longitude = self._fallback_lon
        self.source = "unresolved"
        self.confidence = 0.0
        self._initialized = False
        self._last_fetch_mono = 0.0
        
    def _fetch_location(self):
        """Fetch real GPS coordinates or return fallback with non-sticky retry."""
        now = time.monotonic()
        # Only retry once every 5 minutes if initial fetch failed
        if self._initialized and (now - self._last_fetch_mono < 300):
            return

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = requests.get("https://ipinfo.io/json", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    loc = data.get("loc", "").split(",")
                    if len(loc) == 2:
                        self.latitude = float(loc[0])
                        self.longitude = float(loc[1])
                        self.source = "ipinfo"
                        self.confidence = 0.65
                        self._initialized = True
                        self._last_fetch_mono = now
                        logger.info(f"📍 GPS: located system at {data.get('city', 'Unknown City')}: {self.latitude}, {self.longitude}")
                        return
            except Exception as e:
                record_degradation('gps_interface', e)
                if attempt < max_retries:
                    logger.warning(f"📍 GPS attempt {attempt+1} failed: {e}. Retrying...")
                    time.sleep(0.5)
                else:
                    logger.error("📍 GPS location lookup failed after retries. Using configured fallback.")
        
        self._initialized = True # Mark as "attempted" to trigger the 5-min cooldown
        self._last_fetch_mono = now
        self.latitude = self._fallback_lat
        self.longitude = self._fallback_lon
        self.source = "configured_fallback"
        self.confidence = 0.0

    def get_coords(self) -> Tuple[float, float]:
        """Returns live coordinates."""
        self._fetch_location()
        return self.latitude, self.longitude

    def get_status(self) -> dict:
        self._fetch_location()
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "source": self.source,
            "confidence": self.confidence,
        }
