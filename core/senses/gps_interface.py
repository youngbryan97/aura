import logging
import requests
import time
from typing import Tuple, Optional

class GpsInterface:
    """Live IP-Based GPS Interface for Location-Awareness."""
    
    def __init__(self):
        # Default fallback coords
        self._fallback_lat = 37.7749 
        self._fallback_lon = -122.4194
        self.latitude = self._fallback_lat
        self.longitude = self._fallback_lon
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
                        self._initialized = True
                        self._last_fetch_mono = now
                        logger.info(f"📍 GPS: located system at {data.get('city', 'Unknown City')}: {self.latitude}, {self.longitude}")
                        return
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"📍 GPS attempt {attempt+1} failed: {e}. Retrying...")
                    time.sleep(0.5)
                else:
                    logger.error("📍 GPS hardware failure after retries. Using fallback.")
        
        self._initialized = True # Mark as "attempted" to trigger the 5-min cooldown
        self._last_fetch_mono = now
        self.latitude = self._fallback_lat
        self.longitude = self._fallback_lon

    def get_coords(self) -> Tuple[float, float]:
        """Returns live coordinates."""
        self._fetch_location()
        return self.latitude, self.longitude
