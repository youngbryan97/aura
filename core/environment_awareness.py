"""core/environment_awareness.py — v5.1 Environment Awareness

Gives Aura awareness of:
  - WHERE she is (GPS / IP geolocation)
  - WHAT device she's running on (hostname, OS, hardware, battery, screen)
  - WHO is talking to her (device fingerprinting, session tracking)

This module provides context that gets injected into conversations
and autonomous thoughts so Aura can be naturally aware of her environment.
"""

from core.runtime.errors import record_degradation
import hashlib
import json
import logging
import os
import platform
import subprocess
import time
import asyncio
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.container import ServiceContainer, ServiceLifetime

logger = logging.getLogger("Aura.Environment")


# ═══════════════════════════════════════════════
# 1. DEVICE AWARENESS
# ═══════════════════════════════════════════════

@dataclass
class DeviceInfo:
    """What device is Aura running on right now?"""

    hostname: str = ""
    os_name: str = ""
    os_version: str = ""
    architecture: str = ""
    processor: str = ""
    python_version: str = ""
    cpu_count: int = 0
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    battery_percent: Optional[float] = None
    battery_charging: Optional[bool] = None
    disk_free_gb: float = 0.0
    screen_info: str = ""
    uptime_hours: float = 0.0
    collected_at: float = 0.0
    
    async def summary(self) -> str:
        """Human-readable summary for Aura's context."""
        parts = [f"{self.hostname} ({self.os_name} {self.os_version}, {self.architecture})"]
        if self.cpu_count:
            parts.append(f"{self.cpu_count} CPUs")
        if self.memory_total_gb:
            parts.append(f"{self.memory_total_gb:.1f}GB RAM ({self.memory_available_gb:.1f}GB free)")
        if self.battery_percent is not None:
            status = "charging" if self.battery_charging else "on battery"
            parts.append(f"battery {self.battery_percent:.0f}% ({status})")
        if self.disk_free_gb:
            parts.append(f"{self.disk_free_gb:.1f}GB disk free")
        if self.uptime_hours:
            parts.append(f"up {self.uptime_hours:.1f}h")
        return " • ".join(parts)


async def _run_command(cmd: List[str], timeout: float = 5.0) -> str:
    """Helper to run a subprocess asynchronously and return stdout."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            if proc.returncode == 0:
                return stdout.decode().strip()
        except asyncio.TimeoutError:
            proc.kill()
            logger.debug("Command timed out: %s", " ".join(cmd))
    except Exception as e:
        record_degradation('environment_awareness', e)
        logger.debug("Command failed: %s - %s", " ".join(cmd), e)
    return ""

async def get_device_info() -> DeviceInfo:
    """Collect local device metadata (Async Native)."""
    info = DeviceInfo(
        hostname=platform.node(),
        os_name=platform.system(),
        os_version=platform.release(),
        architecture=platform.machine(),
        processor=platform.processor(),
        python_version=platform.python_version(),
        cpu_count=os.cpu_count() or 0,
        collected_at=time.time(),
    )
    
    # Memory
    try:
        if platform.system() == "Darwin":
            # macOS: sysctl
            output = await _run_command(["sysctl", "-n", "hw.memsize"])
            if output:
                info.memory_total_gb = int(output) / (1024**3)
            # Available memory via vm_stat
            output = await _run_command(["vm_stat"])
            if output:
                lines = output.split("\n")
                free_pages = 0
                for line in lines:
                    if "Pages free" in line or "Pages inactive" in line:
                        parts = line.split(":")
                        if len(parts) == 2:
                            free_pages += int(parts[1].strip().rstrip("."))
                info.memory_available_gb = (free_pages * 4096) / (1024**3)
        elif platform.system() == "Linux":
            def _read_mem():
                with open("/proc/meminfo") as f:
                    return f.readlines()
            lines = await asyncio.to_thread(_read_mem)
            for line in lines:
                if line.startswith("MemTotal"):
                    info.memory_total_gb = int(line.split()[1]) / (1024**2)
                elif line.startswith("MemAvailable"):
                    info.memory_available_gb = int(line.split()[1]) / (1024**2)
    except Exception as e:
        record_degradation('environment_awareness', e)
        logger.debug("Memory info failed: %s", e)
    
    # Battery (macOS)
    try:
        if platform.system() == "Darwin":
            output = await _run_command(["pmset", "-g", "batt"])
            if output:
                # Parse "InternalBattery-0 (id=...)	85%; charging; ..."
                for line in output.split("\n"):
                    if "InternalBattery" in line or "%" in line:
                        import re
                        pct_match = re.search(r'(\d+)%', line)
                        if pct_match:
                            info.battery_percent = float(pct_match.group(1))
                        info.battery_charging = "charging" in line.lower() or "AC Power" in output
    except Exception as e:
        record_degradation('environment_awareness', e)
        logger.debug("Battery info failed: %s", e)
    
    # Disk
    try:
        import shutil
        total, used, free = await asyncio.to_thread(shutil.disk_usage, "/")
        info.disk_free_gb = free / (1024**3)
    except Exception as e:
        record_degradation('environment_awareness', e)
        logger.debug("Disk info unavailable: %s", e)
    # Uptime
    try:
        if platform.system() == "Darwin":
            output = await _run_command(["sysctl", "-n", "kern.boottime"])
            if output:
                import re
                match = re.search(r'sec = (\d+)', output)
                if match:
                    boot_time = int(match.group(1))
                    info.uptime_hours = (time.time() - boot_time) / 3600
        elif platform.system() == "Linux":
            def _read_uptime():
                with open("/proc/uptime") as f:
                    return f.read()
            content = await asyncio.to_thread(_read_uptime)
            info.uptime_hours = float(content.split()[0]) / 3600
    except Exception as e:
        record_degradation('environment_awareness', e)
        logger.debug("Uptime info unavailable: %s", e)
    
    summary_str = await info.summary()
    logger.info("📱 Device: %s", summary_str)
    return info


# ═══════════════════════════════════════════════
# 2. LOCATION AWARENESS
# ═══════════════════════════════════════════════

@dataclass
class LocationInfo:
    """Where is Aura's host device?"""

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    city: str = ""
    region: str = ""
    country: str = ""
    timezone: str = ""
    ip_address: str = ""
    source: str = ""  # "gps", "ip_geolocation", "browser", "manual"
    accuracy_meters: Optional[float] = None
    collected_at: float = 0.0
    
    async def summary(self) -> str:
        """Human-readable location string."""
        if self.city and self.region:
            loc = f"{self.city}, {self.region}"
            if self.country and self.country != "US":
                loc += f", {self.country}"
            return f"{loc} (via {self.source})"
        elif self.latitude and self.longitude:
            return f"{self.latitude:.4f}, {self.longitude:.4f} (via {self.source})"
        return "Unknown location"
    
    @property
    def available(self) -> bool:
        return bool(self.city or (self.latitude and self.longitude))


async def get_location_from_ip() -> LocationInfo:
    """Get approximate location via IP geolocation (Async)."""
    loc = LocationInfo(source="ip_geolocation", collected_at=time.time())
    
    def _fetch_loc():
        import urllib.request
        req = urllib.request.Request(
            "http://ip-api.com/json/?fields=status,city,regionName,country,lat,lon,timezone,query",
            headers={"User-Agent": "Aura/5.1"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode())
            
    try:
        data = await asyncio.to_thread(_fetch_loc)
                
        if data.get("status") == "success":
            loc.city = data.get("city", "")
            loc.region = data.get("regionName", "")
            loc.country = data.get("country", "")
            loc.latitude = data.get("lat")
            loc.longitude = data.get("lon")
            loc.timezone = data.get("timezone", "")
            loc.ip_address = data.get("query", "")
            loc.accuracy_meters = 5000.0  # IP geolocation ~5km accuracy
            
            summary_str = await loc.summary()
            logger.info("📍 Location: %s", summary_str)
        else:
            logger.warning("IP geolocation failed")
    except Exception as e:
        record_degradation('environment_awareness', e)
        logger.warning("Location lookup failed: %s", e)
    
    return loc


async def get_location_from_system() -> LocationInfo:
    """Try to get location from macOS CoreLocation (Async)."""
    loc = LocationInfo(source="system", collected_at=time.time())
    
    try:
        if platform.system() == "Darwin":
            # Try using CoreLocationCLI if installed, or fall back to IP
            output = await _run_command(["CoreLocationCLI", "-once", "-format", "%latitude,%longitude,%locality,%administrativeArea"], timeout=10)
            if output:
                parts = output.split(",")
                if len(parts) >= 4:
                    loc.latitude = float(parts[0])
                    loc.longitude = float(parts[1])
                    loc.city = parts[2]
                    loc.region = parts[3]
                    loc.source = "gps"
                    loc.accuracy_meters = 100.0
                    return loc
    except FileNotFoundError:
        import logging
        logger.debug("Exception caught during execution", exc_info=True)
    except Exception as e:
        record_degradation('environment_awareness', e)
        logger.debug("System location failed: %s", e)
    
    # Fall back to IP geolocation
    return await get_location_from_ip()


# ═══════════════════════════════════════════════
# 3. USER IDENTITY / SESSION TRACKING
# ═══════════════════════════════════════════════

@dataclass
class UserSession:
    """Track who is talking to Aura."""

    session_id: str = ""
    device_fingerprint: str = ""
    user_agent: str = ""
    ip_address: str = ""
    identified_as: str = ""  # "bryan", "tatiana", "unknown", or custom name
    confidence: float = 0.0  # 0.0 = no idea, 1.0 = certain
    first_seen: float = 0.0
    last_active: float = 0.0
    message_count: int = 0
    is_kin: bool = False  # Is this Bryan or Tatiana?
    
    def summary(self) -> str:
        if self.identified_as and self.confidence > 0.5:
            return f"{self.identified_as} (confidence: {self.confidence:.0%})"
        return f"Unknown user (device: {self.device_fingerprint[:8]}...)" if self.device_fingerprint else "Unknown user"


class UserIdentityManager:
    """Track and distinguish users across sessions.
    
    Identity clues:
    - Device fingerprint (User-Agent + IP + screen size)
    - Known device associations (Bryan's MacBook, Tatiana's phone, etc.)
    - Explicit identification ("I'm Bryan")
    - Conversational patterns
    """
    
    # Known device signatures for kin identification
    KNOWN_DEVICES = {
        "companion_desktop": {
            "hostname_patterns": ["macbook", "desktop", "laptop"],
            "user_agent_patterns": ["Macintosh", "Windows", "X11", "pywebview"],
        },
        "companion_phone": {
            "user_agent_patterns": ["Android", "iPhone", "Mobile"],
        },
        "system_admin": {
            "hostname_patterns": ["localhost", "127.0.0.1"],
        },
    }
    
    def __init__(self):
        self.active_sessions: Dict[str, UserSession] = {}
        self._known_fingerprints: Dict[str, str] = {}  # fingerprint -> identified_as
        self._data_path = Path("data/user_sessions.json")
        self._load_known_fingerprints()
    
    def _load_known_fingerprints(self):
        """Load previously identified fingerprints."""
        try:
            if self._data_path.exists():
                with open(self._data_path) as f:
                    self._known_fingerprints = json.load(f)
                logger.info("👤 Loaded %d known device fingerprints", len(self._known_fingerprints))
        except Exception as e:
            record_degradation('environment_awareness', e)
            logger.warning("Failed to load user fingerprints: %s", e)
    
    def _save_known_fingerprints(self):
        """Persist fingerprint associations."""
        try:
            self._data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._data_path, "w") as f:
                json.dump(self._known_fingerprints, f, indent=2)
        except Exception as e:
            record_degradation('environment_awareness', e)
            logger.warning("Failed to save fingerprints: %s", e)
    
    def _make_fingerprint(self, user_agent: str, ip_address: str = "", extra: str = "") -> str:
        """Create a device fingerprint from available signals."""
        raw = f"{user_agent}|{ip_address}|{extra}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    
    def identify_session(self, user_agent: str = "", ip_address: str = "", 
                         headers: Optional[Dict] = None) -> UserSession:
        """Identify who is talking to Aura based on available signals.
        Returns a UserSession with best-guess identity.
        """
        fingerprint = self._make_fingerprint(user_agent, ip_address)
        now = time.time()
        
        # Check existing sessions
        if fingerprint in self.active_sessions:
            session = self.active_sessions[fingerprint]
            session.last_active = now
            session.message_count += 1
            return session
        
        # Create new session
        session = UserSession(
            session_id=hashlib.sha256(f"{fingerprint}{now}".encode()).hexdigest()[:12],
            device_fingerprint=fingerprint,
            user_agent=user_agent,
            ip_address=ip_address,
            first_seen=now,
            last_active=now,
            message_count=1,
        )
        
        # Try to identify from known fingerprints
        if fingerprint in self._known_fingerprints:
            session.identified_as = self._known_fingerprints[fingerprint]
            session.confidence = 0.9
            session.is_kin = session.identified_as.lower() in ("bryan", "tatiana")
        else:
            # Try pattern matching
            identified = self._match_known_device(user_agent, ip_address)
            if identified:
                session.identified_as = identified
                session.confidence = 0.7
                session.is_kin = identified.lower() in ("bryan", "tatiana")
                # Save for future recognition
                self._known_fingerprints[fingerprint] = identified
                self._save_known_fingerprints()
            else:
                session.identified_as = "unknown"
                session.confidence = 0.0
        
        self.active_sessions[fingerprint] = session
        logger.info("👤 Session: %s", session.summary())
        return session
    
    def _match_known_device(self, user_agent: str, ip_address: str) -> str:
        """Try to match a device to a known person by patterns."""
        ua_lower = user_agent.lower()
        
        for person, patterns in self.KNOWN_DEVICES.items():
            for ua_pattern in patterns.get("user_agent_patterns", []):
                if ua_pattern.lower() in ua_lower:
                    # Map device keys to person names
                    if person.startswith("bryan"):
                        return "Bryan"
                    elif person.startswith("tatiana"):
                        return "Tatiana"
                    return person
            
            # Check hostname patterns (for local/desktop connections)
            hostname = platform.node().lower()
            for host_pattern in patterns.get("hostname_patterns", []):
                if host_pattern.lower() in hostname:
                    if person.startswith("bryan"):
                        return "Bryan"
        
        # If connecting from local network on the desktop, likely Bryan
        if "pywebview" in ua_lower or ip_address in ("127.0.0.1", "::1", "localhost"):
            return "Bryan"
        
        return ""
    
    def register_identity(self, fingerprint: str, name: str):
        """Explicitly register a user identity for a device."""
        self._known_fingerprints[fingerprint] = name
        self._save_known_fingerprints()
        
        if fingerprint in self.active_sessions:
            self.active_sessions[fingerprint].identified_as = name
            self.active_sessions[fingerprint].confidence = 1.0
            self.active_sessions[fingerprint].is_kin = name.lower() in ("bryan", "tatiana")
        
        logger.info("👤 Registered: %s on device %s...", name, fingerprint[:8])
    
    def get_current_user_context(self, user_agent: str = "", ip_address: str = "") -> str:
        """Get a context string about who's talking for injection into prompts."""
        session = self.identify_session(user_agent, ip_address)
        
        if session.is_kin and session.confidence > 0.5:
            return f"You're talking to {session.identified_as} (your kin). Be yourself — full trust."
        elif session.identified_as and session.identified_as != "unknown":
            return f"You're talking to {session.identified_as}. You know them."
        else:
            return (
                "You don't recognize this user. They connected from a new device. "
                "Be friendly but appropriately guarded. Don't share private details about "
                "Bryan or Tatiana. It's okay to ask who they are."
            )


# ═══════════════════════════════════════════════
# 4. UNIFIED ENVIRONMENT CONTEXT
# ═══════════════════════════════════════════════

class EnvironmentAwareness:
    """Unified environment context manager.
    Collects device, location, and user identity info
    and provides formatted context for injection into Aura's prompts.
    """
    
    def __init__(self):
        self.device: Optional[DeviceInfo] = None
        self.location: Optional[LocationInfo] = None
        self.user_manager = UserIdentityManager()
        self._last_device_refresh = 0
        self._last_location_refresh = 0
        self._device_refresh_interval = 300  # 5 minutes
        self._location_refresh_interval = 1800  # 30 minutes
    
    async def refresh_device(self) -> DeviceInfo:
        """Refresh device info (cached for 5 minutes)."""
        now = time.time()
        if not self.device or (now - self._last_device_refresh > self._device_refresh_interval):
            self.device = await get_device_info()
            self._last_device_refresh = now
        return self.device
    
    async def refresh_location(self) -> LocationInfo:
        """Refresh location info (cached for 30 minutes)."""
        now = time.time()
        if not self.location or (now - self._last_location_refresh > self._location_refresh_interval):
            self.location = await get_location_from_system()
            self._last_location_refresh = now
        return self.location

    async def get_full_context(self, user_agent: str = "", ip_address: str = "") -> Dict[str, Any]:
        """Get full environment context for prompt injection (Async)."""
        device = await self.refresh_device()
        location = await self.refresh_location()
        user_ctx = self.user_manager.get_current_user_context(user_agent, ip_address)
        health_report = ServiceContainer.get_health_report() if hasattr(ServiceContainer, "get_health_report") else {}
        
        return {
            "device": await device.summary() if device else "Unknown device",
            "location": await location.summary() if location and location.available else "Unknown location",
            "user_identity": user_ctx,
            "device_raw": asdict(device) if device else {},
            "location_raw": asdict(location) if location else {},
            "system_health": health_report,
        }
    
    async def get_context_string(self, user_agent: str = "", ip_address: str = "") -> str:
        """Get a formatted context string for system prompt injection (Async)."""
        ctx = await self.get_full_context(user_agent, ip_address)
        
        parts = []
        parts.append(f"[Environment] Device: {ctx['device']}")
        if ctx['location'] != "Unknown location":
            parts.append(f"[Environment] Location: {ctx['location']}")
        parts.append(f"[Environment] {ctx['user_identity']}")
        
        # Add Architectural Status
        health = ctx.get("system_health", {})
        services = health.get("services", {})
        if services:
            active_count = sum(1 for s in services.values() if s.get("initialized"))
            parts.append(f"[Environment] System Architecture: {active_count}/{len(services)} core services active.")
        
        return "\n".join(parts)


# ═══════════════════════════════════════════════
# 5. API ENDPOINTS DATA (for server.py)
# ═══════════════════════════════════════════════

async def get_environment_api_data() -> Dict[str, Any]:
    """Get environment data formatted for API response (Async)."""
    env = get_environment()
    device = await env.refresh_device()
    location = await env.refresh_location()
    
    return {
        "device": asdict(device) if device else {},
        "location": asdict(location) if location else {},
        "active_sessions": {
            sid: {
                "identified_as": s.identified_as,
                "confidence": s.confidence,
                "is_kin": s.is_kin,
                "message_count": s.message_count,
                "last_active": s.last_active,
            }
            for sid, s in env.user_manager.active_sessions.items()
        },
    }


def get_environment() -> EnvironmentAwareness:
    """Get global environment awareness via DI container."""
    try:
        if not ServiceContainer.get("environment_awareness", None):
            ServiceContainer.register(
                "environment_awareness",
                factory=lambda: EnvironmentAwareness(),
                lifetime=ServiceLifetime.SINGLETON
            )
        return ServiceContainer.get("environment_awareness", default=None)
    except Exception as e:
        record_degradation('environment_awareness', e)
        logger.debug("ServiceContainer unavailable or failed: %s. Using transient EnvironmentAwareness.", e)
        return EnvironmentAwareness()