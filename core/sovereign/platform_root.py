"""core/sovereign/platform_root.py
====================================
Sovereign Platform Root for Metal Persistence.
Ensures the MTLCompilerService remains active and responsive by maintaining 
a direct, high-conductivity connection to the hardware.
"""

import asyncio
import logging
import time
import threading
import subprocess
from typing import Optional, Dict, Any

from core.runtime.desktop_boot_safety import inprocess_mlx_metal_enabled

try:
    import psutil
except ImportError:
    psutil = None

_METAL_ALLOWED, _METAL_REASON = inprocess_mlx_metal_enabled()
if _METAL_ALLOWED:
    try:
        import mlx.core as mx
    except ImportError:
        mx = None
else:
    mx = None

logger = logging.getLogger("Aura.PlatformRoot")

class PlatformRoot:
    """The Mycelial Root for Hardware Connectivity.
    
    Pins the Metal compiler to the Aura process lifecycle and prevents 
    aggressive OS reclamation or XPC lookup failures.
    """
    
    _instance: Optional["PlatformRoot"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PlatformRoot, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
            
        self._initialized = True
        self.running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_pulse = 0.0
        self.device_active = False
        self._pulse_interval = 15.0
        self._metal_allowed, self._metal_reason = _METAL_ALLOWED, _METAL_REASON
        
        logger.info("🌿 [PLATFORM ROOT] Sovereign Root initialized. Building direct hardware connection...")
        self._connect_hardware()

    def _connect_hardware(self):
        """Establish the direct link to the Metal device."""
        if mx is None:
            logger.error("❌ [PLATFORM ROOT] MLX not found. Hardware binding impossible.")
            return
        if not self._metal_allowed:
            logger.info(
                "🛡️ [PLATFORM ROOT] Skipping Metal hardware binding (%s).",
                self._metal_reason,
            )
            self.device_active = False
            return

        try:
            # Force device selection and basic allocation
            mx.set_default_device(mx.gpu)
            logger.info("✅ [PLATFORM ROOT] Hardware Bound: %s", mx.default_device())
            self.device_active = True
            self.pulse() # Initial pulse
        except Exception as e:
            logger.error("❌ [PLATFORM ROOT] Fatal Hardware Binding Error: %s", e)
            self.device_active = False

    def pulse(self, force_wake: bool = False):
        """Execute a sub-conductive hardware pulse to keep the compiler active."""
        if not self._metal_allowed:
            return False
        if mx is None or not self.device_active:
            if force_wake:
                self._connect_hardware()
            else:
                return

        try:
            start = time.monotonic()
            # Construct a small graph that requires compilation/evaluation
            # This 'pins' the MTLCompilerService to our process
            if force_wake:
                # More intensive op to force a major lookup
                a = mx.random.normal((100, 100))
                b = mx.random.normal((100, 100))
                c = mx.matmul(a, b)
                mx.eval(c)
                logger.info("🔥 [PLATFORM ROOT] Force wake pulse executed.")
            else:
                a = mx.array([1.0, 2.0, 3.0])
                b = mx.array([4.0, 5.0, 6.0])
                mx.eval(a * b + (a / b))
            
            self._last_pulse = time.monotonic()
            latency = (self._last_pulse - start) * 1000
            if not force_wake:
                logger.debug("⚡ [PLATFORM ROOT] Sub-conductive pulse: %.2fms", latency)
            return True
        except Exception as e:
            msg = str(e)
            if "MTLCompilerService" in msg or "error 3" in msg:
                logger.critical("🚨 [PLATFORM ROOT] COMPILER DISCONNECT DETECTED: %s", msg)
                self.device_active = False
                # Trigger emergency re-build
                self._connect_hardware()
            else:
                logger.error("[PLATFORM ROOT] Pulse failure: %s", e)
            return False

    def force_compiler_wake(self):
        """Explicitly pull the MTLCompilerService back into memory."""
        logger.info("💉 [PLATFORM ROOT] Triggering FORCE compiler wake...")
        return self.pulse(force_wake=True)

    async def start_monitor(self):
        """Starts the background persistence loop."""
        if self.running:
            return
            
        self.running = True
        logger.info("🌿 [PLATFORM ROOT] Persistence Monitor Active.")
        
        while self.running:
            try:
                # 1. Dynamic Heartbeat: Pulse faster under RAM pressure
                mem_percent = 0.0
                if psutil:
                    mem_percent = psutil.virtual_memory().percent
                
                if mem_percent > 85.0:
                    self._pulse_interval = 2.0
                    logger.warning("📉 [PLATFORM ROOT] High RAM (%s%%): Increasing pulse frequency to 2s.", mem_percent)
                else:
                    self._pulse_interval = 15.0

                self.pulse()
                
                # 2. Verify MTLCompilerService via system-level probe
                if time.monotonic() - self._last_pulse > 30:
                     # If we haven't pulsed successfully, check if the process even exists
                     self._verify_sys_process()
                
                await asyncio.sleep(self._pulse_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[PLATFORM ROOT] Monitor loop error: %s", e)
                await asyncio.sleep(5.0)

    def _verify_sys_process(self):
        """System-level check for the compiler service."""
        try:
            # Check if MTLCompilerService is in our process pool
            out = subprocess.check_output(["ps", "aux"], text=True, stderr=subprocess.DEVNULL)
            if "MTLCompilerService" not in out:
                logger.warning("⚠️ [PLATFORM ROOT] MTLCompilerService not visible in process list.")
                # We can't 'start' it directly easily as it's an XPC service,
                # but a high-level MLX op usually triggers a lookup.
                self.force_compiler_wake()
        except Exception as _e:
            logger.debug('Ignored Exception in platform_root.py: %s', _e)

    def stop(self):
        self.running = False

def get_platform_root() -> PlatformRoot:
    return PlatformRoot()
