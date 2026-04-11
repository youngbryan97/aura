import logging
import os
import time
import threading
import psutil
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("Aura.MetabolicMonitor")

@dataclass
class MetabolismSnapshot:
    cpu_percent: float
    ram_rss_mb: float
    ram_percent: float
    disk_usage_percent: float
    llm_latency_avg: float
    health_score: float
    timestamp: float = field(default_factory=time.time)

class MetabolicMonitor:
    """Tracks physical system resources and calculates 'metabolic health' for Aura.
    
    Phase 21 A+ Upgrade: Runs in a dedicated background thread to ensure
    telemetry remains active even if the main asyncio loop stalls.
    """
    
    def __init__(self, ram_threshold_mb: int = 8192, cpu_threshold: float = 80.0):
        self.process = psutil.Process(os.getpid())
        self.ram_threshold_mb = ram_threshold_mb
        self.cpu_threshold = cpu_threshold
        
        self.latency_history: List[float] = []
        self.max_latency_history = 10
        
        self._last_snapshot: Optional[MetabolismSnapshot] = None
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Prime CPU counter
        self.process.cpu_percent()

    def start(self, interval: float = 5.0):
        """Start the background monitoring thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, args=(interval,), daemon=True, name="Aura-ANS-Metabolism")
        if self._thread:
            self._thread.start()
        else:
            logger.error("Failed to create metabolism thread.")
        logger.info("🫁 Autonomic Nervous System (Metabolism) decoupled and active.")

    def stop(self):
        """Stop the background monitoring thread."""
        self._running = False
        thread = self._thread
        if thread:
            thread.join(timeout=2.0)
            self._thread = None

    def _run_loop(self, interval: float):
        """Internal loop for background thread."""
        while self._running:
            try:
                self.get_current_metabolism()
                time.sleep(interval)
            except Exception as e:
                logger.error("Metabolic background loop error: %s", e)
                time.sleep(interval * 2)

    def record_latency(self, seconds: float):
        """Track LLM response latency (Thread-safe)."""
        with self._lock:
            self.latency_history.append(seconds)
            if len(self.latency_history) > self.max_latency_history:
                self.latency_history.pop(0)

    def _validate_metrics(self, cpu: float, rss_mb: float) -> bool:
        """Ensure metrics are within sane physical bounds."""
        if cpu < 0 or cpu > 1000: # Some OS report > 100% for multi-core but 1000 is a safe cap
            return False
        if rss_mb < 0 or rss_mb > 1024 * 64: # 64GB safety cap
            return False
        return True

    def get_current_metabolism(self) -> MetabolismSnapshot:
        """Collect current resource stats and calculate health score (Thread-safe)."""
        try:
            # 1. CPU (Per-process for Aura only)
            cpu = self.process.cpu_percent()
            
            # 2. RAM (RSS is actual physical memory used)
            mem_info = self.process.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)
            
            if not self._validate_metrics(cpu, rss_mb):
                logger.warning("⚠️ Anomalous metabolic metrics detected. Using defaults.")
                cpu = min(max(0, cpu), 100)
                rss_mb = min(max(0, rss_mb), self.ram_threshold_mb)

            system_ram_percent = psutil.virtual_memory().percent
            
            # 3. Disk
            disk = psutil.disk_usage('/').percent
            
            # 4. Latency
            with self._lock:
                avg_latency = sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0.5
            
            # 5. Calculate Health Score (1.0 = Perfect, 0.0 = Critical)
            ram_factor = max(0.0, 1.0 - (rss_mb / self.ram_threshold_mb)) if rss_mb > self.ram_threshold_mb / 2 else 1.0
            cpu_factor = max(0.0, 1.0 - (cpu / self.cpu_threshold)) if cpu > self.cpu_threshold / 2 else 1.0
            latency_factor = max(0.0, 1.0 - (avg_latency / 10.0))
            
            health_score = (ram_factor * 0.4) + (cpu_factor * 0.4) + (latency_factor * 0.2)
            health_score = max(0.0, min(1.0, health_score))
            
            snapshot = MetabolismSnapshot(
                cpu_percent=cpu,
                ram_rss_mb=rss_mb,
                ram_percent=system_ram_percent,
                disk_usage_percent=disk,
                llm_latency_avg=avg_latency,
                health_score=health_score
            )
            
            with self._lock:
                self._last_snapshot = snapshot
            
            # Phase 11.3: Push to Unified Registry (Synchronization)
            try:
                from core.state_registry import get_registry
                get_registry().sync_update(
                    health_score=health_score,
                    cpu_load=cpu,
                    memory_usage=rss_mb
                )
            except Exception as e:
                logger.debug("Registry sync failed in metabolism: %s", e)
            
            # Phase 21: Auto-emit telemetry if event bus is available
            # (Note: This might be tricky from a thread depending on event bus implementation)
            # For now, we rely on the Orchestrator or Server to pull from get_status_report
            
            return snapshot
            
        except Exception as e:
            logger.error("Failed to collect metabolic data: %s", e)
            return MetabolismSnapshot(0, 0, 0, 0, 0, 0.5)

    def get_status_report(self) -> Dict:
        """Friendly dict for telemetry (Thread-safe)."""
        with self._lock:
            s = self._last_snapshot
            
        if not s:
            s = self.get_current_metabolism()
            
        return {
            "health": round(s.health_score * 100),
            "cpu": f"{s.cpu_percent:.1f}%",
            "ram": f"{s.ram_rss_mb:.0f}MB",
            "latency": f"{s.llm_latency_avg:.2f}s",
            "status": "OPTIMAL" if s.health_score > 0.8 else "STRESSED" if s.health_score > 0.4 else "CRITICAL"
        }

class PersistentComputeCostTracker:
    """Tracks the metabolic cost of cognitive operations with disk persistence.
    
    Costs are measured in 'ergs' (a synthetic unit combining token count, 
    compute time, and hardware intensity). This data is saved to disk
    to ensure Aura tokens (ergs) are never zeroed on restart.
    """
    
    def __init__(self):
        from core.config import config
        self.state_path = config.paths.data_dir / "metabolic_state.json"
        self.total_ergs = 0.0
        self.session_start = time.time()
        self.cost_history: List[Dict] = []
        self._lock = threading.RLock()
        self._load_state()
        
    def _load_state(self):
        """Load persistent erg count from disk."""
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                self.total_ergs = data.get("total_ergs", 0.0)
                logger.info("🔋 Loaded %.2f persistent ergs.", self.total_ergs)
            except Exception as e:
                logger.warning("Failed to load metabolic state: %s", e)

    def _save_state(self):
        """Save current erg count to disk."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({
                "total_ergs": self.total_ergs,
                "last_updated": time.time()
            }))
        except Exception as e:
            logger.debug("Failed to save metabolic state: %s", e)

    def record_operation(self, op_type: str, tokens: int, duration_s: float, model_tier: str = "primary"):
        """Record the cost of a cognitive operation."""
        # Cost multipliers
        tier_mult = {"primary": 1.5, "secondary": 1.0, "tertiary": 0.5}.get(model_tier, 1.0)
        
        # Erg calculation: (tokens * base) + (duration * intensity)
        # Base: 0.1 ergs per token
        # Intensity: 10 ergs per second of GPU flight
        ergs = (tokens * 0.1) + (duration_s * 10 * tier_mult)
        
        with self._lock:
            self.total_ergs += ergs
            entry = {
                "timestamp": time.time(),
                "op_type": op_type,
                "ergs": ergs,
                "tokens": tokens,
                "duration": duration_s
            }
            self.cost_history.append(entry)
            if len(self.cost_history) > 100:
                self.cost_history.pop(0)
            
            # Persist on significant operations (every 100 ergs or 10 ops)
            if self.total_ergs % 100 < 50 or len(self.cost_history) % 10 == 0:
                self._save_state()
                
        logger.debug(f"🔋 Metabolic Cost: {ergs:.2f} ergs ({op_type})")
        return ergs

    def get_metabolic_rate(self, window_s: int = 60) -> float:
        """Calculate average ergs per second over a window."""
        now = time.time()
        cutoff = now - window_s
        
        with self._lock:
            recent = [e["ergs"] for e in self.cost_history if e["timestamp"] > cutoff]
            
        if not recent:
            return 0.0
        return sum(recent) / window_s

    def get_burn_report(self) -> Dict:
        """Summary of energy consumption."""
        with self._lock:
            return {
                "total_ergs": f"{self.total_ergs:.2f}",
                "avg_rate": f"{self.get_metabolic_rate():.3f}",
                "uptimes_s": round(time.time() - self.session_start),
                "history_len": len(self.cost_history)
            }

# Singleton Support
_cost_tracker = None

def get_cost_tracker() -> PersistentComputeCostTracker:
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = PersistentComputeCostTracker()
    return _cost_tracker