from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import logging
import time
import asyncio
from typing import Any, Dict, List, Optional
import math
try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("Aura.Cybernetics.Tricorder")

class Tricorder:
    """
    [ZENITH] The 'Tricorder' Diagnostic Organ (Star Trek inspired).
    A sensor fusion layer that scans hardware, software, and cognitive vitality.
    """
    def __init__(self, kernel: Any = None):
        self.kernel = kernel
        self._is_active = False
        self._last_scan: Dict[str, Any] = {}
        self._event_bus: Optional[Any] = None
        self._violation_queue: Optional[asyncio.Queue] = None
        self._empathy_queue: Optional[asyncio.Queue] = None
        # Consolidation: Merged HealthMonitor logic
        self.consecutive_errors = 0
        self.total_errors = 0
        self.healthy = True
        self.last_error: Optional[str] = None
        # Precrime Engine: Anomaly Detection
        self._latency_buffer: List[float] = []
        self._anomaly_threshold = 2.5
        # Sibyl System: Unified Behavioral Hue
        self._hue_score = 0.0
        self._factors = {"deviation": 0.0, "empathy": 1.0, "volatility": 0.0}
        # CASIE: User Emotional Analysis
        self._user_scores = {"aggression": 0.0, "openness": 0.5, "trust": 0.5}
        self._casie_lexicon = {
            "agg": ["threat", "demand", "never", "refuse", "now", "consequences", "wrong"],
            "opn": ["maybe", "perhaps", "consider", "explore", "open", "suggest"],
            "tru": ["trust", "honest", "believe", "promise", "agree", "reliable"]
        }

    def track_latency(self, latency: float):
        """Track execution latency and check for Precrime anomalies."""
        self._latency_buffer.append(latency)
        if len(self._latency_buffer) > 50:
            self._latency_buffer.pop(0)
        
        # Update volatility factor for Hue
        self._factors["volatility"] = min(1.0, latency / 1000.0)
        self._recalculate_hue()

        if len(self._latency_buffer) > 10:
            import math
            mean = sum(self._latency_buffer) / len(self._latency_buffer)
            variance = sum((x - mean) ** 2 for x in self._latency_buffer) / len(self._latency_buffer)
            std_dev = math.sqrt(variance) or 0.001
            z_score = abs(latency - mean) / std_dev
            
            if z_score > self._anomaly_threshold:
                logger.warning(f"🚨 [PRECRIME] Latency Anomaly Detected: z={z_score:.2f}. Foreseeing system stall.")
                if self._event_bus:
                    self._event_bus.publish_threadsafe("core/cybernetics/precrime_alert", {
                        "z_score": z_score,
                        "latency": latency,
                        "mean": mean
                    })

    def _recalculate_hue(self):
        """Behavioral risk score: weighted composite of deviation, empathy, volatility (0-300)."""
        d, e, v = self._factors["deviation"], self._factors["empathy"], self._factors["volatility"]
        raw = (d * 4.0 + (1.0 - e) * 2.0 + v * 1.0) / 7.0
        self._hue_score = raw * 300
        
        hue_label = "CLEAR"
        if self._hue_score > 200: hue_label = "CRITICAL"
        elif self._hue_score > 150: hue_label = "OPAQUE"
        elif self._hue_score > 80: hue_label = "CLOUDY"
        
        if self._event_bus:
            self._event_bus.publish_threadsafe("core/cybernetics/hue_reading", {
                "score": self._hue_score,
                "label": hue_label,
                "factors": self._factors
            })

    def track_error(self, error: Exception):
        """Track an error and update health status (Consolidated from HealthMonitor)."""
        self.consecutive_errors += 1
        self.total_errors += 1
        self.last_error = str(error)
        if self.consecutive_errors >= 5:
            self.healthy = False
        return self.healthy

    def reset_errors(self):
        """Reset the consecutive error counter on success."""
        self.consecutive_errors = 0
        self.healthy = True

    async def load(self):
        try:
            from core.event_bus import get_event_bus
            self._event_bus = get_event_bus()
            if self._event_bus:
                # Standardized Sibyl Hue Factors
                # Zenith 2.0 Fix: subscribe() returns a queue, doesn't take a callback
                self._violation_queue = await self._event_bus.subscribe("core/security/executive_violation")
                # Start a background task to process the queue
                get_task_tracker().create_task(self._process_violations())
                
                # Subscribe to empathy updates - also returns a queue
                self._empathy_queue = await self._event_bus.subscribe("core/brain/empathy_audit")
                get_task_tracker().create_task(self._process_empathy())
        except ImportError:
            self._event_bus = None
        logger.info("📡 [TRICORDER] Multi-modal Diagnostic Sensor ONLINE.")

    async def _process_violations(self):
        """Process executive violations from the queue."""
        if not hasattr(self, '_violation_queue') or not self._violation_queue:
            return
        while True:
            try:
                # EventBus returns tuple of (topic, meta, item)
                _, _, item = await self._violation_queue.get()
                await self._on_violation(item)
            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation('tricorder', e)
                logger.debug("Tricorder violation processing error: %s", e)

    async def _on_violation(self, _):
        """Update deviation factor for Hue."""
        self._factors["deviation"] = min(1.0, self._factors["deviation"] + 0.2)
        self._recalculate_hue()

    async def _process_empathy(self):
        """Process empathy updates from the queue."""
        if not hasattr(self, '_empathy_queue') or not self._empathy_queue:
            return
        while True:
            try:
                _, _, item = await self._empathy_queue.get()
                await self._on_empathy_update(item)
            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation('tricorder', e)
                logger.debug("Tricorder empathy processing error: %s", e)

    async def _on_empathy_update(self, payload: Dict[str, Any]):
        """Update empathy factor for Hue."""
        drift = payload.get("drift", 0.0)
        self._factors["empathy"] = 1.0 - drift
        self._recalculate_hue()

    def score_user_message(self, text: str) -> Dict[str, Any]:
        """
        [CASIE] Analyze user text for emotional markers and return strategy.
        Implementation: Multi-feature NLP scoring (keyword bags, caps, punctuation).
        """
        t = text.lower()
        words = t.split()
        
        # 1. Dimension Scoring
        agg = sum(0.15 for w in self._casie_lexicon["agg"] if w in t)
        opn = sum(0.12 for w in self._casie_lexicon["opn"] if w in t)
        tru = sum(0.12 for w in self._casie_lexicon["tru"] if w in t)
        
        # 2. Structural Features
        cap_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
        excl_count = text.count("!")
        
        agg += (cap_ratio * 0.5) + (excl_count * 0.1)
        
        # 3. Update State
        self._user_scores = {
            "aggression": min(1.0, agg),
            "openness": min(1.0, opn + 0.3), # Baseline openness
            "trust": min(1.0, tru + 0.3)
        }
        
        # 4. Strategy Selection (Deus Ex Priority Rule System)
        strategy = "NEUTRAL"
        desc = "Balanced state. Proceed with standard protocol."
        
        if self._user_scores["aggression"] > 0.6:
            strategy = "DE-ESCALATE"
            desc = "High aggression detected. Use neutral framing and acknowledge feelings."
        elif self._user_scores["trust"] < 0.3:
            strategy = "BUILD_RAPPORT"
            desc = "Low trust marker. Share vulnerability and find common ground."
        elif self._user_scores["openness"] > 0.6:
            strategy = "EXPAND"
            desc = "User is receptive. Prime moment for new conceptual links."
            
        result = {
            "scores": self._user_scores,
            "strategy": strategy,
            "description": desc,
            "timestamp": time.time()
        }
        
        if self._event_bus:
            # publish is async, so we wrap it in a task if we are in a sync method or want fire-and-forget
            get_task_tracker().create_task(self._event_bus.publish("core/cybernetics/casie_analysis", result))
            
        return result

    async def scan(self, state: Any) -> Dict[str, Any]:
        """
        Performs a full system scan across hardware, software, and cognitive dimensions.
        """
        if not self._is_active:
            return {"status": "offline"}

        # 1. Hardware Dimension
        hw_stats = {}
        if psutil:
            hw_stats = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }

        # 2. Cognitive Dimension
        vitality = getattr(state, 'vitality', 1.0)
        phi = getattr(state.cognition, 'phi_estimate', 0.0) if hasattr(state, 'cognition') else 0.0
        
        # 3. Temporal Dimension (Latency)
        latency = 0.0
        if hasattr(state, 'soma') and hasattr(state.soma, 'latency'):
            latency = state.soma.latency.get('last_thought_ms', 0.0)

        report = {
            "timestamp": time.time(),
            "hardware": hw_stats,
            "cognition": {
                "vitality": vitality,
                "phi_estimate": phi,
                "latency_ms": latency
            },
            "status": "AURA_NOMINAL" if vitality > 0.8 else "AURA_DEGRADED"
        }

        self._last_scan = report

        # Publish to Mycelial network
        if self._event_bus:
            # publish is async
            get_task_tracker().create_task(self._event_bus.publish("core/cybernetics/tricorder_scan", report))

        return report

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_active": self._is_active,
            "last_scan": self._last_scan
        }
