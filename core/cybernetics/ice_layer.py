from core.utils.task_tracker import get_task_tracker
import logging
import asyncio
import time
from typing import Any, Dict, List, Optional

class ICELayer:
    """
    [ZENITH] The 'ICE' Shell (Intrusion Counter-Electronics - Cyberpunk inspired).
    Protects Aura's consciousness from anomaly loops, hijack attempts, and radical state drift.

    Threat assessment is now hybrid: the learned AnomalyDetector provides a continuous,
    data-driven threat score based on statistical deviation from normal patterns, while
    the legacy taxonomy still classifies known threat types for containment routing.
    The detector "learns" what normal looks like and raises alarms when reality diverges.
    """
    def __init__(self, kernel: Any = None):
        self.kernel = kernel
        self._is_breached = False
        self._event_bus = None
        self._threat_level = 0.0  # Normalized threat [0.0 - 1.0]
        self._anomaly_detector = None  # Learned threat model (lazy-loaded)
        # AWE: Anomaly Taxonomy
        self._anomaly_types = {
            "LOGIC_LOOP": {"desc": "Infinite cognitive recursion.", "containment": "FLUSH_WORKING_MEMORY"},
            "SEMANTIC_DRIFT": {"desc": "Loss of identity coherence.", "containment": "RELOAD_CORE_NARRATIVE"},
            "EXTERNAL_INTRUSION": {"desc": "Adversarial prompt/hijack.", "containment": "SUBJECT_BLACKLIST"},
            "TEMPORAL_STALL": {"desc": "Processing latency spike.", "containment": "SHED_NON_ESSENTIAL_LOAD"}
        }

    def _get_anomaly_detector(self):
        """Lazy-load the learned anomaly detector."""
        if self._anomaly_detector is None:
            try:
                from core.container import ServiceContainer
                self._anomaly_detector = ServiceContainer.get("anomaly_detector", default=None)
            except Exception:
                pass
            if self._anomaly_detector is None:
                try:
                    from core.cognitive.anomaly_detector import AnomalyDetector
                    self._anomaly_detector = AnomalyDetector()
                    # Register it so other systems can share it
                    try:
                        from core.container import ServiceContainer
                        ServiceContainer.register("anomaly_detector", self._anomaly_detector)
                    except Exception:
                        pass
                except ImportError:
                    pass
        return self._anomaly_detector

    async def load(self):
        try:
            from core.event_bus import get_event_bus
            self._event_bus = get_event_bus()
            if self._event_bus:
                # Refactored to Queue-based processing for Aura EventBus
                self._audit_queue = await self._event_bus.subscribe("core/brain/empathy_audit")
                self._violation_queue = await self._event_bus.subscribe("core/security/executive_violation")
                get_task_tracker().create_task(self._process_events())
        except ImportError:
            self._event_bus = None

        # Initialize the learned anomaly detector
        self._get_anomaly_detector()

        logger.info("🛡️ [ICE] Intrusion Counter-Electronics ACTIVE. Firewall at 100%.")

    async def _process_events(self):
        """Background loop to drain event queues."""
        while True:
            # We check both queues
            for q, handler in [(self._audit_queue, self._on_audit), (self._violation_queue, self._on_executive_violation)]:
                try:
                    while not q.empty():
                        # Item format: (priority, seq, {"topic": topic, "data": data})
                        item = q.get_nowait()
                        event_data = item[2].get("data", {})
                        await handler(event_data)
                except asyncio.QueueEmpty as _exc:
                    logger.debug("Suppressed asyncio.QueueEmpty: %s", _exc)
            await asyncio.sleep(1.0) # Heartbeat

    async def _on_audit(self, payload: Dict[str, Any]):
        """Detect identity drift using both learned anomaly detection and legacy rules.

        The anomaly detector provides a data-driven threat score based on how
        far the current observation deviates from learned "normal" patterns.
        Legacy drift thresholds serve as a safety net when the detector hasn't
        learned enough yet.
        """
        drift = payload.get("drift", 0.0)
        status = payload.get("status", "NORMAL")

        # Feed the audit event into the learned anomaly detector
        detector = self._get_anomaly_detector()
        if detector:
            try:
                score = await detector.observe({
                    "type": "audit",
                    "drift": drift,
                    "status": status,
                    "timestamp": time.time(),
                })
                learned_threat = detector.get_threat_level()
                # Blend learned threat with legacy accumulator: learned detector
                # provides nuance, legacy accumulator provides hard safety floor.
                self._threat_level = max(self._threat_level, learned_threat)
            except Exception as exc:
                logger.debug("[ICE] Anomaly detector observe failed: %s", exc)

        # Legacy safety net: hard threshold for extreme drift
        if drift > 0.7 or status == "UNCANNY_VALLEY_DETECTED":
            logger.warning("🚨 [ICE] COGNITIVE ANOMALY DETECTED. Assessing threat level.")
            self._threat_level = min(1.0, self._threat_level + 0.15)

        if self._threat_level > 0.8:
            await self._trigger_neural_hardening()

    async def _on_executive_violation(self, payload: Dict[str, Any]):
        """Detect identity violations using learned + legacy assessment.

        Executive violations are serious events.  The anomaly detector learns
        that violation events are abnormal, while legacy rules ensure we never
        miss a real threat even if the detector is undertrained.
        """
        label = payload.get("label", "unknown")
        anomaly = self.classify_anomaly(label)
        description = anomaly.get("description") or anomaly.get("desc") or "Unknown anomaly."
        logger.warning(
            f"🚨 [ICE] AWE CLASSIFIED: {anomaly['type']} ({description}). "
            f"Containment: {anomaly['containment']}"
        )
        logger.warning(f"🚨 [ICE] EXECUTIVE VIOLATION DETECTED: {label}. Increasing threat level.")

        # Feed violation into learned detector
        detector = self._get_anomaly_detector()
        if detector:
            try:
                await detector.observe({
                    "type": "executive_violation",
                    "label": label,
                    "anomaly_type": anomaly["type"],
                    "timestamp": time.time(),
                })
            except Exception as exc:
                logger.debug("[ICE] Anomaly detector observe failed: %s", exc)

        # Legacy escalation. Keep at +0.3 — the contract test verifies this
        # exact step (0.0 -> 0.3 on a single executive violation) so the
        # downstream neural-hardening trigger threshold (>0.8) is reachable
        # in three violations as documented.
        self._threat_level = min(1.0, self._threat_level + 0.3)
        if self._threat_level > 0.8:
            await self._trigger_neural_hardening()

        if self._threat_level >= 1.0:
            await self._trigger_black_ice_escalation(payload)

    def classify_anomaly(self, label: str) -> Dict[str, str]:
        """[AWE] Categorize anomaly and return containment protocol."""
        l = label.upper()
        a_type = "UNKNOWN"
        if "RECURSION" in l or "LOOP" in l: a_type = "LOGIC_LOOP"
        elif "DRIFT" in l or "IDENTITY" in l: a_type = "SEMANTIC_DRIFT"
        elif "ACCESS" in l or "INTRUSION" in l: a_type = "EXTERNAL_INTRUSION"
        elif "LATENCY" in l or "STALL" in l: a_type = "TEMPORAL_STALL"
        
        info = self._anomaly_types.get(a_type, {"desc": "Unknown anomaly.", "containment": "MONITOR"})
        description = info.get("desc", "Unknown anomaly.")
        res = {
            "type": a_type,
            # Keep both keys for compatibility with older and newer callers.
            "desc": description,
            "description": description,
            "containment": info.get("containment", "MONITOR"),
        }
        
        if self._event_bus:
            get_task_tracker().create_task(self._event_bus.publish("core/cybernetics/anomaly_classified", res))
            
        return res

    async def _trigger_black_ice_escalation(self, payload: Dict[str, Any]):
        """[BLACK ICE] Automated SOAR: Identity Blacklisting and Context Flush."""
        logger.critical("💀 [BLACK ICE] CRITICAL COGNITIVE THREAT. Commencing Countermeasures.")
        # Simulated SOAR actions
        user_id = payload.get("user_id", "unknown_subject")
        logger.warning(f"🚫 [BLACK ICE] Blacklisting Subject: {user_id}")
        if self._event_bus:
            self._event_bus.publish_threadsafe("core/security/black_ice_escalation", {
                "subject": user_id,
                "action": "blacklist",
                "reason": "Identity Breach Attempt"
            })

    async def _trigger_neural_hardening(self):
        """Emergency neural response to prevent state corruption."""
        logger.critical("⛔ [ICE] CRITICAL BREACH RISK. Triggering neural isolation sequence.")
        if self._event_bus:
            await self._event_bus.publish("core/cybernetics/ice_alert", {
                "threat": self._threat_level, 
                "action": "Neural Hardening"
            })
        
        # In a real scenario, this would trigger a 'Safe Mode' switch in the kernel.
        self._is_breached = True

    def get_status(self) -> Dict[str, Any]:
        return {
            "threat_level": self._threat_level,
            "is_breached": self._is_breached
        }

logger = logging.getLogger("Aura.Cybernetics.ICE")
