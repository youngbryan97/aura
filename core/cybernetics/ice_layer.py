import asyncio
import logging
import math
import time
import uuid
from collections import deque
from typing import Any

from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker


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
        self._statistical_novelty = 0.0
        self._anomaly_detector = None  # Learned threat model (lazy-loaded)
        self._running = True
        self._task: asyncio.Task | None = None
        self._last_threat_update = time.monotonic()
        self._signal_times: deque[float] = deque(maxlen=16)
        self._signal_window_s = 3600.0
        self._stable_audits = 0
        self._incident: dict[str, Any] | None = None
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
            except (ImportError, AttributeError, RuntimeError) as exc:
                record_degradation("ice_layer", exc)
                logger.debug("[ICE] Shared anomaly detector lookup failed: %s", exc)
            if self._anomaly_detector is None:
                try:
                    from core.cognitive.anomaly_detector import AnomalyDetector
                    self._anomaly_detector = AnomalyDetector()
                    # Register it so other systems can share it
                    try:
                        from core.container import ServiceContainer
                        ServiceContainer.register("anomaly_detector", self._anomaly_detector)
                    except (ImportError, AttributeError, RuntimeError) as exc:
                        record_degradation("ice_layer", exc)
                        logger.debug("[ICE] Anomaly detector registration failed: %s", exc)
                except ImportError as exc:
                    logger.debug(
                        "%s unavailable (%s: %s); the reading is missing from the record",
                        "AnomalyDetector",
                        type(exc).__name__,
                        exc,
                    )
        return self._anomaly_detector

    async def load(self):
        self._running = True
        try:
            from core.event_bus import get_event_bus
            self._event_bus = get_event_bus()
            if self._event_bus:
                # Refactored to Queue-based processing for Aura EventBus
                self._audit_queue = await self._event_bus.subscribe("core/brain/empathy_audit")
                self._violation_queue = await self._event_bus.subscribe("core/security/executive_violation")
                self._task = get_task_tracker().create_task(
                    self._process_events(),
                    name="ice_layer.process_events",
                )
        except ImportError:
            self._event_bus = None

        # Initialize the learned anomaly detector
        self._get_anomaly_detector()

        logger.info("🛡️ [ICE] Intrusion Counter-Electronics ACTIVE. Firewall at 100%.")

    async def shutdown(self):
        """Stop the ICE event processing loop before kernel teardown completes."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._task = None
        self._event_bus = None
        logger.info("🛡️ [ICE] Intrusion Counter-Electronics OFFLINE.")

    async def _process_events(self):
        """Background loop to drain event queues."""
        while self._running:
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

    async def _on_audit(self, payload: dict[str, Any]):
        """Detect identity drift using both learned anomaly detection and legacy rules.

        The anomaly detector provides a data-driven threat score based on how
        far the current observation deviates from learned "normal" patterns.
        Legacy drift thresholds serve as a safety net when the detector hasn't
        learned enough yet.
        """
        drift = payload.get("drift", 0.0)
        status = payload.get("status", "NORMAL")
        try:
            drift = float(drift)
        # not a failure: a value that is not a number is not one this can read.
        except (TypeError, ValueError):
            drift = 0.0
        if not math.isfinite(drift):
            drift = 0.0
        now = time.monotonic()
        self._decay_threat(now)

        # Feed the audit event into the learned anomaly detector
        detector = self._get_anomaly_detector()
        if detector:
            try:
                await detector.observe({
                    "type": "audit",
                    "drift": drift,
                    "status": status,
                    "timestamp": time.time(),
                })
                learned_threat = detector.get_threat_level()
                # Statistical novelty is evidence for investigation, not proof
                # of hostile intent. Keep it visible but do not promote a
                # generic runtime outlier directly into a breach latch.
                self._statistical_novelty = max(
                    0.0,
                    min(1.0, float(learned_threat)),
                )
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('ice_layer', exc)
                logger.debug("[ICE] Anomaly detector observe failed: %s", exc)

        # Legacy safety net: hard threshold for extreme drift
        direct_signal = drift > 0.7 or status == "UNCANNY_VALLEY_DETECTED"
        if direct_signal:
            logger.warning("🚨 [ICE] COGNITIVE ANOMALY DETECTED. Assessing threat level.")
            self._signal_times.append(now)
            self._stable_audits = 0
            self._threat_level = min(
                1.0,
                self._threat_level + 0.35 + min(0.15, self._statistical_novelty * 0.15),
            )
        else:
            self._stable_audits += 1
            self._threat_level = max(0.0, self._threat_level - 0.12)

        self._prune_signals(now)
        if self._threat_level > 0.8 and len(self._signal_times) >= 2:
            await self._trigger_neural_hardening(
                reason="corroborated_identity_drift",
                evidence={
                    "drift": drift,
                    "status": status,
                    "corroborating_signals": len(self._signal_times),
                    "statistical_novelty": self._statistical_novelty,
                },
            )
        elif self._is_breached and self._stable_audits >= 3:
            await self._clear_neural_hardening(reason="three_stable_audits")

    async def _on_executive_violation(self, payload: dict[str, Any]):
        """Detect identity violations using learned + legacy assessment.

        Executive violations are serious events.  The anomaly detector learns
        that violation events are abnormal, while legacy rules ensure we never
        miss a real threat even if the detector is undertrained.
        """
        label = payload.get("label", "unknown")
        now = time.monotonic()
        self._decay_threat(now)
        anomaly = self.classify_anomaly(label)
        description = anomaly.get("description") or anomaly.get("desc") or "Unknown anomaly."
        logger.warning(
            f"🚨 [ICE] AWE CLASSIFIED: {anomaly['type']} ({description}). "
            f"Containment: {anomaly['containment']}"
        )
        logger.warning("🚨 [ICE] EXECUTIVE VIOLATION DETECTED: %s. Increasing threat level.", label)

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
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('ice_layer', exc)
                logger.debug("[ICE] Anomaly detector observe failed: %s", exc)

        # Legacy escalation (slightly softened since detector provides continuous signal)
        self._signal_times.append(now)
        self._prune_signals(now)
        self._stable_audits = 0
        increment = 0.65 if anomaly["type"] == "EXTERNAL_INTRUSION" else 0.25
        self._threat_level = min(1.0, self._threat_level + increment)
        if self._threat_level > 0.8 and (
            anomaly["type"] == "EXTERNAL_INTRUSION" or len(self._signal_times) >= 2
        ):
            await self._trigger_neural_hardening(
                reason=f"executive_violation:{anomaly['type'].lower()}",
                evidence={"label": str(label), "anomaly": anomaly},
            )

        if self._threat_level >= 1.0:
            await self._trigger_black_ice_escalation(payload)

    def classify_anomaly(self, label: str) -> dict[str, str]:
        """[AWE] Categorize anomaly and return containment protocol."""
        label_upper = label.upper()
        a_type = "UNKNOWN"
        if "RECURSION" in label_upper or "LOOP" in label_upper:
            a_type = "LOGIC_LOOP"
        elif "DRIFT" in label_upper or "IDENTITY" in label_upper:
            a_type = "SEMANTIC_DRIFT"
        elif "ACCESS" in label_upper or "INTRUSION" in label_upper:
            a_type = "EXTERNAL_INTRUSION"
        elif "LATENCY" in label_upper or "STALL" in label_upper:
            a_type = "TEMPORAL_STALL"
        
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

    async def _trigger_black_ice_escalation(self, payload: dict[str, Any]):
        """[BLACK ICE] Automated SOAR: Identity Blacklisting and Context Flush."""
        logger.critical("💀 [BLACK ICE] CRITICAL COGNITIVE THREAT. Commencing Countermeasures.")
        # Simulated SOAR actions
        user_id = payload.get("user_id", "unknown_subject")
        logger.warning("🚫 [BLACK ICE] Blacklisting Subject: %s", user_id)
        if self._event_bus:
            self._event_bus.publish_threadsafe("core/security/black_ice_escalation", {
                "subject": user_id,
                "action": "blacklist",
                "reason": "Identity Breach Attempt"
            })

    def _decay_threat(self, now: float) -> None:
        elapsed = max(0.0, float(now) - float(self._last_threat_update))
        self._last_threat_update = float(now)
        if elapsed <= 0.0:
            return
        # Ten-minute half-life: stale evidence cannot hold Aura in an
        # indefinite breach state, while repeated corroborated events still
        # accumulate faster than they decay.
        self._threat_level *= math.pow(0.5, elapsed / 600.0)
        if self._threat_level < 1e-4:
            self._threat_level = 0.0

    def _prune_signals(self, now: float) -> None:
        cutoff = float(now) - self._signal_window_s
        while self._signal_times and self._signal_times[0] < cutoff:
            self._signal_times.popleft()

    async def _trigger_neural_hardening(
        self,
        *,
        reason: str,
        evidence: dict[str, Any],
    ) -> None:
        """Emergency neural response to prevent state corruption."""
        if self._is_breached:
            return
        incident_id = f"ice-{uuid.uuid4()}"
        self._incident = {
            "incident_id": incident_id,
            "reason": str(reason),
            "evidence": dict(evidence),
            "activated_at": time.time(),
            "state": "contained",
        }
        logger.critical(
            "⛔ [ICE] CORROBORATED BREACH RISK. Neural containment active "
            "(incident=%s reason=%s).",
            incident_id,
            reason,
        )
        self._is_breached = True
        if self._event_bus:
            await self._event_bus.publish("core/cybernetics/ice_alert", {
                "threat": self._threat_level,
                "action": "Neural Hardening",
                **self._incident,
            })

    async def _clear_neural_hardening(self, *, reason: str) -> None:
        if not self._is_breached:
            return
        incident = dict(self._incident or {})
        incident["state"] = "recovered"
        incident["cleared_at"] = time.time()
        incident["clear_reason"] = str(reason)
        self._incident = incident
        self._is_breached = False
        self._threat_level = min(self._threat_level, 0.35)
        self._signal_times.clear()
        logger.info(
            "[ICE] Neural containment cleared after verified recovery "
            "(incident=%s reason=%s).",
            incident.get("incident_id", "unknown"),
            reason,
        )
        if self._event_bus:
            await self._event_bus.publish(
                "core/cybernetics/ice_recovery",
                dict(incident),
            )

    def get_status(self) -> dict[str, Any]:
        return {
            "threat_level": self._threat_level,
            "statistical_novelty": self._statistical_novelty,
            "is_breached": self._is_breached,
            "corroborating_signals": len(self._signal_times),
            "incident": dict(self._incident) if self._incident else None,
        }

logger = logging.getLogger("Aura.Cybernetics.ICE")
