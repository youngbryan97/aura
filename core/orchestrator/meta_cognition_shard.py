from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import time
try:
    import numpy as np
except ImportError:
    np = None
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from core.runtime.background_policy import background_activity_reason

logger = logging.getLogger("Aura.MetaCognition")

class MetaCognitionShard:
    """
    [PHASE 9] META-COGNITION SHARD
    Proactive background shard that audits Aura's internal state every 60s.
    Detects loops, stalls, or degradation and pushes corrections.
    """
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self.is_running = False
        self._audit_task: Optional[asyncio.Task] = None
        self._last_audit_time: float = 0
        self._history_hash_log = [] # To detect repetition loops
        self._audit_lock = asyncio.Lock() # Prevent re-entrant audits
        
    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._audit_task = get_task_tracker().create_task(self._audit_loop())
        logger.info("🧠 Meta-Cognition Shard ONLINE.")

    def _background_block_reason(self) -> str:
        try:
            return str(
                background_activity_reason(
                    self.orchestrator,
                    min_idle_seconds=180.0,
                    max_memory_percent=78.0,
                    max_failure_pressure=0.10,
                    require_conversation_ready=False,
                )
                or ""
            )
        except Exception as exc:
            logger.debug("Meta-Cognition background gate skipped: %s", exc)
            return ""

    async def _audit_loop(self):
        while self.is_running:
            try:
                await asyncio.sleep(60)
                reason = self._background_block_reason()
                if reason:
                    logger.debug("🧠 Meta-Cognitive Audit deferred: %s", reason)
                    continue
                await self.perform_audit()
            except Exception as e:
                logger.error(f"Meta-Cognition audit loop failed: {e}")
                await asyncio.sleep(10)

    async def evolve(self):
        """v35 Hardening: Atomic evolution with validation."""
        try:
            # 1. Self-Diagnosis
            if not await self._validate_stability():
                 logger.warning("Stabilizer: Meta-Evolution aborted due to system instability.")
                 return
            
            # 2. Perform Atomic Audit/Evolution
            await self.perform_audit()
            
            logger.info("🧠 Meta-Evolution cycle completed successfully (v35).")
        except Exception as e:
            logger.error(f"Meta-Evolution failed: {e}")

    async def _validate_stability(self) -> bool:
        """Checks if the system is stable enough for self-modification."""
        try:
            # Check for recent latency spikes or affective collapse
            if await self._detect_latency_spike(): return False
            if await self._detect_affective_collapse(): return False
            
            # Check if orchestrator is in a healthy state
            if not getattr(self.orchestrator.status, "healthy", True):
                return False
                
            return True
        except Exception:
            return False

    async def perform_audit(self):
        """Analyze system health and conversation metrics."""
        logger.info("🧠 Running Meta-Cognitive Audit...")
        
        # 1. Loop Detection
        if await self._detect_repetition_loop():
            await self._push_correction("repetition_break", "System detected a repetition loop. Increasing temperature and diversifying response strategy.")
            
        # 2. Performance Audit
        if await self._detect_latency_spike():
            await self._push_correction("latency_mitigation", "System latency is high. Enabling aggressive pruning and short-form responses.")

        # 3. Emotional Coherence Audit (via Mycelium)
        if await self._detect_affective_collapse():
            await self._push_correction("emotional_reset", "Internal affective state is unstable. Applying grounding protocols.")

        self._last_audit_time = time.time()

    async def _detect_repetition_loop(self) -> bool:
        """Detect loops in conversation history or state hashes."""
        try:
            # Standardize: check conversation_history directly if available
            history = getattr(self.orchestrator, 'conversation_history', [])
            if not history:
                conv_mem = getattr(self.orchestrator, 'memory', None)
                if conv_mem and hasattr(conv_mem, "get_recent_history"):
                    history = await conv_mem.get_recent_history(limit=8)
            
            if len(history) < 4:
                return False
                
            # 1. Multi-turn loop detection (e.g. A->B->A->B)
            msg_list = list(history)
            last_four = [str(m.get("content", "")).lower().strip() for m in msg_list[-4:]]
            if len(last_four) == 4 and last_four[0] == last_four[2] and last_four[1] == last_four[3]:
                logger.warning("🧠 DETECTED MULTI-TURN REPETITION LOOP (N=2)")
                return True
                
            # 2. Trivial N=1 loop (Assistant repeating itself 3 times)
            asst_messages = [str(m.get("content", "")).lower().strip() for m in history if m.get("role") == "assistant"]
            if len(asst_messages) >= 3 and len(set(asst_messages[-3:])) == 1:
                logger.warning("🧠 DETECTED TRIVIAL REPETITION LOOP (N=1)")
                return True
        except Exception as e:
            logger.debug(f"Repetition detection error: {e}")
        return False

    async def _detect_latency_spike(self) -> bool:
        # Check if the currently processing message has exceeded threshold
        try:
            if getattr(self.orchestrator.status, "is_processing", False):
                start_time = getattr(self.orchestrator, "_current_processing_start", 0)
                if start_time > 0:
                    delta = time.monotonic() - start_time
                    if delta > 30.0:  # 30 second timeout for cognitive cycles
                        logger.warning(f"🧠 DETECTED LATENCY SPIKE / STALL (Delta: {delta:.2f}s)")
                        return True
        except Exception as e:
            logger.warning("Latency check failed: %s", e)
        return False

    async def _detect_affective_collapse(self) -> bool:
        # Check liquid_state for extreme volatility
        try:
            ls = getattr(self.orchestrator, "liquid_state", None)
            if ls and hasattr(ls, "v") and np:
                volatility = np.mean(np.abs(ls.v))
                if volatility > 0.8:
                    logger.warning(f"🧠 DETECTED AFFECTIVE COLLAPSE (Volatility: {volatility:.2f})")
                    return True
        except Exception as e:
            logger.warning("Affective audit failed: %s", e)
        return False

    async def _push_correction(self, correction_type: str, hint: str):
        """Inject a corrective shard into the next inference cycle."""
        logger.info(f"🧠 Pushing {correction_type} correction: {hint}")
        try:
            if hasattr(self.orchestrator, "add_correction_shard"):
                # Format hint to include type
                formatted_hint = f"[{correction_type.upper()}] {hint}"
                self.orchestrator.add_correction_shard(formatted_hint)
            else:
                logger.debug(f"Correction logged (orchestrator missing add_correction_shard): {hint}")
        except Exception as e:
            logger.error(f"Failed to push correction: {e}")
