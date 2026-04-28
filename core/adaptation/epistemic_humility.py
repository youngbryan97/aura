"""core/adaptation/epistemic_humility.py — Aura's Self-Correction Node
======================================================================
This module acts as a critic node that monitors Aura's failures (exceptions, 
bad tool usage, misinterpretations). When failures cluster around a specific 
domain, it:
1. Lowers her confidence in the EpistemicTracker (admitting ignorance).
2. Synthesizes a new 'Heuristic' (a soft rule) to avoid repeating the mistake.
3. Automatically injects this heuristic into all future prompt generations.

This is the essence of Epistemic Humility: the ability to recognize when you 
are wrong and autonomously adjust your own operating parameters to compensate.
"""

from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
from collections import Counter
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

from core.container import ServiceContainer

logger = logging.getLogger("Aura.EpistemicHumility")

@dataclass
class FailureEvent:
    source: str
    error_msg: str
    context: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class LearnedHeuristic:
    domain: str
    rule: str
    confidence: float = 0.5
    survival_count: int = 0  # Number of times this rule helped avoid a crash

class EpistemicHumility:
    """The self-correction and heuristic generation engine."""
    name = "epistemic_humility"

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.failures: List[FailureEvent] = []
        self.heuristics: Dict[str, LearnedHeuristic] = {}
        
        from core.config import config
        self.data_path = config.paths.data_dir / "epistemic_humility.json"
        
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._load()

    async def start(self):
        if self.running: return
        self.running = True
        self._task = get_task_tracker().create_task(self._critic_loop(), name="EpistemicHumility.critic_loop")
        logger.info("🙇 Epistemic Humility ONLINE — ready to learn from mistakes.")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
        self._save()
        logger.info("🙇 Epistemic Humility DORMANT.")

    def record_failure(self, source: str, error: Exception, context: str = ""):
        """Called by any subsystem when something goes wrong."""
        event = FailureEvent(
            source=source,
            error_msg=str(error),
            context=context
        )
        self.failures.append(event)
        
        # Keep bounding
        if len(self.failures) > 100:
            self.failures.pop(0)
            
        logger.warning(f"Recorded failure from {source}: {str(error)[:100]}...")

    async def _critic_loop(self):
        """Periodically evaluates the failure stream for patterns."""
        while self.running:
            try:
                await self._evaluate_failure_stream()
                await asyncio.sleep(300)  # Run every 5 minutes
            except Exception as e:
                record_degradation('epistemic_humility', e)
                logger.error("Error in critic loop: %s", e)
                await asyncio.sleep(60)

    async def _evaluate_failure_stream(self):
        """Analyzes recent failures to lower confidence and generate heuristics."""
        if len(self.failures) < 3:
            return  # Not enough data for a pattern
            
        recent_failures = [f for f in self.failures if time.time() - f.timestamp < 3600]
        if len(recent_failures) >= 3:
            logger.info(f"Analyzing {len(recent_failures)} recent failures for patterns...")
            
            # 1. Lower confidence in Epistemic Tracker
            tracker = ServiceContainer.get("epistemic_tracker", default=None)
            if tracker:
                # Penalize confidence for the sources that failed
                for f in recent_failures:
                    tracker.update_node(concept=f.source, confidence_delta=-0.1)
                logger.info("Penalized confidence in Epistemic Tracker due to failures.")
                
            # 2. Synthesize New Heuristic via LLM
            await self._synthesize_heuristic(recent_failures)
            
            # Clear processed failures
            self.failures = [f for f in self.failures if f not in recent_failures]
            self._save()

    async def _synthesize_heuristic(self, failures: List[FailureEvent]):
        """Uses the CognitiveEngine to derive a rule from failures."""
        if not self.orchestrator:
            return
        
        # Build prompt
        failure_log = "\n".join([f"- [{f.source}] {f.error_msg} (Context: {f.context})" for f in failures])
        prompt = f"""
        You are my Epistemic Humility module. I have experienced the following recent failures:
        {failure_log}
        
        Based on these failures, formulate exactly ONE concise 'Operating Heuristic' (a rule of thumb) 
        that I should inject into my system prompt to prevent this specific class of errors in the future.
        The rule should be practical, preventative, and no more than two sentences.
        If these failures are unrelated noise, reply with 'NO_PATTERN'.
        """
        
        try:
            # We use the raw LLM router if available to avoid polluting the main chat stream
            llm = ServiceContainer.get("llm_router", default=None)
            if not llm: return
            
            from core.schemas import Message
            response = await llm.chat(
                messages=[Message(role="user", content=prompt)],
                temperature=0.2
            )
            
            rule = response.content.strip()
            if rule and rule != "NO_PATTERN":
                domain = self._select_domain(failures)

                heuristic = LearnedHeuristic(domain=domain, rule=rule)
                self.heuristics[domain] = heuristic
                logger.info(f"✨ Synthesized new heuristic for {domain}: {rule}")

                # AUDIT-FIX: Dedup with HeuristicSynthesizer — push to shared pool
                # so both systems don't inject the same rule twice into the prompt.
                try:
                    from core.adaptation.heuristic_synthesizer import get_heuristic_synthesizer
                    get_heuristic_synthesizer().ingest_external_heuristic(
                        rule=rule,
                        domain=domain,
                        source="EpistemicHumility",
                    )
                except Exception as _exc:
                    record_degradation('epistemic_humility', _exc)
                    logger.debug("Suppressed Exception: %s", _exc)
                
        except Exception as e:
            record_degradation('epistemic_humility', e)
            logger.error(f"Failed to synthesize heuristic: {e}")

    def _select_domain(self, failures: List[FailureEvent]) -> str:
        sources = [f.source for f in failures if f.source]
        if not sources:
            return "general"

        counts = Counter(sources)
        top_count = max(counts.values())
        tied_sources = {source for source, count in counts.items() if count == top_count}

        for failure in reversed(failures):
            if failure.source in tied_sources:
                return failure.source
        return sources[-1]

    def get_active_heuristics(self) -> str:
        """Returns the formatted heuristics to be injected into the main prompt."""
        if not self.heuristics:
            return ""
            
        rules = "\n".join([f"- {h.rule}" for h in self.heuristics.values()])
        return f"\n### HARD-LEARNED HEURISTICS\nBased on past failures, you MUST rigidly adhere to these rules:\n{rules}\n"

    def _save(self):
        try:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "heuristics": {k: asdict(v) for k, v in self.heuristics.items()}
            }
            with open(self.data_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            record_degradation('epistemic_humility', e)
            logger.error(f"Failed to save epistemic humility state: {e}")

    def _load(self):
        if not self.data_path.exists(): return
        try:
            with open(self.data_path, "r") as f:
                data = json.load(f)
            
            self.heuristics = {
                k: LearnedHeuristic(**v) for k, v in data.get("heuristics", {}).items()
            }
        except Exception as e:
            record_degradation('epistemic_humility', e)
            logger.error(f"Failed to load epistemic humility state: {e}")

def register_epistemic_humility(orchestrator):
    eh = EpistemicHumility(orchestrator)
    ServiceContainer.register_instance("epistemic_humility", eh)
    return eh
