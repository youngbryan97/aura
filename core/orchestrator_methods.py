"""Orchestrator methods — CNS-based message processing (experimental mixin).
These methods are intended for future integration with the NeuroWeb CNS.
Currently unused — the main orchestrator handles all message processing.
"""
from core.runtime.errors import record_degradation
import logging
from typing import Any

logger = logging.getLogger(__name__)


class OrchestratorCNSMixin:
    """Mixin for CNS-based orchestrator processing (future use)."""

    cns: Any
    emitter: Any
    cognitive_engine: Any
    _execute_task: Any

    async def process_user_input_cns(self, message: str) -> None:
        """Public alias for _process_message_cns to satisfy Server API."""
        return await self._process_message_cns(message)

    async def _process_message_cns(self, message: str) -> None:
        """Process a user message through the CNS.
        """
        try:
            # 1. CNS Processing (The New Path)
            cns_response = await self.cns.process_stimulus(message)
            
            if cns_response["status"] == "inhibited":
                self.emitter.emit("thought", f"Inhibited: {cns_response['reason']}", level="info")
                return # Do not execute if inhibited
                
            execution_plan = cns_response.get("execution")
            if execution_plan:
                neuron = execution_plan["neuron"]
                synapse = execution_plan["synapse"]
                
                self.emitter.emit("thought", f"Synapse Fired: {synapse.intent_pattern} -> {neuron.name}", level="success")
                
                # Execute Logic (Simplified wrapper)
                # In robust version, this would use the synapse parameters
                task_payload = {
                     "skill": neuron.id.replace("skill:", ""), # Extract skill name
                     "params": {"query": message} # Rough param mapping (we need a proper mapper)
                }
                
                await self._execute_task(task_payload)
            else:
                 # Fallback to old cognitive engine if CNS has no path
                 # This ensures we don't break existing functionality while transitioning
                 self.emitter.emit("thought", "No neural path found. Engaging cognitive engine...", level="info")
                 await self.cognitive_engine.process(message, self)
                 
        except Exception as e:
            record_degradation('orchestrator_methods', e)
            logger.error("CNS Processing Error: %s", e)
            self.emitter.emit("error", f"Neural Error: {e}")
