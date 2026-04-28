from core.runtime.errors import record_degradation
import asyncio
import json
import logging
import time
import httpx
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.skills.base_skill import BaseSkill
from core.container import ServiceContainer

# Configure logger
logger = logging.getLogger("Skills.InterAgent")

class InterAgentCommSkill(BaseSkill):
    """Skill to facilitate communication with external agents (e.g., Gemini, ChatGPT).
    v3.4 Patch: Uses mock 'conversation_loop' if needed.
    """
    
    name = "inter_agent_comm"
    description = "Send a message to an external agent (Gemini, etc) to request assistance."

    def __init__(self):
        # Ensure we have a place to store these outbound messages
        self.comm_log_path = Path("data/comm_logs.jsonl")
        self.comm_log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("✅ InterAgentComm initialized (v3.4 Patch Applied)")

    async def execute(self, goal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the communication request (Async)."""
        params = goal.get("params", {}) if "params" in goal else goal
        agent_name = params.get("agent_name") or params.get("recipient")
        message = params.get("message")
        
        if not agent_name or not message:
            return {"ok": False, "error": "Missing agent_name or message in params"}
        try:
            timestamp = time.time()
            logger.info("📡 Transmitting to %s: %s...", agent_name, message[:50])

            # 1. Log locally
            payload = {
                "timestamp": timestamp,
                "direction": "outbound",
                "target": agent_name,
                "content": message,
                "context": context
            }
            await asyncio.to_thread(self._log_communication, payload)

            # 2. Issue 68: Lazy fetch orchestrator
            orchestrator = ServiceContainer.get("orchestrator", default=None)
            swarm_data = []
            if orchestrator and hasattr(orchestrator, 'belief_sync'):
                logger.info("🌌 Querying swarm for context on: %s", agent_name)
                swarm_data = await orchestrator.belief_sync.query_peers(agent_name)

            # 3. Transmit payload across network
            target_endpoint = f"http://localhost:8000/api/v1/agents/{agent_name}/message"
            logger.info("🚀 Initiating Swarm HTTP POST to %s", target_endpoint)
            
            try:
                # We do a fire-and-forget style async post, but wait briefly for a 200 OK acceptance
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        target_endpoint,
                        json={"sender": "Aura", "message": message, "swarm_context": swarm_data[:2]}
                    )
                    
                if resp.status_code in (200, 201, 202):
                    status_msg = f"Message successfully transmitted to {agent_name} (HTTP {resp.status_code})."
                else:
                    status_msg = f"Agent {agent_name} rejected the payload: HTTP {resp.status_code}"
            except httpx.RequestError as e:
                logger.warning("Network failure reaching agent %s: %s", agent_name, e)
                status_msg = f"Network failure—{agent_name} unreachable: {e}"

            # 4. Status encapsulation

            return {
                "ok": True,
                "status": "logged_and_queried",
                "message": status_msg,
                "details": {
                    "target": agent_name,
                    "timestamp": timestamp,
                    "swarm_insights_count": len(swarm_data),
                    "swarm_insights": swarm_data[:5] # Return first 5
                }
            }

        except Exception as e:
            record_degradation('inter_agent_comm', e)
            logger.error("Inter-agent comm failed: %s", e)
            return {
                "ok": False,
                "error": str(e)
            }

    def _log_communication(self, data: Dict[str, Any]):
        """Append communication record to log file"""
        try:
            with open(self.comm_log_path, 'a') as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            record_degradation('inter_agent_comm', e)
            logger.error("Failed to write comm log: %s", e)