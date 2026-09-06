import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.network_gateway import get_network_gateway
from core.skills.base_skill import BaseSkill

# Configure logger
logger = logging.getLogger("Skills.InterAgent")

class InterAgentCommSkill(BaseSkill):
    """Skill for communication with registered local peer agents.
    v3.4 Patch: Uses mock 'conversation_loop' if needed.
    """
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    
    name = "inter_agent_comm"
    retry_safe = False  # external send/act — never double-fire on retry
    description = "Send a message to a registered local peer agent to request assistance."

    def __init__(self):
        # Ensure we have a place to store these outbound messages
        self.comm_log_path = Path("data/comm_logs.jsonl")
        self.comm_log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("✅ InterAgentComm initialized (v3.4 Patch Applied)")

    async def execute(self, goal: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
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
                resp = await asyncio.to_thread(
                    get_network_gateway().request,
                    "POST",
                    target_endpoint,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"sender": "Aura", "message": message, "swarm_context": swarm_data[:2]}),
                    timeout=10.0,
                    source="skills.inter_agent_comm.send",
                )
                    
                status_code = int(resp.get("status_code") or 0)
                if status_code in (200, 201, 202):
                    status_msg = f"Message successfully transmitted to {agent_name} (HTTP {status_code})."
                else:
                    status_msg = f"Agent {agent_name} rejected the payload: HTTP {status_code}"
            except (OSError, ConnectionError, TimeoutError, TypeError, ValueError) as e:
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

        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('inter_agent_comm', e)
            logger.error("Inter-agent comm failed: %s", e)
            return {
                "ok": False,
                "error": str(e)
            }

    def _log_communication(self, data: dict[str, Any]):
        """Append communication record to log file"""
        try:
            get_file_write_gateway().append_text(
                self.comm_log_path,
                json.dumps(data) + "\n",
                source="skills.inter_agent_comm.log",
            )
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            record_degradation('inter_agent_comm', e)
            logger.error("Failed to write comm log: %s", e)
