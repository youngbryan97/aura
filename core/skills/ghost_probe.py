"""core/skills/ghost_probe.py
Phase 16.4: Ghost Deployment Skill.
Allows Aura to spawn external monitoring probes.
"""
from core.skills.base_skill import BaseSkill
from core.container import ServiceContainer
from typing import Any, Dict
from pydantic import BaseModel, Field
import logging
import time

logger = logging.getLogger("Aura.Skills.GhostProbe")

class GhostProbeParams(BaseModel):
    probe_id: str = Field(..., description="Unique ID for the probe")
    target: str = Field(..., description="File path or resource to monitor")
    type: str = Field("file", description="Type of probe (file/ping)")
    duration: int = Field(3600, description="Duration in seconds")

class GhostProbeSkill(BaseSkill):
    """Skill to deploy and manage Ghost Probes."""
    
    name = "deploy_ghost_probe"
    description = "Deploy a lightweight background probe to monitor a file or resource."
    input_model = GhostProbeParams

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    async def execute(self, params: GhostProbeParams, context: dict = None) -> Dict[str, Any]:
        manager = ServiceContainer.get("probe_manager", default=None)
        if not manager:
            return {"ok": False, "error": "ProbeManager service not available."}
            
        success = await manager.deploy_probe(
            params.probe_id, 
            params.target, 
            params.type, 
            params.duration
        )
        
        if success:
            return {"ok": True, "summary": f"Ghost Probe '{params.probe_id}' successfully deployed to watch {params.target} ({params.type}) for {params.duration}s."}
        else:
            return {"ok": False, "error": f"Failed to deploy Ghost Probe '{params.probe_id}'. It might already exist or there was a system error."}

    async def list_probes(self) -> Dict[str, Any]:
        """List all active ghost probes."""
        manager = ServiceContainer.get("probe_manager", default=None)
        if not manager: 
            return {"ok": False, "error": "ProbeManager offline."}
        
        if not manager.probes:
            return {"ok": True, "probes": [], "summary": "No active ghost probes."}
            
        probes = []
        for pid, meta in manager.probe_metadata.items():
            probes.append({
                "id": pid,
                "type": meta['type'],
                "target": meta['target'],
                "expires_in": int(meta['expiry'] - time.time())
            })
            
        return {
            "ok": True, 
            "probes": probes, 
            "summary": f"Found {len(probes)} active ghost probes."
        }