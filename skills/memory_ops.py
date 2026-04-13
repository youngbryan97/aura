# skills/memory_ops.py
import logging
from typing import Any, Dict, Optional

from infrastructure import BaseSkill

logger = logging.getLogger("Skills.MemoryOps")

class MemoryOpsSkill(BaseSkill):
    name = "memory_ops"
    description = "Manage and update Aura's Knowledge Base (Verified Facts) with structured data support."
    inputs = {
        "action": "The operation (learn_fact, retrieve, append_list, remove_from_list, clear).",
        "key": "The fact key (e.g., 'user_name', 'project_goals').",
        "value": "The fact value (can be string, list, or dict)."
    }

    async def execute(self, goal: Dict, context: Dict) -> Dict:
        mem_sys = context.get("memory")
        if not mem_sys:
            return {"ok": False, "error": "Memory system not available in context."}

        params = goal.get("params", {})
        action = params.get("action")
        key = params.get("key")
        value = params.get("value")

        if action == "learn_fact":
            if not key or value is None:
                return {"ok": False, "error": "Missing key or value for learning fact."}
            
            # Call the learn_fact method on MemoryNexus
            if hasattr(mem_sys, 'learn_fact'):
                await mem_sys.learn_fact(key, value)
                return {
                    "ok": True, 
                    "summary": f"Learned fact: {key} = {value}",
                    "result": f"Knowledge base updated: {key} is now known."
                }
            else:
                return {"ok": False, "error": "Memory system does not support explicit fact learning."}
        
        elif action == "retrieve":
            if not key:
                return {"ok": False, "error": "Missing key for fact retrieval."}
            
            knowledge = getattr(mem_sys, 'knowledge', {})
            val = knowledge.get(key)
            if val is not None:
                return {"ok": True, "result": val}
            else:
                return {"ok": False, "error": f"No fact found for key: {key}"}

        # v2.0 Structured Memory Operations
        elif action == "append_list":
            if not key or value is None:
                return {"ok": False, "error": "Missing key or value for append_list."}
                
            knowledge = getattr(mem_sys, 'knowledge', {})
            current = knowledge.get(key, [])
            if not isinstance(current, list):
                return {"ok": False, "error": f"Key {key} exists but is not a list."}
                
            if value not in current:
                current.append(value)
                if hasattr(mem_sys, 'learn_fact'):
                    await mem_sys.learn_fact(key, current)
                return {"ok": True, "result": f"Appended to {key}.", "current_list": current}
            return {"ok": True, "result": f"Value already in {key}.", "current_list": current}

        elif action == "remove_from_list":
            if not key or value is None:
                return {"ok": False, "error": "Missing key or value for remove_from_list."}
                
            knowledge = getattr(mem_sys, 'knowledge', {})
            current = knowledge.get(key, [])
            if not isinstance(current, list):
                return {"ok": False, "error": f"Key {key} exists but is not a list."}
                
            if value in current:
                current.remove(value)
                if hasattr(mem_sys, 'learn_fact'):
                    await mem_sys.learn_fact(key, current)
                return {"ok": True, "result": f"Removed from {key}.", "current_list": current}
            return {"ok": False, "error": f"Value not in {key}."}

        elif action == "clear":
            if not key:
                return {"ok": False, "error": "Missing key to clear."}
            if hasattr(mem_sys, 'forget_fact'):
                await mem_sys.forget_fact(key)
                return {"ok": True, "result": f"Cleared memory for {key}."}
            return {"ok": False, "error": "Memory system does not support explicit fact clearing."}

        return {"ok": False, "error": f"Unknown action: {action}"}
