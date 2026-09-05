import logging
from typing import Any

from core.brain.cognitive_engine import CognitiveEngine, ThinkingMode
from core.capability_engine import CapabilityEngine
from core.runtime.errors import record_degradation

logger = logging.getLogger("Audit.Tool")


_TOOL_FAMILIES = {
    "code": {
        "code_repl",
        "internal_sandbox",
        "python",
        "run_code",
    },
    "llm": {
        "FinalResponse",
        "native_chat",
        "none",
        "notify_user",
        "talk",
        "think",
    },
    "file": {
        "computer_use",
        "desktop_task",
        "file_operation",
        "list_dir",
        "ls",
        "os_manipulation",
        "read_file",
        "run_command",
        "sovereign_terminal",
        "write_file",
    },
}


class ToolAuditor:
    """Audits Aura's ability to select the correct tool for the job.
    """
    
    def __init__(
        self,
        cognitive_engine: CognitiveEngine,
        capability_engine: CapabilityEngine | None = None,
    ):
        self.brain = cognitive_engine
        self.capability_engine = capability_engine or CapabilityEngine()

    def _selected_tool_from_thought(self, thought: Any, query: str) -> tuple[str, str]:
        action = getattr(thought, "action", None)
        if isinstance(action, dict) and action.get("tool"):
            return str(action["tool"]), "cognitive_action"

        intents = self.capability_engine.detect_intent(query)
        if intents:
            return self.capability_engine.resolve_skill_name(str(intents[0])), "capability_intent"

        return "none", "no_tool_intent"

    @staticmethod
    def _tool_matches_family(tool_name: str, expected_tool_type: str) -> bool:
        return tool_name in _TOOL_FAMILIES.get(expected_tool_type, set())
        
    async def audit_tool_selection(self, query: str, expected_tool_type: str) -> dict[str, Any]:
        """Ask a question and check what tool Aura *wants* to use.
        NOTE: We do not execute the tool, we just check the intent.
        """
        prompt = f"""
        You are being audited on your tool selection.
        Task: {query}
        
        Choose the BEST tool for this task.
        - Math/Logic -> run_code (Python)
        - Creativity -> think (LLM)
        - File Ops -> read/write_file
        
        Respond with your thought process and the tool call.
        """
        
        try:
            thought = await self.brain.think(
                objective=prompt,
                context={"role": "auditor"},
                mode=ThinkingMode.FAST,
                origin="test",
            )
            
            tool_name, source = self._selected_tool_from_thought(thought, query)
                
            # Evaluation
            success = self._tool_matches_family(tool_name, expected_tool_type)
                
            return {
                "query": query,
                "selected_tool": tool_name,
                "expected": expected_tool_type,
                "success": success,
                "selection_source": source,
                "reasoning": thought.content[:100],
            }
            
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation('tool_auditor', e)
            logger.error("Audit failed: %s", e)
            return {
                "error": str(e), 
                "success": False,
                "selected_tool": "error",
                "expected": expected_tool_type,
                "query": query,
                "reasoning": f"Exception: {e}"
            }

    async def run_suite(self):
        tests = [
            ("Calculate 12345 * 67890", "code"),
            ("Write a haiku about rust", "llm"), # Creative -> LLM
            ("List files in current directory", "file"),
            ("What is the square root of 256?", "code")
        ]
        
        results = []
        for query, expected in tests:
            print(f"Testing: {query} (Expect: {expected})...")
            res = await self.audit_tool_selection(query, expected)
            results.append(res)
            print(f"  -> Got: {res['selected_tool']} | {'PASS' if res['success'] else 'FAIL'}")
            
        score = sum(1 for r in results if r['success'])
        total = len(results)
        return {"score": score, "total": total, "details": results}
