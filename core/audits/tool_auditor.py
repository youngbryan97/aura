import asyncio
import logging
from typing import Any, Dict, List

from core.brain.cognitive_engine import CognitiveEngine, ThinkingMode

logger = logging.getLogger("Audit.Tool")

class ToolAuditor:
    """Audits Aura's ability to select the correct tool for the job.
    """
    
    def __init__(self, cognitive_engine: CognitiveEngine):
        self.brain = cognitive_engine
        
    async def audit_tool_selection(self, query: str, expected_tool_type: str) -> Dict[str, Any]:
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
                mode=ThinkingMode.FAST
            )
            
            tool_name = "none"
            if hasattr(thought, "action") and thought.action:
                tool_name = thought.action.get("tool")
                
            # Evaluation
            success = False
            if expected_tool_type == "code":
                success = tool_name in ["run_code", "python"]
            elif expected_tool_type == "llm":
                # 'think' is internal, so if no tool is called or 'final_answer', it's LLM
                success = tool_name in ["none", "think", "notify_user"]
            elif expected_tool_type == "file":
                success = tool_name in ["read_file", "write_file", "list_dir", "ls", "run_command"]
                
            return {
                "query": query,
                "selected_tool": tool_name,
                "expected": expected_tool_type,
                "success": success,
                "reasoning": thought.content[:100]
            }
            
        except Exception as e:
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