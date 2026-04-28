from core.runtime.errors import record_degradation
import json
import re
import logging
from typing import Any, Dict

logger = logging.getLogger("Aura.Utils.JSONRepair")

def robust_json_parse(raw_output: str) -> Dict[str, Any]:
    """Strips markdown formatting and conversational filler before parsing.
    
    This is the v51 'Hammer' fix for LLMs that insist on wrapping JSON in 
    prose or markdown blocks.
    """
    if not raw_output:
        return {}
        
    try:
        # 1. Try to find the outermost curly braces (Objects)
        match = re.search(r'(\{.*\})', raw_output, re.DOTALL)
        if match:
            clean_json = match.group(1).strip()
            try:
                return json.loads(clean_json)
            except json.JSONDecodeError:
                # If balanced extraction fails, fallback to heuristic repair
                pass  # no-op: intentional
        
        # 2. Try to find the outermost square brackets (Arrays)
        match_arr = re.search(r'(\[.*\])', raw_output, re.DOTALL)
        if match_arr:
            clean_json = match_arr.group(1).strip()
            try:
                return json.loads(clean_json)
            except json.JSONDecodeError as _exc:
                logger.debug("Suppressed json.JSONDecodeError: %s", _exc)

        # 3. Last resort: standard parse
        return json.loads(raw_output)
    except Exception:
        # 4. Deep Recovery: Fallback to the recursive SelfHealingJSON engine
        try:
            from core.utils.json_utils import SelfHealingJSON
            repairer = SelfHealingJSON()
            return repairer.parse_sync(raw_output)
        except Exception as e:
            record_degradation('json_repair', e)
            logger.error(f"FATAL: JSON Repair engine failed: {e}")
            return {}