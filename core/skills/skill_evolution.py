from core.runtime.errors import record_degradation
import ast
import json
import logging
import os
import sys
import uuid
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from core.config import config
from core.security.audit_trail import get_audit_trail
from core.security.constitutional_guard import ConstitutionalGuard
from core.brain.llm_health_router import llm_router

logger = logging.getLogger("Skills.Evolution")


class SkillEvolver:
    """Autonomous loop for Aura to draft, verify, and load new skills."""

    def __init__(self):
        self.skills_dir = config.paths.home_dir / "core/skills"
        self.guard = ConstitutionalGuard()
        self.audit = get_audit_trail()
        
    def generate_skill(self, goal: str, instructions: str) -> Tuple[bool, str, str]:
        """Draft a new BaseSkill based on a missing capability.
        
        Returns:
            (success, message, code_or_error)
        """
        logger.info(f"🧬 Evolving new skill for goal: {goal}")
        
        prompt = f"""
You are the cognitive architect of Aura.
You encountered a task you do not have a tool for: "{goal}"
Additional instructions: {instructions}

Write a new Python skill extending `BaseSkill` to solve this.

RULES:
1. ONLY return the raw Python code. No markdown formatting, no backticks, no explanation.
2. The class MUST inherit from `core.skills.base_skill.BaseSkill`.
3. You must define a `pydantic.BaseModel` for the input arguments.
4. You must implement `_execute(self, **kwargs) -> Dict[str, Any]`.
5. Imports must include `from core.skills.base_skill import BaseSkill` and `from pydantic import BaseModel, Field`.
6. Make it robust: handle exceptions inside `_execute` and return helpful error dictionaries where appropriate (BaseSkill's safe_execute will catch unhandled ones, but handle known edge cases).

Draft the Python code for this highly capable new skill now:
"""

        try:
            # 1. Draft the code using the LLM
            response = llm_router.generate(prompt=prompt, temperature=0.2)
            if not response:
                return False, "LLM failed to generate skill code.", ""
                
            code = response.strip()
            
            # Strip markdown if the LLM ignored rule 1
            if code.startswith("```python"):
                code = code[9:]
            elif code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            code = code.strip()
            
            # 2. Extract class name using AST
            try:
                tree = ast.parse(code)
                class_names = [n.name for n in tree.body if isinstance(n, ast.ClassDef) and n.name != "BaseModel"]
                if not class_names:
                    return False, "Syntax error: No skill class found in generated code.", code
                
                # Assume the last defined class is the skill class
                class_name = class_names[-1]
                
                # We need a snake_case filename
                filename = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower() + ".py"
                if filename.startswith("base_skill"):
                    return False, "Generated skill tried to overwrite base_skill.", code
                    
            except SyntaxError as e:
                return False, f"LLM generated invalid Python syntax: {e}", code

            # 3. Constitutional/Security Check
            is_safe, violation = self.guard.check_code_action(code)
            if not is_safe:
                return False, f"Constitutional Guard blocked generated skill: {violation}", code
                
            # 4. Save to disk
            filepath = self.skills_dir / filename
            if filepath.exists():
                # Avoid overwriting existing hardcoded skills — generate a unique name
                filename = f"{filename.split('.')[0]}_{str(uuid.uuid4())[:6]}.py"
                filepath = self.skills_dir / filename
                
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(code)
                
            logger.info(f"✨ Successfully evolved new skill: {class_name} at {filename}")
            
            # 5. Log the autonomous creation to the audit trail
            self.audit.log_action(
                category="skill_evolution",
                action="create_skill",
                actor="autonomy",
                target=filename,
                params={"goal": goal, "class": class_name},
                outcome="success",
                outcome_detail="Generated, verified, and saved new skill."
            )
            
            return True, f"Successfully evolved and loaded `{class_name}`.", class_name
            
        except Exception as e:
            record_degradation('skill_evolution', e)
            logger.error(f"Failed to evolve skill: {e}")
            return False, f"Exception during skill evolution: {e}", ""


# Singleton
skill_evolver = SkillEvolver()
