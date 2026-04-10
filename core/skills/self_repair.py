"""Self-Repair Skill — Autonomous diagnosis and repair of broken components.

Analyzes errors, locates the source file, reads the code, asks the LLM
for a targeted fix, and saves a repair proposal. Integrates with the
learning system to remember what worked.
"""
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from core.config import config
from core.container import ServiceContainer
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.SelfRepair")


class SelfRepairInput(BaseModel):
    component: Optional[str] = Field(None, description="Name of the broken component or skill.")
    error: Optional[str] = Field(None, description="Error message or failure pattern observed.")
    auto_apply: bool = Field(False, description="If True, attempt to apply the fix automatically (requires GrowthLadder approval).")


class SelfRepairSkill(BaseSkill):
    """Autonomous self-healing of Aura's own tools and modules."""

    name = "self_repair"
    description = "Diagnose and repair broken skills or system components. Analyzes errors, reads source, proposes fixes."
    input_model = SelfRepairInput
    timeout_seconds = 60.0
    metabolic_cost = 2

    async def execute(self, params: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}

        if isinstance(params, dict):
            component = params.get("component")
            error = params.get("error")
            auto_apply = params.get("auto_apply", False)
        elif isinstance(params, SelfRepairInput):
            component = params.component
            error = params.error
            auto_apply = params.auto_apply
        else:
            component = str(params) if params else None
            error = None
            auto_apply = False

        if not component:
            return {
                "ok": True,
                "summary": "Self-Repair system is online. Specify a component to repair.",
                "available_actions": ["repair_component", "scan_system"],
            }

        logger.info("Self-Repair: diagnosing '%s' (error: %s)", component, error)

        # 1. Locate the component file
        target_path = self._locate_component(component)
        if not target_path:
            return {"ok": False, "error": f"Could not locate component: {component}"}

        # 2. Read the source
        try:
            source_code = Path(target_path).read_text(errors="replace")
        except Exception as e:
            return {"ok": False, "error": f"Could not read {target_path}: {e}"}

        # 3. Get the LLM to diagnose and propose a fix
        brain = ServiceContainer.get("cognitive_engine", default=None)
        if not brain or not hasattr(brain, "think"):
            return {
                "ok": False,
                "error": "Self-Repair requires cognitive engine (not available).",
                "file": target_path,
            }

        try:
            from core.brain.types import ThinkingMode
            prompt = (
                f"FILE: {target_path}\n"
                f"CONTENT:\n{source_code[:4000]}\n\n"
                f"ERROR: {error or 'Unknown — please diagnose'}\n\n"
                "TASK: Identify the bug and provide a minimal, precise Python fix.\n"
                "Rules:\n"
                "- Show the exact code to change (old → new).\n"
                "- If it's a missing import, show the import line.\n"
                "- If it's a missing dependency, say which package.\n"
                "- Be surgical — minimal changes only.\n"
                "- Do NOT rewrite the entire file."
            )
            result = await brain.think(prompt, mode=ThinkingMode.CRITICAL)
            fix_content = getattr(result, "content", str(result))
        except Exception as e:
            return {"ok": False, "error": f"Diagnosis failed: {e}"}

        # 4. Check for dependency requirements
        if "pip install" in fix_content:
            package_match = re.search(r'pip install\s+([\w-]+)', fix_content)
            package = package_match.group(1) if package_match else "unknown"
            if not re.match(r'^[a-zA-Z0-9_-]+$', package):
                return {"ok": False, "error": f"Suspicious package name: {package}"}
            return {
                "ok": False,
                "action": "manual_install_required",
                "summary": f"Missing dependency: '{package}'. Run: pip install {package}",
                "proposal": fix_content[:500],
            }

        # 5. Save repair proposal
        patch_dir = config.paths.data_dir / "repairs"
        patch_dir.mkdir(parents=True, exist_ok=True)
        patch_path = patch_dir / f"repair_{component}_{int(time.time())}.patch"
        patch_path.write_text(fix_content)

        # 6. Record in learning system
        try:
            learning = ServiceContainer.get("learning_system", default=None)
            if learning and hasattr(learning, "record_fix"):
                learning.record_fix(
                    component=component,
                    error=error or "",
                    fix=fix_content[:500],
                    success=True,
                )
        except Exception:
            pass

        # 7. Record in WorldState
        try:
            from core.world_state import get_world_state
            get_world_state().record_event(
                f"Self-repair proposal for {component}",
                source="self_repair",
                salience=0.5,
                ttl=3600,
            )
        except Exception:
            pass

        return {
            "ok": True,
            "summary": f"Repair proposal generated for {component}. Saved to {patch_path.name}",
            "proposal": fix_content[:500],
            "file": target_path,
            "patch_path": str(patch_path),
        }

    @staticmethod
    def _locate_component(name: str) -> Optional[str]:
        """Search common directories for the component file."""
        base = config.paths.base_dir
        search_dirs = ["core/skills", "core", "skills", "infrastructure", "core/brain",
                       "core/consciousness", "core/agency", "core/phases"]
        for d in search_dirs:
            candidate = base / d / f"{name}.py"
            if candidate.exists():
                return str(candidate)
        # Fuzzy search
        for d in search_dirs:
            dir_path = base / d
            if dir_path.is_dir():
                for f in dir_path.glob("*.py"):
                    if name.replace("-", "_") in f.stem:
                        return str(f)
        return None
