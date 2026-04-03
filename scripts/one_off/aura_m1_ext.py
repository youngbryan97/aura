import os
import sys
import logging
import ast
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("Aura.M1ExtShim")

def pin_to_p_cores():
    """Stub for Apple Silicon P-core pinning."""
    logger.debug("M1 Shim: Requested P-core pinning (No-op)")

def pin_to_e_cores():
    """Stub for Apple Silicon E-core pinning."""
    logger.debug("M1 Shim: Requested E-core pinning (No-op)")

def neon_dot_product(a: List[float], b: List[float]) -> float:
    """Python fallback for NEON-accelerated dot product."""
    try:
        import numpy as np
        return float(np.dot(a, b))
    except ImportError:
        return sum(x * y for x, y in zip(a, b))

def build_skill_index() -> Dict[str, Any]:
    """
    Python-based skill discovery to satisfy Transcendent indexing requirements.
    Scans core/skills/ and project skills/ directory.
    """
    index = {}
    
    # Search paths
    project_root = Path(__file__).parent
    skill_paths = [
        (project_root / "core" / "skills", "core.skills"),
        (project_root / "skills", "skills")
    ]
    
    for skill_dir, module_prefix in skill_paths:
        if not skill_dir.exists():
            continue
            
        for path in skill_dir.glob("*.py"):
            if path.name.startswith("_"):
                continue
                
            try:
                with open(path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                    
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        skill_name = None
                        description = "Core system skill."
                        
                        # Look for name and description class attributes
                        for item in node.body:
                            if isinstance(item, ast.Assign):
                                for target in item.targets:
                                    if isinstance(target, ast.Name):
                                        if target.id == "name" and isinstance(item.value, ast.Constant):
                                            skill_name = item.value.value
                                        elif target.id == "description" and isinstance(item.value, ast.Constant):
                                            description = item.value.value
                        
                        if skill_name:
                            index[skill_name] = {
                                "description": description,
                                "module_path": f"{module_prefix}.{path.stem}",
                                "class_name": node.name,
                                "execution_profile": "cpu",
                                "timeout_seconds": 30,
                                "memory_mb_estimate": 256
                            }
            except Exception as e:
                logger.error(f"Failed to index skill {path}: {e}")
                
    return index

if __name__ == "__main__":
    # Self-test
    idx = build_skill_index()
    print(f"✅ Shim index built: {len(idx)} skills found.")
