import sys
import os
from pathlib import Path

# Add the engine to path - v8.0: Robust portability
engine_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(engine_path)

from core.skill_registry import skill_registry

def run_audit():
    skill_dir = Path(engine_path) / "skills"
    print(f"--- Auditing Skills in {skill_dir} ---")
    
    skill_registry.auto_discover(skill_dir)
    
    print(f"\nSuccessfully Registered Skills ({len(skill_registry)}):")
    for skill_name in skill_registry.get_available_skills():
        instance = skill_registry.get(skill_name)
        file_origin = getattr(instance, '__module__', 'unknown')
        print(f"✓ {skill_name} (from {file_origin})")
        
    print("\n--- Files in skills/ directory ---")
    for file in skill_dir.glob("*.py"):
        if file.name == "__init__.py": continue
        stem = file.stem
        if stem not in skill_registry.get_available_skills():
            print(f"GHOST/UNREGISTERED: {file.name}")

if __name__ == "__main__":
    run_audit()
