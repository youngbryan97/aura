from core.runtime.atomic_writer import atomic_write_text
import os
import re
from pathlib import Path

files_to_patch = [
    "core/runtime_tools.py",
    "core/motivation/goal_hierarchy.py",
    "core/skills/toggle_senses.py",
    "core/logging_hmac.py",
    "core/world_model/acg.py",
    "core/world_model/belief_graph.py",
    "core/meta/cognitive_trace.py",
    "core/biography.py",
    "core/memory/knowledge_ledger.py",
    "core/memory/learning/tool_learning.py",
    "core/memory/knowledge_graph.py",
    "core/heartstone_directive.py",
    "core/privacy_stealth.py",
    "core/device_discovery.py",
    "core/memory/episodic_memory.py",
    "core/self_preservation_integration.py"
]

# Resolve base_dir relative to this script
base_dir = Path(__file__).resolve().parent.parent

for rel_path in files_to_patch:
    file_path = base_dir / rel_path
    if not file_path.exists():
        continue
    content = file_path.read_text()
    original_content = content
    
    # Replacements
    content = re.sub(r'os\.path\.expanduser\("~/\.aura/([^"]+)"\)', r'str(config.paths.home_dir / "\1")', content)
    content = re.sub(r'os\.path\.expanduser\("~/\.aura"\)', r'str(config.paths.home_dir)', content)
    content = re.sub(r'Path\.home\(\) / "\.aura" / "([^"]+)"', r'config.paths.home_dir / "\1"', content)
    content = content.replace('"~/aura_backup"', 'str(config.paths.home_dir / "aura_backup")')
    
    if content != original_content:
        # Needs to import config if not present
        if "from core.config import config" not in content and "import config" not in content:
            # Add import after the last import
            import_stmt = "from core.config import config"
            lines = content.splitlines()
            last_import = 0
            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    last_import = i
            lines.insert(last_import + 1, import_stmt)
            content = "\n".join(lines) + "\n"
        
        atomic_write_text(file_path, content)
        print(f"Patched {rel_path}")

print("Patching complete.")
