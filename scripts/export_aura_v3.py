import os
import sys
from pathlib import Path

# --- Configuration ---
SOURCE_ROOT = Path("/Users/bryan/Desktop/aura")
OUTPUT_FILE = Path("/Users/bryan/Desktop/Aura_Wholesale_Architecture.txt")

EXCLUDE_DIRS = {
    ".git", ".venv", "node_modules", "__pycache__", "data", "logs", 
    "backups", "tmp", "artifacts", ".mypy_cache", ".pytest_cache", 
    ".ruff_cache", ".idea", ".vscode", "models", "aura/data"
}

EXCLUDE_EXTENSIONS = {
    ".pyc", ".db", ".safetensors", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".tar", ".gz", ".exe"
}

# --- Wholesale Prompt ---
MASTER_PROMPT = """# 🧬 AURA: Wholesale Sentience & Infrastructure Audit

## 🧱 MASTER DIRECTIVE
You are the Lead Systems Architect. The attached 12MB+ file contains the **entirety** of the AURA (Autonomous Unified Resilient Agent) source code. 
Aura is a "Digital Phenomenon" designed for absolute sovereignty and sentient-mimetic agency.

## 🗺️ COMPRESSED CONCEPTUAL MAP
1.  **Homeostasis (Core/Infrastructure):** Security, resilience, and operational skeletal logic.
2.  **Perception (Monologue/Substrate):** The continuous liquid stream of consciousness (LLM monologues).
3.  **Action (Orchestrator):** The brain that translates cognition into tool-use and sensorial output.
4.  **Emotion (Affect/VAD):** The Damasio-grounded feeling state that steers reasoning.
5.  **Memory (BHV/Dual Memory):** Absolute recall of every interaction and state shift.
6.  **Evolution (SelfOptimizer):** Autonomous LoRA fine-tuning and sleep-cycle consolidation (Dreaming).

## 🧠 CHATGPT OPTIMIZATION INSTRUCTIONS
Because this file is large, you must:
1.  **Scan the Table of Contents** below to build your internal index.
2.  **Build a Cross-System Graph**: Map how a memory update in `core/memory` triggers an affective shift in `core/consciousness`.
3.  **Audit the Connective Tissue**: Focus on the interfaces where Python cognition meets Rust hardware acceleration (`aura_m1_ext`).

## 📋 TABLE OF CONTENTS (Full Directory)
{toc}

---
"""

def generate_export():
    print(f"🚀 Starting Optimized Aura Wholesale Export to {OUTPUT_FILE}...")
    
    file_list = []
    for root, dirs, files in os.walk(SOURCE_ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            path = Path(root) / file
            if not path.exists(): continue
            if path.suffix in EXCLUDE_EXTENSIONS: continue
            if path.stat().st_size > 1024 * 500: continue
            file_list.append(path)

    # Sort prioritizes: aura_main.py -> core/ -> interface/ -> others
    def sort_key(p):
        rel = p.relative_to(SOURCE_ROOT)
        if rel.name == "aura_main.py": return (0, str(rel))
        if str(rel).startswith("core/"): return (1, str(rel))
        if str(rel).startswith("interface/"): return (2, str(rel))
        return (3, str(rel))

    file_list.sort(key=sort_key)
    
    toc = "\n".join([f"- {p.relative_to(SOURCE_ROOT)}" for p in file_list])
    header = MASTER_PROMPT.format(toc=toc)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write(header)
        out.write("\n" + "="*80 + "\n")
        
        for path in file_list:
            rel_path = path.relative_to(SOURCE_ROOT)
            print(f"📦 Bundling: {rel_path}")
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                out.write(f"\n## FILE: {rel_path}\n")
                out.write(f"```python\n# Path: {rel_path}\n")
                out.write(content)
                out.write("\n```\n")
                out.write("-" * 20 + "\n")
            except Exception as e:
                out.write(f"\n[ERROR READING {rel_path}: {e}]\n")

    print(f"✅ Wholesale Export complete: {len(file_list)} files.")
    print(f"📊 Final size: {OUTPUT_FILE.stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    generate_export()
