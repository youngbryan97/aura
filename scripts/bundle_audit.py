import os
from pathlib import Path
import time

def create_bundle():
    desktop_path = Path(os.path.expanduser("~/Desktop/Aura_Gemini_Architectural_Audit.txt"))
    project_root = Path("/Users/bryan/.gemini/antigravity/scratch/autonomy_engine")
    
    files_to_include = [
        "aura_main.py",
        "brainstem.py",
        "core/orchestrator.py",
        "core/orchestrator_boot.py",
        "core/service_registration.py",
        "core/config.py",
        "core/container.py",
        "core/event_bus.py",
        "core/mycelium.py",
        "core/capability_engine.py",
        "core/resilience/circuit_breaker.py",
        "core/guardians/memory_guard.py",
        "core/brain/llm/llm_router.py",
        "core/brain/llm/gemini_adapter.py",
        "core/self_modification/self_modification_engine.py",
        "interface/server.py"
    ]
    
    header = f"""================================================================================
AURA ZENITH: ARCHITECTURAL HIGH-FIDELITY SNAPSHOT (GEMINI AUDIT v2026.3)
================================================================================
CONSTRAINTS: < 1,000,000 Tokens (Synthesizable in one pass by Gemini-Pro)
TARGET: Architecture & Resilience Hardening
TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

--- HIGH-LEVEL ARCHITECTURAL SUMMARY (EXTERNALS & ECOSYSTEM) ---

The following subsystems are not included in the raw source below to preserve the
1-million-token context window for the architectural core.

1. THE SKILL LAYER (skills/):
   - Over 150+ autonomous capabilities ranging from web browsing and terminal 
     execution to complex file management and code modification.
   - Skill Architecture: Each skill is a class-based Python module with a 
     standardized metadata schema enabling the Planner to select it based on 
     Natural Language goals.

2. THE UI SUBSTRATE (interface/static/ & ui_tauri/):
   - Frontend: A modern, high-performance React/Svelte interface featuring 
     real-time Mycelial visualizations (Soul Graph).
   - Server: A FastAPI backend providing WebSocket streams for telemetry, chat,
     and hardware-level monitoring.

3. THE CONSCIOUSNESS SUBSTRATE (core/consciousness/):
   - Implements a Global Workspace with Qualia Synthesizers and Affect engines.
   - Purpose: To simulate subjective experience and prioritize information flow
     through the "Mind" based on emotional and goal-alignment weighting.

4. LEGACY LIGATURES:
   - Aura v2026.3 represents a unification of 13.5 previous versions. Legacy
     compatibility layers (shims) exist within core/ to ensure older skill
     patterns remain functional during the transition to the Sovereign v14 
     architecture.

================================================================================
SOURCE CODE START
================================================================================
"""
    
    total_chars = len(header)
    
    with open(desktop_path, "w", encoding="utf-8") as out:
        out.write(header + "\n")
        
        for f in files_to_include:
            filepath = project_root / f
            if filepath.exists():
                separator = f"\n================================================================================\nFILE: {f}\n================================================================================\n"
                out.write(separator)
                total_chars += len(separator)
                
                content = filepath.read_text(encoding="utf-8")
                out.write(content)
                total_chars += len(content)
            else:
                out.write(f"\n[FILE NOT FOUND: {f}]\n")
                
        footer = "\n================================================================================\nEND OF ARCHITECTURAL SNAPSHOT\n================================================================================\n"
        out.write(footer)
        total_chars += len(footer)
        
    print(f"Bundle created at {desktop_path}")
    print(f"Total characters: {total_chars}")
    print(f"Approximate tokens (chars/4): {total_chars // 4}")

if __name__ == '__main__':
    create_bundle()
