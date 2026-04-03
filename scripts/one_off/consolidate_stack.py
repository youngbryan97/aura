import os
from pathlib import Path

# Files to include in the consolidated stack
files = [
    "core/orchestrator/main.py",
    "core/orchestrator/services.py",
    "core/orchestrator/mixins/boot/boot_resilience.py",
    "core/brain/llm/mlx_client.py",
    "core/brain/llm/mlx_worker.py",
    "core/brain/cognitive_engine.py",
    "core/utils/concurrency.py",
    "interface/server.py",
    "core/phases/affect_update.py",
    "core/coordinators/metabolic_coordinator.py",
    "core/consciousness/liquid_substrate.py"
]

output_file = "/Users/bryan/Desktop/aura_chat_stack.txt"
project_root = "/Users/bryan/Desktop/aura"

with open(output_file, "w") as out:
    for rel_path in files:
        abs_path = os.path.join(project_root, rel_path)
        if os.path.exists(abs_path):
            out.write(f"\n{'='*80}\n")
            out.write(f"FILE: {rel_path}\n")
            out.write(f"{'='*80}\n\n")
            with open(abs_path, "r") as f:
                out.write(f.read())
            out.write("\n")
        else:
            out.write(f"\nMISSING FILE: {rel_path}\n")

print(f"✅ Consolidated chat stack written to {output_file}")
