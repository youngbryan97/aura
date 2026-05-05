import re
import os

def fix_write_text(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    original = content
    
    # Simple regex to replace atomic_write_text(var, content) with atomic_write_text(var, content)
    # We'll use a regex that matches expressions like `atomic_write_text(my_path, some_content)`
    # This might have issues with nested parentheses but it works for most cases
    
    # Need to match: [some valid expression]...)
    # Let's match from the start of the object to the write_text
    
    lines = content.split('\n')
    new_lines = []
    changed = False
    
    for line in lines:
        if '' in line:
            # find what's before 
            parts = line.split('')
            if len(parts) >= 2:
                for i in range(len(parts) - 1):
                    left = parts[i]
                    right = parts[i+1]
                    # extract the variable/expression on the left
                    # this is tricky, let's just find the last word/attribute
                    m = re.search(r'([a-zA-Z0-9_\.]+)$', left)
                    if m:
                        obj = m.group(1)
                        # replace the atomic_write_text(obj,  with atomic_write_text(obj,
                        left_rem = left[:-len(obj)]
                        parts[i] = left_rem
                        parts[i+1] = f'atomic_write_text({obj}, ' + right
                        changed = True
                
                new_lines.append("".join(parts))
                continue
        new_lines.append(line)
        
    content = '\n'.join(new_lines)
    if changed:
        if 'atomic_write_text' not in original:
            for i, line in enumerate(new_lines):
                if line.startswith('import ') or line.startswith('from '):
                    new_lines.insert(i, 'from core.runtime.atomic_writer import atomic_write_text')
                    break
            content = '\n'.join(new_lines)
            
        with open(filepath, 'w') as f:
            f.write(content)

# We will run this on all files that had DIRECT_WRITE_TEXT warnings
files_to_fix = [
    "core/brain/personality_engine.py",
    "core/brain/personality_kernel.py",
    "core/adaptation/abstraction_engine.py",
    "core/adaptation/adaptive_immunity.py",
    "core/ops/metabolic_monitor.py",
    "core/skills/self_repair.py",
    "core/skills/memory_sync.py",
    "core/skills/memory_ops.py",
    "core/embodiment/resistance_sandbox.py",
    "core/embodiment/world_bridge.py",
    "core/sovereignty/wallet.py",
    "core/self_modification/growth_ladder.py",
    "core/self_modification/code_repair.py",
    "core/self_modification/safe_pipeline.py",
    "core/self_modification/safe_modification.py",
    "core/self_modification/shadow_runtime.py",
    "core/self_modification/self_modification_engine.py",
    "core/control/dynamic_router.py",
    "core/brain/llm/mlx_client.py",
    "core/brain/llm/local_server_client.py"
]

for filepath in files_to_fix:
    if os.path.exists(filepath):
        fix_write_text(filepath)
