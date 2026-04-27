import re
import os

EXCLUDE = {'.git', 'venv', '__pycache__', 'node_modules', '.aura'}

def fix_asyncio(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception:
        return

    original = content

    content = re.sub(r'\basyncio\.create_task\(', 'get_task_tracker().create_task(', content)
    content = re.sub(r'\basyncio\.ensure_future\(', 'get_task_tracker().track(', content)
    
    content = re.sub(r'get_task_tracker\(\)\.track\(\s*get_task_tracker\(\)\.create_task\((.*?)\)\s*\)', r'get_task_tracker().track(\1)', content)
    content = re.sub(r'get_task_tracker\(\)\.track_task\(\s*get_task_tracker\(\)\.create_task\((.*?)\)\s*\)', r'get_task_tracker().track_task(\1)', content)
    
    if content != original:
        if 'get_task_tracker' not in original and 'get_task_tracker().' in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    lines.insert(i, 'from core.utils.task_tracker import get_task_tracker')
                    break
            content = '\n'.join(lines)
        with open(filepath, 'w') as f:
            f.write(content)

def fix_write_text(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception:
        return

    original = content
    lines = content.split('\n')
    new_lines = []
    changed = False
    
    for line in lines:
        if '.write_text(' in line:
            parts = line.split('.write_text(')
            if len(parts) >= 2:
                for i in range(len(parts) - 1):
                    left = parts[i]
                    right = parts[i+1]
                    m = re.search(r'([a-zA-Z0-9_\.]+)$', left)
                    if m:
                        obj = m.group(1)
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

for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in EXCLUDE]
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            fix_asyncio(path)
            fix_write_text(path)
