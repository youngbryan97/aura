import re
import os

def fix_write_text(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

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

for root, _, files in os.walk('core'):
    for f in files:
        if f.endswith('.py'):
            fix_write_text(os.path.join(root, f))
