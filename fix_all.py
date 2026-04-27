from core.runtime.atomic_writer import atomic_write_text
import re
import os

def fix_asyncio(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # Fix: get_task_tracker().create_task(...) -> get_task_tracker().create_task(...)
    # Exclude cases where it's already get_task_tracker().track(...)
    
    # Simple strategy: just replace asyncio.create_task with get_task_tracker().create_task
    # and asyncio.ensure_future with get_task_tracker().track
    content = re.sub(r'\basyncio\.create_task\(', 'get_task_tracker().create_task(', content)
    content = re.sub(r'\basyncio\.ensure_future\(', 'get_task_tracker().track(', content)
    
    # Remove nested track(create_task(...)) if any
    content = re.sub(r'get_task_tracker\(\)\.track\(\s*get_task_tracker\(\)\.create_task\((.*?)\)\s*\)', r'get_task_tracker().track(\1)', content)
    content = re.sub(r'get_task_tracker\(\)\.track_task\(\s*get_task_tracker\(\)\.create_task\((.*?)\)\s*\)', r'get_task_tracker().track_task(\1)', content)

    # DIRECT_WRITE_TEXT fixes:
    # Usually: atomic_write_text(path, ...)
    # Let's see if we can find them
    
    if content != original:
        if 'get_task_tracker' not in original and 'get_task_tracker().' in content:
            # add import
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    lines.insert(i, 'from core.utils.task_tracker import get_task_tracker')
                    break
            content = '\n'.join(lines)
        with open(filepath, 'w') as f:
            f.write(content)

for root, _, files in os.walk('core'):
    for f in files:
        if f.endswith('.py'):
            fix_asyncio(os.path.join(root, f))
