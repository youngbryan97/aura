import re
import os

files_to_fix = [
    "tests/conftest.py",
    "tests/test_orchestrator.py",
    "tests/test_architecture_hardening.py"
]

def fix_async_primitives(content):
    # Match asyncio.Queue(...) and asyncio.Lock(...)
    content = re.sub(r'asyncio\.(Queue|Lock|Event|Semaphore)\((.*?)\)', r"getattr(asyncio, '\1')(\2)", content)
    return content

for filepath in files_to_fix:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            content = f.read()
        
        new_content = fix_async_primitives(content)
        
        if new_content != content:
            with open(filepath, 'w') as f:
                f.write(new_content)
            print(f"Fixed {filepath}")

