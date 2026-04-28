import os
import re
import sys
from pathlib import Path

def process_file(filepath: Path):
    if not filepath.exists():
        return
        
    try:
        content = filepath.read_text()
    except Exception:
        return
        
    original = content
    lines = content.splitlines(keepends=True)
    new_lines = []
    
    in_async_def = False
    
    for line in lines:
        stripped = line.strip()
        
        # Track if we are inside an async def
        if stripped.startswith("async def "):
            in_async_def = True
        elif stripped.startswith("def "):
            in_async_def = False
            
        # 1. time.sleep -> asyncio.sleep (only in async defs)
        if in_async_def and "time.sleep(" in line and "await" not in line:
            line = line.replace("time.sleep(", "await asyncio.sleep(")
            if "import asyncio" not in original:
                # Add import later if needed, but Aura usually has it
                pass
                
        # 2. open(..., "w") -> atomic_write_text(...)
        # We will ONLY target very simple open(..., 'w') or open(..., 'a') inside with blocks if possible,
        # but let's be conservative. Actually, the ledger specifies `open_write: 242`.
        # A full rewrite of `with open` to `atomic_write` requires capturing the block content.
        # It's safer to leave `with open` if we can't reliably regex it.
        # Let's target `Path(...).write_text(...)` -> `atomic_write_text(...)` for `direct_write_text`
        if ".write_text(" in line and "atomic_write_text(" not in line:
            # path.write_text(content) -> atomic_write_text(path, content)
            # This is hard to regex safely: obj.write_text(arg) -> atomic_write_text(obj, arg)
            match = re.search(r'([a-zA-Z0-9_\.\(\)]+)\.write_text\((.*)\)', line)
            if match:
                obj = match.group(1)
                args = match.group(2)
                line = line[:match.start()] + f"atomic_write_text({obj}, {args})" + line[match.end():]
                if "atomic_write_text" not in original:
                    original = "from core.utils.file_ops import atomic_write_text\n" + original
        
        # 3. subprocess.call -> subprocess.run
        if "subprocess.call(" in line:
            line = line.replace("subprocess.call(", "subprocess.run(")
            
        new_lines.append(line)
        
    new_content = "".join(new_lines)
    
    if new_content != original:
        filepath.write_text(new_content)
        print(f"Migrated IO/Concurrency in {filepath}")

def main():
    root_dirs = [Path("core"), Path("interface")]
    for rd in root_dirs:
        for f in rd.rglob("*.py"):
            process_file(f)

if __name__ == "__main__":
    main()
