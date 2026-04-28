import re
import sys
from pathlib import Path

def migrate_file(filepath):
    path = Path(filepath)
    if not path.exists():
        return
        
    content = path.read_text()
    original = content
    
    if "except Exception as " not in content:
        return

    if "from core.runtime.errors import record_degradation" not in content:
        # Find the end of __future__ imports if they exist
        future_match = list(re.finditer(r'^from __future__ import .*$', content, re.MULTILINE))
        if future_match:
            idx = future_match[-1].end()
            content = content[:idx] + "\nfrom core.runtime.errors import record_degradation\n" + content[idx:]
        else:
            match = re.search(r'^(?:import |from [a-zA-Z0-9_\.]+ import )', content, flags=re.MULTILINE)
            if match:
                idx = match.start()
                content = content[:idx] + "from core.runtime.errors import record_degradation\n" + content[idx:]
            else:
                content = "from core.runtime.errors import record_degradation\n" + content

    subsystem = path.stem
    
    def replacer(match):
        indent = match.group(1)
        exc_name = match.group(2)
        added_indent = "\t" if "\t" in indent else "    "
        return f"{indent}except Exception as {exc_name}:\n{indent}{added_indent}record_degradation('{subsystem}', {exc_name})\n"
        
    new_content = re.sub(r'^([ \t]+)except Exception as (\w+):[ \t]*\n', replacer, content, flags=re.MULTILINE)
    
    if new_content != original:
        path.write_text(new_content)
        print(f"Migrated {path}")

if __name__ == "__main__":
    for f in sys.argv[1:]:
        migrate_file(f)
