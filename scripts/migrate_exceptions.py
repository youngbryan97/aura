import re
import sys
from pathlib import Path

def migrate_file(filepath):
    path = Path(filepath)
    if not path.exists():
        return
        
    content = path.read_text()
    original = content
    
    # Ensure import exists
    if "from core.runtime.errors import record_degradation" not in content and "except Exception as " in content:
        import_match = list(re.finditer(r'^import .*$|^from .* import .*$', content, re.MULTILINE))
        if import_match:
            last_import = import_match[-1]
            content = content[:last_import.end()] + "\nfrom core.runtime.errors import record_degradation\n" + content[last_import.end():]
        else:
            content = "from core.runtime.errors import record_degradation\n" + content

    subsystem = path.stem
    
    def replacer(match):
        indent = match.group(1)
        exc_name = match.group(2)
        return f"{indent}except Exception as {exc_name}:\n{indent}    record_degradation('{subsystem}', {exc_name})\n"
        
    new_content = re.sub(r'^([ \t]+)except Exception as (\w+):[ \t]*\n', replacer, content, flags=re.MULTILINE)
    
    if new_content != original:
        path.write_text(new_content)
        print(f"Migrated {path}")

if __name__ == "__main__":
    for f in sys.argv[1:]:
        migrate_file(f)
