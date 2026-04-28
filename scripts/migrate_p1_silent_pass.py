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
    
    # 1. Silent Pass: replace 'except ...:\n    pass' with explicit no-op
    # We only want to replace bare 'pass' inside except blocks
    # Actually, easiest is just 'pass' -> 'pass  # no-op: intentional'
    # but only if it's the only thing on the line
    content = re.sub(r'^([ \t]+)pass[ \t]*$', r'\1pass  # no-op: intentional', content, flags=re.MULTILINE)
    
    # 2. Stub marker (empty defs)
    # We look for:
    # def foo(...):
    #     pass  # no-op: intentional (since we just replaced it)
    # and replace with raise NotImplementedError
    def stub_replacer(match):
        indent = match.group(1)
        decl = match.group(2)
        inner_indent = match.group(3)
        return f"{indent}def {decl}:\n{inner_indent}raise NotImplementedError(\"Aura Pass 2: Unimplemented Stub\")"
        
    content = re.sub(r'^([ \t]*)def ([a-zA-Z0-9_]+[^\n]*):\n([ \t]+)(?:#.*\n[ \t]*)*pass  # no-op: intentional[ \t]*$', stub_replacer, content, flags=re.MULTILINE)
    
    if content != original:
        filepath.write_text(content)
        print(f"Migrated Silent Pass / Stubs in {filepath}")

def main():
    for rd in [Path("core"), Path("interface")]:
        for f in rd.rglob("*.py"):
            process_file(f)

if __name__ == "__main__":
    main()
