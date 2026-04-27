import os
import re
from pathlib import Path

SOURCE_DIR = "/Users/bryan/.aura/live-source"

FUTURE_IMPORT = "from __future__ import annotations"

# Regex to detect a future import line (allow leading whitespace)
future_pattern = re.compile(r"^(\s*)from __future__ import annotations\s*$")


def fix_file(file_path: Path):
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return False
    lines = text.splitlines()
    # Find the future import line(s)
    future_lines = [i for i, line in enumerate(lines) if future_pattern.match(line)]
    if not future_lines:
        return False
    # Remove all future import lines
    future_content = [lines[i] for i in future_lines]
    for i in reversed(future_lines):
        del lines[i]
    # Determine where to insert: after module docstring if present, otherwise at top
    insert_index = 0
    # Detect docstring at start of file (triple quotes)
    if lines:
        first_line = lines[0].lstrip()
        if first_line.startswith('"""') or first_line.startswith("'''"):
            # Find closing docstring
            quote = first_line[:3]
            for idx in range(1, len(lines)):
                if lines[idx].strip().endswith(quote):
                    insert_index = idx + 1
                    break
    # Insert future import(s) (preserve original indentation of first occurrence)
    indent = future_content[0][:future_content[0].find('from')]
    new_line = f"{indent}{FUTURE_IMPORT}"
    lines.insert(insert_index, new_line)
    # Write back
    new_text = "\n".join(lines) + "\n"
    file_path.write_text(new_text, encoding="utf-8")
    return True


def main():
    base = Path(SOURCE_DIR)
    py_files = list(base.rglob('*.py'))
    fixed = 0
    for p in py_files:
        if fix_file(p):
            fixed += 1
    print(f"Processed {len(py_files)} .py files, fixed {fixed} files with future import relocation.")

if __name__ == "__main__":
    main()
