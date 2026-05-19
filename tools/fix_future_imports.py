#!/usr/bin/env python3
"""Fix files where 'import logging' was inserted before 'from __future__' imports.

Moves the logging import lines to after the __future__ imports.
"""

import sys
from pathlib import Path

BROKEN_FILES = [
    "core/environments/terminal_grid/nethack_adapter.py",
    "core/learning/rsi_lineage.py",
    "core/runtime/receipts.py",
    "core/runtime/diagnostics_bundle.py",
    "core/runtime/persistence_ownership.py",
    "core/runtime/causal_trace.py",
    "core/runtime/autonomy_conductor.py",
    "core/runtime/backup_restore.py",
    "core/runtime/audit_chain.py",
    "core/governance/will_client.py",
    "core/brain/llm/substrate_token_generator.py",
    "core/self_modification/mutation_safety.py",
    "core/self_modification/structural_mutator.py",
]


def fix_file(filepath: Path):
    content = filepath.read_text(encoding="utf-8")
    lines = content.split("\n")
    
    # Find and remove the misplaced logging lines at the top
    logging_lines = []
    remaining = []
    removed_logging = False
    
    for i, line in enumerate(lines):
        if not removed_logging and (line.strip() == "import logging" or line.strip().startswith('logger = logging.getLogger(')):
            logging_lines.append(line)
            continue
        if logging_lines and line.strip() == "":
            # Skip the blank line after the removed logging imports
            remaining.append(line)
            removed_logging = True
            continue
        remaining.append(line)
        if logging_lines and not removed_logging:
            removed_logging = True
    
    if not logging_lines:
        print(f"  ⚠️  No misplaced logging found in {filepath.name}")
        return
    
    # Find where to insert (after from __future__ and its blank line)
    insert_at = 0
    found_future = False
    for i, line in enumerate(remaining):
        if "from __future__" in line:
            found_future = True
            insert_at = i + 1
        elif found_future and line.strip() == "":
            insert_at = i + 1
            break
        elif found_future and line.strip():
            insert_at = i
            break
    
    # Check if logging already exists further down
    rest_content = "\n".join(remaining[insert_at:])
    if "import logging" in rest_content:
        # Already has logging import, just remove the duplicate at top
        filepath.write_text("\n".join(remaining), encoding="utf-8")
        print(f"  ✅ Removed duplicate logging import from {filepath.name}")
        return
    
    # Insert logging lines after __future__
    for j, ll in enumerate(logging_lines):
        remaining.insert(insert_at + j, ll)
    
    filepath.write_text("\n".join(remaining), encoding="utf-8")
    print(f"  ✅ Fixed import order in {filepath.name}")


def main():
    root = Path(__file__).resolve().parents[1]
    
    for rel in BROKEN_FILES:
        filepath = root / rel
        if filepath.exists():
            fix_file(filepath)
        else:
            print(f"  ⚠️  Not found: {rel}")
    
    # Compile check
    import subprocess
    python = str(root / ".venv" / "bin" / "python")
    failures = 0
    for rel in BROKEN_FILES:
        filepath = root / rel
        if not filepath.exists():
            continue
        result = subprocess.run(
            [python, "-m", "py_compile", str(filepath)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  ❌ STILL BROKEN: {rel}")
            print(f"     {result.stderr.strip()}")
            failures += 1
        else:
            print(f"  ✅ Compiles: {rel}")
    
    if failures:
        print(f"\n⚠️  {failures} files still broken.")
    else:
        print(f"\n✅ All {len(BROKEN_FILES)} files compile clean.")


if __name__ == "__main__":
    main()
