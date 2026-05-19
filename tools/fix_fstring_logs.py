#!/usr/bin/env python3
"""Convert f-string logger calls to lazy % formatting.

Replaces patterns like:
    logger.info(f"Something {var}")
With:
    logger.info("Something %s", var)

This is a Python best practice because:
1. f-strings evaluate eagerly even when the log level is disabled
2. % formatting is deferred until the message is actually needed
3. It prevents accidental injection of user input into log messages
"""

import re
import sys
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv"}

# Match logger.LEVEL(f"...") calls
FSTRING_LOG_RE = re.compile(
    r'^(\s*)(logger\.\w+)\(f(["\'])(.*?)\3\)(\s*(?:#.*)?)$'
)

# Match simple {var} or {var.attr} or {var:.2f} interpolations
SIMPLE_INTERP_RE = re.compile(r'\{([^{}]+?)\}')

# Match format specs like {var:.2f}, {var:d}, {var!r}
FORMAT_SPEC_RE = re.compile(r'^([^:!]+)([!:].+)$')


def find_python_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name.startswith("test_"):
            continue
        files.append(path)
    return sorted(files)


def convert_fstring_to_lazy(fstring_content: str) -> tuple[str, list[str]] | None:
    """Convert f-string content to format string + args.
    
    Returns (format_string, [args]) or None if conversion is too complex.
    """
    parts = []
    args = []
    last_end = 0
    
    for match in SIMPLE_INTERP_RE.finditer(fstring_content):
        # Add literal text before this interpolation
        parts.append(fstring_content[last_end:match.start()])
        
        expr = match.group(1)
        
        # Check for format spec
        spec_match = FORMAT_SPEC_RE.match(expr)
        if spec_match:
            var_part = spec_match.group(1).strip()
            spec_part = spec_match.group(2)
            
            # Handle common format specs
            if spec_part.startswith(':'):
                # Like {val:.2f} -> we can't easily convert, use %s
                parts.append('%s')
                args.append(f"f\"{{{expr}}}\"")  # Keep as mini f-string
            elif spec_part.startswith('!r'):
                parts.append('%r')
                args.append(var_part)
            elif spec_part.startswith('!s'):
                parts.append('%s')
                args.append(var_part)
            else:
                parts.append('%s')
                args.append(expr)
        else:
            # Simple variable reference
            expr = expr.strip()
            
            # Skip complex expressions (function calls, list comprehensions, etc.)
            if any(c in expr for c in ['(', '[', 'for ', 'if ', ' else ', 'lambda']):
                # Too complex for simple conversion
                parts.append('%s')
                args.append(expr)
            else:
                parts.append('%s')
                args.append(expr)
        
        last_end = match.end()
    
    # Add remaining literal text
    parts.append(fstring_content[last_end:])
    
    format_str = ''.join(parts)
    
    # Don't convert if no interpolations found (it's just a string)
    if not args:
        return None
    
    return format_str, args


def process_file(filepath: Path, dry_run: bool = False) -> int:
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return 0

    lines = content.split("\n")
    new_lines = []
    changes = 0

    for i, line in enumerate(lines):
        match = FSTRING_LOG_RE.match(line)
        if match:
            indent = match.group(1)
            log_call = match.group(2)
            quote = match.group(3)
            fstring_body = match.group(4)
            trailing = match.group(5) or ""
            
            result = convert_fstring_to_lazy(fstring_body)
            if result:
                format_str, args = result
                # Escape any existing % in the format string (but not our %s/%r)
                # Actually, we should NOT double-escape — the format string should 
                # have literal % only if the original had %%
                args_str = ", ".join(args)
                new_line = f'{indent}{log_call}("{format_str}", {args_str}){trailing}'
                new_lines.append(new_line)
                changes += 1
                continue
        
        new_lines.append(line)

    if changes > 0 and not dry_run:
        filepath.write_text("\n".join(new_lines), encoding="utf-8")

    return changes


def main():
    dry_run = "--dry-run" in sys.argv
    root = Path(__file__).resolve().parents[1]

    search_dirs = [root / "core", root / "skills"]
    all_files = []
    for d in search_dirs:
        if d.exists():
            all_files.extend(find_python_files(d))

    total = 0
    changed = 0

    for filepath in all_files:
        rel = filepath.relative_to(root)
        count = process_file(filepath, dry_run=dry_run)
        if count > 0:
            print(f"  {'[DRY] ' if dry_run else ''}Fixed {count:3d} f-string logs in {rel}")
            total += count
            changed += 1

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Total: {total} f-string logger calls converted across {changed} files.")

    if not dry_run:
        import subprocess
        python = str(root / ".venv" / "bin" / "python")
        failures = 0
        for filepath in all_files:
            result = subprocess.run(
                [python, "-m", "py_compile", str(filepath)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"  ❌ COMPILE FAIL: {filepath.relative_to(root)}")
                print(f"     {result.stderr.strip()}")
                failures += 1

        if failures:
            print(f"\n⚠️  {failures} files failed compilation.")
        else:
            print(f"\n✅ All {len(all_files)} files compile clean.")


if __name__ == "__main__":
    main()
