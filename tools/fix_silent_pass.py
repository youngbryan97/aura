#!/usr/bin/env python3
"""Replace silent 'pass' in except blocks with proper logging.

Finds patterns like:
    except SomeError:
        pass

And replaces with:
    except SomeError as _exc:
        logger.debug("Suppressed %s in %s: %s", type(_exc).__name__, __name__, _exc)

Also handles inline: `except SomeError: pass`

Skips blocks that already have logging, record_degradation, or a comment 
explaining the intentional no-op.
"""

import re
import sys
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv"}

# Patterns that indicate the pass is intentional/documented
INTENTIONAL_MARKERS = [
    "# no-op",
    "# intentional",
    "# expected",
    "# ignore",
    "# safe to ignore",
    "# non-critical",
]


def find_python_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name.startswith("test_"):
            continue
        files.append(path)
    return sorted(files)


def extract_module_name(filepath: Path, root: Path) -> str:
    """Get a reasonable module name from filepath."""
    try:
        rel = filepath.relative_to(root)
        return str(rel.with_suffix("")).replace("/", ".")
    except ValueError:
        return filepath.stem


def process_file(filepath: Path, root: Path, dry_run: bool = False) -> int:
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return 0

    lines = content.split("\n")
    new_lines = []
    changes = 0
    module_name = extract_module_name(filepath, root)
    needs_logger = False
    has_logger = "logger" in content or "logging" in content
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Pattern 1: Inline `except SomeError: pass`
        inline_match = re.match(
            r"^(\s*)except\s+(\([^)]+\)|\w[\w.,\s]*)\s*:\s*pass\s*$", stripped
        )
        if inline_match and not any(m in line for m in INTENTIONAL_MARKERS):
            indent = line[:len(line) - len(line.lstrip())]
            exc_types = inline_match.group(2).strip()
            new_lines.append(f"{indent}except {exc_types} as _exc:")
            new_lines.append(f'{indent}    logger.debug("Suppressed %s in {module_name}: %s", type(_exc).__name__, _exc)')
            changes += 1
            needs_logger = True
            i += 1
            continue

        # Pattern 2: Multi-line except + pass on next line
        except_match = re.match(
            r"^(\s*)except\s+(\([^)]+\)|\w[\w.,\s]*)\s*:\s*$", stripped
        )
        if not except_match:
            # Also match `except (X, Y) as name:`
            except_match = re.match(
                r"^(\s*)except\s+(\([^)]+\)|\w[\w.,\s]*)\s+as\s+\w+\s*:\s*$", stripped
            )

        if except_match and i + 1 < len(lines):
            next_stripped = lines[i + 1].strip()
            if next_stripped == "pass":
                # Check if there's a comment on the pass line or nearby
                has_comment = False
                if i + 2 < len(lines):
                    comment_line = lines[i + 2].strip()
                    if comment_line.startswith("#"):
                        has_comment = any(m in comment_line.lower() for m in INTENTIONAL_MARKERS)

                pass_line_comment = any(m in lines[i + 1].lower() for m in INTENTIONAL_MARKERS)

                if not has_comment and not pass_line_comment:
                    indent = line[:len(line) - len(line.lstrip())]
                    body_indent = lines[i + 1][:len(lines[i + 1]) - len(lines[i + 1].lstrip())]

                    # Check if the except already has `as name`
                    as_match = re.search(r"as\s+(\w+)", line)
                    if as_match:
                        var_name = as_match.group(1)
                        new_lines.append(line)  # Keep except line as-is
                        new_lines.append(f'{body_indent}logger.debug("Suppressed %s in {module_name}: %s", type({var_name}).__name__, {var_name})')
                    else:
                        # Add `as _exc` to the except line
                        new_except = re.sub(r":\s*$", " as _exc:", stripped)
                        new_lines.append(f"{indent}{new_except}")
                        new_lines.append(f'{body_indent}logger.debug("Suppressed %s in {module_name}: %s", type(_exc).__name__, _exc)')

                    changes += 1
                    needs_logger = True
                    i += 2  # Skip the pass line
                    continue

        new_lines.append(line)
        i += 1

    # Add logger import if needed and not present
    if changes > 0 and needs_logger and not has_logger:
        # Insert import at top of file (after any __future__ imports)
        insert_at = 0
        for idx, ln in enumerate(new_lines):
            if ln.strip().startswith("from __future__"):
                insert_at = idx + 1
            elif ln.strip() and not ln.strip().startswith("#") and not ln.strip().startswith('"""') and not ln.strip().startswith("'''"):
                break
        new_lines.insert(insert_at, "import logging")
        new_lines.insert(insert_at + 1, f'logger = logging.getLogger("{module_name}")')
        new_lines.insert(insert_at + 2, "")

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
        count = process_file(filepath, root, dry_run=dry_run)
        if count > 0:
            print(f"  {'[DRY] ' if dry_run else ''}Fixed {count:3d} silent pass blocks in {rel}")
            total += count
            changed += 1

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Total: {total} silent pass blocks fixed across {changed} files.")

    if not dry_run:
        # Compile check
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
