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

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.subprocess_gateway import get_subprocess_gateway

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


def find_logger_insert_index(lines: list[str]) -> int:
    """Insert after shebang/comments, module docstring, and future imports."""
    source = "\n".join(lines)
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1

    try:
        module = ast.parse(source)
    except SyntaxError:
        module = None

    if module and module.body:
        first = module.body[0]
        if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
            insert_at = max(insert_at, int(getattr(first, "end_lineno", insert_at) or insert_at))

    while insert_at < len(lines):
        stripped = lines[insert_at].strip()
        if not stripped or stripped.startswith("#"):
            insert_at += 1
            continue
        if stripped.startswith("from __future__"):
            insert_at += 1
            continue
        break
    return insert_at


def has_module_logger(content: str) -> bool:
    return bool(re.search(r"^\s*logger\s*=", content, re.MULTILINE))


def has_logging_import(content: str) -> bool:
    return bool(re.search(r"^\s*import\s+logging\b", content, re.MULTILINE))


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
    has_logger = has_module_logger(content)
    has_logging = has_logging_import(content)
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
        insert_at = find_logger_insert_index(new_lines)
        injected = [f'logger = logging.getLogger("{module_name}")', ""]
        if not has_logging:
            injected.insert(0, "import logging")
        for offset, line_to_insert in enumerate(injected):
            new_lines.insert(insert_at + offset, line_to_insert)

    if changes > 0 and not dry_run:
        filepath.write_text("\n".join(new_lines), encoding="utf-8")

    return changes


def main():
    dry_run = "--dry-run" in sys.argv
    root = ROOT

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
        python = str(root / ".venv" / "bin" / "python")
        failures = 0
        for filepath in all_files:
            result = get_subprocess_gateway().run(
                [python, "-m", "py_compile", str(filepath)],
                cwd=root,
                timeout=30,
                read_only=True,
                source="fix_silent_pass:py_compile",
                accelerator_capability="none",
            )
            if result.returncode != 0:
                print(f"  COMPILE FAIL: {filepath.relative_to(root)}")
                print(f"     {result.stderr.strip()}")
                failures += 1

        if failures:
            print(f"\n {failures} files failed compilation.")
        else:
            print(f"\nAll {len(all_files)} files compile clean.")


if __name__ == "__main__":
    main()
