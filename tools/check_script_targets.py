#!/usr/bin/env python3
"""tools/check_script_targets.py — shell scripts may not name paths that are gone.

`scripts/run_audit_suite.sh quick` invoked `crucible_test.py` for months after
commit 494cb0a4b deleted that file. The script runs under `set -euo pipefail`
and pytest exits 4 on an unrecognised path, so `quick` aborted before the first
test — while TESTING.md, README.md and docs/AURA_TEST_COMMANDS.md all named the
script as the validation entrypoint. Nobody noticed, because a broken
entrypoint and a passing one both print nothing anyone reads.

A deleted file is easy to grep for. The reason it survived is that no gate ever
looked, so this one does: every repo-relative path a tracked shell script names
as a literal must exist on disk.

What counts as a path here is deliberately narrow. Only literals that start
with a known source directory (`tests/`, `core/`, `tools/`, `scripts/`, ...) or
that end in a source extension are checked, because a shell script is full of
strings that look like paths and are not — URLs, glob patterns, `$VAR`
expansions, remote paths on a deploy host. Anything containing a shell
metacharacter or a variable reference is skipped rather than guessed at: this
gate is worth having only if a green run means something, and a gate that
guesses produces suppressions instead of fixes.

    python tools/check_script_targets.py           # check
    python tools/check_script_targets.py --list    # show every path it resolved
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: A literal is treated as a repo path when it starts with one of these.
SOURCE_ROOTS = (
    "core/",
    "tests/",
    "tools/",
    "scripts/",
    "config/",
    "interface/",
    "docs/",
)

#: ...or when it ends with one of these and contains a separator.
SOURCE_SUFFIXES = (".py", ".sh", ".json", ".toml", ".cfg", ".ini")

#: Shell constructs that make a literal unresolvable without running the shell.
DYNAMIC = re.compile(r"[$*?{}\[\]()<>|&;`~!]|\\\s")

#: Candidate path-ish tokens: any run of path characters. Requiring a separator
#: here would miss the defect that motivated the gate — `crucible_test.py` is a
#: bare filename. `looks_like_repo_path` does the narrowing instead.
TOKEN = re.compile(r"[A-Za-z0-9_./-]+")

#: A match preceded by one of these is the tail of a larger expression —
#: `"${ROOT}/scripts/x.py"` contains the literal-looking `/scripts/x.py`, whose
#: existence says nothing because `$ROOT` is not this repo. Checking the
#: character before the match is what distinguishes the two; checking only the
#: matched text cannot, because the `$` and `{` fall outside it.
INTERPOLATED_PREFIX = frozenset("}$)_-0123456789")

#: `<<EOF`, `<<-'EOF'`, `<<"EOF"` — the body that follows is printed text.
HEREDOC = re.compile(r"<<-?\s*['\"]?(?P<tag>[A-Za-z_][A-Za-z0-9_]*)['\"]?\s*$")

#: Path prefixes describing a *remote or transient* filesystem, not this repo.
SKIP_SCRIPT_PREFIXES = (
    "cloud/",
    "scratchpad/",
)
SKIP_SCRIPTS = frozenset({
    "scripts/deploy_hetzner.sh",
    "scripts/install_service.sh",
    "scripts/setup_arm64.sh",
})


@dataclass(frozen=True)
class Missing:
    script: str
    line: int
    path: str

    def __str__(self) -> str:
        return f"{self.script}:{self.line}: names '{self.path}', which does not exist"


def tracked_shell_scripts() -> list[Path]:
    """Every tracked *.sh, from git so untracked scratch files are ignored."""
    out = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "ls-files", "*.sh"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [
        PROJECT_ROOT / rel
        for rel in out.splitlines()
        if rel
        and rel not in SKIP_SCRIPTS
        and not rel.startswith(SKIP_SCRIPT_PREFIXES)
    ]


def looks_like_repo_path(token: str) -> bool:
    if DYNAMIC.search(token):
        return False
    if token.startswith(("http:", "https:", "//", "/")):
        return False
    if token.startswith(SOURCE_ROOTS):
        return True
    return token.endswith(SOURCE_SUFFIXES)


def strip_comment(line: str) -> str:
    """Drop a trailing `#` comment when the `#` is not inside a word."""
    idx = line.find(" #")
    return line if idx < 0 else line[:idx]


def is_printed_text(line: str) -> bool:
    """Whether the line only prints its argument.

    A path inside `echo`/`printf`/`cat <<EOF` is prose about a command, not an
    operand the shell will open, so its absence breaks nothing at runtime.
    scripts/fine_tune_persona.sh prints an upstream HuggingFace `run_clm.py`
    invocation it never runs. Excluding printed text keeps the gate's failures
    real; without this the only way to get to green would be an allowlist, and
    allowlists are where genuine findings go to be forgotten.
    """
    head = line.split(None, 1)[0] if line.split() else ""
    return head in {"echo", "printf", "cat"}


def scan(script: Path) -> tuple[list[Missing], list[str]]:
    rel_script = script.relative_to(PROJECT_ROOT).as_posix()
    missing: list[Missing] = []
    resolved: list[str] = []
    text = script.read_text(encoding="utf-8", errors="replace")
    heredoc_terminator: str | None = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        # A heredoc body is printed text whose lines carry no `cat` prefix, so
        # `is_printed_text` cannot see it; track the terminator instead.
        if heredoc_terminator is not None:
            if raw.strip() == heredoc_terminator:
                heredoc_terminator = None
            continue
        opener = HEREDOC.search(raw)
        if opener:
            heredoc_terminator = opener.group("tag")
            continue

        line = strip_comment(raw).strip()
        if not line or line.startswith("#") or is_printed_text(line):
            continue
        for match in TOKEN.finditer(line):
            start = match.start()
            if start > 0 and line[start - 1] in INTERPOLATED_PREFIX:
                continue
            candidate = match.group(0).strip("'\"").rstrip(":,")
            if not looks_like_repo_path(candidate):
                continue
            resolved.append(f"{rel_script}:{lineno}: {candidate}")
            if not (PROJECT_ROOT / candidate).exists():
                missing.append(Missing(rel_script, lineno, candidate))
    return missing, resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="print every path this gate resolved"
    )
    args = parser.parse_args()

    all_missing: list[Missing] = []
    all_resolved: list[str] = []
    scripts = tracked_shell_scripts()
    for script in scripts:
        missing, resolved = scan(script)
        all_missing.extend(missing)
        all_resolved.extend(resolved)

    if args.list:
        for entry in all_resolved:
            print(entry)

    if all_missing:
        print(
            f"❌ {len(all_missing)} path(s) named by shell scripts do not exist:",
            file=sys.stderr,
        )
        for item in all_missing:
            print(f"   {item}", file=sys.stderr)
        return 1

    print(
        f"✅ {len(all_resolved)} repo path(s) across {len(scripts)} shell script(s) exist"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
