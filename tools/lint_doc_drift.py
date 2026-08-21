#!/usr/bin/env python3
"""Every file a document names has to be a file that exists.

`make writing` reads the prose. `make claim-constants` reads the numbers a claim
cites. Nothing read the references — the paths, the links, the `make` targets,
the line counts — and those are what a reader follows first. ARTIFACT_INDEX.md
sent every one of its twelve links to `artifacts/current/`, a directory that had
been renamed; the file had looked correct in every review since, because a dead
relative link renders as ordinary blue text.

The extraction is narrow on purpose. A path counts as a claim only where the
document writes it as a path: inside a code span, inside a fenced block, or as
the target of a relative Markdown link. Prose that happens to contain a slash is
not a reference, and treating it as one is how a linter earns the right to be
switched off. `make` is read the same way — as a command inside code, never as
the English verb.

Two numeric claims are checked because both are written as present fact and both
go stale silently: the length of a named file, and how many modules a named
directory holds.

Append-only records are out of scope. The execution tracker names files that
were real on the day of the entry; correcting it would falsify the record. See
docs/DOC_STATUS.md for which document is which.

    python tools/lint_doc_drift.py                 # every current document
    python tools/lint_doc_drift.py FILE [FILE...]  # specific files
    python tools/lint_doc_drift.py --json OUT      # machine-readable findings
    python tools/lint_doc_drift.py --write-baseline
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "config" / "doc_drift_baseline.json"

#: Records, not descriptions. Each entry is true as of the date it was written
#: and is never revised; a reference that has since moved is part of the record.
EXCLUDE_PREFIX = (
    "docs/AURA_EXECUTION_TRACKER.md",
    "docs/AURA_EXECUTION_PLAN.md",
    "docs/AURA_PROMPT_COVERAGE_AUDIT.md",
    "docs/RLC_SPARK_EXECUTION_LEDGER.md",
    "docs/evidence/",
    # Proposals. docs/DOC_STATUS.md: "Several name modules that were never
    # built; that is what a proposal is, not a broken reference."
    "scoping/",
    "archive/",
    "dev_archive/",
    "scratch/",
    "scratchpad/",
    "artifacts/",
    "aura_bench/results/",
    "dist/",
    "node_modules/",
    ".claude/",
)
EXCLUDE_DATED = re.compile(r"_20\d\d[_-]\d\d[_-]\d\d\.md$|_2026_\d\d\.md$")

LINK = re.compile(r"\[(?:[^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
CODE_SPAN = re.compile(r"`([^`\n]+)`")
MAKE_CALL = re.compile(r"\bmake\s+([a-z][a-z0-9_.-]*)")
HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.M)

#: A code span is a path claim only with a directory part and a known extension.
#: A bare `README.md` or `config.py` is far more often a name than a location.
PATH_CLAIM = re.compile(
    r"^(?:\./)?((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+"
    r"\.(?:py|md|json|ya?ml|toml|sh|txt|cfg|ini|js|ts|tsx|html|css|rs))(?::\d+)?$"
)
LINE_COUNT = re.compile(r"`([^`\n]+\.py)`\s*\((\d[\d,]*)\s+lines\)")
#: A document that cites a test is claiming that test holds the thing it just
#: said. The threat model cited `tests/test_steering_injection.py` as its proof
#: of prompt-injection defence; that file tests activation steering, a
#: different sense of the word. Whether a test proves the right claim is not
#: decidable here, but a cited file with no tests in it is, and that is the
#: state a rename leaves behind. Only files named `test_*` are read this way —
#: a Locust file under tests/ collects `@task` methods and no `def test_`.
TEST_DEF = re.compile(r"^\s*(?:async\s+)?def\s+test_", re.M)

#: A documented environment variable that nothing reads is a lever with no
#: cable attached. The operator sets it, the value is accepted, and the
#: behaviour does not move — which reads as the setting being ineffective
#: rather than absent. Only names written inside code are checked, and only
#: the AURA_ namespace, which this repository owns.
#: Three things read as an env name and are not one: a document filename
#: (`AURA_MASTER_SPEC.md`), a documented family written with a wildcard
#: (`AURA_LLM__*`), and a name a document mentions in order to say it does not
#: exist. The last is why the negation cue is here — OPERATOR_GUIDE.md warns
#: readers off `AURA_MEM_THRESHOLDS`, and flagging that sentence would ask the
#: author to delete a correction.
ENV_VAR = re.compile(r"\bAURA_[A-Z0-9_]{2,}\b(?![.*/])(?!_\*)")
ENV_NEGATED = re.compile(
    r"\b(no|not|never|does not exist|doesn't exist|removed|retired|"
    r"unused|nothing reads)\b", re.I)
MODULE_COUNT = re.compile(r"(\d[\d,]*)\s+modules? in\s+`([^`\n]+/)`")
#: The size of the test suite, which eight documents stated and all eight got
#: wrong together. config/test_inventory.json is the one recorded value;
#: `make test-inventory` refreshes it.
SUITE_SIZE = re.compile(
    r"\*{0,2}([\d,]{4,})\*{0,2}\s+tests?\b[^.\n]{0,40}?\b(?:across|in)\s+"
    r"\*{0,2}([\d,]{3,})\*{0,2}\s+(?:test\s+)?files?")
QUOTED_COUNT = re.compile(r"[\"“][^\"”\n]*\d[\d,]{3,}[^\"”\n]*[\"”]")

#: Some documents name a file precisely because it is not there.
#: docs/DOC_STATUS.md keeps the correction log and says so outright — "naming
#: an absent file is the point" — and a scoping proposal names the modules it
#: proposes building. The cue is read over the whole paragraph rather than the
#: one line, because "these were never built" and the list that follows it are
#: rarely the same line.
ABSENT_CUE = re.compile(
    r"\b(never (existed|built|shipped)|no such file|does not exist|do not exist|"
    r"is absent|are absent|was absent|as deleted|retired|removed|deleted|"
    r"not built|never had|no longer exists?|proposes? building|"
    r"until something runs|runtime-created|generated outputs?)\b", re.I)

#: A path with a placeholder in it names a shape, not a file:
#: `frames/NNNNNNNN.json` is how the ghost substrate writes frame 42.
PLACEHOLDER = re.compile(r"NNN|XXX|####|<[^>]+>|\{[^}]+\}|\*|\.\.\.|YYYY|MM_DD")

#: A path introduced as somewhere output arrives is a destination, the same as
#: a gitignored one — it is absent until the command that fills it has run.
DESTINATION_CUE = re.compile(
    r"\b(outputs? land|lands? at|written to|writes? to|will be written|"
    r"results? land|report lands)\b", re.I)
INVENTORY = ROOT / "config" / "test_inventory.json"

EXTERNAL = ("http://", "https://", "mailto:", "file:", "ftp:", "tel:")

#: A path the repository deliberately ignores is a destination, not a file that
#: should already be there. ARTIFACT_INDEX.md exists to say where `make
#: final-proof` writes its bundles; every one of those paths is absent on a
#: clean tree, and that is the correct state.
_ignored_cache: dict[str, bool] = {}


def is_output_path(rel: str) -> bool:
    """True when .gitignore claims this path — it names where output goes."""
    if rel not in _ignored_cache:
        hit = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", rel],
            cwd=ROOT, capture_output=True,
        )
        _ignored_cache[rel] = hit.returncode == 0
    return _ignored_cache[rel]


#: Naming an output path is fine; linking to one is not. A reader who clicks a
#: link expects the file, and git decides what a reader can reach. Every one of
#: ARTIFACT_INDEX.md's twelve links pointed into `artifacts/current/`, which is
#: ignored — on disk for the person who ran the proof, a 404 for everyone else.
_published_cache: dict[str, bool] = {}


def is_published(rel: str) -> bool:
    """True when git tracks this file, or any file beneath this directory."""
    if rel not in _published_cache:
        listed = subprocess.run(
            ["git", "ls-files", "--", rel],
            cwd=ROOT, capture_output=True, text=True,
        ).stdout.strip()
        _published_cache[rel] = bool(listed)
    return _published_cache[rel]


def slug(text: str) -> str:
    """GitHub's heading anchor: lowercased, punctuation dropped, spaces hyphened."""
    t = re.sub(r"[`*_\[\]()]", "", text.strip().lower())
    t = re.sub(r"[^\w\s-]", "", t)
    return re.sub(r"\s+", "-", t).strip("-")


def tracked_docs() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    return [
        p
        for p in out
        if not p.startswith(EXCLUDE_PREFIX) and not EXCLUDE_DATED.search(p)
    ]


#: Sources of an env name, in order of how they are written:
#: literally, or built from a prefix — `f"AURA_FLAG_{name.upper()}"` produces
#: every flag override without any of them appearing in the tree.
SOURCE_GLOBS = ("*.py", "*.sh", "*.yml", "*.yaml", "*.json", "*.toml",
                "Makefile", "*.plist", "*.command", "*.js", "*.ts")


def readable_env_names() -> tuple[set[str], tuple[str, ...]]:
    """Literal AURA_* names in tracked source, and the prefixes built at runtime."""
    literal = set(subprocess.run(
        ["git", "grep", "-hoE", r"AURA_[A-Z0-9_]{2,}", "--", *SOURCE_GLOBS],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout.split())
    built = tuple(sorted({
        prefix
        for prefix in subprocess.run(
            ["git", "grep", "-hoE", r"AURA_[A-Z0-9_]*\{", "--", *SOURCE_GLOBS],
            cwd=ROOT, capture_output=True, text=True,
        ).stdout.replace("{", "").split()
        # `f"AURA_{service.upper()}_..."` yields the bare namespace, which
        # would vouch for every name there is. A prefix has to narrow something.
        if prefix != "AURA_"
    }))
    return literal, built


def recorded_suite_size() -> tuple[int, int] | None:
    """(tests, files) as last recorded, or None when nothing has been."""
    if not INVENTORY.exists():
        return None
    data = json.loads(INVENTORY.read_text())
    return int(data["collected"]), int(data["files"])


def make_targets() -> set[str]:
    mk = (ROOT / "Makefile").read_text(errors="replace")
    return set(re.findall(r"^([A-Za-z0-9_.-]+):(?!=)", mk, re.M))


def anchors_of(rel: str) -> set[str]:
    try:
        txt = (ROOT / rel).read_text(errors="replace")
    except OSError:
        return set()
    return {slug(m.group(1)) for m in HEADING.finditer(txt)}


def _resolve(doc: Path, target: str) -> Path | None:
    """Repo-relative resolution of a link target, or None if it escapes the tree."""
    try:
        hit = (ROOT / doc.parent / target).resolve()
        hit.relative_to(ROOT)
    except (ValueError, OSError):
        return None
    return hit


def scan(rel: str, targets: set[str], anchor_cache: dict[str, set[str]],
         env_names: tuple[set[str], tuple[str, ...]],
         suite: tuple[int, int] | None) -> list[dict]:
    doc = Path(rel)
    try:
        lines = (ROOT / rel).read_text(errors="replace").splitlines()
    except OSError:
        return []
    if rel not in anchor_cache:
        anchor_cache[rel] = anchors_of(rel)

    found: list[dict] = []

    def note(line_no: int, kind: str, detail: str, ctx: str) -> None:
        found.append(
            {"doc": rel, "line": line_no, "kind": kind, "detail": detail,
             "context": ctx.strip()[:180]}
        )

    # Paragraph index: blank-line delimited, so a cue anywhere in the
    # paragraph covers every path named in it.
    paragraphs: list[str] = []
    para_of: list[int] = []
    current: list[str] = []
    for raw in lines:
        if raw.strip():
            current.append(raw)
        elif current:
            paragraphs.append("\n".join(current))
            current = []
        para_of.append(len(paragraphs))
    if current:
        paragraphs.append("\n".join(current))

    def names_an_absence(line_no: int) -> bool:
        idx = para_of[line_no - 1]
        if idx >= len(paragraphs):
            return False
        para = paragraphs[idx]
        return bool(ABSENT_CUE.search(para) or DESTINATION_CUE.search(para))

    in_fence = False
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue

        spans = [m.group(1) for m in CODE_SPAN.finditer(line)]

        # A lever the operator can set has to be a lever something reads.
        if not ENV_NEGATED.search(line):
            literal, built = env_names
            for scope in ([line] if in_fence else []) + spans:
                for m in ENV_VAR.finditer(scope):
                    name = m.group(0)
                    if name in literal or name.startswith(built):
                        continue
                    note(i, "env_var_has_no_reader", name, scope)

        # A path written as a path.
        for span in spans:
            claim = PATH_CLAIM.match(span.strip())
            if not claim:
                continue
            cited = claim.group(1)
            if not (ROOT / cited).exists():
                if (not is_output_path(cited)
                        and not PLACEHOLDER.search(cited)
                        and not names_an_absence(i)):
                    note(i, "missing_path", cited, line)
            elif cited.startswith("tests/") and Path(cited).name.startswith("test_"):
                body = (ROOT / cited).read_text(errors="replace")
                if not TEST_DEF.search(body):
                    note(i, "cited_test_has_no_tests", cited, line)

        # `make` as a command, never as the verb.
        for scope in ([line] if in_fence else []) + spans:
            for m in MAKE_CALL.finditer(scope):
                if m.group(1) not in targets:
                    note(i, "missing_make_target", m.group(1), scope)

        # Length of a named file, written as present fact.
        for m in LINE_COUNT.finditer(line):
            path, claimed = m.group(1), int(m.group(2).replace(",", ""))
            f = ROOT / path
            if not f.exists():
                note(i, "missing_path", path, line)
                continue
            actual = len(f.read_text(errors="replace").splitlines())
            if actual != claimed:
                note(i, "stale_line_count", f"{path}: says {claimed:,}, is {actual:,}", line)

        # How big the test suite is. A count inside quotation marks is a
        # citation of what some document said, not a claim about the tree —
        # DOC_STATUS.md's correction log quotes the numbers it replaced.
        if suite is not None and not QUOTED_COUNT.search(line):
            for m in SUITE_SIZE.finditer(line):
                tests = int(m.group(1).replace(",", ""))
                files = int(m.group(2).replace(",", ""))
                if (tests, files) != suite:
                    note(i, "stale_suite_size",
                         f"says {tests:,} tests / {files:,} files, "
                         f"recorded {suite[0]:,} / {suite[1]:,}", line)

        # How many modules a named directory holds.
        for m in MODULE_COUNT.finditer(line):
            claimed, d = int(m.group(1).replace(",", "")), m.group(2)
            target_dir = ROOT / d
            if not target_dir.is_dir():
                note(i, "missing_path", d, line)
                continue
            actual = len(list(target_dir.glob("*.py")))
            if actual != claimed:
                note(i, "stale_module_count", f"{d}: says {claimed}, is {actual}", line)

        if in_fence:
            continue

        # Links, and the headings they point at.
        for m in LINK.finditer(line):
            target = m.group(1)
            if target.startswith(EXTERNAL):
                continue
            if target.startswith("#"):
                if slug(target[1:]) not in anchor_cache[rel]:
                    note(i, "bad_anchor", target, line)
                continue
            frag = None
            if "#" in target:
                target, frag = target.split("#", 1)
            if not target:
                continue
            hit = _resolve(doc, target)
            if hit is None:
                continue
            hit_rel = str(hit.relative_to(ROOT))
            if not is_published(hit_rel):
                # The output-path exemption is for code spans only. Naming
                # where a proof writes is documentation; linking there is a
                # promise the reader can follow it, and they cannot.
                kind = "unpublished_link" if hit.exists() else "dead_link"
                note(i, kind, hit_rel, line)
            elif frag and hit.suffix == ".md":
                if hit_rel not in anchor_cache:
                    anchor_cache[hit_rel] = anchors_of(hit_rel)
                if slug(frag) not in anchor_cache[hit_rel]:
                    note(i, "bad_anchor", f"{hit_rel}#{frag}", line)

    return found


def load_baseline() -> dict[str, int]:
    if not BASELINE.exists():
        return {}
    return json.loads(BASELINE.read_text()).get("per_doc", {})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="documents to check (default: all)")
    ap.add_argument("--json", dest="json_out", help="write findings here")
    ap.add_argument("--write-baseline", action="store_true",
                    help="record the current counts (the ratchet only goes down)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    docs = args.files or tracked_docs()
    targets = make_targets()
    env_names = readable_env_names()
    suite = recorded_suite_size()
    cache: dict[str, set[str]] = {}

    findings: list[dict] = []
    for rel in docs:
        findings.extend(scan(rel, targets, cache, env_names, suite))

    per_doc: dict[str, int] = {}
    for f in findings:
        per_doc[f["doc"]] = per_doc.get(f["doc"], 0) + 1

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"findings": findings, "per_doc": per_doc}, indent=1))

    if args.write_baseline:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(
            {"per_doc": dict(sorted(per_doc.items())),
             "total": len(findings)}, indent=1) + "\n")
        print(f"baseline written: {len(findings)} findings across {len(per_doc)} documents")
        return 0

    base = load_baseline()
    regressions = []
    for doc, count in sorted(per_doc.items()):
        allowed = base.get(doc, 0)
        if count > allowed:
            regressions.append((doc, allowed, count))

    if not args.quiet:
        for f in findings:
            allowed = base.get(f["doc"], 0)
            mark = " " if allowed else "·"
            print(f"{mark} {f['doc']}:{f['line']}  {f['kind']}: {f['detail']}")

    if regressions:
        print()
        for doc, allowed, count in regressions:
            print(f"REGRESSION {doc}: {count} findings, baseline allows {allowed}")
        print(f"\n{len(regressions)} document(s) gained broken references.")
        return 1

    total_allowed = sum(base.values())
    print(f"\n{len(findings)} finding(s); baseline allows {total_allowed}.")
    if len(findings) < total_allowed:
        print("Baseline is loose — rerun with --write-baseline to tighten it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
