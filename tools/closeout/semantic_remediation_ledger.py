#!/usr/bin/env python3
"""Record and summarize remediation of hash-bound semantic-review findings.

The review inventory (`semantic_review_ledger.py`) proves that spans were
read and findings were recorded. It deliberately claims nothing about repair:
its manifest lists ``finding_remediated`` under ``claim_not_supported``, and
each finding carries only category/description/evidence_lines/finding_id/
repair_group/severity/title — there is no status field anywhere.

The consequence was that closure existed only in git commit prose. Multiple
agents worked the same campaign concurrently with no shared, machine-readable
answer to "how many findings are actually closed?", so neither Codex nor a
later session could tell remediated findings from untouched ones, and a
concurrent merge that reverted a batch (this happened twice) was invisible to
any counter.

This ledger closes that gap. It is append-only, one entry per finding, and
every entry pins the file hash AT CLOSURE so drift is detectable later:

    {finding_id, file, severity, title, status, commit, file_sha256_at_close,
     inventory_file_sha256, evidence, note, agent, recorded_at_unix}

Statuses
--------
remediated          the defect was fixed in code
assessed_no_change  verified against current source; no change warranted
                    (already fixed by later work, or the finding does not
                    hold) — ``note`` must say why
superseded          a different change removed the code path entirely
wont_fix            deliberate, documented design decision — ``note`` required

Usage
-----
    semantic_remediation_ledger.py record --finding <id> [...] \
        --status remediated --commit <sha> --evidence "tests/test_x.py" \
        --note "why"
    semantic_remediation_ledger.py status [--by-file] [--severity critical]
    semantic_remediation_ledger.py verify
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.atomic_writer import atomic_append_text  # noqa: E402

SEMANTIC_DIR = ROOT / "artifacts" / "closeout" / "semantic_review"
DEFAULT_LEDGER = SEMANTIC_DIR / "cp126" / "REMEDIATION_LEDGER.jsonl"
DEFAULT_INVENTORY = SEMANTIC_DIR / "cp126" / "inventory_through_batch0037.jsonl.gz"
REMEDIATION_ENTRY_SCHEMA = "aura.closeout.semantic_remediation_entry.v1"

VALID_STATUSES = {
    "remediated",
    "assessed_no_change",
    "superseded",
    "wont_fix",
    # The code already did the right thing and that was CHECKED against
    # current source. A distinct claim from "remediated": no change was made
    # in this pass, so collapsing it into remediated would overstate the work
    # and lose the fact that the behaviour predates it. Thirteen rows carried
    # this status and the reader could not recognise it, so thirteen closed
    # findings counted as open.
    "verified_already_remediated",
    # The finding as written describes a wider exploitable scope than the
    # source supports. Not "wont_fix" (nothing was declined) and not
    # "assessed_no_change" (the assessment CHANGED what the finding claims).
    "analyzed_scope_corrected",
    # Remediated, with the part that was NOT done named and justified in the
    # note. Distinct from "remediated" (nothing left) and from
    # "analyzed_scope_corrected" (the finding overstated the problem): here
    # the finding was right, most of it is closed, and the residue is a
    # dependency that does not exist yet — a signing chain, an external
    # control. Four rows carried this and no reader recognised it, so four
    # closed findings counted as open. The SECOND time an unrecognised
    # status silently reopened work, which is why
    # tests/test_a_status_no_reader_knows.py now fails on the first one
    # rather than waiting for someone to run `verify`.
    "remediated_scope_named",
}
# Statuses that assert the finding required no code change must explain why;
# an unexplained "assessed" is indistinguishable from a skipped finding.
STATUSES_REQUIRING_NOTE = {
    "assessed_no_change",
    "wont_fix",
    "superseded",
    "verified_already_remediated",
    "analyzed_scope_corrected",
    # The residue is the whole point of this status, so an unexplained one
    # is a closure claim with no scope attached.
    "remediated_scope_named",
}


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def load_inventory(inventory: Path) -> dict[str, dict[str, Any]]:
    """Map finding_id -> {file, severity, title, inventory_file_sha256}."""
    findings: dict[str, dict[str, Any]] = {}
    opener = gzip.open if inventory.suffix == ".gz" else open
    with opener(inventory, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for finding in record.get("findings", []):
                findings[finding["finding_id"]] = {
                    "file": record["file"],
                    "severity": finding.get("severity", "unknown"),
                    "title": finding.get("title", ""),
                    "repair_group": finding.get("repair_group", ""),
                    "inventory_file_sha256": record.get("file_sha256", ""),
                }
    return findings


def load_ledger(ledger: Path) -> dict[str, dict[str, Any]]:
    """Map finding_id -> latest entry (last write wins, history preserved)."""
    entries: dict[str, dict[str, Any]] = {}
    if not ledger.exists():
        return entries
    with ledger.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            finding_id = entry.get("finding_id")
            if finding_id:
                entries[finding_id] = entry
    return entries


def _resolve_finding_id(token: str, inventory: dict[str, dict[str, Any]]) -> str | None:
    """Accept a full id or the 8-char suffix used in commit messages."""
    if token in inventory:
        return token
    matches = [fid for fid in inventory if fid.endswith(token)]
    if len(matches) == 1:
        return matches[0]
    return None


def _resolve_commit(ref: str) -> str:
    """Return the canonical commit SHA for ``ref`` or an empty string."""

    try:
        proc = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "rev-parse", "--verify", f"{ref}^{{commit}}"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    resolved = proc.stdout.strip().lower()
    if proc.returncode != 0 or len(resolved) != 40:
        return ""
    try:
        int(resolved, 16)
    except ValueError:
        return ""
    return resolved


def _git_file_sha256(commit: str, path: str) -> str | None:
    """Hash ``path`` exactly as stored by ``commit``; None means absent."""

    try:
        proc = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "show", f"{commit}:{path}"],
            cwd=str(ROOT),
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return hashlib.sha256(proc.stdout).hexdigest()


def cmd_record(args: argparse.Namespace) -> int:
    inventory = load_inventory(Path(args.inventory))
    ledger_path = Path(args.ledger)

    if args.status not in VALID_STATUSES:
        print(f"error: --status must be one of {sorted(VALID_STATUSES)}", file=sys.stderr)
        return 2
    if args.status in STATUSES_REQUIRING_NOTE and not args.note.strip():
        print(
            f"error: --note is required for status '{args.status}' "
            "(an unexplained assessment is indistinguishable from a skip)",
            file=sys.stderr,
        )
        return 2

    commit_ref = str(args.commit or "HEAD").strip()
    commit = _resolve_commit(commit_ref)
    if not commit:
        print(
            f"error: --commit does not resolve to a git commit: {commit_ref!r}",
            file=sys.stderr,
        )
        return 2

    resolved_findings: list[tuple[str, dict[str, Any]]] = []
    unknown: list[str] = []
    for token in args.finding:
        finding_id = _resolve_finding_id(token, inventory)
        if finding_id is None:
            unknown.append(token)
        else:
            resolved_findings.append((finding_id, inventory[finding_id]))
    if unknown:
        print(
            f"error: {len(unknown)} unknown finding id(s): {', '.join(unknown)}",
            file=sys.stderr,
        )
        return 2

    file_hashes: dict[str, str] = {}
    for _finding_id, meta in resolved_findings:
        path = str(meta["file"])
        if path in file_hashes:
            continue
        current_hash = _sha256_file(ROOT / path)
        committed_hash = _git_file_sha256(commit, path) or ""
        if current_hash != committed_hash:
            print(
                "error: finding source is not the source stored by closure commit: "
                f"{path} (current={current_hash or 'absent'}, "
                f"commit={committed_hash or 'absent'}, commit_sha={commit})",
                file=sys.stderr,
            )
            return 2
        file_hashes[path] = current_hash

    now = time.time()
    entries: list[str] = []
    for finding_id, meta in resolved_findings:
        entry = {
            "schema": REMEDIATION_ENTRY_SCHEMA,
            "finding_id": finding_id,
            "file": meta["file"],
            "severity": meta["severity"],
            "title": meta["title"],
            "repair_group": meta["repair_group"],
            "status": args.status,
            "commit": commit,
            "agent": args.agent,
            "evidence": list(args.evidence),
            "note": args.note.strip(),
            # Hash at closure: if the file later changes, a verify pass can
            # tell that the fix may no longer be present (a concurrent
            # merge reverted two whole batches during this campaign).
            "file_sha256_at_close": file_hashes[str(meta["file"])],
            "inventory_file_sha256": meta["inventory_file_sha256"],
            "recorded_at_unix": now,
            "claim_supported": "finding_remediation_recorded_by_agent",
            "claim_not_supported": [
                "independent_verification",
                "full_closeout_complete",
            ],
        }
        entries.append(json.dumps(entry, sort_keys=True) + "\n")

    if entries:
        atomic_append_text(ledger_path, "".join(entries))
    print(f"recorded {len(entries)} finding(s) as {args.status}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    inventory = load_inventory(Path(args.inventory))
    ledger = load_ledger(Path(args.ledger))

    total = len(inventory)
    closed_ids = {fid for fid, e in ledger.items() if e.get("status") in VALID_STATUSES}
    open_ids = set(inventory) - closed_ids

    sev_total = Counter(m["severity"] for m in inventory.values())
    sev_closed = Counter(inventory[f]["severity"] for f in closed_ids if f in inventory)
    # A row with no status is a real condition — a hand-edited or
    # partially-written entry — and sorting None against str raises. Naming
    # it keeps the malformed row visible instead of crashing the report that
    # would have shown it.
    status_counts = Counter(
        str(e.get("status") or "(missing status)") for e in ledger.values()
    )

    print("Semantic remediation status")
    print("=" * 58)
    print(f"  findings in inventory : {total}")
    print(f"  recorded closed       : {len(closed_ids)}  ({100.0 * len(closed_ids) / max(1, total):.1f}%)")
    print(f"  remaining open        : {len(open_ids)}")
    print()
    print("  by severity:")
    for sev in ("critical", "high", "medium", "low", "info"):
        if sev_total.get(sev):
            print(
                f"    {sev:<9} {sev_closed.get(sev, 0):>5} / {sev_total[sev]:<5} closed"
                f"   ({sev_total[sev] - sev_closed.get(sev, 0)} open)"
            )
    print()
    print("  by closure status:")
    for status, count in sorted(status_counts.items()):
        print(f"    {status:<20} {count}")

    # A row whose status is missing or unrecognised is silently excluded from
    # closed_ids above. Eleven such rows once sat in this ledger carrying a
    # commit and a passing test each — genuinely fixed findings, reported as
    # open, for three days. Dropping a malformed row without saying so is the
    # absence of a check reported as a passed check: the report looked
    # complete precisely because the rows it could not read were invisible.
    malformed = [
        fid
        for fid, entry in ledger.items()
        if entry.get("status") not in VALID_STATUSES
    ]
    if malformed:
        print()
        print(
            f"  ⚠️  {len(malformed)} ledger row(s) carry no recognised status and are "
            "NOT counted as closed:"
        )
        for fid in sorted(malformed)[:20]:
            print(f"      {fid}")
        if len(malformed) > 20:
            print(f"      … and {len(malformed) - 20} more")
        print(
            "  Re-record them with `record --finding <id> --status <status>`; "
            "the counts above understate progress until you do."
        )

    if args.by_file:
        per_file_total: Counter = Counter()
        per_file_closed: Counter = Counter()
        for fid, meta in inventory.items():
            per_file_total[meta["file"]] += 1
            if fid in closed_ids:
                per_file_closed[meta["file"]] += 1
        print()
        print("  top open files:")
        ranked = sorted(
            per_file_total.items(),
            key=lambda kv: -(kv[1] - per_file_closed.get(kv[0], 0)),
        )
        for path, count in ranked[: args.limit]:
            remaining = count - per_file_closed.get(path, 0)
            if remaining <= 0:
                continue
            print(f"    {remaining:>4} open / {count:<4} total  {path}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Detect ledger entries whose file changed since closure (possible revert)."""
    inventory = load_inventory(Path(args.inventory))
    ledger = load_ledger(Path(args.ledger))

    drifted: list[tuple[str, str]] = []
    orphaned: list[str] = []
    unreadable: list[tuple[str, str]] = []
    for finding_id, entry in ledger.items():
        status = str(entry.get("status") or "")
        if status not in VALID_STATUSES:
            unreadable.append((finding_id, status or "<missing>"))
        if finding_id not in inventory:
            orphaned.append(finding_id)
            continue
        current = _sha256_file(ROOT / entry["file"])
        recorded = entry.get("file_sha256_at_close", "")
        if recorded and current and current != recorded:
            drifted.append((finding_id, entry["file"]))

    print(f"ledger entries      : {len(ledger)}")
    print(f"orphaned (unknown id): {len(orphaned)}")
    print(f"unreadable status    : {len(unreadable)}")
    print(f"files changed since closure: {len({f for _, f in drifted})}")
    if drifted and args.show:
        print()
        print("  changed since their fix was recorded (re-verify these):")
        for path in sorted({f for _, f in drifted}):
            count = sum(1 for _, p in drifted if p == path)
            print(f"    {count:>4} finding(s)  {path}")
    if unreadable:
        print()
        print("  rows whose status no reader recognises — these count as OPEN:")
        for finding_id, status in sorted(unreadable):
            print(f"    {finding_id}  status={status}")

    # Drift is expected during an active campaign (later batches touch the
    # same file), so it is reported, not failed.
    #
    # An unreadable status is not. `status` printed a warning about those and
    # exited zero, so a row carrying a commit and a passing test could sit
    # uncounted indefinitely — which is what `remediated_scope_named` did.
    # Warning about a defect and then reporting success is the same shape as
    # the findings this campaign exists to close.
    return 1 if (orphaned or unreadable) else 0


def _cp126_commits_by_file() -> dict[str, list[str]]:
    """Files touched by commits that describe CP126 remediation work.

    Multiple agents worked this campaign concurrently and none of them cited
    finding ids in commit messages, so a file's remediation history can only
    be recovered from the commit log. This attributes WORK to files; it does
    NOT claim any specific finding was closed.
    """
    try:
        proc = subprocess.run(
            [
                "git", "-c", "core.fsmonitor=false", "log", "--name-only", "--no-merges",
                "--format=%x00%h %s",
                "--grep=CP126", "--grep=semantic findings", "--grep=semantic review",
                "-i", "--all",
            ],
            cwd=str(ROOT), capture_output=True, text=True, timeout=120, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}

    by_file: dict[str, list[str]] = defaultdict(list)
    current = ""
    for raw in proc.stdout.splitlines():
        if raw.startswith("\x00"):
            current = raw[1:].strip()
            continue
        path = raw.strip()
        if path and current and path not in ("",):
            if current not in by_file[path]:
                by_file[path].append(current)
    return dict(by_file)


def cmd_triage(args: argparse.Namespace) -> int:
    """Honest three-state view of the campaign.

    Distinguishes findings that are provably untouched from those in files
    that HAVE been worked, without fabricating closure for either.
    """
    inventory = load_inventory(Path(args.inventory))
    ledger = load_ledger(Path(args.ledger))
    recorded = {fid for fid, e in ledger.items() if e.get("status") in VALID_STATUSES}
    worked = _cp126_commits_by_file()

    # file -> (inventory hash, findings)
    per_file: dict[str, list[str]] = defaultdict(list)
    file_hash: dict[str, str] = {}
    for fid, meta in inventory.items():
        per_file[meta["file"]].append(fid)
        file_hash[meta["file"]] = meta["inventory_file_sha256"]

    buckets: dict[str, list[str]] = {
        "recorded_closed": [],
        "untouched": [],
        "worked_unverified": [],
        "changed_unattributed": [],
        "file_missing": [],
    }
    for path, fids in per_file.items():
        if path.startswith(("archive/", "artifacts/")):
            continue
        target = ROOT / path
        current = _sha256_file(target)
        for fid in fids:
            if fid in recorded:
                buckets["recorded_closed"].append(fid)
            elif not current:
                buckets["file_missing"].append(fid)
            elif current == file_hash[path]:
                buckets["untouched"].append(fid)
            elif path in worked:
                buckets["worked_unverified"].append(fid)
            else:
                buckets["changed_unattributed"].append(fid)

    total = sum(len(v) for v in buckets.values())
    print("CP126 remediation triage")
    print("=" * 66)
    print(f"  {'recorded closed (in ledger)':<38} {len(buckets['recorded_closed']):>6}")
    print(f"  {'UNTOUCHED file (definitely open)':<38} {len(buckets['untouched']):>6}")
    print(f"  {'file worked by a CP126 commit':<38} {len(buckets['worked_unverified']):>6}  <- verify per finding")
    print(f"  {'file changed, no CP126 commit':<38} {len(buckets['changed_unattributed']):>6}  <- verify per finding")
    print(f"  {'file no longer present':<38} {len(buckets['file_missing']):>6}")
    print(f"  {'-' * 46}")
    print(f"  {'total (excl. archive/artifacts)':<38} {total:>6}")
    print()
    print("  Only 'recorded closed' is a claim. The two 'verify' buckets are")
    print("  work other agents may already have done — they are NOT counted as")
    print("  closed until someone checks the finding against current source.")

    if args.queue:
        print()
        print(f"  next files by UNTOUCHED findings (guaranteed-real work), top {args.limit}:")
        untouched_by_file: Counter = Counter()
        crit_by_file: Counter = Counter()
        for fid in buckets["untouched"]:
            meta = inventory[fid]
            untouched_by_file[meta["file"]] += 1
            if meta["severity"] == "critical":
                crit_by_file[meta["file"]] += 1
        for path, count in untouched_by_file.most_common(args.limit):
            print(f"    {count:>4} open ({crit_by_file.get(path, 0)} crit)  {path}")
    return 0


REVIEW_COMMIT = "b7bc66f2c87e6a24cd7b8280db6912714b5700c1"


def _file_at_commit(path: str, commit: str) -> list[str] | None:
    try:
        proc = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "show", f"{commit}:{path}"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.splitlines()


def _normalize(line: str) -> str:
    return " ".join(line.split())


def _has_contiguous_run(current_lines: list[str], cited: list[str]) -> bool:
    """True when the cited lines still appear as a consecutive run."""
    if not cited:
        return False
    if len(cited) == 1:
        return cited[0] in current_lines
    first = cited[0]
    for index, line in enumerate(current_lines):
        if line != first:
            continue
        window = current_lines[index : index + len(cited)]
        if window == cited:
            return True
    return False


def cmd_sweep(args: argparse.Namespace) -> int:
    """Per-finding evidence check against current source.

    For every finding in a file that has CHANGED since the review, pull the
    exact lines the reviewer cited from the review commit and look for them
    in the current file. This does not prove a fix is correct, but it
    separates two very different situations that file-level triage cannot:

      still_present    the cited defect lines survive verbatim -> OPEN
      span_changed     the cited lines are gone -> plausibly addressed,
                       needs a human/agent read before it can be recorded

    Findings whose file is byte-identical to the review are reported as
    untouched without any git work.
    """
    inventory_path = Path(args.inventory)
    ledger = load_ledger(Path(args.ledger))
    recorded = {f for f, e in ledger.items() if e.get("status") in VALID_STATUSES}

    # Re-read the inventory with evidence_lines, which load_inventory drops.
    per_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    file_hash: dict[str, str] = {}
    opener = gzip.open if inventory_path.suffix == ".gz" else open
    with opener(inventory_path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            path = record["file"]
            file_hash[path] = record.get("file_sha256", "")
            for finding in record.get("findings", []):
                per_file[path].append(
                    {
                        "finding_id": finding["finding_id"],
                        "severity": finding.get("severity", "unknown"),
                        "title": finding.get("title", ""),
                        "evidence_lines": finding.get("evidence_lines", []),
                    }
                )

    counts: Counter = Counter()
    still_present: list[tuple[str, str, str, str]] = []
    weak_present: list[tuple[str, str, str, str]] = []
    span_changed: list[tuple[str, str, str, str]] = []

    targets = sorted(per_file)
    if args.file:
        targets = [t for t in targets if t in set(args.file)]

    for path in targets:
        if path.startswith(("archive/", "artifacts/")):
            continue
        target = ROOT / path
        current_hash = _sha256_file(target)
        if not current_hash:
            counts["file_missing"] += len(per_file[path])
            continue
        if current_hash == file_hash[path]:
            counts["untouched"] += len(per_file[path])
            continue

        original = _file_at_commit(path, args.review_commit)
        if original is None:
            counts["no_review_blob"] += len(per_file[path])
            continue
        try:
            current_text = target.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            counts["file_missing"] += len(per_file[path])
            continue
        current_lines_norm = [_normalize(line) for line in current_text.splitlines()]
        current_norm = set(current_lines_norm)

        for finding in per_file[path]:
            fid = finding["finding_id"]
            if fid in recorded:
                counts["recorded_closed"] += 1
                continue
            lines = [n for n in finding["evidence_lines"] if isinstance(n, int)]
            # Sample the cited lines; substantive ones only (skip blanks and
            # bare punctuation, which match everywhere).
            cited: list[str] = []
            for number in lines[: args.max_lines]:
                if 1 <= number <= len(original):
                    text = _normalize(original[number - 1])
                    if len(text) >= args.min_line_chars:
                        cited.append(text)
            if not cited:
                counts["no_usable_evidence"] += 1
                continue
            # Contiguity matters far more than membership. A fix usually
            # leaves structural lines ("def foo(", a common guard) intact, so
            # counting individual survivors massively over-reports "open".
            # Require the cited lines to survive AS A CONSECUTIVE RUN in the
            # current file: that is what "this exact code is unchanged" means.
            surviving = sum(1 for text in cited if text in current_norm)
            ratio = surviving / len(cited)
            if ratio >= args.present_ratio and len(cited) >= 2:
                if not _has_contiguous_run(current_lines_norm, cited):
                    ratio = 0.0
            row = (path, fid, finding["severity"], finding["title"])
            if ratio >= args.present_ratio:
                # CONFIDENCE MATTERS. With only ONE usable evidence line the
                # contiguity test degrades to bare membership, and a single
                # surviving line proves nothing: a fix frequently ADDS a guard
                # around code the reviewer cited, leaving that line intact.
                # Two findings in actuator_registry were confirmed fixed by
                # reading source despite landing here, both single-line. Only
                # multi-line contiguous survivals are treated as strong.
                if len(cited) >= 2:
                    counts["still_present_strong"] += 1
                    still_present.append(row)
                else:
                    counts["still_present_weak"] += 1
                    weak_present.append(row)
            else:
                counts["span_changed"] += 1
                span_changed.append(row)

    print("CP126 per-finding sweep")
    print("=" * 70)
    for key in (
        "recorded_closed", "untouched", "still_present_strong",
        "still_present_weak", "span_changed",
        "no_usable_evidence", "no_review_blob", "file_missing",
    ):
        if counts.get(key):
            print(f"  {key:<20} {counts[key]:>6}")
    print()
    print("  still_present_strong = 2+ cited lines survive as a contiguous run")
    print("                        -> high confidence OPEN")
    print("  still_present_weak   = only ONE usable cited line survived; this is")
    print("                        membership, not evidence. Measured ~66% of this")
    print("                        bucket and confirmed false positives by reading")
    print("                        source -> must be read, never assumed open")
    print("  span_changed         = cited block gone -> plausibly addressed;")
    print("                        validated 6/6 against known-fixed findings")

    if args.list_changed:
        print()
        print(f"  span_changed criticals (top {args.limit}):")
        for path, fid, _sev, title in [r for r in span_changed if r[2] == "critical"][: args.limit]:
            print(f"    {fid[-8:]}  {path}")
            print(f"              {title[:78]}")
    if args.list_present:
        print()
        print(f"  still_present criticals (top {args.limit}):")
        for path, fid, _sev, title in [r for r in still_present if r[2] == "critical"][: args.limit]:
            print(f"    {fid[-8:]}  {path}")
            print(f"              {title[:78]}")
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "still_present_strong": [list(r) for r in still_present],
                    "still_present_weak": [list(r) for r in weak_present],
                    "span_changed": [list(r) for r in span_changed],
                    "counts": dict(counts),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n  wrote {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    common_inventory = os.environ.get("AURA_SEMANTIC_INVENTORY", str(DEFAULT_INVENTORY))
    common_ledger = os.environ.get("AURA_SEMANTIC_REMEDIATION_LEDGER", str(DEFAULT_LEDGER))

    record = sub.add_parser("record", help="Append remediation receipts for findings.")
    record.add_argument("--finding", action="append", required=True,
                        help="Finding id (full, or the 8-char suffix used in commits).")
    record.add_argument("--status", required=True, help=f"One of {sorted(VALID_STATUSES)}")
    record.add_argument("--commit", default="", help="Commit sha (defaults to HEAD).")
    record.add_argument("--evidence", action="append", default=[],
                        help="Test file or proof reference; repeatable.")
    record.add_argument("--note", default="", help="Why (required for non-remediated statuses).")
    record.add_argument("--agent", default=os.environ.get("AURA_AGENT", "unknown"))
    record.add_argument("--ledger", default=common_ledger)
    record.add_argument("--inventory", default=common_inventory)
    record.set_defaults(func=cmd_record)

    status = sub.add_parser("status", help="Summarize remediation progress.")
    status.add_argument("--by-file", action="store_true")
    status.add_argument("--limit", type=int, default=25)
    status.add_argument("--ledger", default=common_ledger)
    status.add_argument("--inventory", default=common_inventory)
    status.set_defaults(func=cmd_status)

    verify = sub.add_parser("verify", help="Detect drift/reverts since closure.")
    verify.add_argument("--show", action="store_true")
    verify.add_argument("--ledger", default=common_ledger)
    verify.add_argument("--inventory", default=common_inventory)
    verify.set_defaults(func=cmd_verify)

    triage = sub.add_parser(
        "triage",
        help="Three-state view: recorded closed / untouched / needs verification.",
    )
    triage.add_argument("--queue", action="store_true", help="List next files to work.")
    triage.add_argument("--limit", type=int, default=20)
    triage.add_argument("--ledger", default=common_ledger)
    triage.add_argument("--inventory", default=common_inventory)
    triage.set_defaults(func=cmd_triage)

    sweep = sub.add_parser("sweep", help="Per-finding evidence check vs current source.")
    sweep.add_argument("--file", action="append", default=[], help="Limit to these files.")
    sweep.add_argument("--review-commit", default=REVIEW_COMMIT)
    sweep.add_argument("--max-lines", type=int, default=6)
    sweep.add_argument("--min-line-chars", type=int, default=12)
    sweep.add_argument("--present-ratio", type=float, default=0.5)
    sweep.add_argument("--list-changed", action="store_true")
    sweep.add_argument("--list-present", action="store_true")
    sweep.add_argument("--limit", type=int, default=25)
    sweep.add_argument("--out", default="")
    sweep.add_argument("--ledger", default=common_ledger)
    sweep.add_argument("--inventory", default=common_inventory)
    sweep.set_defaults(func=cmd_sweep)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
