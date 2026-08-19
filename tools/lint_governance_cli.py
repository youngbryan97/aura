"""CLI, baseline persistence, and scoped test helpers for governance lint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _scanner() -> Any:
    scanner = sys.modules.get("tools.lint_governance")
    if scanner is None:
        raise RuntimeError("governance scanner facade is not loaded")
    return scanner


def _baseline_payload(buckets: Sequence[Any]) -> dict[str, Any]:
    rows = [asdict(bucket) for bucket in buckets]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": _scanner().BASELINE_SCHEMA_VERSION,
        "description": (
            "Exact AST effect-ownership debt ratchet; refresh only after reviewed change"
        ),
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "buckets": rows,
    }


def write_baseline(path: Path, buckets: Sequence[Any]) -> None:
    payload = _baseline_payload(buckets)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_baseline(path: Path) -> list[Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"effect ownership baseline is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"effect ownership baseline is unreadable: {path}: {exc}"
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != _scanner().BASELINE_SCHEMA_VERSION
    ):
        raise ValueError("effect ownership baseline schema_version is invalid")
    raw_rows = payload.get("buckets")
    if not isinstance(raw_rows, list):
        raise ValueError("effect ownership baseline buckets must be a list")
    bucket_type = _scanner().EffectBucket
    buckets: list[Any] = []
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"effect ownership baseline bucket {index} is not a mapping"
            )
        try:
            buckets.append(
                bucket_type(
                    category=str(row["category"]),
                    path=str(row["path"]),
                    scope=str(row["scope"]),
                    callee=str(row["callee"]),
                    count=int(row["count"]),
                    canonical_owner=bool(row["canonical_owner"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"effect ownership baseline bucket {index} is invalid: {exc}"
            ) from exc
    expected = str(payload.get("inventory_sha256") or "")
    actual = _baseline_payload(sorted(buckets))["inventory_sha256"]
    if expected != actual:
        raise ValueError(
            "effect ownership baseline inventory_sha256 does not match its buckets"
        )
    return sorted(buckets)


def compare_inventory(
    current: Sequence[Any],
    baseline: Sequence[Any],
) -> tuple[list[str], list[str]]:
    current_by_key = {bucket.key(): bucket for bucket in current}
    baseline_by_key = {bucket.key(): bucket for bucket in baseline}
    regressions: list[str] = []
    stale: list[str] = []
    for key in sorted(set(current_by_key) | set(baseline_by_key)):
        observed = current_by_key.get(key)
        expected = baseline_by_key.get(key)
        label = " | ".join(key)
        if expected is None and observed is not None:
            regressions.append(f"NEW {label} count={observed.count}")
        elif observed is None and expected is not None:
            stale.append(f"REMOVED {label} baseline={expected.count}")
        elif observed is not None and expected is not None:
            if observed.count > expected.count:
                regressions.append(
                    f"INCREASED {label} baseline={expected.count} "
                    f"current={observed.count}"
                )
            elif observed.count < expected.count:
                stale.append(
                    f"DECREASED {label} baseline={expected.count} "
                    f"current={observed.count}"
                )
            if expected.canonical_owner and not observed.canonical_owner:
                regressions.append(f"OWNER_DEMOTED {label} baseline=True current=False")
            elif observed.canonical_owner and not expected.canonical_owner:
                stale.append(f"OWNER_PROMOTED {label} baseline=False current=True")
    return regressions, stale


def _summary(buckets: Sequence[Any]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for bucket in buckets:
        row = summary.setdefault(
            bucket.category,
            {"calls": 0, "buckets": 0, "debt_calls": 0},
        )
        row["calls"] += bucket.count
        row["buckets"] += 1
        if not bucket.canonical_owner:
            row["debt_calls"] += bucket.count
    return summary


def _parser() -> argparse.ArgumentParser:
    scanner = _scanner()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=scanner.ROOT)
    parser.add_argument("--baseline", type=Path, default=scanner.DEFAULT_BASELINE)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Replace the baseline with the current reviewed inventory",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: Iterable[str] = ()) -> int:
    scanner = _scanner()
    args = _parser().parse_args(list(argv))
    root = args.root.expanduser().resolve()
    baseline_path = args.baseline.expanduser()
    if not baseline_path.is_absolute():
        baseline_path = root / baseline_path

    buckets, problems = scanner.scan_repository(root)
    problems.extend(scanner.audit_subprocess_accelerator_declarations(root))
    report: dict[str, Any] = {
        "ok": False,
        "root": str(root),
        "baseline": str(baseline_path),
        "summary": _summary(buckets),
        "bucket_count": len(buckets),
        "problems": [asdict(problem) for problem in problems],
        "regressions": [],
        "stale_baseline": [],
    }
    if problems:
        print(f"governance effect ownership: analyzer failed on {len(problems)} file(s)")
        for problem in problems[:40]:
            print(f"  {problem.path}: {problem.problem}")
        _write_report(args.json_out, report)
        return 2

    if args.write_baseline:
        write_baseline(baseline_path, buckets)
        report["ok"] = True
        report["baseline_written"] = True
        _write_report(args.json_out, report)
        debt_calls = sum(bucket.count for bucket in buckets if not bucket.canonical_owner)
        print(
            "governance effect ownership baseline written: "
            f"{len(buckets)} buckets, {debt_calls} migration-debt calls"
        )
        return 0

    try:
        baseline = load_baseline(baseline_path)
    except ValueError as exc:
        print(f"governance effect ownership: configuration error: {exc}")
        _write_report(args.json_out, report)
        return 2

    regressions, stale = compare_inventory(buckets, baseline)
    report["regressions"] = regressions
    report["stale_baseline"] = stale
    report["ok"] = not regressions and not stale
    _write_report(args.json_out, report)

    summary = _summary(buckets)
    debt_calls = sum(row["debt_calls"] for row in summary.values())
    total_calls = sum(row["calls"] for row in summary.values())
    if regressions or stale:
        print(
            "governance effect ownership: baseline drift "
            f"({len(regressions)} regression(s), {len(stale)} stale bucket(s))"
        )
        for issue in (regressions + stale)[:80]:
            print(f"  {issue}")
        print(
            "Review the call-site changes. After debt reductions or approved "
            "canonical-owner changes, refresh with "
            "tools/lint_governance.py --write-baseline."
        )
        return 1

    governed = sum(
        row["debt_calls"]
        for category, row in summary.items()
        if category in GATEWAY_CATEGORIES
    )
    raw = debt_calls - governed
    report["debt_governed_calls"] = governed
    report["debt_raw_calls"] = raw
    _write_report(args.json_out, report)

    print(
        "governance effect ownership: baseline matched; "
        f"{total_calls} recognized calls in {len(buckets)} buckets, "
        f"{debt_calls} calls remain migration debt"
    )
    # One undifferentiated debt number invites the reading that every one of
    # them is an exploitable bypass, and the two halves are not the same claim.
    # A call routed through the subprocess or file-write gateway is following
    # the convention CLAUDE.md documents; it is debt only in the narrow sense
    # that ActionExecutor is not its owner. A raw primitive is ungoverned.
    print(
        f"  ├─ through a declared gateway, not ActionExecutor-owned: {governed}"
    )
    payable = 0
    for bucket in buckets:
        if bucket.canonical_owner or bucket.category in GATEWAY_CATEGORIES:
            continue
        if any(token in bucket.callee for token in PAYABLE_PRIMITIVES):
            payable += bucket.count
    report["debt_raw_payable_calls"] = payable
    print(f"  └─ RAW ungoverned primitives:                          {raw}")
    print(
        f"       of which durable_unlink/durable_replace could own:  {payable}"
    )
    for category, row in sorted(summary.items()):
        tier = "gateway" if category in GATEWAY_CATEGORIES else "RAW" if row["debt_calls"] else ""
        print(
            f"  {category}: calls={row['calls']} buckets={row['buckets']} "
            f"debt_calls={row['debt_calls']}{f'  [{tier}]' if tier else ''}"
        )
    return 0


#: Effect categories whose call sites go through a declared gateway. These
#: are migration debt only in the sense that ActionExecutor is not their
#: canonical owner — the call itself is the pattern CLAUDE.md documents
#: ("all consequential file writes go through file_write_gateway"). Lumping
#: them together with raw primitives produces one large number that reads as
#: "N governance bypasses" and is not that.
#: Raw primitives that HAVE a governed equivalent in the atomic writer, so the
#: debt is payable by migration. Everything else in the raw tier is debt only in
#: the sense that no governed primitive exists for it yet.
#:
#: mkdir is the reason this third split matters. It is over half the raw tier,
#: and the gateway's only directory primitive — ensure_directory — creates a
#: PRIVATE 0o700 directory and requires an active governance scope. Substituting
#: it for a plain `path.mkdir(parents=True, exist_ok=True)` would tighten
#: permissions at 450+ call sites and fail outright wherever no governed scope
#: is open. Counting those as migratable debt implies a migration that would
#: break the system.
PAYABLE_PRIMITIVES = ("unlink", "remove", "rename", "replace")

GATEWAY_CATEGORIES = frozenset(
    {
        "file_write_gateway",
        "direct_atomic_file_write",
        "subprocess_gateway",
        "network_gateway",
        "memory_write_gateway",
        "will_decision",
    }
)


def _write_report(path: Path | None, report: Mapping[str, Any]) -> None:
    if path is None:
        return
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "compare_inventory",
    "load_baseline",
    "main",
    "write_baseline",
]


if __name__ == "__main__":  # pragma: no cover - entry point
    # Run directly, this module used to parse nothing, write nothing, print
    # nothing and exit 0 — which reads exactly like a clean run, and a
    # `--write-baseline` that silently does nothing is worse than one that
    # fails. It holds main() but takes its scanner from tools.lint_governance,
    # so being an entry point means importing that first.
    import sys
    from pathlib import Path

    # Run as a script, `tools` is not on the path — the directory holding this
    # file is. The repository root is its parent.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import tools.lint_governance  # noqa: F401  (installs the scanner facade)

    raise SystemExit(main(sys.argv[1:]))
