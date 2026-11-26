#!/usr/bin/env python3
"""tools/check_evidence_integrity.py — a claim may not outrank its evidence.

Claim 14 (AGI-Candidate) sat at `locally demonstrated` for a month while the
same table cell explained that its primary evidence was an unfair comparison:
baselines strangled at 160 tokens against an effectively unbounded,
solver-assisted treatment, on tasks that cannot be answered inside 160 tokens.
The retraction and the classification lived side by side and the classification
won, because nothing checked.

Two things had to be true for that to happen, and this gate closes both:

1.  The retraction was PROSE. `BASELINES.json` still read `"status": "RUN"`
    with clean pass rates, so every automated consumer saw a passing artifact.
    Evidence whose validity has been withdrawn now carries a machine-readable
    `RETRACTION.json` beside it.

2.  Nothing derived a claim's ceiling from the state of its evidence. A claim
    now cannot be classified above `not proven` while citing a retracted
    artifact.

This does not decide whether evidence is good. It enforces the one thing that
should never need judgement: a claim cannot be stronger than the evidence it
names, and a retraction must reach every claim that leans on it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CLAIMS_MATRIX = ROOT / "CLAIMS_MATRIX.md"
RETRACTION_SCHEMA = "aura.evidence_retraction.v1"
RETRACTION_FILENAME = "RETRACTION.json"

#: Classifications ordered weakest to strongest. Anything above `not proven`
#: asserts something, and an assertion is what retracted evidence cannot carry.
CLASSIFICATION_RANK = {
    "not proven": 0,
    "deprecated": 0,
    "retired": 0,
    "blocked": 0,
    "locally demonstrated": 1,
    "causally demonstrated": 2,
    "externally validated": 3,
}
ASSERTING_FLOOR = 1

#: A claim row: | **N. Name** | `classification` | evidence prose |
_CLAIM_ROW = re.compile(
    r"^\|\s*\*\*(?P<number>\d+)\.\s*(?P<name>[^*]+?)\*\*\s*\|\s*`(?P<classification>[^`]+)`\s*\|(?P<evidence>.*)\|\s*$"
)
#: Any artifact path mentioned in an evidence cell.
_ARTIFACT_PATH = re.compile(r"`(?P<path>artifacts/[A-Za-z0-9_\-./]+)`")


class IntegrityFailure(Exception):
    """A claim outranks its evidence."""


def load_retraction(directory: Path) -> dict[str, Any] | None:
    """Read a retraction sidecar, or None when the artifact is not retracted.

    A malformed sidecar is treated as a retraction, not as its absence. The
    failure mode this gate exists to prevent is evidence quietly counting as
    valid, so an unreadable validity statement must fail closed.
    """
    path = directory / RETRACTION_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "schema": RETRACTION_SCHEMA,
            "status": "retracted",
            "reason": f"retraction sidecar is unreadable ({type(exc).__name__}); "
            "treating the evidence as retracted rather than assuming it is fine",
            "malformed": True,
        }
    if not isinstance(data, dict) or data.get("schema") != RETRACTION_SCHEMA:
        return {
            "schema": RETRACTION_SCHEMA,
            "status": "retracted",
            "reason": f"retraction sidecar does not declare {RETRACTION_SCHEMA}",
            "malformed": True,
        }
    if str(data.get("status", "")).strip().lower() != "retracted":
        return None
    return data


def find_retractions(root: Path) -> dict[str, dict[str, Any]]:
    """Every retracted artifact directory, keyed by repo-relative path."""
    retracted: dict[str, dict[str, Any]] = {}
    artifacts_root = root / "artifacts"
    if not artifacts_root.is_dir():
        return retracted
    for path in artifacts_root.rglob(RETRACTION_FILENAME):
        directory = path.parent
        record = load_retraction(directory)
        if record is not None:
            retracted[str(directory.relative_to(root))] = record
    return retracted


def parse_claims(matrix_text: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for line_no, line in enumerate(matrix_text.splitlines(), start=1):
        match = _CLAIM_ROW.match(line)
        if match is None:
            continue
        claims.append(
            {
                "line": line_no,
                "number": int(match.group("number")),
                "name": match.group("name").strip(),
                "classification": match.group("classification").strip().lower(),
                "evidence": match.group("evidence"),
            }
        )
    return claims


def cited_artifacts(evidence: str) -> list[str]:
    """Artifact paths a claim cites, normalised without a trailing slash."""
    return [match.group("path").rstrip("/") for match in _ARTIFACT_PATH.finditer(evidence)]


def retracted_paths(retracted_dir: str, record: dict[str, Any]) -> list[str]:
    """The paths this retraction actually withdraws.

    `retracted_files` has been in the schema since the agi_live retraction and
    nothing read it, so a retraction could only ever be all-or-nothing on a
    directory. That is wrong for a bundle: `artifacts/proof_bundle/latest`
    holds nineteen independent artifacts, and withdrawing UNDENIABLE_RSI.json
    must not withdraw BOOT_HEALTH.json beside it.

    An empty or missing list still means the whole directory, which keeps the
    agi_live behaviour and keeps a malformed sidecar failing closed.
    """
    files = record.get("retracted_files")
    if not isinstance(files, list) or not files:
        return [retracted_dir]
    named = [
        f"{retracted_dir}/{str(name).strip().lstrip('/')}"
        for name in files
        if str(name).strip()
    ]
    return named or [retracted_dir]


def _covers(cited: str, retracted: str) -> bool:
    """Does citing ``cited`` mean leaning on ``retracted``?

    Citing a directory cites everything under it, so `artifacts/current/` picks
    up a retraction three levels down. Citing a file inside a retracted bundle
    counts too, and citing the directory that contains a retracted file counts:
    a claim resting on a bundle rests on the withdrawn part of it until the row
    says otherwise.
    """
    return (
        cited == retracted
        or cited.startswith(retracted + "/")
        or retracted.startswith(cited + "/")
    )


def check(root: Path = ROOT) -> dict[str, Any]:
    matrix_path = root / "CLAIMS_MATRIX.md"
    if not matrix_path.is_file():
        raise IntegrityFailure(f"{matrix_path} is missing; there is nothing to check")

    retractions = find_retractions(root)
    claims = parse_claims(matrix_path.read_text(encoding="utf-8"))
    if not claims:
        raise IntegrityFailure(
            "no claim rows parsed out of CLAIMS_MATRIX.md — the table shape "
            "changed and this gate silently stopped checking anything"
        )

    violations: list[dict[str, Any]] = []
    for claim in claims:
        rank = CLASSIFICATION_RANK.get(claim["classification"])
        if rank is None:
            violations.append(
                {
                    "claim": claim["number"],
                    "name": claim["name"],
                    "line": claim["line"],
                    "problem": "unknown_classification",
                    "detail": (
                        f"classification {claim['classification']!r} is not one of "
                        f"{sorted(CLASSIFICATION_RANK)}. An unrecognised label cannot "
                        "be ranked, so it cannot be checked."
                    ),
                }
            )
            continue
        if rank < ASSERTING_FLOOR:
            continue
        for cited in cited_artifacts(claim["evidence"]):
            for retracted_dir, record in retractions.items():
                withdrawn = retracted_paths(retracted_dir, record)
                if not any(_covers(cited, path) for path in withdrawn):
                    continue
                # An asserting claim may still NAME retracted evidence when it
                # is naming it as retracted — that is how a row explains its own
                # demotion. What it may not do is count it as support.
                if "RETRACTION.json" in claim["evidence"] or "etracted" in claim["evidence"]:
                    continue
                violations.append(
                    {
                        "claim": claim["number"],
                        "name": claim["name"],
                        "line": claim["line"],
                        "problem": "asserting_claim_cites_retracted_evidence",
                        "classification": claim["classification"],
                        "artifact": retracted_dir,
                        "retracted_paths": withdrawn,
                        "detail": record.get("reason", "")[:400],
                    }
                )

    return {
        "schema": "aura.evidence_integrity.v1",
        "claims_checked": len(claims),
        "retracted_artifacts": sorted(retractions),
        "violations": violations,
        "passed": not violations,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="write the report here")
    args = parser.parse_args(argv)

    try:
        report = check()
    except IntegrityFailure as exc:
        print(f"evidence integrity FAILED: {exc}", file=sys.stderr)
        return 1

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if report["retracted_artifacts"]:
        print(f"retracted evidence bundles: {', '.join(report['retracted_artifacts'])}")

    if not report["passed"]:
        print(
            f"evidence integrity FAILED: {len(report['violations'])} claim(s) outrank their evidence",
            file=sys.stderr,
        )
        for violation in report["violations"]:
            print(
                f"  claim {violation['claim']} ({violation['name']}) at "
                f"CLAIMS_MATRIX.md:{violation['line']}: {violation['problem']}",
                file=sys.stderr,
            )
            if violation.get("artifact"):
                print(f"    cites retracted {violation['artifact']}", file=sys.stderr)
            if violation.get("detail"):
                print(f"    {violation['detail']}", file=sys.stderr)
        return 1

    print(
        f"evidence integrity: {report['claims_checked']} claims checked, "
        f"none outrank their evidence"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
