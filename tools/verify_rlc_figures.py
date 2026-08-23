#!/usr/bin/env python3
"""Every RLC figure a document quotes must be recomputable from the evidence.

`make doc-drift` checks that referenced files exist. `make claim-constants`
checks constants cited from CLAIMS_MATRIX.md. Neither reads a *measured* number
quoted in prose, and that is the number that goes wrong.

docs/INTRINSIC_RECURRENCE.md said the live serving path verified "120/120 at a
median 47 ms and a maximum 318 ms". The 47 ms belonged to CP556, a 90-task run.
The 318 ms belonged to nothing: no artifact, ledger entry, or receipt in the
repository contains it. The sentence read as the most concrete in its paragraph
and had survived every review, because a fabricated measurement looks exactly
like a real one once it is prose.

So each figure below names the artifact field it came from, and the gate
recomputes it. A quoted number that no longer matches its source is a finding; a
quoted number whose source is missing is a finding. Counts that only ever grow
(modules, tools, tests, evidence directories) are held to a floor instead, since
an exact count would fail on the next checkpoint and earn this gate the right to
be switched off.

    python tools/verify_rlc_figures.py           # check every RLC document
    python tools/verify_rlc_figures.py --json OUT
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent

CP566 = (
    "artifacts/closeout/latent_cortex/"
    "cp566_resident_mixed_multidomain_replication/adjudication.json"
)
CP567_RUNTIME = (
    "artifacts/closeout/latent_cortex/cp567_semantic_neural_runtime/"
    "runtime_verification.json"
)
CP568_SHADOW = (
    "artifacts/closeout/latent_cortex/cp568_semantic_neural_shadow/"
    "runtime_verification.json"
)
CP568_ACTIVE = (
    "artifacts/closeout/latent_cortex/cp568_semantic_neural_active_r1/"
    "runtime_verification.json"
)
PREREG = "artifacts/current/latent_campaign_prereg_20260717.json"
RUN2 = "artifacts/current/latent_campaign_1p5b_run2.json"
SWEEP_32B = "artifacts/current/latent_lab_32b_exp1_templated.json"

#: Documents whose RLC figures are checked. Append-only records are excluded on
#: purpose: the tracker and the ledger state what was true on the day of the
#: entry, and correcting them would falsify the record.
DOCUMENTS = (
    "README.md",
    "docs/RECURSIVE_LATENT_CORTEX.md",
    "docs/INTRINSIC_RECURRENCE.md",
    "CHANGELOG.md",
)


class MissingEvidence(Exception):
    """The artifact a figure cites is not in the tree."""


def _load(relative: str) -> Any:
    path = ROOT / relative
    if not path.exists():
        raise MissingEvidence(relative)
    return json.loads(path.read_text())


@dataclass(frozen=True)
class Figure:
    """One quoted figure, its source, and how to recompute it."""

    #: What the documents must say. Checked as a literal substring so the gate
    #: never has to guess whether prose is an assertion.
    quoted: str
    #: Where the number lives.
    source: str
    #: Recomputes the quoted string from the loaded artifact.
    recompute: Callable[[Any], str]
    #: Documents that must contain ``quoted`` when they mention the figure at
    #: all. Empty means "check wherever it appears".
    why: str = ""


def _arm(field: str) -> Callable[[Any], str]:
    return lambda d: str(d["independent_exact_by_arm"][field])


def _p50(d: Any) -> str:
    return f"{d['p50_latency_ms']:.3f}"


def _max(d: Any) -> str:
    return f"{d['max_latency_ms']:.3f}"


def _ablation_totals(run: Any) -> dict[str, int]:
    arms = run["results"]["ablations"]["arms"]
    return {
        name: sum(int(cell["successes"]) for cell in families.values())
        for name, families in arms.items()
    }


def _ordinary_by_family(adj: Any) -> dict[str, int]:
    """Ordinary-decode exact counts, derived rather than quoted.

    Treatment answered every task, and no family regressed, so a family's
    ordinary successes are its size minus its gains. This is what makes
    "misleading premise gained nothing" readable: its ordinary decode was
    already at ceiling, not absent.
    """
    per_family = int(adj["tasks_per_family"])
    return {
        family: per_family - int(gained)
        for family, gained in adj["gains_by_family"].items()
    }


FIGURES: tuple[Figure, ...] = (
    Figure("60/60", CP566, lambda d: f"{_arm('treatment')(d)}/{d['task_count']}",
           why="CP566 treatment arm"),
    Figure("16/60", CP566, lambda d: f"{_arm('ordinary_base')(d)}/{d['task_count']}",
           why="CP566 ordinary decode"),
    Figure("matched wire base at 7", CP566,
           lambda d: f"matched wire base at {_arm('matched_wire_base')(d)}"),
    Figure("coefficient lesion at 5", CP566,
           lambda d: f"coefficient lesion at {_arm('coefficient_lesion')(d)}"),
    Figure("5.7 × 10⁻¹⁴", CP566,
           lambda d: _sci(d["paired_one_sided_exact_p"]),
           why="CP566 paired one-sided exact p"),
    Figure("BOUNDED_WOW_SIGNAL", CP566, lambda d: str(d["verdict"])),
    Figure("120/120", CP568_ACTIVE,
           lambda d: f"{d['exact_count']}/{d['task_count']}"),
    Figure("46.160 ms", CP567_RUNTIME, lambda d: f"{_p50(d)} ms"),
    Figure("83.188 ms", CP567_RUNTIME, lambda d: f"{_max(d)} ms"),
    Figure("34.686 / 63.737 ms", CP568_SHADOW,
           lambda d: f"{_p50(d)} / {_max(d)} ms"),
    Figure("6.501 / 17.102 ms", CP568_ACTIVE,
           lambda d: f"{_p50(d)} / {_max(d)} ms"),
    Figure("21/72", RUN2, lambda d: f"{_ablation_totals(d)['vanilla']}/72"),
    Figure("(7–13/72)", RUN2, lambda d: _arm_range(d)),
    # Accuracies are recorded to four places and quoted to three. Rounding is
    # the only tolerance this gate allows; every other figure must match byte
    # for byte, which is what would have caught the 318 ms.
    Figure("latent 0.375", SWEEP_32B, lambda d: f"latent {_best_latent(d):.3f}"),
    Figure("vanilla 0.417", SWEEP_32B,
           lambda d: "vanilla "
           f"{float(d['results']['exp1']['baseline']['accuracy']):.3f}"),
)


def _sci(value: float) -> str:
    """Render a p-value the way the documents write it."""
    exponent = 0
    mantissa = float(value)
    while mantissa < 1.0:
        mantissa *= 10.0
        exponent -= 1
    digits = "⁰¹²³⁴⁵⁶⁷⁸⁹"
    tail = "".join(digits[int(c)] for c in str(abs(exponent)))
    return f"{mantissa:.1f} × 10⁻{tail}"


def _arm_range(run: Any) -> str:
    totals = _ablation_totals(run)
    latent = [count for name, count in totals.items() if name != "vanilla"]
    return f"({min(latent)}–{max(latent)}/72)"


def _best_latent(sweep: Any) -> float:
    curve = sweep["results"]["exp1"]["curve"]
    return max(float(point["accuracy"]) for point in curve)


#: Counts that only grow. A document may state them; the gate holds the floor.
FLOORS: tuple[tuple[str, str, int], ...] = (
    ("core/brain/llm/latent_cortex/*.py", "156 modules", 156),
    ("tools/*", "55 `tools/` entry points", 55),
    ("tests/*", "231 test files", 231),
    ("artifacts/closeout/latent_cortex/*", "over 250 frozen evidence entries", 250),
)


def _floor_count(pattern: str) -> int:
    directory, _, glob = pattern.rpartition("/")
    entries = sorted((ROOT / directory).glob(glob))
    if directory == "tools" or directory == "tests":
        return sum(
            1 for e in entries if re.search(r"latent|recurren|rlc", e.name)
        )
    return len(entries)


#: Where an RLC latency may legitimately come from. Every retained runtime
#: verification, including the per-task rows.
LATENCY_SOURCES = (CP567_RUNTIME, CP568_SHADOW, CP568_ACTIVE)

#: The RLC section of a mixed document. Latency prose outside it belongs to
#: another subsystem and is not this gate's business.
_RLC_SECTION = re.compile(
    r"^##\s+Recursive Latent Cortex\b.*?(?=^##\s|\Z)", re.M | re.S
)

_LATENCY = re.compile(r"(\d+(?:\.\d+)?)\s*ms\b")


def _sourced_latencies() -> set[str]:
    """Every millisecond figure the evidence can support, as written."""
    values: set[float] = set()
    for source in LATENCY_SOURCES:
        try:
            data = _load(source)
        except MissingEvidence:
            continue
        for key in ("p50_latency_ms", "mean_latency_ms", "max_latency_ms"):
            if key in data:
                values.add(float(data[key]))
        for row in data.get("rows", ()):
            if "latency_ms" in row:
                values.add(float(row["latency_ms"]))
    written: set[str] = set()
    for value in values:
        written.add(f"{value:.3f}")
        written.add(f"{value:.2f}")
        written.add(f"{value:.1f}")
        written.add(f"{value:g}")
        written.add(str(int(round(value))))
    return written


def _unsourced_latencies(corpus: dict[str, str]) -> list[dict[str, Any]]:
    """A latency no receipt contains is the defect this gate exists for."""
    allowed = _sourced_latencies()
    findings: list[dict[str, Any]] = []
    for name, text in corpus.items():
        if name == "README.md":
            section = _RLC_SECTION.search(text)
            text = section.group(0) if section else ""
        elif not name.endswith(
            ("RECURSIVE_LATENT_CORTEX.md", "INTRINSIC_RECURRENCE.md")
        ):
            continue
        for match in _LATENCY.finditer(text):
            written = match.group(1)
            if written in allowed:
                continue
            if written.rstrip("0").rstrip(".") in allowed:
                continue
            findings.append(
                {
                    "figure": f"{written} ms",
                    "problem": "latency_unsourced",
                    "detail": (
                        "no retained runtime verification contains this value"
                    ),
                    "documents": [name],
                }
            )
    return findings


def check() -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    corpus = {
        name: (ROOT / name).read_text()
        for name in DOCUMENTS
        if (ROOT / name).exists()
    }

    for figure in FIGURES:
        try:
            recomputed = figure.recompute(_load(figure.source))
        except MissingEvidence as exc:
            findings.append(
                {
                    "figure": figure.quoted,
                    "problem": "evidence_missing",
                    "detail": f"{exc} is not in the tree",
                }
            )
            continue
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            findings.append(
                {
                    "figure": figure.quoted,
                    "problem": "evidence_unreadable",
                    "detail": f"{figure.source}: {exc}",
                }
            )
            continue
        if recomputed != figure.quoted:
            where = [n for n, text in corpus.items() if figure.quoted in text]
            findings.append(
                {
                    "figure": figure.quoted,
                    "problem": "figure_drifted",
                    "detail": f"{figure.source} now says {recomputed!r}",
                    "documents": where,
                }
            )

    for pattern, quoted, floor in FLOORS:
        actual = _floor_count(pattern)
        if actual < floor:
            findings.append(
                {
                    "figure": quoted,
                    "problem": "count_below_floor",
                    "detail": f"{pattern} holds {actual}, document claims {floor}",
                }
            )

    findings.extend(_unsourced_latencies(corpus))

    adjudication = _load(CP566)
    ordinary = _ordinary_by_family(adjudication)
    per_family = int(adjudication["tasks_per_family"])
    at_ceiling = [f for f, exact in ordinary.items() if exact == per_family]
    if at_ceiling != ["frontier_misleading_premise"]:
        findings.append(
            {
                "figure": "misleading premise was already at ceiling (15/15)",
                "problem": "ceiling_claim_drifted",
                "detail": f"families at ceiling: {at_ceiling or 'none'}",
            }
        )
    if sum(ordinary.values()) != int(
        adjudication["independent_exact_by_arm"]["ordinary_base"]
    ):
        findings.append(
            {
                "figure": "per-family ordinary counts",
                "problem": "ordinary_derivation_inconsistent",
                "detail": f"derived {sum(ordinary.values())} from gains_by_family",
            }
        )

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    findings = check()
    if args.json:
        args.json.write_text(json.dumps(findings, indent=1))
    if findings:
        for finding in findings:
            print(f"  {finding['problem']:32s} {finding['figure']}")
            print(f"      {finding['detail']}")
            if finding.get("documents"):
                print(f"      quoted in: {', '.join(finding['documents'])}")
        print(f"\n{len(findings)} figure(s) no longer match their evidence.")
        return 1
    if not args.quiet:
        print(
            f"OK: {len(FIGURES)} quoted figures recomputed from evidence, "
            f"{len(FLOORS)} counts at or above floor."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
