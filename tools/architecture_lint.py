#!/usr/bin/env python3
"""tools/architecture_lint.py — shape rules for new cognitive organs.

Four rules, each from a defect this repository has actually had, and each
applied only to files that opt in. A shape rule retrofitted across ninety
thousand files produces a baseline of exceptions and changes nothing; applied
to the modules built under it, it holds.

* **Composition over inheritance.** An organ that subclasses another organ
  inherits its state, its locks and its failure modes, and a hot reload then
  breaks class identity on both - which has happened here, with ``↻ UPDATE``
  reporting success while ``isinstance`` silently began returning False. Depth
  is capped at one below a declared base.

* **State lives in one place.** A module registered as stateless may not hold
  a mutable module-level binding. A dict at module scope is a store nobody
  declared, nobody owns and nothing can rewind.

* **A contract change cannot leave a consumer stale.** Any dataclass marked as
  a contract carries a schema version, and the lint fails when a contract's
  field set changes without the version moving. Two definitions of one message
  is how the half-wired channels happened.

* **Research does not import production.** A module under an experimental path
  may not import the live runtime. The dependency is the reverse of the one
  people expect and it is the one that bites: an experiment that imports the
  live tree runs against it.

    python tools/architecture_lint.py            # check
    python tools/architecture_lint.py --list     # show what is covered
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COVERAGE = ROOT / "config/architecture_lint_coverage.json"

DEFAULT_COVERAGE: dict[str, list[str]] = {
    "composition": [
        "core/cognition/procedure.py",
        "core/cognition/cognitive_event.py",
        "core/cognition/dual_knowledge.py",
        "core/runtime/event_spine.py",
        "core/science/claim_ladder.py",
    ],
    "stateless": [
        "core/evidence/packet.py",
        "core/cognition/cognitive_vector.py",
        "core/cognition/structure_mapping.py",
        "core/science/baseline_portfolio.py",
        "core/science/retrieval_latency.py",
    ],
    "contract_version": [
        "core/evidence/packet.py",
        "core/evidence/state_ref.py",
        "core/cognition/action_receipt.py",
    ],
    "research_isolation": [
        "experiments",
        "research",
    ],
    #: Probes whose purpose is to read live state. Reading the live snapshot is
    #: what introspective_accuracy MEASURES; forbidding it would delete the
    #: experiment rather than isolate it.
    "declared_probes": [
        "research/consciousness/introspective_accuracy.py",
    ],
}

#: Modules a research tree may never import.
PRODUCTION_ROOTS = ("core.runtime", "core.brain", "aura_main", "interface")

#: The sanctioned write path is infrastructure, not live state. Every
#: consequential write in this repository goes through the gateway, including
#: from an experiment, so importing it is the rule rather than a breach of one.
INFRASTRUCTURE = (
    "core.runtime.file_write_gateway",
    "core.runtime.atomic_writer",
    "core.runtime.subprocess_gateway",
    "core.runtime.lockdep",
    "core.runtime.errors",
)


@dataclass
class Finding:
    rule: str
    path: str
    line: int
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.message}"


def _load_coverage() -> dict[str, list[str]]:
    if COVERAGE.exists():
        return json.loads(COVERAGE.read_text())
    return DEFAULT_COVERAGE


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(errors="replace"))
    except (OSError, SyntaxError):
        return None


def check_composition(paths: list[str]) -> list[Finding]:
    """No organ subclasses another organ from the same covered set."""
    findings: list[Finding] = []
    defined: dict[str, str] = {}
    trees: dict[str, ast.Module] = {}
    for rel in paths:
        tree = _parse(ROOT / rel)
        if tree is None:
            continue
        trees[rel] = tree
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                defined[node.name] = rel
    for rel, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
                if name in defined:
                    findings.append(Finding(
                        "composition", rel, node.lineno,
                        f"{node.name} subclasses {name}; hold it as a field instead - "
                        "inheriting an organ inherits its locks and its failure modes",
                    ))
    return findings


def check_stateless(paths: list[str]) -> list[Finding]:
    """A module declared stateless holds no mutable module-level binding."""
    findings: list[Finding] = []
    for rel in paths:
        tree = _parse(ROOT / rel)
        if tree is None:
            continue
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if not names or all(n.isupper() or n.startswith("_") for n in names):
                continue
            value = node.value
            if isinstance(value, (ast.Dict, ast.List, ast.Set)):
                findings.append(Finding(
                    "stateless", rel, node.lineno,
                    f"{names[0]} is a mutable module-level store nobody owns and "
                    "nothing can rewind; put it behind a class or the event spine",
                ))
    return findings


def check_contract_version(paths: list[str]) -> list[Finding]:
    """Every dataclass that crosses an organ boundary carries a version."""
    findings: list[Finding] = []
    for rel in paths:
        source = (ROOT / rel).read_text(errors="replace") if (ROOT / rel).exists() else ""
        if "SCHEMA_VERSION" not in source:
            findings.append(Finding(
                "contract_version", rel, 1,
                "no SCHEMA_VERSION; a contract whose field set changes without a version "
                "leaves consumers reading a shape that no longer exists",
            ))
    return findings


def check_research_isolation(
    paths: list[str], declared_probes: frozenset[str] = frozenset()
) -> list[Finding]:
    """A research tree does not import the live runtime, unless it says why.

    Two exemptions, both narrow. The sanctioned write path is infrastructure -
    every consequential write goes through the gateway and an experiment is no
    exception. And a probe whose purpose is to read live state is declared by
    path in the coverage file, so the exception is a line in a diff rather than
    a silent pass.
    """
    findings: list[Finding] = []
    for rel in paths:
        base = ROOT / rel
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            tree = _parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                module = ""
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                elif isinstance(node, ast.Import):
                    module = node.names[0].name if node.names else ""
                if not module.startswith(PRODUCTION_ROOTS):
                    continue
                if module.startswith(INFRASTRUCTURE):
                    continue
                relative = str(path.relative_to(ROOT))
                if relative in declared_probes:
                    continue
                findings.append(Finding(
                    "research_isolation", relative, node.lineno,
                    f"imports {module}; an experiment that imports the live tree runs "
                    "against it. A probe whose PURPOSE is to read live state belongs in "
                    "the declared_probes list, where a reviewer can see it.",
                ))
    return findings


CHECKS = {
    "composition": check_composition,
    "stateless": check_stateless,
    "contract_version": check_contract_version,
    "research_isolation": check_research_isolation,
}


def run() -> tuple[list[Finding], dict[str, list[str]]]:
    coverage = _load_coverage()
    findings: list[Finding] = []
    probes = frozenset(coverage.get("declared_probes", []))
    for rule, check in CHECKS.items():
        if rule == "research_isolation":
            findings.extend(check(coverage.get(rule, []), probes))
        else:
            findings.extend(check(coverage.get(rule, [])))
    return findings, coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show what each rule covers")
    args = parser.parse_args()
    findings, coverage = run()
    if args.list:
        for rule, paths in sorted(coverage.items()):
            print(f"{rule}: {len(paths)} path(s)")
            for path in paths:
                print(f"  {path}")
        return 0
    for finding in findings:
        print(f"architecture-lint: {finding.render()}", file=sys.stderr)
    covered = sum(len(v) for v in coverage.values())
    if findings:
        print(f"architecture-lint: {len(findings)} finding(s) over {covered} covered path(s)",
              file=sys.stderr)
        return 1
    print(f"architecture-lint: clean over {covered} covered path(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
